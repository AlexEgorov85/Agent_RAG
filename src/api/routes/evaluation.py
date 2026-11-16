"""Evaluation / Benchmark API (docs/specs/evaluation-webui.md §8).

Adapter читает артефакты ``eval.runner``: summary/cases/traces. Frontend не
пересчитывает метрики; adapter дополнительно считает только дельты сравнения и
диагностику покрытия по сохранённым данным. Benchmark не выполняется в
HTTP-запросе — создание запуска ставит задачу в очередь worker'а.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Mapping

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.bootstrap import ensure_app_state
from src.api.errors import ApiError
from src.api.evaluation_schemas import EvaluationRunCreate
from src.api.evaluation_store import (
    ACTIVE_RUN_STATUSES,
    RETRIEVAL_MODES,
    TERMINAL_RUN_STATUSES,
    EvaluationRuntime,
    create_evaluation_runtime,
)
from src.api.routes.deps import Owner

router = APIRouter()

SUITE_IDS: tuple[str, ...] = (
    "smoke",
    "full",
    "easy",
    "medium",
    "hard",
    "multi_turn",
)
SPLITS: tuple[str, ...] = ("dev", "holdout", "all")
MAX_PAGE = 100
MAX_TRACE_BYTES = 5 * 1024 * 1024

_LOSS_CAUSE = {
    "never_retrieved": "retrieval",
    "lost_at_fusion": "fusion",
    "lost_at_rerank": "rerank",
    "lost_at_context": "context",
    "not_cited": "citation",
    "unsupported_claim": "answer",
    "dialogue_context_lost": "dialogue",
}


# ---------------------------------------------------------------------------
# Runtime wiring
# ---------------------------------------------------------------------------


def ensure_evaluation_state(app: Any) -> EvaluationRuntime | None:
    """Создать storage/worker при первом обращении, если раздел включён."""
    ensure_app_state(app)
    config = getattr(app.state, "config", None)
    if config is None or not getattr(config, "eval_ui_enabled", False):
        return None
    lock = getattr(app.state, "_evaluation_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state._evaluation_lock = lock
    with lock:
        runtime = getattr(app.state, "evaluation", None)
        if runtime is None:
            runtime = create_evaluation_runtime(
                config, getattr(app.state, "services", None)
            )
            app.state.evaluation = runtime
        return runtime


def _missing_runtime() -> ApiError:
    return ApiError(
        404,
        "evaluation_disabled",
        "Раздел Evaluation отключён конфигурацией.",
    )


def _runtime(request: Request) -> EvaluationRuntime:
    runtime = ensure_evaluation_state(request.app)
    if runtime is None:
        raise _missing_runtime()
    return runtime


RuntimeDep = Annotated[EvaluationRuntime, Depends(_runtime)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _duration_ms(row: Mapping[str, Any]) -> int | None:
    started = _parse_iso(row.get("started_at"))
    finished = _parse_iso(row.get("finished_at"))
    if started is None or finished is None:
        return None
    return max(int((finished - started).total_seconds() * 1000), 0)


def _json_field(row: Mapping[str, Any], key: str) -> Any:
    raw = row.get(key)
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return None


def _summary(row: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = _json_field(row, "summary_json")
    return payload if isinstance(payload, dict) else None


def _progress(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_field(row, "progress_json")
    progress = payload if isinstance(payload, dict) else {}
    total = progress.get("tests_total")
    completed = progress.get("tests_completed")
    total_value = int(total) if isinstance(total, int) and not isinstance(total, bool) else 0
    completed_value = (
        int(completed) if isinstance(completed, int) and not isinstance(completed, bool) else 0
    )
    percent = progress.get("percent")
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        percent = round(completed_value * 100.0 / total_value, 1) if total_value else 0.0
    return {
        "tests_total": total_value,
        "tests_completed": completed_value,
        "current_test_id": progress.get("current_test_id"),
        "current_turn": progress.get("current_turn"),
        "percent": float(percent),
    }


def _run_versions(row: Mapping[str, Any], summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = summary or {}
    git = data.get("git") if isinstance(data.get("git"), Mapping) else {}
    corpus = data.get("corpus") if isinstance(data.get("corpus"), Mapping) else {}
    models = data.get("models") if isinstance(data.get("models"), Mapping) else {}
    prompts = data.get("prompts") if isinstance(data.get("prompts"), Mapping) else {}
    return {
        "git_sha": (git or {}).get("sha") or row.get("git_sha") or "",
        "git_dirty": bool((git or {}).get("dirty")),
        "corpus_content_sha256": (corpus or {}).get("content_sha256")
        or row.get("corpus_content_sha256")
        or "",
        "corpus_fingerprint": (corpus or {}).get("index_fingerprint"),
        "config_fingerprint": data.get("config_fingerprint"),
        "llm_model": (models or {}).get("llm"),
        "reranker_model": (models or {}).get("reranker"),
        "judge_model": (models or {}).get("judge"),
        "prompt_hashes": dict(prompts or {}),
    }


def _run_dataset(row: Mapping[str, Any], summary: Mapping[str, Any] | None) -> dict[str, Any]:
    data = summary or {}
    dataset = data.get("dataset") if isinstance(data.get("dataset"), Mapping) else {}
    coverage = data.get("coverage") if isinstance(data.get("coverage"), Mapping) else {}
    progress = _progress(row)
    tests_total = progress["tests_total"]
    if not tests_total:
        selected = (coverage or {}).get("tests_selected")
        tests_total = int(selected) if isinstance(selected, int) else 0
    return {
        "version": (dataset or {}).get("version"),
        "sha256": (dataset or {}).get("sha256") or row.get("dataset_sha256"),
        "tests_total": tests_total,
    }


def _artifacts_dir(runtime: EvaluationRuntime, row: Mapping[str, Any]) -> Path | None:
    raw = row.get("artifacts_dir")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_dir():
        return None
    return path


def _report_path(runtime: EvaluationRuntime, row: Mapping[str, Any]) -> Path | None:
    directory = _artifacts_dir(runtime, row)
    if directory is None:
        return None
    for name in ("report.html", "report.md"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _run_payload(runtime: EvaluationRuntime, row: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(row)
    artifacts = _artifacts_dir(runtime, row)
    report = _report_path(runtime, row)
    return {
        "id": row.get("id"),
        "status": row.get("status"),
        "suite": row.get("suite"),
        "split": row.get("split"),
        "retrieval_mode": row.get("retrieval_mode"),
        "judge": bool(row.get("judge")),
        "repeat": int(row.get("repeat") or 1),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "progress": _progress(row),
        "dataset": _run_dataset(row, summary),
        "versions": _run_versions(row, summary),
        "summary": summary,
        "error": _json_field(row, "error_json"),
        "artifacts_url": (
            f"/api/evaluation/runs/{row.get('id')}/report" if report else None
        ),
        "artifacts_available": artifacts is not None,
        "notes": row.get("notes"),
        "duration_ms": _duration_ms(row),
    }


def _run_list_item(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(row)
    coverage = (summary or {}).get("coverage") or {}
    progress = _progress(row)
    selected = int(coverage.get("tests_selected") or 0) or progress["tests_total"]
    return {
        "id": row.get("id"),
        "status": row.get("status"),
        "suite": row.get("suite"),
        "split": row.get("split"),
        "retrieval_mode": row.get("retrieval_mode"),
        "judge": bool(row.get("judge")),
        "repeat": int(row.get("repeat") or 1),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "pass_rate": (summary or {}).get("pass_rate"),
        "execution_coverage": (summary or {}).get("execution_coverage"),
        "tests_total": selected,
        "tests_eligible": int(coverage.get("tests_eligible") or 0),
        "tests_executed": int(coverage.get("executed") or 0),
        "tests_passed": int(coverage.get("passed") or 0),
        "tests_skipped": int(coverage.get("skipped") or 0),
        "duration_ms": _duration_ms(row),
        "git_sha": row.get("git_sha"),
        "git_dirty": bool(((summary or {}).get("git") or {}).get("dirty")),
        "dataset_sha256": row.get("dataset_sha256"),
        "corpus_content_sha256": row.get("corpus_content_sha256"),
        "progress": progress,
        "has_summary": summary is not None,
    }


def _get_run_or_404(runtime: EvaluationRuntime, run_id: str) -> dict[str, Any]:
    row = runtime.store.get_run(run_id)
    if row is None:
        raise ApiError(404, "run_not_found", "Запуск не найден.")
    return row


def _page(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(int(limit), MAX_PAGE)), max(0, int(offset))


# ---------------------------------------------------------------------------
# Suites / runs
# ---------------------------------------------------------------------------


@router.get("/evaluation/suites", tags=["evaluation"])
def list_suites(runtime: RuntimeDep, owner: Owner) -> dict[str, Any]:
    """Список наборов тестов и доступных retrieval-режимов."""
    dataset = runtime.dataset()
    items: list[dict[str, Any]] = []
    for suite in SUITE_IDS:
        if not dataset.available:
            items.append(
                {
                    "suite": suite,
                    "tests": None,
                    "dev": None,
                    "holdout": None,
                    "requires_holdout_confirmation": suite == "full",
                }
            )
            continue
        if suite == "smoke":
            selected = [
                test for test in dataset.tests if "smoke" in (test.get("tags") or [])
            ]
            selected = [test for test in selected if test.get("split") == "dev"]
        elif suite == "full":
            selected = list(dataset.tests)
        else:
            selected = [
                test for test in dataset.tests if test.get("suite") == suite
            ]
        dev = sum(1 for test in selected if test.get("split") == "dev")
        holdout = sum(1 for test in selected if test.get("split") == "holdout")
        items.append(
            {
                "suite": suite,
                "tests": len(selected),
                "dev": dev,
                "holdout": holdout,
                "requires_holdout_confirmation": bool(holdout)
                and suite in {"full", "easy", "medium", "hard", "multi_turn"},
            }
        )
    return {
        "items": items,
        "dataset_available": dataset.available,
        "dataset_sha256": dataset.dataset_sha256,
        "expected_corpus_sha256": dataset.expected_corpus_sha256,
        "retrieval_modes": list(runtime.config.capabilities_modes()),
        "max_repeat": 5,
        "max_queued": int(runtime.config.eval_queue_size),
    }


def _selected_tests_total(
    runtime: EvaluationRuntime, *, suite: str, split: str, mode: str
) -> int:
    dataset = runtime.dataset()
    if not dataset.available:
        return 0
    if suite == "smoke":
        selected = [
            test
            for test in dataset.tests
            if "smoke" in (test.get("tags") or []) and test.get("split") == "dev"
        ]
    elif suite == "full":
        selected = list(dataset.tests)
    else:
        selected = [test for test in dataset.tests if test.get("suite") == suite]
    if split != "all":
        selected = [test for test in selected if test.get("split") == split]
    modes = len(RETRIEVAL_MODES) if mode == "all" else 1
    return len(selected) * modes


def _new_run_id() -> str:
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:7]}"


def _validate_mode(runtime: EvaluationRuntime, mode: str) -> None:
    supported = set(runtime.config.capabilities_modes())
    if mode != "all" and mode not in supported:
        raise ApiError(
            422,
            "mode_not_supported",
            "Режим retrieval не поддержан backend.",
            details={"retrieval_mode": mode, "supported": sorted(supported)},
        )


@router.post("/evaluation/runs", status_code=202, tags=["evaluation"])
def create_run(
    payload: EvaluationRunCreate,
    runtime: RuntimeDep,
    owner: Owner,
) -> JSONResponse:
    """Поставить evaluation-прогон в очередь."""
    existing = None
    if payload.client_request_id:
        existing = runtime.store.find_by_client_request_id(payload.client_request_id)
    if existing is not None:
        return JSONResponse(
            status_code=202,
            content={
                "id": existing["id"],
                "status": existing["status"],
                "status_url": f"/api/evaluation/runs/{existing['id']}",
                "created_at": existing["created_at"],
                "duplicate": True,
            },
        )

    suite = (payload.suite or "").strip()
    if suite not in SUITE_IDS:
        raise ApiError(
            422,
            "invalid_suite",
            "Suite вне whitelist.",
            details={"allowed": list(SUITE_IDS)},
        )

    repeat = payload.repeat
    if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= 5:
        raise ApiError(422, "invalid_repeat", "repeat должен быть в диапазоне 1–5.")

    split = payload.split or ("dev" if suite == "smoke" else "all" if suite == "full" else "dev")
    if split not in SPLITS:
        raise ApiError(
            422, "invalid_split", "Split вне dev/holdout/all.", details={"allowed": list(SPLITS)}
        )
    if suite == "smoke" and split != "dev":
        raise ApiError(422, "invalid_split", "Smoke всегда выполняется на split=dev.")

    mode = (payload.retrieval_mode or "hybrid").strip()
    _validate_mode(runtime, mode)

    if payload.judge and not runtime.config.judge_available():
        raise ApiError(422, "judge_unavailable", "LLM-судья недоступен в этой конфигурации.")

    requires_holdout = split in {"all", "holdout"} and suite == "full"
    if split == "holdout":
        requires_holdout = True
    if requires_holdout and not payload.confirm_holdout:
        raise ApiError(
            422,
            "holdout_confirmation_required",
            "Для holdout требуется confirm_holdout=true.",
        )

    running, queued = runtime.store.count_active()
    if queued >= int(runtime.config.eval_queue_size):
        raise ApiError(
            429,
            "queue_full",
            "Очередь benchmark заполнена.",
            retryable=True,
            details={"max_queued": int(runtime.config.eval_queue_size)},
        )
    if running and int(runtime.config.eval_queue_size) <= 0:
        raise ApiError(
            409, "run_in_progress", "Уже выполняется benchmark.", retryable=True
        )

    run_id = _new_run_id()
    request_payload = {
        "suite": suite,
        "split": split,
        "retrieval_mode": mode,
        "judge": bool(payload.judge),
        "repeat": repeat,
        "notes": payload.cleaned_notes(),
        "client_request_id": payload.client_request_id,
        "tests_total": _selected_tests_total(
            runtime, suite=suite, split=split, mode=mode
        ),
    }
    row = runtime.store.create_run(
        run_id=run_id,
        suite=suite,
        split=split,
        retrieval_mode=mode,
        judge=bool(payload.judge),
        repeat=repeat,
        request=request_payload,
        notes=payload.cleaned_notes(),
        client_request_id=payload.client_request_id,
        tests_total=request_payload["tests_total"],
    )
    runtime.worker.submit(run_id)
    return JSONResponse(
        status_code=202,
        content={
            "id": row["id"],
            "status": row["status"],
            "status_url": f"/api/evaluation/runs/{row['id']}",
            "created_at": row["created_at"],
            "duplicate": False,
        },
    )


@router.get("/evaluation/runs", tags=["evaluation"])
def list_runs(
    runtime: RuntimeDep,
    owner: Owner,
    suite: str | None = None,
    status: str | None = None,
    split: str | None = None,
    retrieval_mode: str | None = None,
    limit: int = Query(20, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Список прогонов с фильтрами."""
    page_limit, page_offset = _page(limit, offset)
    if status and status not in {*ACTIVE_RUN_STATUSES, *TERMINAL_RUN_STATUSES}:
        raise ApiError(422, "validation_error", "Неизвестный статус запуска.")
    if suite and suite not in SUITE_IDS:
        raise ApiError(422, "invalid_suite", "Suite вне whitelist.")
    rows, total = runtime.store.list_runs(
        suite=suite,
        status=status,
        split=split,
        retrieval_mode=retrieval_mode,
        limit=page_limit,
        offset=page_offset,
    )
    next_offset = page_offset + len(rows) if page_offset + len(rows) < total else None
    return {
        "items": [_run_list_item(row) for row in rows],
        "total": total,
        "next_offset": next_offset,
    }


@router.get("/evaluation/runs/{run_id}", tags=["evaluation"])
def get_run(run_id: str, runtime: RuntimeDep, owner: Owner) -> dict[str, Any]:
    """Детали прогона."""
    return _run_payload(runtime, _get_run_or_404(runtime, run_id))


def _result_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    case = row.get("case") if isinstance(row.get("case"), Mapping) else {}
    return {
        "test_id": row.get("test_id"),
        "retrieval_mode": row.get("retrieval_mode"),
        "attempt_index": int(row.get("attempt_index") or 0),
        "suite": row.get("suite") or case.get("suite"),
        "category": row.get("category") or case.get("category"),
        "difficulty": row.get("difficulty") or case.get("difficulty"),
        "status": row.get("status"),
        "pass": bool(row.get("pass")),
        "metrics": row.get("metrics") or {},
        "answer": case.get("answer") or {},
        "critical_errors": row.get("critical_errors") or [],
        "warnings": case.get("warnings") or [],
        "efficiency": case.get("efficiency") or {},
        "trace_path": row.get("trace_path"),
        "trace_available": bool(row.get("trace_available")),
        "duration_ms": row.get("duration_ms"),
    }


@router.get("/evaluation/runs/{run_id}/tests", tags=["evaluation"])
def list_run_tests(
    run_id: str,
    runtime: RuntimeDep,
    owner: Owner,
    status: str | None = None,
    critical: bool | None = None,
    difficulty: str | None = None,
    category: str | None = None,
    retrieval_mode: str | None = None,
    limit: int = Query(50, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Результаты тестов прогона."""
    _get_run_or_404(runtime, run_id)
    page_limit, page_offset = _page(limit, offset)
    passed: bool | None = None
    status_value: str | None = status
    if status in {"passed", "failed"}:
        passed = status == "passed"
        status_value = None
    rows, total = runtime.store.list_results(
        run_id,
        status=status_value,
        passed=passed,
        critical=critical,
        difficulty=difficulty,
        category=category,
        retrieval_mode=retrieval_mode,
        limit=page_limit,
        offset=page_offset,
    )
    next_offset = page_offset + len(rows) if page_offset + len(rows) < total else None
    return {
        "items": [_result_payload(row) for row in rows],
        "total": total,
        "next_offset": next_offset,
    }


def _resolve_result(
    runtime: EvaluationRuntime,
    run_id: str,
    test_id: str,
    retrieval_mode: str | None,
    attempt_index: int,
) -> tuple[str, dict[str, Any]]:
    if retrieval_mode == "all":
        raise ApiError(
            422,
            "invalid_retrieval_mode",
            "retrieval_mode=all недопустим для одного результата.",
            details={"allowed": list(RETRIEVAL_MODES)},
        )
    if retrieval_mode and retrieval_mode not in RETRIEVAL_MODES:
        raise ApiError(
            422,
            "invalid_retrieval_mode",
            "Неизвестный режим retrieval.",
            details={"allowed": list(RETRIEVAL_MODES)},
        )
    modes = runtime.store.result_modes(run_id, test_id)
    if not modes:
        raise ApiError(404, "test_not_found", "Результат теста не найден в этом запуске.")
    mode = retrieval_mode
    if mode is None:
        if len(modes) > 1:
            raise ApiError(
                409,
                "ambiguous_test_result",
                "У теста несколько режимов retrieval; укажите retrieval_mode.",
                details={"available_modes": modes},
            )
        mode = modes[0]
    row = runtime.store.get_result(run_id, test_id, mode, attempt_index)
    if row is None:
        raise ApiError(
            404,
            "test_not_found",
            "Результат теста с таким attempt_index не найден.",
            details={
                "available_modes": modes,
                "available_attempts": _available_attempts(runtime, run_id, test_id, mode),
            },
        )
    return mode, row


def _available_attempts(
    runtime: EvaluationRuntime, run_id: str, test_id: str, mode: str
) -> list[int]:
    rows, _ = runtime.store.list_results(
        run_id, retrieval_mode=mode, limit=MAX_PAGE, offset=0
    )
    return sorted(
        {
            int(row["attempt_index"])
            for row in rows
            if row.get("test_id") == test_id
        }
    )


def _safe_trace_path(artifacts: Path | None, relative: Any) -> Path | None:
    if artifacts is None or not isinstance(relative, str) or not relative:
        return None
    try:
        resolved = (artifacts / relative).resolve()
        root = artifacts.resolve()
    except OSError:
        return None
    if root != resolved and root not in resolved.parents:
        return None
    return resolved if resolved.is_file() else None


def _load_trace(
    runtime: EvaluationRuntime, run: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _safe_trace_path(_artifacts_dir(runtime, run), row.get("trace_path"))
    if path is None:
        return None
    try:
        size = path.stat().st_size
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if size > MAX_TRACE_BYTES:
        try:
            from eval.trace import truncate_trace

            payload = truncate_trace(payload, max_bytes=MAX_TRACE_BYTES)
        except Exception:  # noqa: BLE001 — без модуля trace честно помечаем обрезку
            payload["truncated"] = True
    return payload


def _trace_turns(trace: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(trace, Mapping):
        return []
    turns = trace.get("turns")
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns if isinstance(turn, Mapping)]


def _year_status(
    test: Mapping[str, Any] | None,
    trace: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    turns = _trace_turns(trace)
    covered: set[int] = set()
    missing: set[int] = set()
    for turn in turns:
        for year in turn.get("covered_years") or []:
            if isinstance(year, int) and not isinstance(year, bool):
                covered.add(year)
        for year in turn.get("missing_years") or []:
            if isinstance(year, int) and not isinstance(year, bool):
                missing.add(year)
    expected = (test or {}).get("expected") or {}
    required: list[int] = []
    absent: list[int] = []
    if isinstance(expected, Mapping):
        for year in expected.get("required_years") or []:
            if isinstance(year, int) and not isinstance(year, bool):
                required.append(year)
        for year in expected.get("missing_years") or []:
            if isinstance(year, int) and not isinstance(year, bool):
                absent.append(year)
    rows: list[dict[str, Any]] = []
    for year in sorted(set(required) | covered | missing):
        if year in absent:
            status = "year_absent"
        elif year in covered:
            status = "covered"
        elif year in missing:
            status = "not_found"
        elif year in required:
            status = "not_found" if turns else "unknown"
        else:
            status = "unknown"
        rows.append(
            {
                "year": year,
                "status": status,
                "expected": year in required or year in absent,
            }
        )
    return rows


@router.get("/evaluation/runs/{run_id}/tests/{test_id}", tags=["evaluation"])
def get_run_test(
    run_id: str,
    test_id: str,
    runtime: RuntimeDep,
    owner: Owner,
    retrieval_mode: str | None = None,
    attempt_index: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Детали результата одного теста."""
    run = _get_run_or_404(runtime, run_id)
    mode, row = _resolve_result(runtime, run_id, test_id, retrieval_mode, attempt_index)
    trace = _load_trace(runtime, run, row)
    dataset = runtime.dataset()
    test = dataset.test_by_id(test_id)
    case = row.get("case") if isinstance(row.get("case"), Mapping) else {}
    turns = _trace_turns(trace)
    query_plan = None
    for turn in turns:
        if isinstance(turn.get("query_plan"), Mapping):
            query_plan = turn["query_plan"]
            break
    retrieval_calls: list[dict[str, Any]] = []
    for turn in turns:
        turn_index = turn.get("turn_index")
        for call in turn.get("retrieval_calls") or []:
            if isinstance(call, Mapping):
                retrieval_calls.append({**call, "turn_index": turn_index})
    manifest: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        for chunk_id in turn.get("evidence_manifest") or []:
            text = str(chunk_id)
            if text not in seen:
                seen.add(text)
                manifest.append(text)
    context_ids: list[str] | None = None
    context_source: str | None = None
    real_context = False
    if turns:
        collected: list[str] = []
        for turn in turns:
            source = turn.get("context_source")
            if context_source is None:
                context_source = source
            if source == "evidence_manifest":
                continue
            real_context = True
            for chunk_id in turn.get("context_chunk_ids") or []:
                text = str(chunk_id)
                if text not in collected:
                    collected.append(text)
        context_ids = collected if real_context else None
    answer: dict[str, Any] = {}
    for turn in reversed(turns):
        candidate = turn.get("answer")
        if isinstance(candidate, Mapping) and candidate.get("text") is not None:
            answer = dict(candidate)
            break
    if not answer:
        answer = {
            "status": case.get("actual_status"),
            "text": None,
            "citations": [],
            "facts": [],
            "warnings": [],
            "calculations": [],
        }
    loss_stage = (case.get("retrieval") or {}).get("loss_stage")
    critical = row.get("critical_errors") or []
    likely_cause = _LOSS_CAUSE.get(str(loss_stage)) if loss_stage else None
    if likely_cause is None and critical:
        likely_cause = "answer"
    notes = [str(item.get("code")) for item in critical if item.get("code")]
    trace_url = None
    if trace is not None:
        trace_url = (
            f"/api/evaluation/runs/{run_id}/tests/{test_id}/trace"
            f"?retrieval_mode={mode}&attempt_index={int(row['attempt_index'])}"
        )
    result = dict(case)
    result.setdefault("status", row.get("status"))
    result.setdefault("pass", bool(row.get("pass")))
    return {
        "test": (
            {
                "id": test.get("id"),
                "suite": test.get("suite"),
                "split": test.get("split"),
                "category": test.get("category"),
                "difficulty": test.get("difficulty"),
                "question": test.get("question"),
                "turns": test.get("turns"),
                "expected_status": test.get("expected_status"),
                "expected": test.get("expected") or {},
                "gold_evidence": test.get("gold_evidence") or {},
                "reference_answer": test.get("reference_answer"),
                "checks": test.get("checks") or {},
            }
            if test is not None
            else {
                "id": test_id,
                "suite": row.get("suite"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "question": None,
                "expected_status": case.get("expected_status"),
                "expected": {},
                "gold_evidence": {},
                "reference_answer": None,
            }
        ),
        "run_id": run_id,
        "retrieval_mode": mode,
        "attempt_index": int(row.get("attempt_index") or 0),
        "result": result,
        "query_plan": query_plan,
        "retrieval_calls": retrieval_calls,
        "evidence": {
            "manifest": manifest,
            "context_chunk_ids": context_ids,
            "context_source": context_source,
            "coverage": _year_status(test, trace),
            "loss_stage": loss_stage,
        },
        "answer": answer,
        "evaluation": {
            "pass": bool(row.get("pass")),
            "critical_errors": critical,
            "warnings": row.get("case", {}).get("warnings") if isinstance(row.get("case"), Mapping) else [],
            "rubric": case.get("rubric_total"),
            "automatic_pass": (case.get("answer") or {}).get("automatic_pass"),
        },
        "diagnostics": {
            "likely_cause": likely_cause,
            "notes": notes,
        },
        "trace_available": trace is not None,
        "trace_url": trace_url,
        "trace_truncated": bool((trace or {}).get("truncated")),
    }


@router.get("/evaluation/runs/{run_id}/tests/{test_id}/trace", tags=["evaluation"])
def get_run_test_trace(
    run_id: str,
    test_id: str,
    runtime: RuntimeDep,
    owner: Owner,
    retrieval_mode: str | None = None,
    attempt_index: int = Query(0, ge=0),
) -> JSONResponse:
    """Trace одного результата."""
    run = _get_run_or_404(runtime, run_id)
    mode, row = _resolve_result(runtime, run_id, test_id, retrieval_mode, attempt_index)
    trace = _load_trace(runtime, run, row)
    if trace is None:
        raise ApiError(
            404,
            "test_not_found",
            "Trace для этого результата недоступен.",
            details={
                "test_id": test_id,
                "retrieval_mode": mode,
                "attempt_index": int(row.get("attempt_index") or 0),
            },
        )
    return JSONResponse(
        content=trace,
        headers={"Cache-Control": "private, no-store"},
    )


@router.get("/evaluation/runs/{run_id}/report", tags=["evaluation"])
def get_run_report(run_id: str, runtime: RuntimeDep, owner: Owner) -> Response:
    """Отчёт прогона."""
    row = _get_run_or_404(runtime, run_id)
    path = _report_path(runtime, row)
    if path is None:
        raise ApiError(404, "report_not_found", "Отчёт запуска не найден.")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise ApiError(404, "report_not_found", "Отчёт запуска не найден.") from None
    if path.suffix == ".html":
        return HTMLResponse(
            content=text,
            headers={
                "Cache-Control": "private, no-store",
                "Content-Security-Policy": (
                    "sandbox; default-src 'none'; style-src 'unsafe-inline'"
                ),
            },
        )
    return Response(
        content=text,
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "private, no-store"},
    )


@router.post("/evaluation/runs/{run_id}/cancel", tags=["evaluation"])
def cancel_run(run_id: str, runtime: RuntimeDep, owner: Owner) -> dict[str, Any]:
    """Запросить отмену прогона."""
    if not runtime.config.eval_cancel_available():
        raise ApiError(
            409, "cancel_not_supported", "Отмена запусков отключена конфигурацией."
        )
    _get_run_or_404(runtime, run_id)
    status = runtime.store.request_cancel(run_id)
    if status is None:
        raise ApiError(
            409,
            "cancel_not_supported",
            "Запуск уже завершён; отмена не поддерживается.",
        )
    runtime.worker.cancel(run_id, status=status)
    return {"id": run_id, "status": status, "cancel_supported": True}


# ---------------------------------------------------------------------------
# Test catalog
# ---------------------------------------------------------------------------


@router.get("/evaluation/tests", tags=["evaluation"])
def list_tests(
    runtime: RuntimeDep,
    owner: Owner,
    difficulty: str | None = None,
    category: str | None = None,
    suite: str | None = None,
    status: str | None = None,
    critical: bool | None = None,
    q: str | None = None,
    retrieval_mode: str = "hybrid",
    run_id: str | None = None,
    limit: int = Query(50, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Каталог тестов датасета."""
    if retrieval_mode == "all":
        raise ApiError(
            422,
            "invalid_retrieval_mode",
            "Каталог не смешивает режимы: укажите один retrieval_mode.",
            details={"allowed": list(RETRIEVAL_MODES)},
        )
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ApiError(422, "invalid_retrieval_mode", "Неизвестный режим retrieval.")
    dataset = runtime.dataset()
    if not dataset.available:
        return {
            "items": [],
            "total": 0,
            "next_offset": None,
            "dataset_available": False,
            "run_id": None,
            "retrieval_mode": retrieval_mode,
        }
    source_run = (
        runtime.store.get_run(run_id) if run_id else runtime.store.last_completed_run()
    )
    if run_id and source_run is None:
        raise ApiError(404, "run_not_found", "Запуск не найден.")
    results = (
        runtime.store.last_results_for_run(str(source_run["id"]), retrieval_mode)
        if source_run is not None
        else {}
    )
    query = (q or "").strip().casefold()
    items: list[dict[str, Any]] = []
    for test in dataset.tests:
        if difficulty and test.get("difficulty") != difficulty:
            continue
        if category and test.get("category") != category:
            continue
        if suite and test.get("suite") != suite and not (
            suite == "smoke" and "smoke" in (test.get("tags") or [])
        ):
            continue
        question = str(test.get("question") or "")
        if suite == "multi_turn" and not question:
            turns = test.get("turns") or []
            if turns and isinstance(turns[0], Mapping):
                question = str(turns[0].get("question") or "")
        if query and query not in str(test.get("id") or "").casefold() and query not in question.casefold():
            continue
        result = results.get(str(test.get("id")))
        if status == "not_run" and result is not None:
            continue
        if status == "passed" and (result is None or not result.get("pass")):
            continue
        if status == "failed" and (result is None or result.get("pass")):
            continue
        if critical is not None:
            has_critical = bool(result and result.get("critical_errors"))
            if has_critical != critical:
                continue
        items.append(
            {
                "id": test.get("id"),
                "suite": test.get("suite"),
                "category": test.get("category"),
                "difficulty": test.get("difficulty"),
                "split": test.get("split"),
                "question": question,
                "last_run": (
                    {
                        "run_id": source_run["id"],
                        "created_at": source_run.get("created_at"),
                        "retrieval_mode": retrieval_mode,
                        "pass": bool(result.get("pass")),
                        "status": result.get("status"),
                        "critical_errors": result.get("critical_errors") or [],
                        "trace_available": bool(result.get("trace_available")),
                    }
                    if result is not None and source_run is not None
                    else None
                ),
            }
        )
    page_limit, page_offset = _page(limit, offset)
    page = items[page_offset : page_offset + page_limit]
    next_offset = page_offset + len(page) if page_offset + len(page) < len(items) else None
    return {
        "items": page,
        "total": len(items),
        "next_offset": next_offset,
        "dataset_available": True,
        "run_id": source_run["id"] if source_run is not None else None,
        "retrieval_mode": retrieval_mode,
    }


@router.get("/evaluation/tests/{test_id}", tags=["evaluation"])
def get_test(test_id: str, runtime: RuntimeDep, owner: Owner) -> dict[str, Any]:
    """Определение теста по ID."""
    dataset = runtime.dataset()
    test = dataset.test_by_id(test_id) if dataset.available else None
    history = runtime.store.test_history(test_id)
    if test is None and not history:
        raise ApiError(404, "test_not_found", "Тест не найден.")
    return {
        "test": (
            {
                "id": test.get("id"),
                "suite": test.get("suite"),
                "split": test.get("split"),
                "category": test.get("category"),
                "difficulty": test.get("difficulty"),
                "question": test.get("question"),
                "turns": test.get("turns"),
                "expected_status": test.get("expected_status"),
                "expected": test.get("expected") or {},
                "gold_evidence": test.get("gold_evidence") or {},
                "reference_answer": test.get("reference_answer"),
                "checks": test.get("checks") or {},
                "tags": test.get("tags") or [],
            }
            if test is not None
            else None
        ),
        "history": history,
        "dataset_available": dataset.available,
    }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pass_rate", ("pass_rate",)),
    ("execution_coverage", ("execution_coverage",)),
    ("pass_at_1", ("pass_at_1",)),
    ("recall_at_5", ("groups", "retrieval", "recall_at_5")),
    ("recall_at_10", ("groups", "retrieval", "recall_at_10")),
    ("mrr", ("groups", "retrieval", "mrr")),
    ("evidence_coverage", ("groups", "retrieval", "evidence_coverage")),
    ("context_coverage", ("groups", "retrieval", "context_coverage")),
    ("automatic_pass_rate", ("groups", "answer", "automatic_pass_rate")),
    ("rubric_mean", ("groups", "answer", "rubric_mean")),
    ("fact_accuracy", ("groups", "answer", "fact_accuracy")),
    ("citation_validity", ("groups", "answer", "citation_validity")),
    ("citation_coverage", ("groups", "answer", "citation_coverage")),
    ("dialogue_success_rate", ("groups", "dialogue", "scenario_success_rate")),
    ("mean_latency_ms", ("efficiency", "mean_latency_ms")),
)


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _compare_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(row) or {}
    versions = _run_versions(row, summary)
    return {
        "id": row.get("id"),
        "status": row.get("status"),
        "suite": row.get("suite"),
        "split": row.get("split"),
        "retrieval_mode": row.get("retrieval_mode"),
        "created_at": row.get("created_at"),
        "git_sha": versions["git_sha"] or None,
        "git_dirty": versions["git_dirty"],
        "dataset_sha256": row.get("dataset_sha256"),
        "corpus_content_sha256": row.get("corpus_content_sha256"),
        "corpus_fingerprint": versions["corpus_fingerprint"],
        "llm_model": versions["llm_model"],
    }


@router.get("/evaluation/compare", tags=["evaluation"])
def compare_runs(
    runtime: RuntimeDep,
    owner: Owner,
    base: str,
    candidate: str,
) -> dict[str, Any]:
    """Сравнить два прогона и вернуть дельты."""
    base_row = _get_run_or_404(runtime, base)
    candidate_row = _get_run_or_404(runtime, candidate)
    base_summary = _summary(base_row)
    candidate_summary = _summary(candidate_row)
    if (
        base_row.get("status") != "completed"
        or candidate_row.get("status") != "completed"
        or base_summary is None
        or candidate_summary is None
    ):
        raise ApiError(
            409,
            "run_not_finished",
            "Сравнение доступно только для завершённых запусков с summary.",
        )

    reasons: list[str] = []
    if (base_row.get("dataset_sha256") or "") != (candidate_row.get("dataset_sha256") or ""):
        reasons.append("dataset_sha256_mismatch")
    if (base_row.get("corpus_content_sha256") or "") != (
        candidate_row.get("corpus_content_sha256") or ""
    ):
        reasons.append("corpus_content_sha256_mismatch")
    if (base_row.get("suite"), base_row.get("split")) != (
        candidate_row.get("suite"),
        candidate_row.get("split"),
    ):
        reasons.append("suite_or_split_mismatch")
    base_suites = (base_summary.get("dataset") or {}).get("suites") or {}
    candidate_suites = (candidate_summary.get("dataset") or {}).get("suites") or {}
    if base_suites and candidate_suites and base_suites != candidate_suites:
        reasons.append("test_set_mismatch")
    comparable = not reasons

    version_differences: list[str] = []
    base_versions = _run_versions(base_row, base_summary)
    candidate_versions = _run_versions(candidate_row, candidate_summary)
    for key, reason in (
        ("git_sha", "git_sha_differs"),
        ("git_dirty", "git_dirty_differs"),
        ("llm_model", "llm_model_differs"),
        ("reranker_model", "reranker_model_differs"),
        ("judge_model", "judge_model_differs"),
        ("config_fingerprint", "retrieval_config_differs"),
        ("corpus_fingerprint", "corpus_index_fingerprint_differs"),
    ):
        if base_versions.get(key) != candidate_versions.get(key):
            version_differences.append(reason)
    if (
        (base_summary.get("config") or {}).get("retrieval_mode")
        != (candidate_summary.get("config") or {}).get("retrieval_mode")
    ):
        if "retrieval_mode_differs" not in version_differences:
            version_differences.append("retrieval_mode_differs")

    metrics: list[dict[str, Any]] = []
    for key, path in _METRICS:
        before = _number(_dig(base_summary, path))
        after = _number(_dig(candidate_summary, path))
        delta = (
            after - before
            if comparable and before is not None and after is not None
            else None
        )
        metrics.append(
            {"key": key, "base": before, "candidate": after, "delta": delta}
        )

    base_results = {
        (
            str(row["test_id"]),
            str(row["retrieval_mode"]),
            int(row["attempt_index"]),
        ): row
        for row in runtime.store.all_results(base)
    }
    candidate_results = {
        (
            str(row["test_id"]),
            str(row["retrieval_mode"]),
            int(row["attempt_index"]),
        ): row
        for row in runtime.store.all_results(candidate)
    }
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    test_changes: list[dict[str, Any]] = []
    for key in sorted(set(base_results) & set(candidate_results)):
        before = base_results[key]
        after = candidate_results[key]
        if bool(before.get("pass")) == bool(after.get("pass")):
            continue
        from_state = "pass" if before.get("pass") else "fail"
        to_state = "pass" if after.get("pass") else "fail"
        reasons_list = [
            str(item.get("code"))
            for item in (after.get("critical_errors") or [])
            if isinstance(item, Mapping) and item.get("code")
        ]
        change = {
            "test_id": key[0],
            "retrieval_mode": key[1],
            "attempt_index": key[2],
            "from": from_state,
            "to": to_state,
            "reasons": reasons_list,
            "difficulty": after.get("difficulty") or before.get("difficulty"),
            "suite": after.get("suite") or before.get("suite"),
        }
        test_changes.append(change)
        if from_state == "pass":
            regressions.append(change)
        else:
            improvements.append(change)

    return {
        "comparable": comparable,
        "reasons": reasons,
        "version_differences": version_differences,
        "base": _compare_meta(base_row),
        "candidate": _compare_meta(candidate_row),
        "metrics": metrics,
        "regressions": regressions,
        "improvements": improvements,
        "test_changes": test_changes,
    }
