"""Runner benchmark: запуск suite через публичный ``AgentService.ask()``.

CLI:
    .venv\\Scripts\\python.exe -m eval.runner --validate-only
    .venv\\Scripts\\python.exe -m eval.runner --suite smoke
    .venv\\Scripts\\python.exe -m eval.runner --suite smoke --fake-service

Runner не передаёт агенту ``expected``/``gold_evidence`` и не меняет его
поведение. Один сценарий получает один ``conversation_id``. Все артефакты
пишутся сначала во временный каталог и атомарно переименовываются в
``eval/results/{run_id}/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from eval.dataset import (
    DatasetBundle,
    DatasetError,
    chunk_store as make_chunk_store,
    corpus_content_sha256,
    corpus_documents,
    corpus_years,
    load_chunks,
    validate_dataset,
)
from eval.evaluators.answer import evaluate_answer
from eval.evaluators.dialogue import evaluate_dialogue
from eval.evaluators.retrieval import evaluate_retrieval
from eval.metrics import aggregate_mean, gold_groups
from eval.report import (
    atomic_write_json,
    atomic_write_text,
    write_report,
)
from eval.trace import build_trace, build_turn_trace, trace_path
from src.core.llm_lock import (
    LLMBusy,
    LLMExecutionLock,
    LLMLockUnavailable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "dev" / "chunks"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUITES = ("smoke", "full", "easy", "medium", "hard", "multi_turn")
RETRIEVAL_MODES = ("hybrid", "dense", "keyword")
_SECRET_TEXT = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=]\s*\S+|"
    r"access[_-]?token\s*[:=]\s*\S+|bearer\s+[A-Za-z0-9._~+/=-]+)"
)


class RunnerError(RuntimeError):
    """Ошибка runner/окружения с машинным кодом."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class TurnTimeout(RunnerError):
    """Turn не завершился за отведённое время."""

    def __init__(self, seconds: float) -> None:
        super().__init__("turn_timeout", f"Turn превысил лимит {seconds:g} с")


def safe_error_text(exc: BaseException) -> str:
    """Текст ошибки без ключей и тел ответов."""
    message = _SECRET_TEXT.sub("[REDACTED]", str(exc))
    return message[:500]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_info(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Git SHA и dirty без падения вне git-репозитория."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        revision = sha.stdout.strip() or "unknown"
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
        dirty = False
    return {"sha": revision, "dirty": dirty}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_filters(raw: Any) -> Any:
    """Собрать ``SearchFilters`` из JSON-поля ``filters``."""
    from src.retrieval.hybrid_retriever import SearchFilters

    if raw is None:
        return SearchFilters()
    if not isinstance(raw, Mapping):
        raise RunnerError("invalid_filters", "filters должен быть объектом")
    set_fields = {
        "years",
        "report_periods",
        "issuers",
        "kinds",
        "allowed_doc_ids",
    }
    values: dict[str, Any] = {}
    for key, value in raw.items():
        values[key] = set(value) if key in set_fields and value is not None else value
    try:
        return SearchFilters(**values).normalized()
    except (TypeError, ValueError) as exc:
        raise RunnerError("invalid_filters", f"Некорректные filters: {exc}") from exc


def _test_filters(spec: Mapping[str, Any]) -> Any:
    return _safe_filters(spec.get("filters"))


@dataclass
class FakeAgentService:
    """Детерминированный offline-service для проверки артефактов.

    Он делает простой лексический поиск по chunk store, не вызывает LLM и не
    получает эталон. Это не оценка качества агента: отчёт помечается
    ``fake_service=true``.
    """

    chunks: Sequence[Mapping[str, Any]]
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    supported_retrieval_modes: tuple[str, ...] = RETRIEVAL_MODES

    def __post_init__(self) -> None:
        self._store = make_chunk_store(self.chunks)

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", text.casefold())
            if token not in {
                "какой", "какая", "какие", "какова", "что", "как", "для",
                "это", "было", "были", "был", "при", "или", "чем", "над",
            }
        }

    def _search(
        self, question: str, filters: Any, mode: str
    ) -> list[Mapping[str, Any]]:
        years = set(getattr(filters, "years", None) or set())
        if getattr(filters, "year_from", None):
            years = set(range(filters.year_from, (filters.year_to or filters.year_from) + 1))
        docs = set(getattr(filters, "allowed_doc_ids", None) or set())
        query_tokens = self._tokens(question)
        ranked: list[tuple[float, Mapping[str, Any]]] = []
        for chunk in self.chunks:
            if years and chunk.get("year") not in years:
                continue
            if docs and chunk.get("doc_id") not in docs:
                continue
            text_tokens = self._tokens(str(chunk.get("text") or ""))
            overlap = len(query_tokens & text_tokens)
            if overlap <= 0:
                continue
            ranked.append((float(overlap), chunk))
        ranked.sort(key=lambda item: (-item[0], str(item[1].get("chunk_id"))))
        return [chunk for _, chunk in ranked[: self.top_k]]

    def ask(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: Any = None,
    ) -> Any:
        from src.agent.state import AgentRunResult, Answer, Citation, ToolEvent

        started = time.perf_counter()
        mode = self.retrieval_mode
        hits = self._search(query, filters, mode)
        now = time.perf_counter()
        chunk_ids = [str(chunk.get("chunk_id")) for chunk in hits]
        citations = [
            Citation(
                chunk_id=str(chunk.get("chunk_id")),
                doc_id=chunk.get("doc_id"),
                year=chunk.get("year"),
                page_pdf=chunk.get("page_pdf"),
                page_label=chunk.get("page_label"),
                section_path=chunk.get("section_path"),
                quote=str(chunk.get("text") or "")[:300],
            )
            for chunk in hits[:3]
        ]
        if chunk_ids:
            answer = Answer(
                "answer",
                "Демонстрационный ответ fake-service: найдено фрагментов — "
                f"{len(chunk_ids)}. {str(hits[0].get('text') or '')[:300]}",
                citations=citations,
            )
            status = "completed"
        else:
            answer = Answer(
                "abstain",
                "В доступных материалах не найдено подтверждение для ответа.",
                warnings=["no_evidence"],
            )
            status = "abstained"
        years = sorted(
            {
                int(chunk["year"])
                for chunk in hits
                if isinstance(chunk.get("year"), int) and not isinstance(chunk.get("year"), bool)
            }
        )
        stage = [
            {"chunk_id": chunk_id, "rank": index, "score": float(len(chunk_ids) - index + 1)}
            for index, chunk_id in enumerate(chunk_ids, start=1)
        ]
        stages: dict[str, Any] = {
            "dense": stage if mode in {"hybrid", "dense"} else None,
            "bm25": stage if mode in {"hybrid", "keyword"} else None,
            "rrf": stage,
            "rerank": stage,
            "returned": stage,
        }
        available = {
            name: items is not None for name, items in stages.items()
        }
        diagnostics = {
            "query_plan": {
                "intent": "factoid",
                "years": sorted(getattr(filters, "years", None) or []) if filters else [],
                "year_from": getattr(filters, "year_from", None) if filters else None,
                "year_to": getattr(filters, "year_to", None) if filters else None,
                "section_filter": getattr(filters, "section_path_prefix", None) if filters else None,
                "entities": [],
                "needs_rewriting": False,
                "needs_graph": False,
                "answer_mode": "concise",
            },
            "search_attempts": [
                {
                    "kind": "broad",
                    "year": years[0] if len(years) == 1 else None,
                    "years": years,
                    "section_path": None,
                    "result_count": len(chunk_ids),
                    "covered_years": years,
                }
            ],
            "retrieval_calls": [
                {
                    "retrieval_mode": mode,
                    "counts": {
                        "dense": len(stage) if stages["dense"] is not None else 0,
                        "bm25": len(stage) if stages["bm25"] is not None else 0,
                        "rrf": len(stage),
                        "rerank": len(stage),
                        "returned": len(stage),
                    },
                    "stages": stages,
                    "available": available,
                    "fallback": {"reranker": False, "dense": False, "bm25": False},
                    "unavailable": {},
                }
            ],
            "evidence_manifest": chunk_ids,
            "context_chunk_ids": chunk_ids[:3],
            "context_source": "context",
            "coverage": {
                "covered_years": years,
                "missing_years": [],
            },
            "verification": None,
            "structural_changes": [],
        }
        tool_event = ToolEvent(
            tool_name="search_reports",
            arguments={
                "query": query,
                "retrieval_mode": mode,
                "top_k": self.top_k,
                "filters": filters,
            },
            status="completed",
            result_count=len(chunk_ids),
            duration_ms=(now - started) * 1000,
        )
        elapsed = (time.perf_counter() - started) * 1000
        return AgentRunResult(
            request_id=f"fake-{conversation_id}-{int(time.time() * 1000)}",
            conversation_id=conversation_id,
            answer=answer,
            tool_events=[tool_event],
            effective_filters=filters,
            warnings=["fake_service"],
            timings={"total_ms": elapsed, "answer_ms": 0.0},
            status=status,
            calculations=list(answer.calculations),
            diagnostics=diagnostics,
        )


def _service_modes(service: Any) -> set[str]:
    modes = getattr(service, "supported_retrieval_modes", None)
    if modes is None:
        return {"hybrid"}
    return {str(mode) for mode in modes}


def _ask_with_timeout(
    service: Any,
    question: str,
    *,
    conversation_id: str,
    filters: Any,
    timeout: float | None,
    llm_lock: Any | None = None,
    wait_for_llm: bool = True,
) -> Any:
    """Вызов ``ask()`` в daemon-потоке, чтобы не зависнуть навсегда.

    Lock берётся на один turn в **вызывающем** потоке и освобождается в
    ``finally``. Если turn не уложился в timeout, abandoned-поток больше не
    удерживает lock и не блокирует следующие тесты на LLM_LOCK_TIMEOUT_S.
    """
    if llm_lock is not None:
        llm_lock.acquire(wait=wait_for_llm)
    try:
        if timeout is None or timeout <= 0:
            return service.ask(
                question, conversation_id=conversation_id, filters=filters
            )
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)

        def call() -> None:
            try:
                result_queue.put(
                    service.ask(
                        question,
                        conversation_id=conversation_id,
                        filters=filters,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - ошибка переносится в runner
                result_queue.put(exc)

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TurnTimeout(timeout)
        value = result_queue.get()
        if isinstance(value, BaseException):
            raise value
        return value
    finally:
        if llm_lock is not None:
            llm_lock.release()


def _select_tests(
    bundle: DatasetBundle,
    *,
    suite: str,
    split: str,
    ids: Sequence[str] | None,
    tag: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if suite not in SUITES:
        raise DatasetError("invalid_suite", f"Неизвестный suite: {suite}")
    if split not in {"dev", "holdout", "all"}:
        raise DatasetError("invalid_split", f"Неизвестный split: {split}")
    selected = list(bundle.tests)
    if suite == "smoke":
        selected = [test for test in selected if "smoke" in (test.get("tags") or [])]
        split = "dev"
    elif suite != "full":
        selected = [test for test in selected if test.get("suite") == suite]
    if split != "all":
        selected = [test for test in selected if test.get("split") == split]
    if tag:
        selected = [test for test in selected if tag in (test.get("tags") or [])]
    if ids:
        wanted = list(dict.fromkeys(str(item) for item in ids))
        selected = [test for test in selected if test.get("id") in wanted]
    if limit is not None:
        if limit < 0:
            raise DatasetError("invalid_limit", "--limit не может быть отрицательным")
        selected = selected[:limit]
    if not selected:
        raise DatasetError("no_tests_selected", "Фильтры не выбрали ни одного теста")
    return selected


def _expand_modes(
    tests: Sequence[Mapping[str, Any]], retrieval_mode: str, service: Any
) -> list[tuple[Mapping[str, Any], str, bool]]:
    if retrieval_mode not in {*RETRIEVAL_MODES, "all"}:
        raise DatasetError("invalid_retrieval_mode", "Режим вне hybrid/dense/keyword/all")
    modes = list(RETRIEVAL_MODES) if retrieval_mode == "all" else [retrieval_mode]
    supported = _service_modes(service)
    pairs: list[tuple[Mapping[str, Any], str, bool]] = []
    for test in tests:
        for mode in modes:
            pairs.append((test, mode, mode in supported))
    return pairs


def _safe_conversation_id(run_id: str, test_id: str, mode: str, attempt: int) -> str:
    return f"bench-{run_id}-{test_id}-{mode}-{attempt}"


def _case_status(result_status: str) -> str:
    return "failed" if result_status == "failed" else "completed"


def _test_gold_per_turn(
    test: Mapping[str, Any],
    turn_index: int | None = None,
) -> Mapping[str, Any]:
    turns = test.get("turns")
    if isinstance(turns, list) and turns:
        index = 0 if turn_index is None else turn_index
        if 0 <= index < len(turns) and isinstance(turns[index], Mapping):
            gold = turns[index].get("gold_evidence")
            return gold if isinstance(gold, Mapping) else {}
        return {}
    gold = test.get("gold_evidence")
    return gold if isinstance(gold, Mapping) else {}


def _evidence_ok(test: Mapping[str, Any], retrieval: Mapping[str, Any]) -> bool:
    """Проверить min_coverage по каждому turn'у (или по тесту целиком)."""
    test_turns = test.get("turns")
    if isinstance(test_turns, list) and test_turns:
        rows = {
            int(row.get("turn_index", index)): row
            for index, row in enumerate(retrieval.get("turns", []) or [])
            if isinstance(row, Mapping)
        }
        for index, turn in enumerate(test_turns):
            if not isinstance(turn, Mapping):
                continue
            gold = turn.get("gold_evidence")
            if not isinstance(gold, Mapping) or not gold_groups(gold):
                continue
            coverage = (rows.get(index) or {}).get("evidence_coverage")
            if coverage is None:
                continue
            minimum = float(gold.get("min_coverage", 1.0) or 0)
            if coverage < minimum:
                return False
        return True
    gold = test.get("gold_evidence")
    coverage = retrieval.get("evidence_coverage")
    if not isinstance(gold, Mapping) or coverage is None:
        return True
    minimum = float(gold.get("min_coverage", 1.0) or 0)
    return coverage >= minimum


def _retrieval_csv_rows(
    *,
    test: Mapping[str, Any],
    trace: Mapping[str, Any],
    retrieval_mode: str,
    attempt_index: int,
    loss_stage: str | None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for turn in trace.get("turns", []) or []:
        if not isinstance(turn, Mapping):
            continue
        turn_index = int(turn.get("turn_index", 0))
        gold_ids = {
            chunk_id
            for group in gold_groups(_test_gold_per_turn(test, turn_index))
            for chunk_id in group.get("any_of", [])
        }
        for call in turn.get("retrieval_calls", []) or []:
            if not isinstance(call, Mapping):
                continue
            stages = call.get("stages")
            if not isinstance(stages, Mapping):
                continue
            for stage, items in stages.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, Mapping) or not item.get("chunk_id"):
                        continue
                    chunk_id = str(item["chunk_id"])
                    rows.append(
                        [
                            test.get("id"),
                            retrieval_mode,
                            attempt_index,
                            call.get("call_index"),
                            stage,
                            item.get("rank"),
                            chunk_id,
                            item.get("score"),
                            1 if chunk_id in gold_ids else 0,
                            loss_stage,
                        ]
                    )
        context_ids = turn.get("context_chunk_ids")
        if turn.get("context_source") != "evidence_manifest" and isinstance(context_ids, list):
            for rank, chunk_id in enumerate(context_ids, start=1):
                rows.append(
                    [
                        test.get("id"),
                        retrieval_mode,
                        attempt_index,
                        None,
                        "context",
                        rank,
                        str(chunk_id),
                        None,
                        1 if str(chunk_id) in gold_ids else 0,
                        loss_stage,
                    ]
                )
    return rows


def _critical_aggregate(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for case in cases:
        for item in case.get("critical_errors", []) or []:
            code = str(item.get("code") or "unknown")
            row = counts.setdefault(code, {"code": code, "count": 0, "test_ids": []})
            row["count"] += 1
            test_id = str(case.get("id"))
            if test_id not in row["test_ids"]:
                row["test_ids"].append(test_id)
    return sorted(counts.values(), key=lambda item: (-item["count"], item["code"]))


def _failed_check_rows(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    answer = case.get("answer")
    if not isinstance(answer, Mapping):
        return []
    rows = answer.get("failed_checks")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _warning_aggregate(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Сводка warning-кодов по cases, чтобы отчёт не скрывал предупреждения.

    ``citation_quote_mismatch`` при ``strict_quotes=false`` не валит тест, но
    обязан быть виден: иначе отчёт показывает pass и умалчивает о проблеме.
    """
    counts: dict[str, dict[str, Any]] = {}
    for case in cases:
        test_id = str(case.get("id"))
        for code in case.get("warnings", []) or []:
            name = str(code or "unknown")
            row = counts.setdefault(
                name, {"code": name, "count": 0, "test_ids": []}
            )
            row["count"] += 1
            if test_id not in row["test_ids"]:
                row["test_ids"].append(test_id)
    return sorted(counts.values(), key=lambda item: (-item["count"], item["code"]))


def _failed_checks_aggregate(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Сводка проваленных checks (включая non-mandatory) с их detail."""
    counts: dict[str, dict[str, Any]] = {}
    for case in cases:
        test_id = str(case.get("id"))
        for check in _failed_check_rows(case):
            name = str(check.get("name") or "unknown")
            row = counts.setdefault(
                name,
                {
                    "name": name,
                    "count": 0,
                    "mandatory": False,
                    "test_ids": [],
                    "details": [],
                },
            )
            row["count"] += 1
            row["mandatory"] = row["mandatory"] or bool(check.get("mandatory"))
            if test_id not in row["test_ids"]:
                row["test_ids"].append(test_id)
            detail = str(check.get("detail") or "").strip()
            if detail and detail not in row["details"] and len(row["details"]) < 10:
                row["details"].append(detail)
    return sorted(counts.values(), key=lambda item: (-item["count"], item["name"]))


def _problem_tests(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    problems = []
    for case in cases:
        if case.get("pass"):
            continue
        if case.get("status") == "skipped":
            continue
        reasons: list[str] = []
        for item in case.get("critical_errors", []) or []:
            code = str(item.get("code"))
            if code and code not in reasons:
                reasons.append(code)
        loss = (case.get("retrieval") or {}).get("loss_stage")
        if loss and loss not in {"ok", None} and loss not in reasons:
            reasons.append(loss)
        # Показываем все warnings, а не только заранее известный список:
        # иначе новый код (например, citation_quote_mismatch) исчезает из
        # отчёта, хотя он есть в cases.jsonl.
        for warning in case.get("warnings", []) or []:
            code = str(warning or "")
            if code and code not in reasons:
                reasons.append(code)
        for check in _failed_check_rows(case):
            name = str(check.get("name") or "")
            if not name:
                continue
            label = f"check_failed:{name}"
            if label not in reasons:
                reasons.append(label)
        if not reasons:
            reasons.append("test_failed")
        problems.append({"id": case.get("id"), "reasons": reasons})
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique = []
    for item in problems:
        key = (str(item["id"]), tuple(item["reasons"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _group_stats(cases: Sequence[Mapping[str, Any]], *, bucket: str) -> dict[str, Any]:
    eligible = [
        case
        for case in cases
        if case.get("status") in {"completed", "failed", "timeout"}
    ]
    executed = eligible
    passed = sum(1 for case in eligible if case.get("pass"))
    return {
        "pass_rate": passed / len(eligible) if eligible else None,
        "eligible": len(eligible),
        "executed": len(executed),
        "passed": passed,
    }


def _build_groups(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    retrieval_cases = [case for case in cases if case.get("retrieval")]
    answer_cases = [
        case for case in cases
        if case.get("status") in {"completed", "failed", "timeout"}
    ]
    multi_cases = [case for case in cases if case.get("suite") == "multi_turn"]
    reasoning_cases = [
        case for case in cases
        if case.get("suite") != "multi_turn"
        and case.get("category") in {"multi_hop", "temporal", "structural"}
    ]

    def metric(key: str) -> float | None:
        return aggregate_mean(
            [
                (case.get("retrieval") or {}).get(key)
                for case in retrieval_cases
            ]
        )

    by_mode: dict[str, Any] = {}
    for mode in RETRIEVAL_MODES:
        mode_cases = [case for case in cases if case.get("retrieval_mode") == mode]
        if not mode_cases:
            continue
        mode_retrieval = [case for case in mode_cases if case.get("retrieval")]
        by_mode[mode] = {
            "recall_at_5": aggregate_mean(
                [(case.get("retrieval") or {}).get("recall_at_5") for case in mode_retrieval]
            ),
            "recall_at_10": aggregate_mean(
                [(case.get("retrieval") or {}).get("recall_at_10") for case in mode_retrieval]
            ),
            "mrr": aggregate_mean(
                [(case.get("retrieval") or {}).get("mrr") for case in mode_retrieval]
            ),
            "evidence_coverage": aggregate_mean(
                [
                    (case.get("retrieval") or {}).get("evidence_coverage")
                    for case in mode_retrieval
                ]
            ),
            **_group_stats(mode_cases, bucket="mode"),
        }

    answer_metrics = {
        key: aggregate_mean(
            [(case.get("answer") or {}).get(key) for case in answer_cases]
        )
        for key in (
            "fact_accuracy",
            "citation_validity",
            "citation_coverage",
            "calculation_accuracy",
            "abstention_accuracy",
            "status_accuracy",
        )
    }
    answer_metrics["automatic_pass_rate"] = _group_stats(answer_cases, bucket="answer")[
        "pass_rate"
    ]
    rubric_values = [
        case.get("rubric_total")
        for case in answer_cases
        if isinstance(case.get("rubric_total"), (int, float))
    ]
    answer_metrics["rubric_mean"] = aggregate_mean(rubric_values)

    dialogue_checks: dict[str, Any] = {}
    for case in multi_cases:
        checks = (case.get("dialogue") or {}).get("checks")
        if not isinstance(checks, Mapping):
            continue
        for name, value in checks.items():
            if value is None:
                continue
            dialogue_checks.setdefault(name, []).append(1.0 if value else 0.0)
    return {
        "retrieval": {
            "recall_at_5": metric("recall_at_5"),
            "recall_at_10": metric("recall_at_10"),
            "mrr": metric("mrr"),
            "ndcg_at_5": metric("ndcg_at_5"),
            "retrieval_coverage": metric("retrieval_coverage"),
            "evidence_coverage": metric("evidence_coverage"),
            "context_coverage": metric("context_coverage"),
            "by_mode": by_mode,
        },
        "answer": answer_metrics,
        "reasoning": {
            "pass_rate": _group_stats(reasoning_cases, bucket="reasoning")["pass_rate"],
            "rubric_mean": None,
            "causality_issues": 0,
        },
        "dialogue": {
            "scenario_success_rate": (
                _group_stats(multi_cases, bucket="dialogue")["pass_rate"]
                if multi_cases
                else None
            ),
            "by_check": {
                name: aggregate_mean(values)
                for name, values in dialogue_checks.items()
            },
        },
    }


def _coverage(cases: Sequence[Mapping[str, Any]], selected: int) -> dict[str, Any]:
    statuses = [case.get("status") for case in cases]
    skipped = sum(1 for status in statuses if status == "skipped")
    errors = sum(1 for status in statuses if status == "error")
    timeouts = sum(1 for status in statuses if status == "timeout")
    executed = sum(
        1 for status in statuses if status in {"completed", "failed", "timeout"}
    )
    unsupported = sum(
        1
        for case in cases
        if case.get("status") == "skipped"
        and (case.get("skip_reason") == "unsupported_mode")
    )
    eligible = selected - skipped - errors
    passed = sum(1 for case in cases if case.get("pass"))
    return {
        "tests_selected": selected,
        "tests_eligible": eligible,
        "executed": executed,
        "passed": passed,
        "failed": max(eligible - passed, 0),
        "skipped": skipped,
        "unsupported": unsupported,
        "timeouts": timeouts,
        "errors": errors,
        "judge_errors": sum(1 for case in cases if case.get("judge_error")),
    }


def _pass_at_1(cases: Sequence[Mapping[str, Any]]) -> float | None:
    first = [case for case in cases if case.get("attempt_index") == 0]
    eligible = [
        case
        for case in first
        if case.get("status") in {"completed", "failed", "timeout"}
    ]
    if not eligible:
        return None
    return sum(1 for case in eligible if case.get("pass")) / len(eligible)


def _stability(cases: Sequence[Mapping[str, Any]], repeat: int) -> float | None:
    if repeat <= 1:
        return None
    pairs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for case in cases:
        if case.get("status") == "skipped":
            continue
        key = (str(case.get("id")), str(case.get("retrieval_mode")))
        pairs.setdefault(key, []).append(case)
    stable = 0
    total = 0
    for rows in pairs.values():
        total += 1
        if len(rows) == repeat and all(row.get("pass") for row in rows):
            stable += 1
    return stable / total if total else None


_REWRITE_STAGE_NAMES = frozenset({"rewrite", "rewritten", "query_variants"})


def _query_variants(turn: Mapping[str, Any]) -> set[str]:
    """Различные запросы, реально отправленные в retrieval на этом turn'е."""
    variants: set[str] = set()
    calls = turn.get("retrieval_calls")
    if not isinstance(calls, list):
        return variants
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        queries = call.get("queries")
        if not isinstance(queries, list):
            continue
        for query in queries:
            if isinstance(query, str) and query.strip():
                variants.add(query.strip().casefold())
    return variants


def _rewriting_observed(turn: Mapping[str, Any]) -> bool:
    """Есть ли в trace наблюдаемая rewriting/query-expansion стадия.

    Учитываются только факты: несколько различных query variants в одном
    retrieval-вызове или непустая стадия с именем ``rewrite``. Один
    ``query_plan.needs_rewriting=true`` — это запрос на переписывание, а не
    доказательство, что оно выполнено, поэтому сам по себе capability не
    включает.
    """
    if len(_query_variants(turn)) >= 2:
        return True
    calls = turn.get("retrieval_calls")
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        stages = call.get("stages")
        if not isinstance(stages, Mapping):
            continue
        for name, items in stages.items():
            if str(name).casefold() in _REWRITE_STAGE_NAMES and isinstance(items, list) and items:
                return True
    return False


def _capabilities(cases: Sequence[Mapping[str, Any]], service: Any, judge: bool) -> dict[str, Any]:
    traces = [case.get("_trace") for case in cases if isinstance(case.get("_trace"), Mapping)]
    turns = [
        turn
        for trace in traces
        for turn in trace.get("turns", []) or []
        if isinstance(turn, Mapping)
    ]
    query_plan = any(
        isinstance(turn.get("query_plan"), Mapping) for turn in turns
    )
    coverage = any(
        turn.get("covered_years") is not None or turn.get("missing_years") is not None
        for turn in turns
    )
    structural = any(
        isinstance(turn.get("structural_changes"), list)
        and bool(turn.get("structural_changes"))
        for turn in turns
    )
    calculations = any(
        bool((turn.get("answer") or {}).get("calculations")) for turn in turns
    )
    graph = any(
        isinstance((call.get("stages") or {}).get("graph"), list)
        for turn in turns
        for call in turn.get("retrieval_calls", []) or []
        if isinstance(call, Mapping)
    )
    verification = any(
        isinstance((turn.get("answer") or {}).get("verification"), Mapping)
        for turn in turns
    )
    return {
        "retrieval_modes": sorted(_service_modes(service)),
        "query_plan": query_plan,
        "coverage": coverage,
        "rewriting": any(_rewriting_observed(turn) for turn in turns),
        "structural_changes": structural,
        "calculations": calculations,
        "graph": graph,
        "verification": verification,
        "judge": bool(judge),
    }


def _service_graph_fingerprint(service: Any) -> str | None:
    """Fingerprint графа, если сервис его явно раскрывает (duck-typing)."""
    candidates = [
        service,
        getattr(service, "tools", None),
        getattr(getattr(service, "tools", None), "graph_search_backend", None),
    ]
    for owner in candidates:
        if owner is None:
            continue
        value = getattr(owner, "graph_fingerprint", None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_graph_fingerprint(graph_dir: Path | str | None = None) -> str | None:
    """Fingerprint активного graph-артефакта или ``None``, если его нет.

    Используется тот же порядок выбора артефакта, что и в
    ``AgentService`` (полный манифест важнее legacy-файла), поэтому
    summary фиксирует граф, который действительно мог быть загружен.
    Никакого fingerprint не синтезируется: отсутствие артефакта — ``None``.
    """
    directory = Path(graph_dir) if graph_dir is not None else DEFAULT_GRAPH_DIR
    if not directory.is_dir():
        return None
    try:
        from src.agent.factory import _graph_artifact_candidates
        from src.graph.graph_builder import load_graph
    except Exception:  # noqa: BLE001 - fingerprint опционален для запуска
        return None
    try:
        candidates = _graph_artifact_candidates(directory)
    except OSError:
        return None
    for path in candidates:
        try:
            loaded = load_graph(path)
        except (OSError, ValueError):
            continue
        fingerprint = getattr(loaded, "graph_fingerprint", None)
        if isinstance(fingerprint, str) and fingerprint:
            return fingerprint
    return None


def _limitations(
    *,
    bundle: DatasetBundle,
    selected: int,
    fake_service: bool,
    judge_requested: bool,
    cases: Sequence[Mapping[str, Any]],
) -> list[str]:
    limitations: list[str] = []
    manifest = bundle.manifest or {}
    if manifest.get("partial"):
        limitations.append(
            f"dev/smoke-набор: {len(bundle.tests)} тестов, не полный 30-тестовый набор; "
            "holdout отсутствует."
        )
    overlap = [warning for warning in bundle.warnings if warning.code == "holdout_gold_overlap"]
    if overlap:
        limitations.append(
            "holdout-gold пересекается с dev-gold: "
            + ", ".join(str(issue.test_id) for issue in overlap)
            + "; holdout-метрики не изолированы от dev."
        )
    if fake_service:
        limitations.append(
            "fake-service: артефакты проверяют инфраструктуру, а не качество реального агента."
        )
    if judge_requested:
        limitations.append("LLM-судья запрошен, но модель судьи не подключена; rubric = null.")
    if any(
        (case.get("retrieval") or {}).get("context_coverage") is None
        and test_context_missing(case)
        for case in cases
    ):
        limitations.append(
            "context_chunk_ids недоступны для части тестов: context coverage = not_measured."
        )
    if any(case.get("status") == "skipped" for case in cases):
        limitations.append("Часть пар (test, retrieval_mode) пропущена как unsupported_mode.")
    if selected < 30:
        limitations.append(
            "Малый набор: pass rate не следует трактовать как статистически значимый."
        )
    return limitations


def test_context_missing(case: Mapping[str, Any]) -> bool:
    trace = case.get("_trace")
    if not isinstance(trace, Mapping):
        return False
    return any(
        turn.get("context_source") == "evidence_manifest"
        for turn in trace.get("turns", []) or []
        if isinstance(turn, Mapping)
    )


def _summary(
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    command: str,
    git: Mapping[str, Any],
    bundle: DatasetBundle,
    chunks: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    selected: int,
    suite: str,
    split: str,
    retrieval_mode: str,
    judge: bool,
    repeat: int,
    config: Mapping[str, Any],
    service: Any,
    limitations: list[str],
    errors: Sequence[Mapping[str, Any]],
    fake_service: bool,
    graph_fingerprint: str | None = None,
) -> dict[str, Any]:
    coverage = _coverage(cases, selected)
    pass_rate = (
        coverage["passed"] / coverage["tests_eligible"]
        if coverage["tests_eligible"]
        else None
    )
    execution_coverage = (
        coverage["executed"] / coverage["tests_selected"]
        if coverage["tests_selected"]
        else None
    )
    public_cases = [
        {key: value for key, value in case.items() if not key.startswith("_")}
        for case in cases
    ]
    corpus_hash = corpus_content_sha256(chunks)
    documents = sorted(corpus_documents(chunks))
    manifest = bundle.manifest or {}
    dataset_version = manifest.get("dataset_version", "1.1")
    config_fingerprint = _sha256_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    by_difficulty: dict[str, Any] = {}
    for name in ("easy", "medium", "hard", "multi_turn"):
        bucket = [
            case
            for case in public_cases
            if (
                case.get("suite") == "multi_turn"
                if name == "multi_turn"
                else case.get("difficulty") == name and case.get("suite") != "multi_turn"
            )
        ]
        by_difficulty[name] = _group_stats(bucket, bucket=name)
    by_category: dict[str, Any] = {}
    for case in public_cases:
        category = str(case.get("category") or "unknown")
        by_category.setdefault(category, []).append(case)
    by_split: dict[str, Any] = {}
    for split_name in ("dev", "holdout"):
        by_split[split_name] = _group_stats(
            [case for case in public_cases if case.get("split") == split_name],
            bucket=split_name,
        )
    by_mode: dict[str, Any] = {}
    for mode in RETRIEVAL_MODES:
        bucket = [case for case in public_cases if case.get("retrieval_mode") == mode]
        if bucket:
            by_mode[mode] = _group_stats(bucket, bucket=mode)

    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": command,
        "suite": suite,
        "split": split,
        "judge": judge,
        "fake_service": fake_service,
        "git": dict(git),
        "dataset": {
            "version": dataset_version,
            "sha256": bundle.dataset_sha256,
            "tests": len(bundle.tests),
            "expected_corpus_sha256": bundle.expected_corpus_sha256,
            "suites": dict(bundle.suite_hashes),
            "warnings": [issue.as_dict() for issue in bundle.warnings],
        },
        "corpus": {
            "content_sha256": corpus_hash,
            "chunk_count": len(chunks),
            "documents": documents,
            "index_fingerprint": manifest.get("corpus_fingerprint"),
            "graph_fingerprint": graph_fingerprint,
        },
        "config": dict(config),
        "config_fingerprint": config_fingerprint,
        "models": {
            "llm": None,
            "embeddings": None,
            "reranker": None,
            "judge": None,
            "generation": None,
        },
        "prompts": {},
        "capabilities": _capabilities(cases, service, judge),
        "coverage": coverage,
        "pass_rate": pass_rate,
        "execution_coverage": execution_coverage,
        "repeat": repeat,
        "pass_at_1": _pass_at_1(cases),
        # stability измеряется только при repeat >= 2; при одном повторе
        # это None, а не копия pass_at_1.
        "stability": _stability(cases, repeat),
        "groups": _build_groups(cases),
        "by_difficulty": by_difficulty,
        "by_category": {
            name: _group_stats(rows, bucket=name)
            for name, rows in sorted(by_category.items())
        },
        "by_split": by_split,
        "by_mode": by_mode,
        "critical_errors": _critical_aggregate(public_cases),
        "warnings": _warning_aggregate(public_cases),
        "failed_checks": _failed_checks_aggregate(public_cases),
        "problem_tests": _problem_tests(public_cases),
        "efficiency": {
            "mean_tool_calls": aggregate_mean(
                [(case.get("efficiency") or {}).get("tool_calls") for case in public_cases]
            ),
            "mean_search_calls": aggregate_mean(
                [(case.get("efficiency") or {}).get("search_calls") for case in public_cases]
            ),
            "mean_latency_ms": aggregate_mean(
                [(case.get("efficiency") or {}).get("latency_ms") for case in public_cases]
            ),
            "llm_calls": None,
            "llm_concurrency": 1,
        },
        "limitations": limitations,
        "errors": list(errors),
    }


def run_suite(
    *,
    suite: str,
    split: str = "all",
    ids: list[str] | None = None,
    retrieval_mode: str = "hybrid",
    judge: bool = False,
    repeat: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Any | None = None,
    service: Any | None = None,
    llm_lock: Any | None = None,
    wait_for_llm: bool = True,
    service_handles_lock: bool = False,
    out_dir: Path | None = None,
    datasets_dir: Path | None = None,
    chunks_dir: Path | None = None,
    graph_dir: Path | None = None,
    chunks: Sequence[Mapping[str, Any]] | None = None,
    limit: int | None = None,
    tag: str | None = None,
    fake_service: bool = False,
    fresh_service: bool = False,
    timeout: float | None = None,
) -> Path:
    """Запустить suite, записать артефакты и вернуть каталог запуска.

    ``llm_lock``/``wait_for_llm``: lock берётся на один turn; если он занят,
    ``wait_for_llm=False`` поднимает ``llm_busy`` (CLI-код 5). CLI без явного
    lock создаёт ``LLMExecutionLock`` сам. Fake/offline-прогон lock не трогает.
    """
    if repeat < 1:
        raise DatasetError("invalid_repeat", "repeat должен быть >= 1")
    datasets = Path(datasets_dir or DEFAULT_DATASETS_DIR)
    if chunks is None:
        chunks_path = Path(chunks_dir or DEFAULT_CHUNKS_DIR)
        if not chunks_path.exists():
            raise RunnerError(
                "chunk_store_missing",
                f"Chunk store не найден: {chunks_path}",
            )
        chunks = load_chunks(chunks_path)

    bundle = validate_dataset(
        datasets,
        chunks=chunks,
        check_corpus=True,
        require_manifest=True,
    )
    selected_tests = _select_tests(
        bundle,
        suite=suite,
        split=split,
        ids=ids,
        tag=tag,
        limit=limit,
    )

    provided_service = service is not None
    if service is None:
        if fake_service:
            service = FakeAgentService(chunks)
        else:
            service = _build_real_service()
    modes = _expand_modes(selected_tests, retrieval_mode, service)
    if not modes:
        raise DatasetError("no_tests_selected", "Нет пар test/retrieval_mode")

    # Fingerprint только фактически используемого графа:
    # - сервис может раскрыть его сам;
    # - реальный AgentService, собранный здесь, читает data/graph;
    # - fake/provided-service не получает fingerprint «чужого» артефакта.
    graph_fingerprint = _service_graph_fingerprint(service)
    if graph_fingerprint is None and not fake_service and not provided_service:
        graph_fingerprint = load_graph_fingerprint(graph_dir)

    git = git_info()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(git['sha'])[:7]}"
    if llm_lock is None and not fake_service and not service_handles_lock:
        llm_lock = LLMExecutionLock(kind="cli", run_id=run_id)
    base_out = Path(out_dir or DEFAULT_RESULTS_DIR)
    final_dir = base_out / run_id
    suffix = 1
    while final_dir.exists():
        suffix += 1
        final_dir = base_out / f"{run_id}_{suffix}"
    temp_dir = base_out / f".{final_dir.name}.tmp"
    if temp_dir.exists():
        import shutil

        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "traces").mkdir(exist_ok=True)

    started_at = utc_now()
    command = "python -m eval.runner " + " ".join(sys.argv[1:]).strip()
    store = make_chunk_store(chunks)
    known_ids = set(store)
    cases: list[dict[str, Any]] = []
    retrieval_rows: list[list[Any]] = []
    judge_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k": 5,
        "candidates": None,
        "rerank_pool": None,
        "reranker": None,
        "graph": None,
    }

    def notify(event: str, payload: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({"event": event, **payload})
        except Exception:
            pass

    notify(
        "run_started",
        {"run_id": final_dir.name, "tests_selected": len(modes)},
    )
    total_pairs = len(modes)
    completed_pairs = 0
    try:
        for test, mode, supported in modes:
            test_id = str(test.get("id"))
            current_service = service
            if fresh_service and not provided_service and not fake_service:
                current_service = _build_real_service()
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                break
            for attempt_index in range(repeat):
                started_pair = time.perf_counter()
                if not supported:
                    case = {
                        "id": test_id,
                        "attempt_index": attempt_index,
                        "suite": test.get("suite"),
                        "split": test.get("split"),
                        "category": test.get("category"),
                        "difficulty": test.get("difficulty"),
                        "retrieval_mode": mode,
                        "status": "skipped",
                        "skip_reason": "unsupported_mode",
                        "pass": False,
                        "expected_status": test.get("expected_status"),
                        "actual_status": None,
                        "retrieval": {
                            "recall_at_5": None,
                            "recall_at_10": None,
                            "mrr": None,
                            "ndcg_at_5": None,
                            "retrieval_coverage": None,
                            "evidence_coverage": None,
                            "context_coverage": None,
                            "loss_stage": None,
                            "not_measured": ["unsupported_mode"],
                        },
                        "metrics": {
                            "recall_at_5": None,
                            "recall_at_10": None,
                            "mrr": None,
                            "ndcg_at_5": None,
                            "retrieval_coverage": None,
                            "evidence_coverage": None,
                            "context_coverage": None,
                            "loss_stage": None,
                            "not_measured": ["unsupported_mode"],
                        },
                        "answer": {},
                        "critical_errors": [],
                        "warnings": ["unsupported_mode"],
                        "efficiency": {"tool_calls": 0, "search_calls": 0,
                                       "llm_calls": None, "latency_ms": 0.0},
                        "trace_path": None,
                        "trace_available": False,
                    }
                    cases.append(case)
                    completed_pairs += 1
                    notify(
                        "test_completed",
                        {
                            "test_id": test_id,
                            "retrieval_mode": mode,
                            "attempt_index": attempt_index,
                            "status": "skipped",
                            "pass": False,
                            "completed": completed_pairs,
                            "total": total_pairs,
                        },
                    )
                    continue

                conversation_id = _safe_conversation_id(
                    final_dir.name, test_id, mode, attempt_index
                )
                notify(
                    "test_started",
                    {
                        "test_id": test_id,
                        "retrieval_mode": mode,
                        "attempt_index": attempt_index,
                    },
                )
                try:
                    if isinstance(test.get("turns"), list):
                        turn_specs = [
                            turn for turn in test["turns"] if isinstance(turn, Mapping)
                        ]
                        results = []
                        trace_turns = []
                        for turn_index, turn_spec in enumerate(turn_specs):
                            filters = _test_filters(turn_spec)
                            result = _ask_with_timeout(
                                _service_for_mode(current_service, mode),
                                str(turn_spec.get("question") or ""),
                                conversation_id=conversation_id,
                                filters=filters,
                                timeout=timeout,
                                llm_lock=llm_lock,
                                wait_for_llm=wait_for_llm,
                            )
                            results.append(result)
                            trace_turns.append(
                                build_turn_trace(
                                    result,
                                    question=str(turn_spec.get("question") or ""),
                                    turn_index=turn_index,
                                )
                            )
                            if getattr(result, "status", None) == "failed":
                                break
                        test_status = (
                            "failed"
                            if any(getattr(result, "status", None) == "failed"
                                   for result in results)
                            else "completed"
                        )
                    else:
                        filters = _test_filters(test)
                        result = _ask_with_timeout(
                            _service_for_mode(current_service, mode),
                            str(test.get("question") or ""),
                            conversation_id=conversation_id,
                            filters=filters,
                            timeout=timeout,
                            llm_lock=llm_lock,
                            wait_for_llm=wait_for_llm,
                        )
                        results = [result]
                        trace_turns = [
                            build_turn_trace(
                                result,
                                question=str(test.get("question") or ""),
                                turn_index=0,
                            )
                        ]
                        test_status = _case_status(
                            str(getattr(result, "status", "completed"))
                        )
                    pair_duration = (time.perf_counter() - started_pair) * 1000
                    trace = build_trace(
                        run_id=final_dir.name,
                        test=test,
                        turns=trace_turns,
                        retrieval_mode=mode,
                        attempt_index=attempt_index,
                        conversation_id=conversation_id,
                        started_at=started_at,
                        duration_ms=pair_duration,
                        status=test_status,
                        git_sha=str(git["sha"]),
                        git_dirty=bool(git["dirty"]),
                        dataset_sha256=bundle.dataset_sha256,
                        corpus_sha256=corpus_content_sha256(chunks),
                        corpus_fingerprint=(bundle.manifest or {}).get("corpus_fingerprint"),
                        config_fingerprint=_sha256_text(
                            json.dumps(
                                config,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                    )
                    retrieval = evaluate_retrieval(
                        test, trace, known_chunk_ids=known_ids
                    )
                    answer = evaluate_answer(
                        test,
                        trace,
                        chunk_store=store,
                        retrieval_metrics=retrieval,
                    )
                    dialogue = (
                        evaluate_dialogue(test, trace, answer)
                        if isinstance(test.get("turns"), list)
                        else None
                    )
                    judge_result = (
                        {
                            "available": False,
                            "rubric_total": None,
                            "scores": {},
                            "issues": [],
                            "error": "judge_unavailable",
                            "not_measured": ["judge", "rubric_mean"],
                        }
                        if judge
                        else None
                    )
                    if judge_result is not None:
                        judge_rows.append(
                            {
                                "test_id": test_id,
                                "retrieval_mode": mode,
                                "attempt_index": attempt_index,
                                **judge_result,
                            }
                        )

                    critical = list(answer.get("critical_errors", []))
                    warnings = list(answer.get("warnings", []))
                    for result in results:
                        warnings.extend(getattr(result, "warnings", []) or [])
                    if dialogue is not None:
                        critical.extend(dialogue.get("critical_errors", []))
                        warnings.extend(dialogue.get("warnings", []))
                    seen_critical: set[tuple[Any, Any, Any]] = set()
                    deduped_critical: list[dict[str, Any]] = []
                    for item in critical:
                        key = (
                            item.get("code"),
                            item.get("message"),
                            item.get("turn_index"),
                        )
                        if key in seen_critical:
                            continue
                        seen_critical.add(key)
                        deduped_critical.append(item)
                    critical = deduped_critical
                    evidence_coverage = retrieval.get("evidence_coverage")
                    evidence_ok = _evidence_ok(test, retrieval)
                    passed = (
                        test_status == "completed"
                        and bool(answer.get("pass"))
                        and evidence_ok
                        and (dialogue is None or bool(dialogue.get("pass")))
                        and not critical
                    )
                    actual_status: Any
                    turn_statuses = [
                        str((turn.get("answer") or {}).get("status"))
                        for turn in trace_turns
                    ]
                    actual_status = (
                        turn_statuses[0]
                        if len(set(turn_statuses)) <= 1 and turn_statuses
                        else "mixed"
                    )
                    case = {
                        "id": test_id,
                        "attempt_index": attempt_index,
                        "suite": test.get("suite"),
                        "split": test.get("split"),
                        "category": test.get("category"),
                        "difficulty": test.get("difficulty"),
                        "retrieval_mode": mode,
                        "status": test_status,
                        "pass": passed,
                        "expected_status": test.get("expected_status")
                        or (test.get("turns", [{}])[0].get("expected_status")
                            if test.get("turns") else None),
                        "actual_status": actual_status,
                        "turn_statuses": turn_statuses,
                        "retrieval": {
                            key: retrieval.get(key)
                            for key in (
                                "recall_at_5",
                                "recall_at_10",
                                "mrr",
                                "ndcg_at_5",
                                "retrieval_coverage",
                                "evidence_coverage",
                                "context_coverage",
                                "loss_stage",
                            )
                        } | {"not_measured": retrieval.get("not_measured", [])},
                        "metrics": {
                            key: retrieval.get(key)
                            for key in (
                                "recall_at_5",
                                "recall_at_10",
                                "mrr",
                                "ndcg_at_5",
                                "retrieval_coverage",
                                "evidence_coverage",
                                "context_coverage",
                                "loss_stage",
                            )
                        } | {"not_measured": retrieval.get("not_measured", [])},
                        "best_call": retrieval.get("best_call"),
                        "all_calls": retrieval.get("all_calls", []),
                        "answer": {
                            **(answer.get("metrics") or {}),
                            "automatic_pass": bool(answer.get("pass")),
                            # Причины проваленных checks сохраняются в case,
                            # чтобы отчёт мог показать их без чтения traces.
                            "failed_checks": [
                                {
                                    "name": row.get("name"),
                                    "mandatory": bool(row.get("mandatory")),
                                    "detail": row.get("detail", ""),
                                }
                                for row in answer.get("checks", []) or []
                                if isinstance(row, Mapping) and not row.get("pass")
                            ],
                        },
                        "dialogue": (
                            {
                                "pass": dialogue.get("pass"),
                                "checks": dialogue.get("checks", {}),
                                "required_checks": dialogue.get("required_checks", []),
                                "cascade_failure": dialogue.get("cascade_failure"),
                                "metrics": dialogue.get("metrics", {}),
                            }
                            if dialogue is not None
                            else None
                        ),
                        "critical_errors": critical,
                        "warnings": list(dict.fromkeys(warnings)),
                        "efficiency": {
                            "tool_calls": sum(
                                len(getattr(result, "tool_events", []) or [])
                                for result in results
                            ),
                            "search_calls": sum(
                                sum(
                                    1
                                    for event in getattr(result, "tool_events", []) or []
                                    if getattr(event, "tool_name", None) == "search_reports"
                                )
                                for result in results
                            ),
                            "llm_calls": None,
                            "latency_ms": pair_duration,
                        },
                        "trace_path": trace_path(test_id, mode, attempt_index),
                        "trace_available": True,
                        "rubric_total": (
                            judge_result.get("rubric_total") if judge_result else None
                        ),
                        "judge_error": (
                            judge_result.get("error") if judge_result else None
                        ),
                        "_trace": trace,
                    }
                except TurnTimeout as exc:
                    cases.append(
                        _error_case(
                            test=test,
                            mode=mode,
                            attempt_index=attempt_index,
                            status="timeout",
                            error_code=exc.code,
                            message=str(exc),
                        )
                    )
                    completed_pairs += 1
                    notify(
                        "test_completed",
                        {
                            "test_id": test_id,
                            "retrieval_mode": mode,
                            "attempt_index": attempt_index,
                            "status": "timeout",
                            "pass": False,
                            "completed": completed_pairs,
                            "total": total_pairs,
                        },
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - ошибка одного теста
                    message = safe_error_text(exc)
                    errors.append(
                        {
                            "code": "evaluator_error",
                            "test_id": test_id,
                            "message": message,
                        }
                    )
                    cases.append(
                        _error_case(
                            test=test,
                            mode=mode,
                            attempt_index=attempt_index,
                            status="error",
                            error_code="evaluator_error",
                            message=message,
                        )
                    )
                    completed_pairs += 1
                    notify(
                        "test_completed",
                        {
                            "test_id": test_id,
                            "retrieval_mode": mode,
                            "attempt_index": attempt_index,
                            "status": "error",
                            "pass": False,
                            "completed": completed_pairs,
                            "total": total_pairs,
                        },
                    )
                    continue

                cases.append(case)
                retrieval_rows.extend(
                    _retrieval_csv_rows(
                        test=test,
                        trace=case["_trace"],
                        retrieval_mode=mode,
                        attempt_index=attempt_index,
                        loss_stage=(case.get("retrieval") or {}).get("loss_stage"),
                    )
                )
                public_trace = {
                    key: value for key, value in case["_trace"].items()
                }
                atomic_write_json(
                    temp_dir / "traces" / trace_path(test_id, mode, attempt_index).split("/", 1)[1],
                    public_trace,
                )
                completed_pairs += 1
                notify(
                    "test_completed",
                    {
                        "test_id": test_id,
                        "retrieval_mode": mode,
                        "attempt_index": attempt_index,
                        "status": test_status,
                        "pass": passed,
                        "completed": completed_pairs,
                        "total": total_pairs,
                    },
                )
    except Exception as exc:  # noqa: BLE001 - ошибка runner/окружения
        errors.append({"code": "runner_error", "message": safe_error_text(exc)})
        finished_at = utc_now()
        summary = _summary(
            run_id=final_dir.name,
            started_at=started_at,
            finished_at=finished_at,
            command=command,
            git=git,
            bundle=bundle,
            chunks=chunks,
            cases=cases,
            selected=len(modes),
            suite=suite,
            split=split,
            retrieval_mode=retrieval_mode,
            judge=judge,
            repeat=repeat,
            config=config,
            service=service,
            limitations=_limitations(
                bundle=bundle,
                selected=len(modes),
                fake_service=fake_service,
                judge_requested=judge,
                cases=cases,
            ),
            errors=errors,
            fake_service=fake_service,
            graph_fingerprint=graph_fingerprint,
        )
        _write_artifacts(
            temp_dir,
            summary,
            cases,
            retrieval_rows,
            judge_rows,
            judge,
        )
        _finalize_dir(temp_dir, final_dir, run_id=final_dir.name)
        raise RunnerError("runner_error", safe_error_text(exc)) from exc

    finished_at = utc_now()
    summary = _summary(
        run_id=final_dir.name,
        started_at=started_at,
        finished_at=finished_at,
        command=command,
        git=git,
        bundle=bundle,
        chunks=chunks,
        cases=cases,
        selected=len(modes),
        suite=suite,
        split=split,
        retrieval_mode=retrieval_mode,
        judge=judge,
        repeat=repeat,
        config=config,
        service=service,
        limitations=_limitations(
            bundle=bundle,
            selected=len(modes),
            fake_service=fake_service,
            judge_requested=judge,
            cases=cases,
        ),
        errors=errors,
        fake_service=fake_service,
        graph_fingerprint=graph_fingerprint,
    )
    _write_artifacts(temp_dir, summary, cases, retrieval_rows, judge_rows, judge)
    final = _finalize_dir(temp_dir, final_dir, run_id=final_dir.name)
    notify(
        "run_completed",
        {
            "run_id": final.name,
            "pass_rate": summary.get("pass_rate"),
            "execution_coverage": summary.get("execution_coverage"),
        },
    )
    return final


def _service_for_mode(service: Any, mode: str) -> Any:
    """Передать fake-service нужный режим; реальный AgentService не мутируется."""
    if isinstance(service, FakeAgentService):
        return FakeAgentService(
            service.chunks,
            retrieval_mode=mode,
            top_k=service.top_k,
            supported_retrieval_modes=service.supported_retrieval_modes,
        )
    return service


def _error_case(
    *,
    test: Mapping[str, Any],
    mode: str,
    attempt_index: int,
    status: str,
    error_code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "id": test.get("id"),
        "attempt_index": attempt_index,
        "suite": test.get("suite"),
        "split": test.get("split"),
        "category": test.get("category"),
        "difficulty": test.get("difficulty"),
        "retrieval_mode": mode,
        "status": status,
        "pass": False,
        "expected_status": test.get("expected_status"),
        "actual_status": None,
        "retrieval": {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_5": None,
            "retrieval_coverage": None,
            "evidence_coverage": None,
            "context_coverage": None,
            "loss_stage": None,
            "not_measured": ["run_error"],
        },
        "metrics": {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_5": None,
            "retrieval_coverage": None,
            "evidence_coverage": None,
            "context_coverage": None,
            "loss_stage": None,
            "not_measured": ["run_error"],
        },
        "answer": {},
        "critical_errors": [],
        "warnings": [error_code],
        "efficiency": {
            "tool_calls": 0,
            "search_calls": 0,
            "llm_calls": None,
            "latency_ms": 0.0,
        },
        "trace_path": None,
        "trace_available": False,
        "error": {"code": error_code, "message": message},
    }


def _build_real_service() -> Any:
    """Ленивая сборка настоящего AgentService без ключей в артефактах."""
    try:
        from src.agent.service import create_agent_service
    except Exception as exc:  # noqa: BLE001
        raise RunnerError(
            "service_unavailable",
            f"Не удалось импортировать AgentService: {safe_error_text(exc)}",
        ) from exc
    try:
        return create_agent_service()
    except Exception as exc:  # noqa: BLE001
        raise RunnerError(
            "service_unavailable",
            "Не удалось собрать AgentService. Проверьте LLM_API_KEY/LLM_BASE_URL "
            f"и chunk store: {safe_error_text(exc)}",
        ) from exc


def _write_artifacts(
    target: Path,
    summary: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Sequence[Any]],
    judge_rows: Sequence[Mapping[str, Any]],
    judge: bool,
) -> None:
    public_cases = [
        {key: value for key, value in case.items() if not key.startswith("_")}
        for case in cases
    ]
    atomic_write_json(target / "summary.json", summary)
    atomic_write_text(
        target / "cases.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in public_cases
        ),
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "test_id",
            "retrieval_mode",
            "attempt_index",
            "call_index",
            "stage",
            "rank",
            "chunk_id",
            "score",
            "in_gold",
            "loss_stage",
        ]
    )
    writer.writerows(retrieval_rows)
    atomic_write_text(target / "retrieval.csv", buffer.getvalue())
    if judge:
        atomic_write_text(
            target / "judge.jsonl",
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in judge_rows
            ),
        )
    write_report(target, summary)
    review_ids = sorted(
        {
            str(case.get("id"))
            for case in public_cases
            if case.get("suite") == "multi_turn"
            or case.get("difficulty") == "hard"
        }
        - {"None"}
    )
    if review_ids:
        atomic_write_json(
            target / "manual_review.json",
            {
                "reviewer": "",
                "tests": {
                    test_id: {
                        "scores": {
                            "factual_accuracy": None,
                            "completeness": None,
                            "sources": None,
                            "analysis": None,
                            "constraints": None,
                        },
                        "notes": "",
                    }
                    for test_id in review_ids
                },
            },
        )


def _finalize_dir(temp_dir: Path, final_dir: Path, *, run_id: str) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        suffix = 2
        candidate = final_dir
        while candidate.exists():
            candidate = final_dir.parent / f"{run_id}_{suffix}"
            suffix += 1
        final_dir = candidate
    os.replace(temp_dir, final_dir)
    return final_dir


def _validate_only(args: argparse.Namespace) -> int:
    datasets = Path(args.datasets) if args.datasets else DEFAULT_DATASETS_DIR
    chunks_path = Path(args.chunks) if args.chunks else DEFAULT_CHUNKS_DIR
    try:
        bundle = validate_dataset(
            datasets,
            chunks_dir=chunks_path,
            check_corpus=True,
            require_manifest=True,
        )
    except DatasetError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        return 2
    selected = _select_tests(
        bundle,
        suite=args.suite or "full",
        split=args.split or "all",
        ids=None,
        tag=None,
        limit=None,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "dataset_sha256": bundle.dataset_sha256,
                "expected_corpus_sha256": bundle.expected_corpus_sha256,
                "tests": len(bundle.tests),
                "selected": len(selected),
                "suites": bundle.suite_hashes,
                "warnings": [issue.as_dict() for issue in bundle.warnings],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Agent_RAG")
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=SUITES,
        help="smoke | full | easy | medium | hard | multi_turn",
    )
    parser.add_argument("--split", default=None, choices=("dev", "holdout", "all"))
    parser.add_argument("--retrieval-mode", default="hybrid",
                        choices=(*RETRIEVAL_MODES, "all"))
    parser.add_argument("--ids", default=None, help="ID через запятую")
    parser.add_argument("--tag", default=None, help="Фильтр по тегу")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fresh-service", action="store_true")
    parser.add_argument("--fake-service", action="store_true",
                        help="Offline-режим без LLM и сети")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", "--out", dest="output_dir", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--wait-for-llm", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if args.validate_only:
        return _validate_only(args)

    split = args.split or ("dev" if args.suite == "smoke" else "all")
    ids = [item.strip() for item in args.ids.split(",") if item.strip()] if args.ids else None
    try:
        run_dir = run_suite(
            suite=args.suite,
            split=split,
            ids=ids,
            retrieval_mode=args.retrieval_mode,
            judge=bool(args.judge),
            repeat=args.repeat,
            out_dir=Path(args.output_dir) if args.output_dir else None,
            datasets_dir=Path(args.datasets) if args.datasets else None,
            chunks_dir=Path(args.chunks) if args.chunks else None,
            limit=args.limit,
            tag=args.tag,
            fake_service=args.fake_service,
            fresh_service=args.fresh_service,
            timeout=args.timeout,
            wait_for_llm=args.wait_for_llm,
        )
    except LLMBusy as exc:
        print(
            json.dumps(
                {"code": "llm_busy", "message": safe_error_text(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 5
    except LLMLockUnavailable as exc:
        print(
            json.dumps(
                {"code": "llm_lock_unavailable", "message": safe_error_text(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    except DatasetError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        return 2
    except RunnerError as exc:
        print(json.dumps({"code": exc.code, "message": safe_error_text(exc)},
                         ensure_ascii=False, indent=2))
        return 3
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"Run: {run_dir} | pass_rate={summary.get('pass_rate')} "
        f"| coverage={summary.get('execution_coverage')}"
    )
    dataset_warnings = (summary.get("dataset") or {}).get("warnings") or []
    if dataset_warnings:
        codes = sorted({str(item.get("code")) for item in dataset_warnings})
        print(f"Dataset warnings: {len(dataset_warnings)} ({', '.join(codes)})")
    if args.fail_under is not None:
        pass_rate = summary.get("pass_rate")
        if pass_rate is None or pass_rate < args.fail_under:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
