"""Contract/worker-тесты Evaluation API (specs/evaluation-webui.md §8–§10).

Сеть и LLM не используются: runner подменяется fake-функцией, артефакты
(``summary.json``/``cases.jsonl``/traces) пишутся в ``tmp_path``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.evaluation_store import (
    EvaluationStore,
    create_evaluation_runtime,
)
from src.api.config import ApiConfig
from test_web_api_support import FakeAgentService, make_config, make_services, wait_for

DATASETS_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def eval_config(tmp_path: Path, **overrides: Any) -> ApiConfig:
    values: dict[str, Any] = {
        "eval_ui_enabled": True,
        "eval_db_path": tmp_path / "evaluation" / "evaluation.sqlite3",
        "eval_results_dir": tmp_path / "results",
        "eval_datasets_dir": DATASETS_DIR,
        "eval_queue_size": 3,
        "eval_turn_timeout_s": 30.0,
        "eval_retention_runs": 50,
    }
    values.update(overrides)
    return make_config(tmp_path).with_overrides(**values)


def make_case(
    test_id: str = "easy_001",
    *,
    retrieval_mode: str = "hybrid",
    attempt_index: int = 0,
    passed: bool = True,
    status: str = "completed",
    suite: str = "easy",
    difficulty: str = "easy",
    category: str = "fact_lookup",
    critical: list[dict[str, Any]] | None = None,
    trace_available: bool = True,
    loss_stage: str | None = "ok",
    answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    critical_list = list(critical or [])
    return {
        "id": test_id,
        "attempt_index": attempt_index,
        "suite": suite,
        "split": "dev",
        "category": category,
        "difficulty": difficulty,
        "retrieval_mode": retrieval_mode,
        "status": status,
        "pass": passed,
        "expected_status": "answer",
        "actual_status": "answer",
        "retrieval": {
            "recall_at_5": 1.0 if passed else 0.0,
            "recall_at_10": 1.0 if passed else 0.5,
            "mrr": 1.0 if passed else 0.0,
            "retrieval_coverage": 1.0,
            "evidence_coverage": 1.0 if passed else 0.5,
            "context_coverage": 1.0,
            "loss_stage": loss_stage,
        },
        "metrics": {
            "recall_at_5": 1.0 if passed else 0.0,
            "recall_at_10": 1.0 if passed else 0.5,
            "mrr": 1.0 if passed else 0.0,
            "retrieval_coverage": 1.0,
            "evidence_coverage": 1.0 if passed else 0.5,
            "context_coverage": 1.0,
            "loss_stage": loss_stage,
            "not_measured": [],
        },
        "answer": answer
        if answer is not None
        else {
            "automatic_pass": passed,
            "fact_accuracy": 1.0 if passed else 0.0,
            "citation_validity": 1.0,
            "citation_coverage": 1.0,
        },
        "critical_errors": critical_list,
        "warnings": [],
        "efficiency": {
            "tool_calls": 1,
            "search_calls": 1,
            "llm_calls": None,
            "latency_ms": 42.0,
        },
        "trace_path": (
            f"traces/{test_id}.{retrieval_mode}"
            + (f".attempt_{attempt_index}" if attempt_index else "")
            + ".json"
        )
        if trace_available
        else None,
        "trace_available": trace_available,
    }


def make_trace(
    test_id: str = "easy_001",
    *,
    retrieval_mode: str = "hybrid",
    attempt_index: int = 0,
    covered: list[int] | None = None,
    missing: list[int] | None = None,
    context_ids: list[str] | None = None,
) -> dict[str, Any]:
    context = context_ids if context_ids is not None else ["chunk-1", "chunk-2"]
    manifest = ["chunk-1", "chunk-2", "chunk-3"]
    return {
        "trace_schema_version": "1.1",
        "run_id": "fake-run",
        "test_id": test_id,
        "retrieval_mode": retrieval_mode,
        "attempt_index": attempt_index,
        "status": "completed",
        "git_sha": "abc1234",
        "git_dirty": False,
        "turns": [
            {
                "turn_index": 0,
                "question": "Какая выручка за 2023 год?",
                "status": "completed",
                "query_plan": {
                    "intent": "factoid",
                    "years": [2023],
                    "needs_graph": False,
                    "answer_mode": "concise",
                },
                "effective_filters": {"years": [2023]},
                "retrieval_calls": [
                    {
                        "call_index": 0,
                        "tool_name": "search_reports",
                        "query": "выручка 2023",
                        "retrieval_mode": retrieval_mode,
                        "counts": {"dense": 2, "rrf": 2, "rerank": 2, "returned": 2},
                        "stages": {
                            "dense": [{"chunk_id": "chunk-1", "rank": 1}],
                            "bm25": None,
                            "graph": None,
                            "rrf": [{"chunk_id": "chunk-1", "rank": 1}],
                            "rerank": [{"chunk_id": "chunk-1", "rank": 1}],
                            "returned": [{"chunk_id": "chunk-1", "rank": 1}],
                        },
                    }
                ],
                "evidence_manifest": manifest,
                "context_chunk_ids": context,
                "context_source": "context" if context_ids is not None else "evidence_manifest",
                "covered_years": covered if covered is not None else [2023],
                "missing_years": missing or [],
                "answer": {
                    "status": "answer",
                    "text": "Выручка составила 100 млрд руб. [1].",
                    "citations": [
                        {"chunk_id": "chunk-1", "doc_id": "sber_2023", "year": 2023}
                    ],
                    "facts": [],
                    "warnings": [],
                    "calculations": [],
                },
                "run_warnings": [],
                "structural_changes": [],
                "timings": {"total_ms": 42.0},
                "errors": [],
            }
        ],
        "errors": [],
        "truncated": False,
    }


def base_summary(
    *,
    directory: Path,
    suite: str = "smoke",
    split: str = "dev",
    retrieval_mode: str = "hybrid",
    dataset_sha: str = "dataset-sha-1",
    corpus_sha: str = "corpus-sha-1",
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = cases or []
    eligible = sum(1 for case in rows if case.get("status") in {"completed", "failed"})
    passed = sum(1 for case in rows if case.get("pass"))
    return {
        "run_id": directory.name,
        "started_at": "2026-11-14T12:00:00Z",
        "finished_at": "2026-11-14T12:04:00Z",
        "command": "fake",
        "suite": suite,
        "split": split,
        "judge": False,
        "fake_service": False,
        "git": {"sha": "abc1234", "dirty": False},
        "dataset": {
            "version": "1.1",
            "sha256": dataset_sha,
            "tests": 12,
            "expected_corpus_sha256": corpus_sha,
            "suites": {"easy": "h-easy", "medium": "h-medium"},
        },
        "corpus": {
            "content_sha256": corpus_sha,
            "chunk_count": 10,
            "documents": ["sber_2023"],
            "index_fingerprint": "index-fp-1",
        },
        "config": {"retrieval_mode": retrieval_mode, "top_k": 5},
        "config_fingerprint": "config-fp-1",
        "models": {"llm": "test-llm", "reranker": None, "judge": None},
        "prompts": {"answer": "sha256:abc"},
        "capabilities": {},
        "coverage": {
            "tests_selected": len(rows),
            "tests_eligible": eligible,
            "executed": eligible,
            "passed": passed,
            "failed": max(eligible - passed, 0),
            "skipped": 0,
            "unsupported": 0,
            "timeouts": 0,
            "errors": 0,
            "judge_errors": 0,
        },
        "pass_rate": (passed / eligible) if eligible else None,
        "execution_coverage": 1.0 if rows else None,
        "pass_at_1": (passed / eligible) if eligible else None,
        "stability": None,
        "groups": {
            "retrieval": {
                "recall_at_5": 1.0 if passed else 0.0,
                "recall_at_10": 0.9 if passed else 0.5,
                "mrr": 0.8,
                "evidence_coverage": 0.95 if passed else 0.5,
                "context_coverage": 1.0,
            },
            "answer": {
                "automatic_pass_rate": 0.9,
                "rubric_mean": None,
                "fact_accuracy": 0.95,
                "citation_validity": 1.0,
                "citation_coverage": 1.0,
            },
            "reasoning": {"pass_rate": 0.75},
            "dialogue": {"scenario_success_rate": None},
        },
        "by_difficulty": {},
        "by_category": {},
        "by_split": {},
        "by_mode": {},
        "critical_errors": [],
        "problem_tests": [],
        "efficiency": {"mean_latency_ms": 42.0, "llm_concurrency": 1},
        "limitations": [],
        "errors": [],
    }


def write_artifacts(
    directory: Path,
    *,
    cases: list[dict[str, Any]],
    traces: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    suite: str = "smoke",
    split: str = "dev",
    retrieval_mode: str = "hybrid",
    dataset_sha: str = "dataset-sha-1",
    corpus_sha: str = "corpus-sha-1",
    summary_patch: dict[str, Any] | None = None,
    write_summary: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "traces").mkdir(exist_ok=True)
    summary = base_summary(
        directory=directory,
        suite=suite,
        split=split,
        retrieval_mode=retrieval_mode,
        dataset_sha=dataset_sha,
        corpus_sha=corpus_sha,
        cases=cases,
    )
    if summary_patch:
        summary.update(summary_patch)
    if write_summary:
        (directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
    (directory / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    for case in cases:
        path = case.get("trace_path")
        if not case.get("trace_available") or not isinstance(path, str):
            continue
        key = (
            str(case["id"]),
            str(case["retrieval_mode"]),
            int(case.get("attempt_index") or 0),
        )
        trace = (traces or {}).get(key) or make_trace(
            key[0], retrieval_mode=key[1], attempt_index=key[2]
        )
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    (directory / "report.md").write_text(
        f"# Benchmark {directory.name}\n\n| Метрика | Значение |\n", encoding="utf-8"
    )
    return directory


def runner_for(
    cases: list[dict[str, Any]],
    *,
    traces: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    dataset_sha: str = "dataset-sha-1",
    corpus_sha: str = "corpus-sha-1",
    artifacts: bool = True,
    gate: threading.Event | None = None,
    ignore_cancel: bool = False,
    sleep_s: float = 0.0,
    fail_with: BaseException | None = None,
) -> Callable[..., Path]:
    """Fake runner с контрактом worker'а."""

    def runner(
        *,
        request: dict[str, Any],
        out_dir: Path,
        progress: Callable[[dict[str, Any]], None],
        cancel_event: threading.Event,
    ) -> Path:
        directory = Path(out_dir) / f"fake_{uuid.uuid4().hex[:8]}"
        progress({"event": "run_started", "tests_selected": len(cases)})
        if gate is not None:
            gate.wait(10)
        if ignore_cancel:
            time.sleep(sleep_s or 1.0)
        else:
            time.sleep(sleep_s)
        if fail_with is not None:
            raise fail_with
        emitted = 0
        for case in cases:
            if cancel_event.is_set():
                break
            progress(
                {
                    "event": "test_started",
                    "test_id": case["id"],
                    "retrieval_mode": case["retrieval_mode"],
                    "attempt_index": case.get("attempt_index", 0),
                }
            )
            emitted += 1
            progress(
                {
                    "event": "test_completed",
                    "test_id": case["id"],
                    "completed": emitted,
                    "total": len(cases),
                }
            )
        if artifacts:
            write_artifacts(
                directory,
                cases=cases[:emitted] or cases,
                traces=traces,
                suite=str(request.get("suite") or "smoke"),
                split=str(request.get("split") or "dev"),
                retrieval_mode=str(request.get("retrieval_mode") or "hybrid"),
                dataset_sha=dataset_sha,
                corpus_sha=corpus_sha,
            )
        else:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    return runner


@contextmanager
def eval_client(
    tmp_path: Path,
    runner: Callable[..., Path] | None = None,
    *,
    config: ApiConfig | None = None,
) -> Iterator[TestClient]:
    cfg = config or eval_config(tmp_path)
    services = make_services(cfg, FakeAgentService())
    app = create_app(config=cfg, services=services)
    runtime = create_evaluation_runtime(
        cfg, services, runner=runner or runner_for([make_case()])
    )
    app.state.evaluation = runtime
    try:
        with TestClient(app) as client:
            yield client
    finally:
        runtime.shutdown()


def start(client: TestClient, **payload: Any) -> dict[str, Any]:
    body = {
        "suite": "smoke",
        "split": "dev",
        "retrieval_mode": "hybrid",
        "judge": False,
        "repeat": 1,
        **payload,
    }
    response = client.post("/api/evaluation/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()


def wait_status(client: TestClient, run_id: str, *statuses: str, timeout: float = 5.0) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def predicate() -> bool:
        payload = client.get(f"/api/evaluation/runs/{run_id}").json()
        seen.clear()
        seen.update(payload)
        return payload.get("status") in statuses

    assert wait_for(predicate, timeout=timeout), seen
    return seen


def seed_completed_run(
    tmp_path: Path,
    cfg: ApiConfig,
    *,
    run_id: str,
    cases: list[dict[str, Any]],
    traces: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    suite: str = "smoke",
    split: str = "dev",
    retrieval_mode: str = "hybrid",
    dataset_sha: str = "dataset-sha-1",
    corpus_sha: str = "corpus-sha-1",
    summary_patch: dict[str, Any] | None = None,
) -> str:
    store = EvaluationStore(cfg.eval_db_path)
    store.init()
    store.create_run(
        run_id=run_id,
        suite=suite,
        split=split,
        retrieval_mode=retrieval_mode,
        judge=False,
        repeat=1,
        request={"suite": suite, "split": split, "retrieval_mode": retrieval_mode},
        tests_total=len(cases),
    )
    store.mark_running(run_id)
    directory = write_artifacts(
        Path(cfg.eval_results_dir) / run_id,
        cases=cases,
        traces=traces,
        suite=suite,
        split=split,
        retrieval_mode=retrieval_mode,
        dataset_sha=dataset_sha,
        corpus_sha=corpus_sha,
        summary_patch=summary_patch,
    )
    runtime = create_evaluation_runtime(cfg, store=store, runner=runner_for(cases))
    assert runtime.worker.import_artifacts(run_id, directory)
    runtime.shutdown()
    assert store.get_run(run_id)["status"] == "completed"
    return run_id


# ---------------------------------------------------------------------------
# Capabilities / disabled
# ---------------------------------------------------------------------------


def test_evaluation_disabled_returns_honest_404(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path, eval_ui_enabled=False)
    app = create_app(config=cfg, services=make_services(cfg, FakeAgentService()))
    with TestClient(app) as client:
        response = client.get("/api/evaluation/suites")
        capabilities = client.get("/api/capabilities").json()
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "evaluation_disabled"
    assert capabilities["evaluation_ui"] is False
    assert capabilities["evaluation_runs"] is False
    assert capabilities["evaluation_cancel"] is False
    # Обычный чат продолжает работать.
    assert capabilities["coverage"] is True


def test_capabilities_expose_evaluation_flags(tmp_path: Path) -> None:
    with eval_client(tmp_path) as client:
        capabilities = client.get("/api/capabilities").json()
    assert capabilities["evaluation_ui"] is True
    assert capabilities["evaluation_runs"] is True
    assert capabilities["evaluation_sse"] is False
    assert capabilities["evaluation_cancel"] is True
    assert capabilities["retrieval_modes"] == ["hybrid", "dense", "keyword"]
    assert capabilities["judge"] is False
    assert capabilities["max_repeat"] == 5
    assert capabilities["max_queued"] == 3


# ---------------------------------------------------------------------------
# Run creation / worker
# ---------------------------------------------------------------------------


def test_create_run_returns_202_and_worker_imports_artifacts(tmp_path: Path) -> None:
    cases = [make_case("easy_001"), make_case("easy_002", passed=False, critical=[{"code": "invalid_citation"}])]
    with eval_client(tmp_path, runner_for(cases)) as client:
        created = start(client)
        assert created["status"] in {"queued", "running"}
        assert created["status_url"].endswith(created["id"])

        run = wait_status(client, created["id"], "completed")
        assert run["summary"]["pass_rate"] == 0.5
        assert run["dataset"]["sha256"] == "dataset-sha-1"
        assert run["versions"]["git_sha"] == "abc1234"
        assert run["artifacts_url"]

        items = client.get(f"/api/evaluation/runs/{created['id']}/tests").json()
        assert items["total"] == 2
        assert {item["test_id"] for item in items["items"]} == {"easy_001", "easy_002"}

        report = client.get(f"/api/evaluation/runs/{created['id']}/report")
        assert report.status_code == 200
        assert "Benchmark" in report.text


def test_create_run_is_idempotent_by_client_request_id(tmp_path: Path) -> None:
    with eval_client(tmp_path) as client:
        payload = {"client_request_id": "req-1"}
        first = start(client, **payload)
        second = start(client, **payload)
        assert first["id"] == second["id"]
        assert second.get("duplicate") is True
        runs = client.get("/api/evaluation/runs").json()
        assert runs["total"] == 1


def test_create_run_validates_contract_codes(tmp_path: Path) -> None:
    with eval_client(tmp_path) as client:
        def post(body: dict[str, Any]) -> Any:
            return client.post("/api/evaluation/runs", json=body)

        assert post({"suite": "../etc", "split": "dev"}).json()["error"]["code"] == "invalid_suite"
        assert post({"suite": "smoke", "split": "holdout"}).json()["error"]["code"] == "invalid_split"
        assert post({"suite": "smoke", "split": "dev", "repeat": 9}).json()["error"]["code"] == "invalid_repeat"
        assert (
            post({"suite": "smoke", "split": "dev", "retrieval_mode": "bm25"}).json()["error"]["code"]
            == "mode_not_supported"
        )
        assert (
            post({"suite": "smoke", "split": "dev", "judge": True}).json()["error"]["code"]
            == "judge_unavailable"
        )
        holdout = post({"suite": "full", "split": "holdout"})
        assert holdout.status_code == 422
        assert holdout.json()["error"]["code"] == "holdout_confirmation_required"


def test_queue_full_is_reported(tmp_path: Path) -> None:
    gate = threading.Event()
    cfg = eval_config(tmp_path, eval_queue_size=0)
    try:
        with eval_client(tmp_path, runner_for([make_case()], gate=gate), config=cfg) as client:
            response = client.post("/api/evaluation/runs", json={"suite": "smoke", "split": "dev"})
    finally:
        gate.set()
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "queue_full"


def test_run_failed_when_artifacts_are_missing(tmp_path: Path) -> None:
    with eval_client(tmp_path, runner_for([make_case()], artifacts=False)) as client:
        created = start(client)
        run = wait_status(client, created["id"], "failed")
        assert run["summary"] is None
        assert run["error"]["code"] == "artifact_import_error"


def test_runner_error_is_sanitised(tmp_path: Path) -> None:
    error = RuntimeError("LLM failed: api_key=sk-secret-1234567890")
    with eval_client(tmp_path, runner_for([make_case()], fail_with=error)) as client:
        created = start(client)
        run = wait_status(client, created["id"], "failed")
        response = client.get(f"/api/evaluation/runs/{created['id']}")
    assert run["error"]["code"] == "runner_error"
    assert "sk-secret" not in response.text
    assert "api_key" not in response.text
    assert "traceback" not in response.text.casefold()


def test_interrupted_at_startup(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    store = EvaluationStore(cfg.eval_db_path)
    store.init()
    store.create_run(
        run_id="run_stale_queued",
        suite="smoke",
        split="dev",
        retrieval_mode="hybrid",
        judge=False,
        repeat=1,
        request={},
    )
    store.create_run(
        run_id="run_stale_running",
        suite="full",
        split="all",
        retrieval_mode="hybrid",
        judge=False,
        repeat=1,
        request={},
    )
    store.mark_running("run_stale_running")

    # Новый инстанс store имитирует рестарт процесса.
    fresh = EvaluationStore(cfg.eval_db_path)
    runtime = create_evaluation_runtime(cfg, store=fresh, runner=runner_for([make_case()]))
    app = create_app(config=cfg, services=make_services(cfg, FakeAgentService()))
    app.state.evaluation = runtime
    try:
        with TestClient(app) as client:
            queued = client.get("/api/evaluation/runs/run_stale_queued").json()
            running = client.get("/api/evaluation/runs/run_stale_running").json()
    finally:
        runtime.shutdown()
    assert queued["status"] == "interrupted"
    assert running["status"] == "interrupted"
    assert running["error"]["code"] == "interrupted"
    assert running["summary"] is None


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_queued_and_running_runs(tmp_path: Path) -> None:
    gate = threading.Event()
    cfg = eval_config(tmp_path, eval_queue_size=2)
    try:
        with eval_client(tmp_path, runner_for([make_case()], gate=gate), config=cfg) as client:
            first = start(client)
            second = start(client)
            assert wait_status(client, first["id"], "running")["status"] == "running"

            cancelled = client.post(f"/api/evaluation/runs/{second['id']}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert client.get(f"/api/evaluation/runs/{second['id']}").json()["status"] == "cancelled"

            stopping = client.post(f"/api/evaluation/runs/{first['id']}/cancel")
            assert stopping.json()["status"] == "cancel_requested"
            gate.set()
            final = wait_status(client, first["id"], "cancelled", "failed")
            assert final["status"] == "cancelled"

            again = client.post(f"/api/evaluation/runs/{first['id']}/cancel")
            assert again.status_code == 409
            assert again.json()["error"]["code"] == "cancel_not_supported"
    finally:
        gate.set()


def test_cancel_timeout_marks_failed(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path, eval_turn_timeout_s=0.05)
    with eval_client(
        tmp_path,
        runner_for([make_case()], ignore_cancel=True, sleep_s=1.5),
        config=cfg,
    ) as client:
        created = start(client)
        assert wait_status(client, created["id"], "running")["status"] == "running"
        response = client.post(f"/api/evaluation/runs/{created['id']}/cancel")
        assert response.json()["status"] == "cancel_requested"
        final = wait_status(client, created["id"], "failed", timeout=6.0)
    assert final["error"]["code"] == "cancel_timeout"


# ---------------------------------------------------------------------------
# Results / detail / trace
# ---------------------------------------------------------------------------


def test_results_are_addressed_by_mode_and_attempt(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    modes = ["hybrid", "dense", "keyword"]
    cases = [
        make_case("easy_001", retrieval_mode=mode) for mode in modes
    ] + [
        make_case("easy_001", retrieval_mode="hybrid", attempt_index=1, passed=False)
    ]
    traces = {
        (case["id"], case["retrieval_mode"], case["attempt_index"]): make_trace(
            case["id"],
            retrieval_mode=case["retrieval_mode"],
            attempt_index=case["attempt_index"],
            context_ids=["chunk-1", "chunk-2"],
        )
        for case in cases
    }
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_modes",
        cases=cases,
        traces=traces,
        retrieval_mode="all",
    )
    with eval_client(tmp_path, config=cfg) as client:
        ambiguous = client.get("/api/evaluation/runs/run_modes/tests/easy_001")
        assert ambiguous.status_code == 409
        error = ambiguous.json()["error"]
        assert error["code"] == "ambiguous_test_result"
        assert set(error["details"]["available_modes"]) == {"hybrid", "dense", "keyword"}

        invalid = client.get(
            "/api/evaluation/runs/run_modes/tests/easy_001?retrieval_mode=all"
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_retrieval_mode"

        detail = client.get(
            "/api/evaluation/runs/run_modes/tests/easy_001?retrieval_mode=dense"
        )
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["retrieval_mode"] == "dense"
        assert payload["attempt_index"] == 0
        assert payload["test"]["question"]
        assert payload["query_plan"]["intent"] == "factoid"
        assert payload["retrieval_calls"]
        assert payload["evidence"]["manifest"]
        assert payload["evidence"]["context_chunk_ids"] == ["chunk-1", "chunk-2"]
        assert payload["answer"]["citations"]
        assert payload["trace_available"] is True

        attempt = client.get(
            "/api/evaluation/runs/run_modes/tests/easy_001"
            "?retrieval_mode=hybrid&attempt_index=1"
        )
        assert attempt.status_code == 200
        assert attempt.json()["attempt_index"] == 1
        assert attempt.json()["evaluation"]["pass"] is False

        missing = client.get(
            "/api/evaluation/runs/run_modes/tests/easy_001"
            "?retrieval_mode=dense&attempt_index=5"
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "test_not_found"

        trace = client.get(
            "/api/evaluation/runs/run_modes/tests/easy_001/trace?retrieval_mode=dense"
        )
        assert trace.status_code == 200
        assert trace.headers["cache-control"] == "private, no-store"
        assert "prompt" not in trace.text.casefold()


def test_trace_unavailable_is_explicit(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    case = make_case("easy_003", trace_available=False)
    seed_completed_run(tmp_path, cfg, run_id="run_no_trace", cases=[case])
    with eval_client(tmp_path, config=cfg) as client:
        detail = client.get(
            "/api/evaluation/runs/run_no_trace/tests/easy_003?retrieval_mode=hybrid"
        ).json()
        trace = client.get(
            "/api/evaluation/runs/run_no_trace/tests/easy_003/trace?retrieval_mode=hybrid"
        )
    assert detail["trace_available"] is False
    assert detail["trace_url"] is None
    assert detail["evidence"]["context_chunk_ids"] is None
    assert detail["evidence"]["coverage"]
    assert trace.status_code == 404


def test_year_coverage_distinguishes_not_found_and_absent(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    case = make_case("hard_001", suite="hard", difficulty="hard", category="temporal")
    trace = make_trace("hard_001", covered=[2023], missing=[2021])
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_years",
        cases=[case],
        traces={("hard_001", "hybrid", 0): trace},
        suite="hard",
    )
    with eval_client(tmp_path, config=cfg) as client:
        detail = client.get(
            "/api/evaluation/runs/run_years/tests/hard_001?retrieval_mode=hybrid"
        ).json()
    coverage = {row["year"]: row["status"] for row in detail["evidence"]["coverage"]}
    assert coverage[2023] == "covered"
    assert coverage.get(2021) == "not_found"


def test_test_catalog_and_history(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    cases = [
        make_case("hard_001", suite="hard", difficulty="hard", category="temporal", passed=False),
        make_case("easy_001"),
    ]
    seed_completed_run(tmp_path, cfg, run_id="run_catalog", cases=cases, suite="hard")
    with eval_client(tmp_path, config=cfg) as client:
        catalog = client.get("/api/evaluation/tests?difficulty=hard&status=failed").json()
        assert catalog["dataset_available"] is True
        assert [item["id"] for item in catalog["items"]] == ["hard_001"]
        assert catalog["items"][0]["last_run"]["run_id"] == "run_catalog"

        search = client.get("/api/evaluation/tests?q=2023").json()
        assert search["total"] >= 1

        definition = client.get("/api/evaluation/tests/hard_001").json()
        assert definition["test"]["suite"] == "hard"
        assert definition["history"]
        assert definition["history"][0]["run_id"] == "run_catalog"

        missing = client.get("/api/evaluation/tests/unknown_test")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "test_not_found"


def test_history_and_pagination(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    for index in range(3):
        seed_completed_run(
            tmp_path,
            cfg,
            run_id=f"run_{index}",
            cases=[make_case("easy_001", passed=index != 0)],
            suite="smoke",
        )
    with eval_client(tmp_path, config=cfg) as client:
        first = client.get("/api/evaluation/runs?limit=2").json()
        assert first["total"] == 3
        assert len(first["items"]) == 2
        assert first["next_offset"] == 2
        second = client.get("/api/evaluation/runs?limit=2&offset=2").json()
        assert len(second["items"]) == 1
        assert second["next_offset"] is None

        filtered = client.get("/api/evaluation/runs?status=completed&suite=smoke").json()
        assert filtered["total"] == 3


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def test_compare_comparable_runs_reports_deltas_and_regressions(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_base",
        cases=[make_case("easy_001", passed=True), make_case("easy_002", passed=False)],
    )
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_candidate",
        cases=[make_case("easy_001", passed=False), make_case("easy_002", passed=True)],
    )
    with eval_client(tmp_path, config=cfg) as client:
        response = client.get(
            "/api/evaluation/compare?base=run_base&candidate=run_candidate"
        )
        assert response.status_code == 200
        payload = response.json()
    assert payload["comparable"] is True
    assert payload["reasons"] == []
    metrics = {item["key"]: item for item in payload["metrics"]}
    assert metrics["pass_rate"]["base"] == 0.5
    assert metrics["pass_rate"]["candidate"] == 0.5
    assert metrics["pass_rate"]["delta"] == 0.0
    assert [item["test_id"] for item in payload["regressions"]] == ["easy_001"]
    assert [item["test_id"] for item in payload["improvements"]] == ["easy_002"]


def test_compare_not_comparable_when_corpus_differs(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_a",
        cases=[make_case("easy_001")],
        corpus_sha="corpus-a",
    )
    seed_completed_run(
        tmp_path,
        cfg,
        run_id="run_b",
        cases=[make_case("easy_001")],
        corpus_sha="corpus-b",
    )
    with eval_client(tmp_path, config=cfg) as client:
        payload = client.get(
            "/api/evaluation/compare?base=run_a&candidate=run_b"
        ).json()
    assert payload["comparable"] is False
    assert "corpus_content_sha256_mismatch" in payload["reasons"]
    assert all(item["delta"] is None for item in payload["metrics"])
    assert payload["regressions"] == []


def test_compare_requires_finished_runs(tmp_path: Path) -> None:
    cfg = eval_config(tmp_path)
    seed_completed_run(tmp_path, cfg, run_id="run_done", cases=[make_case("easy_001")])
    store = EvaluationStore(cfg.eval_db_path)
    store.init()
    store.create_run(
        run_id="run_waiting",
        suite="smoke",
        split="dev",
        retrieval_mode="hybrid",
        judge=False,
        repeat=1,
        request={},
    )
    with eval_client(tmp_path, config=cfg) as client:
        response = client.get(
            "/api/evaluation/compare?base=run_done&candidate=run_waiting"
        )
        missing = client.get("/api/evaluation/compare?base=run_done&candidate=nope")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "run_not_finished"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"


def test_suites_endpoint_lists_dataset(tmp_path: Path) -> None:
    with eval_client(tmp_path) as client:
        payload = client.get("/api/evaluation/suites").json()
    assert payload["dataset_available"] is True
    suites = {item["suite"]: item for item in payload["items"]}
    assert suites["smoke"]["tests"] == 12
    assert suites["full"]["tests"] == 30
    assert suites["full"]["dev"] == 24
    assert suites["full"]["holdout"] == 6
    assert suites["easy"]["dev"] == 8
    assert suites["easy"]["holdout"] == 2
    assert payload["max_queued"] == 3
