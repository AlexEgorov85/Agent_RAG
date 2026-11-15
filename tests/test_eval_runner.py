"""Офлайн-тесты runner: артефакты, identity, coverage и fake-service."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.dataset import (
    DatasetError,
    SUITE_NAMES,
    corpus_content_sha256,
    dataset_hash,
    suite_hash,
)
from eval.evaluators.answer import evaluate_answer
from eval.evaluators.dialogue import evaluate_dialogue
from eval.runner import (
    _capabilities,
    load_graph_fingerprint,
    run_suite,
)
from eval.trace import build_turn_trace
from src.agent.state import AgentRunResult, Answer, Citation, ToolEvent
from src.retrieval.hybrid_retriever import SearchFilters

CHUNK_ID = "sber_2023:00001:c0:aaaaaaaaaaaa"


def chunks() -> list[dict]:
    return [
        {
            "chunk_id": CHUNK_ID,
            "doc_id": "sber_2023",
            "year": 2023,
            "page_pdf": 8,
            "text": "Чистая прибыль за 2023 год составила 1 508,6 млрд рублей.",
            "source_sha256": "a" * 64,
        },
        {
            "chunk_id": "sber_2022:00002:c0:bbbbbbbbbbbb",
            "doc_id": "sber_2022",
            "year": 2022,
            "page_pdf": 30,
            "text": "В 2022 году Сбер продолжал развивать экосистему.",
            "source_sha256": "b" * 64,
        },
    ]


def answer_test(test_id: str = "easy_001", suite: str = "easy") -> dict:
    return {
        "id": test_id,
        "suite": suite,
        "split": "dev",
        "category": "fact_lookup",
        "difficulty": "easy" if suite == "easy" else "medium",
        "tags": ["smoke"],
        "question": "Какая чистая прибыль Сбербанка за 2023 год?",
        "filters": {"years": [2023]},
        "expected_status": "answer",
        "expected": {
            "required_facts": [
                {
                    "metric": "net_profit",
                    "metric_aliases": ["чистая прибыль"],
                    "year": 2023,
                    "value": 1508.6,
                    "unit": "млрд руб.",
                    "tolerance": 0.01,
                }
            ],
            "required_years": [2023],
            "required_documents": ["sber_2023"],
            "required_answer_elements": ["значение"],
            "required_calculations": [],
            "required_structural_changes": [],
            "missing_years": [],
        },
        "gold_evidence": {
            "min_coverage": 1.0,
            "groups": [{"id": "profit_2023", "any_of": [CHUNK_ID]}],
        },
        "expected_behavior": {
            "retrieval_modes": ["hybrid"],
            "max_search_calls": 2,
            "max_tool_calls": 4,
            "must_use_tools": ["search_reports"],
            "must_use_tools_strict": False,
            "may_use_tools": ["get_chunk"],
            "must_not_use_tools": [],
            "strict_search_budget": False,
            "must_cite_sources": True,
            "must_abstain_if_missing": True,
        },
        "checks": {
            "must_mention": ["чистая прибыль"],
            "must_not_mention": [],
            "forbidden_values": [],
            "forbidden_years": [],
            "expected_warnings": [],
            "must_not_invent_values": True,
            "strict_quotes": False,
        },
        "notes": "offline fixture",
    }


def abstain_test() -> dict:
    return {
        "id": "medium_001",
        "suite": "medium",
        "split": "dev",
        "category": "abstain",
        "difficulty": "medium",
        "tags": ["smoke"],
        "question": "Какая чистая прибыль за 2021 год?",
        "filters": {"years": [2021]},
        "expected_status": "abstain",
        "expected": {
            "required_facts": [],
            "required_years": [],
            "required_documents": [],
            "required_answer_elements": ["причина отказа"],
            "required_calculations": [],
            "required_structural_changes": [],
            "missing_years": [2021],
            "reason_codes": ["no_evidence"],
        },
        "gold_evidence": {"min_coverage": 0.0, "groups": []},
        "expected_behavior": {
            "retrieval_modes": ["hybrid"],
            "max_search_calls": 2,
            "max_tool_calls": 4,
            "must_use_tools": [],
            "must_use_tools_strict": False,
            "may_use_tools": ["search_reports"],
            "must_not_use_tools": [],
            "strict_search_budget": False,
            "must_cite_sources": False,
            "must_abstain_if_missing": True,
        },
        "checks": {
            "must_mention": [],
            "must_not_mention": [],
            "forbidden_values": [],
            "forbidden_years": [],
            "expected_warnings": ["no_evidence"],
            "must_not_invent_values": True,
            "strict_quotes": False,
        },
        "notes": "offline fixture: 2021 отсутствует",
    }


def multi_turn_test() -> dict:
    turn = answer_test("multi_turn_001", "multi_turn")
    turn["suite"] = "multi_turn"
    turn["category"] = "dialogue"
    turn["difficulty"] = "medium"
    turn.pop("filters", None)
    second = json.loads(json.dumps(turn))
    turn["question"] = "Какая чистая прибыль Группы за 2023 год?"
    second["question"] = "Уточни чистую прибыль Группы за 2023 год."
    second["behavior"] = "continue_previous_request"
    return {
        "id": "multi_turn_001",
        "suite": "multi_turn",
        "split": "dev",
        "category": "dialogue",
        "difficulty": "medium",
        "tags": ["smoke"],
        "turns": [turn, second],
        "scenario_checks": [
            "context_preserved",
            "correct_metric",
            "correct_period",
            "no_unnecessary_restart",
            "sources_preserved",
        ],
        "required_checks": [
            "context_preserved",
            "correct_metric",
            "correct_period",
            "no_unnecessary_restart",
            "sources_preserved",
        ],
        "notes": "offline fixture",
    }


def write_dataset(
    directory: Path,
    *,
    tests: dict[str, list[dict]] | None = None,
    chunk_rows: list[dict] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    chunk_rows = chunk_rows or chunks()
    by_suite = {suite: [] for suite in SUITE_NAMES}
    for suite, rows in (tests or {"easy": [answer_test()]}).items():
        by_suite[suite] = rows
    expected = corpus_content_sha256(chunk_rows)
    payloads = {
        suite: {
            "schema_version": "1.1",
            "suite": suite,
            "expected_corpus_sha256": expected,
            "tests": by_suite[suite],
        }
        for suite in SUITE_NAMES
    }
    for suite, payload in payloads.items():
        (directory / f"{suite}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "1.1",
                "partial": True,
                "expected_corpus_sha256": expected,
                "chunk_count": len(chunk_rows),
                "documents": sorted({chunk["doc_id"] for chunk in chunk_rows}),
                "suite_hashes": {
                    suite: suite_hash(payload)
                    for suite, payload in payloads.items()
                },
                "notes": "offline fixture",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class RecordingService:
    """Минимальный fake AgentService с наблюдаемыми citations."""

    supported_retrieval_modes = ("hybrid",)

    def __init__(self, chunk_rows: list[dict], *, quote: str | None = None):
        self.chunk_rows = chunk_rows
        self.calls: list[dict] = []
        self.quote = quote or "Чистая прибыль за 2023 год составила 1 508,6 млрд рублей."

    def ask(self, query: str, *, conversation_id: str, filters: SearchFilters | None = None):
        self.calls.append(
            {
                "query": query,
                "conversation_id": conversation_id,
                "filters": filters,
            }
        )
        if "2021" in query:
            return AgentRunResult(
                request_id="req-abstain",
                conversation_id=conversation_id,
                answer=Answer(
                    "abstain",
                    "В доступных материалах нет данных за 2021 год.",
                    warnings=["no_evidence"],
                ),
                tool_events=[],
                effective_filters=filters or SearchFilters(),
                warnings=["no_evidence"],
                timings={"total_ms": 1.0},
                status="abstained",
                diagnostics={
                    "query_plan": {"intent": "factoid", "years": [2021]},
                    "search_attempts": [],
                    "retrieval_calls": [],
                    "evidence_manifest": [],
                    "context_chunk_ids": None,
                    "context_source": "evidence_manifest",
                    "coverage": {"covered_years": [], "missing_years": [2021]},
                },
            )
        chunk = self.chunk_rows[0]
        return AgentRunResult(
            request_id="req-answer",
            conversation_id=conversation_id,
            answer=Answer(
                "answer",
                "Чистая прибыль за 2023 год составила 1 508,6 млрд руб.",
                citations=[
                    Citation(
                        CHUNK_ID,
                        doc_id="sber_2023",
                        year=2023,
                        page_pdf=8,
                        quote=self.quote,
                    )
                ],
                facts=[
                    {
                        "metric": "net_profit",
                        "year": 2023,
                        "value": 1508.6,
                        "unit": "млрд руб.",
                        "chunk_id": CHUNK_ID,
                    }
                ],
            ),
            tool_events=[
                ToolEvent(
                    tool_name="search_reports",
                    arguments={
                        "query": query,
                        "retrieval_mode": "hybrid",
                        "top_k": 5,
                        "filters": filters or SearchFilters(),
                    },
                    status="completed",
                    result_count=1,
                    duration_ms=1.0,
                )
            ],
            effective_filters=filters or SearchFilters(),
            warnings=[],
            timings={"total_ms": 2.0},
            status="completed",
            diagnostics={
                "query_plan": {
                    "intent": "factoid",
                    "years": [2023],
                    "entities": [],
                    "answer_mode": "concise",
                },
                "search_attempts": [
                    {
                        "kind": "broad",
                        "year": 2023,
                        "years": [2023],
                        "section_path": None,
                        "result_count": 1,
                        "covered_years": [2023],
                    }
                ],
                "retrieval_calls": [
                    {
                        "retrieval_mode": "hybrid",
                        "counts": {"dense": 1, "bm25": 1, "rrf": 1, "rerank": 1, "returned": 1},
                        "stages": {
                            "dense": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 1.0}],
                            "bm25": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 1.0}],
                            "rrf": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 1.0}],
                            "rerank": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 1.0}],
                            "returned": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 1.0}],
                        },
                        "available": {
                            "dense": True,
                            "bm25": True,
                            "rrf": True,
                            "rerank": True,
                            "returned": True,
                        },
                        "fallback": {"reranker": False, "dense": False, "bm25": False},
                    }
                ],
                "evidence_manifest": [CHUNK_ID],
                "context_chunk_ids": [CHUNK_ID],
                "context_source": "context",
                "coverage": {"covered_years": [2023], "missing_years": []},
            },
        )


def test_runner_creates_artifacts_and_counts_coverage(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    write_dataset(
        dataset_dir,
        tests={"easy": [answer_test()], "medium": [abstain_test()]},
    )
    out = tmp_path / "results"
    run_dir = run_suite(
        suite="full",
        split="dev",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks()),
        out_dir=out,
    )

    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "cases.jsonl").is_file()
    assert (run_dir / "retrieval.csv").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "traces").is_dir()
    assert not [path for path in out.iterdir() if path.name.endswith(".tmp")]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["tests_selected"] == 2
    assert summary["coverage"]["tests_eligible"] == 2
    assert summary["coverage"]["executed"] == 2
    assert summary["coverage"]["passed"] == 2
    assert summary["pass_rate"] == 1.0
    assert summary["execution_coverage"] == 1.0
    assert summary["repeat"] == 1
    assert summary["pass_at_1"] == 1.0
    # При одном повторе stability не измеряется и не подменяется pass@1.
    assert summary["stability"] is None
    # Для provided-service graph fingerprint не выдумывается из чужого артефакта.
    assert summary["corpus"]["graph_fingerprint"] is None
    assert summary["groups"]["retrieval"]["recall_at_10"] == 1.0
    assert summary["dataset"]["sha256"] == dataset_hash(
        {
            suite: json.loads((dataset_dir / f"{suite}.json").read_text(encoding="utf-8"))
            for suite in SUITE_NAMES
        }
    )
    cases = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cases) == 2
    by_id = {case["id"]: case for case in cases}
    # У abstain-теста retrieval-метрики не измерялись и остаются None.
    assert by_id["medium_001"]["retrieval"]["recall_at_10"] is None
    assert by_id["medium_001"]["retrieval"]["loss_stage"] is None
    assert "recall_at_10" in by_id["medium_001"]["retrieval"]["not_measured"]

    trace = json.loads(
        (run_dir / "traces" / "easy_001.hybrid.json").read_text(encoding="utf-8")
    )
    assert trace["trace_schema_version"] == "1.1"
    assert trace["turns"][0]["query_plan"]["years"] == [2023]
    assert trace["turns"][0]["evidence_manifest"] == [CHUNK_ID]
    assert "expected" not in trace and "gold_evidence" not in trace
    assert "prompt" not in json.dumps(trace, ensure_ascii=False).casefold()


def test_invalid_dataset_does_not_start_run(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    bad = answer_test()
    bad["gold_evidence"]["groups"][0]["any_of"] = ["sber_2023:fantom"]
    write_dataset(dataset_dir, tests={"easy": [bad]})
    out = tmp_path / "results"
    with pytest.raises(DatasetError):
        run_suite(
            suite="easy",
            split="dev",
            datasets_dir=dataset_dir,
            chunks=chunks(),
            service=RecordingService(chunks()),
            out_dir=out,
        )
    assert not out.exists() or not list(out.iterdir())


def test_one_conversation_id_per_scenario_and_repeat(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    write_dataset(
        dataset_dir,
        tests={"easy": [answer_test()], "multi_turn": [multi_turn_test()]},
    )
    service = RecordingService(chunks())
    run_dir = run_suite(
        suite="full",
        split="dev",
        repeat=2,
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=service,
        out_dir=tmp_path / "results",
    )
    assert run_dir.is_dir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["repeat"] == 2
    assert summary["pass_at_1"] == 1.0
    assert summary["stability"] == 1.0
    # 1 easy * 2 попытки + 2 turn'а multi-turn * 2 попытки = 6 вызовов.
    assert len(service.calls) == 6
    from collections import Counter

    per_conversation = Counter(
        call["conversation_id"] for call in service.calls
    )
    # Два single-turn сценария (по одному turn'у) и два multi-turn (по два turn'а).
    assert sorted(per_conversation.values()) == [1, 1, 2, 2]
    # Внутри multi-turn оба turn'а используют один conversation_id и остаются
    # в одном сценарии; разные попытки не переиспользуют его.
    assert all(set(call) == {"query", "conversation_id", "filters"} for call in service.calls)
    traces = sorted((run_dir / "traces").glob("*.json"))
    assert any(".attempt_1." in path.name for path in traces)


def test_unsupported_mode_is_skipped_not_zero(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    write_dataset(dataset_dir, tests={"easy": [answer_test()]})
    run_dir = run_suite(
        suite="easy",
        split="dev",
        retrieval_mode="dense",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks()),
        out_dir=tmp_path / "results",
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["skipped"] == 1
    assert summary["coverage"]["unsupported"] == 1
    assert summary["pass_rate"] is None
    assert summary["execution_coverage"] == 0.0
    assert summary["groups"]["retrieval"]["recall_at_10"] is None
    cases = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cases[0]["status"] == "skipped"
    assert cases[0]["skip_reason"] == "unsupported_mode"
    assert cases[0]["pass"] is False
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| Pass rate | — |" in report
    assert "| Retrieval Recall@10 | — |" in report


def test_fake_service_marks_artifacts_as_fake(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    write_dataset(dataset_dir, tests={"easy": [answer_test()]})
    run_dir = run_suite(
        suite="easy",
        split="dev",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        fake_service=True,
        out_dir=tmp_path / "results",
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["fake_service"] is True
    assert any("fake-service" in item for item in summary["limitations"])
    assert (run_dir / "traces" / "easy_001.hybrid.json").is_file()


def test_judge_requested_but_unavailable_is_not_zero(tmp_path: Path):
    dataset_dir = tmp_path / "datasets"
    write_dataset(dataset_dir, tests={"easy": [answer_test()]})
    run_dir = run_suite(
        suite="easy",
        split="dev",
        judge=True,
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks()),
        out_dir=tmp_path / "results",
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["judge_errors"] == 1
    assert summary["groups"]["answer"]["rubric_mean"] is None
    lines = (run_dir / "judge.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    assert json.loads(lines[0])["rubric_total"] is None
    assert json.loads(lines[0])["error"] == "judge_unavailable"


def test_mixed_retrieval_calls_match_events_by_tool_name_and_order() -> None:
    first = ToolEvent(
        "search_reports",
        {
            "query": "q1",
            "queries": ["q1", "q1 alt"],
            "top_k": 5,
            "filters": SearchFilters(years={2023}),
        },
        "completed",
        result_count=1,
        diagnostics={
            "retrieval_mode": "hybrid",
            "stages": {"returned": [{"chunk_id": CHUNK_ID, "rank": 1, "score": 0.9}]},
            "available": {"returned": True},
        },
    )
    entities = ToolEvent(
        "get_entities",
        {"entity_names": ["Credit Risk"], "filters": SearchFilters(), "max_hops": 2},
        "completed",
        result_count=1,
        diagnostics={
            "stages": {
                "graph": [
                    {
                        "chunk_id": "graph-chunk",
                        "rank": 1,
                        "score": 0.8,
                        "path": ["entity:risk"],
                        "evidence_chunk_ids": ["graph-chunk"],
                    }
                ]
            },
            "available": {"graph": True},
            "counts": {"graph": 1},
        },
    )
    second = ToolEvent(
        "search_reports",
        {"query": "q2", "top_k": 3, "filters": SearchFilters(years={2022})},
        "completed",
        result_count=0,
        diagnostics={
            "retrieval_mode": "hybrid",
            "stages": {"returned": []},
            "available": {"returned": True},
        },
    )
    result = SimpleNamespace(
        diagnostics={
            "retrieval_calls": [
                {
                    "tool_name": "search_reports",
                    "retrieval_mode": "hybrid",
                    "counts": {"returned": 1},
                    "stages": {
                        "returned": [
                            {"chunk_id": CHUNK_ID, "rank": 1, "score": 0.9}
                        ]
                    },
                    "available": {"returned": True},
                },
                {
                    "tool_name": "get_entities",
                    "entity_names": ["Credit Risk"],
                    "counts": {"graph": 1},
                    "stages": {
                        "graph": [
                            {
                                "chunk_id": "graph-chunk",
                                "rank": 1,
                                "score": 0.8,
                            }
                        ]
                    },
                    "available": {"graph": True},
                },
                {
                    "tool_name": "search_reports",
                    "query": "raw q2",
                    "queries": ["raw q2"],
                    "top_k": 7,
                    "filters": {"years": [2022]},
                    "retrieval_mode": "hybrid",
                    "stages": {"returned": []},
                    "available": {"returned": True},
                },
            ]
        },
        tool_events=[first, entities, second],
        effective_filters=SearchFilters(),
        timings={},
    )

    turn = build_turn_trace(result, question="mixed calls")
    calls = turn["retrieval_calls"]

    assert [call["tool_name"] for call in calls] == [
        "search_reports", "get_entities", "search_reports",
    ]
    # Первый search: raw call не содержит идентичности — берём её из события.
    assert calls[0]["query"] == "q1"
    assert calls[0]["queries"] == ["q1", "q1 alt"]
    assert calls[0]["top_k"] == 5
    assert calls[0]["filters"]["years"] == [2023]
    # get_entities — отдельный graph-call, а не search_reports-метрика.
    assert calls[1]["query"] is None
    assert calls[1]["entity_names"] == ["Credit Risk"]
    assert [item["chunk_id"] for item in calls[1]["stages"]["graph"]] == [
        "graph-chunk"
    ]
    # Второй search: запрос/фильтры из raw call, а не из смещённого события.
    assert calls[2]["query"] == "raw q2"
    assert calls[2]["queries"] == ["raw q2"]
    assert calls[2]["top_k"] == 7
    assert calls[2]["filters"]["years"] == [2022]


def _dialogue_turn(*, entity: str, year: int, citations: list[str] | None = None) -> dict:
    return {
        "query_plan": {"entities": [entity], "years": [year]},
        "effective_filters": {"years": [year]},
        "answer": {
            "status": "answer",
            "text": "Ответ.",
            "citations": [
                {"chunk_id": chunk_id} for chunk_id in (citations or [])
            ],
        },
        "retrieval_calls": [],
        "tool_events": [],
    }


def _dialogue_answer_rows(count: int) -> dict:
    row = {
        "pass": True,
        "critical_errors": [],
        "warnings": [],
        "metrics": {
            "status_accuracy": 1.0,
            "fact_accuracy": 1.0,
            "abstention_accuracy": None,
        },
        "checks": [
            {"name": "required_years", "pass": True, "mandatory": True, "detail": ""}
        ],
        "facts": [],
        "not_measured": [],
    }
    return {"turns": [dict(row) for _ in range(count)]}


def test_context_preserved_allows_expected_metric_change() -> None:
    test = {
        "turns": [
            {"question": "Чистая прибыль за 2023?", "behavior": "continue_previous_request"},
            {"question": "Не прибыль, а ROE за 2023.", "behavior": "change_metric"},
        ],
        "scenario_checks": ["context_preserved"],
        "required_checks": ["context_preserved"],
    }
    trace = {
        "turns": [
            _dialogue_turn(entity="net_profit", year=2023, citations=["c1"]),
            _dialogue_turn(entity="roe", year=2023, citations=["c2"]),
        ]
    }

    result = evaluate_dialogue(test, trace, _dialogue_answer_rows(2))

    # Период сохранён; entity меняется намеренно (change_metric).
    assert result["checks"]["context_preserved"] is True


def test_context_preserved_still_flags_unexpected_entity_loss() -> None:
    test = {
        "turns": [
            {"question": "Чистая прибыль за 2023?", "behavior": "continue_previous_request"},
            {"question": "И ещё раз за 2023.", "behavior": "continue_previous_request"},
        ],
        "scenario_checks": ["context_preserved"],
        "required_checks": ["context_preserved"],
    }
    trace = {
        "turns": [
            _dialogue_turn(entity="net_profit", year=2023, citations=["c1"]),
            _dialogue_turn(entity="roe", year=2023, citations=["c2"]),
        ]
    }

    result = evaluate_dialogue(test, trace, _dialogue_answer_rows(2))

    assert result["checks"]["context_preserved"] is False


def test_answer_evaluator_accepts_unit_word_forms_and_aliases() -> None:
    test = {
        "expected_status": "answer",
        "expected": {
            "required_facts": [
                {
                    "metric": "gigachat_users",
                    "metric_aliases": ["gigachat"],
                    "year": 2023,
                    "value": 2.6,
                    "unit": "млн человек",
                    "unit_aliases": ["млн человек", "млн пользователей"],
                    "tolerance": 0.01,
                }
            ],
            "required_years": [2023],
            "required_documents": [],
        },
        "checks": {"must_mention": ["GigaChat"], "must_not_invent_values": False},
        "gold_evidence": {"min_coverage": 0.0, "groups": []},
    }
    trace = {
        "answer": {
            "status": "answer",
            "text": "В 2023 году у GigaChat было более 2,6 миллионов пользователей.",
            "facts": [
                {
                    "metric": "Количество пользователей GigaChat",
                    "year": 2023,
                    "value": "2,6+ млн",
                    "unit": "пользователей",
                    "chunk_id": "c1",
                }
            ],
            "citations": [{"chunk_id": "c1", "year": 2023}],
        },
        "evidence_manifest": ["c1"],
        "tool_events": [],
        "query_plan": {"years": [2023]},
        "effective_filters": {"years": [2023]},
    }

    result = evaluate_answer(test, trace)

    assert result["pass"] is True, result["critical_errors"]
    assert result["metrics"]["fact_accuracy"] == 1.0


def test_answer_evaluator_accepts_required_year_from_citation() -> None:
    test = {
        "expected_status": "answer",
        "expected": {
            "required_facts": [],
            "required_years": [2023],
            "required_documents": ["sber_2023"],
        },
        "checks": {"must_mention": ["ESG"], "must_not_invent_values": False},
        "gold_evidence": {"min_coverage": 0.0, "groups": []},
    }
    trace = {
        "answer": {
            "status": "answer",
            "text": "Сбер оценивает ESG-риски в кредитном процессе.",
            "facts": [],
            "citations": [{"chunk_id": "c1", "year": 2023, "doc_id": "sber_2023"}],
        },
        "evidence_manifest": ["c1"],
        "tool_events": [],
        "query_plan": {"years": [2023]},
        "effective_filters": {"years": [2023]},
    }

    result = evaluate_answer(test, trace)

    assert result["pass"] is True, result["critical_errors"]


def test_summary_records_holdout_overlap_warning(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    holdout = answer_test("easy_002")
    holdout["split"] = "holdout"
    holdout["tags"] = []
    holdout["question"] = "Сколько чистой прибыли было за 2023 год?"
    write_dataset(
        dataset_dir,
        tests={"easy": [answer_test(), holdout]},
    )
    run_dir = run_suite(
        suite="easy",
        split="dev",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks()),
        out_dir=tmp_path / "results",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert any(
        item["code"] == "holdout_gold_overlap"
        for item in summary["dataset"]["warnings"]
    )
    assert any("holdout" in item.casefold() for item in summary["limitations"])


def test_capabilities_rewriting_requires_observed_query_variants() -> None:
    service = SimpleNamespace(supported_retrieval_modes=("hybrid",))

    rewritten = {
        "_trace": {
            "turns": [
                {
                    "query_plan": {"needs_rewriting": True},
                    "retrieval_calls": [
                        {"queries": ["чистая прибыль", "net profit 2023"]}
                    ],
                }
            ]
        }
    }
    single = {
        "_trace": {
            "turns": [
                {
                    "query_plan": {"needs_rewriting": True},
                    "retrieval_calls": [{"queries": ["чистая прибыль"]}],
                }
            ]
        }
    }
    rewrite_stage = {
        "_trace": {
            "turns": [
                {
                    "query_plan": {},
                    "retrieval_calls": [
                        {"queries": [], "stages": {"rewrite": [{"query": "q"}]}}
                    ],
                }
            ]
        }
    }

    assert _capabilities([rewritten], service, False)["rewriting"] is True
    # needs_rewriting — запрос, а не доказательство выполненного rewriting.
    assert _capabilities([single], service, False)["rewriting"] is False
    assert _capabilities([rewrite_stage], service, False)["rewriting"] is True


def test_load_graph_fingerprint_reads_active_artifact(tmp_path: Path) -> None:
    fingerprint = "f" * 64
    (tmp_path / "graph.json").write_text(
        json.dumps(
            {
                "graph_schema_version": "1.0",
                "graph_fingerprint": fingerprint,
                "nodes": [],
                "edges": [],
                "manifest": {"status": "complete", "chunk_count": 0},
            }
        ),
        encoding="utf-8",
    )

    assert load_graph_fingerprint(tmp_path) == fingerprint
    assert load_graph_fingerprint(tmp_path / "missing") is None


def test_load_graph_fingerprint_is_none_without_valid_artifact(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    assert load_graph_fingerprint(tmp_path) is None


def test_report_shows_warning_and_failed_check_for_passing_case(tmp_path: Path):
    """citation_quote_mismatch не валит тест, но не должен исчезать из отчёта."""
    dataset_dir = tmp_path / "datasets"
    write_dataset(dataset_dir, tests={"easy": [answer_test()]})
    run_dir = run_suite(
        suite="easy",
        split="dev",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks(), quote="Такой цитаты нет в чанке"),
        out_dir=tmp_path / "results",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert cases[0]["pass"] is True
    assert "citation_quote_mismatch" in cases[0]["warnings"]
    assert any(
        row["code"] == "citation_quote_mismatch" for row in summary["warnings"]
    )
    assert any(
        row["name"] == "citation_quotes" for row in summary["failed_checks"]
    )
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Warnings" in report
    assert "citation_quote_mismatch" in report
    assert "citation_quotes" in report


def test_report_shows_structural_failure_reason(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    test = answer_test()
    test["expected"]["required_structural_changes"] = [
        {
            "type": "restructured",
            "section": "Управление рисками",
            "from_doc": "sber_2022",
            "to_doc": "sber_2023",
        }
    ]
    write_dataset(dataset_dir, tests={"easy": [test]})
    run_dir = run_suite(
        suite="easy",
        split="dev",
        datasets_dir=dataset_dir,
        chunks=chunks(),
        service=RecordingService(chunks()),
        out_dir=tmp_path / "results",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["passed"] == 0
    assert any(
        row["code"] == "structural_changes_missing"
        for row in summary["critical_errors"]
    )
    failed = {row["name"]: row for row in summary["failed_checks"]}
    assert failed["structural_changes"]["mandatory"] is True
    problems = {row["id"]: row for row in summary["problem_tests"]}
    assert "structural_changes_missing" in problems["easy_001"]["reasons"]
    assert "check_failed:structural_changes" in problems["easy_001"]["reasons"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "structural_changes_missing" in report
    assert "check_failed:structural_changes" in report

