"""Контракт ``AgentRunResult.diagnostics``: retrieval_calls, coverage и trace.

Тесты проверяют нормализацию без запуска LangGraph, поэтому не зависят от
поведения селектора: ToolEvent'ы собираются напрямую, как их формирует граф.
"""

from __future__ import annotations

from dataclasses import asdict

from src.agent.coverage import CoverageCell, CoverageReport
from src.agent.service import _coverage_payload, _retrieval_calls
from src.agent.state import ToolEvent
from src.retrieval.hybrid_retriever import RetrievalTrace, SearchFilters


def trace() -> RetrievalTrace:
    return RetrievalTrace(
        dense_candidates=3,
        bm25_candidates=2,
        fused_candidates=4,
        rerank_candidates=2,
        returned=1,
        distinct_docs=1,
        stages={
            "dense": [{"chunk_id": "c1", "rank": 1, "score": 0.9}],
            "bm25": [{"chunk_id": "c1", "rank": 2, "score": 7.0}],
            "rrf": [{"chunk_id": "c1", "rank": 1, "score": 0.03}],
            "returned": [{"chunk_id": "c1", "rank": 1, "score": 0.03}],
            "rerank": None,
        },
        available={
            "dense": True, "bm25": True, "rrf": True, "returned": True,
            "rerank": False,
        },
        fallback={"reranker": True},
        unavailable={"rerank": "fallback"},
    )


def search_event(**overrides) -> ToolEvent:
    values = {
        "tool_name": "search_reports",
        "arguments": {
            "query": "чистая прибыль 2023",
            "queries": ["чистая прибыль 2023", "net income 2023"],
            "filters": SearchFilters(years={2023}),
            "retrieval_mode": "hybrid",
            "top_k": 5,
        },
        "status": "completed",
        "result_count": 1,
        "warning": None,
        "diagnostics": trace().to_dict(),
    }
    values.update(overrides)
    return ToolEvent(**values)


def test_search_reports_trace_to_dict_lands_in_retrieval_calls() -> None:
    calls = _retrieval_calls([search_event()])

    assert len(calls) == 1
    call = calls[0]
    assert call["call_index"] == 0
    assert call["tool_name"] == "search_reports"
    assert call["query"] == "чистая прибыль 2023"
    assert call["queries"] == ["чистая прибыль 2023", "net income 2023"]
    assert call["retrieval_mode"] == "hybrid"
    assert call["top_k"] == 5
    assert call["filters"] == {
        "years": [2023],
        "year_from": None,
        "year_to": None,
        "report_periods": None,
        "issuers": None,
        "kinds": None,
        "allowed_doc_ids": None,
        "section_path_prefix": None,
        "breadcrumb_contains": None,
        "max_section_level": None,
    }
    assert call["counts"] == {
        "dense": 3, "bm25": 2, "rrf": 4, "rerank": 2, "returned": 1,
    }
    assert [item["chunk_id"] for item in call["stages"]["dense"]] == ["c1"]
    assert call["stages"]["rerank"] is None
    assert call["available"]["rerank"] is False
    assert call["fallback"] == {"reranker": True}
    assert call["unavailable"] == {"rerank": "fallback"}


def test_call_index_follows_only_retrieval_calls_in_order() -> None:
    events = [
        search_event(),
        ToolEvent("get_structure_tree", {"doc_id": "d1"}, "completed", 2),
        search_event(
            arguments={"query": "q2", "filters": SearchFilters(), "top_k": 3},
        ),
        ToolEvent("get_chunk", {"chunk_id": "c1"}, "completed", 1),
    ]

    calls = _retrieval_calls(events)

    assert [call["call_index"] for call in calls] == [0, 1]
    assert [call["tool_name"] for call in calls] == [
        "search_reports", "search_reports",
    ]
    assert calls[1]["query"] == "q2"
    assert calls[1]["queries"] == ["q2"]


def test_get_entities_is_a_separate_graph_call() -> None:
    events = [
        search_event(),
        ToolEvent(
            "get_entities",
            {
                "entity_names": ["Кредитный риск"],
                "filters": SearchFilters(),
                "max_hops": 2,
            },
            "completed",
            result_count=1,
            diagnostics={
                "stages": {
                    "graph": [
                        {
                            "chunk_id": "c2",
                            "rank": 1,
                            "score": 0.8,
                            "path": ["entity:risk", "chunk:c2"],
                            "evidence_chunk_ids": ["c2"],
                            "matched_entities": ["entity:risk"],
                        }
                    ]
                },
                "available": {"graph": True},
                "fallback": {"graph": False},
                "unavailable": {},
                "counts": {"graph": 1},
            },
        ),
    ]

    calls = _retrieval_calls(events)

    assert [call["call_index"] for call in calls] == [0, 1]
    graph_call = calls[1]
    assert graph_call["tool_name"] == "get_entities"
    assert graph_call["query"] is None
    assert graph_call["entity_names"] == ["Кредитный риск"]
    assert [item["chunk_id"] for item in graph_call["stages"]["graph"]] == ["c2"]
    assert graph_call["counts"] == {"graph": 1}
    assert graph_call["available"] == {"graph": True}


def test_empty_trace_does_not_invent_stages_or_counts() -> None:
    event = ToolEvent(
        "search_reports",
        {"query": "q", "filters": SearchFilters(), "top_k": 5},
        "completed",
        result_count=0,
        diagnostics=None,
    )

    call = _retrieval_calls([event])[0]

    assert call["stages"] == {}
    assert call["counts"] == {}
    assert call["available"] == {}
    assert call["fallback"] == {}
    assert call["unavailable"] == {}


def test_dataclass_asdict_trace_is_normalized_with_derived_counts() -> None:
    event = search_event(diagnostics=asdict(trace()))

    call = _retrieval_calls([event])[0]

    # ``asdict`` не содержит ``counts``; они вычисляются из тех же счётчиков.
    assert call["counts"] == {
        "dense": 3, "bm25": 2, "rrf": 4, "rerank": 2, "returned": 1,
    }
    assert [item["chunk_id"] for item in call["stages"]["returned"]] == ["c1"]
    assert call["available"]["rerank"] is False


def test_event_warnings_are_safe_and_deduplicated() -> None:
    event = search_event(
        warning="graph_unavailable",
        diagnostics={**trace().to_dict(), "warnings": ["graph_unavailable"]},
    )

    call = _retrieval_calls([event])[0]

    assert call["warnings"] == ["graph_unavailable"]


def test_graph_trace_asdict_keeps_graph_diagnostics() -> None:
    from src.retrieval.graph_hybrid import GraphRetrievalTrace

    graph_trace = GraphRetrievalTrace(
        graph_candidates=2,
        graph_added=1,
        graph_matched_entities=["entity:risk"],
        graph_added_chunk_ids=["c2"],
        stages={
            "graph": [
                {
                    "chunk_id": "c2",
                    "rank": 1,
                    "score": 0.8,
                    "path": ["entity:risk", "chunk:c2"],
                    "evidence_chunk_ids": ["c2"],
                    "matched_entities": ["entity:risk"],
                }
            ]
        },
        available={"graph": True},
        fallback={"graph": False},
    )
    event = search_event(diagnostics=asdict(graph_trace))

    call = _retrieval_calls([event])[0]

    assert call["counts"]["graph"] == 2
    assert call["graph_added"] == 1
    assert call["graph_added_chunk_ids"] == ["c2"]
    assert call["graph_matched_entities"] == ["entity:risk"]
    assert [item["chunk_id"] for item in call["stages"]["graph"]] == ["c2"]
    assert call["available"]["graph"] is True


def test_coverage_payload_uses_to_dict_and_keeps_plain_dicts() -> None:
    report = CoverageReport(
        status="partial",
        expected_years=[2023, 2024],
        covered_years=[2023],
        missing_years=[],
        cells=[
            CoverageCell("net_profit", 2023, "covered", ["c1"]),
            CoverageCell("net_profit", 2024, "not_found", []),
        ],
        required_metrics=["net_profit"],
    )

    payload = _coverage_payload(report)

    assert payload is not None
    assert payload["status"] == "partial"
    assert payload["expected_years"] == [2023, 2024]
    assert payload["cells"][0]["metric"] == "net_profit"
    assert payload["cells"][1]["status"] == "not_found"
    assert _coverage_payload({"status": "complete"}) == {"status": "complete"}
    assert _coverage_payload(None) is None


def test_retrieval_calls_do_not_share_mutable_state_with_events() -> None:
    event = search_event()
    call = _retrieval_calls([event])[0]

    call["stages"]["dense"].append({"chunk_id": "c9", "rank": 9, "score": 0.0})
    call["filters"]["years"].append(1999)

    # Исходный trace и фильтры не мутированы повторной нормализацией.
    assert len(event.diagnostics["stages"]["dense"]) == 1
    assert event.arguments["filters"].years == {2023}
    assert _retrieval_calls([event])[0]["stages"]["dense"] == [
        {"chunk_id": "c1", "rank": 1, "score": 0.9}
    ]
