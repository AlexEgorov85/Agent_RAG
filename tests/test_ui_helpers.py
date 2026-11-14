from __future__ import annotations

from types import SimpleNamespace

from src.agent.state import AgentEvent, AgentRunResult, Answer, Citation
from src.retrieval.hybrid_retriever import SearchFilters
from src.ui.app import (
    build_service_filters,
    event_result,
    new_conversation_id,
    normalize_filters,
    safe_tool_arguments,
    summarize_event,
)
from src.ui.citations import citation_metadata, citation_quote


def test_citation_helpers_select_contract_metadata_and_quote() -> None:
    citation = {
        "chunk_id": "sber_2023:00042:c0",
        "doc_id": "sber_2023",
        "year": 2023,
        "issuer": "ПАО Сбербанк",
        "page_pdf": 42,
        "page_label": "41",
        "section_path": "Риски/Кредитный риск",
        "quote": "Короткая цитата",
        "private_field": "must not be rendered",
    }

    assert citation_metadata(citation) == {
        "chunk_id": "sber_2023:00042:c0",
        "doc_id": "sber_2023",
        "year": 2023,
        "issuer": "ПАО Сбербанк",
        "page_pdf": 42,
        "page_label": "41",
        "section_path": "Риски/Кредитный риск",
    }
    assert citation_quote(citation) == "Короткая цитата"


def test_citation_helpers_support_objects_and_missing_quote() -> None:
    citation = SimpleNamespace(chunk_id="x", quote="  ")
    assert citation_metadata(citation) == {"chunk_id": "x"}
    assert citation_quote(citation) is None


def test_event_summary_keeps_only_safe_tool_trace_fields() -> None:
    started = summarize_event(SimpleNamespace(
        type="tool_started",
        tool_name="search_reports",
        short_arguments={"query": "sensitive full user question", "years": [2023], "api_key": "secret"},
    ))
    assert started == {
        "event": "tool_started",
        "tool_name": "search_reports",
        "arguments": {"query": "[запрос скрыт]", "years": [2023]},
        "stage": "Вызов инструмента: search_reports",
    }

    completed = summarize_event({
        "type": "tool_completed",
        "tool_name": "search_reports",
        "result_count": 3,
        "duration_ms": 18,
        "status": "ok",
        "raw_exception": "not included",
    })
    assert completed == {
        "event": "tool_completed",
        "tool_name": "search_reports",
        "result_count": 3,
        "duration_ms": 18,
        "status": "ok",
    }


def test_filter_normalization_and_unique_session_ids() -> None:
    assert normalize_filters([2023, 2023, None], "ПАО Сбербанк", ["sber_2023", ""]) == {
        "years": {2023},
        "issuers": {"ПАО Сбербанк"},
        "allowed_doc_ids": {"sber_2023"},
    }
    assert normalize_filters() == {}
    assert build_service_filters({}) is None
    assert build_service_filters({"years": {2023}}, lambda **kwargs: kwargs) == {"years": {2023}}
    assert new_conversation_id() != new_conversation_id()


def test_safe_tool_arguments_redacts_sensitive_values_and_bounds_lists() -> None:
    args = safe_tool_arguments({
        "query": "question",
        "filters": {"token": "abc", "allowed_doc_ids": ["sber_2023"]},
        "entity_names": ["A", "B", "C", "D", "E", "F"],
    })
    assert args["query"] == "[запрос скрыт]"
    assert args["filters"] == {"allowed_doc_ids": ["sber_2023"]}
    assert args["entity_names"] == ["A", "B", "C", "D", "E"]


def test_actual_agent_events_use_event_type_and_nested_data_safely() -> None:
    started = AgentEvent(
        "tool_started",
        "request-1",
        {
            "tool_name": "search_reports",
            "short_arguments": {
                "query": "full private question",
                "years": [2023],
                "api_key": "private-secret",
            },
            "private_debug": "must not appear",
        },
    )
    completed = AgentEvent(
        "tool_completed",
        "request-1",
        {
            "tool_name": "search_reports",
            "result_count": 4,
            "duration_ms": 12.5,
            "status": "ok",
            "private_debug": "must not appear",
        },
    )
    delta = AgentEvent(
        "answer_delta",
        "request-1",
        {"text": "Confirmed answer text", "provider_body": "private"},
    )
    final_result = AgentRunResult(
        request_id="request-1",
        conversation_id="conversation-1",
        answer=Answer(
            status="answer",
            answer="Confirmed answer text",
            citations=[Citation(chunk_id="sber_2023:00001:c0", year=2023)],
        ),
        tool_events=[],
        effective_filters=SearchFilters(),
        warnings=[],
        timings={"total_ms": 25.0},
    )
    run_completed = AgentEvent(
        "run_completed",
        "request-1",
        {
            "answer_status": "answer",
            "duration_ms": 25.0,
            "private_debug": "must not appear",
        },
        result=final_result,
    )

    started_summary = summarize_event(started)
    assert started_summary["event"] == "tool_started"
    assert started_summary["tool_name"] == "search_reports"
    assert started_summary["arguments"] == {
        "query": "[запрос скрыт]",
        "years": [2023],
    }
    assert summarize_event(completed) == {
        "event": "tool_completed",
        "tool_name": "search_reports",
        "result_count": 4,
        "duration_ms": 12.5,
        "status": "ok",
    }
    assert summarize_event(delta)["delta"] == "Confirmed answer text"
    completed_summary = summarize_event(run_completed)
    assert completed_summary["event"] == "run_completed"
    assert completed_summary["answer_status"] == "answer"
    assert event_result(run_completed) is final_result

    for summary in (started_summary, completed_summary, summarize_event(delta)):
        rendered = repr(summary)
        assert "private" not in rendered
        assert "api_key" not in rendered


def test_event_direct_fields_take_precedence_over_nested_data() -> None:
    summary = summarize_event({
        "event_type": "tool_completed",
        "tool_name": "get_chunk",
        "result_count": 1,
        "data": {
            "tool_name": "search_reports",
            "result_count": 99,
            "duration_ms": 7,
            "status": "ok",
        },
    })
    assert summary["tool_name"] == "get_chunk"
    assert summary["result_count"] == 1
    assert summary["duration_ms"] == 7
