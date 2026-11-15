"""Сериализация backend-контрактов без потери аналитики и без секретов."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.state import AgentEvent, Answer, Citation
from src.api import serialize
from src.api.schemas import FilterSet
from src.retrieval.hybrid_retriever import SearchFilters
from test_web_api_support import make_citation, make_result


def test_result_message_fields_preserve_analysis_contracts() -> None:
    citation = make_citation()
    result = make_result(
        citations=[citation],
        warnings=["agent_warning", "shared"],
        answer_warnings=["shared", "answer_warning"],
        calculations=[{"metric": "revenue", "cagr": 0.1}],
        coverage={
            "status": "partial",
            "expected_years": [2023, 2024],
            "covered_years": [2023],
            "missing_years": [2024],
            "metrics": {"revenue": {"2023": "covered", "2024": "year_absent"}},
        },
        verification={"status": "ok", "checked_claims": 1},
        structural_changes=[
            {
                "change_type": "added",
                "from_year": 2023,
                "to_year": 2024,
                "detail": "ESG",
            }
        ],
        effective_filters=SearchFilters(
            years={2023}, issuers={"ПАО Сбербанк"}
        ),
    )

    metadata = {
        citation.chunk_id: {
            "chunk_id": citation.chunk_id,
            "title": "Годовой отчёт 2023",
            "kind": "table",
            "year": 2023,
            "doc_id": "sber_2023",
            "page_end": 13,
        }
    }
    fields = serialize.result_message_fields(result, metadata.get)

    assert fields["status"] == "completed"
    assert fields["content"] == result.answer.answer
    assert fields["warnings"] == [
        "agent_warning",
        "shared",
        "answer_warning",
    ]
    analysis = fields["analysis"]
    assert analysis["calculations"] == [{"metric": "revenue", "cagr": 0.1}]
    assert analysis["facts"][0]["metric"] == "revenue"
    assert analysis["coverage"]["status"] == "partial"
    assert analysis["structural_changes"][0]["change_type"] == "added"
    assert analysis["verification"] == {"status": "ok", "checked_claims": 1}
    assert fields["effective_filters"] == {
        "years": [2023],
        "issuers": ["ПАО Сбербанк"],
    }

    evidence = fields["evidence"][0]
    assert evidence["evidence_id"] == citation.chunk_id
    assert evidence["document_title"] == "Годовой отчёт 2023"
    assert evidence["source_type"] == "table"
    assert evidence["page_end"] == 13
    assert evidence["has_full_text"] is True
    assert evidence["excerpt"] == citation.quote


def test_answer_json_roundtrip_keeps_analysis_and_effective_filters() -> None:
    result = make_result(
        coverage={"status": "complete"},
        verification={"status": "ok"},
        structural_changes=[{"change_type": "stable"}],
        calculations=[{"metric": "x"}],
    )
    fields = serialize.result_message_fields(result)
    stored = {
        "id": "message-1",
        "research_id": "research-1",
        "role": "assistant",
        "content": fields["content"],
        "status": fields["status"],
        "request_id": "request-1",
        "created_at": "2026-11-14T10:00:00Z",
        "answer_json": serialize.answer_json_from_fields(fields),
        "filters_json": None,
        "error_json": None,
    }
    message = serialize.message_from_stored(stored)
    assert message["analysis"]["coverage"] == {"status": "complete"}
    assert message["analysis"]["verification"] == {"status": "ok"}
    assert message["analysis"]["structural_changes"] == [
        {"change_type": "stable"}
    ]
    assert message["analysis"]["calculations"] == [{"metric": "x"}]
    assert message["effective_filters"] == {"years": [2023]}


def test_citation_mapping_does_not_invent_missing_metadata() -> None:
    citation = Citation(
        chunk_id="chunk-1",
        doc_id=None,
        year=None,
        issuer=None,
        page_pdf=None,
        page_label=None,
        section_path=None,
        quote=None,
    )
    evidence = serialize.citation_to_evidence(citation)
    assert evidence["document_title"] is None
    assert evidence["source_type"] is None
    assert evidence["page_end"] is None
    assert evidence["excerpt"] == ""
    assert evidence["has_full_text"] is False
    assert evidence["section_path"] is None


def test_filters_roundtrip_and_empty_values() -> None:
    assert serialize.filter_set_to_search(None) is None
    assert serialize.filter_set_to_search(
        FilterSet(years=[], issuers=[], allowed_doc_ids=[])
    ) is None

    filters = serialize.filter_set_to_search(
        FilterSet(
            years=[2024, 2023],
            year_from=2023,
            year_to=2024,
            issuers=["ПАО Сбербанк"],
            allowed_doc_ids=["sber_2023"],
            kinds=["table"],
        )
    )
    assert filters is not None
    assert filters.years == {2023, 2024}
    assert filters.year_from == 2023
    assert filters.year_to == 2024
    assert filters.issuers == {"ПАО Сбербанк"}
    assert filters.allowed_doc_ids == {"sber_2023"}
    assert filters.kinds == {"table"}
    assert serialize.filters_to_public(filters) == {
        "years": [2023, 2024],
        "year_from": 2023,
        "year_to": 2024,
        "issuers": ["ПАО Сбербанк"],
        "allowed_doc_ids": ["sber_2023"],
        "kinds": ["table"],
    }


def test_tool_event_public_payload_hides_query_and_filters() -> None:
    event = AgentEvent(
        "tool_started",
        "backend-1",
        {
            "tool_name": "search_reports",
            "short_arguments": {
                "query": "секретный вопрос",
                "queries": ["вариант"],
                "chunk_id": "chunk-1",
                "entity_names": ["Сбербанк"],
                "filters": {"years": [2023]},
                "api_key": "secret",
                "top_k": 5,
                "doc_id": "sber_2023",
            },
        },
    )
    payload = serialize.agent_event_payload(
        event, request_id="public-1", message_id="message-1"
    )
    assert payload == {
        "type": "tool_started",
        "request_id": "public-1",
        "tool_name": "search_reports",
        "short_arguments": {"top_k": 5, "doc_id": "sber_2023"},
    }


def test_run_failed_event_is_safe_and_retryable() -> None:
    event = AgentEvent(
        "run_failed",
        "backend-1",
        {
            "error_code": "agent_run_failed",
            "user_message": "Не удалось безопасно завершить обработку запроса.",
            "raw_exception": "secret traceback",
        },
    )
    payload = serialize.agent_event_payload(
        event, request_id="public-1", message_id="message-1"
    )
    assert payload["code"] == "agent_run_failed"
    assert payload["retryable"] is True
    assert "secret" not in str(payload)
    assert "traceback" not in str(payload)


def test_unknown_backend_events_are_not_republished() -> None:
    event = AgentEvent("internal_debug", "backend-1", {"prompt": "secret"})
    assert (
        serialize.agent_event_payload(
            event, request_id="public-1", message_id="message-1"
        )
        is None
    )


def test_abstained_result_maps_to_abstained_message() -> None:
    result = make_result(
        citations=[],
        answer_status="abstain",
        status="abstained",
        answer_text="В доступных материалах не найдено подтверждение.",
    )
    fields = serialize.result_message_fields(result)
    assert fields["status"] == "abstained"
    assert fields["evidence"] == []


def test_analysis_does_not_require_real_dataclasses() -> None:
    result = SimpleNamespace(
        status="completed",
        answer=Answer("answer", "text", [make_citation()]),
        calculations=[{"metric": "x"}],
        warnings=[],
        structural_changes=[],
        diagnostics={},
        effective_filters=SearchFilters(),
    )
    fields = serialize.result_message_fields(result)
    assert fields["analysis"]["calculations"] == [{"metric": "x"}]
    assert fields["effective_filters"] == {}
