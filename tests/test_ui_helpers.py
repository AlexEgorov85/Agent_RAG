from __future__ import annotations

from types import SimpleNamespace

from src.agent.state import AgentEvent, AgentRunResult, Answer, Citation
from src.retrieval.hybrid_retriever import SearchFilters
from src.ui.app import (
    build_service_filters,
    event_result,
    main,
    new_conversation_id,
    normalize_filters,
    safe_tool_arguments,
    service_options,
    summarize_event,
)
from src.ui.citations import (
    api_base_url,
    citation_metadata,
    citation_pdf_link,
    citation_quote,
)


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


def test_service_options_are_derived_from_the_index() -> None:
    backend = SimpleNamespace(
        available_years=lambda: {2003, 2001, 2002},
        available_doc_ids=lambda: {"doc-b", "doc-a"},
    )
    service = SimpleNamespace(tools=SimpleNamespace(search_backend=backend))

    assert service_options(service) == ((2001, 2002, 2003), ("doc-a", "doc-b"))


def test_service_options_are_empty_without_an_index() -> None:
    assert service_options(None) == ((), ())


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


def test_citation_pdf_link_uses_default_api_base_and_page_fragment(
    monkeypatch,
) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)

    link = citation_pdf_link({"doc_id": "sber_2023", "page_pdf": 42})

    assert link == {
        "available": True,
        "url": "http://127.0.0.1:8000/api/documents/sber_2023/file#page=42",
        "label": "Открыть PDF, стр. 42",
        "page": 42,
    }
    assert api_base_url() == "http://127.0.0.1:8000"


def test_citation_pdf_link_without_page_has_no_fragment(monkeypatch) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)

    link = citation_pdf_link({"doc_id": "sber_2023", "page_pdf": None})

    assert link["available"] is True
    assert link["url"] == "http://127.0.0.1:8000/api/documents/sber_2023/file"
    assert "#page=" not in link["url"]
    assert link["label"] == "Открыть PDF"
    assert link["page"] is None


def test_citation_pdf_link_does_not_invent_url_without_doc_id(
    monkeypatch,
) -> None:
    monkeypatch.setenv("API_BASE_URL", "https://reports.example.test")

    for citation in (
        {"page_pdf": 10},
        {"doc_id": "", "page_pdf": 10},
        {"doc_id": "   ", "page_pdf": 10},
    ):
        link = citation_pdf_link(citation)
        assert link == {
            "available": False,
            "url": None,
            "label": "PDF недоступен",
            "page": None,
        }


def test_citation_pdf_link_honours_api_base_url_override(monkeypatch) -> None:
    monkeypatch.setenv("API_BASE_URL", " https://reports.example.test/root/ ")

    link = citation_pdf_link({"doc_id": "sber_2023", "page_pdf": 7})

    assert link["url"] == (
        "https://reports.example.test/root/api/documents/sber_2023/file#page=7"
    )
    assert api_base_url() == "https://reports.example.test/root"


def test_citation_pdf_link_supports_2015_issuer_without_hiding_it(
    monkeypatch,
) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    citation = {
        "chunk_id": "bbs_2015:00001:c0",
        "doc_id": "bbs_2015",
        "year": 2015,
        "issuer": "БПС-Сбербанк (Беларусь)",
        "section_path": "Основные показатели",
        "page_pdf": 5,
        "quote": "Цитата 2015",
    }

    metadata = citation_metadata(citation)
    link = citation_pdf_link(citation)

    assert metadata["issuer"] == "БПС-Сбербанк (Беларусь)"
    assert metadata["year"] == 2015
    assert metadata["section_path"] == "Основные показатели"
    assert metadata["chunk_id"] == "bbs_2015:00001:c0"
    assert link["available"] is True
    assert link["url"] == (
        "http://127.0.0.1:8000/api/documents/bbs_2015/file#page=5"
    )
    assert link["label"] == "Открыть PDF, стр. 5"


def test_citation_pdf_link_ignores_invalid_page_and_encodes_doc_id(
    monkeypatch,
) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)

    for page in (0, -3, True, "abc", ""):
        link = citation_pdf_link({"doc_id": "sber_2023", "page_pdf": page})
        assert link["available"] is True
        assert link["url"] == (
            "http://127.0.0.1:8000/api/documents/sber_2023/file"
        )
        assert link["label"] == "Открыть PDF"

    encoded = citation_pdf_link({
        "doc_id": "sber 2015/отчёт",
        "page_pdf": 3,
    })
    assert encoded["url"] == (
        "http://127.0.0.1:8000/api/documents/"
        "sber%202015%2F%D0%BE%D1%82%D1%87%D1%91%D1%82/file#page=3"
    )


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value) -> None:
        self[name] = value


class _FakeStreamlit:
    """Minimal st stub for headless ``main()`` tests."""

    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.sidebar = self
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def multiselect(self, *args, **kwargs):
        return []

    def button(self, *args, **kwargs):
        return False

    def chat_input(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        self.calls.append("error")

    def title(self, *args, **kwargs):
        self.calls.append("title")

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            self.calls.append(name)
        return noop


def test_main_renders_without_backend() -> None:
    st = _FakeStreamlit()

    main(agent_service=None, service_factory=lambda: None, st_module=st)

    assert "title" in st.calls
    assert "error" in st.calls
    assert st.session_state.messages == []
    assert st.session_state.conversation_id
