"""Тесты безопасных лимитов RAGSettings, бюджеты селектора и структурные логи."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import ValidationError

from src.agent.agent_graph import _after_assess_decision
from src.agent.selector import PlanAwareSelector
from src.agent.service import log_run_summary, safe_log_fields
from src.agent.state import AgentRunResult, Answer, Citation, ToolEvent
from src.core.rag_config import RAGSettings
from src.retrieval.hybrid_retriever import SearchFilters

LIMIT_ENV = (
    "MAX_TOOL_CALLS",
    "MAX_SEARCH_CALLS",
    "PER_YEAR_K",
    "MAX_YEARS",
    "MAX_STRUCTURE_DOCS",
    "LLM_ROUTER_ENABLED",
    "LLM_REWRITE_ENABLED",
    "LLM_STRUCTURE_VERIFY",
    "RERANK_ENABLED",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LIMIT_ENV:
        monkeypatch.delenv(name, raising=False)


def _settings() -> RAGSettings:
    # ``_env_file=None`` изолирует тест от реального .env проекта.
    return RAGSettings(_env_file=None)


def test_rag_settings_exposes_safe_defaults(clean_env: None) -> None:
    settings = _settings()

    assert settings.max_tool_calls == 4
    assert settings.max_search_calls == 2
    assert settings.per_year_k == 2
    assert settings.max_years == 12
    assert settings.max_structure_docs == 5
    assert settings.max_queries == 3
    assert settings.diversity_enabled is True
    assert settings.max_per_doc == 2
    assert settings.max_per_section == 1
    assert settings.coverage_enabled is True
    assert settings.max_coverage_searches == 1
    assert settings.verification_enabled is True
    assert settings.llm_router_enabled is True
    assert settings.llm_rewrite_enabled is True
    assert settings.llm_structure_verify is True
    assert settings.rerank_enabled is True


def test_rag_settings_reads_limit_env_overrides(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("MAX_SEARCH_CALLS", "2")
    monkeypatch.setenv("PER_YEAR_K", "1")
    monkeypatch.setenv("MAX_YEARS", "4")
    monkeypatch.setenv("MAX_STRUCTURE_DOCS", "2")
    monkeypatch.setenv("MAX_QUERIES", "2")
    monkeypatch.setenv("DIVERSITY_ENABLED", "0")
    monkeypatch.setenv("MAX_PER_DOC", "1")
    monkeypatch.setenv("MAX_PER_SECTION", "2")
    monkeypatch.setenv("COVERAGE_ENABLED", "0")
    monkeypatch.setenv("MAX_COVERAGE_SEARCHES", "3")
    monkeypatch.setenv("VERIFICATION_ENABLED", "0")
    monkeypatch.setenv("LLM_ROUTER_ENABLED", "0")
    monkeypatch.setenv("LLM_REWRITE_ENABLED", "0")
    monkeypatch.setenv("LLM_STRUCTURE_VERIFY", "0")
    monkeypatch.setenv("RERANK_ENABLED", "0")

    settings = _settings()

    assert settings.max_tool_calls == 3
    assert settings.max_search_calls == 2
    assert settings.per_year_k == 1
    assert settings.max_years == 4
    assert settings.max_structure_docs == 2
    assert settings.max_queries == 2
    assert settings.diversity_enabled is False
    assert settings.max_per_doc == 1
    assert settings.max_per_section == 2
    assert settings.coverage_enabled is False
    assert settings.max_coverage_searches == 3
    assert settings.verification_enabled is False
    assert settings.llm_router_enabled is False
    assert settings.llm_rewrite_enabled is False
    assert settings.llm_structure_verify is False
    assert settings.rerank_enabled is False


def test_rag_settings_accepts_truthy_flag_spellings(clean_env: None) -> None:
    settings = RAGSettings(
        _env_file=None,
        llm_router_enabled="yes",  # type: ignore[arg-type]
        rerank_enabled="1",  # type: ignore[arg-type]
    )

    assert settings.llm_router_enabled is True
    assert settings.rerank_enabled is True


def test_rag_settings_rejects_non_positive_budgets(clean_env: None) -> None:
    with pytest.raises(ValidationError):
        RAGSettings(_env_file=None, max_tool_calls=0)
    with pytest.raises(ValidationError):
        RAGSettings(_env_file=None, max_search_calls=0)


def test_temporal_search_limits_are_ready_for_constructor(clean_env: None) -> None:
    settings = RAGSettings(_env_file=None, per_year_k=3, max_years=6)

    assert settings.temporal_search_limits() == {"per_year_k": 3, "max_years": 6}


def _selector_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "query_plan": None,
        "effective_filters": SearchFilters(),
        "question": "Какая выручка?",
        "search_attempts": [{"kind": "broad"}],
        "search_calls_used": 1,
        "expected_years": [2016, 2017],
        "covered_years": [2016],
        "available_documents": {},
        "structure_snapshots": {},
        "structure_comparison_complete": False,
        "structural_changes": [],
    }
    base.update(overrides)
    return base


def test_selector_respects_custom_max_searches() -> None:
    state = _selector_state(search_calls_used=1)

    assert PlanAwareSelector(max_searches=1).choose(state) is None
    assert PlanAwareSelector(max_searches=2).choose(state) is not None


def test_selector_respects_custom_max_structure_docs() -> None:
    from src.agent.state import QueryPlan

    state = _selector_state(
        query_plan=QueryPlan(intent="structural", years=[2016, 2017]),
        expected_years=[2016, 2017],
        available_documents={2016: ["doc2016"], 2017: ["doc2017"]},
    )

    first = PlanAwareSelector(max_structure_docs=1).choose(state)
    assert first is not None
    assert first.arguments["doc_id"] == "doc2017"


def test_after_assess_decision_uses_custom_budgets() -> None:
    from types import SimpleNamespace

    coverage = SimpleNamespace(
        status="partial",
        missing_cells=lambda: [
            SimpleNamespace(status="not_found", year=2022, metric="")
        ],
        uncovered_years=[],
    )
    state = {
        "coverage": coverage,
        "query_plan": SimpleNamespace(intent="temporal"),
        "search_calls_used": 1,
        "tool_calls_used": 1,
        "search_attempts": [{"kind": "broad"}],
        "expected_years": [2022],
        "covered_years": [],
    }

    assert _after_assess_decision(state) == "choose_action"
    assert (
        _after_assess_decision(
            state, max_search_calls=1, max_tool_calls=8
        )
        == "generate_answer"
    )
    assert (
        _after_assess_decision(
            state, max_search_calls=5, max_tool_calls=1
        )
        == "generate_answer"
    )


def test_agent_service_stores_and_applies_budgets() -> None:
    from src.agent.service import AgentService
    from src.agent.state import Answer as ServiceAnswer
    from src.agent.tools import AgentTools

    class MultiYear:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query: str, *, filters: Any, top_k: int) -> dict[str, Any]:
            self.calls += 1
            year = sorted(filters.years or {2023})[0]
            return {
                "hits": [
                    {
                        "chunk_id": f"c{year}",
                        "chunk": {
                            "chunk_id": f"c{year}",
                            "year": year,
                            "doc_id": f"doc{year}",
                            "text": "Прибыль.",
                        },
                    }
                ],
                "warnings": [],
            }

    class Answerer:
        def generate(self, question: str, evidence: Any, manifest: set[str]):
            chunk_id = sorted(manifest)[0]
            chunk = evidence[chunk_id]
            return ServiceAnswer(
                "answer",
                "Ответ.",
                [Citation(chunk_id, year=chunk.get("year"))],
            )

    search = MultiYear()
    service = AgentService(
        AgentTools(search_backend=search),
        answerer=Answerer(),
        max_tool_calls=4,
        max_search_calls=1,
    )

    run = service.ask(
        "Сравни выручку за 2016 и 2017 годы?", conversation_id="budget"
    )

    assert service.max_tool_calls == 4
    assert service.max_search_calls == 1
    assert search.calls == 1
    assert sum(
        1 for event in run.tool_events if event.tool_name == "search_reports"
    ) == 1


def test_agent_service_rejects_zero_budget() -> None:
    from src.agent.service import AgentService
    from src.agent.tools import AgentTools

    with pytest.raises(ValueError):
        AgentService(AgentTools(), max_search_calls=0)


def test_safe_log_fields_drops_prompt_chunks_and_secrets() -> None:
    payload = safe_log_fields(
        {
            "request_id": "req-1",
            "event": "agent_run_completed",
            "tool_sequence": ["search_reports:completed"],
            "coverage_status": "partial",
            "duration_ms": 12.5,
            "question": "секретный вопрос пользователя",
            "chunk_text": "полный текст чанка",
            "api_key": "sk-abcdefghijklmnop",
            "evidence": {"chunk-1": {"text": "тело evidence"}},
        }
    )

    assert payload == {
        "request_id": "req-1",
        "event": "agent_run_completed",
        "tool_sequence": ["search_reports:completed"],
        "coverage_status": "partial",
        "duration_ms": 12.5,
    }


def test_safe_log_fields_scrubs_tokens_and_caps_lists() -> None:
    payload = safe_log_fields(
        {
            "warnings": [f"w{i}" for i in range(30)],
            "event": "API_KEY=supersecret Bearer abcdef",
        }
    )

    assert len(payload["warnings"]) == 16
    assert "supersecret" not in payload["event"]
    assert "[REDACTED]" in payload["event"]


def test_log_run_summary_writes_only_structural_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.agent.service")
    result = AgentRunResult(
        request_id="req-42",
        conversation_id="acceptance-test",
        answer=Answer(
            "answer",
            "Чистая прибыль 1 508,6 млрд рублей.",
            [Citation("c2023", year=2023)],
        ),
        tool_events=[
            ToolEvent(
                "search_reports",
                {"query": "секретный пользовательский запрос"},
                "completed",
                result_count=3,
                duration_ms=10.0,
            )
        ],
        effective_filters=SearchFilters(years={2023}),
        warnings=["incomplete_period"],
        timings={"total_ms": 42.0},
        status="completed",
        diagnostics={
            "query_plan": {"intent": "factoid"},
            "coverage": {"status": "partial"},
            "verification": {"status": "ok"},
        },
    )

    log_run_summary(result)

    records = [item for item in caplog.records if item.name == "src.agent.service"]
    assert records, "структурная запись должна появиться на INFO"
    payload = records[-1].agent
    assert payload["request_id"] == "req-42"
    assert payload["intent"] == "factoid"
    assert payload["tool_sequence"] == ["search_reports:completed"]
    assert payload["coverage_status"] == "partial"
    assert payload["verification_status"] == "ok"
    assert payload["duration_ms"] == 42.0
    assert payload["search_calls"] == 1
    # Текст ответа/запроса/чанков в лог не попадает.
    logged = repr(payload) + caplog.text
    assert "секретный" not in logged
    assert "1 508,6" not in logged
