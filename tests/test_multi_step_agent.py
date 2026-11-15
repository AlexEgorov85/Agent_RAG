"""Интеграция многошагового поиска и детерминированных расчётов."""

from __future__ import annotations

from typing import Any

from src.agent.agent_graph import MAX_SEARCH_CALLS, _after_assess_decision
from src.agent.coverage import build_coverage
from src.agent.selector import PlanAwareSelector
from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation, QueryPlan
from src.agent.tools import AgentTools
from src.retrieval.hybrid_retriever import SearchFilters


class MultiYearSearch:
    """Широкий поиск отдаёт не все годы, недостающие ищутся адресно."""

    def __init__(self, years: set[int]) -> None:
        self.years = set(years)
        self.calls: list[Any] = []

    def __call__(
        self,
        query: str,
        *,
        filters: Any,
        retrieval_mode: str,
        top_k: int,
    ) -> dict[str, Any]:
        self.calls.append(filters)
        requested = sorted(filters.years or [])
        if len(requested) > 1:
            # Имитируем top-k, который покрыл только два года.
            requested = requested[-2:]
        hits = [
            {
                "chunk_id": f"c{year}",
                "chunk": {
                    "chunk_id": f"c{year}",
                    "year": year,
                    "doc_id": f"doc{year}",
                    "text": f"Прибыль за {year} год: {100 + year % 100}.",
                },
            }
            for year in requested
            if year in self.years
        ]
        missing = sorted(year for year in requested if year not in self.years)
        warnings = (
            ["missing years in corpus: " + ", ".join(map(str, missing))]
            if missing
            else []
        )
        return {"hits": hits, "warnings": warnings}


class FactAnswerer:
    def generate(
        self,
        question: str,
        evidence: Any,
        evidence_manifest: set[str],
        **kwargs: Any,
    ) -> Answer:
        citations = []
        facts = []
        for chunk_id, chunk in evidence.items():
            year = chunk.get("year")
            citations.append(
                Citation(chunk_id=chunk_id, year=year, doc_id=chunk.get("doc_id"))
            )
            facts.append(
                {
                    "metric": "Чистая прибыль",
                    "year": year,
                    "value": 100 + (year or 0) % 100,
                    "unit": "млрд руб.",
                    "chunk_id": chunk_id,
                }
            )
        return Answer(
            "answer",
            "Прибыль подтверждена по найденным годам.",
            citations=citations,
            facts=facts,
        )


class Rewriter:
    def rewrite(self, question: str, **kwargs: Any) -> list[str]:
        return ["перефразированный запрос"]


class QueryBackend:
    def __init__(self) -> None:
        self.queries: list[list[str] | None] = []

    def retrieve(
        self,
        query: str,
        *,
        filters: Any,
        top_k: int,
        retrieval_mode: str = "hybrid",
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        self.queries.append(queries)
        year = next(iter(filters.years or {2023}))
        return {
            "hits": [
                {
                    "chunk_id": f"c{year}",
                    "chunk": {
                        "chunk_id": f"c{year}",
                        "year": year,
                        "doc_id": f"doc{year}",
                        "text": f"Прибыль за {year} год.",
                    },
                }
            ],
            "warnings": [],
        }


def test_query_rewriter_feeds_variants_into_broad_search() -> None:
    backend = QueryBackend()
    service = AgentService(
        AgentTools(search_backend=backend),
        answerer=FactAnswerer(),
        router=SimpleRouter({2022, 2023}),
        action_selector=PlanAwareSelector(),
        query_rewriter=Rewriter(),
    )

    service.ask(
        "Как изменилась прибыль за последние 2 года?",
        conversation_id="rewrite",
    )

    assert ["перефразированный запрос"] in backend.queries


class NeedsGraphRouter:
    def analyze(self, question: str, filters: Any) -> QueryPlan:
        del question, filters
        return QueryPlan(intent="multi_hop", needs_graph=True)


class RecordingSearch:
    """Поиск, который запоминает число вызовов и отдаёт один чанк."""

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.calls = 0

    def retrieve(
        self,
        query: str,
        *,
        filters: Any,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        del query, top_k, retrieval_mode, queries
        self.calls += 1
        return {
            "hits": [
                {
                    "chunk_id": self.chunk_id,
                    "chunk": {
                        "chunk_id": self.chunk_id,
                        "year": 2023,
                        "doc_id": "doc2023",
                        "text": "Прибыль за 2023 год: 123.",
                    },
                }
            ],
            "warnings": [],
        }


def test_graph_plan_routes_broad_search_to_graph_backend() -> None:
    plain = RecordingSearch("plain")
    graph = RecordingSearch("graph")
    service = AgentService(
        AgentTools(search_backend=plain, graph_search_backend=graph),
        answerer=FactAnswerer(),
        router=NeedsGraphRouter(),
        action_selector=PlanAwareSelector(),
    )

    result = service.ask("Какие факторы повлияли на прибыль?", conversation_id="graph")

    assert plain.calls == 0
    assert graph.calls == 1
    assert result.answer.status == "answer"


def test_broad_search_forwards_needs_graph_flag() -> None:
    state = {
        "query_plan": QueryPlan(intent="multi_hop", needs_graph=True),
        "effective_filters": SearchFilters(),
        "question": "Какие факторы повлияли на прибыль?",
        "search_attempts": [],
        "search_calls_used": 0,
    }

    action = PlanAwareSelector().choose(state)

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["use_graph"] is True


def test_comparison_without_metrics_searches_each_year_separately() -> None:
    state = {
        "query_plan": QueryPlan(
            intent="multi_hop",
            years=[2017, 2023],
            answer_mode="comparison",
            requires_comparison=True,
        ),
        "effective_filters": SearchFilters(years={2017, 2023}),
        "expected_years": [2017, 2023],
        "covered_years": [],
        "search_attempts": [],
        "search_calls_used": 0,
        "question": "Что изменилось в отчётах за 2017 и 2023 годы?",
    }

    first = PlanAwareSelector().choose(state)

    assert first is not None
    assert first.tool_name == "search_reports"
    assert first.arguments["filters"].years == {2017}
    # Расширенный top-k: нужный раздел года не должен теряться в diversity.
    assert first.arguments["top_k"] == 10
    # Per-year поиск просит реранкер явно; backend применит его, если умеет.
    assert first.arguments["rerank_per_year"] is True

    state["covered_years"] = [2017]
    state["search_attempts"] = [{"kind": "year", "year": 2017}]
    second = PlanAwareSelector().choose(state)

    assert second is not None
    assert second.arguments["filters"].years == {2023}


def test_agent_accepts_selector_rerank_hint_in_search_schema() -> None:
    class ComparisonRouter:
        def analyze(self, question: str, filters: Any) -> QueryPlan:
            del question, filters
            return QueryPlan(
                intent="multi_hop",
                years=[2017, 2023],
                answer_mode="comparison",
                requires_comparison=True,
            )

    service = AgentService(
        AgentTools(search_backend=MultiYearSearch({2017, 2023})),
        answerer=FactAnswerer(),
        router=ComparisonRouter(),
        action_selector=PlanAwareSelector(),
    )

    result = service.ask(
        "Что изменилось в отчётах за 2017 и 2023 годы?",
        conversation_id="comparison-schema",
    )

    assert result.status == "completed"
    assert "invalid_tool_arguments" not in result.warnings
    assert result.tool_events
    assert result.tool_events[0].status == "completed"


def test_period_fragment_uses_previous_question_topic() -> None:
    state = {
        "query_plan": QueryPlan(
            intent="temporal",
            years=[2023],
            answer_mode="comparison",
        ),
        "effective_filters": SearchFilters(years={2023}),
        "expected_years": [2023],
        "covered_years": [],
        "search_attempts": [],
        "search_calls_used": 0,
        "question": "За 2023 год.",
        "history": [
            {"role": "user", "content": "Как менялась чистая прибыль Сбербанка?"},
            {"role": "assistant", "content": "Уточните, пожалуйста, период."},
        ],
    }

    action = PlanAwareSelector().choose(state)

    assert action is not None
    assert action.arguments["query"] == (
        "Как менялась чистая прибыль Сбербанка? За 2023 год."
    )


def test_targeted_year_search_does_not_use_graph() -> None:
    state = {
        "query_plan": QueryPlan(
            intent="temporal",
            years=[2022, 2023],
            requires_comparison=True,
            needs_graph=True,
        ),
        "effective_filters": SearchFilters(years={2022, 2023}),
        "expected_years": [2022, 2023],
        "covered_years": [2022],
        "search_attempts": [{"kind": "broad"}],
        "search_calls_used": 1,
        "available_documents": {},
        "question": "Как изменилась прибыль?",
    }

    action = PlanAwareSelector().choose(state)

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["use_graph"] is False
    assert action.arguments["filters"].years == {2023}



class BadFactAnswerer(FactAnswerer):
    def generate(self, question: str, evidence: Any, evidence_manifest: set[str], **kwargs: Any) -> Answer:
        answer = super().generate(question, evidence, evidence_manifest, **kwargs)
        facts = [dict(fact) for fact in answer.facts]
        if facts:
            facts[0]["value"] = "999999"
        return Answer(
            answer.status,
            answer.answer,
            citations=answer.citations,
            facts=facts,
        )


def test_verification_failure_abstains_and_is_diagnosed() -> None:
    years = {2022, 2023}
    service = AgentService(
        AgentTools(search_backend=MultiYearSearch(years)),
        answerer=BadFactAnswerer(),
        router=SimpleRouter(years),
        action_selector=PlanAwareSelector(),
    )

    result = service.ask(
        "Какая чистая прибыль за 2023 год?",
        conversation_id="verification",
    )

    assert result.answer.status == "abstain", (
        result.answer,
        result.diagnostics.get("verification"),
        result.warnings,
    )
    assert result.diagnostics["verification"]["status"] == "failed"
    assert any(
        issue["code"] == "fact_value_not_in_source"
        for issue in result.diagnostics["verification"]["issues"]
    )


def test_agent_searches_missing_years_and_calculates_change() -> None:
    years = {2019, 2020, 2022, 2023, 2024}
    search = MultiYearSearch(years)
    service = AgentService(
        AgentTools(search_backend=search),
        answerer=FactAnswerer(),
        router=SimpleRouter(years),
        action_selector=PlanAwareSelector(),
    )

    result = service.ask(
        "Как изменилась чистая прибыль за последние 5 лет?",
        conversation_id="multi-step",
    )

    assert result.answer.status == "answer"
    assert result.effective_filters.years == {2020, 2021, 2022, 2023, 2024}
    # Бюджет 2 поиска: широкий + один адресный по непокрытому году.
    assert len(result.tool_events) == 2
    assert "incomplete_period" in result.warnings
    assert result.calculations
    assert "Расчёт по подтверждённым фактам" in result.answer.answer
    assert result.calculations[0]["from_year"] == 2020
    assert result.calculations[0]["to_year"] == 2024

    attempts = result.diagnostics["search_attempts"]
    metric_year_attempts = [
        (item["metric"], item["year"])
        for item in attempts
        if item.get("kind") == "metric_year"
    ]
    assert metric_year_attempts
    assert len(metric_year_attempts) == len(set(metric_year_attempts))
    assert ("net_profit", 2020) in metric_year_attempts
    assert all(
        item.get("section_path") is None
        for item in attempts
        if item.get("kind") == "metric_year"
    )


def coverage_chunk(year: int, text: str) -> dict:
    return {"year": year, "text": text, "doc_id": f"doc{year}"}


def coverage_report(kind: str):
    if kind == "complete":
        return build_coverage(
            [2020, 2022],
            ["net_profit"],
            {
                "c2020": coverage_chunk(2020, "Чистая прибыль 100."),
                "c2022": coverage_chunk(2022, "Чистая прибыль 120."),
            },
            {2020, 2022},
        )
    if kind == "partial":
        return build_coverage(
            [2020, 2022],
            ["net_profit"],
            {"c2020": coverage_chunk(2020, "Чистая прибыль 100.")},
            {2020, 2022},
        )
    if kind == "insufficient":
        return build_coverage([2020, 2022], ["net_profit"], {}, {2020, 2022})
    return build_coverage([], [], {}, None)


def decision_state(
    report,
    *,
    searches: int = 1,
    attempts: list[dict] | None = None,
) -> dict:
    return {
        "coverage": report,
        "query_plan": QueryPlan(
            intent="temporal", years=[2020, 2022], answer_mode="comparison"
        ),
        "search_calls_used": searches,
        "tool_calls_used": searches,
        "search_attempts": attempts or [{"kind": "broad"}],
        "expected_years": [2020, 2022],
        "covered_years": list(report.covered_years),
    }


def test_complete_coverage_routes_to_generation() -> None:
    assert (
        _after_assess_decision(decision_state(coverage_report("complete")))
        == "generate_answer"
    )
    assert (
        _after_assess_decision(decision_state(coverage_report("not_required")))
        == "generate_answer"
    )


def test_partial_coverage_with_budget_continues_search() -> None:
    assert (
        _after_assess_decision(decision_state(coverage_report("partial")))
        == "choose_action"
    )


def test_partial_coverage_with_exhausted_budget_generates() -> None:
    assert (
        _after_assess_decision(
            decision_state(
                coverage_report("partial"), searches=MAX_SEARCH_CALLS
            )
        )
        == "generate_answer"
    )


def test_partial_coverage_without_actionable_gaps_generates() -> None:
    report = coverage_report("partial")
    assert (
        _after_assess_decision(
            decision_state(
                report,
                searches=2,
                attempts=[
                    {"kind": "broad"},
                    {
                        "kind": "metric_year",
                        "metric": "net_profit",
                        "year": 2022,
                        "years": [2022],
                        "section_path": None,
                    },
                ],
            )
        )
        == "generate_answer"
    )


def test_insufficient_coverage_routes_to_generation() -> None:
    assert (
        _after_assess_decision(decision_state(coverage_report("insufficient")))
        == "generate_answer"
    )


def test_structural_plan_keeps_its_tool_loop() -> None:
    state = decision_state(coverage_report("insufficient"))
    state["query_plan"] = QueryPlan(intent="structural", years=[2022, 2023])

    assert _after_assess_decision(state) == "choose_action"


class PartialMetricRouter:
    """Требует две метрики, хотя evidence закрывает только одну."""

    def analyze(self, question: Any, filters: Any) -> QueryPlan:
        del question, filters
        return QueryPlan(
            intent="factoid",
            years=[2023],
            required_metrics=["net_profit", "roa"],
        )


class ProfitOnlySearch:
    def retrieve(
        self,
        query: str,
        *,
        filters: Any,
        top_k: int,
        retrieval_mode: str = "hybrid",
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        del query, filters, top_k, retrieval_mode, queries
        return {
            "hits": [
                {
                    "chunk_id": "c2023",
                    "chunk": {
                        "chunk_id": "c2023",
                        "year": 2023,
                        "doc_id": "doc2023",
                        "text": "Чистая прибыль за 2023 год: 100.",
                    },
                }
            ],
            "warnings": [],
        }


class CapturingAnswerer:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def generate(
        self,
        question: str,
        evidence: Any,
        evidence_manifest: set[str],
        **kwargs: Any,
    ) -> Answer:
        del question, evidence_manifest
        self.kwargs = kwargs
        citations = [
            Citation(
                chunk_id=chunk_id,
                year=chunk.get("year"),
                doc_id=chunk.get("doc_id"),
            )
            for chunk_id, chunk in evidence.items()
        ]
        return Answer("answer", "Ответ по evidence.", citations=citations)


def test_missing_metric_does_not_mark_present_year_as_absent() -> None:
    answerer = CapturingAnswerer()
    service = AgentService(
        AgentTools(search_backend=ProfitOnlySearch()),
        answerer=answerer,
        router=PartialMetricRouter(),
        action_selector=PlanAwareSelector(),
    )

    run = service.ask(
        "Какая чистая прибыль за 2023 год?",
        conversation_id="partial-metric",
    )

    assert run.answer.status == "answer"
    assert answerer.kwargs is not None
    # 2023 есть в корпусе; незакрытая ROA — это not_found, а не missing_year.
    assert answerer.kwargs["missing_years"] == []
    coverage = answerer.kwargs["coverage"]
    assert coverage.missing_years == []
    assert coverage.uncovered_years == [2023]
