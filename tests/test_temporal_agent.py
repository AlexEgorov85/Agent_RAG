"""Интеграция роутера периода и веерного поиска в AgentService."""

from __future__ import annotations

from typing import Any

from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation, QueryPlan
from src.agent.tools import AgentTools
from src.agent.agent_graph import _clarification_prompt
from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
)
from src.retrieval.temporal_search import PerYearSearch


class YearRetriever:
    def __init__(self, years: set[int], *, per_year: int = 2) -> None:
        self.years = set(years)
        self.per_year = per_year
        self.calls: list[SearchFilters] = []

    def available_years(self) -> set[int]:
        return set(self.years)

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        self.calls.append(filters)
        requested = filters.requested_years()
        hits: list[RetrievalHit] = []
        for year in sorted(requested):
            if year not in self.years:
                continue
            for index in range(self.per_year):
                chunk = {
                    "chunk_id": f"c{year}-{index}",
                    "year": year,
                    "kind": "table",
                    "text": f"Прибыль за {year} год: {year}.",
                }
                hits.append(
                    RetrievalHit(chunk_id=chunk["chunk_id"], chunk=chunk)
                )
        return RetrievalResult(
            hits=hits,
            missing_years=sorted(set(requested) - self.years),
            effective_filters=filters,
        )


class FirstCitationAnswerer:
    def generate(
        self,
        question: str,
        evidence: Any,
        evidence_manifest: set[str],
        **kwargs: Any,
    ) -> Answer:
        chunk_id = sorted(evidence_manifest)[0]
        return Answer(
            "answer",
            "Динамика подтверждена источниками.",
            citations=[Citation(chunk_id=chunk_id)],
        )


def test_temporal_question_fans_out_and_reports_missing_year() -> None:
    years = {2019, 2020, 2022, 2023, 2024}
    retriever = YearRetriever(years)
    service = AgentService(
        AgentTools(search_backend=PerYearSearch(retriever, per_year_k=2)),
        answerer=FirstCitationAnswerer(),
        router=SimpleRouter(years),
    )

    result = service.ask(
        "Как изменилась чистая прибыль за последние 5 лет?",
        conversation_id="temporal",
    )

    assert result.answer.status == "answer"
    assert result.effective_filters.years == {2020, 2021, 2022, 2023, 2024}
    # PerYearSearch объединяет 4 доступных года × per_year_k=2, но по контракту
    # search_reports обязан вернуть не больше top_k=5.
    assert result.tool_events[0].result_count == 5
    assert "missing_year" in result.warnings
    assert "incomplete_period" in result.warnings
    assert all(
        citation.chunk_id in {"c2020-0", "c2020-1", "c2022-0", "c2022-1",
                              "c2023-0", "c2023-1", "c2024-0", "c2024-1"}
        for citation in result.answer.citations
    )


class FanOutRetriever:
    """Retriever с полным контрактом поиска: запоминает каждый вызов."""

    def __init__(self, years: set[int], *, per_year: int = 3) -> None:
        self.years = set(years)
        self.per_year = per_year
        self.calls: list[dict[str, Any]] = []

    def available_years(self) -> set[int]:
        return set(self.years)

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: list[str] | None = None,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        self.calls.append(
            {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "mode": retrieval_mode,
                "queries": list(queries or []),
                "use_reranker": use_reranker,
            }
        )
        requested = sorted(filters.requested_years())
        hits = [
            RetrievalHit(
                chunk_id=f"c{year}-{index}",
                chunk={
                    "chunk_id": f"c{year}-{index}",
                    "year": year,
                    "kind": "table",
                    "text": f"Прибыль за {year} год: {year}.",
                },
            )
            for year in requested
            if year in self.years
            for index in range(self.per_year)
        ]
        return RetrievalResult(
            hits=hits,
            missing_years=sorted(set(requested) - self.years),
            effective_filters=filters,
        )


def test_per_year_search_fans_out_and_keeps_top_k_and_missing_2021() -> None:
    retriever = FanOutRetriever({2019, 2020, 2022, 2023, 2024})
    search = PerYearSearch(retriever, per_year_k=2, max_years=6)

    result = search.retrieve(
        "Как изменилась чистая прибыль?",
        filters=SearchFilters(years={2019, 2020, 2021, 2022, 2023, 2024}),
        top_k=5,
        retrieval_mode="hybrid",
        queries=["перефразированный запрос"],
    )

    # 2021 отсутствует в корпусе: отдельного поиска по нему нет.
    assert len(retriever.calls) == 5
    assert all(
        call["filters"].requested_years() in ({2019}, {2020}, {2022}, {2023}, {2024})
        for call in retriever.calls
    )
    assert all(call["top_k"] == 5 for call in retriever.calls)
    assert all(call["mode"] == "hybrid" for call in retriever.calls)
    assert all(
        call["queries"] == ["перефразированный запрос"]
        for call in retriever.calls
    )
    assert len(result.hits) == 5
    assert result.missing_years == [2021]
    years = [hit.chunk["year"] for hit in result.hits]
    assert 2021 not in years
    assert set(years).issubset({2019, 2020, 2022, 2023, 2024})


def test_factoid_question_is_not_fanned_out() -> None:
    retriever = FanOutRetriever({2023})
    service = AgentService(
        AgentTools(search_backend=PerYearSearch(retriever, per_year_k=2)),
        answerer=FirstCitationAnswerer(),
        router=SimpleRouter({2023}),
    )

    result = service.ask(
        "Какая чистая прибыль за 2023 год?",
        conversation_id="factoid",
    )

    assert result.answer.status == "answer"
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["filters"].requested_years() == {2023}
    assert retriever.calls[0]["mode"] == "hybrid"
    assert result.answer.citations


class TrendRouter:
    """Temporal trend-план с метрикой и без периода."""

    def analyze(self, question: str, filters: Any) -> QueryPlan:
        del question, filters
        return QueryPlan(
            intent="temporal",
            years=[2022, 2023],
            answer_mode="comparison",
            required_metrics=["net_profit"],
            requires_comparison=True,
        )


class CountingSearch:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, *, filters: Any, top_k: int) -> dict[str, Any]:
        del query, filters, top_k
        self.calls += 1
        return {"hits": [], "warnings": []}


def test_missing_period_requests_clarification_without_search() -> None:
    search = CountingSearch()
    service = AgentService(
        AgentTools(search_backend=search),
        answerer=FirstCitationAnswerer(),
        router=TrendRouter(),
    )

    result = service.ask(
        "Как менялась чистая прибыль Сбербанка?",
        conversation_id="clarify",
    )

    assert result.answer.status == "abstain"
    assert result.answer.citations == []
    assert "clarification_required" in result.answer.warnings
    assert "период" in result.answer.answer.casefold()
    assert search.calls == 0
    assert result.tool_events == []


def test_explicit_period_in_question_is_not_clarified() -> None:
    search = CountingSearch()
    service = AgentService(
        AgentTools(search_backend=search),
        answerer=FirstCitationAnswerer(),
        router=TrendRouter(),
    )

    result = service.ask(
        "Как менялась чистая прибыль Сбербанка в 2023 году?",
        conversation_id="clarify-explicit",
    )

    assert _clarification_prompt(
        {
            "question": "Как менялась чистая прибыль Сбербанка в 2023 году?",
            "query_plan": TrendRouter().analyze("", None),
            "filters": SearchFilters(),
        }
    ) is None
    # Поиск выполняется, а clarify не подменяет ответ.
    assert "clarification_required" not in result.answer.warnings


def test_period_from_history_is_not_clarified() -> None:
    prompt = _clarification_prompt(
        {
            "question": "Как менялась чистая прибыль?",
            "query_plan": TrendRouter().analyze("", None),
            "filters": SearchFilters(),
            "history": [
                {"role": "user", "content": "Какая чистая прибыль за 2023 год?"},
                {"role": "assistant", "content": "1 508,6 млрд руб."},
            ],
        }
    )

    assert prompt is None


def test_absent_year_abstain_names_the_missing_report() -> None:
    years = {2019, 2020, 2022}
    retriever = YearRetriever(years)
    service = AgentService(
        AgentTools(search_backend=PerYearSearch(retriever, per_year_k=2)),
        router=SimpleRouter(years),
        documents_by_year={year: [f"doc{year}"] for year in years},
    )

    result = service.ask(
        "Какая чистая прибыль за 2021 год?",
        conversation_id="absent-year",
    )

    assert result.answer.status == "abstain"
    assert "2021" in result.answer.answer
    assert "отсутствует" in result.answer.answer.casefold()
    assert "no_evidence_for_period" in result.answer.warnings


def test_missing_metric_in_existing_report_is_named() -> None:
    retriever = YearRetriever({2023}, per_year=0)
    service = AgentService(
        AgentTools(search_backend=PerYearSearch(retriever, per_year_k=2)),
        router=SimpleRouter({2023}),
        documents_by_year={2023: ["doc2023"]},
    )

    result = service.ask(
        "Какая чистая прибыль за 2023 год?",
        conversation_id="missing-metric",
    )

    assert result.answer.status == "abstain"
    assert "не найден показатель" in result.answer.answer.casefold()
    assert "2023" in result.answer.answer
    assert "missing_metric" in result.answer.warnings
    assert "no_evidence_for_period" not in result.answer.warnings
