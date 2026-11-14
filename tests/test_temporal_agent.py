"""Интеграция роутера периода и веерного поиска в AgentService."""

from __future__ import annotations

from typing import Any

from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation
from src.agent.tools import AgentTools
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
    assert result.tool_events[0].result_count == 8
    assert "missing_year" in result.warnings
    assert "incomplete_period" in result.warnings
    assert all(
        citation.chunk_id in {"c2020-0", "c2020-1", "c2022-0", "c2022-1",
                              "c2023-0", "c2023-1", "c2024-0", "c2024-1"}
        for citation in result.answer.citations
    )
