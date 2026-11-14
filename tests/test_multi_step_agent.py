"""Интеграция многошагового поиска и детерминированных расчётов."""

from __future__ import annotations

from typing import Any

from src.agent.selector import PlanAwareSelector
from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation
from src.agent.tools import AgentTools


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
    assert len(result.tool_events) == 4
    assert "missing_year" in result.warnings
    assert "incomplete_period" in result.warnings
    assert result.calculations
    assert "Расчёт по подтверждённым фактам" in result.answer.answer
    assert result.calculations[0]["from_year"] == 2020
    assert result.calculations[0]["to_year"] == 2024
