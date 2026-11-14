"""Тесты веерного поиска по годам поверх обычного retriever."""

from __future__ import annotations

from typing import Any

from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
)
from src.retrieval.temporal_search import PerYearSearch


class FakeRetriever:
    def __init__(
        self,
        years: set[int],
        *,
        hits_per_year: int = 3,
        table_only: bool = False,
    ) -> None:
        self.years = set(years)
        self.hits_per_year = hits_per_year
        self.table_only = table_only
        self.calls: list[dict[str, Any]] = []
        self.reranker = "fake-reranker"

    def available_years(self) -> set[int]:
        return set(self.years)

    def _chunks(self, year: int, table: bool) -> list[dict[str, Any]]:
        prefix = "table" if table else "text"
        return [
            {
                "chunk_id": f"{prefix}-{year}-{index}",
                "year": year,
                "kind": "table" if table else "text",
                "text": f"{year}",
            }
            for index in range(self.hits_per_year)
        ]

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        self.calls.append(
            {
                "query": query,
                "filters": filters,
                "use_reranker": use_reranker,
            }
        )
        requested = filters.requested_years() or set(self.years)
        hits: list[RetrievalHit] = []
        for year in sorted(requested):
            if year not in self.years:
                continue
            use_table = filters.kinds == {"table"}
            if use_table and self.table_only:
                chunks = self._chunks(year, table=True)
            elif use_table:
                chunks = []
            else:
                chunks = self._chunks(year, table=False)
            hits.extend(
                RetrievalHit(chunk_id=chunk["chunk_id"], chunk=chunk)
                for chunk in chunks
            )
        return RetrievalResult(
            hits=hits,
            missing_years=sorted(set(requested) - self.years),
            effective_filters=filters,
        )


def _ids(result: RetrievalResult) -> list[str]:
    return [hit.chunk_id for hit in result.hits]


def test_fan_out_uses_one_search_per_available_year() -> None:
    retriever = FakeRetriever({2019, 2020, 2022})
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "риски",
        filters=SearchFilters(years={2019, 2020, 2021, 2022}),
    )

    assert _ids(result) == [
        "text-2019-0",
        "text-2019-1",
        "text-2020-0",
        "text-2020-1",
        "text-2022-0",
        "text-2022-1",
    ]
    assert result.missing_years == [2021]
    assert any("2021" in warning for warning in result.warnings)
    assert len(retriever.calls) == 3
    assert [
        call["filters"].requested_years() for call in retriever.calls
    ] == [{2019}, {2020}, {2022}]
    assert all(call["use_reranker"] is False for call in retriever.calls)


def test_single_year_keeps_normal_retrieval() -> None:
    retriever = FakeRetriever({2023})
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "прибыль", filters=SearchFilters(years={2023})
    )

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["use_reranker"] is True
    assert len(result.hits) == 3


def test_no_years_keeps_normal_retrieval() -> None:
    retriever = FakeRetriever({2023})
    engine = PerYearSearch(retriever)

    result = engine.retrieve("прибыль")

    assert len(retriever.calls) == 1
    assert len(result.hits) == 3


def test_metric_question_prefers_tables_per_year() -> None:
    retriever = FakeRetriever({2022, 2023}, table_only=True)
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "чистая прибыль", filters=SearchFilters(years={2022, 2023})
    )

    assert _ids(result) == [
        "table-2022-0",
        "table-2022-1",
        "table-2023-0",
        "table-2023-1",
    ]
    assert all(
        call["filters"].kinds == {"table"} for call in retriever.calls
    )


def test_metric_question_falls_back_to_text_when_tables_are_empty() -> None:
    retriever = FakeRetriever({2022, 2023}, table_only=False)
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "чистая прибыль", filters=SearchFilters(years={2022, 2023})
    )

    assert _ids(result) == [
        "text-2022-0",
        "text-2022-1",
        "text-2023-0",
        "text-2023-1",
    ]
    assert len(retriever.calls) == 4


def test_long_period_drops_to_one_hit_per_year() -> None:
    years = set(range(2010, 2017))
    retriever = FakeRetriever(years)
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "риски", filters=SearchFilters(years=years)
    )

    assert len(result.hits) == 7
    assert len(retriever.calls) == 7


def test_period_is_capped_to_latest_available_years() -> None:
    years = set(range(2001, 2016))
    retriever = FakeRetriever(years)
    engine = PerYearSearch(retriever, max_years=10)

    result = engine.retrieve(
        "риски", filters=SearchFilters(years=years)
    )

    called_years = [
        next(iter(call["filters"].requested_years()))
        for call in retriever.calls
    ]
    assert called_years == list(range(2006, 2016))
    assert any("truncated" in warning for warning in result.warnings)


def test_attributes_are_delegated_to_wrapped_retriever() -> None:
    retriever = FakeRetriever({2023})
    engine = PerYearSearch(retriever)

    assert engine.reranker == "fake-reranker"


def test_fan_out_works_for_a_non_modern_corpus() -> None:
    years = {2001, 2002, 2003}
    retriever = FakeRetriever(years)
    engine = PerYearSearch(retriever, per_year_k=1)

    result = engine.retrieve(
        "прибыль", filters=SearchFilters(years=years)
    )

    assert _ids(result) == ["text-2001-0", "text-2002-0", "text-2003-0"]


def test_missing_year_outside_available_is_reported() -> None:
    retriever = FakeRetriever({2023})
    engine = PerYearSearch(retriever)

    result = engine.retrieve(
        "риски", filters=SearchFilters(years={2018, 2021, 2023})
    )

    assert result.missing_years == [2018, 2021]
    assert len(retriever.calls) == 1
    assert len(result.hits) == 3
