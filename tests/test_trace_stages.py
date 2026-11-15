"""Тесты этапов retrieval trace и явной доступности каналов.

Проверяют контракт ``spec_benchmark.md`` §4.3: этап либо измерен
(``available=true``, возможно с пустым списком), либо отсутствует
(``stages[name] is None`` и ``available[name] is False``) — но никогда не
подменяется фиктивными хитами.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.engine.context import pack_context
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RerankResult,
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)
from src.retrieval.temporal_search import PerYearSearch


def chunk(chunk_id: str, *, score: float, **metadata: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "score": score,
        **metadata,
    }


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0]


class FakeIndex:
    def __init__(self, results: Sequence[Mapping[str, Any]]) -> None:
        self.results = list(results)
        self.chunks = self.results
        self.calls: list[Any] = []

    def search(self, query: Any, *, k: int, eligible: Any) -> list[dict[str, Any]]:
        self.calls.append(query)
        return [dict(item) for item in self.results if eligible(item)][:k]


class FallbackReranker:
    def rerank(self, query: str, hits: Sequence[Mapping[str, Any]]) -> RerankResult:
        return RerankResult(hits=list(hits), used_fallback=True)


class ReverseReranker:
    def rerank(self, query: str, hits: Sequence[Mapping[str, Any]]) -> RerankResult:
        reversed_hits = list(reversed(list(hits)))
        return RerankResult(
            hits=[
                {**hit, "rerank_score": float(len(reversed_hits) - index)}
                for index, hit in enumerate(reversed_hits)
            ]
        )


def _engine(
    dense: Sequence[Mapping[str, Any]],
    bm25: Sequence[Mapping[str, Any]],
    *,
    reranker: Any = None,
    top_k: int = 5,
) -> tuple[HybridRetriever, FakeIndex, FakeIndex, FakeEmbedder]:
    dense_index, bm25_index = FakeIndex(dense), FakeIndex(bm25)
    embedder = FakeEmbedder()
    engine = HybridRetriever(
        dense_index,
        bm25_index,
        embedder,
        reranker=reranker,
        top_k=top_k,
    )
    return engine, dense_index, bm25_index, embedder


def _stage_ids(trace: RetrievalTrace, name: str) -> list[str]:
    items = trace.stage_items(name)
    assert items is not None, f"stage {name} must be measured"
    return [item["chunk_id"] for item in items]


def test_unavailable_stage_is_none_not_an_empty_list() -> None:
    engine, _, _, _ = _engine([chunk("d1", score=1)], [chunk("b1", score=1)])

    result = engine.retrieve("question", use_reranker=False)

    # Реранкер явно выключен: этап не измерен, а не «0 результатов».
    assert result.trace.stages["rerank"] is None
    assert result.trace.available["rerank"] is False
    assert result.trace.unavailable["rerank"] == "disabled"
    # Измеренный канал с нулём результатов остаётся доступным и пустым.
    empty_engine, _, _, _ = _engine([], [])
    empty = empty_engine.retrieve("question")
    assert empty.trace.stages["dense"] == []
    assert empty.trace.available["dense"] is True


def test_dense_mode_does_not_call_or_fill_bm25_stage() -> None:
    engine, dense_index, bm25_index, _ = _engine(
        [chunk("dense-hit", score=0.9)],
        [chunk("bm25-hit", score=9.0)],
    )

    result = engine.retrieve("question", retrieval_mode="dense")

    assert bm25_index.calls == []
    assert [hit.chunk_id for hit in result.hits] == ["dense-hit"]
    assert result.trace.available["dense"] is True
    assert result.trace.stages["bm25"] is None
    assert result.trace.available["bm25"] is False
    assert result.trace.unavailable["bm25"] == "not_requested"


def test_keyword_mode_does_not_embed_or_fill_dense_stage() -> None:
    engine, dense_index, _, embedder = _engine(
        [chunk("dense-hit", score=0.9)],
        [chunk("bm25-hit", score=9.0)],
    )

    result = engine.retrieve("question", retrieval_mode="keyword")

    assert embedder.queries == []
    assert dense_index.calls == []
    assert [hit.chunk_id for hit in result.hits] == ["bm25-hit"]
    assert result.trace.stages["dense"] is None
    assert result.trace.available["dense"] is False


def test_returned_stage_matches_final_top_k() -> None:
    dense = [chunk(f"d{index}", score=1 - index / 10) for index in range(5)]
    engine, _, _, _ = _engine(dense, dense, top_k=5)

    result = engine.retrieve("question", top_k=2)

    returned = result.trace.stages["returned"]
    assert returned is not None
    assert result.trace.available["returned"] is True
    assert len(returned) == len(result.hits) == 2
    assert [item["chunk_id"] for item in returned] == [
        hit.chunk_id for hit in result.hits
    ]
    assert [item["rank"] for item in returned] == [1, 2]
    assert result.trace.returned == 2
    # returned — финальный top-k, а не весь пул слияния.
    assert len(result.trace.stages["rrf"] or []) > len(returned)


def test_duplicate_chunk_ids_collapse_with_sequential_ranks() -> None:
    dense = [
        chunk("dup", score=1.0),
        chunk("other", score=0.5),
        chunk("dup", score=0.1),
    ]
    engine, _, _, _ = _engine(dense, [])

    result = engine.retrieve("question")

    dense_stage = result.trace.stages["dense"]
    assert dense_stage is not None
    assert [item["chunk_id"] for item in dense_stage] == ["dup", "other"]
    assert [item["rank"] for item in dense_stage] == [1, 2]
    # Score — технический, от первого (лучшего) вхождения.
    assert dense_stage[0]["score"] == 1.0


def test_reranker_fallback_keeps_rrf_order_and_marks_rerank_unavailable() -> None:
    docs = [chunk("first", score=1), chunk("second", score=0.8)]
    engine, _, _, _ = _engine(docs, docs, reranker=FallbackReranker())

    result = engine.retrieve("question")

    rrf_ids = _stage_ids(result.trace, "rrf")
    assert [hit.chunk_id for hit in result.hits] == rrf_ids[:2]
    assert result.trace.fallback["reranker"] is True
    assert result.trace.stages["rerank"] is None
    assert result.trace.available["rerank"] is False
    assert result.trace.unavailable["rerank"] == "fallback"


def test_successful_reranker_publishes_order_and_scores() -> None:
    docs = [chunk("first", score=1), chunk("second", score=0.8)]
    engine, _, _, _ = _engine(docs, docs, reranker=ReverseReranker())

    result = engine.retrieve("question")

    rerank = result.trace.stages["rerank"]
    assert rerank is not None
    assert [item["chunk_id"] for item in rerank] == ["second", "first"]
    assert [item["rank"] for item in rerank] == [1, 2]
    assert [item["score"] for item in rerank] == [2.0, 1.0]
    assert [hit.chunk_id for hit in result.hits] == ["second", "first"]


class StageTracingRetriever:
    """Отдаёт по одному chunk'у на год плюс общий chunk между годами."""

    def __init__(self, years: set[int]) -> None:
        self.years = set(years)
        self.calls: list[dict[str, Any]] = []

    def available_years(self) -> set[int]:
        return set(self.years)

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        use_reranker: bool = True,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        year = next(iter(filters.requested_years()))
        self.calls.append(
            {
                "year": year,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "queries": queries,
            }
        )
        hits = [
            RetrievalHit(
                chunk_id=f"{year}-primary",
                chunk={"chunk_id": f"{year}-primary", "year": year, "doc_id": "doc"},
            ),
            RetrievalHit(
                chunk_id="shared",
                chunk={"chunk_id": "shared", "year": year, "doc_id": "doc"},
            ),
        ]
        trace = RetrievalTrace(
            retrieval_mode=retrieval_mode,
            fused_candidates=2,
            rerank_candidates=2,
            stages={
                "dense": [
                    {"chunk_id": hit.chunk_id, "rank": index, "score": 1.0}
                    for index, hit in enumerate(hits, start=1)
                ],
                "returned": [
                    {"chunk_id": hit.chunk_id, "rank": index, "score": 0.5}
                    for index, hit in enumerate(hits, start=1)
                ],
            },
            available={"dense": True, "returned": True},
            fallback={"reranker": False},
        )
        trace.mark_unavailable("rerank", "disabled")
        return RetrievalResult(
            hits=hits,
            missing_years=sorted(set(filters.requested_years()) - self.years),
            effective_filters=filters,
            trace=trace,
        )


def test_per_year_search_caps_merged_hits_to_top_k() -> None:
    years = {2019, 2020, 2021, 2022, 2023}
    retriever = StageTracingRetriever(years)
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "риски", filters=SearchFilters(years=years), top_k=3
    )

    assert len(result.hits) == 3
    assert result.trace.returned == 3
    assert all(call["top_k"] == 3 for call in retriever.calls)


def test_per_year_search_merges_stages_without_duplicates() -> None:
    retriever = StageTracingRetriever({2020, 2022})
    engine = PerYearSearch(retriever, per_year_k=2)

    result = engine.retrieve(
        "риски", filters=SearchFilters(years={2020, 2021, 2022})
    )

    assert result.missing_years == [2021]
    dense = _stage_ids(result.trace, "dense")
    # Дубли по chunk_id схлопнуты, порядок — по годам, ранги с 1.
    assert dense == ["2020-primary", "shared", "2022-primary"]
    dense_stage = result.trace.stages["dense"] or []
    assert [item["rank"] for item in dense_stage] == [1, 2, 3]
    # Этап, недоступный во всех внутренних вызовах, остаётся недоступным.
    assert result.trace.stages["rerank"] is None
    assert result.trace.available["rerank"] is False
    assert result.trace.returned == len(result.hits) == 3
    assert result.trace.distinct_docs == 1


def test_packed_context_exposes_context_and_skipped_chunk_ids() -> None:
    def full_chunk(chunk_id: str, text: str) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "doc_id": "doc",
            "year": 2023,
            "kind": "text",
            "text": text,
        }

    first, oversized, last = (
        full_chunk("a", "A" * 40),
        full_chunk("b", "B" * 2_000),
        full_chunk("c", "C" * 40),
    )
    budget = len(pack_context([first, last], max_chars=1_000_000).text)

    packed = pack_context([first, oversized, last], max_chars=budget)

    assert packed.context_chunk_ids == ["a", "c"]
    assert packed.context_chunk_ids == packed.chunk_ids
    assert packed.context_chunk_ids == packed.manifest.included_chunk_ids
    assert packed.skipped_chunk_ids == ["b"]
    assert packed.skipped_chunk_ids == packed.manifest.skipped_chunk_ids
    assert packed.context_stage == [
        {"chunk_id": "a", "rank": 1, "score": None},
        {"chunk_id": "c", "rank": 2, "score": None},
    ]
