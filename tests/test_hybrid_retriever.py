"""Офлайн-юнит-тесты гибридного ретривера BM25 + dense."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from src.retrieval.hybrid_retriever import (
    SECTION_FILTER_FALLBACK,
    SECTION_YEAR_ABSENT,
    HybridRetriever,
    RerankResult,
    SearchFilters,
)
from src.retrieval.reranker import LLMReranker
from src.retrieval.temporal_search import PerYearSearch


def chunk(chunk_id: str, *, score: float, **metadata: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "score": score,
        **metadata,
    }


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return [1.0, 0.0]


class FakeIndex:
    def __init__(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = list(results)
        self.chunks = self.results
        self.error = error
        self.calls: list[tuple[Any, int, Any]] = []

    def search(self, query: Any, *, k: int, eligible: Any) -> list[dict[str, Any]]:
        self.calls.append((query, k, eligible))
        if self.error:
            raise self.error
        # Повторяет поведение продакшн-индексов: фильтрация до обрезки.
        return [dict(item) for item in self.results if eligible(item)][:k]


class FakeReranker:
    def __init__(self, *, fallback: bool = False) -> None:
        self.fallback = fallback
        self.received: list[Mapping[str, Any]] = []

    def rerank(
        self, query: str, hits: Sequence[Mapping[str, Any]]
    ) -> RerankResult:
        self.received = list(hits)
        if self.fallback:
            return RerankResult(
                hits=hits,
                used_fallback=True,
                warning="provider response contained SECRET details",
            )
        return RerankResult(
            hits=[
                {**hit, "rerank_score": float(3 - index)}
                for index, hit in enumerate(hits)
            ],
        )


class FakeOpenAICompatibleClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **kwargs: Any) -> Any:
        prompt = kwargs["messages"][1]["content"]
        payload = json.loads(prompt.splitlines()[-1])
        self.payload = payload
        scores = [
            {
                "chunk_id": candidate["chunk_id"],
                "score": 3 if candidate["chunk_id"] == "relevant" else 0,
            }
            for candidate in payload["candidates"]
        ]
        content = json.dumps({"scores": scores})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ]
        )


def retriever(
    dense: Sequence[Mapping[str, Any]],
    bm25: Sequence[Mapping[str, Any]],
    *,
    embedder: FakeEmbedder | None = None,
    dense_error: Exception | None = None,
    bm25_error: Exception | None = None,
    reranker: FakeReranker | None = None,
) -> tuple[HybridRetriever, FakeIndex, FakeIndex]:
    dense_index = FakeIndex(dense, error=dense_error)
    bm25_index = FakeIndex(bm25, error=bm25_error)
    return (
        HybridRetriever(
            dense_index,
            bm25_index,
            embedder or FakeEmbedder(),
            reranker=reranker,
        ),
        dense_index,
        bm25_index,
    )


def test_filters_are_passed_to_both_channels_before_candidate_limit() -> None:
    dense = [
        chunk("wrong", score=9, year=2022),
        chunk("eligible", score=8, year=2023),
    ]
    bm25 = [
        chunk("wrong-bm25", score=7, year=2022),
        chunk("eligible-bm25", score=6, year=2023),
    ]
    engine, dense_index, bm25_index = retriever(dense, bm25)

    result = engine.retrieve("question", filters=SearchFilters(years={2023}))

    assert [hit.chunk_id for hit in result.hits] == ["eligible", "eligible-bm25"]
    assert dense_index.calls[0][1] == bm25_index.calls[0][1] == 20
    assert all(call[2](dense[1]) for call in (dense_index.calls[0], bm25_index.calls[0]))


def test_fuses_deduplicates_and_preserves_independent_scores_and_ranks() -> None:
    dense = [chunk("shared", score=0.9), chunk("dense-only", score=0.8)]
    bm25 = [chunk("shared", score=12.5), chunk("bm25-only", score=4.0)]
    engine, _, _ = retriever(dense, bm25)

    result = engine.retrieve("question", top_k=5)
    by_id = {hit.chunk_id: hit for hit in result.hits}

    assert len(result.hits) == 3
    assert by_id["shared"].dense_rank == by_id["shared"].bm25_rank == 1
    assert by_id["shared"].dense_score == 0.9
    assert by_id["shared"].bm25_score == 12.5
    assert by_id["shared"].rrf_score == pytest.approx(2 / 61)
    assert by_id["dense-only"].bm25_rank is None
    assert by_id["dense-only"].bm25_score is None
    assert by_id["bm25-only"].dense_rank is None
    assert by_id["bm25-only"].dense_score is None
    assert not hasattr(by_id["shared"], "score")
    assert "score" not in by_id["shared"].chunk


def test_defaults_cap_channel_rerank_pool_and_output() -> None:
    dense = [chunk(f"d-{index}", score=1 - index / 100) for index in range(25)]
    bm25 = [chunk(f"b-{index}", score=100 - index) for index in range(25)]
    reranker = FakeReranker()
    engine, dense_index, bm25_index = retriever(dense, bm25, reranker=reranker)

    result = engine.retrieve("question")

    assert dense_index.calls[0][1] == bm25_index.calls[0][1] == 20
    assert len(reranker.received) == 10
    assert all("chunk_id" in hit and "text" in hit for hit in reranker.received)
    assert all("rrf_score" in hit and "dense_score" in hit for hit in reranker.received)
    assert len(result.hits) == 5
    assert result.trace.fused_candidates == 10
    assert result.trace.rerank_candidates == 10
    assert result.trace.returned == 5
    assert result.hits[0].rerank_score == 3.0


def test_query_variants_are_fused_and_counted_in_trace() -> None:
    docs = [chunk("c1", score=1.0), chunk("c2", score=0.5)]
    engine, _, _ = retriever(docs, docs)

    result = engine.retrieve(
        "original",
        queries=["variant one", "variant two"],
    )

    assert result.trace.query_count == 3
    assert set(result.trace.__dict__) >= {
        "query_count",
        "retrieval_mode",
        "distinct_docs",
        "distinct_sections",
    }
    assert result.trace.retrieval_mode == "hybrid"


def test_diversity_prefers_distinct_sections() -> None:
    dense = [
        chunk("a1", score=3.0, section_path="A"),
        chunk("a2", score=2.5, section_path="A"),
        chunk("b1", score=2.0, section_path="B"),
    ]
    dense_index = FakeIndex(dense)
    bm25_index = FakeIndex(dense)
    engine = HybridRetriever(
        dense_index,
        bm25_index,
        FakeEmbedder(),
        top_k=2,
        diversity=True,
    )

    result = engine.retrieve("question")

    assert [hit.chunk_id for hit in result.hits] == ["a1", "b1"]
    assert result.trace.diversity_removed >= 1


def test_diversity_limits_are_soft_and_do_not_drop_recall() -> None:
    dense = [
        chunk("a1", score=4.0, section_path="A", doc_id="d1"),
        chunk("a2", score=3.5, section_path="A", doc_id="d1"),
        chunk("b1", score=3.0, section_path="B", doc_id="d2"),
        chunk("b2", score=2.5, section_path="B", doc_id="d2"),
        chunk("c1", score=2.0, section_path="C", doc_id="d3"),
    ]
    engine = HybridRetriever(
        FakeIndex(dense),
        FakeIndex(dense),
        FakeEmbedder(),
        top_k=4,
        diversity=True,
        max_per_doc=1,
        max_per_section=1,
    )

    result = engine.retrieve("question")

    # Первые два хита — из разных документов/разделов, затем добор.
    assert [hit.chunk_id for hit in result.hits][:2] == ["a1", "b1"]
    assert len(result.hits) == 4
    assert result.trace.diversity_removed >= 2


def test_retrieval_mode_can_use_only_one_channel() -> None:
    dense = [chunk("dense", score=1.0)]
    bm25 = [chunk("keyword", score=2.0)]
    engine, dense_index, bm25_index = retriever(dense, bm25)

    keyword = engine.retrieve("question", retrieval_mode="keyword")
    dense_only = engine.retrieve("question", retrieval_mode="dense")

    assert [hit.chunk_id for hit in keyword.hits] == ["keyword"]
    assert [hit.chunk_id for hit in dense_only.hits] == ["dense"]
    assert dense_index.calls[-1][1] == 20
    assert bm25_index.calls[-1][1] == 20


def test_invalid_retrieval_mode_is_rejected() -> None:
    engine, _, _ = retriever([], [])

    with pytest.raises(ValueError):
        engine.retrieve("question", retrieval_mode="unknown")


def test_explicit_year_range_reports_missing_years_without_synthesizing_hits() -> None:
    docs = [chunk("y2020", score=1, year=2020), chunk("y2022", score=1, year=2022)]
    engine, _, _ = retriever(docs, docs)

    result = engine.retrieve(
        "question",
        filters=SearchFilters(year_from=2020, year_to=2022),
    )

    assert result.missing_years == [2021]
    assert {hit.chunk_id for hit in result.hits} == {"y2020", "y2022"}
    assert any("2021" in warning for warning in result.warnings)


def test_structure_filters_use_path_boundaries_and_normalized_breadcrumb_segments() -> None:
    docs = [
        chunk("exact", score=3, section_path="Риск", breadcrumbs=["Финансы", "Риск"]),
        chunk(
            "child",
            score=2,
            section_path="Риск/Кредитный риск",
            breadcrumbs=["Финансы", "Риск", "Кредитный риск"],
        ),
        chunk(
            "lookalike",
            score=1,
            section_path="Риск-менеджмент",
            breadcrumbs=["Финансы", "Риск-менеджмент"],
        ),
    ]
    engine, _, _ = retriever(docs, docs)

    result = engine.retrieve(
        "question",
        filters=SearchFilters(
            section_path_prefix=" риск ",
            breadcrumb_contains="ФИНАНСЫ",
            max_section_level=3,
        ),
    )

    assert {hit.chunk_id for hit in result.hits} == {"exact", "child"}


def test_invalid_year_range_fails_clearly() -> None:
    with pytest.raises(ValueError, match="year_from"):
        SearchFilters(year_from=2024, year_to=2020)


@pytest.mark.parametrize("failed_channel", ["dense", "bm25"])
def test_one_channel_failure_degrades_with_warning(failed_channel: str) -> None:
    dense = [chunk("dense-hit", score=0.9)]
    bm25 = [chunk("bm25-hit", score=5)]
    engine, _, _ = retriever(
        dense,
        bm25,
        dense_error=RuntimeError("dense unavailable") if failed_channel == "dense" else None,
        bm25_error=RuntimeError("bm25 unavailable") if failed_channel == "bm25" else None,
    )

    result = engine.retrieve("question")

    assert result.hits
    channel_label = "BM25" if failed_channel == "bm25" else "Dense"
    assert any(f"{channel_label} retrieval failed" in warning for warning in result.warnings)
    assert all("unavailable" not in warning for warning in result.warnings)
    assert getattr(result.trace, f"{failed_channel}_failed") is True
    assert getattr(result.trace, f"{failed_channel}_error_type") == "RuntimeError"


def test_reranker_fallback_is_traced_and_keeps_rrf_order() -> None:
    docs = [chunk("first", score=1), chunk("second", score=0.8)]
    reranker = FakeReranker(fallback=True)
    engine, _, _ = retriever(docs, docs, reranker=reranker)

    result = engine.retrieve("question")

    assert [hit.chunk_id for hit in result.hits] == ["first", "second"]
    assert result.trace.reranker_used_fallback is True
    assert result.trace.reranker_status == "fallback"
    assert any("preserving RRF order" in warning for warning in result.warnings)
    assert all("SECRET" not in warning and "provider response" not in warning for warning in result.warnings)


def test_real_llm_reranker_accepts_adapted_mappings_and_reranks() -> None:
    dense = [
        chunk("less-relevant", score=0.9, text="less relevant content"),
        chunk("relevant", score=0.8, text="answering content"),
    ]
    bm25 = [
        chunk("less-relevant", score=7.0, text="less relevant content"),
        chunk("relevant", score=6.0, text="answering content"),
    ]
    client = FakeOpenAICompatibleClient()
    engine, _, _ = retriever(
        dense,
        bm25,
        reranker=LLMReranker(client=client, model="offline-test"),
    )

    result = engine.retrieve("question")

    assert result.trace.reranker_status == "ok"
    assert result.trace.reranker_used_fallback is False
    assert [hit.chunk_id for hit in result.hits] == ["relevant", "less-relevant"]
    assert result.hits[0].rerank_score == 3
    assert result.hits[1].rerank_score == 0
    assert client.payload is not None
    assert {candidate["chunk_id"] for candidate in client.payload["candidates"]} == {
        "relevant",
        "less-relevant",
    }


def test_reranker_non_candidate_id_is_rejected_and_falls_back_safely() -> None:
    class UnexpectedIdReranker:
        def rerank(
            self, query: str, hits: Sequence[Mapping[str, Any]]
        ) -> RerankResult:
            return RerankResult(
                hits=[{"chunk_id": "not-a-candidate", "rerank_score": 3}]
            )

    docs = [chunk("first", score=1), chunk("second", score=0.8)]
    dense_index, bm25_index = FakeIndex(docs), FakeIndex(docs)
    engine = HybridRetriever(
        dense_index,
        bm25_index,
        FakeEmbedder(),
        reranker=UnexpectedIdReranker(),
    )

    result = engine.retrieve("question")

    assert [hit.chunk_id for hit in result.hits] == ["first", "second"]
    assert result.trace.reranker_used_fallback is True
    assert result.trace.reranker_error_type == "ValueError"
    assert all("not-a-candidate" not in warning for warning in result.warnings)


def test_both_channel_failures_return_no_results_with_warnings() -> None:
    engine, _, _ = retriever(
        [],
        [],
        embedder=FakeEmbedder(error=RuntimeError("embed unavailable")),
        bm25_error=RuntimeError("bm25 unavailable"),
    )

    result = engine.retrieve("question")

    assert result.hits == []
    assert result.trace.dense_failed and result.trace.bm25_failed
    assert len(result.warnings) >= 2


def test_strict_section_hits_do_not_trigger_fallback() -> None:
    docs = [
        chunk("risk", score=3, year=2023, section_path="Риск"),
        chunk("risk-child", score=2, year=2023, section_path="Риск/Кредитный"),
    ]
    engine, dense_index, bm25_index = retriever(docs, docs)

    result = engine.retrieve(
        "вопрос",
        filters=SearchFilters(years={2023}, section_path_prefix="Риск"),
    )

    assert [hit.chunk_id for hit in result.hits] == ["risk", "risk-child"]
    # Один строгий поиск: fallback не выполнялся.
    assert len(dense_index.calls) == len(bm25_index.calls) == 1
    assert SECTION_FILTER_FALLBACK not in result.warnings
    assert result.trace.section_filter_status == "applied"
    assert result.trace.section_filter_strict_hits == 2
    assert result.trace.section_filter_corpus_matches == 2
    assert "section_filter" not in result.trace.fallback


def test_section_mismatch_falls_back_to_title_match_without_hard_filter() -> None:
    docs = [
        chunk(
            "chunk",
            score=1,
            year=2023,
            section_path="Сбер в цифрах и фактах/Ключевые результаты",
        ),
        chunk("unrelated", score=3, year=2023, section_path="Обзор рынка"),
        chunk("wrong-year", score=5, year=2016, section_path="Ключевые результаты"),
    ]
    prefix = "Структура отчёта/Ключевые результаты"
    engine, dense_index, bm25_index = retriever(docs, docs)

    result = engine.retrieve(
        "Как изменилась структура отчёта?",
        filters=SearchFilters(years={2023}, section_path_prefix=prefix),
    )

    # Хиты пошли из поиска без section-фильтра, но год сохранён.
    assert [hit.chunk_id for hit in result.hits] == ["chunk"]
    assert all(hit.chunk["year"] == 2023 for hit in result.hits)
    assert len(dense_index.calls) == len(bm25_index.calls) == 2
    assert SECTION_FILTER_FALLBACK in result.warnings
    assert any(
        "section filter matched no chunks" in warning
        for warning in result.warnings
    )
    # Fallback-хиты не выдаются за точное попадание в раздел.
    assert result.effective_filters.section_path_prefix is None
    assert result.trace.section_filter_status == "fallback_title"
    assert result.trace.section_filter_prefix == prefix
    assert result.trace.section_filter_strict_hits == 0
    assert result.trace.section_filter_corpus_matches == 0
    assert result.trace.section_filter_title == "Ключевые результаты"
    assert result.trace.section_filter_title_hits == 1
    assert result.trace.fallback["section_filter"] is True
    assert result.trace.available["section_filter"] is False
    assert result.trace.unavailable["section_filter"] == "fallback"
    # trace описывает фактическую выдачу, а не отброшенные fallback-хиты.
    assert result.trace.returned == 1
    assert [item["chunk_id"] for item in result.trace.stages["returned"] or []] == [
        "chunk"
    ]
    # Запрос fallback содержит title раздела (последний сегмент пути).
    assert bm25_index.calls[1][0].startswith("Ключевые результаты")


def test_section_fallback_without_title_match_keeps_all_hits() -> None:
    docs = [
        chunk("only", score=1, year=2016, section_path="Обзор рынка"),
    ]
    engine, _, _ = retriever(docs, docs)

    result = engine.retrieve(
        "вопрос",
        filters=SearchFilters(
            years={2016}, section_path_prefix="Портрет Группы/Достижения"
        ),
    )

    assert [hit.chunk_id for hit in result.hits] == ["only"]
    assert SECTION_FILTER_FALLBACK in result.warnings
    assert any(
        "searched without section_path_prefix" in warning
        for warning in result.warnings
    )
    assert result.trace.section_filter_status == "fallback"
    assert result.trace.section_filter_title == "Достижения"
    assert result.trace.section_filter_title_hits == 0


def test_section_fallback_does_not_duplicate_existing_title_in_query() -> None:
    docs = [chunk("c1", score=1, year=2023, section_path="Другое")]
    engine, _, bm25_index = retriever(docs, docs)
    query = "Ключевые результаты: как изменилась структура?"

    engine.retrieve(
        query,
        filters=SearchFilters(years={2023}, section_path_prefix="Отчёт/Ключевые результаты"),
    )

    assert bm25_index.calls[1][0] == query


def test_empty_section_fallback_is_reported_as_fallback_empty() -> None:
    engine, _, _ = retriever([], [])

    result = engine.retrieve(
        "вопрос",
        filters=SearchFilters(section_path_prefix="Несуществующий раздел"),
    )

    assert result.hits == []
    assert SECTION_FILTER_FALLBACK in result.warnings
    assert result.trace.section_filter_status == "fallback_empty"
    assert result.trace.section_filter_strict_hits == 0


def test_section_without_requested_year_does_not_return_other_sections() -> None:
    docs = [
        chunk("risk-2019", score=1, year=2019, section_path="Риск"),
        chunk("other-2023", score=5, year=2023, section_path="Обзор рынка"),
    ]
    engine, dense_index, bm25_index = retriever(docs, docs)

    result = engine.retrieve(
        "вопрос",
        filters=SearchFilters(years={2023}, section_path_prefix="Риск"),
    )

    # Раздел в корпусе есть, но за 2023 год в нём нет чанков: чужие разделы
    # не подставляются, fallback не выполняется.
    assert result.hits == []
    assert SECTION_YEAR_ABSENT in result.warnings
    assert SECTION_FILTER_FALLBACK not in result.warnings
    assert result.missing_years == [2023]
    assert result.effective_filters.section_path_prefix == "Риск"
    assert result.trace.section_filter_status == "year_absent"
    assert result.trace.section_filter_corpus_matches == 1
    assert result.trace.section_filter_prefix == "Риск"
    assert len(dense_index.calls) == len(bm25_index.calls) == 1
    assert any(
        "no chunks for years" in warning for warning in result.warnings
    )


def test_factoid_query_without_section_runs_exactly_one_search() -> None:
    docs = [chunk("c1", score=1, year=2023)]
    engine, dense_index, bm25_index = retriever(docs, docs)

    result = engine.retrieve("Какая чистая прибыль за 2023 год?")

    assert [hit.chunk_id for hit in result.hits] == ["c1"]
    assert len(dense_index.calls) == len(bm25_index.calls) == 1
    assert result.warnings == []
    assert result.trace.section_filter_status == "not_requested"
    assert result.trace.section_filter_prefix is None


def test_per_year_search_keeps_hits_and_warning_after_section_fallback() -> None:
    docs = [
        chunk("c2016", score=1, year=2016, section_path="Достижения"),
        chunk("c2023", score=1, year=2023, section_path="Сбер сегодня"),
    ]
    engine, _, _ = retriever(docs, docs)
    fanout = PerYearSearch(engine, per_year_k=1)

    result = fanout.retrieve(
        "Как изменилась структура отчёта?",
        filters=SearchFilters(
            years={2016, 2023},
            section_path_prefix="Портрет Группы/Достижения",
        ),
    )

    assert {hit.chunk["year"] for hit in result.hits} == {2016, 2023}
    assert SECTION_FILTER_FALLBACK in result.warnings
    assert result.trace.fallback["section_filter"] is True
