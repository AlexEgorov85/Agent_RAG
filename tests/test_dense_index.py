"""Офлайн-тесты плотного косинусного индекса."""

from __future__ import annotations

import pytest

from src.core.dense_index import DenseIndex


def _chunk(chunk_id: str, **fields: object) -> dict[str, object]:
    return {"chunk_id": chunk_id, **fields}


def test_cosine_ordering_normalization_and_score_field() -> None:
    chunks = [_chunk("opposite"), _chunk("diagonal"), _chunk("best")]
    index = DenseIndex(chunks, [[-10, 0], [1, 1], [2, 0]])

    hits = index.search([3, 0], k=3)

    assert [hit["chunk_id"] for hit in hits] == ["best", "diagonal", "opposite"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[1]["score"] == pytest.approx(2**-0.5)


def test_eligibility_is_applied_before_top_k_and_reads_metadata() -> None:
    chunks = [
        _chunk("excluded", metadata={"year": 2022}),
        _chunk("eligible", metadata={"year": 2023}),
    ]
    index = DenseIndex(chunks, [[1, 0], [0.8, 0.6]])

    hits = index.search([1, 0], k=1, eligible=lambda chunk: chunk["metadata"]["year"] == 2023)

    assert [hit["chunk_id"] for hit in hits] == ["eligible"]
    assert hits[0]["score"] == pytest.approx(0.8)


def test_ties_break_by_chunk_id_independent_of_input_order() -> None:
    index = DenseIndex([_chunk("z"), _chunk("a")], [[0, 1], [0, 2]])

    assert [hit["chunk_id"] for hit in index.search([0, 1])] == ["a", "z"]


def test_invalid_vector_counts_and_query_dimensions_are_clear() -> None:
    with pytest.raises(ValueError, match="vector count"):
        DenseIndex([_chunk("a"), _chunk("b")], [[1, 0]])
    index = DenseIndex([_chunk("a")], [[1, 0]])
    with pytest.raises(ValueError, match="dimension"):
        index.search([1, 0, 0])


def test_zero_vectors_have_safe_zero_scores_and_deterministic_order() -> None:
    index = DenseIndex([_chunk("z"), _chunk("a")], [[0, 0], [0, 0]])

    hits = index.search([0, 0], k=2)

    assert [hit["chunk_id"] for hit in hits] == ["a", "z"]
    assert [hit["score"] for hit in hits] == [0.0, 0.0]


def test_empty_index_returns_no_hits() -> None:
    assert DenseIndex([], []).search([1, 0]) == []


def test_id_mapping_vectors_and_index_eligibility() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    index = DenseIndex(chunks, {"b": [0, 1], "a": [1, 0]})

    assert [hit["chunk_id"] for hit in index.search([0, 1], eligible={0})] == ["a"]
