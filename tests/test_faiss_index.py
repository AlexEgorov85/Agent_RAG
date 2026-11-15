"""Тесты постоянного FAISS-индекса (offline, без API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.dense_index import DenseIndex
from src.core.faiss_index import FaissDenseIndex, FaissIndexError

faiss = pytest.importorskip("faiss")


def _chunk(chunk_id: str, **fields: object) -> dict[str, object]:
    return {"chunk_id": chunk_id, **fields}


def _vectors() -> list[list[float]]:
    return [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]]


def test_search_matches_dense_index_and_returns_score() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]
    index = FaissDenseIndex(chunks, _vectors())

    hits = index.search([1.0, 0.0], k=3)

    assert [hit["chunk_id"] for hit in hits] == ["a", "b", "c"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert hits[1]["score"] == pytest.approx(0.8, abs=1e-5)

    dense = DenseIndex(chunks, _vectors())
    assert [hit["chunk_id"] for hit in hits] == [
        hit["chunk_id"] for hit in dense.search([1.0, 0.0], k=3)
    ]


def test_eligibility_is_applied_before_top_k() -> None:
    chunks = [_chunk("a", year=2022), _chunk("b", year=2023), _chunk("c", year=2023)]
    index = FaissDenseIndex(chunks, [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

    hits = index.search([1.0, 0.0], k=1, eligible=lambda chunk: chunk["year"] == 2023)

    assert [hit["chunk_id"] for hit in hits] == ["b"]


def test_iterable_eligibility_and_empty_filter_result() -> None:
    index = FaissDenseIndex(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
    )

    assert [hit["chunk_id"] for hit in index.search([1.0, 0.0], k=2, eligible={1, 2})] == [
        "b",
        "c",
    ]
    assert index.search([1.0, 0.0], k=2, eligible={99}) == []


def test_empty_index_returns_no_hits() -> None:
    assert FaissDenseIndex([], []).search([1.0, 0.0]) == []


def test_query_dimension_is_validated() -> None:
    index = FaissDenseIndex([_chunk("a")], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0, 0.0, 0.0])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    index = FaissDenseIndex(chunks, [[1.0, 0.0], [0.0, 1.0]])

    index.save(
        tmp_path / "idx",
        fingerprint="f" * 64,
        model="fake-embed",
        query_prefix="query:",
    )
    loaded = FaissDenseIndex.load(
        tmp_path / "idx", chunks, fingerprint="f" * 64, model="fake-embed"
    )

    assert loaded is not None
    assert loaded.dimension == 2
    assert [hit["chunk_id"] for hit in loaded.search([0.0, 1.0], k=1)] == ["b"]
    assert (tmp_path / "idx" / "index.faiss").is_file()
    assert (tmp_path / "idx" / "manifest.json").is_file()


def test_load_rejects_other_corpus_fingerprint_and_chunk_order(tmp_path: Path) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    FaissDenseIndex(chunks, [[1.0, 0.0], [0.0, 1.0]]).save(
        tmp_path / "idx", fingerprint="a" * 64, model="fake-embed"
    )

    assert FaissDenseIndex.load(
        tmp_path / "idx", chunks, fingerprint="b" * 64, model="fake-embed"
    ) is None
    assert FaissDenseIndex.load(
        tmp_path / "idx", list(reversed(chunks)), fingerprint="a" * 64, model="fake-embed"
    ) is None
    assert FaissDenseIndex.load(
        tmp_path / "idx", chunks, fingerprint="a" * 64, model="other-model"
    ) is None
    with pytest.raises(FaissIndexError):
        FaissDenseIndex.load(
            tmp_path / "idx",
            chunks,
            fingerprint="b" * 64,
            model="fake-embed",
            strict=True,
        )


def test_load_rejects_corrupted_artifact(tmp_path: Path) -> None:
    chunks = [_chunk("a")]
    FaissDenseIndex(chunks, [[1.0, 0.0]]).save(
        tmp_path / "idx", fingerprint="a" * 64, model="fake-embed"
    )
    (tmp_path / "idx" / "index.faiss").write_bytes(b"not a faiss index")

    assert FaissDenseIndex.load(
        tmp_path / "idx", chunks, fingerprint="a" * 64, model="fake-embed"
    ) is None


def test_load_or_build_persists_once_and_reuses(tmp_path: Path) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    first = FaissDenseIndex.load_or_build(
        tmp_path / "idx",
        chunks,
        vectors,
        fingerprint="c" * 64,
        model="fake-embed",
    )
    assert (tmp_path / "idx" / "index.faiss").is_file()

    second = FaissDenseIndex.load_or_build(
        tmp_path / "idx",
        chunks,
        vectors,
        fingerprint="c" * 64,
        model="fake-embed",
    )
    assert second is not None
    assert [hit["chunk_id"] for hit in first.search([1.0, 0.0], k=1)] == ["a"]
    assert [hit["chunk_id"] for hit in second.search([1.0, 0.0], k=1)] == ["a"]
