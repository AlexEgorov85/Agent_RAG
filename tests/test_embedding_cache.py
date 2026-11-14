"""Offline tests for embedding adaptation and vector cache behavior."""

from __future__ import annotations

import json

import pytest

from src.core.cache import cached_document_vectors, corpus_fingerprint
from src.core.embeddings import CourseEmbeddings


class _EmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        return {
            "data": [
                {"index": index, "embedding": [float(len(text)), float(index + 1)]}
                for index, text in enumerate(inputs)  # type: ignore[union-attr]
            ]
        }


class _Client:
    def __init__(self) -> None:
        self.embeddings = _EmbeddingsEndpoint()


class _FakeEmbedder:
    model = "fake-v1"
    query_prefix = "query:"

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("offline fake failure")
        return [[float(len(text)), 1.0] for text in texts]


def _chunks() -> list[dict[str, object]]:
    return [
        {"chunk_id": "doc:c0", "doc_id": "doc", "text": "first", "source_sha256": "abc"},
        {"chunk_id": "doc:c1", "doc_id": "doc", "text": "second", "source_sha256": "abc"},
        {"chunk_id": "doc:c2", "doc_id": "doc", "text": "third", "source_sha256": "abc"},
    ]


def test_course_embeddings_prefixes_only_queries_and_preserves_batch_order() -> None:
    client = _Client()
    embeddings = CourseEmbeddings(client, "test-model")

    query = embeddings.embed_query("revenue")
    docs = embeddings.embed_documents(["alpha", "beta"])

    assert query == [13.0, 1.0]
    assert docs == [[5.0, 1.0], [4.0, 2.0]]
    assert client.embeddings.calls == [
        {"model": "test-model", "input": ["query:revenue"]},
        {"model": "test-model", "input": ["alpha", "beta"]},
    ]


def test_course_embeddings_reorders_indexed_response_and_rejects_malformed() -> None:
    class ReversedEndpoint:
        def create(self, **_kwargs: object) -> dict[str, object]:
            return {"data": [{"index": 1, "embedding": [2]}, {"index": 0, "embedding": [1]}]}

    adapter = CourseEmbeddings(type("C", (), {"embeddings": ReversedEndpoint()})(), "m")
    assert adapter.embed_documents(["a", "b"]) == [[1.0], [2.0]]

    class BadEndpoint:
        def create(self, **_kwargs: object) -> dict[str, object]:
            return {"data": [{"index": 0, "embedding": ["bad"]}]}

    bad = CourseEmbeddings(type("C", (), {"embeddings": BadEndpoint()})(), "m")
    with pytest.raises(ValueError, match="non-numeric"):
        bad.embed_documents(["a"])


def test_fingerprint_is_order_independent_and_tracks_content_model_and_source() -> None:
    chunks = _chunks()
    first = corpus_fingerprint(chunks, "model-a", embedding_config={"prefix": "query:"})
    second = corpus_fingerprint(list(reversed(chunks)), "model-a", embedding_config={"prefix": "query:"})
    changed = [dict(chunk) for chunk in chunks]
    changed[0]["text"] = "changed"
    with_image_payload = [dict(chunk, image_b64="x" * 100_000) for chunk in chunks]

    assert first == second
    assert first == corpus_fingerprint(
        with_image_payload, "model-a", embedding_config={"prefix": "query:"}
    )
    assert first != corpus_fingerprint(changed, "model-a", embedding_config={"prefix": "query:"})
    assert first != corpus_fingerprint(chunks, "model-b", embedding_config={"prefix": "query:"})
    assert first != corpus_fingerprint(chunks, "model-a", source_hashes={"doc": "new"})


def test_cache_batches_and_valid_hit_avoids_embedding_calls(tmp_path) -> None:
    chunks = _chunks()
    embedder = _FakeEmbedder()

    first = cached_document_vectors(chunks, embedder, tmp_path, batch_size=2)
    assert len(embedder.calls) == 2
    assert embedder.calls == [["first", "second"], ["third"]]
    cache_file = next(tmp_path.glob("embeddings_*.json"))
    assert json.loads(cache_file.read_text(encoding="utf-8"))["chunk_ids"] == [
        "doc:c0",
        "doc:c1",
        "doc:c2",
    ]

    second = cached_document_vectors(chunks, embedder, tmp_path, batch_size=2)
    assert second == first
    assert len(embedder.calls) == 2
    reordered = cached_document_vectors(list(reversed(chunks)), embedder, tmp_path, batch_size=2)
    assert list(reordered) == ["doc:c2", "doc:c1", "doc:c0"]
    assert len(embedder.calls) == 2


def test_failed_build_does_not_replace_existing_cache_file(tmp_path) -> None:
    chunks = _chunks()
    target = tmp_path / "cache.json"
    target.write_text('{"old":"cache"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="fake failure"):
        cached_document_vectors(chunks, _FakeEmbedder(fail_on_call=2), target, batch_size=2)

    assert target.read_text(encoding="utf-8") == '{"old":"cache"}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_non_finite_vectors_are_not_cached(tmp_path) -> None:
    class BadVectors:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float("nan")] for _ in texts]

    target = tmp_path / "cache.json"
    with pytest.raises(ValueError, match="non-finite"):
        cached_document_vectors(_chunks(), BadVectors(), target, batch_size=8)
    assert not target.exists()
