"""Offline, deterministic unit tests for BM25 and RRF core behavior."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.retrieval.bm25 import BM25Index, keyword_search, lemmatize, tokenize
from src.retrieval.rrf import rrf


def _chunk(chunk_id: str, text: str, **metadata: object) -> dict[str, object]:
    return {"chunk_id": chunk_id, "text": text, "metadata": metadata}


def test_russian_tokenization_is_safe_and_case_insensitive() -> None:
    tokens = tokenize("ЁЖИКи, РИСКИ — 2023!")

    assert tokens
    assert all(token == token.lower() for token in tokens)
    assert "2023" in tokens
    assert lemmatize("ЁЖИКи") == tokenize("ЁЖИКи")


def test_bm25_only_returns_positive_scores_and_orders_relevant_chunks() -> None:
    chunks = [
        _chunk("doc:c0", "альфа альфа"),
        _chunk("doc:c1", "альфа бета"),
        _chunk("doc:c2", "гамма"),
    ]

    hits = keyword_search("альфа", chunks, k=20)

    assert [hit["chunk_id"] for hit in hits] == ["doc:c0", "doc:c1"]
    assert all(hit["score"] > 0 for hit in hits)
    assert hits[0]["score"] > hits[1]["score"]


def test_bm25_scores_match_rank_bm25_okapi() -> None:
    chunks = [
        _chunk("doc:c0", "альфа альфа"),
        _chunk("doc:c1", "альфа бета"),
        _chunk("doc:c2", "гамма"),
    ]
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    reference = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    expected_scores = reference.get_scores(tokenize("альфа"))

    hits = BM25Index(chunks).search("альфа", k=20)
    scores_by_id = {hit["chunk_id"]: hit["score"] for hit in hits}

    assert scores_by_id == {
        chunk["chunk_id"]: float(expected_scores[index])
        for index, chunk in enumerate(chunks)
        if expected_scores[index] > 0
    }


def test_bm25_applies_eligibility_before_top_k_without_reindexing() -> None:
    chunks = [
        {"chunk_id": "a", "text": "альфа альфа альфа", "year": 2022},
        {"chunk_id": "b", "text": "альфа", "year": 2023},
        {"chunk_id": "c", "text": "бета гамма"},
    ]
    index = BM25Index(chunks)

    hits = index.search("альфа", k=1, eligible=lambda chunk: chunk.get("year") == 2023)

    assert [hit["chunk_id"] for hit in hits] == ["b"]
    assert hits[0]["score"] > 0


def test_bm25_accepts_eligible_indices_and_nested_chunk_fields() -> None:
    chunks = [
        {"metadata": {"chunk_id": "nested-a", "text": "альфа альфа"}},
        {"metadata": {"chunk_id": "nested-b", "text": "альфа"}},
        {"metadata": {"chunk_id": "nested-c", "text": "бета гамма"}},
    ]

    hits = BM25Index(chunks).search("альфа", k=1, eligible={1})

    assert len(hits) == 1
    assert hits[0]["metadata"]["chunk_id"] == "nested-b"
    assert hits[0]["score"] > 0


def test_rrf_deduplicates_per_chunk_and_preserves_ties_by_first_seen() -> None:
    a = _chunk("a", "A")
    b = _chunk("b", "B")
    rankings = [[a, b, a], [b, a]]

    fused = rrf(rankings, k=10, constant=60)

    assert [item["chunk_id"] for item in fused] == ["a", "b"]
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
    assert fused[0]["rrf_score"] == 1 / 61 + 1 / 62


def test_rrf_default_limit_is_independent_of_constant() -> None:
    ranking = [_chunk(str(index), str(index)) for index in range(25)]

    fused = rrf([ranking])

    assert len(fused) == 10
    assert fused[0]["chunk_id"] == "0"
    assert fused[0]["rrf_score"] == 1 / 61
