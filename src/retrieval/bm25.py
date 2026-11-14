"""A BM25Okapi index for mapping-style chunks.

Chunks use the project contract: ``chunk_id`` and ``text`` are top-level
fields; these fields may also be supplied under a ``metadata`` mapping.
Russian tokenization optionally uses pymorphy3 before BM25Okapi indexing.
"""

from __future__ import annotations

from functools import lru_cache
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rank_bm25 import BM25Okapi

Chunk = Mapping[str, Any]
Eligibility = Callable[[Chunk], bool] | Iterable[str | int] | None

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _surface_tokens(text: str) -> list[str]:
    """Return lowercase Unicode word tokens, normalizing Russian ё to е."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [token.replace("ё", "е") for token in _TOKEN_PATTERN.findall(text.lower())]


@lru_cache(maxsize=1)
def _morph_analyzer() -> Any | None:
    """Create a cached pymorphy3 analyzer when the optional package is present."""
    try:
        import pymorphy3  # type: ignore[import-not-found]

        return pymorphy3.MorphAnalyzer()
    except (ImportError, OSError):
        return None


def tokenize(text: str) -> list[str]:
    """Tokenize text and lemmatize Russian words when pymorphy3 is available.

    If pymorphy3 is missing or cannot analyze an individual token, its
    normalized surface form is retained. This keeps indexing and query
    processing usable without network access or optional NLP dependencies.
    """
    words = _surface_tokens(text)
    analyzer = _morph_analyzer()
    if analyzer is None:
        return words

    normalized: list[str] = []
    for word in words:
        try:
            lemma = analyzer.parse(word)[0].normal_form
            normalized.append(str(lemma).replace("ё", "е").lower() if lemma else word)
        except Exception:
            # A malformed/unrecognized token must not make retrieval fail.
            normalized.append(word)
    return normalized


def lemmatize(text: str) -> list[str]:
    """Return the same Russian-aware normalized token sequence as ``tokenize``."""
    return tokenize(text)


def _field(chunk: Chunk, name: str) -> Any:
    """Read a field from the chunk itself, then its optional metadata mapping."""
    if name in chunk:
        return chunk[name]
    metadata = chunk.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


class BM25Index:
    """In-memory BM25Okapi index over mapping-style chunks.

    The corpus statistics are computed once at construction. Eligibility is
    applied to scored corpus entries at search time, before truncating to
    ``k``; it does not rebuild or renormalize the index.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = list(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self._ids: list[str] = []
        tokenized_corpus: list[list[str]] = []

        for position, chunk in enumerate(self.chunks):
            if not isinstance(chunk, Mapping):
                raise TypeError(f"chunk at position {position} must be a mapping")
            chunk_id = _field(chunk, "chunk_id")
            text = _field(chunk, "text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"chunk at position {position} has no valid chunk_id")
            if not isinstance(text, str):
                raise ValueError(f"chunk {chunk_id!r} has no string text")

            tokens = tokenize(text)
            self._ids.append(chunk_id)
            tokenized_corpus.append(tokens)

        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        eligible: Eligibility = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``k`` positive-score chunks, best first.

        ``eligible`` may be a predicate over chunks or a collection of
        eligible chunk IDs / corpus indices. Result dictionaries are shallow
        copies of source chunks with a numeric ``score`` field added.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not self.chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        eligible_ids: set[str] | None = None
        eligible_indices: set[int] | None = None
        predicate: Callable[[Chunk], bool] | None = None
        if callable(eligible):
            predicate = eligible
        elif eligible is not None:
            values = [eligible] if isinstance(eligible, str) else list(eligible)
            eligible_ids = {value for value in values if isinstance(value, str)}
            eligible_indices = {
                value for value in values if isinstance(value, int) and not isinstance(value, bool)
            }

        # BM25Okapi retains corpus-wide statistics; filter its scores before
        # sorting/truncation so eligibility never causes a top-k shortfall.
        if not self._bm25.avgdl or not any(
            term in self._bm25.idf for term in query_terms
        ):
            return []
        scores = self._bm25.get_scores(query_terms)
        scored: list[tuple[float, str, int]] = []
        for index, chunk in enumerate(self.chunks):
            if predicate is not None and not predicate(chunk):
                continue
            if eligible_ids is not None or eligible_indices is not None:
                if self._ids[index] not in (eligible_ids or set()) and index not in (
                    eligible_indices or set()
                ):
                    continue

            score = float(scores[index])
            if score > 0:
                scored.append((score, self._ids[index], index))

        # Chunk ID ordering is deterministic for equal scores; corpus position
        # breaks ties for any repeated IDs in malformed input.
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            {**dict(self.chunks[index]), "score": score}
            for score, _chunk_id, index in scored[:k]
        ]


def keyword_search(
    query: str,
    chunks: Iterable[Chunk] | BM25Index,
    *,
    k: int = 20,
    eligible: Eligibility = None,
) -> list[dict[str, Any]]:
    """Convenience search API; reuse ``BM25Index`` for repeated queries."""
    index = chunks if isinstance(chunks, BM25Index) else BM25Index(chunks)
    return index.search(query, k=k, eligible=eligible)
