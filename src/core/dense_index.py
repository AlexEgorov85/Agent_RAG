"""Косинусный индекс в памяти по чанкам-словарям (офлайн, детерминированно)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Chunk = Mapping[str, Any]
Eligibility = Callable[[Chunk], bool] | Iterable[int] | None


def _vector(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("vector must contain finite numbers")
    return vector


def _normalize(values: Sequence[float]) -> list[float]:
    vector = _vector(values)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _chunk_id(chunk: Chunk) -> str:
    value = chunk.get("chunk_id")
    if value is None:
        metadata = chunk.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("chunk_id")
    if not isinstance(value, str) or not value:
        raise ValueError("every chunk must have a non-empty chunk_id")
    return value


class DenseIndex:
    """Косинусный индекс; фильтр применимости применяется до обрезки до top-k."""

    def __init__(self, chunks: Sequence[Chunk],
                 vectors: Sequence[Sequence[float]] | Mapping[str, Sequence[float]]):
        self.chunks: list[Chunk] = list(chunks)
        if isinstance(vectors, Mapping):
            if len(vectors) != len(self.chunks):
                raise ValueError("vector count must match chunk count")
            try:
                ordered = [vectors[_chunk_id(chunk)] for chunk in self.chunks]
            except KeyError as exc:
                raise ValueError("vector count must match chunk count") from exc
        else:
            ordered = list(vectors)
            if len(ordered) != len(self.chunks):
                raise ValueError("vector count must match chunk count")

        self.vectors: list[list[float]] = []
        for values in ordered:
            vector = _normalize(values)
            if self.vectors and len(vector) != len(self.vectors[0]):
                raise ValueError("vector dimension mismatch")
            self.vectors.append(vector)
        self.dimension = len(self.vectors[0]) if self.vectors else 0

    def _eligible_indices(self, eligible: Eligibility) -> list[int]:
        if eligible is None:
            return list(range(len(self.chunks)))
        if callable(eligible):
            return [index for index, chunk in enumerate(self.chunks)
                    if eligible(chunk)]
        allowed = {int(index) for index in eligible}
        return [index for index in allowed
                if 0 <= index < len(self.chunks)]

    def search(self, query_vector: Sequence[float], k: int = 5,
               eligible: Eligibility = None) -> list[dict[str, Any]]:
        """Возвращает до ``k`` чанков, упорядоченных по косинусному сходству."""
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks:
            return []
        query = _normalize(query_vector)
        if len(query) != self.dimension:
            raise ValueError("query dimension does not match index dimension")

        scored: list[tuple[float, str, int]] = []
        for index in self._eligible_indices(eligible):
            vector = self.vectors[index]
            score = sum(left * right for left, right in zip(query, vector))
            scored.append((float(score), _chunk_id(self.chunks[index]), index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**dict(self.chunks[index]), "score": score}
                for score, _chunk_id_value, index in scored[:k]]
