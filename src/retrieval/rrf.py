"""Детерминированное reciprocal-rank fusion по чанкам-словарям."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

Chunk = Mapping[str, Any]


def _chunk_id(chunk: Chunk) -> str:
    """Достаёт ID чанка из плоского словаря или из вложенного ``metadata``."""
    value = chunk.get("chunk_id")
    if value is None:
        metadata = chunk.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("chunk_id")
    if not isinstance(value, str) or not value:
        raise ValueError("every RRF candidate must have a non-empty chunk_id")
    return value


def rrf(
    rankings: Iterable[Iterable[Chunk]],
    *,
    k: int = 10,
    constant: int = 60,
) -> list[dict[str, Any]]:
    """Сливает ранжированные списки чанков и возвращает не более ``k`` уникальных.

    Каждый список даёт вклад ``1 / (constant + rank)`` с рангами от 1.
    Повторяющиеся ID учитываются только при первом вхождении в каждом списке.
    При равных итоговых оценках сохраняется порядок первого появления по всем
    входным спискам.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if constant < 0:
        raise ValueError("constant must be non-negative")
    if k == 0:
        return []

    scores: dict[str, float] = {}
    first_chunks: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    next_order = 0

    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = _chunk_id(chunk)
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                first_chunks[chunk_id] = dict(chunk)
                first_seen[chunk_id] = next_order
                next_order += 1
            scores[chunk_id] += 1.0 / (constant + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]))
    return [
        {**first_chunks[chunk_id], "rrf_score": scores[chunk_id]}
        for chunk_id in ordered_ids[:k]
    ]
