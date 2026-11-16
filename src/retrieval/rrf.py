"""Детерминированное reciprocal-rank fusion по чанкам-словарям."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Chunk = Mapping[str, Any]

#: Служебные поля, которые не являются содержимым чанка и сбрасываются при слиянии.
_RESERVED_FIELDS = frozenset(
    {"score", "dense_score", "bm25_score", "rrf_score", "rerank_score", "rerank_rank"}
)


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


def strip_scores(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Копия чанка без служебных score-полей."""
    return {key: value for key, value in chunk.items() if key not in _RESERVED_FIELDS}


def fuse_rankings(
    rankings: Iterable[tuple[str, Sequence[Chunk]]],
    *,
    constant: int = 60,
    channel_names: Sequence[str] = (),
    strip: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Слить именованные списки и вернуть ``(chunk_id, record)`` по убыванию score.

    ``record`` содержит исходный чанк, ``rrf_score`` и, если задан
    ``channel_names``, лучшие ``{channel}_rank``/``{channel}_score`` по каналам.
    ``strip=True`` убирает служебные score-поля из копии чанка.
    """
    if constant < 0:
        raise ValueError("constant must be non-negative")
    fused: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    counter = 0

    for channel, hits in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            chunk_id = _chunk_id(hit)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in fused:
                record: dict[str, Any] = {
                    "chunk": strip_scores(hit) if strip else dict(hit),
                    "rrf_score": 0.0,
                }
                for name in channel_names:
                    record[f"{name}_rank"] = None
                    record[f"{name}_score"] = None
                fused[chunk_id] = record
                first_seen[chunk_id] = counter
                counter += 1
            record = fused[chunk_id]
            if f"{channel}_rank" in record:
                best_rank = record[f"{channel}_rank"]
                if best_rank is None or rank < best_rank:
                    record[f"{channel}_rank"] = rank
                score = float(hit.get("score", 0.0))
                best_score = record[f"{channel}_score"]
                if best_score is None or score > best_score:
                    record[f"{channel}_score"] = score
            record["rrf_score"] += 1.0 / (constant + rank)

    ordered = sorted(
        fused,
        key=lambda chunk_id: (-fused[chunk_id]["rrf_score"], first_seen[chunk_id]),
    )
    return [(chunk_id, fused[chunk_id]) for chunk_id in ordered]


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
    if k == 0:
        return []
    named = ((f"list{index}", list(hits)) for index, hits in enumerate(rankings))
    return [
        {**record["chunk"], "rrf_score": record["rrf_score"]}
        for _chunk_id_value, record in fuse_rankings(named, constant=constant)[:k]
    ]
