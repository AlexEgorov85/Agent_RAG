"""Кэш эмбеддингов, адресуемый по содержимому, с атомарной записью."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    value = chunk.get("chunk_id")
    if value is None:
        metadata = chunk.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("chunk_id")
    if not isinstance(value, str) or not value:
        raise ValueError("every chunk must have a non-empty chunk_id")
    return value


def corpus_fingerprint(chunks: Sequence[Mapping[str, Any]], model: str,
                       embedding_config: Mapping[str, Any] | None = None,
                       source_hashes: Mapping[str, str] | None = None) -> str:
    """Хеширует содержимое чанков, модель, конфигурацию и хеши источников.

    Порядок чанков не важен. Base64-картинки намеренно исключены: они не влияют
    на эмбеддинги и иначе сбрасывали бы кэш при смене настройки payload.
    """
    ordered = sorted(chunks, key=_chunk_id)
    if source_hashes is None:
        derived: dict[str, str] = {}
        for chunk in ordered:
            doc_id = str(chunk.get("doc_id") or "")
            sha = chunk.get("source_sha256")
            if doc_id and isinstance(sha, str):
                derived.setdefault(doc_id, sha)
        source_hashes = derived
    payload = {
        "model": model,
        "embedding_config": dict(embedding_config or {}),
        "source_hashes": dict(sorted(source_hashes.items())),
        "chunks": [
            {
                "chunk_id": _chunk_id(chunk),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text", ""),
                "source_sha256": chunk.get("source_sha256"),
            }
            for chunk in ordered
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _validate_vectors(vectors: Sequence[Sequence[float]],
                      expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise ValueError("embedding count does not match chunk count")
    result: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("embedding must be a non-empty numeric vector")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("embedding contains a non-numeric value")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding contains a non-finite value")
            values.append(number)
        result.append(values)
    dimensions = {len(vector) for vector in result}
    if len(dimensions) > 1:
        raise ValueError("embedding dimensions differ")
    return result


def _cache_target(cache_path: Path, fingerprint: str) -> Path:
    path = Path(cache_path)
    if path.exists() and path.is_dir():
        return path / f"embeddings_{fingerprint}.json"
    if not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        return path / f"embeddings_{fingerprint}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cached_document_vectors(chunks: Sequence[Mapping[str, Any]],
                            embeddings: Any, cache_path: Path,
                            batch_size: int = 8) -> dict[str, list[float]]:
    """Возвращает ``chunk_id → vector`` из кэша или считает их батчами."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not chunks:
        return {}

    model = str(getattr(embeddings, "model", "unknown"))
    config = {"query_prefix": getattr(embeddings, "query_prefix", "")}
    fingerprint = corpus_fingerprint(chunks, model, embedding_config=config)
    target = _cache_target(Path(cache_path), fingerprint)
    chunk_ids = [_chunk_id(chunk) for chunk in chunks]

    if target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint and data.get("model") == model:
                cached_ids = data.get("chunk_ids")
                vectors = _validate_vectors(data.get("vectors", []), len(chunk_ids))
                if isinstance(cached_ids, list) and sorted(cached_ids) == sorted(chunk_ids):
                    by_id = dict(zip(cached_ids, vectors, strict=True))
                    return {chunk_id: by_id[chunk_id] for chunk_id in chunk_ids}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Повреждённый или устаревший кэш пересобирается; старый файл
            # заменяется только после полностью успешной пересборки.
            pass

    vectors: list[list[float]] = []
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            result = embeddings.embed_documents([str(chunk.get("text", ""))
                                                 for chunk in batch])
            vectors.extend(_validate_vectors(result, len(batch)))
        payload = {"fingerprint": fingerprint, "model": model,
                   "chunk_ids": chunk_ids, "vectors": vectors}
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")
        temporary.replace(target)
        return {chunk_id: vector for chunk_id, vector in
                zip(chunk_ids, vectors, strict=True)}
    finally:
        if temporary.exists():
            temporary.unlink()
