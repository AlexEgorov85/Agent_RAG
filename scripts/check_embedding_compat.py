# -*- coding: utf-8 -*-
"""Сравнить векторы нового провайдера эмбеддингов с уже собранным кэшем.

Полезно при переезде на другой OpenAI-совместимый endpoint той же модели
(например, Cloud.ru → NeuralDeep ``giga-embeddings``): если косинус близок к 1,
кэш можно переиспользовать без полной переиндексации. Скрипт делает только
чтение и несколько запросов к embeddings API.

Запуск::

    python scripts/check_embedding_compat.py
    python scripts/check_embedding_compat.py --samples 16 --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.corpus import load_chunks  # noqa: E402
from src.core.embeddings import ReportEmbeddings, make_embeddings_client  # noqa: E402

DEFAULT_QUERY_PREFIX = "query:"
SAFE_REUSE_COSINE = 0.999
CLOSE_COSINE = 0.99


def _cosine(left: list[float], right: list[float]) -> float:
    """Косинус двух векторов; разные размерности дают 0.0."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _load_cache_vectors(cache_dir: Path) -> tuple[dict[str, list[float]], Path | None]:
    """Взять самый большой кэш ``embeddings_*.json``: id чанка → вектор."""
    best: tuple[int, Path, list[str], list[list[float]]] | None = None
    for path in sorted(cache_dir.glob("embeddings_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ids = payload.get("chunk_ids")
        vectors = payload.get("vectors")
        if not isinstance(ids, list) or not isinstance(vectors, list):
            continue
        size = min(len(ids), len(vectors))
        if size > 0 and (best is None or size > best[0]):
            best = (size, path, ids[:size], vectors[:size])
    if best is None:
        return {}, None
    _, path, ids, vectors = best
    return {str(chunk_id): vector for chunk_id, vector in zip(ids, vectors)}, path


def _pick_samples(
    chunks: list[dict[str, Any]],
    cached: dict[str, list[float]],
    limit: int,
) -> list[dict[str, Any]]:
    """Равномерно взять до ``limit`` чанков, для которых есть старый вектор."""
    eligible = [
        chunk for chunk in chunks
        if isinstance(chunk.get("chunk_id"), str)
        and chunk["chunk_id"] in cached
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not eligible or limit <= 0:
        return []
    step = max(1, len(eligible) // limit)
    return eligible[::step][:limit]


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    parser = argparse.ArgumentParser(
        description="Проверить совместимость векторов нового эмбеддера с кэшем.")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT / "data" / "dev" / "chunks")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT / "data" / "dev" / "cache")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--query-prefix",
                        default=os.getenv("EMBED_QUERY_PREFIX",
                                          DEFAULT_QUERY_PREFIX))
    args = parser.parse_args(argv)

    cached, cache_path = _load_cache_vectors(args.cache_dir)
    if not cached or cache_path is None:
        print(f"Не найден кэш embeddings_*.json в {args.cache_dir}",
              file=sys.stderr)
        return 1

    chunks = load_chunks(args.chunks_dir)
    samples = _pick_samples(chunks, cached, args.samples)
    if not samples:
        print("В кэше нет чанков из выбранного корпуса — сравнивать нечего.",
              file=sys.stderr)
        return 1

    model = os.getenv("EMBED_MODEL", "ai-sage/Giga-Embeddings-instruct-3B")
    try:
        client = make_embeddings_client()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    embeddings = ReportEmbeddings(client, model, args.query_prefix)

    print(f"Модель: {model} | кэш: {cache_path.name}")
    print(f"Чанков в кэше: {len(cached)} | сэмпл: {len(samples)}")
    try:
        fresh = embeddings.embed_documents(
            [str(chunk["text"]) for chunk in samples])
    except Exception as exc:  # noqa: BLE001 - без секретов в сообщении
        print(f"Ошибка embeddings API: {type(exc).__name__}", file=sys.stderr)
        return 1

    scores = [
        _cosine(vector, cached[str(chunk["chunk_id"])])
        for chunk, vector in zip(samples, fresh)
    ]
    for chunk, score, vector in zip(samples, scores, fresh):
        print(f"{chunk['chunk_id']}: cosine={score:.6f} dim={len(vector)}")
    average = sum(scores) / len(scores)
    print(f"\nСредний cosine: {average:.6f} | min: {min(scores):.6f}")

    if average >= SAFE_REUSE_COSINE and min(scores) >= CLOSE_COSINE:
        print("Вердикт: векторы совместимы — кэш можно переиспользовать.")
    elif average >= CLOSE_COSINE:
        print("Вердикт: векторы близки; перед переиспользованием прогоните eval.")
    else:
        print("Вердикт: векторы различаются — нужна переиндексация.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
