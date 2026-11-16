# -*- coding: utf-8 -*-
"""Сборка постоянного FAISS-индекса для выбранного корпуса чанков.

Эмбеддинги считаются тем же адаптером и кэшем, что и runtime
(``ReportEmbeddings`` + ``data/cache/embeddings_{fingerprint}.json``).
Готовый артефакт раскладывается как::

    data/indexes/{index_fingerprint}/index.faiss
    data/indexes/{index_fingerprint}/manifest.json

Запуск::

    python scripts/build_indexes.py --dry-run
    python scripts/build_indexes.py
    python scripts/build_indexes.py --chunks-dir data/dev/chunks \
        --cache-dir data/dev/cache --indexes-dir data/dev/indexes
    python scripts/build_indexes.py --reuse-cache-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.cache import (  # noqa: E402
    cached_document_vectors,
    corpus_fingerprint,
)
from src.core.corpus import load_chunks  # noqa: E402
from src.core.embeddings import ReportEmbeddings, make_embeddings_client  # noqa: E402
from src.core.faiss_index import (  # noqa: E402
    FaissDenseIndex,
    FaissIndexError,
)

DEFAULT_QUERY_PREFIX = "query:"


def build_parser() -> argparse.ArgumentParser:
    """CLI сборки FAISS-индекса."""
    parser = argparse.ArgumentParser(
        description="Собрать постоянный FAISS-индекс эмбеддингов чанков.")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT / "data" / "chunks",
                        help="Каталог *_chunks.json (по умолчанию data/chunks)")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT / "data" / "cache",
                        help="Каталог кэша эмбеддингов (по умолчанию data/cache)")
    parser.add_argument("--indexes-dir", type=Path,
                        default=PROJECT / "data" / "indexes",
                        help="Каталог индексов (по умолчанию data/indexes)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Чанков в одном запросе эмбеддингов (по умолчанию 8)")
    parser.add_argument("--model", default=None,
                        help="Модель эмбеддингов (env EMBED_MODEL)")
    parser.add_argument("--query-prefix", default=None,
                        help="Префикс запроса (env EMBED_QUERY_PREFIX)")
    parser.add_argument("--reuse-cache-only", action="store_true",
                        help="Не вызывать API: использовать готовый кэш эмбеддингов")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать объём работы без эмбеддингов/записи")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size должен быть >= 1")


def _cached_vectors_from_file(
    cache_dir: Path, fingerprint: str, *, model: str
) -> dict[str, list[float]] | None:
    """Прочитать готовый кэш без обращения к API."""
    import json

    target = cache_dir / f"embeddings_{fingerprint}.json"
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if payload.get("fingerprint") != fingerprint or payload.get("model") != model:
        return None
    ids = payload.get("chunk_ids")
    vectors = payload.get("vectors")
    if not isinstance(ids, list) or not isinstance(vectors, list) or len(ids) != len(vectors):
        return None
    return dict(zip(ids, vectors, strict=True))


def main(argv: list[str] | None = None) -> int:
    """Точка входа сборки индекса."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        print(f"В {args.chunks_dir} нет чанков.", file=sys.stderr)
        return 1

    model = (args.model or os.getenv("EMBED_MODEL") or "").strip()
    if not model:
        print("Не задан EMBED_MODEL.", file=sys.stderr)
        return 1
    query_prefix = (
        args.query_prefix
        if args.query_prefix is not None
        else os.getenv("EMBED_QUERY_PREFIX", DEFAULT_QUERY_PREFIX)
    )
    fingerprint = corpus_fingerprint(
        chunks, model, embedding_config={"query_prefix": query_prefix}
    )
    target = args.indexes_dir / fingerprint

    print(f"Чанков: {len(chunks)} | документов: "
          f"{len({chunk.get('doc_id') for chunk in chunks})}")
    print(f"Модель: {model} | префикс запроса: {query_prefix!r}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Индекс: {target}")
    if args.dry_run:
        print("[dry-run] эмбеддинги не считались, артефакт не создан.")
        return 0

    vectors = (
        _cached_vectors_from_file(args.cache_dir, fingerprint, model=model)
        if args.reuse_cache_only
        else None
    )
    if vectors is None:
        if args.reuse_cache_only:
            print("Готовый кэш эмбеддингов не найден; API не вызывается.",
                  file=sys.stderr)
            return 1
        try:
            embed_client = make_embeddings_client()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        embeddings = ReportEmbeddings(embed_client, model, query_prefix)
        try:
            vectors = cached_document_vectors(
                chunks, embeddings, args.cache_dir, batch_size=args.batch_size
            )
        except Exception as exc:  # noqa: BLE001 - сообщение без секретов
            print(f"Ошибка вычисления эмбеддингов: {type(exc).__name__}",
                  file=sys.stderr)
            return 1

    try:
        index = FaissDenseIndex(chunks, vectors)
        index.save(
            target,
            fingerprint=fingerprint,
            model=model,
            query_prefix=query_prefix,
        )
    except (FaissIndexError, ValueError, OSError) as exc:
        print(f"Ошибка сборки FAISS-индекса: {exc}", file=sys.stderr)
        return 1

    print(f"Готово: {target / 'index.faiss'}")
    print(f"Векторов: {len(vectors)} | размерность: {index.dimension}")
    print(f"Manifest: {target / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
