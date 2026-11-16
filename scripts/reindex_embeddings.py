# -*- coding: utf-8 -*-
"""Пересчитать кэш эмбеддингов с чекпоинтами и pacing под лимиты провайдера.

Штатный ``build_indexes.py`` пишет кэш только после последнего батча, поэтому
429/обрыв на большом корпусе теряет всю работу. Этот скрипт:

* режет чанки на батчи по числу символов (лимит payload — 105 600);
* выдерживает паузу между запросами под RPM провайдера;
* каждые несколько батчей пишет ``*.partial.json`` и продолжает с него;
* в конце пишет финальный ``embeddings_<fingerprint>.json``.

После успешного кэша FAISS собирается существующим скриптом::

    python scripts/reindex_embeddings.py
    python scripts/build_indexes.py --reuse-cache-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.cache import corpus_fingerprint  # noqa: E402
from src.core.corpus import load_chunks  # noqa: E402
from src.core.embeddings import ReportEmbeddings, make_embeddings_client  # noqa: E402

DEFAULT_QUERY_PREFIX = "query:"
MAX_PAYLOAD_CHARS = 90_000
MAX_BATCH_CHUNKS = 64
CHECKPOINT_EVERY = 5


def _validate_vectors(vectors: Any, expected: int) -> list[list[float]]:
    """Проверить форму ответа провайдера, не доверяя битым векторам."""
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise ValueError("embedding count does not match batch size")
    result: list[list[float]] = []
    dimensions: set[int] = set()
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("embedding must be a non-empty numeric vector")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains a non-finite value")
        dimensions.add(len(values))
        result.append(values)
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions differ")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Атомарная запись JSON: временный файл рядом и ``replace``."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _text_hashes(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """SHA-256 текста каждого чанка: защита от переиспользования чужих векторов."""
    return {
        str(chunk.get("chunk_id", "")): hashlib.sha256(
            str(chunk.get("text", "")).encode("utf-8")).hexdigest()
        for chunk in chunks
    }


def _load_reusable_prefix(
    cache_dir: Path, model: str, chunk_ids: list[str], hashes: dict[str, str]
) -> tuple[list[list[float]], int, Path | None]:
    """Взять самый длинный префикс готовых векторов с совпавшими id и текстом.

    Кэш адресуется fingerprint'ом всего корпуса, поэтому при добавлении нового
    отчёта он формально устаревает. Если старые чанки идут префиксом и их текст
    не изменился, их векторы можно переиспользовать и посчитать только новые.
    """
    best_vectors: list[list[float]] = []
    best_count = 0
    best_path: Path | None = None
    for path in sorted(cache_dir.glob("embeddings_*.json")):
        if path.name.endswith(".partial.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("model") != model:
            continue
        sidecar = path.with_suffix(".text_sha256.json")
        if not sidecar.is_file():
            continue
        try:
            saved_hashes = json.loads(
                sidecar.read_text(encoding="utf-8")).get("text_sha256")
        except (OSError, ValueError):
            continue
        saved_ids = payload.get("chunk_ids")
        saved_vectors = payload.get("vectors")
        if (
            not isinstance(saved_hashes, dict)
            or not isinstance(saved_ids, list)
            or not isinstance(saved_vectors, list)
        ):
            continue
        limit = min(len(saved_ids), len(saved_vectors), len(chunk_ids))
        count = 0
        while count < limit:
            chunk_id = saved_ids[count]
            if chunk_id != chunk_ids[count]:
                break
            if saved_hashes.get(chunk_id) != hashes.get(chunk_id):
                break
            count += 1
        if count <= best_count:
            continue
        try:
            prefix = _validate_vectors(saved_vectors[:count], count)
        except ValueError:
            continue
        best_vectors, best_count, best_path = prefix, count, path
    return best_vectors, best_count, best_path


def _load_partial(
    path: Path, fingerprint: str, model: str, chunk_ids: list[str]
) -> tuple[list[str], list[list[float]]]:
    """Продолжить с чекпоинта, только если он относится к тому же корпусу."""
    if not path.is_file():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    saved_ids = payload.get("chunk_ids")
    saved_vectors = payload.get("vectors")
    if (
        payload.get("fingerprint") != fingerprint
        or payload.get("model") != model
        or not isinstance(saved_ids, list)
        or not isinstance(saved_vectors, list)
        or len(saved_ids) != len(saved_vectors)
        or saved_ids != chunk_ids[: len(saved_ids)]
    ):
        return [], []
    try:
        return list(saved_ids), _validate_vectors(
            saved_vectors, len(saved_ids))
    except ValueError:
        # Битый чекпоинт (например, смешанные размерности после обрыва)
        # не должен блокировать пересчёт: начинаем заново.
        return [], []


def _batches(
    chunks: list[dict[str, Any]], start: int,
    max_chars: int = MAX_PAYLOAD_CHARS,
    max_chunks: int = MAX_BATCH_CHUNKS,
) -> list[tuple[int, int]]:
    """Нарезать оставшиеся чанки по суммарному числу символов."""
    result: list[tuple[int, int]] = []
    index = start
    while index < len(chunks):
        end = index
        chars = 0
        while end < len(chunks) and end - index < max_chunks:
            size = len(str(chunks[end].get("text", "")))
            if end > index and chars + size > max_chars:
                break
            chars += size
            end += 1
        result.append((index, end))
        index = end
    return result


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    parser = argparse.ArgumentParser(
        description="Пересчитать кэш эмбеддингов с чекпоинтами.")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT / "data" / "chunks")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT / "data" / "cache")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_CHUNKS)
    parser.add_argument("--max-chars", type=int, default=MAX_PAYLOAD_CHARS)
    parser.add_argument("--rpm", type=int, default=25,
                        help="Запросов в минуту (лимит free-тарифа — 30)")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--no-reuse", action="store_true",
                        help="Не переиспользовать векторы старого кэша")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_chars < 1 or args.rpm < 1:
        parser.error("--batch-size/--max-chars/--rpm должны быть >= 1")

    model = (os.getenv("EMBED_MODEL") or "").strip()
    if not model:
        print("Не задан EMBED_MODEL.", file=sys.stderr)
        return 1
    query_prefix = os.getenv("EMBED_QUERY_PREFIX", DEFAULT_QUERY_PREFIX)
    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        print(f"В {args.chunks_dir} нет чанков.", file=sys.stderr)
        return 1
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    if not all(chunk_ids):
        print("У каждого чанка должен быть непустой chunk_id.", file=sys.stderr)
        return 1

    fingerprint = corpus_fingerprint(
        chunks, model, embedding_config={"query_prefix": query_prefix})
    target = args.cache_dir / f"embeddings_{fingerprint}.json"
    partial = target.with_suffix(".partial.json")
    print(f"Чанков: {len(chunks)} | модель: {model} | префикс: {query_prefix!r}")
    print(f"Fingerprint: {fingerprint}")
    print(f"Кэш: {target}")

    if target.is_file():
        sidecar = target.with_suffix(".text_sha256.json")
        if not sidecar.is_file():
            _write_json(sidecar, {
                "model": model,
                "chunk_ids": chunk_ids,
                "text_sha256": _text_hashes(chunks),
            })
        print("Готовый кэш уже существует — пересчёт не нужен.")
        return 0

    hashes = _text_hashes(chunks)
    saved_ids, vectors = _load_partial(partial, fingerprint, model, chunk_ids)
    if saved_ids:
        print(f"Продолжаю с чекпоинта: {len(saved_ids)}/{len(chunks)} чанков.")
    elif not args.no_reuse:
        vectors, reused, source = _load_reusable_prefix(
            args.cache_dir, model, chunk_ids, hashes)
        if reused:
            saved_ids = chunk_ids[:reused]
            print(
                f"Переиспользую {reused}/{len(chunks)} векторов "
                f"из {source.name if source else '<cache>'}.")
    start = len(saved_ids)

    try:
        client = make_embeddings_client()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    embeddings = ReportEmbeddings(client, model, query_prefix)
    interval = 60.0 / args.rpm
    last_request = 0.0
    total = len(chunks)
    written = start

    try:
        for number, (begin, end) in enumerate(
                _batches(chunks, start, args.max_chars, args.batch_size), 1):
            pause = interval - (time.monotonic() - last_request)
            if pause > 0:
                time.sleep(pause)
            last_request = time.monotonic()
            result = embeddings.embed_documents(
                [str(chunk.get("text", "")) for chunk in chunks[begin:end]])
            vectors.extend(_validate_vectors(result, end - begin))
            saved_ids.extend(chunk_ids[begin:end])
            written = end
            print(f"  батч {number}: {written}/{total}", flush=True)
            if number % args.checkpoint_every == 0:
                _write_json(partial, {
                    "fingerprint": fingerprint,
                    "model": model,
                    "chunk_ids": saved_ids,
                    "vectors": vectors,
                })
        payload = {
            "fingerprint": fingerprint,
            "model": model,
            "chunk_ids": chunk_ids,
            "vectors": vectors,
        }
        _write_json(target, payload)
        _write_json(target.with_suffix(".text_sha256.json"), {
            "model": model,
            "chunk_ids": chunk_ids,
            "text_sha256": hashes,
        })
        if partial.exists():
            partial.unlink()
    except Exception as exc:  # noqa: BLE001 - секреты не печатаем
        _write_json(partial, {
            "fingerprint": fingerprint,
            "model": model,
            "chunk_ids": saved_ids,
            "vectors": vectors,
        })
        print(
            f"Ошибка на {written}/{total} ({type(exc).__name__}); "
            f"прогресс сохранён в {partial.name}. Перезапустите команду.",
            file=sys.stderr,
        )
        return 1

    print(f"Готово: {target} | векторов: {len(vectors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
