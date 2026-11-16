# -*- coding: utf-8 -*-
"""Сборка полного graph-индекса по каноническим чанкам ``data/chunks``.

Скрипт — обёртка над ``src/graph`` (``LLMEntityExtractor`` +
детерминированный ``build_graph``) с дефолтами полного корпуса и режимом
``--dry-run``, который показывает объём работы без обращения к LLM.

Секреты берутся только из env/``.env`` (``LLM_API_KEY``, ``LLM_BASE_URL``).
Тела ответов провайдера не печатаются.

Запуск::

    python scripts/build_full_graph.py --dry-run
    python scripts/build_full_graph.py
    python scripts/build_full_graph.py --doc sber_2023 --limit 200
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.corpus import load_chunks  # noqa: E402
from src.graph.entity_extractor import (  # noqa: E402
    LLM_ENTITY_EXTRACTOR_NAME,
    LLM_ENTITY_EXTRACTOR_VERSION,
    LLM_ENTITY_PROMPT_VERSION,
    ExtractionResult,
    LLMEntityExtractor,
    _max_confidence,
    _merge_alias_tuples,
)
from src.graph.graph_builder import build_graph, save_graph  # noqa: E402
from src.graph.graph_models import (  # noqa: E402
    EntityRecord,
    GraphBuildConfig,
    RelationRecord,
)

DEFAULT_MODEL = "GigaChat/GigaChat-3-Pro"


def build_parser() -> argparse.ArgumentParser:
    """CLI сборки полного графа."""
    parser = argparse.ArgumentParser(
        description="Собрать graph-индекс по чанкам полного корпуса.")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT / "data" / "chunks",
                        help="Каталог *_chunks.json (по умолчанию data/chunks)")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT / "data" / "graph",
                        help="Каталог graph/manifest JSON (по умолчанию data/graph)")
    parser.add_argument("--structure-dir", type=Path,
                        default=PROJECT / "data" / "structure",
                        help="Каталог *_sections.json (по умолчанию data/structure)")
    parser.add_argument("--doc", action="append", default=None,
                        help="Ограничить doc_id (можно несколько раз)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Максимум чанков после фильтра по --doc")
    parser.add_argument("--limit-per-doc", type=int, default=None,
                        help="Максимум чанков на документ (для пилота)")
    parser.add_argument("--chunks-per-call", type=int, default=4,
                        help="Чанков в одном запросе к LLM (по умолчанию 4)")
    parser.add_argument("--model", default=None,
                        help=f"Модель extraction (env LLM_MODEL или {DEFAULT_MODEL})")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Повторы на пакет при некорректном JSON")
    parser.add_argument("--workers", type=int, default=1,
                        help="Параллельных запросов к LLM (1–8; по умолчанию 1)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Сколько пакетов обработать за этот запуск")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="JSONL-чекпоинт успешных пакетов (по умолчанию "
                             "data/graph/.extraction_checkpoint.jsonl)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Не читать и не писать чекпоинт")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только посчитать чанки/пакеты, без вызова LLM")
    parser.add_argument("--strict", action="store_true",
                        help="Код возврата 1, если extraction оставил failed-чанки")
    return parser


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    """Проверить числовые лимиты CLI."""
    if args.limit is not None and args.limit < 0:
        parser.error("--limit должен быть неотрицательным")
    if args.limit_per_doc is not None and args.limit_per_doc < 1:
        parser.error("--limit-per-doc должен быть >= 1")
    if args.chunks_per_call < 1:
        parser.error("--chunks-per-call должен быть >= 1")
    if not 1 <= args.workers <= 32:
        parser.error("--workers должен быть от 1 до 32")


def _load_structures(doc_ids: set[str], structure_dir: Path) -> dict[str, Any]:
    """Прочитать доступные ``{doc_id}_sections.json``."""
    structures: dict[str, Any] = {}
    for doc_id in sorted(doc_ids):
        path = structure_dir / f"{doc_id}_sections.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            print(f"warning: структура недоступна: {path.name}", file=sys.stderr)
            continue
        if isinstance(payload, dict) and isinstance(payload.get("sections"), list):
            structures[doc_id] = payload
    return structures


def _select_chunks(chunks: list[dict[str, Any]],
                   args: argparse.Namespace) -> list[dict[str, Any]]:
    """Отфильтровать чанки по --doc и лимитам."""
    if args.doc:
        wanted = set(args.doc)
        chunks = [chunk for chunk in chunks
                  if chunk.get("doc_id") in wanted]
    if args.limit_per_doc is not None:
        limited: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id"))
            if counts.get(doc_id, 0) >= args.limit_per_doc:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
            limited.append(chunk)
        chunks = limited
    if args.limit is not None:
        chunks = chunks[:args.limit]
    return chunks


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Записать manifest графа атомарно (UTF-8)."""
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), **manifest}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temp.replace(path)


DEFAULT_CHECKPOINT = PROJECT / "data" / "graph" / ".extraction_checkpoint.jsonl"


def _batch_key(batch: list[dict[str, Any]]) -> str:
    ids = ",".join(str(chunk.get("chunk_id")) for chunk in batch)
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def _entity_from_dict(item: dict[str, Any]) -> EntityRecord:
    return EntityRecord(
        name=item["name"],
        entity_type=item["entity_type"],
        chunk_id=item["chunk_id"],
        aliases=tuple(item.get("aliases") or ()),
        extraction_version=item.get("extraction_version"),
        confidence=item.get("confidence"),
        source=item.get("source", "pre_extracted"),
    )


def _relation_from_dict(item: dict[str, Any]) -> RelationRecord:
    return RelationRecord(
        subject=item["subject"],
        object=item["object"],
        evidence_chunk_ids=tuple(item.get("evidence_chunk_ids") or ()),
        predicate=item.get("predicate", "RELATED_TO"),
        subject_type=item.get("subject_type", "Other"),
        object_type=item.get("object_type", "Other"),
        confidence=item.get("confidence"),
        extraction_version=item.get("extraction_version"),
        source=item.get("source", "pre_extracted"),
    )


def _load_checkpoint(
    path: Path | None, *, model: str
) -> dict[str, dict[str, Any]]:
    """Прочитать только успешно завершённые пакеты для текущей модели."""
    if path is None or not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            item = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict) or item.get("model") != model:
            continue
        key = item.get("batch")
        if isinstance(key, str) and not item.get("failed"):
            records[key] = item
    return records


def _covered_chunk_ids(records: dict[str, dict[str, Any]]) -> set[str]:
    """Чанки, уже обработанные в чекпоинте.

    Новые записи хранят список ``chunk_ids`` пакета целиком. В старых записях
    его нет, поэтому покрытие восстанавливается по ``chunk_id`` сущностей и
    списку ``skipped``; чанки без единой сущности просто попадут в повторную
    обработку, потерь нет.
    """
    covered: set[str] = set()
    for item in records.values():
        for chunk_id in item.get("chunk_ids") or []:
            covered.add(str(chunk_id))
        for entity in item.get("entities") or []:
            chunk_id = entity.get("chunk_id")
            if chunk_id:
                covered.add(str(chunk_id))
        for chunk_id in item.get("skipped") or []:
            covered.add(str(chunk_id))
    return covered


def _append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _extract_parallel(
    chunks: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    max_retries: int,
    chunks_per_call: int,
    workers: int,
    checkpoint_path: Path | None = None,
    max_batches: int | None = None,
) -> ExtractionResult:
    """Извлекает сущности пакетами параллельно и детерминированно склеивает.

    Один пакет = один запрос к LLM. Результаты сливаются по тем же ключам,
    что использует ``LLMEntityExtractor`` (сущность + chunk_id; отношение +
    evidence), поэтому порядок завершения потоков не влияет на граф.
    """
    batches: list[list[dict[str, Any]]] = []
    entities: dict[tuple[str, str, str], Any] = {}
    relations: dict[tuple[Any, ...], Any] = {}
    failed: list[str] = []
    skipped: list[str] = []
    calls = 0
    done = 0

    def merge_records(
        entity_records: list[Any], relation_records: list[Any]
    ) -> None:
        for record in entity_records:
            key = (record.name, record.entity_type, record.chunk_id)
            previous = entities.get(key)
            if previous is None:
                entities[key] = record
            else:
                entities[key] = replace(
                    previous,
                    aliases=_merge_alias_tuples(
                        previous.aliases, record.aliases
                    ),
                    confidence=_max_confidence(
                        previous.confidence, record.confidence
                    ),
                )
        for record in relation_records:
            key = (
                record.subject,
                record.object,
                record.predicate,
                record.evidence_chunk_ids,
                record.subject_type,
                record.object_type,
            )
            previous = relations.get(key)
            if previous is None:
                relations[key] = record
            else:
                relations[key] = replace(
                    previous,
                    confidence=_max_confidence(
                        previous.confidence, record.confidence
                    ),
                )

    saved = _load_checkpoint(checkpoint_path, model=model)
    done_keys = set(saved)
    for item in saved.values():
        calls += int(item.get("calls") or 0)
        skipped.extend(item.get("skipped") or [])
        merge_records(
            [_entity_from_dict(value) for value in item.get("entities") or []],
            [_relation_from_dict(value) for value in item.get("relations") or []],
        )
    done = len(done_keys)

    covered = _covered_chunk_ids(saved)
    todo_chunks = [
        chunk for chunk in chunks
        if str(chunk.get("chunk_id")) not in covered
    ]
    batches = [
        todo_chunks[start:start + chunks_per_call]
        for start in range(0, len(todo_chunks), chunks_per_call)
    ]
    print(
        f"[graph] чанков: {len(chunks)}, из checkpoint {len(chunks) - len(todo_chunks)}, "
        f"к обработке {len(todo_chunks)} в {len(batches)} пакетах",
        flush=True,
    )

    def extract_batch(batch: list[dict[str, Any]]) -> ExtractionResult:
        extractor = LLMEntityExtractor(
            client,
            model,
            max_retries=max_retries,
            chunks_per_call=max(1, len(batch)),
        )
        return extractor.extract(batch)

    pending = [batch for batch in batches if _batch_key(batch) not in done_keys]
    if max_batches is not None:
        pending = pending[:max(0, max_batches)]
        print(f"[graph] бюджет окна: {len(pending)} пакетов", flush=True)
    if done_keys:
        print(
            f"[graph] resume: {len(done_keys)} пакетов из checkpoint, "
            f"к отправке {len(pending)}",
            flush=True,
        )
    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_batch, batch): batch for batch in pending}
        for future in as_completed(futures):
            batch = futures[future]
            processed += 1
            try:
                result = future.result()
            except Exception:  # noqa: BLE001 - один пакет не должен рушить весь граф
                failed.extend(str(chunk.get("chunk_id")) for chunk in batch)
                result = None
            if result is not None:
                calls += result.calls
                skipped.extend(result.skipped_chunk_ids)
                failed.extend(result.failed_chunk_ids)
                merge_records(result.entities, result.relations)
                if checkpoint_path is not None and not result.failed_chunk_ids:
                    _append_checkpoint(
                        checkpoint_path,
                        {
                            "batch": _batch_key(batch),
                            "model": model,
                            "chunk_ids": [
                                str(chunk.get("chunk_id")) for chunk in batch
                            ],
                            "entities": [
                                asdict(record) for record in result.entities
                            ],
                            "relations": [
                                asdict(record) for record in result.relations
                            ],
                            "skipped": list(result.skipped_chunk_ids),
                            "calls": result.calls,
                        },
                    )
            if processed % 10 == 0 or processed == len(pending):
                print(
                    f"[graph] пакетов {processed}/{len(pending)} | calls {calls} | "
                    f"failed {len(set(failed))}",
                    flush=True,
                )

    ordered_entities = sorted(
        entities.values(),
        key=lambda record: (record.chunk_id, record.entity_type, record.name),
    )
    ordered_relations = sorted(
        relations.values(),
        key=lambda record: (
            record.subject,
            record.object,
            record.predicate,
            record.evidence_chunk_ids,
            record.subject_type,
            record.object_type,
        ),
    )
    return ExtractionResult(
        entities=ordered_entities,
        relations=ordered_relations,
        failed_chunk_ids=sorted(set(failed)),
        skipped_chunk_ids=sorted(set(skipped)),
        calls=calls,
    )


def main(argv: list[str] | None = None) -> int:
    """Точка входа сборки полного графа."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    chunks = load_chunks(args.chunks_dir)
    chunks = _select_chunks(chunks, args)
    if not chunks:
        print("Нет чанков для сборки графа. Проверьте --chunks-dir/--doc.",
              file=sys.stderr)
        return 1

    doc_ids = sorted({str(chunk.get("doc_id")) for chunk in chunks})
    batches = (len(chunks) + args.chunks_per_call - 1) // args.chunks_per_call
    print(f"Чанков: {len(chunks)} | документов: {len(doc_ids)} "
          f"({', '.join(doc_ids)}) | пакетов ~{batches}")

    if args.dry_run:
        for doc_id in doc_ids:
            own = sum(1 for chunk in chunks if chunk.get("doc_id") == doc_id)
            print(f"  {doc_id}: {own}")
        missing_structures = [
            doc_id for doc_id in doc_ids
            if not (args.structure_dir / f"{doc_id}_sections.json").is_file()]
        if missing_structures:
            print("Без структуры (SectionNode не будет): "
                  + ", ".join(missing_structures), file=sys.stderr)
        print("[dry-run] LLM не вызывался, артефакты не созданы.")
        return 0

    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL в .env или окружении.",
              file=sys.stderr)
        return 1

    structures = _load_structures(set(doc_ids), args.structure_dir)
    if not structures:
        print("Структуры разделов недоступны: граф будет без SectionNode.",
              file=sys.stderr)

    model = args.model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90,
                    max_retries=2)
    print(f"Extraction: модель {model} | workers {args.workers}", flush=True)
    extraction = _extract_parallel(
        chunks,
        client=client,
        model=model,
        max_retries=args.max_retries,
        chunks_per_call=args.chunks_per_call,
        workers=args.workers,
        checkpoint_path=None if args.no_checkpoint else args.checkpoint,
        max_batches=args.max_batches,
    )

    result = build_graph(
        chunks, structures,
        entities=extraction.entities,
        relations=extraction.relations,
        config=GraphBuildConfig(
            extractor_name=LLM_ENTITY_EXTRACTOR_NAME,
            extractor_version=LLM_ENTITY_EXTRACTOR_VERSION,
            prompt_version=LLM_ENTITY_PROMPT_VERSION,
        ),
    )

    manifest = dict(result.manifest)
    manifest["failed_chunk_ids"] = list(extraction.failed_chunk_ids)
    manifest["skipped_chunk_ids"] = list(extraction.skipped_chunk_ids)
    manifest["status"] = ("incomplete" if extraction.failed_chunk_ids
                          else "complete")
    result.manifest = manifest

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = save_graph(result, args.out_dir / f"{result.graph_fingerprint}.json")
    manifest_path = args.out_dir / f"{result.graph_fingerprint}_manifest.json"
    _write_manifest(manifest_path, manifest)

    print(f"Граф: {graph_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Чанки: {manifest.get('chunk_count', 0)} | "
          f"сущности: {manifest.get('entity_count', 0)} | "
          f"связи: {manifest.get('relation_count', 0)} | "
          f"LLM calls: {extraction.calls}")
    print(f"failed: {len(extraction.failed_chunk_ids)} | "
          f"skipped: {len(extraction.skipped_chunk_ids)} | "
          f"status: {manifest['status']}")
    print(f"fingerprint: {result.graph_fingerprint}")
    if args.strict and extraction.failed_chunk_ids:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
