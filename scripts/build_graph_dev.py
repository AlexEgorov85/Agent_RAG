# -*- coding: utf-8 -*-
"""Сборка dev-графа: LLM entity extraction + детерминированный ``build_graph``.

Скрипт читает чанки из ``--chunks-dir``, опционально ограничивает набор
документов/чанков, извлекает сущности и связи через OpenAI-совместимый API и
сохраняет JSON-артефакт графа в ``--out-dir``.

Секреты берутся только из env/``.env`` (``LLM_API_KEY``, ``LLM_BASE_URL``) и
никогда не выводятся. Тела ответов провайдера тоже не печатаются: при сбое
пакета в сводке видны только его ``chunk_id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.core.corpus import load_chunks
from src.graph.entity_extractor import (
    LLM_ENTITY_EXTRACTOR_NAME,
    LLM_ENTITY_EXTRACTOR_VERSION,
    LLM_ENTITY_PROMPT_VERSION,
    LLMEntityExtractor,
)
from src.graph.graph_builder import build_graph, save_graph
from src.graph.graph_models import GraphBuildConfig

DEFAULT_MODEL = "GigaChat/GigaChat-3-Pro"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать dev-граф: entity extraction (LLM) + structure edges")
    parser.add_argument(
        "--chunks-dir", type=Path, default=PROJECT / "data" / "dev" / "chunks",
        help="Каталог с *_chunks.json (по умолчанию data/dev/chunks)")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT / "data" / "graph",
        help="Каталог для graph/manifest JSON (по умолчанию data/graph)")
    parser.add_argument(
        "--doc", action="append", default=None,
        help="Ограничить doc_id (можно несколько раз)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Максимальное число чанков после фильтрации по --doc")
    parser.add_argument(
        "--chunks-per-call", type=int, default=4,
        help="Чанков в одном запросе к LLM (по умолчанию 4)")
    parser.add_argument(
        "--structure-dir", type=Path,
        default=PROJECT / "data" / "structure",
        help="Каталог с *_sections.json; недоступные структуры пропускаются")
    parser.add_argument(
        "--model", default=None,
        help=f"Модель extraction (по умолчанию env LLM_MODEL или {DEFAULT_MODEL})")
    parser.add_argument(
        "--max-retries", type=int, default=1,
        help="Повторы на пакет при некорректном JSON (по умолчанию 1)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit должен быть неотрицательным")
    if args.chunks_per_call < 1:
        parser.error("--chunks-per-call должен быть >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries должен быть неотрицательным")
    return args


def _load_structures(
    doc_ids: set[str], structure_dir: Path
) -> dict[str, dict[str, Any]]:
    """Прочитать доступные ``{doc_id}_sections.json``; иначе вернуть пусто."""
    structures: dict[str, dict[str, Any]] = {}
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


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    args = _parse_args(argv)

    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL в .env или окружении.",
              file=sys.stderr)
        return 1

    chunks = load_chunks(args.chunks_dir)
    if args.doc:
        wanted = set(args.doc)
        chunks = [chunk for chunk in chunks if chunk.get("doc_id") in wanted]
    if args.limit is not None:
        chunks = chunks[:args.limit]
    if not chunks:
        print("Нет чанков для сборки графа. Проверьте --chunks-dir/--doc.",
              file=sys.stderr)
        return 1

    doc_ids = sorted({str(chunk.get("doc_id")) for chunk in chunks})
    structures = _load_structures(set(doc_ids), args.structure_dir)
    if not structures:
        print("Структуры разделов недоступны: граф будет без SectionNode.",
              file=sys.stderr)

    model = args.model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=2)
    extractor = LLMEntityExtractor(
        client,
        model,
        max_retries=args.max_retries,
        chunks_per_call=args.chunks_per_call,
    )
    batches = (len(chunks) + args.chunks_per_call - 1) // args.chunks_per_call
    print(f"Extraction: документов {len(doc_ids)}, чанков {len(chunks)}, "
          f"пакетов ~{batches}, модель {model}")

    extraction = extractor.extract(chunks)
    result = build_graph(
        chunks,
        structures,
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
    manifest["status"] = (
        "incomplete" if extraction.failed_chunk_ids else "complete"
    )
    result.manifest = manifest

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = save_graph(
        result, args.out_dir / f"{result.graph_fingerprint}.json")
    manifest_path = args.out_dir / f"{result.graph_fingerprint}_manifest.json"
    _write_manifest(manifest_path, manifest)

    print(f"Граф: {graph_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Документы: {len(manifest.get('doc_ids', []))} | "
          f"чанки: {manifest.get('chunk_count', 0)}")
    print(f"Сущности: {manifest.get('entity_count', 0)} | "
          f"связи: {manifest.get('relation_count', 0)} | "
          f"LLM calls: {extraction.calls}")
    print(f"failed: {len(extraction.failed_chunk_ids)} | "
          f"skipped: {len(extraction.skipped_chunk_ids)}")
    if extraction.failed_chunk_ids:
        preview = ", ".join(extraction.failed_chunk_ids[:10])
        suffix = " ..." if len(extraction.failed_chunk_ids) > 10 else ""
        print(f"failed_chunk_ids: {preview}{suffix}", file=sys.stderr)
    if extraction.skipped_chunk_ids:
        preview = ", ".join(extraction.skipped_chunk_ids[:10])
        suffix = " ..." if len(extraction.skipped_chunk_ids) > 10 else ""
        print(f"skipped_chunk_ids: {preview}{suffix}", file=sys.stderr)
    print(f"status: {manifest['status']}")
    print(f"fingerprint: {result.graph_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
