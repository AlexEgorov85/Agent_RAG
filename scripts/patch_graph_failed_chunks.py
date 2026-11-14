# -*- coding: utf-8 -*-
"""Долить упавшие чанки в уже собранный граф без полной пересборки.

Скрипт читает ``failed_chunk_ids`` из manifest, повторно извлекает только эти
чанки, строит для них подграф и объединяет его с существующим графом. Новый
артефакт получает детерминированный fingerprint и manifest со статусом
``complete``. Полезен, когда полная пересборка дорога, а упало несколько
пакетов.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def fingerprint(graph: object) -> str:
    from src.graph.graph_models import serialize_graph

    content = serialize_graph(graph)
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")

    parser = argparse.ArgumentParser(description="Долить упавшие чанки в граф")
    parser.add_argument("--graph", required=True, help="Существующий артефакт графа")
    parser.add_argument("--chunks", default=str(PROJECT / "data" / "dev" / "chunks"))
    parser.add_argument("--out-dir", default=str(PROJECT / "data" / "graph"))
    parser.add_argument("--chunks-per-call", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "GigaChat/GigaChat-3-Pro"))
    args = parser.parse_args(argv)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL")
        return 1

    from openai import OpenAI

    import networkx as nx

    from src.core.corpus import load_chunks
    from src.graph.entity_extractor import LLMEntityExtractor
    from src.graph.graph_builder import build_graph, load_graph, save_graph
    from src.graph.graph_models import GraphBuildConfig, GraphBuildResult

    loaded = load_graph(Path(args.graph))
    failed_ids = list(loaded.manifest.get("failed_chunk_ids") or [])
    if not failed_ids:
        print("В manifest нет failed_chunk_ids — патч не нужен")
        return 0

    chunks = load_chunks(Path(args.chunks))
    by_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
        if isinstance(chunk.get("chunk_id"), str)
    }
    selected = [by_id[chunk_id] for chunk_id in failed_ids if chunk_id in by_id]
    missing = [chunk_id for chunk_id in failed_ids if chunk_id not in by_id]
    if missing:
        print(f"Предупреждение: {len(missing)} failed chunk_id отсутствуют в корпусе")
    if not selected:
        print("Нечего извлекать")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=2)
    extractor = LLMEntityExtractor(
        client,
        args.model,
        max_retries=args.max_retries,
        chunks_per_call=args.chunks_per_call,
    )
    extraction = extractor.extract(selected)
    print(f"Повторно извлечено: чанков {len(selected)}, "
          f"calls {extraction.calls}, entities {len(extraction.entities)}, "
          f"relations {len(extraction.relations)}, "
          f"failed {len(extraction.failed_chunk_ids)}")
    if extraction.failed_chunk_ids:
        print("Часть чанков снова не извлеклась: "
              + ", ".join(extraction.failed_chunk_ids))
        return 1

    config = GraphBuildConfig(
        extractor_name=loaded.manifest.get("extractor_name"),
        extractor_version=loaded.manifest.get("extractor_version"),
        prompt_version=loaded.manifest.get("prompt_version"),
    )
    sub = build_graph(
        selected,
        {},
        entities=extraction.entities,
        relations=extraction.relations,
        config=config,
    )

    merged = nx.compose(loaded.graph.copy(), sub.graph)
    # Сохраняем накопленные алиасы и версии из старого графа.
    for node_id, attrs in loaded.graph.nodes(data=True):
        if node_id not in merged or attrs.get("type") != "Entity":
            continue
        old_aliases = set(attrs.get("aliases") or ())
        new_aliases = set(merged.nodes[node_id].get("aliases") or ())
        merged.nodes[node_id]["aliases"] = sorted(old_aliases | new_aliases)
        old_versions = set(attrs.get("extraction_versions") or ())
        new_versions = set(merged.nodes[node_id].get("extraction_versions") or ())
        merged.nodes[node_id]["extraction_versions"] = sorted(
            old_versions | new_versions)

    new_fp = fingerprint(merged)
    manifest = dict(loaded.manifest)
    manifest.update({
        "graph_fingerprint": new_fp,
        "chunk_count": sum(
            1 for _, data in merged.nodes(data=True) if data.get("type") == "Chunk"),
        "entity_count": sum(
            1 for _, data in merged.nodes(data=True) if data.get("type") == "Entity"),
        "relation_count": sum(
            1 for _, _, data in merged.edges(data=True)
            if data.get("type") == "RELATED_TO"),
        "failed_chunk_ids": [],
        "skipped_chunk_ids": list(manifest.get("skipped_chunk_ids") or []),
        "status": "complete",
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "patched_chunk_ids": sorted(item["chunk_id"] for item in selected),
    })
    result = GraphBuildResult(
        graph=merged, graph_fingerprint=new_fp, manifest=manifest)

    out_dir = Path(args.out_dir)
    graph_path = save_graph(result, out_dir / f"{new_fp}.json")
    manifest_path = out_dir / f"{new_fp}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Граф: {graph_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Сущности: {manifest['entity_count']} | связи: {manifest['relation_count']}")
    print(f"fingerprint: {new_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
