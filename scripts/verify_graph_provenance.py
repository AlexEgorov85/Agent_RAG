# -*- coding: utf-8 -*-
"""Проверка provenance графа: сущности и связи должны опираться на текст.

Проверяет два инварианта:
1. каждая связь ``MENTIONS`` ведёт к чанку, в тексте которого буквально
   встречается canonical_name сущности;
2. каждый evidence-чанк связи ``RELATED_TO`` упоминает subject или object.

Код возврата 1, если найдены нарушения.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

_NON_ALNUM = re.compile(r"[^0-9a-zа-яё]+")


def key(value: object) -> str:
    return _NON_ALNUM.sub("", str(value).casefold())


def mentions(name: object, text: object) -> bool:
    name_key = key(name)
    return len(name_key) >= 3 and name_key in key(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверить provenance графа")
    parser.add_argument("--graph", default=None,
                        help="Артефакт графа (по умолчанию последний data/graph/*.json)")
    parser.add_argument("--chunks", default=str(PROJECT / "data" / "dev" / "chunks"))
    args = parser.parse_args(argv)

    from src.core.corpus import load_chunks
    from src.graph.graph_builder import load_graph

    graph_path = Path(args.graph) if args.graph else None
    if graph_path is None:
        candidates = sorted(
            path for path in (PROJECT / "data" / "graph").glob("*.json")
            if not path.name.endswith("_manifest.json")
        )
        if not candidates:
            print("Граф не найден")
            return 1
        graph_path = candidates[-1]

    chunks = {
        str(chunk["chunk_id"]): chunk
        for chunk in load_chunks(Path(args.chunks))
        if isinstance(chunk.get("chunk_id"), str)
    }
    graph = load_graph(graph_path).graph

    mention_total = 0
    mention_bad: list[tuple[str, str]] = []
    relation_total = 0
    relation_bad: list[tuple[str, list[str]]] = []

    for source, target, data in graph.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "MENTIONS" and source.startswith("chunk:"):
            chunk_id = source.replace("chunk:", "", 1)
            name = graph.nodes[target].get("canonical_name")
            mention_total += 1
            if not mentions(name, chunks.get(chunk_id, {}).get("text", "")):
                mention_bad.append((chunk_id, str(name)))
        elif edge_type == "RELATED_TO":
            relation_total += 1
            subject = str(
                data.get("subject")
                or graph.nodes[source].get("canonical_name")
                or ""
            )
            obj = str(
                data.get("object")
                or graph.nodes[target].get("canonical_name")
                or ""
            )
            bad_evidence = [
                str(chunk_id)
                for chunk_id in data.get("evidence_chunk_ids", [])
                if not (
                    mentions(subject, chunks.get(str(chunk_id), {}).get("text", ""))
                    or mentions(obj, chunks.get(str(chunk_id), {}).get("text", ""))
                )
            ]
            if bad_evidence:
                relation_bad.append((f"{subject} -> {obj}", bad_evidence))

    print(f"Граф: {graph_path}")
    print(f"MENTIONS: {mention_total} | нарушений: {len(mention_bad)}")
    for chunk_id, name in mention_bad[:10]:
        print(f"  BAD mention: {chunk_id} | {name!r}")
    print(f"RELATED_TO: {relation_total} | связей с плохим evidence: {len(relation_bad)}")
    for relation, chunk_ids in relation_bad[:10]:
        print(f"  BAD relation: {relation} | {chunk_ids[:3]}")
    return 0 if not mention_bad and not relation_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
