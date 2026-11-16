# -*- coding: utf-8 -*-
"""Оценка retrieval-режимов на размеченном dev-наборе.

Скрипт считает один раз на каждый размер пула (retrieval + reranker), затем
применяет сетку порогов важности и бюджетов контекста постфактум, без
повторных вызовов LLM. Метрики считаются по реально упакованным чанкам:
hit@k (найден хотя бы один gold), MRR, recall по размеченным чанкам и
«labeled precision» (доля gold среди возвращённых; разметка неполная, поэтому
это нижняя оценка точности).

Запуск:
    .venv\\Scripts\\python.exe -X utf8 scripts/eval_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DEFAULT_QUESTIONS = PROJECT / "data" / "eval" / "retrieval_questions.json"
DEFAULT_OUT = PROJECT / "data" / "eval" / "retrieval_results.json"


@dataclass(frozen=True)
class SelectionConfig:
    name: str
    threshold: int
    max_chars: int
    max_tokens: int | None


def build_configs() -> list[SelectionConfig]:
    configs: list[SelectionConfig] = []
    budgets = [
        ("chars16000", 16_000, None),
        ("tok4000", 1_000_000_000, 4_000),
        ("tok8000", 1_000_000_000, 8_000),
        ("tok16000", 1_000_000_000, 16_000),
        ("tok28672", 1_000_000_000, 28_672),
    ]
    for threshold in (0, 1, 2, 3):
        for name, max_chars, max_tokens in budgets:
            configs.append(SelectionConfig(
                name=f"t{threshold}_{name}",
                threshold=threshold,
                max_chars=max_chars,
                max_tokens=max_tokens,
            ))
    return configs


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("questions", [])
    if not isinstance(payload, list):
        raise ValueError("questions file must be a list or {'questions': [...]}")
    questions = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each question must be an object")
        question = str(item.get("question", "")).strip()
        relevant = [str(value) for value in item.get("relevant_chunk_ids", [])]
        if not question or not relevant:
            raise ValueError("question and relevant_chunk_ids are required")
        questions.append({
            "id": str(item.get("id") or question),
            "question": question,
            "relevant_chunk_ids": relevant,
        })
    return questions


def metrics(returned: list[str], relevant: list[str]) -> dict[str, float]:
    gold = set(relevant)
    matched = [chunk_id for chunk_id in returned if chunk_id in gold]
    first_rank = next(
        (rank for rank, chunk_id in enumerate(returned, start=1)
         if chunk_id in gold),
        0,
    )
    return {
        "hit": 1.0 if matched else 0.0,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "recall": len(set(matched)) / len(gold) if gold else 0.0,
        "labeled_precision": len(set(matched)) / len(returned) if returned else 0.0,
        "returned": float(len(returned)),
    }


def mean(value: list[float]) -> float:
    return statistics.mean(value) if value else 0.0


def apply_config(
    config: SelectionConfig,
    hits: list[Any],
    relevant: list[str],
    pack_context: Any,
) -> dict[str, Any]:
    selected = hits
    if config.threshold > 0:
        selected = [
            hit for hit in hits
            if hit.rerank_score is not None and hit.rerank_score >= config.threshold
        ]
    packed = pack_context(
        selected,
        max_chars=config.max_chars,
        max_tokens=config.max_tokens,
    )
    result = metrics(packed.manifest.included, relevant)
    result["tokens"] = float(packed.manifest.used_tokens)
    result["chars"] = float(packed.manifest.used_chars)
    return result


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")

    parser = argparse.ArgumentParser(description="Оценка retrieval-режимов")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--graph", default=None,
                        help="Артефакт графа (по умолчанию последний data/graph/*.json)")
    parser.add_argument("--chunks", default=str(PROJECT / "data" / "dev" / "chunks"))
    parser.add_argument("--cache", default=str(PROJECT / "data" / "dev" / "cache"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pools", default="10,20,30",
                        help="Размеры пула кандидатов через запятую")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ограничить число вопросов для быстрой проверки")
    parser.add_argument("--include-relation-evidence", action="store_true",
                        help="Добавлять evidence соседних сущностей по RELATED_TO")
    parser.add_argument("--structure-budget", type=int, default=0,
                        help="Сколько структурных соседей (раздел/страница) "
                             "добавлять на один базовый хит; 0 — выключено")
    args = parser.parse_args(argv)

    if args.structure_budget < 0:
        print("--structure-budget должен быть неотрицательным")
        return 2

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL в .env")
        return 1

    pools = [int(value) for value in args.pools.split(",") if value.strip()]
    if not pools or any(pool < 1 for pool in pools):
        print("--pools должен содержать положительные числа")
        return 2

    from openai import OpenAI

    from src.core.cache import cached_document_vectors
    from src.core.corpus import load_chunks
    from src.core.dense_index import DenseIndex
    from src.core.embeddings import ReportEmbeddings, make_embeddings_client
    from src.engine.context import pack_context
    from src.graph.graph_builder import load_graph
    from src.retrieval.bm25 import BM25Index
    from src.retrieval.graph_hybrid import GraphAugmentedRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import LLMReranker

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    graph_path = Path(args.graph) if args.graph else None
    if graph_path is None:
        candidates = sorted(
            path for path in (PROJECT / "data" / "graph").glob("*.json")
            if not path.name.endswith("_manifest.json")
        )
        if not candidates:
            print("Граф не найден: сначала scripts/build_graph_dev.py")
            return 1
        graph_path = candidates[-1]

    chunks = load_chunks(Path(args.chunks))
    if not chunks:
        print(f"Нет чанков в {args.chunks}")
        return 1
    loaded = load_graph(graph_path)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=2)
    embed_client = make_embeddings_client(fallback_client=client)
    embeddings = ReportEmbeddings(
        embed_client,
        os.getenv("EMBED_MODEL", "ai-sage/Giga-Embeddings-instruct-3B"),
        os.getenv("EMBED_QUERY_PREFIX", "query:"),
    )
    vectors = cached_document_vectors(chunks, embeddings, Path(args.cache))
    dense = DenseIndex(chunks, vectors)
    bm25 = BM25Index(chunks)
    reranker = LLMReranker(
        client,
        os.getenv("RERANKER_MODEL", "ai-sage/GigaChat3-10B-A1.8B"),
        allow_missing_scores=True,
        max_candidates=max(pools),
    )
    chunk_store = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
        if isinstance(chunk.get("chunk_id"), str) and chunk["chunk_id"]
    }
    missing_gold = sorted({
        chunk_id
        for item in questions
        for chunk_id in item["relevant_chunk_ids"]
        if chunk_id not in chunk_store
    })
    if missing_gold:
        print("В разметке есть отсутствующие chunk_id:")
        for chunk_id in missing_gold:
            print(f"  {chunk_id}")
        return 1

    configs = build_configs()
    report: dict[str, Any] = {
        "graph": str(graph_path),
        "questions": len(questions),
        "relation_evidence": args.include_relation_evidence,
        "structure_budget": args.structure_budget,
        "pools": {},
        "baselines": {},
    }

    # Baseline 1: production-like top-5 without reranker (cheap, no LLM).
    base_plain = HybridRetriever(
        dense, bm25, embeddings, reranker=None, candidates=20, top_k=5)
    plain_rows = []
    for item in questions:
        result = base_plain.retrieve(item["question"], top_k=5)
        row = metrics([hit.chunk_id for hit in result.hits], item["relevant_chunk_ids"])
        plain_rows.append(row)
    report["baselines"]["no_rerank_top5"] = aggregate(plain_rows)
    print(f"baseline no-rerank top5: {format_row(aggregate(plain_rows))}")

    # Baseline 2: production-like top-5 with reranker.
    base_rerank = HybridRetriever(
        dense, bm25, embeddings,
        reranker=reranker, candidates=20, rerank_pool=10, top_k=5,
    )
    rerank_rows = []
    started = time.perf_counter()
    for item in questions:
        result = base_rerank.retrieve(item["question"], top_k=5)
        row = metrics([hit.chunk_id for hit in result.hits], item["relevant_chunk_ids"])
        rerank_rows.append(row)
    elapsed = time.perf_counter() - started
    report["baselines"]["rerank_top5"] = aggregate(rerank_rows)
    report["baselines"]["rerank_top5"]["latency_s"] = elapsed / len(questions)
    print(f"baseline rerank top5:   {format_row(aggregate(rerank_rows))} "
          f"{elapsed / len(questions):.2f}s/q")

    for pool in pools:
        base = HybridRetriever(
            dense, bm25, embeddings, reranker=None,
            candidates=max(20, pool), top_k=pool,
        )
        wrapper = GraphAugmentedRetriever(
            base, loaded.graph, chunk_store,
            reranker=reranker,
            relevance_threshold=None,
            max_candidates=pool,
            base_candidate_k=pool,
            graph_max_results=10,
            include_relation_evidence=args.include_relation_evidence,
            structure_budget=args.structure_budget,
        )
        per_question: list[dict[str, Any]] = []
        latencies: list[float] = []
        for item in questions:
            started = time.perf_counter()
            result = wrapper.retrieve(item["question"], top_k=None)
            latencies.append(time.perf_counter() - started)
            per_question.append({
                "id": item["id"],
                "question": item["question"],
                "hits": result.hits,
                "graph_added": getattr(result.trace, "graph_added", 0),
                "warnings": list(result.warnings),
            })

        config_rows: dict[str, list[dict[str, Any]]] = {}
        for config in configs:
            rows = []
            for item, stored in zip(questions, per_question):
                row = apply_config(
                    config, stored["hits"], item["relevant_chunk_ids"], pack_context)
                rows.append(row)
            config_rows[config.name] = rows

        pool_report = {
            "latency_s_mean": mean(latencies),
            "latency_s_median": statistics.median(latencies),
            "graph_added_mean": mean([float(item["graph_added"]) for item in per_question]),
            "configs": {
                name: aggregate(rows) for name, rows in config_rows.items()
            },
            "per_question": [
                {
                    "id": stored["id"],
                    "question": stored["question"],
                    "graph_added": stored["graph_added"],
                    "candidates": [hit.chunk_id for hit in stored["hits"]],
                    "scores": [hit.rerank_score for hit in stored["hits"]],
                    "warnings": stored["warnings"],
                }
                for stored in per_question
            ],
        }
        report["pools"][str(pool)] = pool_report

        print(f"\n--- pool={pool}: rerank latency "
              f"mean={pool_report['latency_s_mean']:.2f}s "
              f"median={pool_report['latency_s_median']:.2f}s | "
              f"graph_added={pool_report['graph_added_mean']:.1f} ---")
        ranked = sorted(
            pool_report["configs"].items(),
            key=lambda item: (-item[1]["mrr"], -item[1]["hit"]),
        )
        for name, row in ranked[:6]:
            print(f"  {name:20} {format_row(row)}")

    all_configs = [
        (f"pool{pool}/{name}", row)
        for pool, pool_report in report["pools"].items()
        for name, row in pool_report["configs"].items()
    ]
    best_name, best = max(
        all_configs,
        key=lambda item: (item[1]["mrr"], item[1]["hit"], item[1]["recall"]),
    )
    report["best"] = {"name": best_name, **best}
    print(f"\nBEST by MRR: {best_name} {format_row(best)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out_path}")
    return 0


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: mean([row[key] for row in rows])
        for key in ("hit", "mrr", "recall", "labeled_precision", "returned", "tokens", "chars")
        if any(key in row for row in rows)
    }


def format_row(row: dict[str, float]) -> str:
    return (
        f"hit={row.get('hit', 0):.2f} mrr={row.get('mrr', 0):.2f} "
        f"recall={row.get('recall', 0):.2f} prec={row.get('labeled_precision', 0):.2f} "
        f"chunks={row.get('returned', 0):.1f} tokens={row.get('tokens', 0):.0f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
