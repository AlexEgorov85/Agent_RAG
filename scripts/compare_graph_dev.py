# -*- coding: utf-8 -*-
"""Сравнить hybrid retrieval с graph-augmented обёрткой на dev-корпусе.

Скрипт не меняет существующий hybrid retriever: он собирает обычный
``HybridRetriever`` (dense + BM25 + опциональный reranker) и оборачивает его в
``GraphAugmentedRetriever``. Для каждого вопроса печатаются top-k chunk_id
с graph и без graph, а также число добавленных graph-кандидатов.

Dense-поиск требует embedding-модель, поэтому нужен ``LLM_API_KEY`` и
``LLM_BASE_URL`` (ключ никогда не печатается). Векторы кэшируются.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

DEFAULT_QUESTIONS = (
    "Какие ключевые риски связаны со стратегией развития банка и как они влияют на капитал?",
    "Как стратегические приоритеты банка связаны с кредитным риском и достаточностью капитала?",
    "Какие технологии и продукты упоминаются в связи с рисками цифровой трансформации?",
    "Как изменение ключевой ставки связано с процентным риском и чистой процентной маржой?",
    "Какие ESG-риски и риски устойчивого развития связаны со стратегией банка?",
)


def load_questions(path: str | None) -> list[str]:
    """Прочитать вопросы из JSON-массива или текстового файла (по одному в строке)."""
    if path is None:
        return list(DEFAULT_QUESTIONS)
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(payload, dict):
        payload = payload.get("questions", [])
    if not isinstance(payload, list):
        raise ValueError("questions JSON must be a list or {'questions': [...]}")
    return [str(question).strip() for question in payload if str(question).strip()]


def resolve_graph_path(value: str | None) -> Path:
    """Вернуть явный артефакт графа или последний файл из ``data/graph``."""
    if value:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"graph artifact not found: {path}")
        return path
    candidates = sorted(
        path
        for path in (PROJECT / "data" / "graph").glob("*.json")
        if path.is_file() and not path.name.endswith("_manifest.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "graph artifact not found; pass --graph or build data/graph/*.json"
        )
    return candidates[-1]


def format_ids(result, top_k: int) -> str:
    """Компактный список top-k chunk_id результата."""
    ids = [hit.chunk_id for hit in result.hits[:top_k]]
    return ", ".join(ids) if ids else "—"


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")

    parser = argparse.ArgumentParser(
        description="Сравнить hybrid retrieval с graph-augmented retrieval")
    parser.add_argument("--graph", default=None,
                        help="JSON-артефакт графа (по умолчанию последний data/graph/*.json)")
    parser.add_argument("--chunks", default=str(PROJECT / "data" / "dev" / "chunks"),
                        help="Каталог dev-чанков")
    parser.add_argument("--cache", default=str(PROJECT / "data" / "dev" / "cache"),
                        help="Каталог кэша embeddings")
    parser.add_argument("--questions", default=None,
                        help="JSON-массив или текстовый файл с вопросами "
                             "(по умолчанию 5 встроенных multi-hop вопросов)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--graph-depth", type=int, default=2,
                        help="Максимальная глубина обхода графа")
    parser.add_argument("--graph-max-results", type=int, default=10,
                        help="Максимум graph-кандидатов до слияния")
    parser.add_argument("--graph-weight", type=float, default=0.5,
                        help="Вес synthetic RRF-вклада graph-списка")
    parser.add_argument("--base-candidates", type=int, default=10,
                        help="Сколько базовых кандидатов брать в importance-режиме")
    parser.add_argument("--candidate-pool", type=int, default=10,
                        help="Общий пул кандидатов для реранкера (пакетами по 5)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Отключить LLM reranker (по умолчанию включён)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Порог важности rerank_score 0..3: включает "
                             "importance-режим и позволяет графу вытеснять "
                             "базовые чанки (требует reranker)")
    parser.add_argument("--context-chars", type=int, default=16000,
                        help="Символьный бюджет упаковки контекста")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Токенный бюджет упаковки (например, "
                             "context_window - output - reserve)")
    parser.add_argument("--year", type=int, default=None,
                        help="Ограничить выдачу годом")
    parser.add_argument("--doc", action="append", default=None,
                        help="Ограничить выдачу doc_id (можно несколько раз)")
    args = parser.parse_args(argv)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL в .env")
        return 1
    if args.top_k < 1:
        print("--top-k должен быть положительным")
        return 2
    if args.graph_depth < 0 or args.graph_max_results < 0 or args.graph_weight < 0:
        print("Параметры графа должны быть неотрицательными")
        return 2
    if args.threshold is not None and not 0 <= args.threshold <= 3:
        print("--threshold должен быть от 0 до 3")
        return 2
    if args.threshold is not None and args.no_rerank:
        print("--threshold требует reranker: уберите --no-rerank")
        return 2
    if args.base_candidates < 1 or args.candidate_pool < 1:
        print("--base-candidates и --candidate-pool должны быть положительными")
        return 2

    try:
        graph_path = resolve_graph_path(args.graph)
    except FileNotFoundError as exc:
        print(f"Граф недоступен: {exc}")
        return 1

    from src.core.cache import cached_document_vectors
    from src.core.corpus import load_chunks
    from src.core.dense_index import DenseIndex
    from src.core.embeddings import ReportEmbeddings, make_embeddings_client
    from src.engine.context import pack_context
    from src.graph.graph_builder import load_graph
    from src.retrieval.bm25 import BM25Index
    from src.retrieval.graph_hybrid import GraphAugmentedRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever, SearchFilters
    from src.retrieval.reranker import LLMReranker

    chunks = load_chunks(Path(args.chunks))
    if not chunks:
        print(f"Нет чанков в {args.chunks}. Сначала: python scripts/build_dev_corpus.py")
        return 1

    try:
        loaded = load_graph(graph_path)
    except (OSError, ValueError) as exc:
        print(f"Не удалось загрузить граф {graph_path}: {exc}")
        return 1

    from openai import OpenAI

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
    reranker = None if args.no_rerank else LLMReranker(
        client,
        os.getenv("RERANKER_MODEL", "ai-sage/GigaChat3-10B-A1.8B"),
        allow_missing_scores=True,
        max_candidates=args.candidate_pool,
    )
    # В режиме важности реранкингом управляет обёртка, поэтому базовые и
    # графовые кандидаты соревнуются в одном списке, а не оцениваются дважды.
    base = HybridRetriever(
        dense, bm25, embeddings,
        reranker=None if args.threshold is not None else reranker,
        top_k=args.top_k,
    )

    chunk_store = {
        str(chunk["chunk_id"]): chunk
        for chunk in chunks
        if isinstance(chunk.get("chunk_id"), str) and chunk["chunk_id"]
    }
    graph_retriever = GraphAugmentedRetriever(
        base,
        loaded.graph,
        chunk_store,
        graph_depth=args.graph_depth,
        graph_max_results=args.graph_max_results,
        graph_weight=args.graph_weight,
        reranker=reranker if args.threshold is not None else None,
        relevance_threshold=args.threshold,
        base_candidate_k=args.base_candidates,
        max_candidates=args.candidate_pool,
    )

    filters = SearchFilters(
        years={args.year} if args.year is not None else None,
        allowed_doc_ids=set(args.doc) if args.doc else None,
    )
    questions = load_questions(args.questions)
    if not questions:
        print("Список вопросов пуст")
        return 1

    print(f"Dev-корпус: {len(chunks)} чанков | документы: "
          f"{sorted({chunk.get('doc_id', '') for chunk in chunks})}")
    print(f"Граф: {graph_path} | узлов: {loaded.graph.number_of_nodes()}, "
          f"рёбер: {loaded.graph.number_of_edges()} | "
          f"chunks в manifest: {loaded.manifest.get('chunk_count', '—')}")
    print(f"Reranker: {'off' if args.no_rerank else 'on'} | top_k={args.top_k} | "
          f"graph depth={args.graph_depth}, max={args.graph_max_results}, "
          f"weight={args.graph_weight} | "
          f"threshold={args.threshold if args.threshold is not None else '—'} | "
          f"budget={args.max_context_tokens or '—'} tokens / "
          f"{args.context_chars} chars")

    for index, question in enumerate(questions, start=1):
        print(f"\n=== Q{index}/{len(questions)}: {question} ===")
        baseline = base.retrieve(question, filters=filters, top_k=args.top_k)
        # Обёртка получает top_k=None, чтобы список графовых свидетельств не
        # обрезался после базового top-k; базовая часть всё равно использует
        # args.top_k как настроенный по умолчанию лимит HybridRetriever.
        augmented = graph_retriever.retrieve(
            question, filters=filters, top_k=None)
        base_ids = [hit.chunk_id for hit in baseline.hits[:args.top_k]]
        graph_ids = [hit.chunk_id for hit in augmented.hits[:args.top_k]]
        added = [
            hit.chunk_id
            for hit in augmented.hits
            if hit.chunk_id not in set(base_ids)
        ]
        print(f"base  ({len(base_ids)}): {format_ids(baseline, args.top_k)}")
        print(f"graph ({len(graph_ids)}): {format_ids(augmented, args.top_k)}")
        print(f"added ({len(added)}): {', '.join(added) if added else '—'}")
        packed = pack_context(
            augmented.hits,
            max_chars=args.context_chars,
            max_tokens=args.max_context_tokens,
        )
        print(f"packed: {len(packed.records)} чанков | "
              f"{packed.manifest.used_chars} chars | "
              f"{packed.manifest.used_tokens} tokens"
              + (f" из {packed.manifest.max_tokens}"
                 if packed.manifest.max_tokens is not None else ""))
        trace = augmented.trace
        print(f"trace: graph_candidates={getattr(trace, 'graph_candidates', 0)} "
              f"matched_entities={len(getattr(trace, 'graph_matched_entities', []))} "
              f"failed={getattr(trace, 'graph_failed', False)}")
        if augmented.warnings:
            print("warnings: " + "; ".join(augmented.warnings))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
