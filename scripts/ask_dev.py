# -*- coding: utf-8 -*-
"""End-to-end запрос по dev-корпусу: hybrid retrieval → JSON-ответ с цитатами."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.core.cache import cached_document_vectors
from src.core.corpus import load_chunks
from src.core.dense_index import DenseIndex
from src.core.embeddings import ReportEmbeddings, make_embeddings_client
from src.engine.generator import AnswerGenerator
from src.engine.rag_engine import RAGEngine
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever, SearchFilters
from src.retrieval.reranker import LLMReranker


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    parser = argparse.ArgumentParser(description="Запрос к dev-корпусу Sber RAG")
    parser.add_argument("question", help="Вопрос на русском")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--kind", default=None,
                        choices=["text", "table", "table_fact", "chart"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--agent", action="store_true",
                        help="Прогнать через AgentService (tools + answer generator)")
    parser.add_argument("--doc", action="append", default=None,
                        help="Ограничить dev-корпус doc_id (можно несколько раз)")
    args = parser.parse_args(argv)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        print("Не заданы LLM_API_KEY/LLM_BASE_URL в .env")
        return 1
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    embed_client = make_embeddings_client(fallback_client=client)
    chunks = load_chunks(PROJECT / "data" / "dev" / "chunks")
    if args.doc:
        chunks = [chunk for chunk in chunks
                  if chunk.get("doc_id") in set(args.doc)]
    if not chunks:
        print("Нет чанков. Сначала: python scripts/build_dev_corpus.py")
        return 1
    print(f"Dev-корпус: {len(chunks)} чанков, документы: "
          f"{sorted({chunk['doc_id'] for chunk in chunks})}")

    embed_model = os.getenv("EMBED_MODEL", "ai-sage/Giga-Embeddings-instruct-3B")
    embeddings = ReportEmbeddings(
        embed_client, embed_model, os.getenv("EMBED_QUERY_PREFIX", "query:"))
    vectors = cached_document_vectors(
        chunks, embeddings, PROJECT / "data" / "dev" / "cache")
    dense = DenseIndex(chunks, vectors)
    bm25 = BM25Index(chunks)
    reranker = None if args.no_rerank else LLMReranker(
        client, os.getenv("RERANKER_MODEL", "ai-sage/GigaChat3-10B-A1.8B"))
    retriever = HybridRetriever(dense, bm25, embeddings, reranker=reranker,
                                top_k=args.top_k)
    filters = SearchFilters(
        years={args.year} if args.year is not None else None,
        kinds={args.kind} if args.kind is not None else None)

    result = retriever.retrieve(args.question, filters=filters,
                                top_k=args.top_k)
    print(f"\nНайдено: {len(result.hits)} | warnings={result.warnings}")
    for rank, hit in enumerate(result.hits, 1):
        chunk = hit.chunk
        print(f"\n#{rank} {hit.chunk_id} | год {chunk.get('year')} | "
              f"{chunk.get('kind')} | rrf={hit.rrf_score:.4f} | "
              f"rerank={hit.rerank_score}")
        print(f"  раздел: {chunk.get('section_path') or '—'}")
        print(f"  {str(chunk.get('text', ''))[:500]}")

    if args.retrieval_only:
        return 0
    generator = AnswerGenerator(
        client, os.getenv("LLM_MODEL", "GigaChat/GigaChat-3-Pro"))
    if args.agent:
        from src.agent import AgentService
        from src.agent.tools import AgentTools

        service = AgentService(AgentTools(search_backend=retriever),
                               answerer=generator)
        run = service.ask(args.question, conversation_id="dev-session",
                          filters=filters)
        answer = run.answer
        print("\n=== AGENT ===")
        print("status:", run.status, "| tool_events:",
              [(event.tool_name, event.status, event.result_count)
               for event in run.tool_events])
        print("warnings:", run.warnings)
    else:
        engine = RAGEngine(retriever, generator)
        answer = engine.ask(args.question, filters=filters)
    print("\n=== ОТВЕТ ===")
    print("status:", answer.status)
    print(answer.answer)
    print("citations:", [citation.chunk_id for citation in answer.citations])
    print("warnings:", answer.warnings)
    target = PROJECT / "data" / "dev" / "last_answer.json"
    target.write_text(json.dumps({
        "status": answer.status,
        "answer": answer.answer,
        "citations": [citation.chunk_id for citation in answer.citations],
        "facts": answer.facts,
        "warnings": answer.warnings,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nСохранено: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
