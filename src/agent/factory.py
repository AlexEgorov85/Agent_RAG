"""Сборка готового ``AgentService`` из локального корпуса и Cloud.ru API.

Фабрика нужна UI и скриптам: она связывает уже проверенные компоненты
(retrieval, генерация ответа, инструменты агента), но НЕ создаёт корпус и не
выполняет парсинг. Если корпус ещё не собран, вызывающий получает понятную
ошибку вместо молчаливого «пустого» backend.

Секреты читаются только из env/``.env`` (``get_credentials``) и никогда не
попадают в сообщения об ошибках.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.agent.service import AgentService
from src.agent.tools import AgentTools
from src.core.cache import cached_document_vectors
from src.core.corpus import load_chunks
from src.core.credentials import get_credentials
from src.core.dense_index import DenseIndex
from src.core.embeddings import CourseEmbeddings
from src.core.rag_config import RAGSettings, load_rag_settings
from src.engine.generator import AnswerGenerator
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever

#: Префикс запроса, совместимый с уже собранным кэшем курсового dev-корпуса.
DEFAULT_QUERY_PREFIX = "query:"


def _structure_backend(structure_dir: Path):
    """Инструмент чтения дерева разделов из локальных артефактов."""

    def get_structure_tree(
        doc_id: str, *, section_path: str | None = None, max_depth: int = 4
    ) -> Any:
        path = structure_dir / f"{doc_id}_sections.json"
        if not path.is_file():
            from src.agent.tools import ToolUnavailable

            raise ToolUnavailable("structure_unavailable")
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections = payload.get("sections", []) if isinstance(payload, dict) else []
        wanted = (section_path or "").strip().rstrip("/")
        selected = []
        for section in sections:
            breadcrumbs = section.get("breadcrumbs") or []
            path_text = "/".join(str(item) for item in breadcrumbs)
            level = section.get("level")
            if isinstance(level, int) and level > max_depth:
                continue
            if wanted and not (
                path_text == wanted or path_text.startswith(wanted + "/")
            ):
                continue
            selected.append(
                {
                    "id": section.get("id"),
                    "title": section.get("title"),
                    "level": level,
                    "section_path": path_text,
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                }
            )
        return {
            "doc_id": doc_id,
            "section_path": wanted or None,
            "max_depth": max_depth,
            "sections": selected,
        }

    return get_structure_tree


def _chunk_backend(chunks_by_id: Mapping[str, Mapping[str, Any]]):
    """Инструмент чтения одного чанка по ID (только из корпуса)."""

    def get_chunk(chunk_id: str) -> Any:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            from src.agent.tools import ToolUnavailable

            raise ToolUnavailable("chunk_unavailable")
        return dict(chunk)

    return get_chunk


def _graph_backend(graph_dir: Path):
    """Инструмент graph-retrieval, если артефакт графа уже построен."""
    candidates = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.is_file() and not path.name.endswith("_manifest.json")
    )
    if not candidates:
        return None

    from src.graph.graph_builder import load_graph
    from src.retrieval.graph_retriever import resolve_entity_ids, retrieve_graph

    try:
        loaded = load_graph(candidates[-1])
    except (OSError, ValueError):
        return None

    def get_entities(
        entity_names, *, filters, max_hops: int
    ) -> dict[str, Any]:
        entity_ids = resolve_entity_ids(loaded.graph, list(entity_names))
        results = retrieve_graph(
            loaded.graph,
            entity_ids,
            doc_ids=filters.allowed_doc_ids,
            years=filters.years,
            issuers=filters.issuers,
            section_path=filters.section_path_prefix,
            max_depth=max_hops,
        )
        return {
            "nodes": entity_ids,
            "results": [
                {
                    "chunk_id": item.chunk_id,
                    "graph_score": item.graph_score,
                    "path": list(item.path),
                    "evidence_chunk_ids": list(item.evidence_chunk_ids),
                }
                for item in results
            ],
            "evidence_chunk_ids": sorted(
                {
                    chunk_id
                    for item in results
                    for chunk_id in item.evidence_chunk_ids
                }
            ),
        }

    return get_entities


def create_agent_service(
    *,
    chunks_dir: Path | str | None = None,
    structure_dir: Path | str | None = None,
    graph_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    client: Any = None,
    use_reranker: bool | None = None,
    settings: RAGSettings | None = None,
) -> AgentService:
    """Собрать ``AgentService`` на локальном корпусе.

    Args:
        chunks_dir: Каталог ``*_chunks.json``; по умолчанию из ``RAGSettings``.
        structure_dir: Каталог ``*_sections.json`` (по умолчанию ``data/structure``).
        graph_dir: Каталог артефактов графа (по умолчанию ``data/graph``).
        cache_dir: Каталог кэша векторов; по умолчанию ``RAGSettings.cache_dir``.
        client: Готовый OpenAI-совместимый клиент (для тестов/smoke).
        use_reranker: Включить LLM-реранкер. ``None`` — из env ``RERANK_ENABLED``.
        settings: Готовая конфигурация (для тестов).

    Returns:
        Настроенный ``AgentService`` с retrieval, генерацией и локальными tools.

    Raises:
        RuntimeError: корпус не найден/пуст или не задан ключ API.
    """
    settings = settings or load_rag_settings()
    chunks_path = Path(chunks_dir or settings.chunks_dir)
    if not chunks_path.is_dir():
        raise RuntimeError(
            f"Корпус не найден: {chunks_path}. Соберите чанки перед запуском UI."
        )
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise RuntimeError(f"В {chunks_path} нет чанков.")

    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=get_credentials(settings),
            base_url=settings.base_url,
            timeout=90,
            max_retries=2,
        )

    embeddings = CourseEmbeddings(
        client,
        settings.embeddings_model,
        os.getenv("EMBED_QUERY_PREFIX", DEFAULT_QUERY_PREFIX),
    )
    vectors = cached_document_vectors(
        chunks, embeddings, Path(cache_dir or settings.cache_dir)
    )
    dense = DenseIndex(chunks, vectors)
    bm25 = BM25Index(chunks)

    if use_reranker is None:
        use_reranker = os.getenv("RERANK_ENABLED", "0").strip() == "1"
    reranker = None
    if use_reranker and settings.rerank_model:
        from src.retrieval.reranker import LLMReranker

        reranker = LLMReranker(client, settings.rerank_model)

    retriever = HybridRetriever(
        dense,
        bm25,
        embeddings,
        reranker=reranker,
        candidates=settings.candidates,
        rerank_pool=settings.rerank_pool,
        top_k=settings.top_k,
    )
    tools = AgentTools(
        search_backend=retriever,
        structure_backend=_structure_backend(
            Path(structure_dir or "data/structure")
        ),
        chunk_backend=_chunk_backend({chunk["chunk_id"]: chunk for chunk in chunks}),
        entity_backend=_graph_backend(Path(graph_dir or "data/graph")),
    )
    answerer = AnswerGenerator(
        client,
        settings.model,
        max_context_chars=settings.context_chars,
    )
    return AgentService(tools, answerer=answerer)
