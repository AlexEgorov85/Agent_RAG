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
import warnings
from pathlib import Path
from typing import Any, Mapping

from src.agent.llm_router import LLMRouter
from src.agent.selector import PlanAwareSelector
from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.tools import AgentTools
from src.core.cache import cached_document_vectors, corpus_fingerprint
from src.core.corpus import enrich_chunks_with_registry, load_chunks
from src.core.credentials import get_credentials
from src.core.source_registry import DEFAULT_REGISTRY_PATH, load_registry
from src.core.dense_index import DenseIndex
from src.core.embeddings import ReportEmbeddings, make_embeddings_client
from src.core.faiss_index import FaissDenseIndex, FaissIndexError
from src.core.rag_config import RAGSettings, load_rag_settings
from src.engine.generator import AnswerGenerator
from src.engine.structure_compare import StructureAnalyzer
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rewrite import QueryRewriter
from src.retrieval.temporal_search import PerYearSearch

#: Префикс запроса, совместимый с уже собранным кэшем dev-корпуса.
DEFAULT_QUERY_PREFIX = "query:"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str, default: bool) -> bool:
    """Runtime-override флага из env; пустое значение оставляет default.

    ``RAGSettings`` уже читает те же переменные, но env остаётся явным
    runtime-переключателем (например, ``RERANK_ENABLED=0`` в оболочке).
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    """Читает целочисленный runtime-override из env; мусор оставляет default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not an integer, using {default}",
                      RuntimeWarning, stacklevel=2)
        return default


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


def _load_graph_artifact(graph_dir: Path):
    """Один раз загружает активный graph-артефакт (или ``None``).

    Активным считается только артефакт с манифестом ``status="complete"``:
    частичный индекс (``failed_chunk_ids``) не должен попадать в поиск
    (``spec_graph.md`` §4). Среди полных выбирается наиболее полный по
    ``chunk_count``, при равенстве — самый свежий. Артефакт без манифеста
    (``graph.json`` из тестов и локальных сборок) допустим как fallback, но
    только если полных манифестированных артефактов нет.
    """
    from src.graph.graph_builder import load_graph

    for path in _graph_artifact_candidates(graph_dir):
        try:
            return load_graph(path)
        except (OSError, ValueError):
            continue
    return None


def _graph_artifact_candidates(graph_dir: Path) -> list[Path]:
    """Упорядочить артефакты графа: сначала полные, затем без манифеста."""
    if not graph_dir.is_dir():
        return []
    artifacts = [
        path
        for path in graph_dir.glob("*.json")
        if path.is_file() and not path.name.endswith("_manifest.json")
    ]
    by_name = {path.name: path for path in artifacts}

    complete: list[tuple[int, str, float, Path]] = []
    manifest_backed: set[str] = set()
    for manifest_path in graph_dir.glob("*_manifest.json"):
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        fingerprint = manifest.get("graph_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            manifest_backed.add(f"{fingerprint}.json")
        if manifest.get("status") != "complete":
            continue
        created_at = manifest.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = ""
        chunk_count = manifest.get("chunk_count")
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
            chunk_count = 0
        names = [
            f"{fingerprint}.json" if isinstance(fingerprint, str) else "",
            manifest_path.name[: -len("_manifest.json")] + ".json",
        ]
        for name in names:
            artifact = by_name.get(name)
            if artifact is not None:
                try:
                    mtime = artifact.stat().st_mtime
                except OSError:
                    mtime = 0.0
                complete.append((chunk_count, created_at, mtime, artifact))
                break

    # Больший полный граф покрывает больше документов; при равенстве — свежий.
    complete.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    ordered = [path for _, _, _, path in complete]
    seen = {path.name for path in ordered}
    # Артефакты с неполным манифестом не рассматриваются вовсе; без
    # манифеста — legacy-путь (граф, собранный скриптом или тестом).
    fallback = sorted(
        (
            path
            for path in artifacts
            if path.name not in seen and path.name not in manifest_backed
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return ordered + fallback


def _entity_backend(loaded):
    """Инструмент graph-retrieval поверх уже загруженного артефакта графа."""
    if loaded is None:
        return None
    from src.retrieval.graph_retriever import resolve_entity_ids, retrieve_graph

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


def _load_source_registry(path: Path):
    """Прочитать реестр источников; отсутствие файла не блокирует запуск."""
    try:
        return load_registry(path)
    except (OSError, ValueError):
        # Реестр — метаданные, а не корпус: битый/отсутствующий файл не должен
        # мешать чату работать, просто issuer останется неизвестным.
        return []


def _index_target(
    embeddings: ReportEmbeddings,
    chunks: list[dict[str, Any]],
    *,
    settings: RAGSettings,
    indexes_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> tuple[Path, str]:
    """Вернуть каталог индекса и fingerprint корпуса/модели."""
    if indexes_dir is None:
        indexes_dir = (
            Path(cache_dir).parent / "indexes"
            if cache_dir is not None
            else settings.indexes_dir
        )
    fingerprint = corpus_fingerprint(
        chunks,
        embeddings.model,
        embedding_config={"query_prefix": embeddings.query_prefix},
    )
    return Path(indexes_dir) / fingerprint, fingerprint


def _load_dense_index(
    chunks: list[dict[str, Any]],
    embeddings: ReportEmbeddings,
    *,
    settings: RAGSettings,
    indexes_dir: Path | str | None,
    cache_dir: Path | str | None,
) -> tuple[Any | None, Path, str]:
    """Попробовать загрузить FAISS-индекс без вычисления эмбеддингов."""
    if not settings.faiss_enabled:
        return None, Path(), ""
    target, fingerprint = _index_target(
        embeddings,
        chunks,
        settings=settings,
        indexes_dir=indexes_dir,
        cache_dir=cache_dir,
    )
    try:
        loaded = FaissDenseIndex.load(
            target,
            chunks,
            fingerprint=fingerprint,
            model=embeddings.model,
        )
    except (FaissIndexError, ImportError, ValueError, OSError) as exc:
        warnings.warn(
            f"faiss_load_failed: {exc}", RuntimeWarning, stacklevel=2
        )
        loaded = None
    return loaded, target, fingerprint


def _build_dense_index(
    chunks: list[dict[str, Any]],
    vectors: Mapping[str, list[float]],
    embeddings: ReportEmbeddings,
    *,
    settings: RAGSettings,
    target: Path,
    fingerprint: str,
) -> Any:
    """Собрать FAISS-индекс по готовым векторам с безопасным fallback."""
    if not settings.faiss_enabled:
        return DenseIndex(chunks, vectors)
    try:
        return FaissDenseIndex.load_or_build(
            target,
            chunks,
            vectors,
            fingerprint=fingerprint,
            model=embeddings.model,
            query_prefix=embeddings.query_prefix,
            persist=settings.faiss_persist,
        )
    except (FaissIndexError, ImportError, ValueError, OSError) as exc:
        warnings.warn(
            f"faiss_fallback_to_dense: {exc}", RuntimeWarning, stacklevel=2
        )
        return DenseIndex(chunks, vectors)


def create_agent_service(
    *,
    chunks_dir: Path | str | None = None,
    structure_dir: Path | str | None = None,
    graph_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    indexes_dir: Path | str | None = None,
    source_registry_path: Path | str | None = None,
    client: Any = None,
    embedding_client: Any = None,
    use_reranker: bool | None = None,
    settings: RAGSettings | None = None,
) -> AgentService:
    """Собрать ``AgentService`` на локальном корпусе.

    Args:
        chunks_dir: Каталог ``*_chunks.json``; по умолчанию из ``RAGSettings``.
        structure_dir: Каталог ``*_sections.json`` (по умолчанию ``data/structure``).
        graph_dir: Каталог артефактов графа (по умолчанию ``data/graph``).
        cache_dir: Каталог кэша векторов; по умолчанию ``RAGSettings.cache_dir``.
        indexes_dir: Каталог постоянных индексов; по умолчанию
            ``RAGSettings.indexes_dir``. Если задан ``cache_dir``, но не задан
            ``indexes_dir``, индекс кладётся рядом с кэшем (``../indexes``),
            чтобы тесты и локальные сборки не писали в общий ``data/indexes``.
        source_registry_path: Реестр источников для ``issuer``/``country`` чанков.
            ``None`` — env ``SOURCE_REGISTRY_PATH`` или ``data/source_registry.json``.
        client: Готовый OpenAI-совместимый клиент (для тестов/smoke).
        embedding_client: Отдельный клиент эмбеддингов; ``None`` — из
            ``EMBED_BASE_URL``/``EMBED_API_KEY`` или общий ``client``.
        use_reranker: Включить LLM-реранкер. ``None`` — из env ``RERANK_ENABLED``
            (по умолчанию включён; ``0`` отключает).
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

    registry_path = Path(
        source_registry_path
        or os.getenv("SOURCE_REGISTRY_PATH")
        or DEFAULT_REGISTRY_PATH
    )
    records = _load_source_registry(registry_path)
    source_status_by_doc: dict[str, str] = {}
    if records:
        # Чанки из PDF не содержат issuer: без обогащения фильтр issuers
        # отбрасывает весь корпус, а citations остаются без эмитента.
        enrich_chunks_with_registry(chunks, records)
        for record in records:
            source_status_by_doc[record.doc_id] = record.source_status

    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=get_credentials(settings),
            base_url=settings.base_url,
            timeout=90,
            max_retries=2,
        )
        if embedding_client is None:
            embedding_client = make_embeddings_client(
                getattr(settings, "embed_base_url", ""),
                getattr(settings, "embed_api_key", ""),
                fallback_client=client,
            )
    elif embedding_client is None:
        # Явно переданный клиент — инъекция для тестов; используем его и для
        # эмбеддингов, не собирая отдельный провайдер.
        embedding_client = client

    embeddings = ReportEmbeddings(
        embedding_client,
        settings.embeddings_model,
        os.getenv("EMBED_QUERY_PREFIX", DEFAULT_QUERY_PREFIX),
    )
    dense, index_target, index_fingerprint = _load_dense_index(
        chunks,
        embeddings,
        settings=settings,
        indexes_dir=indexes_dir,
        cache_dir=cache_dir,
    )
    if dense is None:
        vectors = cached_document_vectors(
            chunks, embeddings, Path(cache_dir or settings.cache_dir)
        )
        dense = _build_dense_index(
            chunks,
            vectors,
            embeddings,
            settings=settings,
            target=index_target,
            fingerprint=index_fingerprint,
        )
    bm25 = BM25Index(chunks)

    if use_reranker is None:
        # Реранкер есть в production и включён по умолчанию; явное
        # ``RERANK_ENABLED=0`` отключает его (например, ради latency/cost).
        use_reranker = _env_flag("RERANK_ENABLED", settings.rerank_enabled)
    reranker = None
    if use_reranker and settings.rerank_model:
        from src.retrieval.reranker import LLMReranker

        # Пул graph-кандидатов шире одного запроса реранкера: LLMReranker
        # сам режет его на батчи, но должен принять весь пул, иначе
        # graph+rerank молча деградирует к RRF.
        reranker = LLMReranker(
            client,
            settings.rerank_model,
            allow_missing_scores=True,
            max_candidates=settings.candidates,
        )

    retriever = HybridRetriever(
        dense,
        bm25,
        embeddings,
        reranker=reranker,
        candidates=settings.candidates,
        rerank_pool=settings.rerank_pool,
        top_k=settings.top_k,
        diversity=settings.diversity_enabled,
        max_per_doc=settings.max_per_doc,
        max_per_section=settings.max_per_section,
    )
    available_years = {
        int(chunk["year"])
        for chunk in chunks
        if chunk.get("year") is not None
    }
    documents_by_year: dict[int, list[str]] = {}
    for chunk in chunks:
        year = chunk.get("year")
        doc_id = chunk.get("doc_id")
        if year is None or not isinstance(doc_id, str) or not doc_id:
            continue
        try:
            normalized_year = int(year)
        except (TypeError, ValueError):
            continue
        ids = documents_by_year.setdefault(normalized_year, [])
        if doc_id not in ids:
            ids.append(doc_id)
    for ids in documents_by_year.values():
        ids.sort()

    # Документы с другим эмитентом (sber_2015 — БПС-Сбербанк) не участвуют
    # во временных сравнениях по умолчанию; прямой вопрос про год остаётся
    # доступным (spec_target.md §3.1).
    noncomparable_doc_ids = {
        doc_id
        for doc_id, status in source_status_by_doc.items()
        if status != "comparable"
    }
    if not records:
        from src.core.source_registry import NONCOMPARABLE_DOC_IDS

        noncomparable_doc_ids = set(NONCOMPARABLE_DOC_IDS)

    # Веерный поиск по годам собирается поверх hybrid retriever: фактологический
    # запрос с одним годом (или без периода) уходит в retriever напрямую, а
    # сравнение нескольких лет получает минимум per_year_k хитов на год.
    per_year_search = PerYearSearch(
        retriever,
        per_year_k=settings.per_year_k,
        max_years=settings.max_years,
        documents_by_year=documents_by_year,
        noncomparable_doc_ids=noncomparable_doc_ids,
    )

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    # Артефакт графа загружается один раз: он нужен и инструменту
    # get_entities, и graph-augmented поиску для use_graph=True.
    loaded_graph = _load_graph_artifact(Path(graph_dir or "data/graph"))
    graph_search_backend = None
    if loaded_graph is not None:
        from src.retrieval.graph_hybrid import GraphAugmentedRetriever

        # Реранкер общий: базовые и графовые кандидаты должны попасть в один
        # пул, иначе graph-хиты дописываются в конец и обрезаются top_k
        # (spec_graph.md §5/§8.5).
        graph_search_backend = GraphAugmentedRetriever(
            retriever,
            loaded_graph.graph,
            chunks_by_id,
            reranker=reranker,
            max_candidates=settings.candidates,
            base_candidate_k=settings.rerank_pool,
            # Структурный канал: соседи по разделу (CONTAINS) и странице.
            # Выключен при ``GRAPH_STRUCTURE_BUDGET=0``; бюджет измерен офлайн
            # (tmp_graph_ab/structure_budget_probe.py), см. spec_graph.md §8.6.
            structure_budget=_env_int("GRAPH_STRUCTURE_BUDGET", 2),
        )

    tools = AgentTools(
        search_backend=per_year_search,
        graph_search_backend=graph_search_backend,
        structure_backend=_structure_backend(
            Path(structure_dir or "data/structure")
        ),
        chunk_backend=_chunk_backend(chunks_by_id),
        entity_backend=_entity_backend(loaded_graph),
        top_k=settings.top_k,
    )
    answerer = AnswerGenerator(
        client,
        settings.model,
        max_context_chars=settings.context_chars,
        max_context_tokens=settings.context_tokens,
    )

    use_llm_router = _env_flag(
        "LLM_ROUTER_ENABLED", settings.llm_router_enabled
    )
    if use_llm_router:
        router = LLMRouter(
            client,
            settings.router_model or settings.model,
            sorted(available_years),
        )
    else:
        router = SimpleRouter(available_years)

    use_structure_llm = _env_flag(
        "LLM_STRUCTURE_VERIFY", settings.llm_structure_verify
    )
    structure_analyzer = StructureAnalyzer(
        client if use_structure_llm else None,
        (settings.router_model or settings.model) if use_structure_llm else None,
    )
    use_rewriter = _env_flag(
        "LLM_REWRITE_ENABLED", settings.llm_rewrite_enabled
    )
    query_rewriter = (
        QueryRewriter(
            client,
            settings.router_model or settings.model,
        )
        if use_rewriter
        else None
    )

    return AgentService(
        tools,
        answerer=answerer,
        router=router,
        action_selector=PlanAwareSelector(
            max_searches=settings.max_search_calls,
            max_structure_docs=settings.max_structure_docs,
            max_queries=settings.max_queries,
            coverage_enabled=settings.coverage_enabled,
            max_coverage_searches=settings.max_coverage_searches,
        ),
        documents_by_year=documents_by_year,
        structure_analyzer=structure_analyzer,
        query_rewriter=query_rewriter,
        max_tool_calls=settings.max_tool_calls,
        max_search_calls=settings.max_search_calls,
        verification_enabled=settings.verification_enabled,
    )
