"""Адаптеры из белого списка вокруг переданного поиска и локальных артефактов."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.retrieval.hybrid_retriever import SearchFilters

#: Предупреждение, которое возвращается вместе с обычным результатом, если
#: запрошен graph-поиск, но graph backend не настроен.
GRAPH_UNAVAILABLE = "graph_unavailable"

#: Явная ошибка несовместимости backend'а с запрошенным retrieval_mode.
BACKEND_MODE_ERROR = "backend_does_not_support_mode"

#: Режимы, которые меняют поведение каналов и не могут быть молча проигнорированы.
_EXPLICIT_MODES = frozenset({"dense", "keyword"})

#: Жёсткий потолок top_k для ``search_reports``. Selector просит 10 для
#: per-year сравнений; старое ограничение в 5 молча срезало этот запрос.
MAX_SEARCH_TOP_K = 10


def _supports_keyword(target: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or (
            item.name == keyword
            and item.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for item in parameters
    )


class ToolUnavailable(RuntimeError):
    """Поддерживаемый необязательный инструмент не настроен или его индекс недоступен."""


class SearchBackend(Protocol):
    def retrieve(self, query: str, *, filters: SearchFilters) -> Any: ...


class StructureBackend(Protocol):
    def get_structure_tree(
        self, doc_id: str, *, section_path: str | None, max_depth: int
    ) -> Any: ...


class EntityBackend(Protocol):
    def get_entities(
        self, entity_names: Sequence[str], *, filters: SearchFilters, max_hops: int
    ) -> Any: ...


class ChunkBackend(Protocol):
    def get_chunk(self, chunk_id: str) -> Any: ...


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _valid_filters(value: Any) -> SearchFilters:
    if not isinstance(value, SearchFilters):
        raise ValueError("filters must be SearchFilters")
    return value.normalized()


def _with_warning(result: Any, warning: str) -> Any:
    """Добавляет безопасное предупреждение к результату, не ломая его форму."""
    if result is None:
        return result
    if isinstance(result, Mapping):
        updated = dict(result)
        warnings = updated.get("warnings")
        warnings = list(warnings) if isinstance(warnings, (list, tuple)) else []
        if warning not in warnings:
            warnings.append(warning)
        updated["warnings"] = warnings
        return updated
    warnings = getattr(result, "warnings", None)
    if isinstance(warnings, (list, tuple)):
        updated_warnings = list(warnings)
        if warning not in updated_warnings:
            updated_warnings.append(warning)
        try:
            return replace(result, warnings=updated_warnings)
        except (TypeError, ValueError, AttributeError):
            # Неизменяемый/чужой объект не пересобираем принудительно.
            return result
    return result


@dataclass
class AgentTools:
    """Фасад настроенных на сервере инструментов. Отсутствующие опциональные backend'ы безопасно недоступны."""

    search_backend: SearchBackend | Callable[..., Any] | None = None
    #: Отдельный backend для ``use_graph=True`` (гибридный поиск + граф).
    #: Он не переписывает запрос: ``search_reports`` передаёт аргументы как есть.
    graph_search_backend: SearchBackend | Callable[..., Any] | None = None
    structure_backend: StructureBackend | Callable[..., Any] | None = None
    entity_backend: EntityBackend | Callable[..., Any] | None = None
    chunk_backend: ChunkBackend | Callable[..., Any] | None = None
    #: Значение top_k по умолчанию; запрошенное значение ограничивается
    #: ``MAX_SEARCH_TOP_K``, а не этим полем.
    top_k: int = 5
    max_depth: int = 4
    max_hops: int = 2

    ALLOWLIST = frozenset(
        {"search_reports", "get_structure_tree", "get_entities", "get_chunk"}
    )

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.max_depth < 0 or self.max_hops < 0:
            raise ValueError("tool depth limits must be non-negative")

    @staticmethod
    def _retrieve_search(
        backend: SearchBackend | Callable[..., Any],
        *,
        query: str,
        filters: SearchFilters,
        mode: str,
        top_k: int,
        queries: list[str] | None,
        rerank_per_year: bool | None = None,
    ) -> Any:
        """Вызывает выбранный backend, не подменяя запрошенный режим.

        Запрос не переписывается: инструмент только валидирует аргументы и
        передаёт их. Если backend не умеет переключать режим, ``dense``/
        ``keyword`` получают явную ошибку вместо молчаливого ``hybrid``.
        ``rerank_per_year`` передаётся только backend'ам, которые его
        поддерживают (например, ``PerYearSearch``).
        """
        retrieve = getattr(backend, "retrieve", None)
        target = retrieve if callable(retrieve) else backend
        if mode in _EXPLICIT_MODES and not _supports_keyword(
            target, "retrieval_mode"
        ):
            raise ValueError(BACKEND_MODE_ERROR)
        if callable(retrieve):
            kwargs: dict[str, Any] = {"filters": filters}
            if _supports_keyword(retrieve, "top_k"):
                kwargs["top_k"] = top_k
            if _supports_keyword(retrieve, "retrieval_mode"):
                kwargs["retrieval_mode"] = mode
            if queries and _supports_keyword(retrieve, "queries"):
                kwargs["queries"] = queries
            if rerank_per_year is not None and _supports_keyword(
                retrieve, "rerank_per_year"
            ):
                kwargs["rerank_per_year"] = rerank_per_year
            return retrieve(query, **kwargs)
        kwargs = {}
        if _supports_keyword(backend, "top_k"):
            kwargs["top_k"] = top_k
        if _supports_keyword(backend, "retrieval_mode"):
            kwargs["retrieval_mode"] = mode
        if queries and _supports_keyword(backend, "queries"):
            kwargs["queries"] = queries
        if rerank_per_year is not None and _supports_keyword(
            backend, "rerank_per_year"
        ):
            kwargs["rerank_per_year"] = rerank_per_year
        return backend(query, filters=filters, **kwargs)

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name not in self.ALLOWLIST:
            raise ValueError("unknown_tool")
        args = dict(arguments)
        if name == "search_reports":
            use_graph = args.get("use_graph", False)
            if not isinstance(use_graph, bool):
                raise ValueError("invalid_use_graph")
            backend = self.search_backend
            graph_fallback = False
            if use_graph:
                graph_backend = self.graph_search_backend
                if graph_backend is not None:
                    backend = graph_backend
                elif backend is not None:
                    # Граф — опциональная надстройка: запрос не падает, а
                    # деградирует до обычного поиска с явным предупреждением.
                    graph_fallback = True
            if backend is None:
                raise ToolUnavailable("search_unavailable")
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("invalid_query")
            filters = _valid_filters(args.get("filters"))
            mode = args.get("retrieval_mode", "hybrid")
            if mode not in {"hybrid", "dense", "keyword"}:
                raise ValueError("invalid_retrieval_mode")
            top_k = args.get("top_k", self.top_k)
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError("invalid_top_k")
            queries = args.get("queries")
            if queries is not None:
                if not isinstance(queries, (list, tuple)) or not all(
                    isinstance(item, str) and item.strip() for item in queries
                ):
                    raise ValueError("invalid_queries")
                queries = list(dict.fromkeys(item.strip() for item in queries))[:3]
            rerank_per_year = args.get("rerank_per_year")
            if rerank_per_year is not None and not isinstance(
                rerank_per_year, bool
            ):
                raise ValueError("invalid_rerank_per_year")

            # top_k — значение по умолчанию, а не потолок: selector просит 10
            # для per-year сравнений. Потолок остаётся, но явный.
            top_k = min(top_k, MAX_SEARCH_TOP_K)
            result = self._retrieve_search(
                backend,
                query=query,
                filters=filters,
                mode=mode,
                top_k=top_k,
                queries=queries,
                rerank_per_year=rerank_per_year,
            )
            if graph_fallback:
                result = _with_warning(result, GRAPH_UNAVAILABLE)
            return result

        if name == "get_structure_tree":
            backend = self.structure_backend
            if backend is None:
                raise ToolUnavailable("structure_unavailable")
            doc_id = args.get("doc_id")
            depth = args.get("max_depth", self.max_depth)
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("invalid_doc_id")
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
                raise ValueError("invalid_max_depth")
            depth = min(depth, self.max_depth)
            path = args.get("section_path")
            if path is not None and not isinstance(path, str):
                raise ValueError("invalid_section_path")
            if hasattr(backend, "get_structure_tree"):
                return backend.get_structure_tree(doc_id, section_path=path, max_depth=depth)
            return backend(doc_id, section_path=path, max_depth=depth)

        if name == "get_entities":
            backend = self.entity_backend
            if backend is None:
                raise ToolUnavailable("graph_unavailable")
            names = args.get("entity_names")
            if not isinstance(names, (list, tuple)) or not all(
                isinstance(item, str) and item.strip() for item in names
            ):
                raise ValueError("invalid_entity_names")
            hops = args.get("max_hops", self.max_hops)
            if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
                raise ValueError("invalid_max_hops")
            hops = min(hops, self.max_hops, 2)
            filters = _valid_filters(args.get("filters"))
            if hasattr(backend, "get_entities"):
                return backend.get_entities(names, filters=filters, max_hops=hops)
            return backend(names, filters=filters, max_hops=hops)

        backend = self.chunk_backend
        if backend is None:
            raise ToolUnavailable("chunk_unavailable")
        chunk_id = args.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("invalid_chunk_id")
        if hasattr(backend, "get_chunk"):
            return backend.get_chunk(chunk_id)
        return backend(chunk_id)


def result_hits(result: Any) -> list[Any]:
    """Приводит результат поиска (RetrievalResult/список/словарь) к списку хитов."""
    hits = _field(result, "hits", None)
    if hits is None:
        hits = result if isinstance(result, (list, tuple)) else []
    return list(hits) if isinstance(hits, (list, tuple)) else []


def chunk_from_hit(hit: Any) -> tuple[str | None, Any | None]:
    chunk = _field(hit, "chunk", hit)
    chunk_id = _field(hit, "chunk_id", _field(chunk, "chunk_id"))
    return (chunk_id if isinstance(chunk_id, str) and chunk_id else None, chunk)


def result_warnings(result: Any) -> list[str]:
    warnings = _field(result, "warnings", [])
    return [str(item) for item in warnings] if isinstance(warnings, (list, tuple)) else []
