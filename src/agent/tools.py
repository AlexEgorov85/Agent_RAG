"""Allowlisted adapters around injected retrieval and local artifact tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.retrieval.hybrid_retriever import SearchFilters


class ToolUnavailable(RuntimeError):
    """A supported optional tool is not configured or its index is unavailable."""


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


@dataclass
class AgentTools:
    """Server-configured tool facade. Missing optional backends fail safely."""

    search_backend: SearchBackend | Callable[..., Any] | None = None
    structure_backend: StructureBackend | Callable[..., Any] | None = None
    entity_backend: EntityBackend | Callable[..., Any] | None = None
    chunk_backend: ChunkBackend | Callable[..., Any] | None = None
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

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name not in self.ALLOWLIST:
            raise ValueError("unknown_tool")
        args = dict(arguments)
        if name == "search_reports":
            backend = self.search_backend
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
            top_k = min(top_k, self.top_k)
            if hasattr(backend, "retrieve"):
                result = backend.retrieve(query, filters=filters)
            else:
                result = backend(query, filters=filters, retrieval_mode=mode, top_k=top_k)
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
    """Normalize RetrievalResult/list/mapping search outputs."""
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
