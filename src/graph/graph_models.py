"""Typed data contracts for the in-memory Graph RAG slice.

The graph uses string IDs and JSON-compatible attributes so it can be safely
serialized without pickle.  The ingestion layer remains the source of truth for
chunks and section structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import networkx as nx

GRAPH_SCHEMA_VERSION = "1.0"
ENTITY_TYPES = frozenset({
    "Organization", "Person", "FinancialMetric", "Risk", "Strategy",
    "Product", "Technology", "Other",
})


@dataclass(frozen=True)
class DocumentNode:
    """Document metadata node, keyed as ``doc:{doc_id}``."""

    doc_id: str
    year: int | None = None
    report_period: str | None = None
    issuer: str | None = None
    country: str | None = None
    title: str | None = None
    source_sha256: str | None = None

    @property
    def node_id(self) -> str:
        return f"doc:{self.doc_id}"


@dataclass(frozen=True)
class SectionGraphNode:
    """Graph representation of an ingestion ``SectionNode``."""

    doc_id: str
    stable_section_id: str
    title: str
    level: int
    section_path: str
    page_start: int
    page_end: int

    @property
    def node_id(self) -> str:
        return f"section:{self.doc_id}:{self.stable_section_id}"


@dataclass(frozen=True)
class ChunkNode:
    """Retrieval-facing chunk node."""

    chunk_id: str
    doc_id: str
    year: int | None = None
    issuer: str | None = None
    kind: str | None = None
    page_pdf: int | None = None
    section_path: str | None = None
    evidence_id: str | None = None
    text: str = ""

    @property
    def node_id(self) -> str:
        return f"chunk:{self.chunk_id}"


@dataclass(frozen=True)
class EntityRecord:
    """Pre-extracted entity mention with chunk-level provenance.

    ``chunk_id`` is required: entity extraction is intentionally external to
    this module and this record never triggers an LLM call.
    """

    name: str
    entity_type: str
    chunk_id: str
    aliases: tuple[str, ...] = ()
    extraction_version: str | None = None
    confidence: float | None = None
    source: str = "pre_extracted"


@dataclass(frozen=True)
class RelationRecord:
    """Pre-extracted entity relation, backed by one or more evidence chunks."""

    subject: str
    object: str
    evidence_chunk_ids: tuple[str, ...]
    predicate: str = "RELATED_TO"
    subject_type: str = "Other"
    object_type: str = "Other"
    confidence: float | None = None
    extraction_version: str | None = None
    source: str = "pre_extracted"


@dataclass(frozen=True)
class GraphBuildConfig:
    """Deterministic graph build options.

    ``extractor_name`` and version fields are descriptive only.  Entity and
    relation extraction itself is supplied as records by the caller.
    """

    extractor_name: str | None = None
    extractor_version: str | None = None
    prompt_version: str | None = None
    schema_version: str = GRAPH_SCHEMA_VERSION


@dataclass
class GraphBuildResult:
    """Built graph, deterministic fingerprint, and compact manifest."""

    graph: nx.MultiDiGraph
    graph_fingerprint: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible graph artifact."""
        return serialize_graph(self.graph, self.graph_fingerprint, self.manifest)


@dataclass(frozen=True)
class GraphRetrievalResult:
    """A filtered chunk candidate and the graph path explaining its retrieval."""

    chunk_id: str
    graph_score: float
    matched_entities: tuple[str, ...]
    path: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]


@dataclass
class GraphView:
    """Size-limited JSON-compatible subgraph payload for visualization."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


def entity_node_id(name: str, entity_type: str = "Other") -> str:
    """Create a deterministic entity ID using type and normalized name."""
    canonical_type = entity_type if entity_type in ENTITY_TYPES else "Other"
    normalized_name = " ".join(name.split()).casefold()
    return f"entity:{canonical_type}:{normalized_name}"


def _json_value(value: Any) -> Any:
    """Convert common non-JSON collection values into stable JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_json_value(item) for item in value]
        return sorted(values, key=repr) if isinstance(value, (set, frozenset)) else values
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def serialize_graph(
    graph: nx.MultiDiGraph,
    graph_fingerprint: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a MultiDiGraph deterministically using ordinary JSON values."""
    nodes = [
        {"node_id": str(node_id), **_json_value(attrs)}
        for node_id, attrs in graph.nodes(data=True)
    ]
    nodes.sort(key=lambda node: node["node_id"])
    edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        edge_attrs = _json_value(attrs)
        if "source" in edge_attrs:
            edge_attrs["provenance_source"] = edge_attrs.pop("source")
        edges.append({
            **edge_attrs,
            "source": str(source),
            "target": str(target),
            "key": str(key),
        })
    edges.sort(key=lambda edge: (
        edge["source"], edge["target"], edge["key"],
        str(edge.get("type", "")), repr(edge),
    ))
    result: dict[str, Any] = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }
    if graph_fingerprint is not None:
        result["graph_fingerprint"] = graph_fingerprint
    if manifest is not None:
        result["manifest"] = _json_value(dict(manifest))
    return result
