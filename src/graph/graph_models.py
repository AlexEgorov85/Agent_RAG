"""Типизированные контракты данных для in-memory среза Graph RAG.

Граф использует строковые ID и JSON-совместимые атрибуты, поэтому безопасно
сериализуется без pickle. Источником истины для чанков и структур разделов
остаётся ingestion-слой.
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
    """Узел метаданных документа с ключом ``doc:{doc_id}``."""

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
    """Графовое представление ``SectionNode`` из ingestion-слоя."""

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
    """Узел чанка, используемый при поиске."""

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
    """Заранее извлечённое упоминание сущности с происхождением на уровне чанка.

    ``chunk_id`` обязателен: извлечение сущностей намеренно вынесено из этого
    модуля, и запись никогда не запускает вызов LLM.
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
    """Заранее извлечённая связь сущностей, подкреплённая чанками-доказательствами."""

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
    """Опции детерминированной сборки графа.

    Поля ``extractor_name`` и версий — только описательные. Сами сущности и
    связи передаются вызывающим кодом готовыми записями.
    """

    extractor_name: str | None = None
    extractor_version: str | None = None
    prompt_version: str | None = None
    schema_version: str = GRAPH_SCHEMA_VERSION


@dataclass
class GraphBuildResult:
    """Собранный граф, детерминированный отпечаток и компактный манифест."""

    graph: nx.MultiDiGraph
    graph_fingerprint: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Возвращает стабильный JSON-совместимый артефакт графа."""
        return serialize_graph(self.graph, self.graph_fingerprint, self.manifest)


@dataclass(frozen=True)
class GraphRetrievalResult:
    """Отфильтрованный чанк-кандидат и графовый путь, объясняющий его находку."""

    chunk_id: str
    graph_score: float
    matched_entities: tuple[str, ...]
    path: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]


@dataclass
class GraphView:
    """Ограниченный по размеру JSON-совместимый подграф для визуализации."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


def entity_node_id(name: str, entity_type: str = "Other") -> str:
    """Создаёт детерминированный ID сущности из типа и нормализованного имени."""
    canonical_type = entity_type if entity_type in ENTITY_TYPES else "Other"
    normalized_name = " ".join(name.split()).casefold()
    return f"entity:{canonical_type}:{normalized_name}"


def _json_value(value: Any) -> Any:
    """Преобразует нестандартные коллекции в стабильные JSON-значения."""
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
    """Детерминированно сериализует MultiDiGraph обычными JSON-значениями."""
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
