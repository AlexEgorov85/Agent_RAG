"""Детерминированное построение графа NetworkX из чанков и структур документов."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from src.core.io import atomic_write_json
from src.core.record import field_value as _read

from src.graph.graph_models import (
    ENTITY_TYPES,
    ChunkNode,
    DocumentNode,
    EntityRecord,
    GraphBuildConfig,
    GraphBuildResult,
    GRAPH_SCHEMA_VERSION,
    RelationRecord,
    SectionGraphNode,
    entity_node_id,
    normalize_entity_name,
    serialize_graph,
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    if hasattr(value, "__dict__"):
        return _canonical(vars(value))
    return str(value)


def _record(record: EntityRecord | RelationRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    return record.__dict__.copy()


def _entity_parts(name: str, entity_type: str) -> tuple[str, str]:
    normalized_type = entity_type if entity_type in ENTITY_TYPES else "Other"
    return entity_node_id(name, normalized_type), normalized_type


def _put_entity(
    graph: nx.MultiDiGraph,
    name: str,
    entity_type: str,
    aliases: Sequence[str] = (),
    extraction_version: str | None = None,
) -> str:
    node_id, canonical_type = _entity_parts(name, entity_type)
    canonical_name = " ".join(name.split())
    if graph.has_node(node_id):
        attrs = graph.nodes[node_id]
        merged = set(attrs.get("aliases", ()))
        merged.update(str(alias) for alias in aliases if alias)
        if canonical_name.casefold() != node_id.rsplit(":", 1)[-1]:
            merged.add(canonical_name)
        attrs["aliases"] = sorted(merged, key=str.casefold)
        versions = set(attrs.get("extraction_versions", ()))
        if extraction_version:
            versions.add(extraction_version)
        attrs["extraction_versions"] = sorted(versions)
    else:
        alias_set = {str(alias) for alias in aliases if alias}
        if canonical_name.casefold() != node_id.rsplit(":", 1)[-1]:
            alias_set.add(canonical_name)
        graph.add_node(
            node_id,
            type="Entity",
            canonical_name=canonical_name,
            entity_type=canonical_type,
            aliases=sorted(alias_set, key=str.casefold),
            extraction_versions=[extraction_version] if extraction_version else [],
        )
    return node_id


def build_graph(
    chunks: Sequence[Mapping[str, Any]],
    structures: Mapping[str, Any],
    *,
    entities: Sequence[EntityRecord | Mapping[str, Any]] = (),
    relations: Sequence[RelationRecord | Mapping[str, Any]] = (),
    config: GraphBuildConfig | None = None,
) -> GraphBuildResult:
    """Строит узлы и рёбра графа из детерминированных артефактов ingestion-слоя.

    Структуры ожидаются как ``DocumentStructure`` (или их словари); чанки
    используют текущий словарный контракт К2. Необязательные записи сущностей
    требуют ``name``, ``entity_type``/``type`` и ``chunk_id``. Связи требуют
    subject/object и валидные ``chunk_id`` доказательств. Экстрактор и LLM не
    вызываются.
    """
    config = config or GraphBuildConfig()
    graph = nx.MultiDiGraph()
    chunks_by_id: dict[str, Mapping[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        chunk_id, doc_id = chunk.get("chunk_id"), chunk.get("doc_id")
        if not chunk_id or not doc_id:
            raise ValueError("Every chunk must contain non-empty chunk_id and doc_id.")
        if str(chunk_id) in chunks_by_id:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        chunks_by_id[str(chunk_id)] = chunk
        doc_id = str(doc_id)
        doc_data = documents.setdefault(doc_id, {})
        for field in ("year", "report_period", "issuer", "country", "title", "source_sha256"):
            if field in chunk and chunk[field] is not None:
                previous = doc_data.get(field)
                if previous is not None and previous != chunk[field]:
                    raise ValueError(f"Inconsistent {field} metadata for doc_id {doc_id}.")
                doc_data[field] = chunk[field]

    section_nodes: dict[str, dict[str, Any]] = {}
    section_ids_by_doc: dict[str, dict[str, str]] = {}
    for doc_id, structure in sorted(structures.items()):
        documents.setdefault(str(doc_id), {})
        sections = _read(structure, "sections", ()) or ()
        by_stable_id = {str(_read(section, "id", "")): section for section in sections}
        section_ids_by_doc[str(doc_id)] = {}
        for section in sections:
            stable_id = str(_read(section, "id", ""))
            if not stable_id:
                raise ValueError(f"Section in {doc_id} has no stable id.")
            breadcrumbs = _read(section, "breadcrumbs", ()) or ()
            title = str(_read(section, "title", ""))
            path = "/".join(str(item) for item in breadcrumbs) or title
            node = SectionGraphNode(
                doc_id=str(doc_id),
                stable_section_id=stable_id,
                title=title,
                level=int(_read(section, "level", 0)),
                section_path=path,
                page_start=int(_read(section, "page_start", 0)),
                page_end=int(_read(section, "page_end", 0)),
            )
            section_nodes[node.node_id] = {
                "type": "Section",
                "doc_id": node.doc_id,
                "stable_section_id": node.stable_section_id,
                "title": node.title,
                "level": node.level,
                "section_path": node.section_path,
                "page_start": node.page_start,
                "page_end": node.page_end,
            }
            section_ids_by_doc[str(doc_id)][stable_id] = node.node_id

    for doc_id, metadata in sorted(documents.items()):
        node = DocumentNode(
            doc_id=doc_id,
            year=_as_optional_int(metadata.get("year")),
            report_period=_optional_str(metadata.get("report_period")),
            issuer=_optional_str(metadata.get("issuer")),
            country=_optional_str(metadata.get("country")),
            title=_optional_str(metadata.get("title")),
            source_sha256=_optional_str(metadata.get("source_sha256")),
        )
        graph.add_node(
            node.node_id, type="Document", doc_id=node.doc_id, year=node.year,
            report_period=node.report_period, issuer=node.issuer,
            country=node.country, title=node.title, source_sha256=node.source_sha256,
        )
    for node_id, attrs in sorted(section_nodes.items()):
        graph.add_node(node_id, **attrs)

    # Иерархия структуры берётся из переданных артефактов разделов.
    for doc_id, structure in sorted(structures.items()):
        sections = _read(structure, "sections", ()) or ()
        for section in sections:
            stable_id = str(_read(section, "id", ""))
            target = section_ids_by_doc[str(doc_id)][stable_id]
            parent_id = _read(section, "parent_id")
            if parent_id and str(parent_id) in section_ids_by_doc[str(doc_id)]:
                graph.add_edge(
                    section_ids_by_doc[str(doc_id)][str(parent_id)], target,
                    key="has_child", type="HAS_CHILD", source="structure",
                    confidence=1.0, extraction_version=None,
                )
            else:
                graph.add_edge(
                    f"doc:{doc_id}", target, key="has_section",
                    type="HAS_SECTION", source="structure", confidence=1.0,
                    extraction_version=None,
                )

    # Привязываем чанк к разделу по точному пути, иначе — по тому же правилу
    # стартовой страницы, что и обогатитель чанков в ingestion.
    section_by_path: dict[str, dict[str, str]] = {}
    for node_id, attrs in section_nodes.items():
        section_by_path.setdefault(attrs["doc_id"], {})[attrs["section_path"]] = node_id
    for chunk_id, chunk in sorted(chunks_by_id.items()):
        doc_id = str(chunk["doc_id"])
        section_path = chunk.get("section_path")
        section_id = section_by_path.get(doc_id, {}).get(str(section_path)) if section_path else None
        if section_id is None and doc_id in structures and chunk.get("page_pdf") is not None:
            candidates = [
                (attrs["page_start"], attrs["level"], node_id)
                for node_id, attrs in section_nodes.items()
                if attrs["doc_id"] == doc_id and attrs["page_start"] <= int(chunk["page_pdf"])
            ]
            if candidates:
                section_id = max(candidates)[2]
        chunk_node = ChunkNode(
            chunk_id=chunk_id,
            doc_id=doc_id,
            year=_as_optional_int(chunk.get("year")),
            issuer=_optional_str(chunk.get("issuer")),
            kind=_optional_str(chunk.get("kind")),
            page_pdf=_as_optional_int(chunk.get("page_pdf")),
            section_path=_optional_str(section_path),
            evidence_id=_optional_str(chunk.get("evidence_id")),
            text=str(chunk.get("text") or ""),
        )
        graph.add_node(
            chunk_node.node_id, type="Chunk", chunk_id=chunk_node.chunk_id,
            doc_id=chunk_node.doc_id, year=chunk_node.year, issuer=chunk_node.issuer,
            kind=chunk_node.kind, page_pdf=chunk_node.page_pdf,
            section_path=chunk_node.section_path, evidence_id=chunk_node.evidence_id,
            text=chunk_node.text,
        )
        if section_id is not None:
            graph.add_edge(
                section_id, chunk_node.node_id, key="contains", type="CONTAINS",
                source="structure", confidence=1.0, extraction_version=None,
            )

    for raw_entity in entities:
        item = _record(raw_entity)
        name = str(item.get("name", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not name or chunk_id not in chunks_by_id:
            raise ValueError("Entity records need a name and an existing chunk_id.")
        entity_type = str(item.get("entity_type", item.get("type", "Other")))
        entity_id = _put_entity(
            graph, name, entity_type, item.get("aliases", ()) or (),
            item.get("extraction_version"),
        )
        confidence = item.get("confidence")
        graph.add_edge(
            f"chunk:{chunk_id}", entity_id,
            key=f"mentions:{chunk_id}", type="MENTIONS",
            source=item.get("source", "pre_extracted"),
            confidence=confidence,
            extraction_version=item.get("extraction_version"),
            evidence_chunk_ids=[chunk_id],
        )

    for raw_relation in relations:
        item = _record(raw_relation)
        subject, obj = str(item.get("subject", "")).strip(), str(item.get("object", "")).strip()
        evidence = item.get("evidence_chunk_ids", ()) or ()
        evidence_ids = sorted({str(value) for value in evidence})
        if not subject or not obj or not evidence_ids:
            # Связь без явного доказательства небезопасно обходить.
            continue
        missing = set(evidence_ids) - chunks_by_id.keys()
        if missing:
            raise ValueError(f"Relation references unknown evidence chunks: {sorted(missing)}")
        predicate = str(item.get("predicate", "RELATED_TO"))
        if predicate != "RELATED_TO":
            predicate = "RELATED_TO"
        subject_id = _resolve_relation_entity(
            graph, subject, str(item.get("subject_type", "Other")),
            item.get("extraction_version"),
        )
        object_id = _resolve_relation_entity(
            graph, obj, str(item.get("object_type", "Other")),
            item.get("extraction_version"),
        )
        key = f"{predicate}:{','.join(evidence_ids)}"
        graph.add_edge(
            subject_id, object_id, key=key, type=predicate,
            source=item.get("source", "pre_extracted"),
            confidence=item.get("confidence"),
            extraction_version=item.get("extraction_version"),
            evidence_chunk_ids=evidence_ids,
        )

    content = serialize_graph(graph)
    fingerprint_payload = {
        "schema_version": config.schema_version,
        "config": _canonical(config.__dict__),
        "chunks": _canonical(sorted(chunks, key=lambda item: str(item.get("chunk_id", "")))),
        "structures": _canonical(structures),
        "entities": _canonical(entities),
        "relations": _canonical(relations),
        "graph": content,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "graph_schema_version": config.schema_version,
        "graph_fingerprint": fingerprint,
        "doc_ids": sorted(documents),
        "chunk_count": len(chunks_by_id),
        "entity_count": sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "Entity"),
        "relation_count": sum(1 for _, _, data in graph.edges(data=True)
                              if data.get("type") == "RELATED_TO"),
        "extractor_name": config.extractor_name,
        "extractor_version": config.extractor_version,
        "prompt_version": config.prompt_version,
        "failed_chunk_ids": [],
        "skipped_chunk_ids": [],
        "status": "complete",
    }
    return GraphBuildResult(graph=graph, graph_fingerprint=fingerprint, manifest=manifest)


def _as_optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _resolve_relation_entity(
    graph: nx.MultiDiGraph,
    name: str,
    entity_type: str,
    extraction_version: str | None,
) -> str:
    """Переиспользует сущность с тем же именем/псевдонимом.

    Сначала берётся единственная сущность с совпадающим ``entity_type``, затем
    единственная по имени; при неоднозначности создаётся новая сущность.
    """
    normalized = normalize_entity_name(name)
    matching = [
        node_id for node_id, attrs in graph.nodes(data=True)
        if attrs.get("type") == "Entity"
        and (
            str(attrs.get("canonical_name", "")).casefold() == normalized
            or normalized in {str(alias).casefold() for alias in attrs.get("aliases", ())}
        )
    ]
    typed = [node_id for node_id in matching
             if graph.nodes[node_id].get("entity_type") == entity_type]
    if len(typed) == 1:
        return typed[0]
    if len(matching) == 1:
        return matching[0]
    return _put_entity(graph, name, entity_type,
                       extraction_version=extraction_version)


def graph_to_dict(result: GraphBuildResult) -> dict[str, Any]:
    """Возвращает детерминированное JSON-сериализуемое содержимое артефакта графа."""
    return serialize_graph(result.graph, result.graph_fingerprint, result.manifest)


def save_graph(result: GraphBuildResult, path: str | Path) -> Path:
    """Атомарно записывает артефакт графа в UTF-8 JSON.

    Родительский каталог создаётся при необходимости. Сериализация завершается
    во временном файле рядом, и только потом ``os.replace`` публикует артефакт,
    поэтому неудачная запись не оставляет частичный целевой файл.
    """
    target = Path(path)
    atomic_write_json(
        target,
        graph_to_dict(result),
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )
    return target


def load_graph(path: str | Path) -> GraphBuildResult:
    """Загружает и проверяет JSON-артефакт графа в ``MultiDiGraph``.

    Сериализованный ``provenance_source`` восстанавливается в атрибут ребра
    ``source``. ID доказательств нормализуются в отсортированные уникальные
    строки и должны указывать на существующие узлы-чанки.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict):
        raise ValueError("Graph artifact must be a JSON object.")
    if artifact.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported graph schema version: {artifact.get('graph_schema_version')!r}"
        )
    nodes = artifact.get("nodes")
    edges = artifact.get("edges")
    fingerprint = artifact.get("graph_fingerprint")
    manifest = artifact.get("manifest", {})
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Graph artifact nodes and edges must be lists.")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Graph artifact requires a non-empty graph_fingerprint.")
    if not isinstance(manifest, dict):
        raise ValueError("Graph artifact manifest must be an object.")

    graph = nx.MultiDiGraph()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each graph node must be an object.")
        node_id = node.get("node_id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not node_id or not isinstance(node_type, str):
            raise ValueError("Graph nodes require string node_id and type.")
        if node_id in graph:
            raise ValueError(f"Duplicate graph node ID: {node_id}")
        attrs = {key: value for key, value in node.items() if key != "node_id"}
        graph.add_node(node_id, **attrs)

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each graph edge must be an object.")
        source, target = edge.get("source"), edge.get("target")
        key, edge_type = edge.get("key"), edge.get("type")
        if not all(isinstance(value, str) and value for value in
                   (source, target, key, edge_type)):
            raise ValueError("Graph edges require string source, target, key, and type.")
        if source not in graph or target not in graph:
            raise ValueError(f"Dangling graph edge endpoint: {source!r} -> {target!r}")
        attrs = {
            attr_key: value for attr_key, value in edge.items()
            if attr_key not in {"source", "target", "key"}
        }
        if "provenance_source" in attrs:
            attrs["source"] = attrs.pop("provenance_source")
        if "evidence_chunk_ids" in attrs:
            evidence = attrs["evidence_chunk_ids"]
            if not isinstance(evidence, list):
                raise ValueError("evidence_chunk_ids must be a list.")
            evidence_ids = sorted({str(value) for value in evidence})
            if any(
                f"chunk:{chunk_id}" not in graph
                or graph.nodes[f"chunk:{chunk_id}"].get("type") != "Chunk"
                for chunk_id in evidence_ids
            ):
                raise ValueError("Graph edge contains dangling evidence chunk IDs.")
            attrs["evidence_chunk_ids"] = evidence_ids
        elif edge_type in {"MENTIONS", "RELATED_TO"}:
            raise ValueError(f"{edge_type} edge requires evidence_chunk_ids.")
        if edge_type == "RELATED_TO" and not attrs.get("evidence_chunk_ids"):
            raise ValueError("RELATED_TO edge requires non-empty evidence_chunk_ids.")
        graph.add_edge(source, target, key=key, **attrs)

    return GraphBuildResult(
        graph=graph,
        graph_fingerprint=fingerprint,
        manifest=manifest,
    )
