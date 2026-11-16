"""Тесты на фикстурах для сборки, сериализации и поиска по графу."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from src.graph.graph_builder import build_graph, graph_to_dict, load_graph, save_graph
from src.graph.graph_models import EntityRecord, GraphBuildConfig, RelationRecord
from src.ingestion.structure_builder import DocumentStructure, SectionNode
from src.retrieval.graph_retriever import (
    get_graph_subgraph,
    resolve_entity_ids,
    retrieve_graph,
    structural_neighbors,
)


def fixtures():
    structure = DocumentStructure(
        doc_id="report_2023",
        sections=[
            SectionNode("Risk", "Risks", 0, 1, 10, ["Risks"]),
            SectionNode("Risk/Credit", "Credit risk", 1, 2, 5,
                        ["Risks", "Credit risk"], "Risk"),
        ],
    )
    chunks = [
        {
            "chunk_id": "c-risk", "doc_id": "report_2023", "year": 2023,
            "report_period": "2023", "issuer": "Example Bank", "country": "RU",
            "title": "Annual report", "source_sha256": "abc", "kind": "text",
            "page_pdf": 3, "section_path": "Risks/Credit risk",
            "evidence_id": "page-3", "text": "Credit risk evidence.",
        },
        {
            "chunk_id": "c-other", "doc_id": "report_2022", "year": 2022,
            "issuer": "Example Bank", "kind": "table", "page_pdf": 8,
            "text": "Other evidence.",
        },
    ]
    return chunks, {"report_2023": structure}


def test_builds_deterministic_structure_and_chunk_edges():
    chunks, structures = fixtures()
    first = build_graph(chunks, structures)
    second = build_graph(list(reversed(chunks)), structures)

    assert isinstance(first.graph, nx.MultiDiGraph)
    assert first.graph_fingerprint == second.graph_fingerprint
    graph = first.graph
    assert "doc:report_2023" in graph
    assert "section:report_2023:Risk/Credit" in graph
    assert "chunk:c-risk" in graph
    assert graph.has_edge("doc:report_2023", "section:report_2023:Risk")
    assert graph.has_edge("section:report_2023:Risk", "section:report_2023:Risk/Credit")
    assert graph.has_edge("section:report_2023:Risk/Credit", "chunk:c-risk")
    assert all(source in graph and target in graph
               for source, target in graph.edges())


def test_entity_relations_keep_provenance_and_retrieve_filtered_chunk():
    chunks, structures = fixtures()
    result = build_graph(
        chunks,
        structures,
        entities=[
            EntityRecord("Credit Risk", "Risk", "c-risk", ("credit risk",), "v1"),
            EntityRecord("Example Bank", "Organization", "c-risk", (), "v1"),
        ],
        relations=[
            RelationRecord("Credit Risk", "Example Bank", ("c-risk",),
                           extraction_version="v1"),
        ],
        config=GraphBuildConfig(extractor_name="fixture", extractor_version="v1"),
    )
    risk_id = "entity:Risk:credit risk"
    retrieved = retrieve_graph(
        result.graph, {risk_id}, doc_ids={"report_2023"}, years={2023},
        issuers={"Example Bank"}, section_path="Risks", max_depth=1,
    )

    assert [item.chunk_id for item in retrieved] == ["c-risk"]
    assert "c-risk" in retrieved[0].evidence_chunk_ids
    assert retrieved[0].path[0] == risk_id
    relation = next(data for _, _, data in result.graph.edges(data=True)
                    if data.get("type") == "RELATED_TO")
    assert relation["evidence_chunk_ids"] == ["c-risk"]
    assert retrieve_graph(result.graph, {risk_id}, doc_ids={"report_2022"}) == []


def test_resolve_entity_ids_matches_names_aliases_and_types():
    chunks, structures = fixtures()
    result = build_graph(
        chunks,
        structures,
        entities=[
            EntityRecord("Credit Risk", "Risk", "c-risk", ("credit exposure",)),
            EntityRecord("Credit Risk", "FinancialMetric", "c-risk", ("risk score",)),
        ],
    )

    assert resolve_entity_ids(result.graph, ["  CREDIT   EXPOSURE "]) == [
        "entity:Risk:credit risk"
    ]
    assert resolve_entity_ids(
        result.graph, ["credit risk"], entity_types={"FinancialMetric"}
    ) == ["entity:FinancialMetric:credit risk"]
    assert resolve_entity_ids(result.graph, ["not present"]) == []


def test_related_to_evidence_is_retrievable_without_mentions_edge():
    chunks, structures = fixtures()
    result = build_graph(
        chunks,
        structures,
        entities=[EntityRecord("Seed entity", "Organization", "c-other")],
        relations=[RelationRecord(
            "Seed entity", "Unmentioned entity", ("c-risk",),
            subject_type="Organization", object_type="Organization",
            extraction_version="fixture-v1",
        )],
    )
    seed = resolve_entity_ids(result.graph, ["seed entity"])

    # c-risk подтверждается ребром связи, но не имеет ребра MENTIONS.
    assert not any(
        data.get("type") == "MENTIONS"
        and source == "chunk:c-risk"
        for source, _, data in result.graph.edges(data=True)
    )
    retrieved = retrieve_graph(
        result.graph, seed, doc_ids={"report_2023"}, years={2023},
        issuers={"Example Bank"}, section_path="Risks", max_depth=1,
    )
    assert [item.chunk_id for item in retrieved] == ["c-risk"]
    assert "c-risk" in retrieved[0].evidence_chunk_ids
    assert retrieved[0].path[0] == seed[0]
    assert retrieved[0].path[-1] == "chunk:c-risk"
    assert [item.chunk_id for item in retrieve_graph(
        result.graph, seed, doc_ids={"report_2022"}, max_depth=1,
    )] == ["c-other"]


def test_records_without_relation_evidence_are_not_added_or_traversed():
    chunks, structures = fixtures()
    result = build_graph(
        chunks, structures,
        entities=[EntityRecord("Risk", "Risk", "c-risk")],
        relations=[{"subject": "Risk", "object": "Strategy", "evidence_chunk_ids": []}],
    )
    assert not any(data.get("type") == "RELATED_TO"
                   for _, _, data in result.graph.edges(data=True))
    assert retrieve_graph(result.graph, {"entity:Risk:risk"}, max_depth=2)


def test_serialization_is_stable_json_and_graph_view_is_bounded():
    chunks, structures = fixtures()
    result = build_graph(chunks, structures)
    artifact = graph_to_dict(result)

    assert json.dumps(artifact, ensure_ascii=False, sort_keys=True)
    assert artifact == result.to_dict()
    view = get_graph_subgraph(result.graph, doc_ids={"report_2023"}, max_nodes=3)
    assert len(view.nodes) <= 3
    node_ids = {node["node_id"] for node in view.nodes}
    assert all(edge["source"] in node_ids and edge["target"] in node_ids
               for edge in view.edges)


def test_graph_artifact_roundtrip_normalizes_edge_provenance(tmp_path):
    chunks, structures = fixtures()
    result = build_graph(
        chunks,
        structures,
        entities=[EntityRecord("Risk", "Risk", "c-risk", extraction_version="v1")],
        relations=[RelationRecord(
            "Risk", "Bank", ("c-risk",), subject_type="Risk",
            object_type="Organization", source="fixture", extraction_version="v1",
        )],
    )
    target = tmp_path / "nested" / "graph.json"
    assert save_graph(result, target) == target
    loaded = load_graph(target)

    assert loaded.graph_fingerprint == result.graph_fingerprint
    assert loaded.manifest == result.manifest
    assert graph_to_dict(loaded) == graph_to_dict(result)
    relation = next(data for _, _, data in loaded.graph.edges(data=True)
                    if data.get("type") == "RELATED_TO")
    assert relation["source"] == "fixture"
    assert relation["evidence_chunk_ids"] == ["c-risk"]


def test_failed_atomic_graph_save_preserves_existing_target(tmp_path, monkeypatch):
    chunks, structures = fixtures()
    result = build_graph(chunks, structures)
    target = tmp_path / "graph.json"
    target.write_text("previous artifact\n", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("src.core.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_graph(result, target)

    assert target.read_text(encoding="utf-8") == "previous artifact\n"
    assert list(tmp_path.glob(".graph.json.*.tmp")) == []


def test_graph_load_rejects_schema_mismatch_and_dangling_edges(tmp_path):
    chunks, structures = fixtures()
    artifact = graph_to_dict(build_graph(chunks, structures))
    target = tmp_path / "graph.json"

    artifact["graph_schema_version"] = "future"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        load_graph(target)

    artifact["graph_schema_version"] = "1.0"
    artifact["edges"] = [{
        "source": "doc:report_2023", "target": "missing",
        "key": "bad", "type": "HAS_SECTION",
    }]
    target.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="Dangling"):
        load_graph(target)


def structural_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "section:report_2023:Risk/Credit", type="Section", doc_id="report_2023",
        section_path="Risks/Credit risk", level=1, title="Credit risk",
    )
    for chunk_id, page in (("c-a", 3), ("c-b", 3), ("c-c", 4), ("c-d", 9)):
        graph.add_node(
            f"chunk:{chunk_id}", type="Chunk", chunk_id=chunk_id,
            doc_id="report_2023", year=2023, issuer="Example Bank",
            kind="text", page_pdf=page,
        )
    for chunk_id in ("c-a", "c-b", "c-c"):
        graph.add_edge(
            "section:report_2023:Risk/Credit", f"chunk:{chunk_id}",
            key="contains", type="CONTAINS",
        )
    return graph


def test_structural_neighbors_prefer_same_page_then_section_order():
    graph = structural_graph()

    assert structural_neighbors(graph, "c-a", budget=1) == ["c-b"]
    assert structural_neighbors(graph, "c-a", budget=2) == ["c-b", "c-c"]
    # Сосед с другой страницы вне раздела не подмешивается.
    assert "c-d" not in structural_neighbors(graph, "c-a", budget=3)


def test_structural_neighbors_apply_filters_and_ignore_unknown_chunk():
    graph = structural_graph()

    assert structural_neighbors(graph, "c-a", budget=2, years={2022}) == []
    assert structural_neighbors(graph, "c-a", budget=2, doc_ids={"other"}) == []
    assert structural_neighbors(graph, "missing", budget=2) == []
    assert structural_neighbors(graph, "c-a", budget=0) == []
    # Фильтр по разделу отсекает чанки, у которых другой section_path.
    assert structural_neighbors(
        graph, "c-a", budget=2, section_path="Risks/Credit risk",
    ) == ["c-b", "c-c"]
    assert structural_neighbors(
        graph, "c-a", budget=2, section_path="Other",
    ) == []
