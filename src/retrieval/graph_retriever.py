"""Ограниченный графовый поиск, возвращающий ``chunk_id`` источников и пути свидетельств."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import networkx as nx

from src.graph.graph_models import GraphRetrievalResult, GraphView


def resolve_entity_ids(
    graph: nx.MultiDiGraph,
    entity_names: Iterable[str],
    entity_types: Iterable[str] | None = None,
) -> list[str]:
    """Разрешает имена и псевдонимы в отсортированные уникальные ID узлов-сущностей.

    Пробелы схлопываются, регистр игнорируется. Если задан ``entity_types``,
    совпадают только сущности с одним из этих точных канонических типов.
    Неоднозначные псевдонимы намеренно возвращают все совпавшие ID стабильно.
    """
    names = [entity_names] if isinstance(entity_names, str) else entity_names
    if isinstance(entity_types, str):
        allowed_types = {entity_types}
    else:
        allowed_types = set(entity_types) if entity_types is not None else None
    normalized_names = {
        " ".join(str(name).split()).casefold()
        for name in names
        if str(name).strip()
    }
    matches = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "Entity":
            continue
        if allowed_types is not None and attrs.get("entity_type") not in allowed_types:
            continue
        names = [attrs.get("canonical_name", ""), *attrs.get("aliases", ())]
        normalized_node_names = {
            " ".join(str(name).split()).casefold()
            for name in names
            if str(name).strip()
        }
        if normalized_names.intersection(normalized_node_names):
            matches.append(str(node_id))
    return sorted(matches)


def retrieve_graph(
    graph: nx.MultiDiGraph,
    entity_ids: Iterable[str],
    *,
    doc_ids: set[str] | None = None,
    years: set[int] | None = None,
    issuers: set[str] | None = None,
    section_path: str | None = None,
    max_depth: int = 2,
    max_results: int = 10,
    include_relation_evidence: bool = True,
) -> list[GraphRetrievalResult]:
    """Возвращает подходящие чанки, достижимые от переданных ID сущностей.

    Рёбра связей обходятся только при наличии существующих ID чанков-доказательств.
    Глубина ограничена ``max_depth``; фильтры метаданных и разделов применяются
    до ранжирования. Оценки детерминированы и зависят от глубины; они
    предназначены для последующего слияния/реранжирования.

    Args:
        include_relation_evidence: При ``True`` чанки, указанные в пройденных
            рёбрах ``RELATED_TO``, тоже возвращаются как кандидаты. При ``False``
            возвращаются только чанки, напрямую упоминающие стартовые сущности;
            связи по-прежнему ограничивают обход, но не могут подмешать
            свидетельства о соседних сущностях. Вызывающим, которым важна
            точность (graph-augmented retrieval), следует использовать ``False``.
    """
    if max_depth < 0 or max_results < 0:
        raise ValueError("max_depth and max_results must be non-negative.")
    starts = sorted({node_id for node_id in entity_ids
                     if node_id in graph and graph.nodes[node_id].get("type") == "Entity"})
    if max_results == 0 or not starts:
        return []

    candidates: dict[str, dict[str, Any]] = {}

    # Храним лучший детерминированный путь к каждой сущности. RELATED_TO
    # проходим в обе стороны, но путь сохраняет реальное направление рёбер
    # для объяснения.
    reached: dict[str, tuple[int, tuple[str, ...], tuple[str, ...], str]] = {}
    queue = deque((node_id, 0, (node_id,), ()) for node_id in starts)
    while queue:
        node_id, depth, path, evidence = queue.popleft()
        previous = reached.get(node_id)
        if previous is not None and previous[0] <= depth:
            continue
        reached[node_id] = (depth, path, evidence, path[0])
        if depth >= max_depth:
            continue
        neighbors: list[tuple[str, tuple[str, ...]]] = []
        for source, target, attrs in graph.in_edges(node_id, data=True):
            if attrs.get("type") != "RELATED_TO":
                continue
            proof = _eligible_evidence(
                graph, attrs.get("evidence_chunk_ids", ()),
                doc_ids, years, issuers, section_path,
            )
            if proof:
                if include_relation_evidence:
                    _add_relation_candidates(
                        candidates, graph, node_id, source, proof, evidence, path,
                        depth + 1,
                    )
                neighbors.append((source, proof))
        for source, target, attrs in graph.out_edges(node_id, data=True):
            if attrs.get("type") != "RELATED_TO":
                continue
            proof = _eligible_evidence(
                graph, attrs.get("evidence_chunk_ids", ()),
                doc_ids, years, issuers, section_path,
            )
            if proof:
                if include_relation_evidence:
                    _add_relation_candidates(
                        candidates, graph, node_id, target, proof, evidence, path,
                        depth + 1,
                    )
                neighbors.append((target, proof))
        for neighbor, proof in sorted(neighbors, key=lambda item: (item[0], item[1])):
            if neighbor not in path:
                queue.append((neighbor, depth + 1, path + (neighbor,),
                              tuple(sorted(set(evidence).union(proof)))))

    for entity_id, (depth, entity_path, relation_evidence, start_id) in reached.items():
        if not include_relation_evidence and entity_id not in starts:
            # Режим точности: не цитируем чанки соседних сущностей.
            continue
        for chunk_node, _, attrs in graph.in_edges(entity_id, data=True):
            edge_attrs = attrs
            if edge_attrs.get("type") != "MENTIONS":
                continue
            chunk_attrs = graph.nodes.get(chunk_node, {})
            if chunk_attrs.get("type") != "Chunk":
                continue
            chunk_id = str(chunk_attrs.get("chunk_id", ""))
            if not chunk_id or not _eligible(
                graph, chunk_node, chunk_attrs, doc_ids, years, issuers, section_path
            ):
                continue
            direct_proof = _valid_evidence(graph, edge_attrs.get("evidence_chunk_ids", ()))
            all_evidence = tuple(sorted(set(relation_evidence).union(direct_proof)))
            # Свидетельства включают сам возвращаемый чанк — так у кандидата
            # всегда есть источник, готовый для цитаты.
            all_evidence = tuple(sorted(set(all_evidence).union({chunk_id})))
            candidate_path = entity_path + (chunk_node,)
            candidate = candidates.setdefault(chunk_id, {
                "score": 0.0,
                "matched_entities": set(),
                "path": candidate_path,
                "evidence": set(),
                "rank": (depth, candidate_path),
            })
            candidate["score"] = max(candidate["score"], 1.0 / (depth + 1))
            candidate["matched_entities"].add(start_id)
            candidate["evidence"].update(all_evidence)
            if (depth, candidate_path) < candidate["rank"]:
                candidate["rank"] = (depth, candidate_path)
                candidate["path"] = candidate_path

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1]["score"], item[1]["rank"], item[0]),
    )
    return [
        GraphRetrievalResult(
            chunk_id=chunk_id,
            graph_score=float(data["score"]),
            matched_entities=tuple(sorted(data["matched_entities"])),
            path=tuple(data["path"]),
            evidence_chunk_ids=tuple(sorted(data["evidence"])),
        )
        for chunk_id, data in ranked[:max_results]
    ]


def structural_neighbors(
    graph: nx.MultiDiGraph,
    chunk_id: str,
    *,
    budget: int = 2,
    doc_ids: set[str] | None = None,
    years: set[int] | None = None,
    issuers: set[str] | None = None,
    section_path: str | None = None,
) -> list[str]:
    """Соседние чанки по структуре: раздел (``CONTAINS``) и страница.

    Возвращает до ``budget`` ``chunk_id`` того же документа, упорядоченных по
    расстоянию в страницах от исходного чанка. Раздел даёт соседей даже на
    соседних страницах, страница — чанки той же страницы, которых нет в разделе
    (врезки, таблицы). Собственный чанк и чанки, не прошедшие фильтры, не
    попадают в результат; при равном расстоянии порядок определяется ``chunk_id``,
    поэтому результат детерминирован.
    """
    if budget <= 0:
        return []
    node_id = f"chunk:{chunk_id}"
    if node_id not in graph:
        return []
    seed_attrs = graph.nodes[node_id]
    if seed_attrs.get("type") != "Chunk":
        return []
    doc_id = str(seed_attrs.get("doc_id") or "")
    seed_page = _page(seed_attrs)

    ordered: dict[str, tuple[int, str]] = {}
    for parent in sorted(graph.predecessors(node_id)):
        if graph.nodes[parent].get("type") != "Section":
            continue
        for _, target, attrs in graph.out_edges(parent, data=True):
            if attrs.get("type") != "CONTAINS" or target == node_id:
                continue
            _remember_neighbor(
                ordered, graph, target, seed_page,
                doc_ids, years, issuers, section_path,
            )
    for candidate in graph.nodes:
        if candidate == node_id:
            continue
        if not _same_page(graph, candidate, doc_id, seed_page):
            continue
        _remember_neighbor(
            ordered, graph, candidate, seed_page,
            doc_ids, years, issuers, section_path,
        )
    ranked = sorted(ordered.items(), key=lambda item: item[1])
    return [candidate_id for candidate_id, _ in ranked[:budget]]


def _remember_neighbor(
    ordered: dict[str, tuple[int, str]],
    graph: nx.MultiDiGraph,
    candidate: str,
    seed_page: int | None,
    doc_ids: set[str] | None,
    years: set[int] | None,
    issuers: set[str] | None,
    section_path: str | None,
) -> None:
    """Запоминает кандидата, если он существует, подходит и ближе прежнего."""
    attrs = graph.nodes[candidate]
    if attrs.get("type") != "Chunk":
        return
    candidate_id = str(attrs.get("chunk_id") or "")
    if not candidate_id:
        return
    if not _eligible(graph, candidate, attrs, doc_ids, years, issuers, section_path):
        return
    candidate_page = _page(attrs)
    if candidate_page is None or seed_page is None:
        distance = 1 << 30
    else:
        distance = abs(candidate_page - seed_page)
    previous = ordered.get(candidate_id)
    key = (distance, candidate_id)
    if previous is None or key < previous:
        ordered[candidate_id] = key


def _page(attrs: dict[str, Any]) -> int | None:
    """Номер страницы чанка как ``int`` (``None``, если метаданных нет)."""
    value = attrs.get("page_pdf")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _same_page(
    graph: nx.MultiDiGraph, candidate: str, doc_id: str, seed_page: int | None
) -> bool:
    """Совпадает ли чанк-кандидат с сидом по документу и странице."""
    if seed_page is None:
        return False
    attrs = graph.nodes[candidate]
    if attrs.get("type") != "Chunk":
        return False
    if str(attrs.get("doc_id") or "") != doc_id:
        return False
    return _page(attrs) == seed_page


def _eligible_evidence(
    graph: nx.MultiDiGraph,
    evidence: Iterable[Any],
    doc_ids: set[str] | None,
    years: set[int] | None,
    issuers: set[str] | None,
    section_path: str | None,
) -> tuple[str, ...]:
    """Оставляет только чанки-доказательства, которые существуют и проходят фильтры поиска."""
    valid = []
    for chunk_id in _valid_evidence(graph, evidence):
        node_id = f"chunk:{chunk_id}"
        if _eligible(
            graph, node_id, graph.nodes[node_id],
            doc_ids, years, issuers, section_path,
        ):
            valid.append(chunk_id)
    return tuple(valid)


def _add_relation_candidates(
    candidates: dict[str, dict[str, Any]],
    graph: nx.MultiDiGraph,
    current_entity: str,
    related_entity: str,
    evidence_ids: tuple[str, ...],
    prior_evidence: tuple[str, ...],
    entity_path: tuple[str, ...],
    relation_depth: int,
) -> None:
    """Добавляет чанки, указанные в пройденной связи, даже без рёбер MENTIONS."""
    for chunk_id in evidence_ids:
        chunk_node = f"chunk:{chunk_id}"
        if graph.nodes[chunk_node].get("type") != "Chunk":
            continue
        candidate_path = entity_path + (related_entity, chunk_node)
        proof = set(prior_evidence).union(evidence_ids)
        proof.add(chunk_id)
        candidate = candidates.setdefault(chunk_id, {
            "score": 0.0,
            "matched_entities": set(),
            "path": candidate_path,
            "evidence": set(),
            "rank": (relation_depth, candidate_path),
        })
        candidate["score"] = max(
            candidate["score"], 1.0 / (relation_depth + 1),
        )
        candidate["matched_entities"].add(entity_path[0])
        candidate["evidence"].update(proof)
        if (relation_depth, candidate_path) < candidate["rank"]:
            candidate["rank"] = (relation_depth, candidate_path)
            candidate["path"] = candidate_path


def _valid_evidence(graph: nx.MultiDiGraph, evidence: Iterable[Any]) -> tuple[str, ...]:
    valid = set()
    for chunk_id in evidence:
        chunk_id = str(chunk_id)
        node_id = f"chunk:{chunk_id}"
        if node_id in graph and graph.nodes[node_id].get("type") == "Chunk":
            valid.add(chunk_id)
    return tuple(sorted(valid))


def _eligible(
    graph: nx.MultiDiGraph,
    chunk_node: str,
    attrs: dict[str, Any],
    doc_ids: set[str] | None,
    years: set[int] | None,
    issuers: set[str] | None,
    section_path: str | None,
) -> bool:
    if doc_ids is not None and attrs.get("doc_id") not in doc_ids:
        return False
    if years is not None and attrs.get("year") not in years:
        return False
    if issuers is not None and attrs.get("issuer") not in issuers:
        return False
    if section_path is not None:
        paths = [
            graph.nodes[parent].get("section_path", "")
            for parent in graph.predecessors(chunk_node)
            if graph.nodes[parent].get("type") == "Section"
        ]
        wanted = section_path.rstrip("/")
        if not any(path == wanted or path.startswith(wanted + "/") for path in paths):
            return False
    return True


def get_graph_subgraph(
    graph: nx.MultiDiGraph,
    *,
    doc_ids: set[str] | None = None,
    section_path: str | None = None,
    entity_ids: set[str] | None = None,
    max_nodes: int = 200,
) -> GraphView:
    """Возвращает ограниченный вид графа с фильтрами по документу, разделу и сущности."""
    if max_nodes < 0:
        raise ValueError("max_nodes must be non-negative.")
    eligible: set[str] = set()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type")
        if node_type == "Document":
            if doc_ids is None or attrs.get("doc_id") in doc_ids:
                eligible.add(node_id)
        elif node_type == "Section":
            if (doc_ids is None or attrs.get("doc_id") in doc_ids) and (
                section_path is None or attrs.get("section_path") == section_path
                or str(attrs.get("section_path", "")).startswith(section_path.rstrip("/") + "/")
            ):
                eligible.add(node_id)
        elif node_type == "Chunk":
            if (doc_ids is None or attrs.get("doc_id") in doc_ids) and (
                section_path is None or _eligible(
                    graph, node_id, attrs, doc_ids, None, None, section_path
                )
            ):
                eligible.add(node_id)
        elif node_type == "Entity" and entity_ids is not None and node_id in entity_ids:
            eligible.add(node_id)

    # Виды вокруг сущности включают напрямую связанные свидетельства и
    # структурных предков; сортируем перед жёстким ограничением числа узлов.
    selected = set(eligible)
    for node_id in sorted(eligible):
        for neighbor in graph.predecessors(node_id):
            if neighbor in eligible:
                selected.add(neighbor)
        for neighbor in graph.successors(node_id):
            if neighbor in eligible:
                selected.add(neighbor)
    selected = set(sorted(selected)[:max_nodes])
    nodes = [
        {"node_id": str(node_id), **_stable_attrs(graph.nodes[node_id])}
        for node_id in sorted(selected)
    ]
    edges = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        if source in selected and target in selected:
            edge_attrs = _stable_attrs(attrs)
            if "source" in edge_attrs:
                edge_attrs["provenance_source"] = edge_attrs.pop("source")
            edges.append({
                **edge_attrs, "source": str(source), "target": str(target),
                "key": str(key),
            })
    edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["key"]))
    return GraphView(nodes=nodes, edges=edges)


def _stable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Копирует атрибуты графа в JSON-безопасные детерминированные значения."""
    result = {}
    for key, value in sorted(attrs.items()):
        if isinstance(value, (set, frozenset, tuple)):
            value = sorted(value, key=repr)
        result[key] = value
    return result
