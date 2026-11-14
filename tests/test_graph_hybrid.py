"""Офлайн-юнит-тесты обёртки graph-augmented поиска."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import networkx as nx
import pytest

from src.graph.graph_models import GraphRetrievalResult
from src.retrieval import graph_hybrid
from src.retrieval.graph_hybrid import (
    GraphAugmentedRetriever,
    GraphRetrievalTrace,
    default_entity_matcher,
)
from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)

ENTITY_ID = "entity:Risk:credit risk"


def chunk(
    chunk_id: str,
    *,
    doc_id: str = "d1",
    year: int = 2023,
    issuer: str = "Bank",
    kind: str = "text",
    report_period: str | None = None,
    section_path: str | None = "Risks",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "year": year,
        "issuer": issuer,
        "kind": kind,
        "report_period": report_period or str(year),
        "section_path": section_path,
        "text": f"text for {chunk_id}",
    }


def graph_fixture(*, mention_ids: Sequence[str] = ("c1", "c2")) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for chunk_id in ("c1", "c2", "c3", "c4"):
        graph.add_node(
            f"chunk:{chunk_id}", type="Chunk", chunk_id=chunk_id,
            doc_id="d1", year=2023, issuer="Bank", kind="text",
        )
    graph.add_node(
        ENTITY_ID,
        type="Entity",
        canonical_name="Credit Risk",
        entity_type="Risk",
        aliases=["Кредитный   риск"],
    )
    for chunk_id in mention_ids:
        graph.add_edge(
            f"chunk:{chunk_id}", ENTITY_ID,
            key=f"mentions:{chunk_id}", type="MENTIONS",
            evidence_chunk_ids=[chunk_id],
        )
    return graph


def store(*chunk_ids: str) -> dict[str, dict[str, Any]]:
    return {chunk_id: chunk(chunk_id) for chunk_id in chunk_ids}


def hit(chunk_id: str, *, score: float = 0.02, **kwargs: Any) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        chunk=chunk(chunk_id),
        rrf_score=score,
        rerank_score=kwargs.get("rerank_score"),
        rerank_rank=kwargs.get("rerank_rank"),
    )


class FakeBase:
    def __init__(
        self,
        hits: Sequence[RetrievalHit],
        *,
        warnings: Sequence[str] = (),
        trace: RetrievalTrace | None = None,
    ) -> None:
        self.hits = list(hits)
        self.warnings = list(warnings)
        self.trace = trace or RetrievalTrace(fused_candidates=len(self.hits))
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        selected = self.hits if top_k is None else self.hits[:top_k]
        return RetrievalResult(
            hits=list(selected),
            warnings=list(self.warnings),
            effective_filters=filters or SearchFilters(),
            trace=self.trace,
        )


class BrokenGraph:
    def nodes(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("graph is corrupt")


def test_default_entity_matcher_normalizes_case_and_whitespace():
    graph = graph_fixture()

    assert default_entity_matcher("оцени  КРЕДИТНЫЙ\tриск", graph) == [ENTITY_ID]
    assert default_entity_matcher("CREDIT    RISK outlook", graph) == [ENTITY_ID]
    assert default_entity_matcher("unrelated question", graph) == []
    assert default_entity_matcher("", graph) == []


def test_base_only_when_graph_is_empty():
    base = FakeBase([hit("c1"), hit("c2")], warnings=["base warning"])
    retriever = GraphAugmentedRetriever(base, nx.MultiDiGraph(), store("c1", "c2"))

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.hits[0].rrf_score == 0.02
    assert result.warnings[0] == "base warning"
    assert "graph_candidates=0" in result.warnings
    assert "graph_matched_entities=0" in result.warnings
    assert isinstance(result.trace, GraphRetrievalTrace)
    assert result.trace.graph_candidates == 0
    assert result.trace.graph_added == 0
    assert result.trace.graph_failed is False


def test_graph_only_chunks_are_appended_after_base_hits():
    base = FakeBase([hit("c1", score=0.031)])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"), graph_weight=0.5,
    )

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    base_hit, graph_hit = result.hits
    assert base_hit.rrf_score == 0.031
    assert base_hit.dense_rank is None
    assert graph_hit.chunk["text"] == "text for c2"
    assert graph_hit.rerank_score is None
    assert graph_hit.rerank_rank is None
    assert graph_hit.rrf_score == pytest.approx(0.5 / 62)
    assert result.trace.graph_candidates == 2
    assert result.trace.graph_added == 1
    assert "graph_added=1" in result.warnings
    assert result.trace.graph_results[0]["chunk_id"] in {"c1", "c2"}
    assert result.trace.graph_matched_entities == [ENTITY_ID]


def test_dedupe_keeps_base_position_and_orders_graph_only_by_score(monkeypatch):
    base = FakeBase([hit("c1", score=0.04), hit("c3", score=0.02)])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2", "c3", "c4"),
    )

    def fake_retrieve_graph(graph, entity_ids, **kwargs):
        return [
            GraphRetrievalResult("c2", 0.4, (), (), ()),
            GraphRetrievalResult("c4", 0.9, (), (), ()),
            GraphRetrievalResult("c1", 0.9, (), (), ()),
            GraphRetrievalResult("c3", 0.8, (), (), ()),
        ]

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", fake_retrieve_graph)

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c3", "c4", "c2"]
    assert len({item.chunk_id for item in result.hits}) == 4
    assert result.hits[0].rrf_score == 0.04
    assert result.hits[1].rrf_score == 0.02
    assert result.trace.graph_duplicates == 2
    assert result.trace.graph_added == 2
    assert "graph_duplicates=2" in result.warnings


def test_top_k_is_applied_after_merge():
    base = FakeBase([hit("c1"), hit("c3")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2", "c3"))

    result = retriever.retrieve("credit risk", top_k=3)
    assert [item.chunk_id for item in result.hits] == ["c1", "c3", "c2"]
    assert result.trace.graph_added == 1
    assert base.calls[-1]["top_k"] == 3

    result = retriever.retrieve("credit risk", top_k=1)
    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_added == 0

    result = retriever.retrieve("credit risk", top_k=None)
    assert [item.chunk_id for item in result.hits] == ["c1", "c3", "c2"]


def test_filters_are_passed_to_base_and_graph(monkeypatch):
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))
    calls: list[dict[str, Any]] = []

    def spy(graph, entity_ids, **kwargs):
        calls.append({"graph": graph, "entity_ids": entity_ids, **kwargs})
        return []

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", spy)
    filters = SearchFilters(
        years={2023},
        issuers={"Bank"},
        kinds={"text"},
        section_path_prefix="Risks",
        allowed_doc_ids={"d1"},
    )

    result = retriever.retrieve("credit risk", filters=filters, top_k=5)

    assert base.calls == [{"query": "credit risk", "filters": filters, "top_k": 5}]
    assert calls[0]["doc_ids"] == {"d1"}
    assert calls[0]["years"] == {2023}
    assert calls[0]["issuers"] == {"Bank"}
    assert calls[0]["section_path"] == "Risks"
    assert calls[0]["max_depth"] == 2
    assert calls[0]["max_results"] == 10
    assert result.effective_filters == filters


def test_year_range_filters_expand_to_graph_years(monkeypatch):
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))
    calls: list[dict[str, Any]] = []

    def spy(graph, entity_ids, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", spy)
    retriever.retrieve("credit risk", filters=SearchFilters(year_from=2021, year_to=2023))

    assert calls[0]["years"] == {2021, 2022, 2023}


def test_explicit_filters_win_over_forgetful_base(monkeypatch):
    class ForgetfulBase:
        def retrieve(self, query: str, *, filters: Any = None, top_k: Any = None):
            # Заглушка базы игнорирует фильтры и возвращает результат по умолчанию.
            return RetrievalResult(hits=[hit("c1")])

    retriever = GraphAugmentedRetriever(
        ForgetfulBase(), graph_fixture(), store("c1", "c2"))
    calls: list[dict[str, Any]] = []

    def spy(graph, entity_ids, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", spy)
    retriever.retrieve("credit risk", filters=SearchFilters(years={2022}))

    assert calls[0]["years"] == {2022}


def test_no_entities_does_not_call_graph_retrieval(monkeypatch):
    base = FakeBase([hit("c1")])

    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("graph retrieval must not run without entities")

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", fail)
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    result = retriever.retrieve("совершенно другой вопрос")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_candidates == 0
    assert result.trace.graph_added == 0


def test_broken_graph_falls_back_to_base_with_warning():
    base = FakeBase([hit("c1")], warnings=["base-only warning"])
    retriever = GraphAugmentedRetriever(base, BrokenGraph(), store("c1"))

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.hits[0].rrf_score == 0.02
    assert "base-only warning" in result.warnings
    assert "graph_candidates=0" in result.warnings
    assert any("Graph retrieval unavailable" in warning for warning in result.warnings)
    assert result.trace.graph_failed is True
    assert result.trace.graph_error_type == "RuntimeError"


def test_missing_chunk_store_entry_is_skipped():
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1"))

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_candidates == 2
    assert result.trace.graph_skipped_missing_chunks == 1
    assert result.trace.graph_added == 0
    assert "graph_skipped_missing_chunks=1" in result.warnings


def test_graph_candidate_must_pass_kind_filter():
    base = FakeBase([hit("c1")])
    chunks = store("c1", "c2")
    chunks["c2"]["kind"] = "table"
    retriever = GraphAugmentedRetriever(base, graph_fixture(), chunks)

    result = retriever.retrieve(
        "credit risk", filters=SearchFilters(kinds={"text"}),
    )

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_skipped_filtered == 1
    assert "graph_skipped_filtered=1" in result.warnings


@pytest.mark.parametrize("matcher", [
    lambda query: [ENTITY_ID],
    lambda query, graph: [ENTITY_ID],
    lambda query: ["Credit Risk"],
])
def test_injectable_entity_matcher_accepts_ids_or_names(matcher):
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"), entity_matcher=matcher,
    )

    result = retriever.retrieve("query without a textual mention")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.trace.graph_matched_entities == [ENTITY_ID]
    assert result.trace.graph_added == 1


def test_constructor_rejects_invalid_limits():
    base = FakeBase([hit("c1")])
    for kwargs in (
        {"graph_depth": -1},
        {"graph_max_results": -1},
        {"graph_weight": -0.1},
        {"graph_weight": float("nan")},
    ):
        with pytest.raises(ValueError):
            GraphAugmentedRetriever(base, graph_fixture(), store("c1"), **kwargs)


def test_accepts_mapping_hits_from_a_lightweight_base():
    class MappingBase:
        def retrieve(self, query: str, *, filters: Any = None, top_k: Any = None):
            return {
                "hits": [{
                    "chunk_id": "c1",
                    "chunk": chunk("c1"),
                    "rrf_score": 0.05,
                }],
                "warnings": ["mapping warning"],
                "effective_filters": filters,
            }

    retriever = GraphAugmentedRetriever(
        MappingBase(), graph_fixture(), store("c1", "c2"),
    )
    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.hits[0].rrf_score == 0.05
    assert result.warnings[0] == "mapping warning"

def test_default_matcher_ignores_generic_aliases_and_prefers_longest_name():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "entity:Organization:pao sberbank", type="Entity",
        canonical_name="пао сбербанк", entity_type="Organization",
        aliases=["Банк", "Сбербанк"],
    )
    graph.add_node(
        "entity:Organization:sber", type="Entity",
        canonical_name="сбер", entity_type="Organization",
        aliases=["Сбер"],
    )

    assert default_entity_matcher("Что говорится о банке?", graph) == []
    assert default_entity_matcher("Какая стратегия у Сбербанка?", graph) == [
        "entity:Organization:pao sberbank"
    ]
    assert default_entity_matcher("Что известно о Сбере?", graph) == [
        "entity:Organization:sber"
    ]

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pymorphy3") is None,
    reason="pymorphy3 is required for lemma matching",
)
def test_default_matcher_handles_russian_inflections():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "entity:FinancialMetric:key rate", type="Entity",
        canonical_name="ключевая ставка", entity_type="FinancialMetric",
    )
    graph.add_node(
        "entity:Strategy:strategy 2026", type="Entity",
        canonical_name="стратегия 2026", entity_type="Strategy",
    )

    assert default_entity_matcher("Что говорится о ключевой ставке?", graph) == [
        "entity:FinancialMetric:key rate"
    ]
    assert default_entity_matcher("Что известно о стратегии 2026?", graph) == [
        "entity:Strategy:strategy 2026"
    ]

class FakeReranker:
    def __init__(self, scores: dict[str, int], *, used_fallback: bool = False):
        self.scores = scores
        self.used_fallback = used_fallback
        self.warning = None
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, hits: Any):
        from src.retrieval.reranker import RerankResult

        ids = [item["chunk_id"] for item in hits]
        self.calls.append((query, ids))
        ordered = sorted(
            ids, key=lambda chunk_id: (-self.scores.get(chunk_id, 0), ids.index(chunk_id))
        )
        return RerankResult(
            hits=[
                {
                    "chunk_id": chunk_id,
                    "text": "",
                    "rerank_score": self.scores.get(chunk_id, 0),
                    "rerank_rank": rank,
                }
                for rank, chunk_id in enumerate(ordered, start=1)
            ],
            used_fallback=self.used_fallback,
            attempts=1,
            warning=self.warning,
        )


def test_reranker_lets_graph_evidence_displace_a_base_hit():
    base = FakeBase([hit("c1", score=0.9)])
    reranker = FakeReranker({"c1": 1, "c2": 3})
    retriever = GraphAugmentedRetriever(
        base,
        graph_fixture(mention_ids=("c2",)),
        store("c1", "c2"),
        reranker=reranker,
    )

    result = retriever.retrieve("credit risk", top_k=1)

    assert [item.chunk_id for item in result.hits] == ["c2"]
    assert result.hits[0].rerank_score == 3
    assert result.trace.graph_added == 1
    assert result.trace.reranker_status == "used"
    assert reranker.calls[0][1] == ["c1", "c2"]


def test_relevance_threshold_keeps_important_hits_and_may_return_fewer():
    base = FakeBase([hit("c1"), hit("c2")])
    reranker = FakeReranker({"c1": 3, "c2": 1, "c3": 2})
    retriever = GraphAugmentedRetriever(
        base,
        graph_fixture(mention_ids=("c3",)),
        store("c1", "c2", "c3"),
        reranker=reranker,
        relevance_threshold=2,
    )

    result = retriever.retrieve("credit risk", top_k=10)

    assert [item.chunk_id for item in result.hits] == ["c1", "c3"]
    assert all(item.rerank_score >= 2 for item in result.hits)
    assert result.trace.graph_added == 1


def test_relevance_threshold_requires_a_reranker():
    with pytest.raises(ValueError):
        GraphAugmentedRetriever(
            FakeBase([]),
            graph_fixture(),
            store(),
            relevance_threshold=2,
        )

def test_importance_mode_widens_base_pool_and_interleaves_graph():
    base_hits = [hit(f"b{index}", score=0.9 - index / 100) for index in range(8)]
    base = FakeBase(base_hits)
    graph = graph_fixture(mention_ids=("c1", "c2", "c3"))
    reranker = FakeReranker({"c1": 3, "c2": 0, "c3": 2})
    retriever = GraphAugmentedRetriever(
        base,
        graph,
        store(*[f"b{index}" for index in range(8)], "c1", "c2", "c3"),
        reranker=reranker,
        max_candidates=6,
        base_candidate_k=8,
    )

    result = retriever.retrieve("credit risk", top_k=3)

    # У базы запрошен широкий пул, а не три хита.
    assert base.calls[0]["top_k"] == 8
    sent = reranker.calls[0][1]
    assert len(sent) == 6
    assert "c1" in sent and "c2" in sent
    assert [item.chunk_id for item in result.hits][0] == "c1"
    assert result.hits[0].rerank_score == 3
