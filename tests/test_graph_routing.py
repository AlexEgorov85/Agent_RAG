"""Маршрутизация graph-этапа, fallback и provenance свидетельств (P7, G1–G5).

Проверяется backend-контракт без интеграции в factory/tools:

* отсутствие индекса, режим ``dense``/``keyword``, сломанный backend и
  отсутствие кандидатов дают базовый результат с явной причиной в
  ``trace.unavailable["graph"]`` и предупреждением ``graph_unavailable``;
* успешный graph-поиск публикует ``stages["graph"]`` с путём и
  ``evidence_chunk_ids``;
* ``evidence_manifest`` содержит только подтверждённые primary чанки, а
  graph-only чанк ждёт ``get_chunk`` (P7, AC-P7-3);
* ``graph_added=0`` — валидный результат, а не ошибка.
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx
import pytest

from src.graph.graph_models import GraphRetrievalResult
from src.agent.tools import AgentTools
from src.retrieval import graph_hybrid
from src.retrieval.graph_hybrid import (
    GraphAugmentedRetriever,
    GraphRetrievalTrace,
)
from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)

ENTITY_ID = "entity:Risk:credit risk"


def chunk(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "year": 2023,
        "issuer": "Bank",
        "kind": "text",
        "report_period": "2023",
        "section_path": "Risks",
        "text": f"text for {chunk_id}",
    }
    data.update(overrides)
    return data


def graph_fixture(*, mention_ids: Sequence[str] = ("c1", "c2")) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for chunk_id in ("c1", "c2", "c3"):
        graph.add_node(
            f"chunk:{chunk_id}", type="Chunk", chunk_id=chunk_id,
            doc_id="d1", year=2023, issuer="Bank", kind="text",
        )
    graph.add_node(
        ENTITY_ID,
        type="Entity",
        canonical_name="Credit Risk",
        entity_type="Risk",
        aliases=["Кредитный риск"],
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


def hit(chunk_id: str, *, score: float = 0.02) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, chunk=chunk(chunk_id), rrf_score=score)


class FakeBase:
    """Минимальная база без ``retrieval_mode``/``queries`` в сигнатуре."""

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


class ModeAwareBase(FakeBase):
    """База, поддерживающая новые аргументы ``search_reports``."""

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        self.calls.append({
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "queries": queries,
        })
        selected = self.hits if top_k is None else self.hits[:top_k]
        return RetrievalResult(
            hits=list(selected),
            warnings=list(self.warnings),
            effective_filters=filters or SearchFilters(),
            trace=RetrievalTrace(
                retrieval_mode=retrieval_mode,
                query_count=1 + len(queries or []) if queries else 1,
            ),
        )


class BrokenGraph:
    def nodes(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("graph is corrupt")


def _stage_ids(result: RetrievalResult) -> list[str]:
    stage = result.trace.stages.get("graph")
    assert stage is not None, "graph stage must be measured"
    return [item["chunk_id"] for item in stage]


def test_missing_graph_index_returns_base_without_failing():
    base = FakeBase([hit("c1"), hit("c2")], warnings=["base warning"])
    retriever = GraphAugmentedRetriever(base, None, {})

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.warnings[0] == "base warning"
    assert any("graph_unavailable" in warning for warning in result.warnings)
    assert result.trace.graph_failed is False
    assert result.trace.stages["graph"] is None
    assert result.trace.available["graph"] is False
    assert result.trace.unavailable["graph"] == "no_index"
    assert result.trace.fallback["graph"] is True
    assert result.trace.graph_candidates == 0
    assert result.trace.stages["graph"] is None


def test_empty_graph_is_no_index_not_a_failure():
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, nx.MultiDiGraph(), store("c1"))

    result = retriever.retrieve("credit risk")

    assert result.trace.unavailable["graph"] == "no_index"
    assert result.trace.graph_failed is False
    assert result.trace.graph_candidates == 0
    assert result.trace.fallback["graph"] is True


def test_broken_graph_backend_falls_back_to_base_result(monkeypatch):
    base = FakeBase([hit("c1")], warnings=["base-only warning"])
    retriever = GraphAugmentedRetriever(base, BrokenGraph(), store("c1"))

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert "base-only warning" in result.warnings
    assert result.trace.graph_failed is True
    assert result.trace.graph_error_type == "RuntimeError"
    assert result.trace.unavailable["graph"] == "graph_failed"
    assert result.trace.available["graph"] is False
    assert result.trace.fallback["graph"] is True
    assert any("graph_unavailable: graph_failed" in warning
               for warning in result.warnings)


def test_graph_retrieval_error_does_not_leak_candidates(monkeypatch):
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    def explode(*args: Any, **kwargs: Any):
        raise ValueError("graph backend exploded")

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", explode)
    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_failed is True
    assert result.trace.graph_candidates == 0
    assert result.trace.stages["graph"] is None
    assert result.trace.unavailable["graph"] == "graph_failed"
    assert retriever.evidence_manifest(result) == ["c1"]
    assert retriever.graph_evidence_manifest(result) == []


def test_successful_graph_search_publishes_stage_with_provenance():
    base = FakeBase([hit("c1", score=0.04)])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"), graph_weight=0.5,
    )

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.trace.available["graph"] is True
    assert "graph" not in result.trace.unavailable
    assert result.trace.fallback["graph"] is False

    stage = result.trace.stages["graph"]
    assert stage is not None
    assert [entry["chunk_id"] for entry in stage] == ["c1", "c2"]
    assert [entry["rank"] for entry in stage] == [1, 2]
    assert [entry["score"] for entry in stage] == [1.0, 1.0]
    for entry in stage:
        assert entry["path"][0] == ENTITY_ID
        assert entry["chunk_id"] in entry["evidence_chunk_ids"]
        assert entry["matched_entities"] == [ENTITY_ID]

    serialized = result.trace.to_dict()
    assert serialized["stages"]["graph"] == stage
    assert serialized["counts"]["graph"] == 2
    assert serialized["graph"]["added"] == 1
    assert serialized["graph"]["added_chunk_ids"] == ["c2"]
    assert serialized["graph"]["matched_entities"] == [ENTITY_ID]
    # Финальный top-k этапа „returned“ включает graph-добавленный чанк.
    assert [item["chunk_id"] for item in (result.trace.stages["returned"] or [])] == [
        "c1", "c2",
    ]
    assert result.trace.available["returned"] is True
    assert result.trace.distinct_docs == 1


def test_duplicate_evidence_collapses_by_chunk_id(monkeypatch):
    base = FakeBase([hit("c1"), hit("c1", score=0.01)])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"),
    )

    monkeypatch.setattr(
        graph_hybrid,
        "retrieve_graph",
        lambda *args, **kwargs: [
            GraphRetrievalResult("c2", 0.4, (ENTITY_ID,), ("e", "chunk:c2"), ("c2",)),
            GraphRetrievalResult("c2", 0.9, (ENTITY_ID,), ("e", "chunk:c2"), ("c2",)),
            GraphRetrievalResult("c1", 0.9, (ENTITY_ID,), ("e", "chunk:c1"), ("c1",)),
        ],
    )

    result = retriever.retrieve("credit risk")

    ids = [item.chunk_id for item in result.hits]
    assert ids == ["c1", "c2"]
    assert len(ids) == len(set(ids))
    assert result.trace.graph_duplicates == 1
    assert result.trace.graph_added == 1
    assert result.trace.graph_added_chunk_ids == ["c2"]
    # Дубликаты схлопнуты и в опубликованном этапе.
    assert _stage_ids(result) == ["c2", "c1"]
    assert retriever.evidence_manifest(result) == ["c1"]
    assert retriever.graph_evidence_manifest(result) == ["c2"]


def test_graph_only_chunk_stays_out_of_evidence_manifest_until_confirmed():
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    result = retriever.retrieve("credit risk")

    # Graph-кандидат возвращается агенту, но не считается подтверждённым.
    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert retriever.evidence_manifest(result) == ["c1"]
    assert retriever.graph_evidence_manifest(result) == ["c2"]
    graph_hit = retriever.graph_evidence_hits(result)[0]
    assert graph_hit.chunk["text"] == "text for c2"

    # После получения через поиск/get_chunk чанк становится primary.
    confirmed = GraphAugmentedRetriever(
        FakeBase([hit("c1"), hit("c2")]), graph_fixture(), store("c1", "c2"),
    ).retrieve("credit risk")
    assert confirmed.trace.graph_added == 0
    assert retriever.evidence_manifest(confirmed) == ["c1", "c2"]
    assert retriever.graph_evidence_manifest(confirmed) == []


def test_graph_added_zero_is_valid_result():
    # Все графовые кандидаты уже пришли из базового поиска.
    base = FakeBase([hit("c1", score=0.05), hit("c2", score=0.04)])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert result.trace.graph_candidates == 2
    assert result.trace.graph_duplicates == 2
    assert result.trace.graph_added == 0
    assert result.trace.graph_added_chunk_ids == []
    assert result.trace.graph_failed is False
    assert result.trace.available["graph"] is True
    assert result.trace.stages["graph"] is not None
    assert "graph_added=0" in result.warnings
    # Ноль добавленных — не повод помечать весь graph-этап недоступным.
    assert "graph" not in result.trace.unavailable
    assert result.trace.fallback["graph"] is False
    assert retriever.evidence_manifest(result) == ["c1", "c2"]


@pytest.mark.parametrize("mode", ["dense", "keyword"])
def test_dense_and_keyword_modes_disable_graph(mode, monkeypatch):
    base = ModeAwareBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("graph must not run outside hybrid mode")

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", fail)
    result = retriever.retrieve("credit risk", retrieval_mode=mode)

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert base.calls[0]["retrieval_mode"] == mode
    assert result.trace.retrieval_mode == mode
    assert result.trace.stages["graph"] is None
    assert result.trace.available["graph"] is False
    assert result.trace.unavailable["graph"] == "mode_not_supported"
    assert result.trace.fallback["graph"] is True
    assert any("mode_not_supported" in warning for warning in result.warnings)
    assert retriever.evidence_manifest(result) == ["c1"]


def test_graph_is_disabled_by_configuration(monkeypatch):
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"), graph_max_results=0,
    )

    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("disabled graph must not run")

    monkeypatch.setattr(graph_hybrid, "retrieve_graph", fail)
    result = retriever.retrieve("credit risk")

    assert result.trace.unavailable["graph"] == "disabled"
    assert result.trace.graph_failed is False
    assert any("disabled" in warning for warning in result.warnings)


def test_queries_are_forwarded_to_capable_base():
    base = ModeAwareBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    retriever.retrieve(
        "credit risk",
        retrieval_mode="hybrid",
        queries=["какие кредитные риски", "credit risk"],
    )

    assert base.calls[0]["retrieval_mode"] == "hybrid"
    assert base.calls[0]["queries"] == ["какие кредитные риски", "credit risk"]


def test_invalid_retrieval_mode_is_rejected():
    retriever = GraphAugmentedRetriever(
        FakeBase([hit("c1")]), graph_fixture(), store("c1"))

    with pytest.raises(ValueError):
        retriever.retrieve("credit risk", retrieval_mode="semantic")


def test_entity_matcher_failure_is_a_graph_fallback():
    def matcher(*args: Any, **kwargs: Any):
        raise RuntimeError("matcher failed")

    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(
        base, graph_fixture(), store("c1", "c2"), entity_matcher=matcher,
    )

    result = retriever.retrieve("credit risk")

    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert result.trace.graph_failed is True
    assert result.trace.unavailable["graph"] == "graph_failed"
    assert result.trace.fallback["graph"] is True


def test_trace_does_not_mutate_reused_base_trace():
    base = FakeBase([hit("c1")])
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    first = retriever.retrieve("credit risk")
    second = retriever.retrieve("совершенно другой вопрос")

    assert first.trace.available["graph"] is True
    # Второй вызов не унаследовал graph-этап первого через общий base.trace.
    assert second.trace.available["graph"] is False
    assert second.trace.stages["graph"] is None
    assert base.trace.stages == {}


def test_graph_trace_is_still_a_retrieval_trace():
    base = FakeBase([hit("c1")], trace=RetrievalTrace(fused_candidates=3))
    retriever = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))

    result = retriever.retrieve("credit risk")

    assert isinstance(result.trace, RetrievalTrace)
    assert isinstance(result.trace, GraphRetrievalTrace)
    assert result.trace.fused_candidates == 3


# --- AgentTools: маршрутизация use_graph (P7) -----------------------------


class RecordingBackend:
    """Backend, запоминающий точные аргументы вызова."""

    def __init__(self, hits: Sequence[RetrievalHit] | None = None) -> None:
        self.hits = list(hits if hits is not None else [hit("c1")])
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        self.calls.append({
            "query": query,
            "filters": filters,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "queries": queries,
        })
        return RetrievalResult(
            hits=list(self.hits),
            effective_filters=filters or SearchFilters(),
            trace=RetrievalTrace(retrieval_mode=retrieval_mode),
        )


def test_use_graph_routes_to_graph_backend_without_rewriting_query() -> None:
    plain = RecordingBackend([hit("c1")])
    graph_backend = RecordingBackend([hit("c1"), hit("c2")])
    tools = AgentTools(
        search_backend=plain, graph_search_backend=graph_backend, top_k=3
    )

    result = tools.invoke(
        "search_reports",
        {
            "query": "  credit risk  ",
            "filters": SearchFilters(),
            "retrieval_mode": "hybrid",
            "top_k": 10,
            "queries": ["credit risk", "кредитный риск"],
            "use_graph": True,
        },
    )

    assert plain.calls == []
    assert len(graph_backend.calls) == 1
    call = graph_backend.calls[0]
    # Инструмент только валидирует и передаёт: запрос не переписывается.
    assert call["query"] == "  credit risk  "
    assert call["top_k"] == 10
    assert call["queries"] == ["credit risk", "кредитный риск"]
    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    assert "graph_unavailable" not in result.warnings


def test_use_graph_false_routes_to_plain_backend() -> None:
    plain = RecordingBackend([hit("c1")])
    graph_backend = RecordingBackend([hit("c2")])
    tools = AgentTools(search_backend=plain, graph_search_backend=graph_backend)

    result = tools.invoke(
        "search_reports",
        {
            "query": "credit risk",
            "filters": SearchFilters(),
            "use_graph": False,
        },
    )

    assert graph_backend.calls == []
    assert len(plain.calls) == 1
    assert [item.chunk_id for item in result.hits] == ["c1"]


def test_missing_graph_backend_falls_back_with_warning() -> None:
    plain = RecordingBackend([hit("c1")])
    tools = AgentTools(search_backend=plain)

    result = tools.invoke(
        "search_reports",
        {"query": "credit risk", "filters": SearchFilters(), "use_graph": True},
    )

    assert len(plain.calls) == 1
    assert isinstance(result, RetrievalResult)
    assert result.warnings == ["graph_unavailable"]
    assert [item.chunk_id for item in result.hits] == ["c1"]


def test_graph_fallback_preserves_dict_result_shape() -> None:
    backend = lambda query, **kwargs: {"hits": [hit("c1")], "warnings": ["base"]}
    tools = AgentTools(search_backend=backend)

    result = tools.invoke(
        "search_reports",
        {"query": "credit risk", "filters": SearchFilters(), "use_graph": True},
    )

    assert result["warnings"] == ["base", "graph_unavailable"]
    assert result["hits"][0].chunk_id == "c1"


def test_invalid_use_graph_is_rejected() -> None:
    tools = AgentTools(search_backend=RecordingBackend())

    with pytest.raises(ValueError):
        tools.invoke(
            "search_reports",
            {"query": "q", "filters": SearchFilters(), "use_graph": "yes"},
        )


def test_backend_without_mode_raises_explicit_error() -> None:
    called: list[dict[str, Any]] = []

    def legacy_backend(query: str, *, filters: SearchFilters, top_k: int):
        called.append({"query": query, "filters": filters, "top_k": top_k})
        return RetrievalResult(
            hits=[hit("c1")], effective_filters=filters
        )

    tools = AgentTools(search_backend=legacy_backend)

    with pytest.raises(ValueError, match="backend_does_not_support_mode"):
        tools.invoke(
            "search_reports",
            {
                "query": "credit risk",
                "filters": SearchFilters(),
                "retrieval_mode": "dense",
            },
        )
    # Молчаливого hybrid не произошло: backend не вызывался.
    assert called == []

    result = tools.invoke(
        "search_reports",
        {"query": "credit risk", "filters": SearchFilters()},
    )
    assert [item.chunk_id for item in result.hits] == ["c1"]
    assert called[0]["top_k"] == tools.top_k


def test_graph_only_chunk_is_not_primary_evidence_through_tools() -> None:
    base = FakeBase([hit("c1")])
    graph_backend = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))
    tools = AgentTools(
        search_backend=base, graph_search_backend=graph_backend
    )

    result = tools.invoke(
        "search_reports",
        {"query": "credit risk", "filters": SearchFilters(), "use_graph": True},
    )

    assert [item.chunk_id for item in result.hits] == ["c1", "c2"]
    # Primary evidence — только подтверждённое базой; graph-only ждёт get_chunk.
    assert graph_backend.evidence_manifest(result) == ["c1"]
    assert graph_backend.graph_evidence_manifest(result) == ["c2"]
    assert result.trace.available["graph"] is True
    assert result.trace.graph_added == 1


# --- AgentService: state-level primary/graph evidence (P7) ------------------


class SequenceSelector:
    """Выдаёт заранее заданные действия и затем завершает цикл."""

    def __init__(self, actions: Sequence[Any]) -> None:
        self.actions = list(actions)

    def choose(self, state: Any) -> Any:
        return self.actions.pop(0) if self.actions else None


class EvidenceAnswerer:
    """Цитирует всё, что попало в primary evidence."""

    def generate(self, question: str, evidence: Any, evidence_manifest: Any, **_: Any) -> Any:
        from src.agent.state import Answer, Citation

        citations = [
            Citation(
                chunk_id,
                doc_id=(chunk or {}).get("doc_id"),
                year=(chunk or {}).get("year"),
            )
            for chunk_id, chunk in sorted(evidence.items())
        ]
        return Answer(
            "answer" if citations else "abstain",
            "Ответ по подтверждённым свидетельствам.",
            citations=citations,
            warnings=[] if citations else ["no_evidence"],
        )


def graph_service(actions: Sequence[Any], **tools_kwargs: Any) -> Any:
    from src.agent.service import AgentService

    base = FakeBase([hit("c1")])
    graph_backend = GraphAugmentedRetriever(base, graph_fixture(), store("c1", "c2"))
    tools = AgentTools(
        search_backend=base,
        graph_search_backend=graph_backend,
        **tools_kwargs,
    )
    return AgentService(
        tools,
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector(actions),
    )


def search_action(**overrides: Any) -> Any:
    from src.agent.state import AgentAction

    arguments: dict[str, Any] = {"query": "credit risk", "use_graph": True}
    arguments.update(overrides)
    return AgentAction("search_reports", arguments, "fact_lookup")


def test_graph_only_chunk_stays_out_of_primary_manifest_in_agent_state() -> None:
    service = graph_service([search_action()])

    result = service.ask("credit risk", conversation_id="graph-only")

    # Graph-only c2 пришёл в результате, но не стал primary и не цитируется.
    assert [event.tool_name for event in result.tool_events] == ["search_reports"]
    assert result.tool_events[0].result_count == 2
    assert result.diagnostics["evidence_manifest"] == ["c1"]
    assert result.diagnostics["graph_evidence_manifest"] == ["c2"]
    assert [item.chunk_id for item in result.answer.citations] == ["c1"]


def test_get_chunk_confirms_graph_only_chunk_and_makes_it_primary() -> None:
    from src.agent.state import AgentAction

    service = graph_service(
        [
            search_action(),
            AgentAction("get_chunk", {"chunk_id": "c2"}, "inspect_evidence"),
        ],
        chunk_backend=lambda chunk_id: chunk(chunk_id),
    )

    result = service.ask("credit risk", conversation_id="graph-confirm")

    assert [event.tool_name for event in result.tool_events] == [
        "search_reports", "get_chunk",
    ]
    assert result.diagnostics["evidence_manifest"] == ["c1", "c2"]
    assert [item.chunk_id for item in result.answer.citations] == ["c1", "c2"]


def test_duplicate_graph_and_primary_chunk_ids_are_not_duplicated() -> None:
    def backend(query: str, *, filters: SearchFilters, **_: Any) -> dict[str, Any]:
        return {
            "hits": [
                hit("c1"),
                hit("c1"),
                {"chunk_id": "c2", "chunk": chunk("c2")},
            ],
            "trace": {"graph_added_chunk_ids": ["c2", "c2"]},
        }

    from src.agent.service import AgentService

    service = AgentService(
        AgentTools(search_backend=backend),
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector([search_action(use_graph=False)]),
    )

    result = service.ask("credit risk", conversation_id="graph-duplicates")

    assert result.diagnostics["evidence_manifest"] == ["c1"]
    assert result.diagnostics["graph_evidence_manifest"] == ["c2"]


def test_search_without_graph_added_ids_keeps_previous_behaviour() -> None:
    def backend(query: str, *, filters: SearchFilters, **_: Any) -> dict[str, Any]:
        return {
            "hits": [
                hit("c1"),
                {"chunk_id": "c2", "chunk": chunk("c2")},
            ],
            "trace": {},
        }

    from src.agent.service import AgentService

    service = AgentService(
        AgentTools(search_backend=backend),
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector([search_action(use_graph=False)]),
    )

    result = service.ask("credit risk", conversation_id="no-graph-ids")

    assert result.diagnostics["evidence_manifest"] == ["c1", "c2"]
    assert result.diagnostics["graph_evidence_manifest"] == []


def test_get_entities_publishes_graph_diagnostics_on_tool_event() -> None:
    from src.agent.service import AgentService
    from src.agent.state import AgentAction

    entity_result = {
        "nodes": [ENTITY_ID],
        "results": [
            {
                "chunk_id": "c2",
                "graph_score": 0.8,
                "path": [ENTITY_ID, "chunk:c2"],
                "evidence_chunk_ids": ["c2"],
            }
        ],
        "evidence_chunk_ids": ["c2"],
    }
    service = AgentService(
        AgentTools(
            entity_backend=lambda names, *, filters, max_hops: entity_result
        ),
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector(
            [AgentAction("get_entities", {"entity_names": ["Credit Risk"]}, "explore_entities")]
        ),
    )

    result = service.ask("credit risk", conversation_id="entity-diagnostics")

    event = result.tool_events[0]
    assert event.tool_name == "get_entities"
    assert event.result_count == 1
    diagnostics = event.diagnostics
    assert diagnostics is not None
    stage = diagnostics["stages"]["graph"]
    assert stage == [
        {
            "chunk_id": "c2",
            "rank": 1,
            "score": 0.8,
            "path": [ENTITY_ID, "chunk:c2"],
            "evidence_chunk_ids": ["c2"],
        }
    ]
    assert diagnostics["available"] == {"graph": True}
    assert diagnostics["unavailable"] == {}
    assert diagnostics["counts"] == {"graph": 1}
    assert diagnostics["graph_candidates"] == 1
    assert diagnostics["graph_added"] == 1
    assert diagnostics["graph_matched_entities"] == [ENTITY_ID]
    # Через сервис диагностика попадает в отдельный graph retrieval_call.
    graph_call = result.diagnostics["retrieval_calls"][0]
    assert graph_call["tool_name"] == "get_entities"
    assert graph_call["entity_names"] == ["Credit Risk"]
    assert [item["chunk_id"] for item in graph_call["stages"]["graph"]] == ["c2"]


def test_empty_get_entities_result_does_not_invent_a_graph_stage() -> None:
    from src.agent.service import AgentService
    from src.agent.state import AgentAction

    service = AgentService(
        AgentTools(entity_backend=lambda names, *, filters, max_hops: {}),
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector(
            [AgentAction("get_entities", {"entity_names": ["Credit Risk"]}, "explore_entities")]
        ),
    )

    result = service.ask("credit risk", conversation_id="entity-empty")

    event = result.tool_events[0]
    assert event.result_count == 0
    assert not (event.diagnostics or {}).get("stages")
    assert result.diagnostics["graph_evidence_manifest"] == []


def test_query_and_entity_names_are_kept_in_tool_event_arguments() -> None:
    from src.agent.service import AgentService
    from src.agent.state import AgentAction

    service = AgentService(
        AgentTools(
            search_backend=lambda query, *, filters, **_: {
                "hits": [hit("c1")]
            },
            entity_backend=lambda names, *, filters, max_hops: {
                "evidence_chunk_ids": ["c2"]
            },
        ),
        answerer=EvidenceAnswerer(),
        action_selector=SequenceSelector(
            [
                AgentAction(
                    "search_reports",
                    {
                        "query": "чистая прибыль",
                        "queries": ["чистая прибыль", "net income"],
                    },
                    "fact_lookup",
                ),
                AgentAction(
                    "get_entities",
                    {"entity_names": ["Кредитный риск"]},
                    "explore_entities",
                ),
            ]
        ),
    )

    result = service.ask("чистая прибыль", conversation_id="no-redaction")

    search_event, entity_event = result.tool_events
    assert search_event.arguments["query"] == "чистая прибыль"
    assert search_event.arguments["queries"] == ["чистая прибыль", "net income"]
    assert entity_event.arguments["entity_names"] == ["Кредитный риск"]


