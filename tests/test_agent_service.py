"""Офлайн-тесты оркестрации с заглушками backend'ов и генераторов ответа."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.agent import AgentAction, AgentService, Answer, Citation
from src.agent.state import QueryPlan
from src.agent.tools import AgentTools
from src.engine.generator import AnswerGenerator
from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)


def hit(chunk_id: str = "chunk-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk": {
            "chunk_id": chunk_id,
            "doc_id": "report-2023",
            "year": 2023,
            "text": "Net income was 10.",
        },
    }


class FakeSearch:
    def __init__(self, hits=None):
        self.hits = list(hits if hits is not None else [hit()])
        self.calls = []

    def __call__(self, query, *, filters, retrieval_mode, top_k):
        self.calls.append((query, filters, retrieval_mode, top_k))
        return {"hits": self.hits, "warnings": []}


class AnswerFromEvidence:
    def generate(self, question, evidence, evidence_manifest):
        assert evidence_manifest
        chunk_id = sorted(evidence_manifest)[0]
        chunk = evidence[chunk_id]
        return Answer(
            "answer",
            f"Supported response for {question}",
            [Citation(chunk_id, doc_id=chunk.get("doc_id"), year=chunk.get("year"))],
        )


class SequenceSelector:
    def __init__(self, actions):
        self.actions = list(actions)

    def choose(self, state):
        if not self.actions:
            return None
        return self.actions.pop(0)


def service(search=None, **kwargs):
    return AgentService(
        AgentTools(search_backend=search or FakeSearch()),
        answerer=kwargs.pop("answerer", AnswerFromEvidence()),
        **kwargs,
    )


def test_simple_factoid_search_and_answer():
    search = FakeSearch()
    result = service(search).ask("What was net income?", conversation_id="a")
    assert result.answer.status == "answer"
    assert result.answer.citations[0].chunk_id == "chunk-1"
    assert [event.tool_name for event in result.tool_events] == ["search_reports"]
    assert len(search.calls) == 1


def test_tools_forward_top_k_and_retrieval_mode_to_backend():
    class RetrieveBackend:
        def __init__(self):
            self.kwargs = None

        def retrieve(self, query, *, filters, top_k, retrieval_mode):
            self.kwargs = {
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
            }
            return {"hits": []}

    backend = RetrieveBackend()
    tools = AgentTools(search_backend=backend, top_k=3)
    tools.invoke(
        "search_reports",
        {
            "query": "q",
            "filters": SearchFilters(),
            "top_k": 10,
            "retrieval_mode": "keyword",
        },
    )

    # top_k — значение по умолчанию, а не потолок: selector просит 10.
    assert backend.kwargs["top_k"] == 10
    assert backend.kwargs["retrieval_mode"] == "keyword"


def test_tools_clip_requested_top_k_above_the_hard_limit():
    class RetrieveBackend:
        def __init__(self):
            self.top_k = None

        def retrieve(self, query, *, filters, top_k, retrieval_mode):
            del query, filters, retrieval_mode
            self.top_k = top_k
            return {"hits": []}

    backend = RetrieveBackend()
    tools = AgentTools(search_backend=backend)

    tools.invoke(
        "search_reports",
        {"query": "q", "filters": SearchFilters(), "top_k": 50},
    )

    assert backend.top_k == 10


def test_tools_forward_rerank_per_year_only_to_supporting_backend():
    class RerankBackend:
        def __init__(self):
            self.rerank_per_year = None

        def retrieve(
            self,
            query,
            *,
            filters,
            top_k,
            retrieval_mode,
            rerank_per_year=None,
        ):
            del query, filters, top_k, retrieval_mode
            self.rerank_per_year = rerank_per_year
            return {"hits": []}

    backend = RerankBackend()
    tools = AgentTools(search_backend=backend)

    tools.invoke(
        "search_reports",
        {
            "query": "q",
            "filters": SearchFilters(),
            "rerank_per_year": True,
        },
    )

    assert backend.rerank_per_year is True


def test_router_plan_adds_year_and_section_filters():
    search = FakeSearch()

    class Router:
        def analyze(self, question, filters):
            return QueryPlan(
                years=[2023],
                year_from=2022,
                year_to=2024,
                section_filter="Risk management/Credit risk",
            )

    result = AgentService(
        AgentTools(search_backend=search),
        answerer=AnswerFromEvidence(),
        router=Router(),
    ).ask("Risk in 2023?", conversation_id="router-filters")
    actual = search.calls[0][1]
    assert actual.years == {2023}
    assert (actual.year_from, actual.year_to) == (2022, 2024)
    assert actual.section_path_prefix == "Risk management/Credit risk"
    assert result.effective_filters == actual


def test_explicit_filters_take_precedence_over_router_plan():
    search = FakeSearch()
    explicit = SearchFilters(
        years={2020},
        year_from=2020,
        year_to=2022,
        section_path_prefix="User chosen section",
    )

    class Router:
        def analyze(self, question, filters):
            return QueryPlan(
                years=[2024],
                year_from=2024,
                year_to=2025,
                section_filter="Router section",
            )

    result = AgentService(
        AgentTools(search_backend=search),
        answerer=AnswerFromEvidence(),
        router=Router(),
    ).ask("Question", conversation_id="explicit-filters", filters=explicit)
    actual = search.calls[0][1]
    assert actual.years == {2020}
    assert (actual.year_from, actual.year_to) == (2020, 2022)
    assert actual.section_path_prefix == "User chosen section"
    assert result.effective_filters == explicit


def test_multi_tool_structure_then_search():
    search = FakeSearch()
    selector = SequenceSelector(
        [
            AgentAction("get_structure_tree", {"doc_id": "report-2023"}, "clarify_structure"),
            AgentAction(
                "search_reports", {"query": "net income"}, "fact_lookup"
            ),
        ]
    )
    tools = AgentTools(
        search_backend=search,
        structure_backend=lambda doc_id, **kwargs: {"doc_id": doc_id, "sections": []},
    )
    result = AgentService(
        tools, answerer=AnswerFromEvidence(), action_selector=selector
    ).ask("Find the reported net income", conversation_id="multi")
    assert [event.tool_name for event in result.tool_events] == [
        "get_structure_tree", "search_reports"
    ]
    assert result.answer.status == "answer"


def test_graph_unavailable_does_not_block_search():
    selector = SequenceSelector(
        [
            AgentAction("get_entities", {"entity_names": ["Bank"]}, "explore_entities"),
            AgentAction("search_reports", {"query": "income"}, "fact_lookup"),
        ]
    )
    result = AgentService(
        AgentTools(search_backend=FakeSearch()),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    ).ask("Bank income", conversation_id="graph-fallback")
    assert [event.status for event in result.tool_events] == [
        "tool_unavailable", "completed"
    ]
    assert "graph_unavailable" in result.warnings
    assert result.answer.status == "answer"


def test_graph_evidence_authorizes_get_chunk_and_citation():
    selector = SequenceSelector(
        [
            AgentAction("get_entities", {"entity_names": ["Bank"]}, "explore_entities"),
            AgentAction("get_chunk", {"chunk_id": "graph-chunk"}, "inspect_evidence"),
        ]
    )
    requested = []

    def get_chunk(chunk_id):
        requested.append(chunk_id)
        return {
            "chunk_id": chunk_id,
            "doc_id": "graph-report",
            "text": "Graph-backed evidence content.",
        }

    result = AgentService(
        AgentTools(
            entity_backend=lambda names, **kwargs: {
                "evidence_chunk_ids": ["graph-chunk"]
            },
            chunk_backend=get_chunk,
        ),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    ).ask("Graph-backed fact", conversation_id="graph-chunk")
    assert requested == ["graph-chunk"]
    assert [event.tool_name for event in result.tool_events] == [
        "get_entities", "get_chunk"
    ]
    assert result.answer.status == "answer"
    assert result.answer.citations[0].chunk_id == "graph-chunk"


def test_arbitrary_get_chunk_id_is_rejected():
    requested = []
    selector = SequenceSelector(
        [AgentAction("get_chunk", {"chunk_id": "arbitrary"}, "inspect_evidence")]
    )

    def get_chunk(chunk_id):
        requested.append(chunk_id)
        return {"chunk_id": chunk_id, "text": "untrusted"}

    result = AgentService(
        AgentTools(chunk_backend=get_chunk),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    ).ask("No evidence", conversation_id="arbitrary-chunk")
    assert requested == []
    assert not result.tool_events
    assert "chunk_not_authorized" in result.warnings
    assert result.answer.status == "abstain"


def test_unknown_tool_rejected_and_safe_search_fallback():
    selector = SequenceSelector(
        [AgentAction("run_shell", {"command": "not allowed"}, "fact_lookup")]
    )
    search = FakeSearch()
    result = AgentService(
        AgentTools(search_backend=search),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    ).ask("question", conversation_id="unknown")
    assert [event.tool_name for event in result.tool_events] == ["search_reports"]
    assert "unknown_tool_rejected" in result.warnings
    assert len(search.calls) == 1


def test_tool_budgets_stop_loops():
    selector = SequenceSelector(
        [AgentAction("get_structure_tree", {"doc_id": "d"}, "clarify_structure")]
        * 10
    )
    service_instance = AgentService(
        AgentTools(structure_backend=lambda doc_id, **kwargs: {"doc_id": doc_id}),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    )
    result = service_instance.ask("structure?", conversation_id="budget")
    assert len(result.tool_events) == 4
    assert "tool_budget_exhausted" in result.warnings
    assert result.answer.status == "abstain"


def test_search_budget_is_two_calls():
    search = FakeSearch()
    selector = SequenceSelector(
        [
            AgentAction("search_reports", {"query": f"query-{index}"}, "fact_lookup")
            for index in range(8)
        ]
    )
    result = AgentService(
        AgentTools(search_backend=search),
        answerer=AnswerFromEvidence(),
        action_selector=selector,
    ).ask("query", conversation_id="search-budget")
    assert len(search.calls) == 2
    assert len(result.tool_events) == 2
    assert "tool_budget_exhausted" in result.warnings


def test_citations_are_repaired_and_allowlisted():
    class RepairingAnswerer:
        def __init__(self):
            self.calls = 0
            self.repair_flags = []
            self.diagnostics = []

        def generate(
            self, question, evidence, evidence_manifest, *, repair=False, diagnostics=None
        ):
            self.calls += 1
            self.repair_flags.append(repair)
            self.diagnostics.append(diagnostics)
            chunk_id = "outside-manifest" if self.calls == 1 else "chunk-1"
            return Answer("answer", "Claim", [Citation(chunk_id)])

    answerer = RepairingAnswerer()
    result = service(answerer=answerer).ask("Claim?", conversation_id="citation")
    assert answerer.calls == 2
    assert answerer.repair_flags == [False, True]
    assert answerer.diagnostics[1]["reason"] == (
        "answer_schema_or_citation_validation_failed"
    )
    assert answerer.diagnostics[1]["invalid_citation_chunk_ids"] == [
        "outside-manifest"
    ]
    assert result.answer.citations[0].chunk_id == "chunk-1"
    assert "citation_validation_retry" in result.warnings

    class AlwaysInvalid:
        def generate(self, question, evidence, evidence_manifest):
            return Answer("answer", "Claim", [Citation("outside-manifest")])

    failed = service(answerer=AlwaysInvalid()).ask("Claim?", conversation_id="bad-citation")
    assert failed.answer.status == "abstain"
    assert not failed.answer.citations


def test_conversation_isolation_and_stream_ask_parity():
    observed = []

    class ConversationSearch:
        def __call__(self, query, *, filters, retrieval_mode, top_k):
            return {"hits": [hit(f"{query}-chunk")]}

    class IsolatingAnswerer:
        def generate(self, question, evidence, evidence_manifest):
            observed.append((question, set(evidence_manifest)))
            chunk_id = next(iter(evidence_manifest))
            return Answer("answer", question, [Citation(chunk_id)])

    app = AgentService(
        AgentTools(search_backend=ConversationSearch()),
        answerer=IsolatingAnswerer(),
    )
    first = app.ask("first", conversation_id="conversation-one")
    streamed = list(app.stream("second", conversation_id="conversation-two"))
    second = streamed[-1].result
    direct = app.ask("second", conversation_id="conversation-two")
    assert first.conversation_id != second.conversation_id
    assert observed[0][1] == {"first-chunk"}
    assert observed[1][1] == observed[2][1] == {"second-chunk"}
    assert second is not None
    assert second.answer == direct.answer
    assert [
        (event.tool_name, event.status, event.result_count)
        for event in second.tool_events
    ] == [
        (event.tool_name, event.status, event.result_count)
        for event in direct.tool_events
    ]
    assert streamed[-1].event_type == "run_completed"
    assert streamed[-1].result.answer == direct.answer
    assert streamed[-1].result.tool_events
    assert streamed[-1].result.answer.citations == direct.answer.citations


def test_history_is_passed_to_router_selector_and_answerer_per_conversation():
    router_histories = []
    selector_histories = []
    answer_histories = []
    evidence_seen = []

    class HistoryRouter:
        def analyze(self, question, filters, *, history=None):
            router_histories.append((question, history))
            return QueryPlan()

    class HistorySelector:
        def choose(self, state):
            if not state["tool_events"]:
                selector_histories.append(
                    (state["question"], list(state.get("history", [])))
                )
                return AgentAction(
                    "search_reports",
                    {"query": state["question"]},
                    "fact_lookup",
                )
            return None

    class HistoryAnswerer:
        def generate(
            self, question, evidence, evidence_manifest, *, history=None
        ):
            answer_histories.append((question, history))
            evidence_seen.append((question, set(evidence_manifest)))
            # Цитируем свежий хит текущего вопроса, а не перенесённый chunk.
            chunk_id = next(
                item for item in evidence_manifest if item.startswith(question)
            )
            return Answer("answer", f"answer:{question}", [Citation(chunk_id)])

    app = AgentService(
        AgentTools(search_backend=lambda query, **kwargs: {"hits": [hit(f"{query}-id")]}),
        router=HistoryRouter(),
        action_selector=HistorySelector(),
        answerer=HistoryAnswerer(),
    )
    app.ask("first question api_key=superSecret", conversation_id="history-a")
    second = app.ask("second question", conversation_id="history-a")
    isolated = app.ask("other conversation", conversation_id="history-b")

    expected_prior = [
        {"role": "user", "content": "first question api_key=[REDACTED]"},
        {
            "role": "assistant",
            "content": "answer:first question api_key=[REDACTED]",
        },
    ]
    assert router_histories[1] == ("second question", expected_prior)
    assert selector_histories[1] == ("second question", expected_prior)
    assert answer_histories[1] == ("second question", expected_prior)
    assert "superSecret" not in repr(expected_prior)
    assert router_histories[2] == ("other conversation", [])
    assert selector_histories[2] == ("other conversation", [])
    assert answer_histories[2] == ("other conversation", [])
    # Follow-up сохраняет подтверждённые цитаты прошлого хода: второго ответа
    # касаются и перенесённый chunk, и его собственный свежий хит. Текст
    # истории при этом остаётся только контекстом и не цитируется.
    assert evidence_seen[1] == (
        "second question",
        {"second question-id", "first question api_key=superSecret-id"},
    )
    assert second.answer.citations[0].chunk_id == "second question-id"
    assert isolated.answer.citations[0].chunk_id == "other conversation-id"


def test_conversation_history_is_bounded_to_last_ten_turns():
    observed = []

    class HistoryAnswerer:
        def generate(
            self, question, evidence, evidence_manifest, *, history=None
        ):
            observed.append((question, history))
            chunk_id = next(iter(evidence_manifest))
            return Answer("answer", f"answer:{question}", [Citation(chunk_id)])

    app = AgentService(
        AgentTools(search_backend=lambda query, **kwargs: {"hits": [hit(f"{query}-id")]}),
        answerer=HistoryAnswerer(),
    )
    for index in range(12):
        app.ask(f"q{index}", conversation_id="bounded")

    final_history = observed[-1][1]
    assert len(final_history) == 20
    assert final_history[0] == {"role": "user", "content": "q1"}
    assert final_history[-1] == {"role": "assistant", "content": "answer:q10"}
    assert all(set(item) == {"role", "content"} for item in final_history)


# --- diagnostics: context_chunk_ids, retrieval_calls, evidence manifests ----


class _JsonCompletions:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.dumps(
            {
                "status": "answer",
                "answer": "Ответ по упакованному источнику.",
                "citations": [{"chunk_id": self.chunk_id}],
                "facts": [],
                "warnings": [],
            },
            ensure_ascii=False,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )


class _JsonClient:
    def __init__(self, chunk_id: str) -> None:
        self.completions = _JsonCompletions(chunk_id)
        self.chat = SimpleNamespace(completions=self.completions)


def _chunk(chunk_id: str, *, text: str, year: int = 2023) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "report-2023",
        "year": year,
        "text": text,
    }


def _nested_hit(chunk: dict) -> dict:
    return {"chunk_id": chunk["chunk_id"], "chunk": chunk}


def test_diagnostics_context_ids_contain_only_packed_chunks():
    packed = _chunk("packed", text="short evidence")
    oversized = _chunk("oversized", text="H" * 10_000)
    search = FakeSearch([_nested_hit(packed), _nested_hit(oversized)])
    selector = SequenceSelector(
        [AgentAction("search_reports", {"query": "q"}, "fact_lookup")]
    )
    answerer = AnswerGenerator(
        _JsonClient("packed"), "test-model", max_context_chars=2_000
    )
    result = AgentService(
        AgentTools(search_backend=search),
        answerer=answerer,
        action_selector=selector,
    ).ask("Прибыль?", conversation_id="context-ids")

    assert result.answer.status == "answer"
    # context ids — именно упакованные чанки, а не весь evidence manifest.
    assert result.diagnostics["context_chunk_ids"] == ["packed"]
    assert result.diagnostics["context_source"] == "context"
    assert set(result.diagnostics["evidence_manifest"]) == {"packed", "oversized"}


def test_context_source_falls_back_when_generator_hides_context():
    result = service().ask("What was net income?", conversation_id="no-context")

    assert result.diagnostics["context_chunk_ids"] is None
    assert result.diagnostics["context_source"] == "evidence_manifest"
    assert result.diagnostics["evidence_manifest"] == ["chunk-1"]


def test_retrieval_calls_carry_trace_stages_and_counts():
    class TraceSearch:
        def retrieve(self, query, *, filters, top_k=None, retrieval_mode="hybrid"):
            return RetrievalResult(
                hits=[RetrievalHit(chunk_id="chunk-1", chunk=_chunk(
                    "chunk-1", text="Net income was 10."))],
                effective_filters=filters or SearchFilters(),
                trace=RetrievalTrace(
                    retrieval_mode=retrieval_mode,
                    dense_candidates=2,
                    bm25_candidates=3,
                    fused_candidates=4,
                    rerank_candidates=1,
                    returned=1,
                    stages={
                        "dense": [{"chunk_id": "chunk-1", "rank": 1, "score": 0.9}],
                        "bm25": [{"chunk_id": "chunk-1", "rank": 2, "score": 4.2}],
                        "returned": [
                            {"chunk_id": "chunk-1", "rank": 1, "score": 0.9}
                        ],
                    },
                    available={"dense": True, "bm25": True, "returned": True},
                    fallback={"reranker": False},
                    unavailable={"rerank": "disabled"},
                ),
            )

    result = AgentService(
        AgentTools(search_backend=TraceSearch()),
        answerer=AnswerFromEvidence(),
        action_selector=SequenceSelector(
            [AgentAction("search_reports", {"query": "net income"}, "fact_lookup")]
        ),
    ).ask("What was net income?", conversation_id="trace-diagnostics")

    calls = result.diagnostics["retrieval_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["call_index"] == 0
    assert call["tool_name"] == "search_reports"
    assert call["retrieval_mode"] == "hybrid"
    assert call["top_k"] == AgentTools().top_k
    assert call["status"] == "completed"
    assert call["counts"] == {
        "dense": 2, "bm25": 3, "rrf": 4, "rerank": 1, "returned": 1,
    }
    assert [item["chunk_id"] for item in call["stages"]["dense"]] == ["chunk-1"]
    assert call["available"]["bm25"] is True
    assert call["unavailable"] == {"rerank": "disabled"}


def test_diagnostics_keep_primary_and_graph_evidence_manifests_separate():
    result = service().ask("What was net income?", conversation_id="manifests")

    assert result.diagnostics["evidence_manifest"] == ["chunk-1"]
    assert result.diagnostics["graph_evidence_manifest"] == []


def test_smalltalk_answers_without_retrieval():
    search = FakeSearch()
    events = list(service(search).stream("Привет!", conversation_id="smalltalk"))

    result = events[-1].result
    assert [event.event_type for event in events] == ["run_started", "run_completed"]
    assert result is not None
    assert result.status == "completed"
    assert result.answer.status == "answer"
    assert result.answer.citations == []
    assert result.answer.facts == []
    assert result.warnings == []
    assert result.tool_events == []
    assert result.diagnostics["smalltalk"] is True
    assert "Здравствуйте" in result.answer.answer
    assert search.calls == []


def test_greeting_with_question_still_uses_retrieval():
    search = FakeSearch()
    result = service(search).ask(
        "Привет! What was net income in 2023?", conversation_id="greeting-question"
    )

    assert result.answer.citations[0].chunk_id == "chunk-1"
    assert result.diagnostics.get("smalltalk") is None
    assert len(search.calls) == 1


def test_smalltalk_does_not_pollute_history():
    agent = service()
    agent.ask("Привет!", conversation_id="history")
    assert agent._history_snapshot("history") == []

    result = agent.ask("What was net income?", conversation_id="history")
    assert result.answer.citations
    contents = [item["content"] for item in agent._history_snapshot("history")]
    assert all("Здравствуйте" not in content for content in contents)
    assert contents[0] == "What was net income?"
