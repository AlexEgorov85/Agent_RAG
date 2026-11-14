"""Офлайн-тесты оркестрации с заглушками backend'ов и генераторов ответа."""

from __future__ import annotations

from src.agent import AgentAction, AgentService, Answer, Citation
from src.agent.state import QueryPlan
from src.agent.tools import AgentTools
from src.retrieval.hybrid_retriever import SearchFilters


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

    assert backend.kwargs["top_k"] == 3
    assert backend.kwargs["retrieval_mode"] == "keyword"


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
    assert len(result.tool_events) == 8
    assert "tool_budget_exhausted" in result.warnings
    assert result.answer.status == "abstain"


def test_search_budget_is_five_calls():
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
    assert len(search.calls) == 5
    assert len(result.tool_events) == 5
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
            chunk_id = next(iter(evidence_manifest))
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
    # Прошлые ходы — только контекст: второй ответ видит лишь свой свежий хит.
    assert evidence_seen[1] == ("second question", {"second question-id"})
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
