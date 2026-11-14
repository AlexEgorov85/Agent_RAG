"""Bounded LangGraph orchestration graph."""

from __future__ import annotations

import json
import inspect
import time
import uuid
from dataclasses import asdict, is_dataclass
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agent.state import AgentAction, AgentState, Answer, QueryPlan, ToolEvent
from src.agent.tools import (
    AgentTools,
    ToolUnavailable,
    chunk_from_hit,
    result_hits,
    result_warnings,
)
from src.retrieval.hybrid_retriever import SearchFilters

MAX_TOOL_CALLS = 4
MAX_SEARCH_CALLS = 2


class Router(Protocol):
    def analyze(self, question: str, filters: SearchFilters) -> QueryPlan: ...


class ActionSelector(Protocol):
    def choose(self, state: AgentState) -> AgentAction | None: ...


def _supports_keyword(target: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or (
            item.name == keyword
            and item.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for item in parameters
    )


def _as_plan(value: Any) -> QueryPlan:
    if isinstance(value, QueryPlan):
        plan = value
    elif isinstance(value, Mapping):
        allowed = {
            key: value[key] for key in QueryPlan.__dataclass_fields__ if key in value
        }
        try:
            plan = QueryPlan(**allowed)
        except (TypeError, ValueError):
            return QueryPlan()
    else:
        return QueryPlan()
    if (
        plan.intent not in {"factoid", "structural", "temporal", "multi_hop"}
        or plan.answer_mode not in {"concise", "comparison", "outline"}
        or not isinstance(plan.years, list)
        or any(isinstance(year, bool) or not isinstance(year, int) for year in plan.years)
        or not isinstance(plan.entities, list)
        or any(not isinstance(item, str) for item in plan.entities)
        or not isinstance(plan.needs_rewriting, bool)
        or not isinstance(plan.needs_graph, bool)
    ):
        return QueryPlan()
    for year in (plan.year_from, plan.year_to):
        if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
            return QueryPlan()
    if (
        plan.year_from is not None
        and plan.year_to is not None
        and plan.year_from > plan.year_to
    ):
        return QueryPlan()
    if plan.section_filter is not None and not isinstance(plan.section_filter, str):
        return QueryPlan()
    return plan


def _filters_with_plan(explicit: SearchFilters, plan: QueryPlan) -> SearchFilters:
    """Narrow unspecified filters from the router without overriding user scope."""
    user_years_are_explicit = explicit.years is not None
    years = None if explicit.years is None else set(explicit.years)
    year_from = explicit.year_from
    year_to = explicit.year_to

    if not user_years_are_explicit:
        if plan.years:
            years = set(plan.years)
        if plan.year_from is not None:
            year_from = (
                plan.year_from
                if year_from is None
                else max(year_from, plan.year_from)
            )
        if plan.year_to is not None:
            year_to = (
                plan.year_to
                if year_to is None
                else min(year_to, plan.year_to)
            )
        # A router inference conflicting with a user-supplied bound narrows to
        # an empty scope rather than weakening or reversing the user's bound.
        if year_from is not None and year_to is not None and year_from > year_to:
            years = set()
            year_from, year_to = explicit.year_from, explicit.year_to

    return SearchFilters(
        years=years,
        year_from=year_from,
        year_to=year_to,
        report_periods=explicit.report_periods,
        issuers=explicit.issuers,
        kinds=explicit.kinds,
        allowed_doc_ids=explicit.allowed_doc_ids,
        section_path_prefix=(
            explicit.section_path_prefix
            if explicit.section_path_prefix is not None
            else plan.section_filter
        ),
        breadcrumb_contains=explicit.breadcrumb_contains,
        max_section_level=explicit.max_section_level,
    ).normalized()


def _graph_evidence_ids(value: Any) -> set[str]:
    """Collect IDs only from explicit graph-result evidence fields."""
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "evidence_chunk_ids" and isinstance(
                    child, (list, tuple, set)
                ):
                    found.update(
                        chunk_id for chunk_id in child
                        if isinstance(chunk_id, str) and chunk_id
                    )
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return found


def _contained_chunk_id(chunk: Any) -> str | None:
    if isinstance(chunk, Mapping):
        value = chunk.get("chunk_id")
    else:
        value = getattr(chunk, "chunk_id", None)
    return value if isinstance(value, str) and value else None


def _safe_action(value: Any) -> AgentAction | None:
    if value is None:
        return None
    if isinstance(value, AgentAction):
        return value
    if isinstance(value, Mapping):
        try:
            return AgentAction(
                tool_name=str(value["tool_name"]),
                arguments=dict(value["arguments"]),
                reason_code=str(value["reason_code"]),
            )
        except (KeyError, TypeError, ValueError):
            return AgentAction(tool_name="__invalid__", arguments={}, reason_code="fact_lookup")
    return AgentAction(tool_name="__invalid__", arguments={}, reason_code="fact_lookup")


class _DefaultRouter:
    def analyze(self, question: str, filters: SearchFilters) -> QueryPlan:
        del filters
        return QueryPlan()


class _DefaultSelector:
    def choose(self, state: AgentState) -> AgentAction | None:
        if not state.get("tool_events"):
            return AgentAction(
                "search_reports",
                {
                    "query": state["question"],
                    "filters": state["effective_filters"],
                    "retrieval_mode": "hybrid",
                    "top_k": 5,
                },
                "fact_lookup",
            )
        return None


def build_agent_graph(
    tools: AgentTools,
    *,
    router: Router | Callable[..., Any] | None = None,
    action_selector: ActionSelector | Callable[..., Any] | None = None,
    checkpointer: Any = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    answer_generator: Callable[[AgentState], tuple[Answer, list[str]]] | None = None,
    answer_validator: Callable[[Answer, set[str]], bool] | None = None,
):
    """Build a StateGraph that enforces tool-call and search-call budgets."""
    router = router or _DefaultRouter()
    action_selector = action_selector or _DefaultSelector()

    class SearchInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1)
        filters: dict[str, Any] | None = None
        retrieval_mode: str = "hybrid"
        top_k: int = 5

    class StructureInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        doc_id: str = Field(min_length=1)
        section_path: str | None = None
        max_depth: int = 4

    class EntitiesInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        entity_names: list[str]
        filters: dict[str, Any] | None = None
        max_hops: int = 2

    class ChunkInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        chunk_id: str = Field(min_length=1)

    def filters_from_dict(value: dict[str, Any] | None) -> SearchFilters:
        if value is None:
            return SearchFilters()
        return SearchFilters(**value).normalized()

    def encode_result(value: Any) -> str:
        def convert(item: Any) -> Any:
            if is_dataclass(item):
                return asdict(item)
            if isinstance(item, Mapping):
                return {str(key): convert(child) for key, child in item.items()}
            if isinstance(item, (list, tuple, set)):
                return [convert(child) for child in item]
            if hasattr(item, "__dict__"):
                return {
                    str(key): convert(child)
                    for key, child in vars(item).items()
                    if not key.startswith("_")
                }
            return item

        return json.dumps(convert(value), ensure_ascii=False, default=str)

    def wrapped(name: str, arguments: dict[str, Any]) -> str:
        started = time.perf_counter()
        try:
            value = tools.invoke(name, arguments)
            return encode_result({
                "_agent_status": "completed",
                "result": value,
                "_duration_ms": (time.perf_counter() - started) * 1000,
            })
        except ToolUnavailable as exc:
            allowed = {
                "graph_unavailable", "structure_unavailable", "chunk_unavailable",
                "search_unavailable",
            }
            warning = str(exc) if str(exc) in allowed else "tool_unavailable"
            return encode_result({
                "_agent_status": "tool_unavailable",
                "warning": warning,
                "_duration_ms": (time.perf_counter() - started) * 1000,
            })
        except Exception:
            # Never put exception details or provider response bodies into messages.
            return encode_result({
                "_agent_status": "failed",
                "warning": "tool_failed",
                "_duration_ms": (time.perf_counter() - started) * 1000,
            })

    @tool("search_reports", args_schema=SearchInput)
    def search_reports(
        query: str, filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid", top_k: int = 5,
    ) -> str:
        """Search the report corpus with validated filters."""
        return wrapped(
            "search_reports",
            {
                "query": query, "filters": filters_from_dict(filters),
                "retrieval_mode": retrieval_mode, "top_k": top_k,
            },
        )

    @tool("get_structure_tree", args_schema=StructureInput)
    def get_structure_tree(
        doc_id: str, section_path: str | None = None, max_depth: int = 4,
    ) -> str:
        """Read a bounded local document structure tree."""
        return wrapped(
            "get_structure_tree",
            {"doc_id": doc_id, "section_path": section_path, "max_depth": max_depth},
        )

    @tool("get_entities", args_schema=EntitiesInput)
    def get_entities(
        entity_names: list[str], filters: dict[str, Any] | None = None,
        max_hops: int = 2,
    ) -> str:
        """Find bounded entity paths with source evidence."""
        return wrapped(
            "get_entities",
            {
                "entity_names": entity_names, "filters": filters_from_dict(filters),
                "max_hops": max_hops,
            },
        )

    @tool("get_chunk", args_schema=ChunkInput)
    def get_chunk(chunk_id: str) -> str:
        """Read one chunk by its indexed chunk ID."""
        return wrapped("get_chunk", {"chunk_id": chunk_id})

    tool_node = ToolNode([search_reports, get_structure_tree, get_entities, get_chunk])

    def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(kind, data)

    def normalize_request(state: AgentState) -> dict[str, Any]:
        question = " ".join(state.get("question", "").split())
        filters = state.get("filters") or SearchFilters()
        if not isinstance(filters, SearchFilters):
            raise ValueError("filters must be SearchFilters")
        normalized = filters.normalized()
        if not question:
            raise ValueError("query must not be empty")
        return {"question": question, "filters": normalized, "effective_filters": normalized}

    def analyze_query(state: AgentState) -> dict[str, Any]:
        try:
            target = router.analyze if hasattr(router, "analyze") else router
            kwargs = {}
            if _supports_keyword(target, "history"):
                kwargs["history"] = list(state.get("history", []))
            result = target(state["question"], state["filters"], **kwargs)
            plan = _as_plan(result)
        except Exception:
            # Never expose provider exception messages or bodies.
            plan = QueryPlan()
            emit(
                "analysis_completed",
                {
                    "intent": plan.intent,
                    "effective_filters": state["effective_filters"],
                },
            )
            return {
                "query_plan": plan,
                "warnings": state.get("warnings", []) + ["query_analysis_fallback"],
            }
        effective = _filters_with_plan(state["filters"], plan)
        emit(
            "analysis_completed",
            {"intent": plan.intent, "effective_filters": effective},
        )
        return {
            "query_plan": plan,
            "effective_filters": effective,
        }

    def choose_action(state: AgentState) -> dict[str, Any]:
        try:
            result = (
                action_selector.choose(state)
                if hasattr(action_selector, "choose")
                else action_selector(state)
            )
            action = _safe_action(result)
        except Exception:
            action = None
            return {
                "action": None,
                "warnings": state.get("warnings", []) + ["action_selection_failed"],
                "messages": [AIMessage(content="")],
            }
        if action is None:
            return {"action": None, "messages": [AIMessage(content="")]}
        allowed_reasons = {
            "fact_lookup", "clarify_structure", "explore_entities", "inspect_evidence"
        }
        if action.reason_code not in allowed_reasons:
            action = AgentAction(action.tool_name, action.arguments, "fact_lookup")
        if action.tool_name not in tools.ALLOWLIST:
            warnings = state.get("warnings", []) + ["unknown_tool_rejected"]
            # A malformed router/selector cannot execute arbitrary functionality.
            if not state.get("unknown_action_fallback_used"):
                # Continue with the safe search through the standard ToolNode.
                action = AgentAction(
                    "search_reports",
                    {
                        "query": state["question"],
                        "filters": state["effective_filters"],
                        "retrieval_mode": "hybrid",
                        "top_k": 5,
                    },
                    "fact_lookup",
                )
                return build_call(
                    action,
                    warnings=warnings,
                    unknown_action_fallback_used=True,
                )
            return {
                "action": None, "warnings": warnings, "messages": [AIMessage(content="")]
            }
        calls = state.get("tool_calls_used", 0)
        searches = state.get("search_calls_used", 0)
        if calls >= MAX_TOOL_CALLS or (
            action.tool_name == "search_reports" and searches >= MAX_SEARCH_CALLS
        ):
            return {
                "action": None,
                "warnings": state.get("warnings", []) + ["tool_budget_exhausted"],
                "messages": [AIMessage(content="")],
            }
        if action.tool_name == "get_chunk":
            requested_id = action.arguments.get("chunk_id")
            authorized = set(state.get("search_evidence_manifest", set()))
            authorized.update(state.get("graph_evidence_manifest", set()))
            if not isinstance(requested_id, str) or requested_id not in authorized:
                return {
                    "action": None,
                    "warnings": state.get("warnings", []) + ["chunk_not_authorized"],
                    "messages": [AIMessage(content="")],
                }
        arguments = dict(action.arguments)
        if action.tool_name in {"search_reports", "get_entities"}:
            arguments["filters"] = {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in vars(state["effective_filters"]).items()
            }
        schema = {
            "search_reports": SearchInput,
            "get_structure_tree": StructureInput,
            "get_entities": EntitiesInput,
            "get_chunk": ChunkInput,
        }[action.tool_name]
        try:
            arguments = schema.model_validate(arguments).model_dump(exclude_none=True)
        except Exception:
            warnings = state.get("warnings", []) + ["invalid_tool_arguments"]
            if action.tool_name != "search_reports" and not state.get(
                "unknown_action_fallback_used"
            ):
                fallback = AgentAction(
                    "search_reports",
                    {
                        "query": state["question"],
                        "filters": {
                            key: sorted(value) if isinstance(value, set) else value
                            for key, value in vars(state["effective_filters"]).items()
                        },
                        "retrieval_mode": "hybrid",
                        "top_k": 5,
                    },
                    "fact_lookup",
                )
                return build_call(
                    fallback,
                    warnings=warnings,
                    unknown_action_fallback_used=True,
                )
            return {
                "action": None, "warnings": warnings,
                "messages": [AIMessage(content="")],
            }
        # LangGraph's documented ToolNode/tools_condition protocol carries the
        # validated decision; no custom tool-call execution loop is implemented.
        return build_call(AgentAction(action.tool_name, arguments, action.reason_code))

    def build_call(
        action: AgentAction,
        *,
        warnings: list[str] | None = None,
        unknown_action_fallback_used: bool | None = None,
    ) -> dict[str, Any]:
        call_id = str(uuid.uuid4())
        args = dict(action.arguments)
        if isinstance(args.get("filters"), SearchFilters):
            args["filters"] = {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in vars(args["filters"]).items()
            }
        call = {
            "action": AgentAction(action.tool_name, args, action.reason_code),
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": action.tool_name,
                        "args": args,
                        "id": call_id,
                        "type": "tool_call",
                    }],
                )
            ],
        }
        if warnings is not None:
            call["warnings"] = warnings
        if unknown_action_fallback_used is not None:
            call["unknown_action_fallback_used"] = unknown_action_fallback_used
        emit("tool_started", {
            "tool_name": action.tool_name,
            "short_arguments": {
                key: ("[redacted]" if key in {"query", "entity_names", "chunk_id"} else value)
                for key, value in args.items() if key != "filters"
            },
        })
        return call

    def harvest_tool_result(state: AgentState) -> dict[str, Any]:
        action = state.get("action")
        if action is None:
            return {}
        calls = state.get("tool_calls_used", 0)
        searches = state.get("search_calls_used", 0)
        arguments = dict(action.arguments)
        messages = state.get("messages", [])
        tool_message = next(
            (message for message in reversed(messages) if isinstance(message, ToolMessage)),
            None,
        )
        payload: dict[str, Any] = {}
        if tool_message is not None:
            try:
                decoded = json.loads(tool_message.content)
                if isinstance(decoded, dict):
                    payload = decoded
            except (TypeError, json.JSONDecodeError):
                payload = {"_agent_status": "failed", "warning": "tool_failed"}
        result = payload.get("result")
        status = payload.get("_agent_status", "failed")
        warning = payload.get("warning")
        short_args = {
            key: ("[redacted]" if key in {"query", "entity_names", "chunk_id"} else value)
            for key, value in arguments.items()
            if key not in {"filters"}
        }
        if action.tool_name in {"search_reports", "get_entities"}:
            filters = arguments.get("filters")
            if isinstance(filters, SearchFilters):
                filters = {
                    key: sorted(value) if isinstance(value, set) else value
                    for key, value in vars(filters).items()
                }
            short_args["filters"] = filters
        duration = float(payload.get("_duration_ms", 0.0))
        count = len(result_hits(result)) if action.tool_name == "search_reports" else (
            len(result) if isinstance(result, (list, tuple)) else int(result is not None)
        )
        event = ToolEvent(
            tool_name=action.tool_name,
            arguments=short_args,
            status=status,
            result_count=count,
            duration_ms=duration,
            warning=warning,
        )
        results = list(state.get("retrieval_results", []))
        evidence = dict(state.get("evidence", {}))
        manifest = set(state.get("evidence_manifest", set()))
        search_manifest = set(state.get("search_evidence_manifest", set()))
        graph_manifest = set(state.get("graph_evidence_manifest", set()))
        warnings = list(state.get("warnings", []))
        if action.tool_name == "search_reports" and result is not None:
            results.append(result)
            warnings.extend(safe_result_warnings(result))
            for hit in result_hits(result):
                chunk_id, chunk = chunk_from_hit(hit)
                inner_id = _contained_chunk_id(chunk)
                if chunk_id and (inner_id is None or inner_id == chunk_id):
                    manifest.add(chunk_id)
                    search_manifest.add(chunk_id)
                    evidence[chunk_id] = chunk
        elif action.tool_name == "get_entities" and result is not None:
            graph_manifest.update(_graph_evidence_ids(result))
        elif action.tool_name == "get_chunk" and result is not None:
            chunk_id, chunk = chunk_from_hit(result)
            inner_id = _contained_chunk_id(chunk)
            requested_id = arguments.get("chunk_id")
            if (
                chunk_id == requested_id
                and (inner_id is None or inner_id == requested_id)
                and chunk_id in (
                    search_manifest | graph_manifest
                )
            ):
                evidence[chunk_id] = chunk
                manifest.add(chunk_id)
            else:
                warnings.append("chunk_not_authorized")
        elif status in {"tool_unavailable", "failed"} and warning:
            warnings.append(warning)
        if calls + 1 >= MAX_TOOL_CALLS:
            warnings.append("tool_budget_exhausted")
        emit(
            "tool_completed",
            {
                "tool_name": action.tool_name,
                "result_count": count,
                "duration_ms": duration,
                "status": status,
            },
        )
        return {
            "action": None,
            "messages": [],
            "tool_events": list(state.get("tool_events", [])) + [event],
            "retrieval_results": results,
            "evidence_manifest": manifest,
            "search_evidence_manifest": search_manifest,
            "graph_evidence_manifest": graph_manifest,
            "evidence": evidence,
            "warnings": warnings,
            "tool_calls_used": calls + 1,
            "search_calls_used": searches + int(action.tool_name == "search_reports"),
        }

    def assess_evidence(state: AgentState) -> dict[str, Any]:
        # The selector is deliberately consulted again after every result; it
        # may choose a bounded follow-up (e.g. structure -> search).
        enough = bool(state.get("evidence_manifest"))
        return {"status": "evidence_available" if enough else "evidence_pending"}

    def safe_result_warnings(result: Any) -> list[str]:
        safe: list[str] = []
        for warning in result_warnings(result):
            normalized = warning.casefold()
            if "missing_year" in normalized or "missing year" in normalized:
                safe.append("missing_year")
            elif "issuer" in normalized:
                safe.append("issuer_mismatch")
            elif "graph" in normalized:
                safe.append("graph_unavailable")
            elif "fallback" in normalized:
                safe.append("retrieval_fallback")
            else:
                safe.append("retrieval_warning")
        return safe

    def generate_answer(state: AgentState) -> dict[str, Any]:
        emit("answer_started", {})
        started = time.perf_counter()
        if answer_generator is None:
            answer = Answer(
                "abstain",
                "Ответ не сформирован: генератор ответа не настроен.",
                warnings=["generation_unavailable"],
            )
            answer_warnings: list[str] = []
        else:
            try:
                answer, answer_warnings = answer_generator(state)
            except Exception:
                # Keep generator/provider errors out of state and UI.
                answer = Answer(
                    "abstain",
                    "Ответ не удалось безопасно сформировать.",
                    warnings=["generation_unavailable"],
                )
                answer_warnings = []
        timings = dict(state.get("timings", {}))
        timings["answer_ms"] = (time.perf_counter() - started) * 1000
        return {
            "answer": answer,
            "warnings": list(state.get("warnings", [])) + answer_warnings,
            "timings": timings,
            "answer_repairs_used": state.get("answer_repairs_used", 0)
            + int("citation_validation_retry" in answer_warnings),
        }

    def validate_answer(state: AgentState) -> dict[str, Any]:
        answer = state.get("answer")
        if answer is None:
            answer = Answer(
                "abstain",
                "Ответ не удалось безопасно сформировать.",
                warnings=["generation_unavailable"],
            )
        manifest = set(state.get("evidence_manifest", set()))
        valid = (
            answer_validator(answer, manifest)
            if answer_validator is not None
            else (
                answer.status == "abstain" and not answer.citations
                or answer.status == "answer"
                and bool(answer.citations)
                and all(item.chunk_id in manifest for item in answer.citations)
            )
        )
        if not valid:
            answer = Answer(
                "abstain",
                "Ответ не прошёл проверку источников.",
                warnings=["citation_validation_failed"],
            )
        return {
            "answer": answer,
            "status": "abstained" if answer.status == "abstain" else "completed",
        }

    graph = StateGraph(AgentState)
    graph.add_node("normalize_request", normalize_request)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("choose_action", choose_action)
    graph.add_node("tools", tool_node)
    graph.add_node("harvest_tool_result", harvest_tool_result)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer)
    graph.add_edge(START, "normalize_request")
    graph.add_edge("normalize_request", "analyze_query")
    graph.add_edge("analyze_query", "choose_action")

    def after_tools(state: AgentState) -> str:
        return "harvest_tool_result"

    def after_assess(state: AgentState) -> str:
        if state.get("tool_calls_used", 0) >= MAX_TOOL_CALLS:
            return "validate_answer"
        return "choose_action"

    graph.add_conditional_edges(
        "choose_action",
        tools_condition,
        {"tools": "tools", END: "generate_answer"},
    )
    graph.add_conditional_edges("tools", after_tools)
    graph.add_edge("harvest_tool_result", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", after_assess)
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("validate_answer", END)
    return graph.compile(checkpointer=checkpointer)
