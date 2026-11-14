"""Ограниченный граф оркестрации на LangGraph."""

from __future__ import annotations

import json
import inspect
import re
import time
import uuid
from dataclasses import asdict, is_dataclass, replace
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agent.selector import PlanAwareSelector, content_attempt_key
from src.agent.state import AgentAction, AgentState, Answer, QueryPlan, ToolEvent
from src.agent.tools import (
    AgentTools,
    ToolUnavailable,
    chunk_from_hit,
    result_hits,
    result_warnings,
)
from src.engine.calculations import parse_number
from src.engine.structure_compare import (
    StructureAnalyzer,
    compare_content_changes,
)
from src.retrieval.hybrid_retriever import SearchFilters

#: Бюджеты — защитный механизм, а не основной способ управления исследованием.
MAX_TOOL_CALLS = 8
MAX_SEARCH_CALLS = 5

_METRIC_TERMS = (
    "прибыль",
    "выручка",
    "roe",
    "roa",
    "актив",
    "капитал",
    "доход",
    "расход",
    "маржа",
)


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
    """Сужает незаданные пользователем фильтры по плану роутера, не меняя его выбор."""
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
        # Если вывод роутера конфликтует с границей пользователя, сужаем
        # диапазон до пустого, а не ослабляем или переворачиваем его.
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
    """Собирает ID только из явных полей evidence в результате графа."""
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


def _chunk_year(chunk: Any) -> int | None:
    if isinstance(chunk, Mapping):
        value = chunk.get("year")
    else:
        value = getattr(chunk, "year", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _year_from_doc_id(doc_id: Any) -> int | None:
    if not isinstance(doc_id, str):
        return None
    match = re.search(r"(?:18|19|20|21)\d{2}", doc_id)
    return int(match.group(0)) if match is not None else None


def _requested_years(value: Any) -> set[int]:
    if isinstance(value, SearchFilters):
        if value.years:
            return set(value.years)
        start = value.year_from if value.year_from is not None else value.year_to
        end = value.year_to if value.year_to is not None else value.year_from
        if start is None or end is None:
            return set()
        return set(range(min(start, end), max(start, end) + 1))
    if not isinstance(value, Mapping):
        return set()
    years = value.get("years")
    if isinstance(years, (list, tuple, set)):
        return {
            int(year)
            for year in years
            if isinstance(year, int) and not isinstance(year, bool)
        }
    start = value.get("year_from")
    end = value.get("year_to")
    if not isinstance(start, int) or isinstance(start, bool):
        start = None
    if not isinstance(end, int) or isinstance(end, bool):
        end = None
    if start is None and end is None:
        return set()
    start = start if start is not None else end
    end = end if end is not None else start
    return set(range(min(start, end), max(start, end) + 1))


def _filters_dict(filters: SearchFilters) -> dict[str, Any]:
    return {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in vars(filters).items()
    }


def _intersect(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, set) and isinstance(right, set):
        result = left & right
        return result if result else left
    return left


def _narrower_prefix(base: str | None, action: str | None) -> str | None:
    if base is None:
        return action
    if action is None:
        return base
    base_text = base.strip("/").casefold()
    action_text = action.strip("/").casefold()
    if action_text == base_text or action_text.startswith(base_text + "/"):
        return action
    return base


def _merge_action_filters(
    effective: SearchFilters,
    action_filters: Any,
) -> SearchFilters:
    """Сужает пользовательские/плановые фильтры, но никогда не расширяет их."""
    if not isinstance(action_filters, SearchFilters):
        if not isinstance(action_filters, Mapping):
            return effective
        allowed = set(SearchFilters.__dataclass_fields__)
        try:
            action_filters = SearchFilters(
                **{
                    key: value
                    for key, value in action_filters.items()
                    if key in allowed
                }
            ).normalized()
        except (TypeError, ValueError):
            return effective

    if action_filters.normalized() == effective.normalized():
        return effective

    base_years = _requested_years(effective)
    action_years = _requested_years(action_filters)
    if base_years and action_years:
        years = base_years & action_years
        if not years:
            years = base_years
    elif action_years:
        years = action_years
    else:
        years = base_years

    year_from = effective.year_from
    year_to = effective.year_to
    if years:
        year_from = year_to = None

    return SearchFilters(
        years=set(years) if years else None,
        year_from=year_from,
        year_to=year_to,
        report_periods=_intersect(
            effective.report_periods, action_filters.report_periods
        ),
        issuers=_intersect(effective.issuers, action_filters.issuers),
        kinds=_intersect(effective.kinds, action_filters.kinds),
        section_path_prefix=_narrower_prefix(
            effective.section_path_prefix,
            action_filters.section_path_prefix,
        ),
        breadcrumb_contains=(
            effective.breadcrumb_contains or action_filters.breadcrumb_contains
        ),
        max_section_level=(
            action_filters.max_section_level
            if effective.max_section_level is None
            else effective.max_section_level
            if action_filters.max_section_level is None
            else min(
                effective.max_section_level, action_filters.max_section_level
            )
        ),
        allowed_doc_ids=_intersect(
            effective.allowed_doc_ids, action_filters.allowed_doc_ids
        ),
    ).normalized()


def _fact_year(fact: Any) -> int | None:
    if not isinstance(fact, Mapping):
        return None
    value = fact.get("year", fact.get("period"))
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = re.search(r"\b(18|19|20|21)\d{2}\b", value)
        if match is not None:
            return int(match.group(0))
    return None


def _metric_terms(question: str) -> list[str]:
    text = question.casefold()
    return [term for term in _METRIC_TERMS if term in text]


def _chunk_mentions_metric(chunk: Any, terms: list[str]) -> bool:
    if not terms:
        return True
    if isinstance(chunk, Mapping):
        text = f"{chunk.get('text', '')} {chunk.get('html', '')}"
    else:
        text = f"{getattr(chunk, 'text', '')} {getattr(chunk, 'html', '')}"
    normalized = str(text).casefold()
    return any(term in normalized for term in terms)


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
    structure_analyzer: StructureAnalyzer | None = None,
    query_rewriter: Any = None,
):
    """Строит StateGraph с лимитами на вызовы инструментов и поиска."""
    router = router or _DefaultRouter()
    action_selector = action_selector or PlanAwareSelector()
    structure_analyzer = structure_analyzer or StructureAnalyzer()

    class SearchInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1)
        queries: list[str] | None = None
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
            # Никогда не помещаем текст исключений или тела ответов провайдера в сообщения.
            return encode_result({
                "_agent_status": "failed",
                "warning": "tool_failed",
                "_duration_ms": (time.perf_counter() - started) * 1000,
            })

    @tool("search_reports", args_schema=SearchInput)
    def search_reports(
        query: str, queries: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_mode: str = "hybrid", top_k: int = 5,
    ) -> str:
        """Ищет по корпусу отчётов с проверенными фильтрами."""
        return wrapped(
            "search_reports",
            {
                "query": query, "queries": queries,
                "filters": filters_from_dict(filters),
                "retrieval_mode": retrieval_mode, "top_k": top_k,
            },
        )

    @tool("get_structure_tree", args_schema=StructureInput)
    def get_structure_tree(
        doc_id: str, section_path: str | None = None, max_depth: int = 4,
    ) -> str:
        """Читает ограниченное дерево разделов локального документа."""
        return wrapped(
            "get_structure_tree",
            {"doc_id": doc_id, "section_path": section_path, "max_depth": max_depth},
        )

    @tool("get_entities", args_schema=EntitiesInput)
    def get_entities(
        entity_names: list[str], filters: dict[str, Any] | None = None,
        max_hops: int = 2,
    ) -> str:
        """Ищет ограниченные пути сущностей вместе с исходными свидетельствами."""
        return wrapped(
            "get_entities",
            {
                "entity_names": entity_names, "filters": filters_from_dict(filters),
                "max_hops": max_hops,
            },
        )

    @tool("get_chunk", args_schema=ChunkInput)
    def get_chunk(chunk_id: str) -> str:
        """Читает один чанк по его проиндексированному ``chunk_id``."""
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
            # Никогда не раскрываем текст исключений и тела ответов провайдера.
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
                "expected_years": sorted(
                    _requested_years(state.get("filters") or SearchFilters())
                ),
                "rewritten_queries": [],
                "warnings": state.get("warnings", []) + ["query_analysis_fallback"],
            }
        effective = _filters_with_plan(state["filters"], plan)
        # Явные фильтры пользователя уже учтены в effective и важнее плана.
        expected_years = _requested_years(effective) or set(plan.years)
        rewritten_queries: list[str] = []
        analysis_warnings: list[str] = []
        if plan.needs_rewriting and query_rewriter is not None:
            try:
                target = (
                    query_rewriter.rewrite
                    if hasattr(query_rewriter, "rewrite")
                    else query_rewriter
                )
                rewrite_kwargs = {}
                if _supports_keyword(target, "history"):
                    rewrite_kwargs["history"] = list(
                        state.get("history", [])
                    )
                variants = target(state["question"], **rewrite_kwargs)
                if isinstance(variants, (list, tuple)):
                    rewritten_queries = [
                        item
                        for item in variants
                        if isinstance(item, str) and item.strip()
                    ][:2]
            except Exception:
                analysis_warnings.append("query_rewrite_fallback")
        emit(
            "analysis_completed",
            {"intent": plan.intent, "effective_filters": effective},
        )
        return {
            "query_plan": plan,
            "effective_filters": effective,
            "expected_years": sorted(expected_years),
            "rewritten_queries": rewritten_queries,
            "warnings": list(state.get("warnings", [])) + analysis_warnings,
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
            # Сломанный роутер/селектор не может запустить произвольный код.
            if not state.get("unknown_action_fallback_used"):
                # Продолжаем безопасным поиском через штатный ToolNode.
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
            merged = _merge_action_filters(
                state["effective_filters"], arguments.get("filters")
            )
            arguments["filters"] = _filters_dict(merged)
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
        # Штатный протокол LangGraph ToolNode/tools_condition выполняет
        # проверенное решение; собственный цикл вызова инструментов не нужен.
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
                key: ("[redacted]" if key in {
                    "query", "queries", "entity_names", "chunk_id"
                } else value)
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
            key: ("[redacted]" if key in {
                "query", "queries", "entity_names", "chunk_id"
            } else value)
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
        if action.tool_name == "search_reports":
            count = len(result_hits(result))
        elif action.tool_name == "get_structure_tree" and isinstance(
            result, Mapping
        ):
            count = len(result.get("sections", []) or [])
        else:
            count = (
                len(result)
                if isinstance(result, (list, tuple))
                else int(result is not None)
            )
        diagnostics: dict[str, Any] | None = None
        if action.tool_name == "search_reports" and result is not None:
            if isinstance(result, Mapping):
                trace = result.get("trace")
                if isinstance(trace, Mapping):
                    diagnostics = dict(trace)
            else:
                trace = getattr(result, "trace", None)
                if is_dataclass(trace):
                    diagnostics = asdict(trace)
        event = ToolEvent(
            tool_name=action.tool_name,
            arguments=short_args,
            status=status,
            result_count=count,
            duration_ms=duration,
            warning=warning,
            diagnostics=diagnostics,
        )
        results = list(state.get("retrieval_results", []))
        evidence = dict(state.get("evidence", {}))
        manifest = set(state.get("evidence_manifest", set()))
        search_manifest = set(state.get("search_evidence_manifest", set()))
        graph_manifest = set(state.get("graph_evidence_manifest", set()))
        structure_documents = dict(state.get("structure_documents", {}))
        structure_snapshots = dict(state.get("structure_snapshots", {}))
        search_attempts = list(state.get("search_attempts", []))
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
        elif action.tool_name == "get_structure_tree" and result is not None:
            if isinstance(result, Mapping):
                doc_id = result.get("doc_id") or arguments.get("doc_id")
                sections = result.get("sections")
                year = _year_from_doc_id(doc_id)
                if isinstance(year, int) and isinstance(sections, list):
                    structure_documents[year] = str(doc_id)
                    structure_snapshots[year] = list(sections)
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

        covered_years = sorted(
            {
                year
                for chunk in evidence.values()
                if (year := _chunk_year(chunk)) is not None
            }
        )
        requested_years = _requested_years(state.get("effective_filters", SearchFilters()))
        if len(requested_years) > 1 and requested_years - set(covered_years):
            warnings.append("incomplete_period")

        plan = state.get("query_plan") or QueryPlan()
        if action.tool_name == "search_reports":
            filters = arguments.get("filters")
            if isinstance(filters, SearchFilters):
                filters = _filters_dict(filters)
            requested_now = _requested_years(filters)
            if (
                plan.intent == "structural"
                and action.reason_code == "inspect_evidence"
            ):
                kind = "structure_content"
            elif searches == 0:
                kind = "broad"
            elif len(requested_now) == 1:
                kind = "year"
            else:
                kind = "broad"
            search_attempts.append(
                {
                    "kind": kind,
                    "year": (
                        next(iter(requested_now))
                        if len(requested_now) == 1
                        else None
                    ),
                    "years": sorted(requested_now),
                    "section_path": (
                        filters.get("section_path_prefix")
                        if isinstance(filters, Mapping)
                        else None
                    ),
                    "result_count": count,
                    "covered_years": covered_years,
                }
            )
        elif action.tool_name == "get_structure_tree":
            search_attempts.append(
                {
                    "kind": "structure",
                    "year": _year_from_doc_id(arguments.get("doc_id")),
                    "doc_id": arguments.get("doc_id"),
                    "result_count": count,
                }
            )
        elif action.tool_name == "get_entities":
            search_attempts.append(
                {"kind": "graph", "result_count": count}
            )

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
            "structure_documents": structure_documents,
            "structure_snapshots": structure_snapshots,
            "search_attempts": search_attempts,
            "covered_years": covered_years,
            "warnings": warnings,
            "tool_calls_used": calls + 1,
            "search_calls_used": searches + int(action.tool_name == "search_reports"),
        }

    def assess_evidence(state: AgentState) -> dict[str, Any]:
        # Селектор намеренно опрашивается заново после каждого результата:
        # он может выбрать ограниченное продолжение (например, структура → поиск).
        plan = state.get("query_plan") or QueryPlan()
        effective = state.get("effective_filters") or SearchFilters()
        evidence = state.get("evidence", {})
        expected = set(state.get("expected_years", []))
        if not expected:
            expected = set(plan.years)
        if not expected and plan.year_from is not None and plan.year_to is not None:
            expected = set(range(plan.year_from, plan.year_to + 1))
        if not expected:
            expected = _requested_years(effective)
        terms = _metric_terms(state.get("question", ""))
        covered = {
            year
            for chunk in evidence.values()
            if _chunk_mentions_metric(chunk, terms)
            and (year := _chunk_year(chunk)) is not None
        }
        missing = sorted(expected - covered)
        warnings = list(state.get("warnings", []))
        updates: dict[str, Any] = {
            "expected_years": sorted(expected),
            "covered_years": sorted(covered),
            "missing_years": missing,
        }

        if plan.intent == "structural" and not state.get(
            "structure_comparison_complete"
        ):
            snapshots = state.get("structure_snapshots", {})
            documents = state.get("available_documents", {})
            attempts = state.get("search_attempts", [])
            attempted_structure = {
                item.get("year")
                for item in attempts
                if isinstance(item, dict) and item.get("kind") == "structure"
            }
            target_years = sorted(expected) or sorted(documents) or sorted(snapshots)
            unresolved = [
                year
                for year in target_years
                if year not in snapshots
                and year not in attempted_structure
                and documents.get(year)
            ]
            if not unresolved:
                if len(snapshots) >= 2:
                    analysis = structure_analyzer.compare(
                        snapshots, sorted(snapshots)
                    )
                    updates.update(analysis)
                updates["structure_comparison_complete"] = True
                if len(target_years) > len(snapshots):
                    warnings.append("structure_scope_limited")
                if not snapshots and target_years:
                    warnings.append("structure_unavailable")

        matches = updates.get(
            "section_matches", state.get("section_matches", [])
        )
        if matches and evidence:
            found_content = compare_content_changes(matches, evidence)
            existing_content = list(state.get("content_changes", []))
            known_content = {
                (
                    item.get("from_year"),
                    item.get("to_year"),
                    (item.get("to_section") or {}).get("section_path"),
                    item.get("change_type"),
                )
                for item in existing_content
                if isinstance(item, Mapping)
            }
            new_content = [
                item
                for item in found_content
                if (
                    item.get("from_year"),
                    item.get("to_year"),
                    (item.get("to_section") or {}).get("section_path"),
                    item.get("change_type"),
                )
                not in known_content
            ]
            if new_content:
                updates["content_changes"] = existing_content + new_content
                structural = list(
                    updates.get(
                        "structural_changes",
                        state.get("structural_changes", []),
                    )
                )
                structural.extend(new_content)
                updates["structural_changes"] = structural

        if plan.answer_mode == "comparison" and missing:
            updates["status"] = "evidence_incomplete"
        elif evidence:
            updates["status"] = "evidence_available"
        else:
            updates["status"] = "evidence_pending"
        updates["warnings"] = warnings
        return updates

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
                # Ошибки генератора/провайдера не попадают в состояние и UI.
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
        elif answer.status == "answer":
            extra: list[str] = []
            plan = state.get("query_plan") or QueryPlan()
            expected = set(state.get("expected_years", [])) or set(plan.years)
            facts = [fact for fact in answer.facts if isinstance(fact, Mapping)]
            fact_years = {
                year
                for fact in facts
                if (year := _fact_year(fact)) is not None
            }
            if (
                plan.answer_mode == "comparison"
                and len(expected) > 1
                and facts
                and len(fact_years) < 2
            ):
                extra.append("incomplete_period")
            if expected and fact_years and expected - fact_years:
                extra.append("incomplete_period")

            values: dict[tuple[str, int], str] = {}
            units: dict[str, set[str]] = {}
            for fact in facts:
                metric = fact.get("metric", fact.get("claim"))
                year = _fact_year(fact)
                if not isinstance(metric, str) or year is None:
                    continue
                normalized_metric = metric.strip().casefold()
                unit = fact.get("unit")
                if isinstance(unit, str) and unit.strip():
                    units.setdefault(normalized_metric, set()).add(
                        unit.strip().casefold()
                    )
                if "value" in fact:
                    key = (normalized_metric, year)
                    parsed = parse_number(fact.get("value"))
                    value = str(parsed) if parsed is not None else str(
                        fact.get("value")
                    )
                    previous = values.get(key)
                    if previous is not None and previous != value:
                        extra.append("conflicting_values")
                    values[key] = value
            if any(len(items) > 1 for items in units.values()):
                extra.append("unit_mismatch")
            if plan.intent == "structural":
                if not state.get("structure_comparison_complete"):
                    extra.append("structure_unverified")
                checked = {
                    content_attempt_key(item)
                    for item in state.get("search_attempts", [])
                    if isinstance(item, dict)
                    and item.get("kind") == "structure_content"
                }
                unverified = []
                for change in state.get("structural_changes", []):
                    if not isinstance(change, Mapping):
                        continue
                    if change.get("change_type") == "stable":
                        continue
                    if change.get("change_type") == "removed":
                        year = change.get("from_year")
                        section = change.get("from_section")
                    else:
                        year = change.get("to_year") or change.get("from_year")
                        section = change.get("to_section") or change.get(
                            "from_section"
                        )
                    path = (
                        section.get("section_path")
                        if isinstance(section, Mapping)
                        else None
                    )
                    verified = any(
                        path == checked_path
                        and (not years or year in years)
                        for years, checked_path in checked
                    )
                    if not verified:
                        unverified.append(change)
                if unverified:
                    extra.append("structure_content_unverified")
            if extra:
                answer = replace(
                    answer,
                    warnings=list(
                        dict.fromkeys(list(answer.warnings) + extra)
                    ),
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
        # Даже при исчерпанном бюджете сначала пытаемся сформировать ответ по
        # уже собранным evidence; choose_action сам вернёт None и уведёт граф
        # в generate_answer.
        del state
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
