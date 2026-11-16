"""Ограниченный граф оркестрации на LangGraph."""

from __future__ import annotations

import json
import math
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

from src.agent.coverage import (
    build_coverage,
    extract_metrics,
    normalize_metric,
)
from src.agent.metric_evidence import attach_metric_evidence
from src.agent.selector import (
    PlanAwareSelector,
    aspect_search_candidates,
    content_attempt_key,
    plan_section_path,
    requested_years,
)
from src.agent.state import AgentAction, AgentState, Answer, QueryPlan, ToolEvent
from src.agent.tools import (
    AgentTools,
    ToolUnavailable,
    chunk_from_hit,
    result_hits,
    result_warnings,
)
from src.core.introspection import supports_keyword
from src.core.record import chunk_year as _chunk_year
from src.engine.calculations import parse_number
from src.engine.structure_compare import (
    StructureAnalyzer,
    compare_content_changes,
)
from src.retrieval.hybrid_retriever import SearchFilters

#: Бюджеты — защитный механизм, а не основной способ управления исследованием.
#: ``spec_agent.md`` §3.1: не более 4 tool calls и 2 поисков за turn.
MAX_TOOL_CALLS = 4
MAX_SEARCH_CALLS = 2

#: Политика clarify: temporal/multi_hop-вопрос с показателем, но без периода в
#: самом вопросе/фильтрах/истории, не даёт корректного сравнения. Такой запрос
#: уточняется до поиска, а не превращается в abstain «no_evidence».
_EXPLICIT_YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_EXPLICIT_PERIOD = re.compile(
    r"(?:\b(?:18|19|20|21)\d{2}\s*[-–—]\s*(?:18|19|20|21)\d{2}\b|"
    r"\bс\s+(?:18|19|20|21)\d{2}\b|"
    r"\bпо\s+(?:18|19|20|21)\d{2}\b|"
    r"\bпоследн\w+\s+(?:\d+\s+)?(?:год|лет)\w*|"
    r"\bза\s+все\s+(?:годы|время)\w*|"
    r"\bвесь\s+период\b|\bпо\s+всем\s+годам\b|"
    r"\bгод\s+к\s+году\b|\bг/г\b|"
    r"\bпрошл\w+\s+год\w*|\bпозапрошл\w+\s+год\w*)",
    re.I,
)
_TREND_QUESTION = re.compile(
    r"\b(?:менял|измен|динамик|тренд|эволюц|сравн)\w*",
    re.I,
)
_STRUCTURAL_QUESTION = re.compile(
    r"(?i)\b(?:структур\w*|подраздел\w*|оглавлен\w*|дерев\w*|"
    r"раздел\w*|оглавлен\w*)"
)


def _has_explicit_period(text: str) -> bool:
    """Есть ли в тексте явный год, диапазон или относительный период."""
    return bool(_EXPLICIT_YEAR.search(text) or _EXPLICIT_PERIOD.search(text))


def _clarification_prompt(state: AgentState) -> str | None:
    """Текст уточнения периода или ``None``, если запрос можно искать.

    Уточнение запрашивается только для trend/temporal/multi_hop-вопроса с
    показателем, когда период не задан ни пользовательскими фильтрами, ни самим
    вопросом, ни историей диалога. Это защищает от подмены периода «всем
    корпусом».
    """
    plan = state.get("query_plan") or QueryPlan()
    question = state.get("question", "")
    if plan.intent not in {"temporal", "multi_hop"}:
        return None
    if not plan.required_metrics:
        return None
    if not (plan.requires_comparison or _TREND_QUESTION.search(question)):
        return None
    if requested_years(state.get("filters") or SearchFilters()):
        return None
    if _has_explicit_period(question):
        return None
    for item in state.get("history", []) or []:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if isinstance(content, str) and _has_explicit_period(content):
            return None
    return (
        "Уточните, пожалуйста, период: за какой год или диапазон лет "
        "показать динамику? Например, «за 2022–2023 годы» или «за 2023 год»."
    )



class Router(Protocol):
    """Протокол роутера: классифицирует вопрос в ``QueryPlan``."""
    def analyze(self, question: str, filters: SearchFilters) -> QueryPlan:
        """Вернуть план поиска для вопроса и явных фильтров."""
        ...


class ActionSelector(Protocol):
    """Протокол выбора следующего действия агента."""
    def choose(self, state: AgentState) -> AgentAction | None:
        """Выбрать один вызов инструмента или ``None``."""
        ...


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
    if (
        not isinstance(plan.aspects, list)
        or any(not isinstance(item, str) for item in plan.aspects)
        or not isinstance(plan.required_metrics, list)
        or not isinstance(plan.requires_comparison, bool)
    ):
        return QueryPlan()

    aspects = [
        item.strip() for item in plan.aspects if item.strip()
    ][:4]
    metrics: list[str] = []
    for item in plan.required_metrics:
        metric = normalize_metric(item)
        if metric is not None and metric not in metrics:
            metrics.append(metric)
    metrics = metrics[:5]
    year_count = len(plan.years)
    if plan.year_from is not None and plan.year_to is not None:
        year_count = max(year_count, plan.year_to - plan.year_from + 1)
    return replace(
        plan,
        aspects=aspects,
        required_metrics=metrics,
        requires_comparison=(
            plan.requires_comparison
            or plan.answer_mode == "comparison"
            or year_count > 1
        ),
    )


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
            else plan_section_path(plan.section_filter)
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


def _trace_value(value: Any, name: str) -> Any:
    """Читает поле trace из результата поиска (объект или словарь)."""
    trace = value.get("trace") if isinstance(value, Mapping) else getattr(
        value, "trace", None
    )
    if isinstance(trace, Mapping):
        return trace.get(name)
    return getattr(trace, name, None)


def _graph_added_chunk_ids(value: Any) -> set[str]:
    """ID graph-only чанков: они ждут подтверждения и не являются primary."""
    raw = _trace_value(value, "graph_added_chunk_ids")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    return {str(item) for item in raw if isinstance(item, str) and item}


def _graph_score(value: Any) -> float | None:
    """Конечная графовая оценка или ``None``, если она не измерена."""
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _graph_entity_diagnostics(result: Any) -> dict[str, Any] | None:
    """Собирает diagnostics ``get_entities`` из ``result["results"]``.

    Этап ``graph`` публикуется только по фактическим результатам: пустой
    ответ не получает выдуманных хитов или рангов.
    """
    if not isinstance(result, Mapping):
        return None
    raw_results = result.get("results")
    items = (
        [item for item in raw_results if isinstance(item, Mapping)]
        if isinstance(raw_results, (list, tuple))
        else []
    )
    raw_evidence = result.get("evidence_chunk_ids")
    evidence_ids = (
        [str(item) for item in raw_evidence if isinstance(item, str) and item]
        if isinstance(raw_evidence, (list, tuple, set, frozenset))
        else []
    )
    raw_nodes = result.get("nodes")
    nodes = (
        [str(item) for item in raw_nodes if isinstance(item, str) and item]
        if isinstance(raw_nodes, (list, tuple, set, frozenset))
        else []
    )

    stage: list[dict[str, Any]] = []
    for item in items:
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        path = item.get("path")
        path_ids = (
            [str(entry) for entry in path]
            if isinstance(path, (list, tuple))
            else []
        )
        item_evidence = item.get("evidence_chunk_ids")
        evidence = (
            [str(entry) for entry in item_evidence]
            if isinstance(item_evidence, (list, tuple, set, frozenset))
            else []
        )
        stage.append(
            {
                "chunk_id": chunk_id,
                "rank": len(stage) + 1,
                "score": _graph_score(item.get("graph_score")),
                "path": path_ids,
                "evidence_chunk_ids": evidence,
            }
        )

    if not stage and not evidence_ids and not nodes:
        # Совсем пустой результат: не публикуем синтетический этап.
        return None

    added_ids = {entry["chunk_id"] for entry in stage} | set(evidence_ids)
    candidates = len(stage) if stage else len(evidence_ids)
    measured = bool(stage or evidence_ids)
    return {
        "stages": {"graph": stage} if stage else {},
        "available": {"graph": measured},
        "fallback": {"graph": not measured},
        "unavailable": {} if measured else {"graph": "no_candidates"},
        "counts": {"graph": candidates},
        "graph_candidates": candidates,
        "graph_added": len(added_ids),
        "graph_matched_entities": nodes,
    }


def _debug_answer_node_failure(exc: BaseException) -> None:
    """Записать падение узла ответа при ``RAG_DEBUG_GENERATION=1``."""
    import os
    import traceback
    from pathlib import Path

    if os.getenv("RAG_DEBUG_GENERATION") != "1":
        return
    try:
        target = Path("logs") / "graph_debug.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    except OSError:
        pass


def _contained_chunk_id(chunk: Any) -> str | None:
    if isinstance(chunk, Mapping):
        value = chunk.get("chunk_id")
    else:
        value = getattr(chunk, "chunk_id", None)
    return value if isinstance(value, str) and value else None


def _year_from_doc_id(doc_id: Any) -> int | None:
    if not isinstance(doc_id, str):
        return None
    match = re.search(r"(?:18|19|20|21)\d{2}", doc_id)
    return int(match.group(0)) if match is not None else None


def _filters_dict(filters: SearchFilters) -> dict[str, Any]:
    return filters.to_payload()


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

    base_years = requested_years(effective)
    action_years = requested_years(action_filters)
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


def _coverage_has_actionable_gaps(state: AgentState, coverage: Any) -> bool:
    """Есть ли gap, который можно закрыть новым поиском с учётом попыток."""
    missing_cells = getattr(coverage, "missing_cells", None)
    if not callable(missing_cells):
        return False
    attempted_years: set[int] = set()
    attempted_pairs: set[tuple[str, int]] = set()
    for attempt in state.get("search_attempts", []):
        if not isinstance(attempt, Mapping):
            continue
        year = attempt.get("year")
        if not isinstance(year, int) or isinstance(year, bool):
            continue
        attempted_years.add(year)
        metric = attempt.get("metric")
        if isinstance(metric, str) and metric:
            attempted_pairs.add((normalize_metric(metric) or metric, year))
    for cell in missing_cells():
        if getattr(cell, "status", None) == "year_absent":
            continue
        year = getattr(cell, "year", None)
        metric = getattr(cell, "metric", "")
        if metric:
            if isinstance(year, int) and (metric, year) not in attempted_pairs:
                return True
            continue
        if isinstance(year, int) and year not in attempted_years:
            return True
    is_year_absent = getattr(coverage, "is_year_absent", None)
    for year in getattr(coverage, "uncovered_years", []):
        if year in attempted_years:
            continue
        if callable(is_year_absent) and is_year_absent(year):
            continue
        return True
    return False


def _after_assess_decision(
    state: AgentState,
    *,
    max_search_calls: int = MAX_SEARCH_CALLS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> str:
    """Маршрутизация после coverage: генерация или ещё один адресный поиск."""
    plan = state.get("query_plan") or QueryPlan()
    if plan.intent == "structural":
        # Структурный сценарий использует отдельные инструменты и не
        # описывается metric×year-матрицей; сохраняем прежний цикл.
        return "choose_action"
    budget = (
        state.get("search_calls_used", 0) < max_search_calls
        and state.get("tool_calls_used", 0) < max_tool_calls
    )
    # Аспект запроса может быть не покрыт даже при полном year-coverage:
    # длинный вопрос размывает поиск, поэтому нужен один адресный добивочный
    # поиск по формулировке аспекта.
    if budget and aspect_search_candidates(state, plan):
        return "choose_action"
    coverage = state.get("coverage")
    status = getattr(coverage, "status", None)
    if status is None:
        return "choose_action"
    if status in {"complete", "not_required", "insufficient"}:
        return "generate_answer"
    if status == "partial":
        if budget and _coverage_has_actionable_gaps(state, coverage):
            return "choose_action"
        return "generate_answer"
    return "generate_answer"


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
        """Детерминированный fallback: пустой план без LLM."""
        del filters
        return QueryPlan()


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
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_search_calls: int = MAX_SEARCH_CALLS,
):
    """Строит StateGraph с лимитами на вызовы инструментов и поиска.

    ``max_tool_calls``/``max_search_calls`` приходят из ``RAGSettings``;
    модульные константы остаются безопасным fallback для прямых вызовов.
    """
    if max_tool_calls < 1 or max_search_calls < 1:
        raise ValueError("agent budgets must be positive")
    router = router or _DefaultRouter()
    action_selector = action_selector or PlanAwareSelector()
    structure_analyzer = structure_analyzer or StructureAnalyzer()

    class SearchInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1)
        queries: list[str] | None = None
        filters: dict[str, Any] | None = None
        retrieval_mode: str = "hybrid"
        #: Граф — опциональная надстройка над гибридным поиском (P7).
        use_graph: bool = False
        top_k: int = 5
        #: Внутренняя подсказка селектора: включить реранкер на каждый год
        #: в temporal fan-out (PerYearSearch).
        rerank_per_year: bool | None = None
        #: Внутренняя подсказка селектора для адресного поиска (metric, year).
        metric: str | None = None

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
        retrieval_mode: str = "hybrid", use_graph: bool = False,
        top_k: int = 5, metric: str | None = None,
        rerank_per_year: bool | None = None,
    ) -> str:
        """Ищет по корпусу отчётов с проверенными фильтрами."""
        del metric  # внутренняя подсказка селектора, backend её не получает
        return wrapped(
            "search_reports",
            {
                "query": query, "queries": queries,
                "filters": filters_from_dict(filters),
                "retrieval_mode": retrieval_mode, "use_graph": use_graph,
                "top_k": top_k, "rerank_per_year": rerank_per_year,
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
        # Перенесённые цитаты прошлого хода уже покрывают свой год; без этого
        # сравнительный вопрос заново искал бы год, который есть в evidence,
        # и тратил бы бюджет на повторный широкий поиск.
        covered = sorted(
            {
                year
                for chunk in (state.get("evidence") or {}).values()
                if isinstance(
                    year := _chunk_year(chunk), int
                )
            }
        )
        return {
            "question": question,
            "filters": normalized,
            "effective_filters": normalized,
            "covered_years": covered,
        }

    def analyze_query(state: AgentState) -> dict[str, Any]:
        try:
            target = router.analyze if hasattr(router, "analyze") else router
            kwargs = {}
            if supports_keyword(target, "history"):
                kwargs["history"] = list(state.get("history", []))
            result = target(state["question"], state["filters"], **kwargs)
            plan = _as_plan(result)
            # Follow-up ??? ??????? (?? ?? 2024 ?????) ????????? ??????? ??
            # ???????: ????? ????? ???? ??? coverage ? ????? ?????? ? abstain.
            if not plan.required_metrics:
                from src.agent.coverage import extract_metrics

                # 1) ??????? ????? ? ???????; 2) follow-up ????????? ?? ??
                # ?????????? ???????? ???????????? (?? ?? ??????? ??????????:
                # ??? ??????????? ??? ??????? ?????? ? ???? ??????????????).
                inherited = extract_metrics(str(state.get("question") or ""))
                if not inherited:
                    user_text = " ".join(
                        str(item.get("content") or "")
                        for item in list(state.get("history", []))[-6:]
                        if isinstance(item, Mapping)
                        and item.get("role") == "user"
                    )
                    inherited = extract_metrics(user_text) if user_text else []
                if inherited:
                    plan = replace(plan, required_metrics=inherited[:5])
            if (
                plan.intent == "structural"
                and not _STRUCTURAL_QUESTION.search(state["question"])
            ):
                # Structural intent включает дорогой обход деревьев разделов.
                # Если вопрос не про структуру/раздел, это ложная
                # классификация роутера: обычный поиск по содержанию дешевле
                # и не подменяет период/раздел.
                plan = replace(
                    plan,
                    intent=(
                        "multi_hop"
                        if plan.requires_comparison or plan.answer_mode == "comparison"
                        else "temporal"
                    ),
                )
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
                    requested_years(state.get("filters") or SearchFilters())
                ),
                "rewritten_queries": [],
                "warnings": state.get("warnings", []) + ["query_analysis_fallback"],
            }
        effective = _filters_with_plan(state["filters"], plan)
        # Явные фильтры пользователя уже учтены в effective и важнее плана.
        expected_years = requested_years(effective) or set(plan.years)
        if plan.requires_comparison or plan.answer_mode == "comparison":
            # Follow-up сравнение («а каким он был в 2022?») должно помнить
            # год предыдущего ответа, иначе «насколько вырос» не с чем
            # сравнить. Годы берутся из уже подтверждённого evidence.
            carried_years = {
                year
                for chunk in (state.get("evidence") or {}).values()
                if (year := _chunk_year(chunk)) is not None
            }
            if carried_years:
                merged_years = sorted(set(plan.years) | carried_years)[:5]
                plan = replace(plan, years=merged_years)
                effective = _filters_with_plan(state["filters"], plan)
                expected_years = requested_years(effective) or set(plan.years)
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
                if supports_keyword(target, "history"):
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

    def clarify_period(state: AgentState) -> dict[str, Any]:
        """Фиксирует необходимость уточнения периода до любого поиска."""
        text = _clarification_prompt(state)
        if text is None:
            return {}
        emit("clarification_required", {"reason": "missing_period"})
        return {"clarification": text, "status": "clarification_required"}

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
        if calls >= max_tool_calls or (
            action.tool_name == "search_reports" and searches >= max_search_calls
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
        tool_args = dict(args)
        # ``metric`` — внутренняя подсказка селектора, инструменту она не нужна.
        tool_args.pop("metric", None)
        if isinstance(tool_args.get("filters"), SearchFilters):
            tool_args["filters"] = {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in vars(tool_args["filters"]).items()
            }
        call = {
            "action": AgentAction(action.tool_name, args, action.reason_code),
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": action.tool_name,
                        "args": tool_args,
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
            # query/queries/entity_names — пользовательский текст, а не секрет
            # (spec_benchmark §4.5); секреты и ключи сюда не попадают.
            key: ("[redacted]" if key == "chunk_id" else value)
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
        elif action.tool_name == "get_entities" and isinstance(result, Mapping):
            graph_results = result.get("results")
            if isinstance(graph_results, (list, tuple)):
                count = len(graph_results)
            else:
                evidence_ids = result.get("evidence_chunk_ids")
                count = (
                    len(evidence_ids)
                    if isinstance(evidence_ids, (list, tuple, set, frozenset))
                    else 0
                )
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
        elif action.tool_name == "get_entities" and result is not None:
            diagnostics = _graph_entity_diagnostics(result)
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
            # Graph-only чанки возвращаются вместе с обычными хитами, но
            # подтверждены только графом: они ждут get_chunk и не участвуют
            # в citations, пока не станут primary.
            graph_only_ids = _graph_added_chunk_ids(result)
            for hit in result_hits(result):
                chunk_id, chunk = chunk_from_hit(hit)
                inner_id = _contained_chunk_id(chunk)
                if not chunk_id or (inner_id is not None and inner_id != chunk_id):
                    continue
                if chunk_id in graph_only_ids:
                    if (
                        chunk_id not in manifest
                        and chunk_id not in search_manifest
                    ):
                        graph_manifest.add(chunk_id)
                    continue
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
        wanted_years = requested_years(state.get("effective_filters", SearchFilters()))
        if len(wanted_years) > 1 and wanted_years - set(covered_years):
            warnings.append("incomplete_period")

        plan = state.get("query_plan") or QueryPlan()
        if action.tool_name == "search_reports":
            filters = arguments.get("filters")
            if isinstance(filters, SearchFilters):
                filters = _filters_dict(filters)
            requested_now = requested_years(filters)
            metric = arguments.get("metric")
            metric = metric if isinstance(metric, str) and metric else None
            if metric:
                kind = "metric_year"
            elif (
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
            attempt = {
                "kind": kind,
                "year": (
                    next(iter(requested_now))
                    if len(requested_now) == 1
                    else None
                ),
                "years": sorted(requested_now),
                "section_path": (
                    None
                    if metric
                    else (
                        filters.get("section_path_prefix")
                        if isinstance(filters, Mapping)
                        else None
                    )
                ),
                "result_count": count,
                "covered_years": covered_years,
            }
            if metric:
                attempt["metric"] = metric
            search_attempts.append(attempt)
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

        if calls + 1 >= max_tool_calls:
            warnings.append("tool_budget_exhausted")
        attach_metric_evidence(
            tools,
            plan,
            evidence,
            manifest,
            search_manifest if action.tool_name == "search_reports" else None,
        )
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
            expected = requested_years(effective)
        required_metrics = list(plan.required_metrics) or extract_metrics(
            state.get("question", "")
        )
        available_years = set(state.get("available_documents", {}).keys())
        coverage = build_coverage(
            sorted(expected),
            required_metrics,
            evidence,
            available_years,
        )
        covered = set(coverage.covered_years)
        uncovered = sorted(expected - covered)
        warnings = list(state.get("warnings", []))
        updates: dict[str, Any] = {
            "expected_years": sorted(expected),
            "covered_years": sorted(covered),
            # ``missing_years`` — только годы, которых нет в корпусе.
            # Незакрытая metric-ячейка не делает год отсутствующим.
            "missing_years": list(coverage.missing_years),
            "coverage": coverage,
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

        if plan.answer_mode == "comparison" and uncovered:
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
        clarification = state.get("clarification")
        if clarification:
            # Политика clarify: без периода сравнение было бы подменой
            # (весь корпус вместо запрошенного пользователем). Поиск не нужен.
            answer = Answer(
                "abstain",
                clarification,
                warnings=["clarification_required"],
            )
            answer_warnings: list[str] = []
        elif answer_generator is None:
            answer = Answer(
                "abstain",
                "Ответ не сформирован: генератор ответа не настроен.",
                warnings=["generation_unavailable"],
            )
            answer_warnings: list[str] = []
        else:
            try:
                answer, answer_warnings = answer_generator(state)
            except Exception as exc:
                _debug_answer_node_failure(exc)
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
    graph.add_node("clarify_period", clarify_period)
    graph.add_node("choose_action", choose_action)
    graph.add_node("tools", tool_node)
    graph.add_node("harvest_tool_result", harvest_tool_result)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer)
    graph.add_edge(START, "normalize_request")
    graph.add_edge("normalize_request", "analyze_query")
    graph.add_edge("analyze_query", "clarify_period")

    def after_clarify(state: AgentState) -> str:
        return "generate_answer" if state.get("clarification") else "choose_action"

    graph.add_conditional_edges(
        "clarify_period",
        after_clarify,
        {"generate_answer": "generate_answer", "choose_action": "choose_action"},
    )

    def after_tools(state: AgentState) -> str:
        return "harvest_tool_result"

    def after_assess(state: AgentState) -> str:
        # Coverage управляет переходом только для plan-aware селектора.
        # Пользовательский selector может намеренно выстраивать цепочку
        # инструментов; его ограничивают бюджеты choose_action.
        if not isinstance(action_selector, PlanAwareSelector):
            return "choose_action"
        # Полный/неприменимый отчёт сразу уходит в генерацию, partial — в
        # селектор только при наличии actionable gap и бюджета. Если
        # choose_action вернёт None, tools_condition уведёт граф в
        # generate_answer.
        return _after_assess_decision(
            state,
            max_search_calls=max_search_calls,
            max_tool_calls=max_tool_calls,
        )

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
