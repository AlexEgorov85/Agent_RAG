"""Селектор действий: несколько осмысленных поисков вместо одного top-k.

Селектор не вызывает инструменты напрямую. Он только предлагает следующее
действие, а существующий StateGraph исполняет его через ``ToolNode``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.agent.state import AgentAction, AgentState, QueryPlan
from src.retrieval.hybrid_retriever import SearchFilters

DEFAULT_MAX_SEARCHES = 5
DEFAULT_MAX_STRUCTURE_DOCS = 5
DEFAULT_MAX_CONTENT_SEARCHES = 5


def _requested_years(filters: SearchFilters) -> set[int]:
    if filters.years:
        return set(filters.years)
    if filters.year_from is not None or filters.year_to is not None:
        start = filters.year_from if filters.year_from is not None else filters.year_to
        end = filters.year_to if filters.year_to is not None else filters.year_from
        if start is None or end is None:
            return set()
        return set(range(min(start, end), max(start, end) + 1))
    return set()


def _target_years(state: AgentState, plan: QueryPlan) -> list[int]:
    years = set(state.get("expected_years", []))
    if not years:
        years = set(plan.years)
    if not years and plan.year_from is not None and plan.year_to is not None:
        years = set(range(plan.year_from, plan.year_to + 1))
    if not years and plan.intent in {"structural", "temporal", "multi_hop"}:
        years.update(state.get("available_documents", {}).keys())
    if not years:
        years.update(
            _requested_years(
                state.get("effective_filters") or SearchFilters()
            )
        )
    return sorted(years)


def _attempted_years(state: AgentState) -> set[int]:
    years: set[int] = set()
    for attempt in state.get("search_attempts", []):
        if not isinstance(attempt, dict):
            continue
        year = attempt.get("year")
        if isinstance(year, int):
            years.add(year)
    return years


def _query_for_year(question: str, year: int) -> str:
    if str(year) in question:
        return question
    return f"{question} {year} год"


def content_attempt_key(item: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
    years = item.get("years")
    if isinstance(years, (list, tuple, set)):
        normalized = tuple(
            sorted(
                int(year)
                for year in years
                if isinstance(year, int) and not isinstance(year, bool)
            )
        )
    else:
        year = item.get("year")
        normalized = (
            (year,)
            if isinstance(year, int) and not isinstance(year, bool)
            else ()
        )
    return normalized, item.get("section_path")


class PlanAwareSelector:
    """Использует QueryPlan и уже собранные evidence для следующего шага."""

    def __init__(
        self,
        *,
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_structure_docs: int = DEFAULT_MAX_STRUCTURE_DOCS,
        max_content_searches: int = DEFAULT_MAX_CONTENT_SEARCHES,
    ) -> None:
        if min(max_searches, max_structure_docs, max_content_searches) < 1:
            raise ValueError("selector limits must be positive")
        self.max_searches = max_searches
        self.max_structure_docs = max_structure_docs
        self.max_content_searches = max_content_searches

    def choose(self, state: AgentState) -> AgentAction | None:
        plan = state.get("query_plan") or QueryPlan()
        if plan.intent == "structural":
            return self._choose_structural(state, plan)
        return self._choose_search_or_graph(state, plan)

    def _search_action(
        self,
        query: str,
        filters: SearchFilters,
        reason_code: str,
        *,
        queries: list[str] | None = None,
    ) -> AgentAction:
        arguments: dict[str, Any] = {
            "query": query,
            "filters": filters,
            "retrieval_mode": "hybrid",
            "top_k": 5,
        }
        if queries:
            arguments["queries"] = queries
        return AgentAction("search_reports", arguments, reason_code)

    def _choose_search_or_graph(
        self, state: AgentState, plan: QueryPlan
    ) -> AgentAction | None:
        attempts = list(state.get("search_attempts", []))
        searches = state.get("search_calls_used", 0)
        if searches >= self.max_searches:
            return None

        if (
            plan.needs_graph
            and plan.entities
            and not state.get("graph_evidence_manifest")
            and not any(item.get("kind") == "graph" for item in attempts)
        ):
            return AgentAction(
                "get_entities",
                {"entity_names": plan.entities},
                "explore_entities",
            )

        if not any(item.get("kind") == "broad" for item in attempts):
            filters = state.get("effective_filters") or SearchFilters()
            return self._search_action(
                state.get("question", ""),
                filters,
                "fact_lookup",
                queries=list(state.get("rewritten_queries", [])),
            )

        expected = _target_years(state, plan)
        covered = set(state.get("covered_years", []))
        attempted = _attempted_years(state)
        missing = [
            year
            for year in expected
            if year not in covered and year not in attempted
        ]
        if missing and searches < self.max_searches:
            year = missing[0]
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={year},
                year_from=None,
                year_to=None,
            )
            return self._search_action(
                _query_for_year(state.get("question", ""), year),
                filters,
                "fact_lookup",
            )
        return None

    def _choose_structural(
        self, state: AgentState, plan: QueryPlan
    ) -> AgentAction | None:
        attempts = list(state.get("search_attempts", []))
        snapshots = state.get("structure_snapshots", {})
        documents = state.get("available_documents", {})
        years = _target_years(state, plan)[-self.max_structure_docs :]

        for year in years:
            if year in snapshots:
                continue
            doc_ids = list(documents.get(year) or [])
            allowed = (
                state.get("effective_filters") or SearchFilters()
            ).allowed_doc_ids
            if allowed:
                narrowed = [doc_id for doc_id in doc_ids if doc_id in allowed]
                doc_ids = narrowed or sorted(allowed)
            if not doc_ids:
                doc_ids = sorted(allowed or [])
            if not doc_ids:
                continue
            if any(
                item.get("kind") == "structure"
                and item.get("year") == year
                for item in attempts
            ):
                continue
            return AgentAction(
                "get_structure_tree",
                {"doc_id": doc_ids[0], "max_depth": 4},
                "clarify_structure",
            )

        if not state.get("structure_comparison_complete"):
            # Сравнение выполняет assess_evidence, когда все доступные
            # структуры уже собраны.
            return None

        changes = [
            change
            for change in state.get("structural_changes", [])
            if isinstance(change, dict)
            and change.get("change_type") != "stable"
        ]
        stable_matches = [
            match
            for match in state.get("section_matches", [])
            if isinstance(match, dict)
            and match.get("change_type") == "stable"
        ]
        checked = {
            content_attempt_key(item)
            for item in attempts
            if item.get("kind") == "structure_content"
        }

        if not changes and not stable_matches:
            if not any(
                item.get("kind") == "structure_content"
                for item in attempts
            ):
                filters = state.get("effective_filters") or SearchFilters()
                if plan.section_filter:
                    filters = replace(
                        filters, section_path_prefix=plan.section_filter
                    )
                return self._search_action(
                    state.get("question", ""),
                    filters,
                    "inspect_evidence",
                )
            return None

        searches = state.get("search_calls_used", 0)
        for change in changes:
            if searches >= self.max_searches:
                break
            if change.get("change_type") == "removed":
                target_year = change.get("from_year")
                section = change.get("from_section")
            else:
                target_year = change.get("to_year") or change.get("from_year")
                section = change.get("to_section") or change.get("from_section")
            if not isinstance(target_year, int):
                continue
            if not isinstance(section, dict):
                continue
            section_path = section.get("section_path")
            title = section.get("title")
            keys = {((target_year,), section_path)}
            if change.get("detail") == "content_changed":
                years = {
                    year
                    for year in (
                        change.get("from_year"),
                        change.get("to_year"),
                    )
                    if isinstance(year, int)
                }
                if years:
                    keys.add((tuple(sorted(years)), section_path))
            if keys & checked:
                continue
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={target_year},
                year_from=None,
                year_to=None,
            )
            if isinstance(section_path, str) and section_path:
                filters = replace(filters, section_path_prefix=section_path)
            query = (
                f"{title} {state.get('question', '')}"
                if isinstance(title, str) and title
                else state.get("question", "")
            )
            return self._search_action(
                query,
                filters,
                "inspect_evidence",
            )

        # Даже у структурно стабильного раздела содержание могло измениться.
        for match in stable_matches:
            if searches >= self.max_searches:
                break
            from_year = match.get("from_year")
            to_year = match.get("to_year")
            section = match.get("to_section") or match.get("from_section")
            if not isinstance(from_year, int) or not isinstance(to_year, int):
                continue
            if not isinstance(section, dict):
                continue
            section_path = section.get("section_path")
            title = section.get("title")
            key = ((from_year, to_year), section_path)
            if key in checked:
                continue
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={from_year, to_year},
                year_from=None,
                year_to=None,
            )
            if isinstance(section_path, str) and section_path:
                filters = replace(filters, section_path_prefix=section_path)
            query = (
                f"{title} {state.get('question', '')}"
                if isinstance(title, str) and title
                else state.get("question", "")
            )
            return self._search_action(
                query,
                filters,
                "inspect_evidence",
            )
        return None
