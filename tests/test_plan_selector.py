"""Тесты последовательного выбора действий и сужения фильтров."""

from __future__ import annotations

from src.agent.agent_graph import _merge_action_filters
from src.agent.selector import PlanAwareSelector
from src.agent.state import QueryPlan
from src.retrieval.hybrid_retriever import SearchFilters


def state(**overrides):
    base = {
        "query_plan": QueryPlan(
            intent="temporal",
            years=[2020, 2022],
            answer_mode="comparison",
        ),
        "effective_filters": SearchFilters(years={2020, 2022}),
        "question": "Как менялась прибыль?",
        "search_attempts": [],
        "search_calls_used": 0,
        "expected_years": [2020, 2022],
        "covered_years": [],
        "available_documents": {},
        "structure_snapshots": {},
        "structure_comparison_complete": False,
        "structural_changes": [],
    }
    base.update(overrides)
    return base


def test_starts_with_a_broad_search_over_the_period() -> None:
    action = PlanAwareSelector().choose(state())

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2020, 2022}


def test_searches_only_a_missing_year_with_narrowed_filter() -> None:
    action = PlanAwareSelector().choose(
        state(
            search_calls_used=1,
            covered_years=[2022],
            search_attempts=[{"kind": "broad"}],
        )
    )

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2020}
    assert "2020" in action.arguments["query"]


def test_does_not_repeat_an_attempted_year() -> None:
    action = PlanAwareSelector().choose(
        state(
            search_calls_used=2,
            covered_years=[2022],
            search_attempts=[
                {"kind": "broad"},
                {"kind": "year", "year": 2020},
            ],
        )
    )

    assert action is None


def test_stops_when_all_expected_years_are_covered() -> None:
    action = PlanAwareSelector().choose(
        state(
            search_calls_used=1,
            covered_years=[2020, 2022],
            search_attempts=[{"kind": "broad"}],
        )
    )

    assert action is None


def test_structural_plan_loads_each_document_tree() -> None:
    selector = PlanAwareSelector()
    structural_state = state(
        query_plan=QueryPlan(
            intent="structural",
            years=[2022, 2023],
            answer_mode="comparison",
        ),
        effective_filters=SearchFilters(years={2022, 2023}),
        expected_years=[2022, 2023],
        available_documents={2022: ["doc22"], 2023: ["doc23"]},
    )

    first = selector.choose(structural_state)
    assert first is not None
    assert first.tool_name == "get_structure_tree"
    assert first.arguments["doc_id"] == "doc22"

    structural_state["structure_snapshots"] = {2022: [{"title": "A"}]}
    structural_state["search_attempts"] = [
        {"kind": "structure", "year": 2022, "doc_id": "doc22"}
    ]
    second = selector.choose(structural_state)
    assert second is not None
    assert second.tool_name == "get_structure_tree"
    assert second.arguments["doc_id"] == "doc23"


def test_structural_changes_trigger_content_search() -> None:
    structural_state = state(
        query_plan=QueryPlan(
            intent="structural",
            years=[2022, 2023],
            answer_mode="comparison",
        ),
        effective_filters=SearchFilters(years={2022, 2023}),
        expected_years=[2022, 2023],
        available_documents={2022: ["doc22"], 2023: ["doc23"]},
        structure_snapshots={
            2022: [{"title": "A"}],
            2023: [{"title": "A"}],
        },
        structure_comparison_complete=True,
        structural_changes=[
            {
                "change_type": "renamed",
                "from_year": 2022,
                "to_year": 2023,
                "from_section": {"title": "A", "section_path": "A"},
                "to_section": {"title": "B", "section_path": "B"},
            }
        ],
        search_attempts=[
            {"kind": "structure", "year": 2022},
            {"kind": "structure", "year": 2023},
        ],
    )

    action = PlanAwareSelector().choose(structural_state)

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2023}
    assert action.arguments["filters"].section_path_prefix == "B"


def test_structural_without_changes_falls_back_to_content_search() -> None:
    structural_state = state(
        query_plan=QueryPlan(
            intent="structural",
            years=[2022],
            section_filter="Риски",
        ),
        effective_filters=SearchFilters(years={2022}),
        expected_years=[2022],
        structure_comparison_complete=True,
    )

    action = PlanAwareSelector().choose(structural_state)

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].section_path_prefix == "Риски"


def test_action_filters_can_narrow_but_not_expand_user_filters() -> None:
    effective = SearchFilters(
        years={2022, 2023, 2024},
        section_path_prefix="Риски",
    )

    narrowed = _merge_action_filters(
        effective,
        SearchFilters(
            years={2023},
            section_path_prefix="Риски/Кредитный риск",
        ),
    )
    assert narrowed.years == {2023}
    assert narrowed.section_path_prefix == "Риски/Кредитный риск"

    expanded = _merge_action_filters(
        effective,
        SearchFilters(
            years={2019},
            section_path_prefix="Финансы",
        ),
    )
    assert expanded.years == {2022, 2023, 2024}
    assert expanded.section_path_prefix == "Риски"
