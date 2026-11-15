"""Тесты последовательного выбора действий и сужения фильтров."""

from __future__ import annotations

from src.agent.agent_graph import _merge_action_filters
from src.agent.coverage import build_coverage
from src.agent.selector import PlanAwareSelector, _attempted_metric_years
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


def test_max_queries_caps_broad_rewrites() -> None:
    plan = QueryPlan(
        intent="temporal",
        years=[2022, 2023],
        answer_mode="comparison",
        needs_rewriting=True,
        aspects=["net_profit", "revenue"],
        required_metrics=["net_profit", "revenue"],
    )

    action = PlanAwareSelector(max_queries=1).choose(
        state(
            query_plan=plan,
            effective_filters=SearchFilters(years={2022, 2023}),
            question="Как менялась прибыль?",
        )
    )

    assert action is not None
    assert action.arguments["queries"] == ["Как менялась прибыль?"]


def test_coverage_can_be_disabled_without_gap_searches() -> None:
    plan = QueryPlan(
        intent="temporal",
        years=[2020, 2022],
        answer_mode="comparison",
        required_metrics=["net_profit"],
    )
    coverage = build_coverage(
        [2020, 2022],
        ["net_profit"],
        {},
        {2020, 2022},
    )
    base = state(
        query_plan=plan,
        search_attempts=[{"kind": "broad"}],
        search_calls_used=1,
        coverage=coverage,
    )

    enabled = PlanAwareSelector().choose(base)
    disabled = PlanAwareSelector(coverage_enabled=False).choose(base)

    assert enabled is not None
    assert disabled is None


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


def test_missing_metric_is_added_to_targeted_year_query() -> None:
    action = PlanAwareSelector().choose(
        state(
            query_plan=QueryPlan(
                intent="temporal",
                years=[2022, 2023],
                answer_mode="comparison",
                required_metrics=["net_profit"],
                requires_comparison=True,
            ),
            search_calls_used=1,
            covered_years=[2022],
            search_attempts=[{"kind": "broad"}],
            expected_years=[2022, 2023],
        )
    )

    assert action is not None
    assert action.arguments["filters"].years == {2023}
    assert "Чистая прибыль" in action.arguments["query"]


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


ALL_YEARS = [2020, 2021, 2022, 2023, 2024]
ALL_METRICS = ["revenue", "net_profit", "operating_income"]


def coverage_chunk(year: int, text: str) -> dict:
    return {"year": year, "text": text, "doc_id": f"doc{year}"}


def metric_year_state() -> dict:
    full_text = "Выручка. Чистая прибыль. Операционная прибыль."
    evidence = {
        f"c{year}": coverage_chunk(year, full_text)
        for year in (2020, 2022, 2024)
    }
    # 2023: net_profit и operating_income есть, revenue отсутствует.
    evidence["c2023"] = coverage_chunk(
        2023, "Чистая прибыль. Операционная прибыль."
    )
    report = build_coverage(
        ALL_YEARS,
        ALL_METRICS,
        evidence,
        {2020, 2022, 2023, 2024},
    )
    return state(
        query_plan=QueryPlan(
            intent="temporal",
            years=ALL_YEARS,
            answer_mode="comparison",
            required_metrics=ALL_METRICS,
            requires_comparison=True,
        ),
        effective_filters=SearchFilters(years=set(ALL_YEARS)),
        expected_years=ALL_YEARS,
        covered_years=report.covered_years,
        coverage=report,
        search_attempts=[{"kind": "broad"}],
        search_calls_used=1,
    )


def test_selector_targets_exact_missing_metric_year() -> None:
    action = PlanAwareSelector().choose(metric_year_state())

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2023}
    assert action.arguments["metric"] == "revenue"
    assert "Выручка" in action.arguments["query"]
    assert "2023" in action.arguments["query"]


def test_selector_does_not_repeat_attempted_metric_year() -> None:
    base = metric_year_state()
    base["search_attempts"] = [
        {"kind": "broad"},
        {
            "kind": "metric_year",
            "metric": "revenue",
            "year": 2023,
            "years": [2023],
            "section_path": None,
        },
    ]
    base["search_calls_used"] = 2

    assert PlanAwareSelector().choose(base) is None


def test_selector_never_searches_a_year_absent_from_corpus() -> None:
    evidence = {
        f"c{year}": coverage_chunk(
            year, "Выручка. Чистая прибыль. Операционная прибыль."
        )
        for year in (2020, 2022, 2023, 2024)
    }
    report = build_coverage(
        ALL_YEARS,
        ALL_METRICS,
        evidence,
        {2020, 2022, 2023, 2024},
    )
    base = state(
        query_plan=QueryPlan(
            intent="temporal",
            years=ALL_YEARS,
            required_metrics=ALL_METRICS,
            requires_comparison=True,
        ),
        effective_filters=SearchFilters(years=set(ALL_YEARS)),
        expected_years=ALL_YEARS,
        covered_years=report.covered_years,
        coverage=report,
        search_attempts=[{"kind": "broad"}],
        search_calls_used=1,
    )

    action = PlanAwareSelector().choose(base)

    assert action is None
    assert report.is_year_absent(2021) is True


def test_attempted_metric_years_helper() -> None:
    assert _attempted_metric_years(
        {
            "search_attempts": [
                {"kind": "metric_year", "metric": "revenue", "year": 2023},
                {"kind": "metric_year", "metric": "Чистая прибыль", "year": 2022},
                {"kind": "broad"},
            ]
        }
    ) == {("revenue", 2023), ("net_profit", 2022)}


def test_aspects_are_added_to_broad_queries() -> None:
    question = "Сравни выручку и прибыль за 2023–2024 годы"
    plan = QueryPlan(
        intent="temporal",
        years=[2023, 2024],
        answer_mode="comparison",
        aspects=[question, "Выручка", "активы", "капитал"],
        needs_rewriting=True,
        required_metrics=["revenue"],
        requires_comparison=True,
    )

    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            question=question,
            effective_filters=SearchFilters(years={2023, 2024}),
            expected_years=[2023, 2024],
        )
    )

    assert action is not None
    assert action.arguments["queries"] == [question, "Выручка", "активы"]


def test_aspects_are_deduplicated_casefold_and_capped() -> None:
    question = "Сравни прибыль и активы"
    plan = QueryPlan(
        intent="temporal",
        years=[2023, 2024],
        aspects=[question.upper(), "ПРИБЫЛЬ", "активы", "капитал"],
        requires_comparison=True,
        required_metrics=["net_profit", "assets"],
    )

    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            question=question,
            effective_filters=SearchFilters(years={2023, 2024}),
            expected_years=[2023, 2024],
        )
    )

    assert action is not None
    assert action.arguments["queries"] == [question, "ПРИБЫЛЬ", "активы"]


def test_simple_factoid_without_aspects_uses_question_only() -> None:
    question = "Какая выручка в 2023 году?"
    plan = QueryPlan(
        intent="factoid",
        years=[2023],
        required_metrics=["revenue"],
    )

    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            question=question,
            effective_filters=SearchFilters(years={2023}),
            expected_years=[2023],
        )
    )

    assert action is not None
    assert action.arguments["queries"] == [question]


def test_aspects_are_not_used_without_complexity_flags() -> None:
    question = "Какая выручка в 2023 году?"
    plan = QueryPlan(
        intent="factoid",
        years=[2023],
        aspects=["доходы", "продажи"],
        required_metrics=["revenue"],
    )

    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            question=question,
            effective_filters=SearchFilters(years={2023}),
            expected_years=[2023],
        )
    )

    assert action is not None
    assert action.arguments["queries"] == [question]


def _chunk(year: int, text: str, section: str | None = None, kind: str = "text") -> dict:
    return {
        "year": year,
        "doc_id": f"doc{year}",
        "text": text,
        "section_path": section,
        "kind": kind,
    }


def test_year_suffixed_aspect_searches_the_year_without_the_topic() -> None:
    plan = QueryPlan(
        intent="multi_hop",
        years=[2022, 2023],
        answer_mode="comparison",
        aspects=["net_profit_2022", "net_profit_2023"],
        requires_comparison=True,
    )
    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            effective_filters=SearchFilters(years={2022, 2023}),
            expected_years=[2022, 2023],
            covered_years=[2022, 2023],
            search_attempts=[{"kind": "broad"}],
            search_calls_used=1,
            question="Сопоставь показатели двух отчётов.",
            evidence={
                "c22": _chunk(2022, "Чистая прибыль составила 270,5 млрд руб."),
                "c23": _chunk(2023, "Ключевые финансовые результаты отчёта."),
            },
        )
    )

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2023}
    assert action.arguments["metric"] == "чистая прибыль"
    assert "чистая прибыль" in action.arguments["query"]


def test_table_of_contents_does_not_count_as_aspect_evidence() -> None:
    toc = (
        "Отчет по рискам 11 Сбер сегодня 104 Отчет по рискам 12 "
        "Культура целей 13 Ключевые финансовые результаты 20 "
        "Стратегия 2026 21 Приложения 158"
    )
    plan = QueryPlan(
        intent="multi_hop",
        years=[2023],
        aspects=["financial_results"],
        requires_comparison=True,
    )
    action = PlanAwareSelector().choose(
        state(
            query_plan=plan,
            effective_filters=SearchFilters(years={2023}),
            expected_years=[2023],
            covered_years=[2023],
            search_attempts=[{"kind": "broad"}],
            search_calls_used=1,
            evidence={
                "toc": _chunk(2023, toc, kind="table"),
                "body": _chunk(2023, "Чистая прибыль и рентабельность капитала."),
            },
        )
    )

    assert action is not None
    assert action.arguments["metric"] == (
        "ключевые финансовые результаты чистая прибыль "
        "рентабельность капитала достаточность капитала CET1"
    )
    assert action.arguments["filters"].years == {2023}


def test_structural_content_is_completed_by_section_path() -> None:
    structural_state = state(
        query_plan=QueryPlan(
            intent="structural",
            years=[2022, 2023],
            answer_mode="comparison",
        ),
        effective_filters=SearchFilters(years={2022, 2023}),
        expected_years=[2022, 2023],
        available_documents={2022: ["doc2022"], 2023: ["doc2023"]},
        structure_snapshots={2022: [{"title": "A"}], 2023: [{"title": "A"}]},
        structure_comparison_complete=True,
        search_attempts=[
            {"kind": "structure", "year": 2022, "doc_id": "doc2022"},
            {"kind": "structure", "year": 2023, "doc_id": "doc2023"},
            {
                "kind": "structure_content",
                "year": None,
                "years": [2022, 2023],
                "section_path": None,
            },
        ],
        search_calls_used=1,
        evidence={
            "risk23": _chunk(2023, "Управление рисками.", "Сбер в цифрах/Риски"),
            "risk22": _chunk(2022, "Управление рисками.", "Управление рисками"),
        },
    )

    action = PlanAwareSelector().choose(structural_state)

    assert action is not None
    assert action.tool_name == "search_reports"
    assert action.arguments["filters"].years == {2023}
    assert action.arguments["filters"].section_path_prefix == "Сбер в цифрах/Риски"
    assert action.reason_code == "inspect_evidence"
