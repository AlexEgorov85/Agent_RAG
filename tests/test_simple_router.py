"""Тесты универсального роутера периода (без хардкода годов корпуса)."""

from __future__ import annotations

import pytest

from src.agent.simple_router import SimpleRouter

YEARS = {2016, 2017, 2019, 2020, 2022, 2023, 2024}


def _plan(question: str, years: set[int] = YEARS):
    return SimpleRouter(years).analyze(question, None)


def test_last_n_years_are_anchored_to_corpus_maximum() -> None:
    result = _plan(
        "Как изменилась чистая прибыль за последние 5 лет?"
    )

    assert result.intent == "temporal"
    assert result.years == [2020, 2021, 2022, 2023, 2024]
    assert result.answer_mode == "comparison"


def test_last_n_years_work_for_an_arbitrary_corpus() -> None:
    result = _plan(
        "Как менялась выручка за последние 3 года?",
        years={2001, 2002, 2003, 2005},
    )

    assert result.years == [2003, 2004, 2005]


def test_word_number_is_supported() -> None:
    result = _plan("Динамика прибыли за последние пять лет")

    assert result.years == [2020, 2021, 2022, 2023, 2024]


def test_fused_period_word_is_supported() -> None:
    result = _plan("Покажи десятилетнюю динамику прибыли")

    assert result.years == sorted(YEARS)
    assert result.answer_mode == "comparison"


def test_explicit_range_is_expanded() -> None:
    result = _plan("Сравни прибыль с 2019 по 2023 год")

    assert result.years == [2019, 2020, 2021, 2022, 2023]
    assert result.answer_mode == "comparison"


def test_range_and_discrete_year_can_be_combined() -> None:
    result = _plan("Сравни прибыль 2019-2020 и 2023")

    assert result.years == [2019, 2020, 2023]


def test_discrete_years_are_kept() -> None:
    result = _plan("Сравни прибыль 2019 и 2023")

    assert result.years == [2019, 2023]
    assert result.answer_mode == "comparison"


def test_missing_years_are_not_removed_from_the_plan() -> None:
    result = _plan("Сравни прибыль с 2019 по 2023", years={2019, 2020, 2022})

    assert result.years == [2019, 2020, 2021, 2022, 2023]


def test_all_years_phrase_uses_the_whole_corpus() -> None:
    result = _plan("Как менялись подходы к безопасности за все время?")

    assert result.years == sorted(YEARS)
    assert result.answer_mode == "comparison"


def test_evolution_question_without_period_uses_available_years() -> None:
    result = _plan("Как менялись подходы к кибербезопасности?")

    assert result.years == sorted(YEARS)
    assert result.answer_mode == "comparison"


def test_single_year_question_stays_concise() -> None:
    result = _plan("Какая чистая прибыль за 2023 год?")

    assert result.years == [2023]
    assert result.answer_mode == "concise"


def test_recent_years_phrase_uses_the_whole_corpus() -> None:
    result = _plan("Как менялась прибыль в последние годы?")

    assert result.years == sorted(YEARS)


def test_latest_year_phrase_uses_corpus_maximum() -> None:
    result = _plan("Какая прибыль за последний год?")

    assert result.years == [2024]


def test_previous_year_is_relative_to_corpus() -> None:
    result = _plan("Какая прибыль за прошлый год?")

    assert result.years == [2023]


def test_year_over_year_adds_previous_year() -> None:
    result = _plan("Прибыль 2024 к прошлому году")

    assert result.years == [2023, 2024]
    assert result.answer_mode == "comparison"


def test_open_after_range() -> None:
    result = _plan("Что изменилось после 2022?")

    assert result.years == [2022, 2023, 2024]


def test_open_before_range() -> None:
    result = _plan("Динамика прибыли до 2020 года")

    assert result.years == [2016, 2017, 2018, 2019, 2020]


def test_since_range() -> None:
    result = _plan("Как менялась прибыль с 2020 года?")

    assert result.years == [2020, 2021, 2022, 2023, 2024]


def test_financial_structure_is_not_document_structure() -> None:
    result = _plan("Какая структура капитала за 2023 год?")

    assert result.intent == "temporal"
    assert result.years == [2023]


def test_document_structure_question_is_structural() -> None:
    result = _plan("Как менялась структура годовых отчётов в 2022-2023 годах?")

    assert result.intent == "structural"
    assert result.years == [2022, 2023]


def test_factoid_without_period_is_not_temporal() -> None:
    result = _plan("Какая ставка налога на прибыль?")

    assert result.intent == "factoid"
    assert result.years == []


def test_amount_is_not_treated_as_year() -> None:
    result = _plan("Выручка 1508,6 млрд рублей")

    assert result.intent == "factoid"
    assert result.years == []


def test_last_n_longer_than_corpus_uses_all_available_years() -> None:
    result = _plan("Как менялась прибыль за последние 30 лет?")

    assert result.years == sorted(YEARS)
    assert result.answer_mode == "comparison"


def test_max_period_years_is_enforced() -> None:
    result = SimpleRouter(YEARS, max_period_years=3).analyze(
        "Как менялась прибыль за все время?", None
    )

    assert result.years == [2022, 2023, 2024]


def test_router_requires_positive_period_limit() -> None:
    with pytest.raises(ValueError):
        SimpleRouter(YEARS, max_period_years=0)
