"""Тесты детерминированных расчётов по фактам."""

from __future__ import annotations

import pytest

from src.engine.calculations import (
    format_calculations,
    parse_number,
    summarize_facts,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 508,6 млрд руб.", 1508.6),
        ("1,508.6", 1508.6),
        ("1.508,6", 1508.6),
        ("12,55%", 12.55),
        ("(1 508,6)", -1508.6),
        ("−12,5", -12.5),
        (1508, 1508.0),
        ("нет данных", None),
    ],
)
def test_parse_number(raw, expected) -> None:
    assert parse_number(raw) == expected


def test_summarize_facts_computes_change_and_cagr() -> None:
    facts = [
        {
            "metric": "Чистая прибыль",
            "year": 2020,
            "value": "100",
            "unit": "млрд руб.",
            "chunk_id": "c2020",
        },
        {
            "metric": "Чистая прибыль",
            "year": 2024,
            "value": "146,41",
            "unit": "млрд руб.",
            "chunk_id": "c2024",
        },
    ]

    [summary] = summarize_facts(facts)

    assert summary["from_value"] == 100
    assert summary["to_value"] == 146.41
    assert summary["absolute_change"] == pytest.approx(46.41)
    assert summary["percent_change"] == pytest.approx(46.41)
    assert summary["cagr"] == pytest.approx(10.0, abs=0.01)
    assert summary["direction"] == "increase"
    assert summary["chunk_ids"] == ["c2020", "c2024"]


def test_summarize_facts_skips_single_year_and_missing_values() -> None:
    assert summarize_facts(
        [{"metric": "Прибыль", "year": 2024, "value": 10}]
    ) == []
    assert summarize_facts(
        [{"metric": "Прибыль", "year": 2024, "value": "н/д"}]
    ) == []


def test_format_calculations_labels_result_as_calculation() -> None:
    facts = [
        {"metric": "Прибыль", "year": 2020, "value": 100},
        {"metric": "Прибыль", "year": 2021, "value": 120},
    ]
    text = format_calculations(summarize_facts(facts))

    assert text.startswith("Расчёт по подтверждённым фактам")
    assert "2020→2021" in text
    assert "+20.00" in text
