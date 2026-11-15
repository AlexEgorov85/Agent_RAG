"""Тесты verification: citations, числа, единицы, периоды и причинность."""

from __future__ import annotations

import json

import pytest

from src.agent.coverage import build_coverage
from src.agent.state import Answer, Citation, QueryPlan
from src.agent.verification import verify_answer


def evidence() -> dict[str, dict]:
    return {
        "c1": {
            "chunk_id": "c1",
            "year": 2022,
            "kind": "text",
            "text": "Чистая прибыль составила 100 млрд руб.",
        },
        "c2": {
            "chunk_id": "c2",
            "year": 2023,
            "kind": "text",
            "text": "Чистая прибыль составила 120 млрд руб.",
        },
    }


def codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def profit_facts() -> list[dict]:
    return [
        {
            "metric": "Чистая прибыль",
            "year": 2022,
            "value": "100",
            "unit": "млрд руб.",
            "chunk_id": "c1",
        },
        {
            "metric": "Чистая прибыль",
            "year": 2023,
            "value": "120",
            "unit": "млрд руб.",
            "chunk_id": "c2",
        },
    ]


def test_valid_answer_is_ok() -> None:
    answer = Answer(
        "answer",
        "Чистая прибыль выросла с 100 до 120 млрд руб.",
        citations=[Citation("c1"), Citation("c2")],
        facts=profit_facts(),
    )

    result = verify_answer(answer, evidence(), QueryPlan())

    assert result.status == "ok"
    assert result.issues == []
    assert result.ok is True
    assert result.supported_citations == ["c1", "c2"]


def test_value_not_found_in_source_fails() -> None:
    answer = Answer(
        "answer",
        "Прибыль 999 млрд руб.",
        citations=[Citation("c1")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2022,
                "value": "999",
                "unit": "млрд руб.",
                "chunk_id": "c1",
            }
        ],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "failed"
    assert "fact_value_not_in_source" in codes(result)
    issue = next(
        issue
        for issue in result.issues
        if issue.code == "fact_value_not_in_source"
    )
    assert issue.severity == "fatal"
    assert issue.fact_index == 0
    assert result.unsupported_facts == [0]


def test_fact_without_citation_fails() -> None:
    answer = Answer(
        "answer",
        "Прибыль 100 млрд руб.",
        citations=[Citation("c1")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2022,
                "value": "100",
                "chunk_id": "c2",
            }
        ],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "failed"
    assert "fact_without_citation" in codes(result)
    assert result.unsupported_facts == [0]


def test_fact_year_mismatch_uses_fact_year_not_chunk_year() -> None:
    answer = Answer(
        "answer",
        "В 2023 году прибыль составила 100 млрд руб.",
        citations=[Citation("c1")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2023,
                "value": "100",
                "chunk_id": "c1",
            }
        ],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "failed"
    assert "fact_year_mismatch" in codes(result)
    # Значение найдено в чанке, но год — нет: подменять год чанка нельзя.
    assert "fact_value_not_in_source" not in codes(result)


def test_fact_year_from_multiyear_chunk_is_accepted() -> None:
    source = {
        "c3": {
            "chunk_id": "c3",
            "year": 2023,
            "text": "2022: 100 млрд руб.; 2023: 120 млрд руб.",
        }
    }
    answer = Answer(
        "answer",
        "За 2022 год прибыль 100 млрд руб.",
        citations=[Citation("c3")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2022,
                "value": "100",
                "chunk_id": "c3",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert "fact_year_mismatch" not in codes(result)


def test_citation_outside_manifest_is_fatal() -> None:
    answer = Answer(
        "answer",
        "Прибыль 100 млрд руб.",
        citations=[Citation("missing")],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "failed"
    assert "citation_outside_manifest" in codes(result)


def test_quote_not_found_is_warning_and_does_not_crash() -> None:
    answer = Answer(
        "answer",
        "Прибыль 100 млрд руб.",
        citations=[
            Citation("c1", quote="Такого фрагмента в источнике нет."),
        ],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "warning"
    assert "quote_not_found" in codes(result)
    assert result.supported_citations == []


def test_quote_substring_with_normalized_spaces_is_accepted() -> None:
    answer = Answer(
        "answer",
        "Прибыль 100 млрд руб.",
        citations=[
            Citation("c1", quote="Чистая прибыль   составила 100 млрд руб."),
        ],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert "quote_not_found" not in codes(result)
    assert result.supported_citations == ["c1"]


def test_table_without_anchors_is_warning() -> None:
    source = {
        "t1": {
            "chunk_id": "t1",
            "year": 2022,
            "kind": "table",
            "html": "<table><tr><td>Выручка</td><td>100</td></tr></table>",
            "text": "Выручка | 100",
        }
    }
    answer = Answer(
        "answer",
        "Выручка 100 млрд руб.",
        citations=[Citation("t1")],
        facts=[
            {
                "kind": "table",
                "metric": "Выручка",
                "year": 2022,
                "value": "100",
                "chunk_id": "t1",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert result.status == "warning"
    assert "table_anchor_missing" in codes(result)


def test_table_anchor_must_exist_in_source() -> None:
    source = {
        "t1": {
            "chunk_id": "t1",
            "year": 2022,
            "kind": "table",
            "text": "Выручка | 100",
        }
    }
    answer = Answer(
        "answer",
        "Выручка 100 млрд руб.",
        citations=[Citation("t1")],
        facts=[
            {
                "kind": "table",
                "metric": "Выручка",
                "year": 2022,
                "value": "100",
                "row_label": "Выручка",
                "column_label": "2021",
                "chunk_id": "t1",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert "table_anchor_missing" in codes(result)


def test_table_with_valid_anchors_passes() -> None:
    source = {
        "t1": {
            "chunk_id": "t1",
            "year": 2022,
            "kind": "table",
            "text": "Выручка | 2022 | 100",
        }
    }
    answer = Answer(
        "answer",
        "Выручка 100 млрд руб.",
        citations=[Citation("t1")],
        facts=[
            {
                "kind": "table",
                "metric": "Выручка",
                "year": 2022,
                "value": "100",
                "row_label": "Выручка",
                "column_label": "2022",
                "chunk_id": "t1",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert result.status == "ok"


def test_causal_claim_unverified_is_warning() -> None:
    answer = Answer(
        "answer",
        "Прибыль выросла, потому что выросли доходы.",
        citations=[Citation("c2")],
        facts=profit_facts()[1:],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "warning"
    assert "causal_claim_unverified" in codes(result)

    supported = verify_answer(
        answer,
        {
            "c2": {
                "year": 2023,
                "text": (
                    "Чистая прибыль составила 120 млрд руб., "
                    "потому что выросли доходы."
                ),
            }
        },
    )
    assert "causal_claim_unverified" not in codes(supported)


def test_missing_gap_must_be_disclosed() -> None:
    report = build_coverage(
        [2022, 2023],
        ["net_profit"],
        {"c1": evidence()["c1"]},
        {2022, 2023},
    )
    answer = Answer(
        "answer",
        "В 2022 году прибыль составила 100 млрд руб.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence(), coverage=report)

    assert "gap_not_disclosed" in codes(result)

    disclosed = verify_answer(
        Answer(
            "answer",
            "Данных за 2023 год нет; в 2022 году прибыль 100 млрд руб.",
            citations=[Citation("c1")],
            facts=profit_facts()[:1],
        ),
        evidence(),
        coverage=report,
    )
    assert "gap_not_disclosed" not in codes(disclosed)


def test_gap_not_disclosed_accepts_mapping_coverage() -> None:
    answer = Answer(
        "answer",
        "Прибыль 100 млрд руб.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )
    coverage = {
        "status": "partial",
        "missing_years": [2023],
        "uncovered_years": [2023],
        "cells": [
            {
                "metric": "net_profit",
                "year": 2023,
                "status": "not_found",
                "chunk_ids": [],
            }
        ],
    }

    result = verify_answer(answer, evidence(), coverage=coverage)

    assert "gap_not_disclosed" in codes(result)


def test_numeric_claim_without_source_is_warning() -> None:
    answer = Answer(
        "answer",
        "Прибыль выросла на 25%.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "warning"
    assert "numeric_claim_unverified" in codes(result)


def test_year_like_quantity_with_countable_noun_is_checked() -> None:
    answer = Answer(
        "answer",
        "Сеть насчитывает 2 000 отделений и 1500 сотрудников.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "warning"
    numeric = [
        issue
        for issue in result.issues
        if issue.code == "numeric_claim_unverified"
    ]
    assert numeric
    literals = " ".join(issue.message for issue in numeric)
    assert "2 000" in literals
    assert "1500" in literals


def test_counted_quantity_is_not_hidden_as_a_year() -> None:
    answer = Answer(
        "answer",
        "Число клиентов — 2000, количество офисов: 1900.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert "numeric_claim_unverified" in codes(result)


def test_year_in_proper_context_is_not_a_numeric_claim() -> None:
    answer = Answer(
        "answer",
        "В 2022 году чистая прибыль составила 100 млрд руб.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(answer, evidence())

    assert "numeric_claim_unverified" not in codes(result)
    assert result.status == "ok"


def test_large_value_with_scale_is_verified_not_treated_as_year() -> None:
    source = {
        "c9": {
            "chunk_id": "c9",
            "year": 2022,
            "text": "Выручка составила 1 508 млрд руб.",
        }
    }
    answer = Answer(
        "answer",
        "Выручка составила 1 508 млрд руб.",
        citations=[Citation("c9")],
        facts=[
            {
                "metric": "Выручка",
                "year": 2022,
                "value": "1 508",
                "unit": "млрд руб.",
                "chunk_id": "c9",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert "numeric_claim_unverified" not in codes(result)
    assert "fact_value_not_in_source" not in codes(result)


def test_calculation_block_is_not_treated_as_claim() -> None:
    answer = Answer(
        "answer",
        "Прибыль выросла.\n\n"
        "Расчёт по подтверждённым фактам:\n"
        "- «Чистая прибыль», 2022→2023: +20.00 млрд руб. (20.0%)",
        citations=[Citation("c1"), Citation("c2")],
        facts=profit_facts(),
    )

    result = verify_answer(answer, evidence())

    assert "numeric_claim_unverified" not in codes(result)
    assert result.status == "ok"


@pytest.mark.parametrize(
    ("value", "unit", "chunk_text"),
    [
        ("1 508,6", "млрд руб.", "Выручка 1 508,6 млрд руб."),
        ("(1 234,5)", "млрд руб.", "Убыток (1 234,5) млрд руб."),
        ("−12,5", "%", "Снижение −12,5%"),
        ("12,5", "%", "Рентабельность 12,5%"),
        ("100", "млн руб.", "Выручка 0,1 млрд руб."),
    ],
)
def test_number_normalization_and_scale(
    value: str, unit: str, chunk_text: str
) -> None:
    source = {"n1": {"chunk_id": "n1", "year": 2022, "text": chunk_text}}
    answer = Answer(
        "answer",
        "Показатель приведён.",
        citations=[Citation("n1")],
        facts=[
            {
                "metric": "Показатель",
                "year": 2022,
                "value": value,
                "unit": unit,
                "chunk_id": "n1",
            }
        ],
    )

    result = verify_answer(answer, source)

    assert "fact_value_not_in_source" not in codes(result)


def test_different_units_are_not_the_same_value() -> None:
    answer = Answer(
        "answer",
        "Прибыль 120 млн руб.",
        citations=[Citation("c2")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2023,
                "value": "120",
                "unit": "млн руб.",
                "chunk_id": "c2",
            }
        ],
    )

    result = verify_answer(answer, evidence())

    assert result.status == "failed"
    assert "fact_value_not_in_source" in codes(result)


def test_unit_mismatch_is_warning() -> None:
    source = {
        "u1": {"chunk_id": "u1", "text": "Выручка 100 млрд руб. (100 000 млн руб.)"},
        "u2": {"chunk_id": "u2", "text": "Выручка 120 млрд руб. (120 000 млн руб.)"},
    }
    answer = Answer(
        "answer",
        "Выручка 100 и 120 000.",
        citations=[Citation("u1"), Citation("u2")],
        facts=[
            {
                "metric": "Выручка",
                "value": "100",
                "unit": "млрд руб.",
                "chunk_id": "u1",
            },
            {
                "metric": "Выручка",
                "value": "120 000",
                "unit": "млн руб.",
                "chunk_id": "u2",
            },
        ],
    )

    result = verify_answer(answer, source)

    assert result.status == "warning"
    assert "unit_mismatch" in codes(result)


def test_partial_coverage_warns_for_comparison() -> None:
    report = build_coverage(
        [2022, 2023],
        ["net_profit"],
        {"c1": evidence()["c1"]},
        {2022, 2023},
    )
    answer = Answer(
        "answer",
        "Прибыль в 2022 году составила 100 млрд руб.",
        citations=[Citation("c1")],
        facts=profit_facts()[:1],
    )

    result = verify_answer(
        answer,
        evidence(),
        QueryPlan(
            intent="temporal",
            years=[2022, 2023],
            answer_mode="comparison",
            requires_comparison=True,
        ),
        report,
    )

    assert "period_incomplete" in codes(result)


def test_structure_unverified_warning() -> None:
    result = verify_answer(
        Answer("answer", "Раздел изменился."),
        {},
        QueryPlan(intent="structural"),
        structure_complete=False,
    )

    assert result.status == "warning"
    assert "structure_unverified" in codes(result)


def test_abstain_is_always_ok() -> None:
    result = verify_answer(
        Answer("abstain", "Нет данных.", citations=[]),
        {},
    )

    assert result.status == "ok"
    assert result.issues == []


def test_repair_diagnostics_are_serializable() -> None:
    answer = Answer(
        "answer",
        "Прибыль 999 млрд руб.",
        citations=[Citation("c1")],
        facts=[
            {
                "metric": "Прибыль",
                "year": 2023,
                "value": "999",
                "chunk_id": "c1",
            }
        ],
    )

    payload = verify_answer(answer, evidence()).to_dict()

    assert payload["status"] == "failed"
    assert payload["ok"] is False
    assert payload["issues"]
    issue = payload["issues"][0]
    assert {
        "code",
        "severity",
        "message",
        "chunk_ids",
        "chunk_id",
        "fact_index",
    } <= set(issue)
    assert json.loads(json.dumps(payload)) == payload
