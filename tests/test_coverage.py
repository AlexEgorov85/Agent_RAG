"""Тесты CoverageReport: год отсутствует vs метрика не найдена."""

from __future__ import annotations

from src.agent.coverage import (
    build_coverage,
    extract_metrics,
    normalize_metric,
)


def chunk(year: int, text: str) -> dict:
    return {"year": year, "text": text, "html": None}


def test_distinguishes_absent_year_from_missing_metric() -> None:
    evidence = {
        "c2022": chunk(2022, "Чистая прибыль составила 100."),
        "c2023": chunk(2023, "Выручка составила 200."),
    }

    report = build_coverage(
        [2021, 2022, 2023],
        ["net_profit"],
        evidence,
        {2022, 2023},
    )
    cells = {(cell.year, cell.status) for cell in report.cells}

    assert (2021, "year_absent") in cells
    assert (2022, "covered") in cells
    assert (2023, "not_found") in cells
    assert report.missing_years == [2021]
    assert report.covered_years == [2022]
    assert report.uncovered_years == [2021, 2023]
    assert report.status == "partial"
    assert report.is_year_absent(2021) is True


def test_missing_cells_and_is_covered_helpers() -> None:
    report = build_coverage(
        [2021, 2022, 2023],
        ["net_profit"],
        {
            "c2022": chunk(2022, "Чистая прибыль 100."),
            "c2023": chunk(2023, "Выручка 200."),
        },
        {2022, 2023},
    )

    assert report.is_covered("net_profit", 2022) is True
    assert report.is_covered("net_profit", 2023) is False
    assert {
        (cell.year, cell.status) for cell in report.missing_cells()
    } == {(2021, "year_absent"), (2023, "not_found")}


def test_complete_report_when_all_cells_covered() -> None:
    evidence = {
        "c2022": chunk(2022, "Чистая прибыль 100."),
        "c2023": chunk(2023, "Чистая прибыль 120."),
    }

    report = build_coverage(
        [2022, 2023], ["net_profit"], evidence, {2022, 2023}
    )

    assert report.status == "complete"
    assert report.covered_years == [2022, 2023]
    assert report.uncovered_years == []


def test_year_level_coverage_without_metrics() -> None:
    evidence = {"c2023": chunk(2023, "Любой текст.")}

    report = build_coverage([2023], [], evidence, {2023})

    assert report.status == "complete"
    assert report.covered_years == [2023]


def test_extract_and_normalize_metrics() -> None:
    assert extract_metrics("Как изменилась чистая прибыль и ROE?") == [
        "net_profit",
        "roe",
    ]
    assert normalize_metric("Чистая прибыль") == "net_profit"
    assert normalize_metric("неизвестный показатель") is None


def test_normalize_operating_profit_is_not_net_profit() -> None:
    assert normalize_metric("Операционная прибыль") == "operating_income"
    assert normalize_metric("операционную прибыль, млрд руб.") == (
        "operating_income"
    )
    assert "net_profit" not in extract_metrics("Операционная прибыль выросла")


def test_normalize_cet1_is_not_balance_capital() -> None:
    assert normalize_metric("Достаточность капитала CET1") == "cet1"
    assert normalize_metric("достаточность базового капитала") == "cet1"
    assert normalize_metric("CET1") == "cet1"
    assert normalize_metric("капитал") == "capital"


def test_normalize_roe_wins_over_nested_capital_alias() -> None:
    assert normalize_metric("Рентабельность капитала") == "roe"
    assert normalize_metric("ROE") == "roe"
    assert "capital" not in extract_metrics("Рентабельность капитала составила 25%")
    assert "assets" not in extract_metrics("Рентабельность активов выросла")


def test_operating_profit_chunk_does_not_cover_net_profit() -> None:
    report = build_coverage(
        [2023],
        ["net_profit", "operating_income"],
        {"c1": chunk(2023, "Операционная прибыль составила 300 млрд руб.")},
        {2023},
    )

    assert report.metrics["net_profit"]["2023"] == "not_found"
    assert report.metrics["operating_income"]["2023"] == "covered"
    assert report.missing_metrics == ["net_profit"]


def test_to_dict_is_serializable() -> None:
    report = build_coverage(
        [2022], ["net_profit"], {"c1": chunk(2022, "Прибыль 1")}, {2022}
    )

    payload = report.to_dict()

    assert payload["status"] == "complete"
    assert payload["cells"][0]["metric"] == "net_profit"
    assert payload["cells"][0]["status"] == "covered"
    assert payload["metrics"]["net_profit"]["2022"] == "covered"
    assert payload["metrics"]["net_profit"]["*"] == "covered"
    assert payload["missing_metrics"] == []
    assert payload["missing_metric_years"] == []
    assert payload["distinct_docs"] == 0
    assert payload["distinct_sections"] == 0
    assert payload["notes"] == []


def test_year_absent_with_other_years_covered_is_complete() -> None:
    report = build_coverage(
        [2021, 2022, 2023],
        ["revenue"],
        {
            "c2022": chunk(2022, "Выручка 200."),
            "c2023": chunk(2023, "Выручка 250."),
        },
        {2022, 2023},
    )

    assert report.status == "complete"
    assert report.covered_years == [2022, 2023]
    assert report.missing_years == [2021]
    assert report.is_year_absent(2021) is True
    assert any("нет в корпусе" in note for note in report.notes)


def test_empty_evidence_is_insufficient() -> None:
    report = build_coverage([2022], ["revenue"], {}, {2022})

    assert report.status == "insufficient"
    assert report.covered_years == []
    assert report.cells[0].status == "not_found"
    assert any("нет собранных свидетельств" in note for note in report.notes)


def test_only_not_found_cells_is_insufficient() -> None:
    report = build_coverage(
        [2022], ["revenue"], {"c1": chunk(2022, "Прибыль 1")}, {2022}
    )

    assert report.status == "insufficient"
    assert report.required_metrics == ["revenue"]
    assert report.missing_metrics == ["revenue"]


def test_metrics_matrix_and_missing_metric_years() -> None:
    report = build_coverage(
        [2022, 2023],
        ["revenue", "net_profit"],
        {
            "c2022": chunk(2022, "Выручка 200."),
            "c2023": chunk(2023, "Чистая прибыль 120."),
        },
        {2022, 2023},
    )

    assert report.metrics["revenue"] == {
        "2022": "covered",
        "2023": "not_found",
        "*": "not_found",
    }
    assert report.metrics["net_profit"] == {
        "2022": "not_found",
        "2023": "covered",
        "*": "not_found",
    }
    assert report.covered_years == []
    assert report.status == "partial"
    assert set(report.missing_metrics) == {"revenue", "net_profit"}
    assert {
        (item["metric"], item["year"], item["status"])
        for item in report.missing_metric_years
    } == {
        ("revenue", 2023, "not_found"),
        ("net_profit", 2022, "not_found"),
    }


def test_metric_year_is_covered_only_when_all_metrics_are_found() -> None:
    report = build_coverage(
        [2022],
        ["revenue", "net_profit"],
        {"c1": chunk(2022, "Выручка и чистая прибыль 200.")},
        {2022},
    )

    assert report.covered_years == [2022]
    assert report.status == "complete"
    assert report.metrics["revenue"]["*"] == "covered"
    assert report.metrics["net_profit"]["*"] == "covered"


def test_distinct_docs_and_sections() -> None:
    report = build_coverage(
        [2022],
        ["revenue"],
        {
            "c1": {
                "year": 2022,
                "text": "Выручка 1.",
                "doc_id": "doc-1",
                "section_path": "Финансы",
            },
            "c2": {
                "year": 2022,
                "text": "Выручка 2.",
                "doc_id": "doc-1",
                "section_path": "Финансы/Выручка",
            },
            "c3": {
                "year": 2022,
                "text": "Выручка 3.",
                "doc_id": "doc-2",
                "section_path": "Финансы",
            },
        },
        {2022},
    )

    assert report.distinct_docs == 2
    assert report.distinct_sections == 2


def test_simple_factoid_without_years_or_metrics_is_not_required() -> None:
    report = build_coverage([], [], {}, None)

    assert report.status == "not_required"
    assert report.metrics == {}
    assert report.missing_metrics == []
    assert report.notes == []
