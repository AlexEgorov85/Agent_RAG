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


def test_to_dict_is_serializable() -> None:
    report = build_coverage(
        [2022], ["net_profit"], {"c1": chunk(2022, "Прибыль 1")}, {2022}
    )

    payload = report.to_dict()

    assert payload["status"] == "complete"
    assert payload["cells"][0]["metric"] == "net_profit"
    assert payload["cells"][0]["status"] == "covered"
