"""Coverage-матрица «метрика × год».

Отделяет три разных состояния:

* ``covered`` — в найденных chunk'ах есть упоминание метрики за год;
* ``not_found`` — год в корпусе есть, но метрика в evidence не найдена;
* ``year_absent`` — самого года нет в корпусе.

Модуль не вызывает LLM и не извлекает числа: он только классифицирует уже
собранные evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CoverageStatus = Literal["covered", "not_found", "year_absent"]
ReportStatus = Literal["not_required", "complete", "partial", "insufficient"]

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "net_profit": (
        "чистая прибыль",
        "чистую прибыль",
        "прибыль",
        "net profit",
        "net income",
    ),
    "revenue": ("выручка", "выручку", "доходы", "revenue"),
    "operating_income": (
        "операционная прибыль",
        "операционную прибыль",
        "operating income",
    ),
    "roe": ("roe", "рентабельность капитала"),
    "roa": ("roa", "рентабельность активов"),
    "assets": ("активы", "активов", "assets"),
    "capital": ("капитал", "капитала", "equity"),
    "margin": ("маржа", "маржу", "margin"),
}

METRIC_LABELS = {
    "net_profit": "Чистая прибыль",
    "revenue": "Выручка",
    "operating_income": "Операционная прибыль",
    "roe": "ROE",
    "roa": "ROA",
    "assets": "Активы",
    "capital": "Капитал",
    "margin": "Маржа",
}


def normalize_metric(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.casefold().split())
    if not text:
        return None
    if text in METRIC_ALIASES:
        return text
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias in text or text in alias for alias in aliases):
            return metric
    return None


def extract_metrics(text: str) -> list[str]:
    normalized = text.casefold()
    found: list[str] = []
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            found.append(metric)
    return found[:5]


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return f"{chunk.get('text', '')} {chunk.get('html', '')}"
    return f"{getattr(chunk, 'text', '')} {getattr(chunk, 'html', '')}"


def chunk_matches_metric(chunk: Any, metric: str) -> bool:
    normalized = _chunk_text(chunk).casefold()
    aliases = METRIC_ALIASES.get(metric, (metric,))
    return any(alias in normalized for alias in aliases)


@dataclass(frozen=True)
class CoverageCell:
    metric: str
    year: int | None
    status: CoverageStatus
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_label": metric_label(self.metric),
            "year": self.year,
            "status": self.status,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass(frozen=True)
class CoverageReport:
    status: ReportStatus
    expected_years: list[int]
    covered_years: list[int]
    missing_years: list[int]
    cells: list[CoverageCell] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)

    @property
    def uncovered_years(self) -> list[int]:
        return sorted(set(self.expected_years) - set(self.covered_years))

    def is_year_absent(self, year: int) -> bool:
        return any(
            cell.year == year and cell.status == "year_absent"
            for cell in self.cells
        )

    def missing_cells(self) -> list[CoverageCell]:
        """Ячейки, по которым ещё нужен адресный поиск."""
        return [
            cell for cell in self.cells if cell.status != "covered"
        ]

    def is_covered(self, metric: str, year: int) -> bool:
        return any(
            cell.metric == metric
            and cell.year == year
            and cell.status == "covered"
            for cell in self.cells
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_years": list(self.expected_years),
            "covered_years": list(self.covered_years),
            "missing_years": list(self.missing_years),
            "uncovered_years": self.uncovered_years,
            "required_metrics": list(self.required_metrics),
            "cells": [cell.to_dict() for cell in self.cells],
        }


def _year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_coverage(
    expected_years: list[int],
    required_metrics: list[str],
    evidence: dict[str, Any],
    available_years: set[int] | None = None,
) -> CoverageReport:
    years = sorted({int(year) for year in expected_years})
    metrics = [
        metric
        for metric in (normalize_metric(item) for item in required_metrics)
        if metric is not None
    ]
    available = {
        int(year) for year in (available_years or set())
    }
    if not available:
        # Без карты корпуса не можем отличить «года нет» от «нет evidence».
        available = set(years) | {
            year
            for chunk in evidence.values()
            if isinstance(chunk, dict)
            and (year := _year(chunk.get("year"))) is not None
        }

    by_year: dict[int, list[tuple[str, Any]]] = {}
    for chunk_id, chunk in evidence.items():
        if not isinstance(chunk, dict):
            continue
        year = _year(chunk.get("year"))
        if year is not None:
            by_year.setdefault(year, []).append((chunk_id, chunk))

    cells: list[CoverageCell] = []
    for year in years:
        if year not in available:
            if metrics:
                for metric in metrics:
                    cells.append(
                        CoverageCell(metric, year, "year_absent", [])
                    )
            else:
                cells.append(
                    CoverageCell("", year, "year_absent", [])
                )
            continue
        chunks = by_year.get(year, [])
        if metrics:
            for metric in metrics:
                chunk_ids = [
                    chunk_id
                    for chunk_id, chunk in chunks
                    if chunk_matches_metric(chunk, metric)
                ]
                cells.append(
                    CoverageCell(
                        metric,
                        year,
                        "covered" if chunk_ids else "not_found",
                        chunk_ids,
                    )
                )
        else:
            chunk_ids = [chunk_id for chunk_id, _ in chunks]
            cells.append(
                CoverageCell(
                    "",
                    year,
                    "covered" if chunk_ids else "not_found",
                    chunk_ids,
                )
            )

    if not years:
        status: ReportStatus = "not_required"
    elif cells and all(cell.status == "covered" for cell in cells):
        status = "complete"
    elif any(cell.status == "covered" for cell in cells):
        status = "partial"
    else:
        status = "insufficient"

    covered_years = sorted(
        {
            year
            for year in years
            if cells
            and all(
                cell.status == "covered"
                for cell in cells
                if cell.year == year
            )
            and any(cell.year == year for cell in cells)
        }
    )
    missing_years = sorted(set(years) - available)
    return CoverageReport(
        status=status,
        expected_years=years,
        covered_years=covered_years,
        missing_years=missing_years,
        cells=cells,
        required_metrics=metrics,
    )
