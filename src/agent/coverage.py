"""Coverage-матрица «метрика × год».

Отделяет три разных состояния:

* ``covered`` — в найденных chunk'ах есть упоминание метрики за год;
* ``not_found`` — год в корпусе есть, но метрика в evidence не найдена;
* ``year_absent`` — самого года нет в корпусе.

Модуль не вызывает LLM и не извлекает числа: он только классифицирует уже
собранные evidence. ``year_absent`` — ограничение корпуса, а не ошибка поиска.
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
    #: Достаточность капитала (CET1) — отдельный показатель, а не «капитал»:
    #: иначе нормализация подменяла его балансовым капиталом.
    "cet1": (
        "достаточность капитала cet1",
        "достаточность базового капитала",
        "достаточность основного капитала",
        "базового капитала",
        "cet1",
    ),
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
    "cet1": "Достаточность капитала CET1",
    "margin": "Маржа",
}


def _alias_spans(text: str) -> list[tuple[int, int, str]]:
    """Все вхождения алиасов метрик в уже нормализованный текст.

    Возвращает ``(start, end, metric)``; вложенные алиасы («прибыль» внутри
    «операционная прибыль») здесь не отбрасываются — это делает вызывающий
    код, который знает, какой алиас специфичнее.
    """
    spans: list[tuple[int, int, str]] = []
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            start = text.find(alias)
            while start >= 0:
                spans.append((start, start + len(alias), metric))
                start = text.find(alias, start + 1)
    return spans


def _exact_alias_metric(text: str) -> str | None:
    """Метрика, алиас которой совпадает со всем текстом."""
    for metric, aliases in METRIC_ALIASES.items():
        if text in aliases:
            return metric
    return None


def normalize_metric(value: Any) -> str | None:
    """Нормализует название показателя в канонический ключ.

    Порядок выбора: точное совпадение ключа, точное совпадение алиаса, затем
    самый длинный (специфичный) алиас внутри текста. Длина важнее порядка
    словаря: «операционная прибыль» не должна становиться ``net_profit``
    только потому, что «прибыль» — тоже алиас, а «достаточность капитала
    CET1» — ``capital``.
    """
    if not isinstance(value, str):
        return None
    text = " ".join(value.casefold().split())
    if not text:
        return None
    if text in METRIC_ALIASES:
        return text
    exact = _exact_alias_metric(text)
    if exact is not None:
        return exact
    best: tuple[int, int, str] | None = None
    for start, end, metric in _alias_spans(text):
        key = (end - start, -start)
        if best is None or key > best[:2]:
            best = (key[0], key[1], metric)
    if best is not None:
        return best[2]
    # Совместимость: короткая форма («операционная») — это начало алиаса.
    short: tuple[int, str] | None = None
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if text in alias:
                if short is None or len(alias) < short[0]:
                    short = (len(alias), metric)
    return short[1] if short is not None else None


def _matched_metrics(text: str) -> list[str]:
    """Метрики, упомянутые в тексте, без подмены вложенными алиасами.

    Длинный алиас поглощает вложенный: в «операционная прибыль» нет отдельной
    «чистой прибыли», а в «рентабельность капитала» — отдельного «капитала».
    """
    normalized = text.casefold()
    spans = sorted(
        _alias_spans(normalized),
        key=lambda item: (-(item[1] - item[0]), item[0]),
    )
    kept: list[tuple[int, int, str]] = []
    for start, end, metric in spans:
        if any(
            start < other_end and other_start < end
            for other_start, other_end, _ in kept
        ):
            continue
        kept.append((start, end, metric))
    kept.sort(key=lambda item: item[0])
    return list(dict.fromkeys(metric for _, _, metric in kept))


def extract_metrics(text: str) -> list[str]:
    """Метрики, упомянутые в тексте (не больше пяти, в порядке упоминания)."""
    return _matched_metrics(text)[:5]


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return f"{chunk.get('text', '')} {chunk.get('html', '')}"
    return f"{getattr(chunk, 'text', '')} {getattr(chunk, 'html', '')}"


def _chunk_field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name)
    return getattr(chunk, name, None)


def chunk_matches_metric(chunk: Any, metric: str) -> bool:
    """Чанк подтверждает именно эту метрику, а не родственный показатель.

    Используется та же логика «длинный алиас поглощает вложенный», поэтому
    чанк про операционную прибыль не закрывает покрытие чистой прибыли.
    """
    matched = _matched_metrics(_chunk_text(chunk))
    if metric in matched:
        return True
    if metric in METRIC_ALIASES:
        return False
    # Неизвестная метрика: сохраняем прежнее прямое сравнение по ключу.
    return metric.casefold() in _chunk_text(chunk).casefold()


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


def _aggregate_status(statuses: list[str]) -> str:
    """Сводный статус метрики по всем требуемым годам."""
    values = set(statuses)
    if not values:
        return "not_found"
    if values == {"covered"}:
        return "covered"
    if values == {"year_absent"}:
        return "year_absent"
    return "not_found"


@dataclass(frozen=True)
class CoverageReport:
    status: ReportStatus
    expected_years: list[int]
    covered_years: list[int]
    missing_years: list[int]
    cells: list[CoverageCell] = field(default_factory=list)
    required_metrics: list[str] = field(default_factory=list)
    #: metric -> year|"*" -> status; "*" — сводка по всем годам метрики.
    metrics: dict[str, dict[str, str]] = field(default_factory=dict)
    missing_metrics: list[str] = field(default_factory=list)
    #: [{metric, metric_label, year, status}] для незакрытых metric-ячеек.
    missing_metric_years: list[dict[str, Any]] = field(default_factory=list)
    distinct_docs: int = 0
    distinct_sections: int = 0
    notes: list[str] = field(default_factory=list)

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

    def has_actionable_gaps(self) -> bool:
        """Есть ли пробел, который можно закрыть ещё одним поиском.

        ``year_absent`` — ограничение корпуса, его повторно не ищем.
        """
        for cell in self.missing_cells():
            if cell.status == "year_absent":
                continue
            if cell.metric:
                if cell.year is not None:
                    return True
                continue
            if cell.year is not None:
                return True
        for year in self.uncovered_years:
            if not self.is_year_absent(year):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_years": list(self.expected_years),
            "covered_years": list(self.covered_years),
            "missing_years": list(self.missing_years),
            "uncovered_years": self.uncovered_years,
            "required_metrics": list(self.required_metrics),
            "cells": [cell.to_dict() for cell in self.cells],
            "metrics": {
                metric: dict(statuses)
                for metric, statuses in self.metrics.items()
            },
            "missing_metrics": list(self.missing_metrics),
            "missing_metric_years": [
                dict(item) for item in self.missing_metric_years
            ],
            "distinct_docs": self.distinct_docs,
            "distinct_sections": self.distinct_sections,
            "notes": list(self.notes),
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


def _evidence_years(evidence: dict[str, Any]) -> list[int]:
    years: set[int] = set()
    for chunk in evidence.values():
        year = _year(_chunk_field(chunk, "year"))
        if year is not None:
            years.add(year)
    return sorted(years)


def _distinct_evidence_values(
    evidence: dict[str, Any], name: str
) -> int:
    values: set[str] = set()
    for chunk in evidence.values():
        value = _chunk_field(chunk, name)
        if isinstance(value, str) and value.strip():
            values.add(value.strip())
    return len(values)


def _notes(
    status: ReportStatus,
    evidence: dict[str, Any],
    missing_years: list[int],
    not_found_cells: list[CoverageCell],
) -> list[str]:
    if status == "not_required":
        return []
    notes: list[str] = []
    if not evidence:
        notes.append("нет собранных свидетельств")
    for year in missing_years:
        notes.append(f"{year}: года нет в корпусе")
    for cell in not_found_cells:
        label = metric_label(cell.metric)
        if cell.year is not None:
            notes.append(f"{label} за {cell.year} год: показатель не найден")
        else:
            notes.append(f"{label}: показатель не найден")
    if status == "insufficient" and evidence and not not_found_cells:
        notes.append("ни одна требуемая ячейка не покрыта")
    return notes


def build_coverage(
    expected_years: list[int],
    required_metrics: list[str],
    evidence: dict[str, Any],
    available_years: set[int] | None = None,
) -> CoverageReport:
    """Строит матрицу «метрика × год» по уже собранным evidence.

    Если годы не заданы, но метрики требуются, матрица строится по годам,
    реально присутствующим в evidence (spec P3, шаг 3). Простой factoid без
    лет и метрик получает статус ``not_required``.
    """
    years = sorted({int(year) for year in expected_years})
    metrics = [
        metric
        for metric in (normalize_metric(item) for item in required_metrics)
        if metric is not None
    ]
    # Убираем дубли, сохраняя порядок router'а.
    metrics = list(dict.fromkeys(metrics))
    if not years and metrics:
        years = _evidence_years(evidence)

    available = {
        int(year) for year in (available_years or set())
    }
    if not available:
        # Без карты корпуса не можем отличить «года нет» от «нет evidence».
        available = set(years) | set(_evidence_years(evidence))

    by_year: dict[int, list[tuple[str, Any]]] = {}
    for chunk_id, chunk in evidence.items():
        year = _year(_chunk_field(chunk, "year"))
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

    if not years and not metrics:
        status: ReportStatus = "not_required"
    elif not evidence or not cells:
        status = "insufficient"
    elif all(cell.status in {"covered", "year_absent"} for cell in cells):
        status = "complete"
    elif not any(cell.status == "covered" for cell in cells):
        status = "insufficient"
    else:
        status = "partial"

    def cells_for_year(year: int) -> list[CoverageCell]:
        return [cell for cell in cells if cell.year == year]

    covered_years = sorted(
        {
            year
            for year in years
            if not any(
                cell.status == "year_absent" for cell in cells_for_year(year)
            )
            and cells_for_year(year)
            and all(
                cell.status == "covered" for cell in cells_for_year(year)
            )
        }
    )
    missing_years = sorted(set(years) - available)

    matrix: dict[str, dict[str, str]] = {}
    for metric in metrics:
        per_year: dict[str, str] = {}
        for year in years:
            matching = [
                cell
                for cell in cells
                if cell.metric == metric and cell.year == year
            ]
            per_year[str(year)] = (
                matching[0].status if matching else "not_found"
            )
        per_year["*"] = _aggregate_status(list(per_year.values()))
        matrix[metric] = per_year

    missing_metrics = [
        metric
        for metric in metrics
        if matrix.get(metric, {}).get("*") != "covered"
    ]
    missing_metric_years = [
        {
            "metric": cell.metric,
            "metric_label": metric_label(cell.metric),
            "year": cell.year,
            "status": cell.status,
        }
        for cell in cells
        if cell.metric and cell.status != "covered"
    ]
    not_found_cells = [
        cell
        for cell in cells
        if cell.metric and cell.status == "not_found"
    ]

    return CoverageReport(
        status=status,
        expected_years=years,
        covered_years=covered_years,
        missing_years=missing_years,
        cells=cells,
        required_metrics=metrics,
        metrics=matrix,
        missing_metrics=missing_metrics,
        missing_metric_years=missing_metric_years,
        distinct_docs=_distinct_evidence_values(evidence, "doc_id"),
        distinct_sections=_distinct_evidence_values(evidence, "section_path"),
        notes=_notes(status, evidence, missing_years, not_found_cells),
    )
