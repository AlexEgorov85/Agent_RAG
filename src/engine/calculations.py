"""Детерминированные расчёты по структурированным фактам ответа.

LLM извлекает факты и объясняет их, но арифметику выполняет Python. Каждый
расчёт сохраняет ссылки на chunk_id исходных фактов, поэтому результат можно
проверить по тем же citations, что и сам ответ.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

_NUMBER = re.compile(r"[-−–—]?\d[\d \u00a0.,]*")


def parse_number(value: Any) -> float | None:
    """Разбирает финансовое число из числа или строки вроде ``1 508,6``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip().replace("\u00a0", " ")
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")

    match = _NUMBER.search(text)
    if match is None:
        return None
    raw = (
        match.group(0)
        .strip()
        .rstrip(".,")
        .replace(" ", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    if raw in {"-", ""}:
        return None

    try:
        if "," in raw and "." in raw:
            # Десятичный разделитель — последний из встреченных.
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif "," in raw:
            parts = raw.split(",")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
                raw = raw.replace(",", ".")
            else:
                raw = raw.replace(",", "")
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return -number if negative and number > 0 else number


def _fact_year(fact: Mapping[str, Any]) -> int | None:
    value = fact.get("year")
    if value is None:
        value = fact.get("period")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\b(18|19|20|21)\d{2}\b", value)
        if match is not None:
            return int(match.group(0))
    return None


def _fact_metric(fact: Mapping[str, Any]) -> str | None:
    for key in ("metric", "name", "claim"):
        value = fact.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _fact_value(fact: Mapping[str, Any]) -> float | None:
    for key in ("value", "value_num", "amount"):
        if key in fact:
            parsed = parse_number(fact.get(key))
            if parsed is not None:
                return parsed
    return None


def _fact_unit(fact: Mapping[str, Any]) -> str:
    value = fact.get("unit")
    return value.strip() if isinstance(value, str) else ""


def summarize_facts(
    facts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Строит проверяемые сводки по годам для каждой метрики и единицы."""
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)

    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        metric = _fact_metric(fact)
        year = _fact_year(fact)
        value = _fact_value(fact)
        if metric is None or year is None or value is None:
            continue
        unit = _fact_unit(fact)
        key = (metric.casefold(), unit.casefold())
        chunk_id = fact.get("chunk_id")
        chunk_ids = (
            [item for item in chunk_id if isinstance(item, str)]
            if isinstance(chunk_id, list)
            else [chunk_id] if isinstance(chunk_id, str) else []
        )
        # При дубликате года сохраняем первое значение; конфликт отмечает
        # отдельная проверка в validate_answer.
        groups[key].setdefault(
            year,
            {
                "metric": metric,
                "unit": unit,
                "year": year,
                "value": value,
                "chunk_ids": chunk_ids,
            },
        )

    summaries: list[dict[str, Any]] = []
    for (_metric_key, _unit_key), by_year in groups.items():
        points = sorted(by_year.values(), key=lambda item: item["year"])
        if len(points) < 2:
            continue
        first = points[0]
        last = points[-1]
        span = last["year"] - first["year"]
        absolute = last["value"] - first["value"]
        percent = (
            absolute / abs(first["value"]) * 100.0
            if first["value"] != 0
            else None
        )
        cagr = None
        if (
            span > 0
            and first["value"] > 0
            and last["value"] > 0
        ):
            cagr = ((last["value"] / first["value"]) ** (1 / span) - 1) * 100.0

        if absolute > 0:
            direction = "increase"
        elif absolute < 0:
            direction = "decrease"
        else:
            direction = "flat"

        summaries.append(
            {
                "metric": first["metric"],
                "unit": first["unit"],
                "from_year": first["year"],
                "to_year": last["year"],
                "from_value": first["value"],
                "to_value": last["value"],
                "absolute_change": absolute,
                "percent_change": percent,
                "cagr": cagr,
                "min": min(points, key=lambda item: item["value"]),
                "max": max(points, key=lambda item: item["value"]),
                "years": [point["year"] for point in points],
                "direction": direction,
                "chunk_ids": sorted(
                    {
                        chunk_id
                        for point in points
                        for chunk_id in point["chunk_ids"]
                    }
                ),
            }
        )
    return summaries


def format_calculations(
    calculations: Iterable[Mapping[str, Any]],
) -> str:
    """Короткая текстовая сводка расчётов; не выдаёт её за факт из отчёта."""
    lines: list[str] = []
    for item in calculations:
        metric = str(item.get("metric") or "Показатель")
        unit = str(item.get("unit") or "").strip()
        suffix = f" {unit}" if unit else ""
        from_year = item.get("from_year")
        to_year = item.get("to_year")
        absolute = item.get("absolute_change")
        percent = item.get("percent_change")
        if not isinstance(absolute, (int, float)):
            continue
        line = (
            f"«{metric}», {from_year}→{to_year}: "
            f"{absolute:+.2f}{suffix}"
        )
        if isinstance(percent, (int, float)):
            line += f" ({percent:+.1f}%)"
        cagr = item.get("cagr")
        if isinstance(cagr, (int, float)):
            line += f", CAGR {cagr:+.1f}%"
        lines.append(line)
    if not lines:
        return ""
    return "Расчёт по подтверждённым фактам:\n- " + "\n- ".join(lines)
