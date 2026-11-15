"""Проверка ответа: citations, числа, единицы, периоды и причинность.

Модуль не вызывает LLM. Он проверяет только то, что можно проверить по уже
собранным evidence, и возвращает машинно-читаемые issues.

Контракт из ``spec_agent_enhancement.md`` §4.7 сохранён по смыслу, но не по
букве: ``VerificationIssue`` хранит ``chunk_ids`` (так его использует
``AgentService``), а совместимые со спецификацией ``chunk_id``/``fact_index``
доступны как свойства/поля. ``severity`` имеет значения ``fatal``/``warning``;
старое значение ``error`` читается как синоним ``fatal``, чтобы уже
сохранённые diagnostics не ломались.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.coverage import (
    METRIC_ALIASES,
    CoverageReport,
    metric_label,
)
from src.agent.state import Answer, QueryPlan
from src.engine.calculations import parse_number

Severity = Literal["fatal", "warning"]
Status = Literal["ok", "warning", "failed"]

#: ``error`` — устаревший alias ``fatal`` из ранней версии модуля.
_FATAL_SEVERITIES = {"fatal", "error"}

# Число: «1 508,6», «1,508.6», «12,5%», «(1 234)», «−12,5», «100».
# Пробел внутри числа допускается только как разделитель тысяч, поэтому
# перечисление «2022, 2023» распадается на два токена.
_NUMBER = re.compile(
    r"(?<![\d.,])"
    r"[-−–—]?"
    r"(?:\(\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?\)"
    r"|\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)*)"
)
_YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_CAUSAL = re.compile(
    r"\b(?:потому что|поскольку|вследствие|из-за|привело к|привели к|"
    r"приводит к|благодаря|в результате|обусловлен\w*|"
    r"является причиной|влияет на|влияют на|за счёт|за счет)\b",
    re.I,
)
_DISCLOSURE = re.compile(
    r"\bнет\b|отсутств\w*|не найден\w*|недоступ\w*|не раскрыт\w*|"
    r"не представлен\w*|пропуск\w*|не содержится|не содержит|"
    r"не удалось найти|не обнаруж\w*|не указан\w*|"
    r"неполн\w*|частичн\w*|не публику\w*|не привод\w*",
    re.I,
)
_UNIT_MARKER = re.compile(
    r"%|процент\w*|п\.\s*п\.|руб\w*|₽|\$|€|долл\w*|евро|"
    r"трлн\w*|млрд\w*|млн\w*|тыс\w*|\bраз(?:а|ов)?\b",
    re.I,
)
#: Счётные существительные: число 1900–2100 рядом с ними — количество, а не
#: год («2 000 отделений»). Без этого проверка чисел молча теряла такие факты.
_COUNTABLE_NOUN = (
    r"отделени\w*|сотрудник\w*|сотрудниц\w*|клиент\w*|офис\w*|"
    r"устройств\w*|партн[её]р\w*|проект\w*|филиал\w*|банкомат\w*|"
    r"терминал\w*|пользовател\w*|работник\w*|специалист\w*|"
    r"точек|точк\w*|подразделени\w*|компан\w*|магазин\w*|"
    r"сервис\w*|операц\w*|человек\w*|"
    r"количеств\w*|численност\w*|числ\w*|шт\.?"
)
_COUNTABLE_AFTER = re.compile(
    r"^[\s,.:;()«»\"'–—-]*(?:" + _COUNTABLE_NOUN + r")", re.I
)
_COUNTABLE_BEFORE = re.compile(
    r"(?:" + _COUNTABLE_NOUN + r")[\s,.:;()«»\"'–—-]*$", re.I
)
_NOUN_WINDOW = 40
_SCALES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"трлн|триллион"), 1e12),
    (re.compile(r"млрд|миллиард"), 1e9),
    (re.compile(r"млн|миллион"), 1e6),
    (re.compile(r"тыс|тысяч"), 1e3),
)
_SCALE_WINDOW = 14
#: Блок расчётов дописывает ``AgentService``; его числа производные и не
#: обязаны встречаться в источниках.
_CALC_HEADER = "Расчёт по подтверждённым фактам"
_GAP_WINDOW = 120


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    severity: Severity
    message: str
    chunk_ids: list[str] = field(default_factory=list)
    fact_index: int | None = None

    @property
    def chunk_id(self) -> str | None:
        """Первый chunk_id: совместимость со схемой §4.7."""
        return self.chunk_ids[0] if self.chunk_ids else None

    @property
    def is_fatal(self) -> bool:
        return self.severity in _FATAL_SEVERITIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            # ``chunk_ids`` — рабочий контракт AgentService/document_id;
            # ``chunk_id``/``fact_index`` добавлены для совместимости со spec.
            "chunk_ids": list(self.chunk_ids),
            "chunk_id": self.chunk_id,
            "fact_index": self.fact_index,
        }


@dataclass(frozen=True)
class VerificationResult:
    status: Status
    issues: list[VerificationIssue] = field(default_factory=list)
    supported_citations: list[str] = field(default_factory=list)
    unsupported_facts: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True, если нет fatal-issues; warnings ответ не блокируют."""
        return self.status != "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "supported_citations": list(self.supported_citations),
            "unsupported_facts": list(self.unsupported_facts),
        }


def _text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return f"{chunk.get('text', '')} {chunk.get('html', '')}"
    return f"{getattr(chunk, 'text', '')} {getattr(chunk, 'html', '')}"


def _chunk_field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(name)
    return getattr(chunk, name, None)


def _normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def _fold(text: str) -> str:
    return _normalize(text).casefold()


def _number_tokens(text: str) -> list[tuple[float, int, int]]:
    tokens: list[tuple[float, int, int]] = []
    for match in _NUMBER.finditer(text):
        value = parse_number(match.group(0))
        if value is not None:
            tokens.append((value, match.start(), match.end()))
    return tokens


def _scale_near(text: str, end: int) -> float | None:
    """Масштаб единицы сразу после числа: «100 млрд» → 1e9."""
    tail = text[end : end + _SCALE_WINDOW].casefold()
    best: tuple[int, float] | None = None
    for pattern, factor in _SCALES:
        found = pattern.search(tail)
        if found is not None and (best is None or found.start() < best[0]):
            best = (found.start(), factor)
    return best[1] if best is not None else None


def _unit_scale(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.casefold()
    best: tuple[int, float] | None = None
    for pattern, factor in _SCALES:
        found = pattern.search(text)
        if found is not None and (best is None or found.start() < best[0]):
            best = (found.start(), factor)
    return best[1] if best is not None else None


def _same_number(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.005, abs(right) * 1e-6)


def _values_match(
    left: float,
    left_scale: float | None,
    right: float,
    right_scale: float | None,
) -> bool:
    """Сравнить значения с учётом масштаба; разные единицы — не одно число."""
    if _same_number(left, right):
        if (
            left_scale is not None
            and right_scale is not None
            and not _same_number(left * left_scale, right * right_scale)
        ):
            return False
        return True
    if left_scale is not None and right_scale is not None:
        return _same_number(left * left_scale, right * right_scale)
    return False


def _has_value(
    text: str, value: float, scale: float | None = None
) -> bool:
    for candidate, _start, end in _number_tokens(text):
        if _values_match(candidate, _scale_near(text, end), value, scale):
            return True
    return False


def _years(text: str) -> set[int]:
    return {int(match.group(0)) for match in _YEAR.finditer(text)}


def _parse_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = _YEAR.search(value)
        return int(match.group(0)) if match is not None else None
    return None


def _fact_chunk_ids(fact: Mapping[str, Any]) -> list[str]:
    """``chunk_id`` или ``chunk_ids``; нестроковые элементы игнорируются."""
    for key in ("chunk_id", "chunk_ids"):
        value = fact.get(key)
        if isinstance(value, str) and value:
            return [value]
        if isinstance(value, (list, tuple)):
            return [item for item in value if isinstance(item, str) and item]
    return []


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


def _fact_year(fact: Mapping[str, Any]) -> int | None:
    for key in ("year", "period"):
        if key in fact:
            parsed = _parse_year(fact.get(key))
            if parsed is not None:
                return parsed
    return None


def _fact_is_table(fact: Mapping[str, Any], chunks: list[Any]) -> bool:
    kind = fact.get("kind")
    if isinstance(kind, str) and kind.casefold().startswith("table"):
        return True
    return any(
        isinstance(_chunk_field(chunk, "kind"), str)
        and str(_chunk_field(chunk, "kind")).casefold().startswith("table")
        for chunk in chunks
    )


def _normalized_contains(source: str, needle: str) -> bool:
    folded_source = _fold(source)
    folded_needle = _fold(needle)
    return bool(folded_needle) and folded_needle in folded_source


def _quote_found(quote: str, chunk: Any) -> bool:
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return False
    source = _normalize(
        f"{_chunk_field(chunk, 'text') or ''} "
        f"{_chunk_field(chunk, 'html') or ''}"
    )
    if normalized_quote in source:
        return True
    # LLM иногда меняет регистр или типографские кавычки/тире; это всё ещё
    # тот же фрагмент, поэтому проверяем мягкий вариант.
    if normalized_quote.casefold() in source.casefold():
        return True
    return _fold(normalized_quote) in _fold(source)


def _coverage_status(coverage: Any) -> str | None:
    if isinstance(coverage, CoverageReport):
        return coverage.status
    if isinstance(coverage, Mapping):
        status = coverage.get("status")
        return status if isinstance(status, str) else None
    return None


def _coverage_missing(coverage: Any) -> list[int]:
    if isinstance(coverage, CoverageReport):
        return list(coverage.missing_years)
    if isinstance(coverage, Mapping):
        value = coverage.get("missing_years")
        return (
            [year for year in value if isinstance(year, int)]
            if isinstance(value, (list, tuple))
            else []
        )
    return []


def _coverage_cells(coverage: Any) -> list[Any]:
    if isinstance(coverage, CoverageReport):
        return list(coverage.cells)
    if isinstance(coverage, Mapping):
        value = coverage.get("cells")
        return list(value) if isinstance(value, (list, tuple)) else []
    return []


def _cell_field(cell: Any, name: str) -> Any:
    if isinstance(cell, Mapping):
        return cell.get(name)
    return getattr(cell, name, None)


def _coverage_gap_years(coverage: Any) -> list[int]:
    if _coverage_status(coverage) in {"not_required", "complete"}:
        return []
    years: set[int] = set()
    for year in _coverage_missing(coverage):
        years.add(year)
    if isinstance(coverage, CoverageReport):
        years.update(coverage.uncovered_years)
    elif isinstance(coverage, Mapping):
        value = coverage.get("uncovered_years")
        if isinstance(value, (list, tuple)):
            years.update(year for year in value if isinstance(year, int))
    for cell in _coverage_cells(coverage):
        if _cell_field(cell, "status") in {"year_absent", "not_found"}:
            year = _cell_field(cell, "year")
            if isinstance(year, int) and not isinstance(year, bool):
                years.add(year)
    return sorted(years)


def _coverage_gap_metrics(coverage: Any) -> list[tuple[str, int | None]]:
    if _coverage_status(coverage) in {"not_required", "complete"}:
        return []
    gaps: list[tuple[str, int | None]] = []
    for cell in _coverage_cells(coverage):
        if _cell_field(cell, "status") != "not_found":
            continue
        metric = _cell_field(cell, "metric")
        if not isinstance(metric, str) or not metric:
            continue
        year = _cell_field(cell, "year")
        valid_year = (
            year if isinstance(year, int) and not isinstance(year, bool) else None
        )
        gaps.append((metric, valid_year))
    return gaps


def _gap_named(answer_text: str, year: int) -> bool:
    """Год назван рядом со словом о пропуске/отсутствии данных."""
    for match in re.finditer(rf"\b{year}\b", answer_text):
        window = answer_text[
            max(0, match.start() - _GAP_WINDOW) : match.end() + _GAP_WINDOW
        ]
        if _DISCLOSURE.search(window):
            return True
    return False


def _metric_named(answer_text: str, metric: str) -> bool:
    folded = answer_text.casefold()
    labels = {metric.casefold(), metric_label(metric).casefold()}
    labels.update(alias for alias in METRIC_ALIASES.get(metric, ()))
    for label in labels:
        if not label:
            continue
        for match in re.finditer(re.escape(label), folded):
            window = folded[
                max(0, match.start() - _GAP_WINDOW) : match.end() + _GAP_WINDOW
            ]
            if _DISCLOSURE.search(window):
                return True
    return False


def _numeric_claim_marker(text: str, end: int) -> bool:
    return _UNIT_MARKER.search(text[max(0, end - 1) : end + 10]) is not None


def _countable_noun_near(text: str, start: int, end: int) -> bool:
    """Рядом с числом стоит счётное существительное или слово количества."""
    after = text[end : end + _NOUN_WINDOW]
    if _COUNTABLE_AFTER.match(after):
        return True
    before = text[max(0, start - _NOUN_WINDOW) : start]
    return _COUNTABLE_BEFORE.search(before) is not None


def _is_year_token(
    value: float,
    scale: float | None,
    marked: bool,
    *,
    text: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> bool:
    """Похоже ли число на год, который не нужно проверять по источникам.

    Диапазон 1900–2100 сам по себе не признак года: «2 000 отделений» —
    количество. Годом считаем только число без единицы и без счётного
    существительного рядом.
    """
    if not (
        scale is None
        and not marked
        and 1900 <= abs(value) <= 2100
        and float(value).is_integer()
    ):
        return False
    if text is not None and start is not None and end is not None:
        return not _countable_noun_near(text, start, end)
    return True


def _verified_numbers(
    facts: list[float],
    fact_scales: list[float | None],
    source_text: str,
) -> list[tuple[float, float | None]]:
    verified: list[tuple[float, float | None]] = [
        (value, scale) for value, scale in zip(facts, fact_scales)
    ]
    for value, _start, end in _number_tokens(source_text):
        verified.append((value, _scale_near(source_text, end)))
    return verified


def _narrative(text: str) -> str:
    marker = text.find(_CALC_HEADER)
    return text[:marker] if marker >= 0 else text


def verify_answer(
    answer: Answer,
    evidence: Mapping[str, Mapping[str, Any]],
    plan: QueryPlan | None = None,
    coverage: Any = None,
    *,
    structure_complete: bool = True,
) -> VerificationResult:
    """Проверить ответ по evidence; fatal-issues дают ``status="failed"``."""
    if answer.status != "answer":
        return VerificationResult("ok")

    evidence = evidence or {}
    issues: list[VerificationIssue] = []
    cited = {citation.chunk_id for citation in answer.citations}
    citation_year: dict[str, int] = {
        citation.chunk_id: citation.year
        for citation in answer.citations
        if isinstance(citation.year, int) and not isinstance(citation.year, bool)
    }
    supported: list[str] = []

    for citation in answer.citations:
        chunk = evidence.get(citation.chunk_id)
        if chunk is None:
            issues.append(
                VerificationIssue(
                    "citation_outside_manifest",
                    "fatal",
                    "Цитата отсутствует в evidence_manifest.",
                    [citation.chunk_id],
                )
            )
            continue
        quote = citation.quote
        if isinstance(quote, str) and quote.strip() and not _quote_found(
            quote, chunk
        ):
            issues.append(
                VerificationIssue(
                    "quote_not_found",
                    "warning",
                    "Цитата не является фрагментом указанного чанка.",
                    [citation.chunk_id],
                )
            )
        else:
            supported.append(citation.chunk_id)

    units: dict[str, set[str]] = {}
    fact_years: set[int] = set()
    fact_values: list[float] = []
    fact_scales: list[float | None] = []
    unsupported_facts: list[int] = []

    for index, fact in enumerate(answer.facts):
        if not isinstance(fact, Mapping):
            continue
        chunk_ids = _fact_chunk_ids(fact)
        if not chunk_ids or any(chunk_id not in cited for chunk_id in chunk_ids):
            issues.append(
                VerificationIssue(
                    "fact_without_citation",
                    "fatal",
                    "Факт не ссылается на возвращённую citation.",
                    chunk_ids,
                    fact_index=index,
                )
            )
            unsupported_facts.append(index)
            continue

        chunks = [
            evidence[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in evidence
        ]
        # Кэш chunk_id → текст: у больших ответов факты часто делят чанки.
        source_text = " ".join(_text(chunk) for chunk in chunks)
        fact_scales_value = _unit_scale(fact.get("unit"))
        value = _fact_value(fact)
        fact_supported = True

        metric = _fact_metric(fact)
        if isinstance(metric, str):
            unit = fact.get("unit")
            if isinstance(unit, str) and unit.strip():
                units.setdefault(metric.casefold(), set()).add(
                    unit.strip().casefold()
                )

        if value is not None:
            fact_values.append(value)
            fact_scales.append(fact_scales_value)
            if chunks and not _has_value(
                source_text, value, fact_scales_value
            ):
                issues.append(
                    VerificationIssue(
                        "fact_value_not_in_source",
                        "fatal",
                        "Значение факта не найдено в цитируемом чанке.",
                        chunk_ids,
                        fact_index=index,
                    )
                )
                fact_supported = False

        year = _fact_year(fact)
        if year is not None:
            fact_years.add(year)
            source_years: set[int] = set()
            for chunk in chunks:
                source_years.update(_years(_text(chunk)))
                meta_year = _parse_year(_chunk_field(chunk, "year"))
                if meta_year is not None:
                    source_years.add(meta_year)
                for chunk_id in chunk_ids:
                    if chunk_id in citation_year:
                        source_years.add(citation_year[chunk_id])
            # Год чанка не подменяет год факта: если факт с другим годом не
            # подтверждён ни метаданными, ни текстом — это mismatch.
            if chunks and year not in source_years:
                issues.append(
                    VerificationIssue(
                        "fact_year_mismatch",
                        "fatal",
                        "Год факта противоречит источнику.",
                        chunk_ids,
                        fact_index=index,
                    )
                )
                fact_supported = False

        if chunks and _fact_is_table(fact, chunks):
            row_label = fact.get("row_label")
            column_label = fact.get("column_label")
            missing = [
                name
                for name, label in (
                    ("row_label", row_label),
                    ("column_label", column_label),
                )
                if not isinstance(label, str) or not label.strip()
            ]
            if missing:
                issues.append(
                    VerificationIssue(
                        "table_anchor_missing",
                        "warning",
                        "Для табличного факта не указаны "
                        + " и ".join(missing)
                        + ".",
                        chunk_ids,
                        fact_index=index,
                    )
                )
            else:
                absent = [
                    name
                    for name, label in (
                        ("row_label", row_label),
                        ("column_label", column_label),
                    )
                    if not _normalized_contains(source_text, str(label))
                ]
                if absent:
                    issues.append(
                        VerificationIssue(
                            "table_anchor_missing",
                            "warning",
                            "Якорь таблицы не найден в источнике: "
                            + ", ".join(absent)
                            + ".",
                            chunk_ids,
                            fact_index=index,
                        )
                    )

        if not fact_supported:
            unsupported_facts.append(index)

    if any(len(values) > 1 for values in units.values()):
        issues.append(
            VerificationIssue(
                "unit_mismatch",
                "warning",
                "Одна метрика приведена в разных единицах.",
            )
        )

    if plan is not None and (
        plan.requires_comparison or plan.answer_mode == "comparison"
    ):
        if coverage is not None and _coverage_status(coverage) not in {
            None,
            "not_required",
            "complete",
        }:
            issues.append(
                VerificationIssue(
                    "period_incomplete",
                    "warning",
                    "Период покрыт не полностью; ответ не должен выглядеть полным.",
                    [str(year) for year in _coverage_gap_years(coverage)],
                )
            )
        if fact_years and len(fact_years) < 2:
            issues.append(
                VerificationIssue(
                    "period_mismatch",
                    "warning",
                    "Для сравнения найден факт только за один год.",
                )
            )

    narrative = _narrative(answer.answer or "")
    source_all = " ".join(
        _text(evidence[chunk_id])
        for chunk_id in cited
        if chunk_id in evidence
    )

    if _CAUSAL.search(narrative) and not _CAUSAL.search(source_all):
        issues.append(
            VerificationIssue(
                "causal_claim_unverified",
                "warning",
                "Причинный вывод не подтверждён причинным утверждением "
                "в источниках.",
                sorted(cited),
            )
        )

    gap_years = _coverage_gap_years(coverage)
    undisclosed_years = [
        year for year in gap_years if not _gap_named(narrative, year)
    ]
    undisclosed_metrics = [
        (metric_label(metric), year)
        for metric, year in _coverage_gap_metrics(coverage)
        if not _metric_named(narrative, metric)
    ]
    if undisclosed_years or undisclosed_metrics:
        details = [str(year) for year in undisclosed_years]
        details.extend(
            f"{label} {year}" if year is not None else label
            for label, year in undisclosed_metrics
        )
        issues.append(
            VerificationIssue(
                "gap_not_disclosed",
                "warning",
                "Известные пробелы не раскрыты в ответе: "
                + ", ".join(details)
                + ".",
            )
        )

    verified = _verified_numbers(fact_values, fact_scales, source_all)
    seen_claims: set[str] = set()
    for value, start, end in _number_tokens(narrative):
        scale = _scale_near(narrative, end)
        marked = _numeric_claim_marker(narrative, end)
        if _is_year_token(
            value, scale, marked, text=narrative, start=start, end=end
        ):
            continue
        literal = _normalize(narrative[start:end])
        if not (
            marked
            or scale is not None
            or abs(value) >= 1000
            or not float(value).is_integer()
        ):
            continue
        if literal in seen_claims:
            continue
        if any(
            _values_match(value, scale, item, item_scale)
            for item, item_scale in verified
        ):
            continue
        seen_claims.add(literal)
        issues.append(
            VerificationIssue(
                "numeric_claim_unverified",
                "warning",
                f"Число «{literal}» в ответе не подтверждено фактом или "
                "цитируемым чанком.",
                sorted(cited),
            )
        )

    if plan is not None and plan.intent == "structural" and not structure_complete:
        issues.append(
            VerificationIssue(
                "structure_unverified",
                "warning",
                "Структурное сравнение завершено не для всех документов.",
            )
        )

    if any(issue.is_fatal for issue in issues):
        status: Status = "failed"
    elif issues:
        status = "warning"
    else:
        status = "ok"
    return VerificationResult(
        status,
        issues,
        supported_citations=list(dict.fromkeys(supported)),
        unsupported_facts=sorted(set(unsupported_facts)),
    )
