"""Детерминированный роутер периода без LLM и без хардкода годов корпуса.

Границы периода всегда выводятся из ``available_years``, которые фабрика
собирает из реально загруженных чанков. Поэтому «последние N лет» означает
последние N лет *доступного корпуса*, а не фиксированную дату.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.agent.state import QueryPlan

#: Четырёхзначный год в разумном для отчётов диапазоне. Это не границы корпуса,
#: а защита от того, чтобы сумма «1 508,6» превратилась в год 1508.
_YEAR_TOKEN = r"(?:18|19|20|21)\d{2}"

_RANGE_PATTERNS = (
    re.compile(rf"\b({_YEAR_TOKEN})\s*[-–—]\s*({_YEAR_TOKEN})\b"),
    re.compile(
        rf"\bс\s+({_YEAR_TOKEN})\s*(?:год\w*)?\s*(?:по|до)\s+({_YEAR_TOKEN})\b",
        re.I,
    ),
    re.compile(
        rf"\bмежду\s+({_YEAR_TOKEN})\s+и\s+({_YEAR_TOKEN})\b", re.I
    ),
    re.compile(
        rf"\b({_YEAR_TOKEN})\s+(?:по|до)\s+({_YEAR_TOKEN})\b", re.I
    ),
)
_YEAR = re.compile(rf"\b({_YEAR_TOKEN})\b")
_AFTER = re.compile(rf"\bпосле\s+({_YEAR_TOKEN})\b", re.I)
_BEFORE = re.compile(rf"\bдо\s+({_YEAR_TOKEN})\b", re.I)
_SINCE = re.compile(rf"\b(?:начиная\s+с|с)\s+({_YEAR_TOKEN})\b", re.I)
_LAST_N = re.compile(
    r"(?:за\s+)?(?:последн\w+\s+)?"
    r"(?<![\d])(\d{1,2}|[а-яё]+)\s+(?:[а-яё]+\s+){0,2}"
    r"(?:лет|год(?:а|ов)?)\b",
    re.I,
)
_LAST_N_HYPHEN = re.compile(
    r"\b(\d{1,2}|[а-яё]+)[- ]летн\w*\b", re.I
)
_LAST_N_FUSED = re.compile(r"\b([а-яё]+)летн\w*\b", re.I)
_ALL_YEARS = re.compile(
    r"\b(?:за\s+все\s+время|за\s+весь\s+период|по\s+всем\s+годам|"
    r"все\s+годы|все\s+отч[её]т\w*|весь\s+корпус|с\s+начала\s+периода)\b",
    re.I,
)
_PREV_YEAR = re.compile(r"\bпрошл\w+\s+год\w*\b", re.I)
_YEAR_BEFORE_PREV = re.compile(r"\bпозапрошл\w+\s+год\w*\b", re.I)
_RECENT_YEARS = re.compile(
    r"\bпоследн(?:ие|их|им|ими)\s+год\w*\b", re.I
)
_LATEST_YEAR = re.compile(
    r"\bпоследн(?:ий|его|ему|ем)\s+год\w*\b", re.I
)
_YOY = re.compile(
    r"\b(?:год\s+к\s+году|г/г|по\s+сравнению\s+с\s+прошл\w+\s+год\w*)\b",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(?:измен|динамик|эволюц|менял|меняются|тренд|сравн|разниц|"
    r"отлича|рост|вырос|увелич|сниж|уменьш|сократ|сокращ|падени|"
    r"колеб|траектор|хронологи)\w*|\bпо\s+годам\b",
    re.I,
)
_STRUCTURAL = re.compile(
    r"\b(?:"
    r"структур\w*(?:\s+[а-яё]+){0,2}\s+(?:отч[её]т|документ|раздел)\w*|"
    r"оглавлен\w*|подраздел\w*|раздел\w*|"
    r"дерев\w*\s+раздел\w*|"
    r"содержани\w*\s+(?:раздел|отч[её]т)\w*"
    r")\b",
    re.I,
)

_WORD_NUMBERS = {
    "один": 1,
    "одного": 1,
    "одну": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "три": 3,
    "трех": 3,
    "четыре": 4,
    "четырех": 4,
    "пять": 5,
    "пяти": 5,
    "шесть": 6,
    "шести": 6,
    "семь": 7,
    "семи": 7,
    "восемь": 8,
    "восьми": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "пятнадцать": 15,
    "двадцать": 20,
}

#: Защита от абсурдно широких диапазонов. Не привязана к годам корпуса.
_MAX_EXPANDED_SPAN = 50


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def _count(token: str) -> int | None:
    value = _normalize(token).strip()
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 50 else None
    return _WORD_NUMBERS.get(value)


def _years_from_text(
    text: str,
    min_year: int | None,
    max_year: int | None,
) -> set[int]:
    """Извлекает годы из диапазонов, открытых границ и отдельных упоминаний."""
    years: set[int] = set()
    range_found = False

    for pattern in _RANGE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        start, end = sorted(int(part) for part in match.groups())
        if end - start <= _MAX_EXPANDED_SPAN:
            years.update(range(start, end + 1))
        elif min_year is not None and max_year is not None:
            left, right = max(start, min_year), min(end, max_year)
            if left <= right:
                years.update(range(left, right + 1))
        else:
            years.update({start, end})
        range_found = True
        break

    # Отдельные годы могут стоять рядом с диапазоном: «2015-2017 и 2020».
    years.update(int(year) for year in _YEAR.findall(text))
    if range_found:
        return years

    after = _AFTER.search(text)
    if after is not None and max_year is not None:
        start = int(after.group(1)) + 1
        if start <= max_year:
            years.update(range(start, max_year + 1))

    before = _BEFORE.search(text)
    if before is not None and min_year is not None:
        end = int(before.group(1))
        if end >= min_year:
            years.update(range(min_year, end + 1))

    since = _SINCE.search(text)
    if since is not None and max_year is not None:
        start = int(since.group(1))
        if start <= max_year:
            years.update(range(start, max_year + 1))

    return years


class SimpleRouter:
    """Правила вместо LLM: период выводится из корпуса, а не из константы.

    ``available_years`` передаёт фабрика, поэтому один и тот же роутер работает
    и на dev-корпусе из четырёх лет, и на полном корпусе из девяти отчётов.
    """

    def __init__(
        self,
        available_years: Iterable[int],
        *,
        max_period_years: int = 30,
    ) -> None:
        self.available_years = tuple(
            sorted({int(year) for year in available_years})
        )
        self.min_year = self.available_years[0] if self.available_years else None
        self.max_year = self.available_years[-1] if self.available_years else None
        if max_period_years < 1:
            raise ValueError("max_period_years must be positive")
        self.max_period_years = max_period_years

    def analyze(self, question: str, filters: object) -> QueryPlan:
        del filters
        text = _normalize(question)
        years = _years_from_text(text, self.min_year, self.max_year)
        has_comparison = bool(_COMPARISON.search(text))
        has_all_years = bool(_ALL_YEARS.search(text))
        has_recent = bool(_RECENT_YEARS.search(text))
        has_latest = bool(_LATEST_YEAR.search(text))
        has_structural = bool(_STRUCTURAL.search(text))

        count_match = (
            _LAST_N.search(text)
            or _LAST_N_HYPHEN.search(text)
            or _LAST_N_FUSED.search(text)
        )
        count = _count(count_match.group(1)) if count_match else None
        has_relative = bool(
            count is not None
            or has_recent
            or has_latest
            or _PREV_YEAR.search(text)
            or _YEAR_BEFORE_PREV.search(text)
            or _YOY.search(text)
        )
        if (
            not years
            and not has_comparison
            and not has_all_years
            and not has_relative
            and not has_structural
        ):
            return QueryPlan()

        anchor = max(years) if years else self.max_year

        if count is not None and self.max_year is not None:
            span = (
                self.max_year - self.min_year + 1
                if self.min_year is not None
                else count
            )
            if count >= span:
                years = set(self.available_years)
            else:
                years = set(
                    range(self.max_year - count + 1, self.max_year + 1)
                )
        elif has_all_years or has_recent:
            years = set(self.available_years)
        else:
            if anchor is not None and _YEAR_BEFORE_PREV.search(text):
                years.add(anchor - 2)
            elif anchor is not None and _PREV_YEAR.search(text):
                years.add(anchor - 1)
            elif anchor is not None and has_latest:
                years.add(anchor)
            if anchor is not None and _YOY.search(text):
                years.update({anchor - 1, anchor})

        if not years and (has_comparison or has_structural):
            # «Как менялись подходы» / «как менялась структура» без периода —
            # весь доступный корпус.
            years = set(self.available_years)

        if len(years) > self.max_period_years:
            years = set(sorted(years)[-self.max_period_years:])

        comparison = has_comparison or len(years) > 1
        if has_structural:
            intent = "structural"
        elif years or has_comparison:
            intent = "temporal"
        else:
            intent = "factoid"
        return QueryPlan(
            intent=intent,
            years=sorted(years),
            needs_rewriting=True,
            answer_mode=(
                "comparison" if comparison and len(years) > 1 else "concise"
            ),
        )
