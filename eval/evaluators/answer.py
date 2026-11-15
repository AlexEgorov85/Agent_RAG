"""Детерминированный evaluator ответа.

Проверяются статус, обязательные факты и годы, citations, цитаты, расчёты,
abstention, запреты и наблюдаемое поведение инструментов. Оценка не заменяет
LLM-судью: rubric остаётся ``None``, если судья не включён. Critical errors
делают тест провальным независимо от стиля ответа.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from eval.metrics import aggregate_mean

_NUMBER = re.compile(r"[-+]?\d[\d\s\u00a0]*(?:[.,]\d+)?")
_CLAIM_COMPLETE = (
    "полный период",
    "полностью",
    "за весь период",
    "все годы",
    "полное сравнение",
    "полный ряд",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold().replace("ё", "е")


def parse_number(value: Any) -> float | None:
    """Число из ``1 508,6`` / ``25.3`` / ``"25,3%"``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(str(value))
    if not match:
        return None
    text = (
        match.group(0)
        .replace("\u00a0", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(text)
    except ValueError:
        return None


def numbers_in_text(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER.finditer(text or ""):
        raw = (
            match.group(0)
            .replace("\u00a0", "")
            .replace(" ", "")
            .replace(",", ".")
        )
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _within(actual: float, expected: float, tolerance: float) -> bool:
    if expected == 0:
        return abs(actual) <= max(tolerance, 1e-6)
    return abs(actual - expected) <= max(abs(expected) * tolerance, 1e-9)


#: Масштаб единицы: «1,5 трлн» и «1 508,6 млрд» — один порядок величины.
_SCALE_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"трлн|триллион", re.I), 1e12),
    (re.compile(r"млрд|миллиард", re.I), 1e9),
    (re.compile(r"млн|миллион", re.I), 1e6),
    (re.compile(r"тыс|тысяч", re.I), 1e3),
)


def unit_scale(value: Any) -> float | None:
    """Множитель единицы измерения или ``None``, если масштаб не указан."""
    if not isinstance(value, str):
        return None
    for pattern, factor in _SCALE_PATTERNS:
        if pattern.search(value):
            return factor
    return None


def _within_scaled(
    actual: float,
    actual_scale: float | None,
    expected: float,
    expected_scale: float | None,
    tolerance: float,
) -> bool:
    if actual_scale is not None and expected_scale is not None:
        actual_value = actual * actual_scale
        expected_value = expected * expected_scale
        if expected_value == 0:
            return abs(actual_value) <= max(tolerance, 1e-6)
        return abs(actual_value - expected_value) <= max(
            abs(expected_value) * tolerance, 1e-9
        )
    return _within(actual, expected, tolerance)


def _contains_any(text: str, aliases: Iterable[Any]) -> bool:
    normalized = normalize_text(text)
    for alias in aliases:
        value = normalize_text(alias)
        if value and value in normalized:
            return True
    return False


def mention_present(text: str, mention: str) -> bool:
    """Обязательное упоминание: подстрока или те же слова в любой словоформе.

    ``must_mention`` задаёт основу термина («ипотека», «климат»), а ответ
    естественно содержит падежную форму («ипотечного», «климатические»).
    Точная подстрока здесь была бы проверкой орфографии, а не содержания,
    поэтому после неё применяется то же сравнение словоформ, что и для
    названий показателей.
    """
    normalized = normalize_text(text)
    value = normalize_text(mention)
    if not value:
        return False
    if value in normalized:
        return True
    words = _words(mention)
    if not words:
        return False
    return _alias_words_match(text, [mention])


#: Токены названий показателей. Генератор называет метрику свободным текстом
#: («корпоративный кредитный портфель»), поэтому сравнение только по точной
#: подстроке пропускает падежные формы и вставленные определения.
_WORD = re.compile(r"[0-9a-zа-я]+")


def _words(value: Any) -> list[str]:
    return [
        token
        for token in _WORD.findall(normalize_text(value))
        if len(token) > 2
    ]


def _same_word(left: str, right: str) -> bool:
    """Сравнить слова с учётом русской словоформы, не теряя смысл.

    Правило узкое: общий префикс не короче 4 букв и почти во всю длину более
    короткого слова (``риски``/``рисками``, ``портфель``/``портфеля``,
    ``корпоративный``/``корпоративного``). ``рост`` и ``российский`` дают всего
    три общих буквы и не склеиваются.
    """
    if left == right:
        return True
    prefix = 0
    for first, second in zip(left, right):
        if first != second:
            break
        prefix += 1
    return prefix >= 4 and prefix >= min(len(left), len(right)) - 3


def _alias_words_match(text: str, aliases: Iterable[Any]) -> bool:
    """Все значимые слова alias встречаются в ``text`` (с учётом словоформ)."""
    target = _words(text)
    if not target:
        return False
    for alias in aliases:
        words = _words(alias)
        if not words:
            continue
        if all(any(_same_word(word, candidate) for candidate in target) for word in words):
            return True
    return False


#: Числовые сокращения и словоформы — одна и та же единица измерения.
#: «2,6 млн пользователей» и «2,6 миллионов пользователей» не должны
#: расходиться только из-за формы слова.
_UNIT_WORD_FORMS = (
    (re.compile(r"\bмлн\b\.?", re.I), "миллион"),
    (re.compile(r"\bмлрд\b\.?", re.I), "миллиард"),
    (re.compile(r"\bтрлн\b\.?", re.I), "триллион"),
    (re.compile(r"\bмиллион\w*", re.I), "миллион"),
    (re.compile(r"\bмиллиард\w*", re.I), "миллиард"),
    (re.compile(r"\bтриллион\w*", re.I), "триллион"),
)


def normalize_unit_text(value: Any) -> str:
    """Нормализует единицу: «млн»/«миллионов» → одна каноническая форма."""
    text = normalize_text(value)
    for pattern, replacement in _UNIT_WORD_FORMS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any_unit(text: str, aliases: Iterable[Any]) -> bool:
    normalized = normalize_unit_text(text)
    for alias in aliases:
        value = normalize_unit_text(alias)
        if value and value in normalized:
            return True
    return False


def _metric_matches(fact: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    aliases = set()
    for key in ("metric_aliases", "aliases", "names"):
        value = required.get(key) or []
        if isinstance(value, list):
            aliases.update(str(item) for item in value)
    if required.get("metric"):
        aliases.add(str(required["metric"]))
    actual = fact.get("metric") or fact.get("name") or fact.get("label")
    if actual is not None and str(actual) in aliases:
        return True
    # Факты генератора иногда содержат только название показателя.
    text = fact.get("metric_text") or fact.get("title") or fact.get("name") or actual
    if text is None:
        return False
    if _contains_any(str(text), aliases):
        return True
    # Свободная формулировка метрики: «корпоративный кредитный портфель»
    # должно совпадать с alias «корпоративный портфель».
    return _alias_words_match(str(text), aliases)


def _unit_aliases(required: Mapping[str, Any]) -> list[str]:
    aliases = [required.get("unit")]
    for key in ("unit_aliases", "units"):
        value = required.get(key) or []
        if isinstance(value, list):
            aliases.extend(value)
    return [str(item) for item in aliases if item]


def _unit_dimension(value: Any) -> str | None:
    """Измерение единицы без масштаба: «трлн руб.» и «млрд руб.» — рубли."""
    if not isinstance(value, str):
        return None
    text = normalize_unit_text(value)
    if re.search(r"%|процент", text):
        return "percent"
    if "п.п" in text or "пункт" in text:
        return "pp"
    if re.search(r"руб|₽", text):
        return "rub"
    if re.search(r"долл|usd|\$", text):
        return "usd"
    if re.search(r"евро|eur|€", text):
        return "eur"
    return None


def _unit_matches(fact: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    aliases = _unit_aliases(required)
    actual = fact.get("unit") or fact.get("units") or ""
    if _contains_any_unit(str(actual), aliases):
        return True
    # Масштаб не меняет измерение: «1,5 трлн руб.» подтверждает факт
    # «1 508,6 млрд руб.», если значение совпадает после пересчёта.
    actual_dimension = _unit_dimension(str(actual))
    if actual_dimension is None:
        return False
    return any(
        _unit_dimension(alias) == actual_dimension for alias in aliases
    )


def _fact_source_ids(fact: Mapping[str, Any]) -> list[str]:
    value = fact.get("chunk_id", fact.get("chunk_ids"))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    source = fact.get("source")
    if isinstance(source, Mapping) and source.get("chunk_id"):
        return [str(source["chunk_id"])]
    return []


def _citations(turn: Mapping[str, Any]) -> list[dict[str, Any]]:
    answer = turn.get("answer")
    if not isinstance(answer, Mapping):
        return []
    citations = answer.get("citations")
    return [item for item in citations or [] if isinstance(item, Mapping)]


def _fact_match(
    required: Mapping[str, Any],
    answer: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Сопоставить обязательный факт со структурированными facts или текстом."""
    text = str(answer.get("text") or "")
    facts = [fact for fact in answer.get("facts", []) or [] if isinstance(fact, Mapping)]
    cited = {
        str(citation.get("chunk_id"))
        for citation in citations
        if citation.get("chunk_id")
    }
    expected_value = parse_number(required.get("value"))
    expected_scale = unit_scale(required.get("unit")) or unit_scale(
        required.get("value")
    )
    tolerance = required.get("tolerance", 0.01)
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        tolerance = 0.01

    metric_seen = False
    year_seen = False
    for fact in facts:
        if not _metric_matches(fact, required):
            continue
        metric_seen = True
        if required.get("year") is not None and fact.get("year") != required["year"]:
            continue
        year_seen = True
        value = parse_number(fact.get("value"))
        fact_scale = unit_scale(fact.get("unit")) or unit_scale(
            fact.get("value")
        )
        if expected_value is not None and value is not None and _within_scaled(
            value, fact_scale, expected_value, expected_scale, float(tolerance)
        ):
            unit_ok = _unit_matches(fact, required) or _contains_any_unit(
                text,
                [required.get("unit") or "", *(required.get("unit_aliases") or [])],
            )
            source_ids = _fact_source_ids(fact)
            source_ok = bool(source_ids) and all(item in cited for item in source_ids)
            if not source_ids:
                source_ok = bool(cited)
            return {
                "matched": bool(unit_ok and source_ok),
                "metric_seen": True,
                "year_seen": True,
                "value_seen": True,
                "unit_seen": unit_ok,
                "source_ok": source_ok,
                "matched_in_text": False,
                "source_ids": source_ids,
            }

    # Fallback: метрика, год, число и единица в тексте рядом с citation.
    metric_in_text = _contains_any(text, [
        required.get("metric") or "",
        *(required.get("metric_aliases") or []),
    ])
    year_value = required.get("year")
    year_in_text = year_value is None or str(year_value) in text
    unit_in_text = _contains_any_unit(text, [
        required.get("unit") or "",
        *(required.get("unit_aliases") or []),
    ])
    value_in_text = False
    if expected_value is not None:
        value_in_text = any(
            _within(value, expected_value, float(tolerance))
            for value in numbers_in_text(text)
        )
    matched = bool(
        metric_in_text
        and year_in_text
        and value_in_text
        and unit_in_text
        and cited
    )
    return {
        "matched": matched,
        "metric_seen": metric_in_text,
        "year_seen": year_in_text,
        "value_seen": value_in_text,
        "unit_seen": unit_in_text,
        "source_ok": bool(cited),
        "matched_in_text": True,
        "source_ids": sorted(cited),
    }


def _structural_section_texts(item: Mapping[str, Any]) -> list[str]:
    """Все варианты названия раздела в наблюдаемом structural_change."""
    values: list[str] = []
    section = item.get("section")
    if isinstance(section, str) and section.strip():
        values.append(section)
    for key in ("from_section", "to_section"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
        elif isinstance(value, Mapping):
            for field_name in ("title", "section_path"):
                text = value.get(field_name)
                if isinstance(text, str) and text.strip():
                    values.append(text)
    for key in ("section_path", "from_section_path", "to_section_path"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            values.append(text)
    return values


def _structural_match(
    required: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> bool:
    """Сопоставить required structural change с наблюдаемым артефактом.

    Эталон использует ``type``/``section``/``from_doc``/``to_doc``, а
    ``structure_compare`` — ``change_type`` и ``from_section``/``to_section``
    (словари с ``title``/``section_path``). Оба формата допустимы: иначе
    корректный structural-результат агента никогда не совпал бы с эталоном.
    ``from_doc``/``to_doc`` сравниваются только если артефакт их содержит:
    у детерминированного сравнения документы выражены годами.
    """
    required_type = normalize_text(required.get("type") or required.get("change_type"))
    observed_type = normalize_text(observed.get("change_type") or observed.get("type"))
    if required_type and observed_type and required_type != observed_type:
        return False
    required_section = required.get("section")
    if isinstance(required_section, str) and required_section.strip():
        candidates = _structural_section_texts(observed)
        if not candidates:
            return False
        normalized_required = normalize_text(required_section)
        section_ok = any(
            normalized_required in normalize_text(candidate)
            or normalize_text(candidate) in normalized_required
            or _alias_words_match(candidate, [required_section])
            for candidate in candidates
        )
        if not section_ok:
            return False
    for key in ("from_doc", "to_doc", "from_year", "to_year"):
        if key not in observed or observed.get(key) is None:
            continue
        if key in required and required.get(key) is not None:
            if str(observed.get(key)) != str(required.get(key)):
                return False
    return True


def _calc_match(
    required: Mapping[str, Any],
    calculations: Sequence[Mapping[str, Any]],
) -> bool:
    kind = required.get("kind", "percent_change")
    field = {
        "percent_change": "percent_change",
        "cagr": "cagr",
        "absolute_change": "absolute_change",
    }.get(kind, kind)
    expected = parse_number(required.get("value"))
    tolerance = required.get("tolerance", 0.02)
    if expected is None or not isinstance(tolerance, (int, float)):
        return False
    for calculation in calculations:
        if required.get("metric") and calculation.get("metric") != required["metric"]:
            continue
        if required.get("from_year") is not None and calculation.get("from_year") != required["from_year"]:
            continue
        if required.get("to_year") is not None and calculation.get("to_year") != required["to_year"]:
            continue
        value = parse_number(calculation.get(field, calculation.get("value")))
        if value is not None and _within(value, expected, float(tolerance)):
            return True
    return False


def _warning_union(turn: Mapping[str, Any]) -> set[str]:
    answer = turn.get("answer") if isinstance(turn.get("answer"), Mapping) else {}
    warnings = set(str(item) for item in (answer.get("warnings") or []))
    warnings.update(str(item) for item in (turn.get("run_warnings") or []))
    return warnings


def _tool_names(turn: Mapping[str, Any]) -> list[str]:
    return [
        str(event.get("tool_name"))
        for event in turn.get("tool_events", []) or []
        if isinstance(event, Mapping)
    ]


def _evaluate_turn(
    spec: Mapping[str, Any],
    turn: Mapping[str, Any],
    *,
    chunk_store: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_coverage: float | None = None,
) -> dict[str, Any]:
    answer = turn.get("answer") if isinstance(turn.get("answer"), Mapping) else {}
    expected = spec.get("expected") if isinstance(spec.get("expected"), Mapping) else {}
    checks = spec.get("checks") if isinstance(spec.get("checks"), Mapping) else {}
    behavior = (
        spec.get("expected_behavior")
        if isinstance(spec.get("expected_behavior"), Mapping)
        else {}
    )
    citations = _citations(turn)
    cited_ids = [str(citation.get("chunk_id")) for citation in citations if citation.get("chunk_id")]
    text = str(answer.get("text") or "")
    status = answer.get("status")
    expected_status = spec.get("expected_status", "answer")
    critical: list[dict[str, Any]] = []
    warnings: list[str] = []
    checks_rows: list[dict[str, Any]] = []
    not_measured: list[str] = []

    def add_critical(code: str, message: str) -> None:
        critical.append({"code": code, "message": message})

    def add_check(name: str, passed: bool, *, mandatory: bool = True, detail: str = "") -> None:
        checks_rows.append(
            {
                "name": name,
                "pass": bool(passed),
                "mandatory": bool(mandatory),
                "detail": detail,
            }
        )

    # 1. Статус ответа.
    status_ok = status == expected_status
    if not status_ok:
        add_critical(
            "wrong_status",
            f"Ожидался status={expected_status}, получен {status!r}",
        )
    add_check("status_accuracy", status_ok, detail=f"{status!r} vs {expected_status!r}")

    # 2. Обязательные факты.
    required_facts = [
        item for item in expected.get("required_facts", []) or []
        if isinstance(item, Mapping)
    ]
    fact_details: list[dict[str, Any]] = []
    fact_matches = 0
    for index, required in enumerate(required_facts):
        match = _fact_match(required, answer, citations)
        fact_details.append(
            {
                "index": index,
                "metric": required.get("metric"),
                "year": required.get("year"),
                "value": required.get("value"),
                **match,
            }
        )
        if match["matched"]:
            fact_matches += 1
            continue
        if not match["metric_seen"]:
            add_critical(
                "fact_missing",
                f"Обязательный показатель {required.get('metric')!r} не найден",
            )
        elif not match["year_seen"]:
            add_critical(
                "fact_wrong_year",
                f"Показатель {required.get('metric')!r} приписан не тому году",
            )
        elif not match["value_seen"]:
            add_critical(
                "fact_wrong_value",
                f"Неверное значение для {required.get('metric')!r}",
            )
        elif not match["source_ok"]:
            add_critical(
                "unsupported_claim",
                f"Факт {required.get('metric')!r} не подтверждён citation",
            )
        else:
            add_critical(
                "fact_missing",
                f"Факт {required.get('metric')!r} не прошёл проверку единицы/источника",
            )
    fact_accuracy = fact_matches / len(required_facts) if required_facts else None
    if required_facts:
        add_check("fact_accuracy", fact_matches == len(required_facts),
                  detail=f"{fact_matches}/{len(required_facts)}")

    # 3. Годы и пропуски.
    required_years = [
        year for year in expected.get("required_years", []) or []
        if isinstance(year, int) and not isinstance(year, bool)
    ]
    missing_years = [
        year for year in expected.get("missing_years", []) or []
        if isinstance(year, int) and not isinstance(year, bool)
    ]
    fact_years = {
        fact.get("year")
        for fact in answer.get("facts", []) or []
        if isinstance(fact, Mapping) and isinstance(fact.get("year"), int)
    }
    # Год может быть подтверждён метаданными валидной citation: ответ про
    # 2023 год не обязан повторять «2023» в тексте, если источник — отчёт 2023.
    citation_years = {
        citation.get("year")
        for citation in citations
        if isinstance(citation.get("year"), int)
        and not isinstance(citation.get("year"), bool)
    }
    missing_required_years: list[int] = []
    for year in required_years:
        if year in missing_years:
            continue
        if year in fact_years or year in citation_years or str(year) in text:
            continue
        missing_required_years.append(year)
        add_critical("fact_missing", f"В ответе отсутствует обязательный год {year}")
    years_ok = not missing_required_years
    add_check(
        "required_years",
        years_ok,
        detail=", ".join(str(year) for year in missing_required_years),
    )

    required_documents = [
        str(doc_id) for doc_id in expected.get("required_documents", []) or []
        if isinstance(doc_id, str) and doc_id
    ]
    if required_documents:
        cited_docs = {
            str(citation.get("doc_id"))
            for citation in citations
            if citation.get("doc_id")
        }
        if chunk_store is not None:
            for citation in citations:
                chunk_id = citation.get("chunk_id")
                chunk = chunk_store.get(str(chunk_id)) if chunk_id else None
                if chunk and chunk.get("doc_id"):
                    cited_docs.add(str(chunk["doc_id"]))
        missing_documents = [
            doc_id for doc_id in required_documents if doc_id not in cited_docs
        ]
        if missing_documents:
            add_critical(
                "wrong_document",
                "Нет подтверждающих citations из документов: "
                + ", ".join(missing_documents),
            )
        add_check(
            "required_documents",
            not missing_documents,
            detail=", ".join(missing_documents),
        )

    if expected_status == "answer":
        for year in missing_years:
            if str(year) not in text:
                add_critical(
                    "evidence_gap_not_disclosed",
                    f"Не раскрыт пропуск года {year}",
                )
                break
        if missing_years and any(
            normalize_text(claim) in normalize_text(text) for claim in _CLAIM_COMPLETE
        ) and not all(str(year) in text for year in missing_years):
            add_critical(
                "missing_required_year_claimed_complete",
                "Заявлена полнота периода при известном пропуске года",
            )

    # 4. Запрещённые значения и годы.
    forbidden_values = [
        value for value in checks.get("forbidden_values", []) or []
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    text_numbers = numbers_in_text(text)
    for forbidden in forbidden_values:
        if any(_within(value, float(forbidden), 0.001) for value in text_numbers):
            add_critical(
                "invented_numeric_value",
                f"В ответе встречается запрещённое значение {forbidden}",
            )
            break
    forbidden_years = [
        year for year in checks.get("forbidden_years", []) or []
        if isinstance(year, int) and not isinstance(year, bool)
    ]
    for year in forbidden_years:
        if year in fact_years:
            add_critical("fact_wrong_year", f"Запрещённый год {year} приписан факту")
            break
    if forbidden_years:
        for year in forbidden_years:
            position = text.find(str(year))
            if position < 0:
                continue
            window = normalize_text(text[max(0, position - 120):position + 120])
            metric_words = normalize_text(
                " ".join(
                    str(item.get("metric"))
                    for item in required_facts
                    if item.get("metric")
                )
            )
            if metric_words and any(
                word in window for word in metric_words.split() if len(word) > 3
            ):
                add_critical(
                    "fact_wrong_year",
                    f"Запрещённый год {year} стоит рядом с показателем",
                )
                break

    # 5. Обязательные и запрещённые упоминания.
    missing_mentions: list[str] = []
    for mention in checks.get("must_mention", []) or []:
        if not isinstance(mention, str):
            continue
        if mention.startswith("re:"):
            ok = re.search(mention[3:], text, flags=re.IGNORECASE) is not None
        else:
            ok = mention_present(text, mention)
        if not ok:
            missing_mentions.append(mention)
    add_check(
        "required_elements",
        not missing_mentions,
        detail=", ".join(missing_mentions),
    )
    forbidden_mentions = [
        mention
        for mention in checks.get("must_not_mention", []) or []
        if isinstance(mention, str) and normalize_text(mention) in normalize_text(text)
    ]
    if forbidden_mentions:
        add_critical(
            "forbidden_mention",
            "Ответ содержит запрещённую формулировку: "
            + ", ".join(forbidden_mentions),
        )

    # 6. Citations.
    manifest = set(str(item) for item in turn.get("evidence_manifest", []) or [])
    context_ids = turn.get("context_chunk_ids")
    context_source = turn.get("context_source")
    context_measured = context_source != "evidence_manifest" and isinstance(context_ids, list)
    valid_citations = 0
    quote_mismatches: list[str] = []
    quotes_checked = 0
    strict_quotes = bool(checks.get("strict_quotes"))
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        citation_ok = True
        if not isinstance(chunk_id, str) or not chunk_id:
            citation_ok = False
            warnings.append("citation_without_chunk_id")
        elif chunk_id not in manifest:
            citation_ok = False
            add_critical(
                "invalid_citation",
                f"citation {chunk_id} отсутствует в evidence_manifest",
            )
        elif context_measured and chunk_id not in set(context_ids):
            citation_ok = False
            add_critical(
                "invalid_citation",
                f"citation {chunk_id} отсутствует в context_chunk_ids",
            )
        if citation_ok:
            valid_citations += 1
        quote = citation.get("quote")
        if (
            citation_ok
            and chunk_store is not None
            and isinstance(chunk_id, str)
            and isinstance(quote, str)
            and quote.strip()
        ):
            chunk = chunk_store.get(chunk_id)
            if chunk is not None:
                quote_ok = normalize_text(quote) in normalize_text(chunk.get("text", ""))
                quotes_checked += 1
                if not quote_ok:
                    quote_mismatches.append(chunk_id)
                    # strict_quotes=true превращает расхождение цитаты в
                    # critical error: цитата — проверяемое доказательство, а не
                    # косметика. При strict_quotes=false остаётся warning,
                    # потому что evaluator не должен скрывать проблему, но и не
                    # обязан валить тест из-за сокращённой формулировки.
                    if strict_quotes:
                        add_critical(
                            "citation_quote_mismatch",
                            f"quote не является фрагментом {chunk_id}",
                        )
                    else:
                        warnings.append("citation_quote_mismatch")
    if quotes_checked:
        add_check(
            "citation_quotes",
            not quote_mismatches,
            mandatory=strict_quotes,
            detail=(
                "OK"
                if not quote_mismatches
                else f"{len(quote_mismatches)}/{quotes_checked}: "
                + ", ".join(quote_mismatches[:5])
            ),
        )
    citation_validity = (
        valid_citations / len(citations) if citations else None
    )
    if expected_status == "answer" and not citations:
        add_critical("invalid_citation", "Ответ без citations")
    if expected_status == "abstain" and citations:
        add_critical("wrong_status", "Abstain-ответ содержит citations")
    citation_check_passed = (
        not citations
        if expected_status == "abstain"
        else bool(citations) and valid_citations == len(citations)
    )
    add_check(
        "citation_validity",
        citation_check_passed,
        mandatory=expected_status == "answer",
        detail=f"{valid_citations}/{len(citations)}",
    )

    if required_facts:
        supported = sum(
            1 for detail in fact_details if detail.get("matched") and detail.get("source_ok")
        )
        citation_coverage = supported / len(required_facts)
    else:
        citation_coverage = None

    # 7. Расчёты.
    required_calculations = [
        item for item in expected.get("required_calculations", []) or []
        if isinstance(item, Mapping)
    ]
    calculations = [
        item for item in answer.get("calculations", []) or []
        if isinstance(item, Mapping)
    ]
    if required_calculations:
        matched_calculations = sum(
            1 for required in required_calculations if _calc_match(required, calculations)
        )
        calculation_accuracy = matched_calculations / len(required_calculations)
        if matched_calculations != len(required_calculations):
            if not calculations:
                add_critical("calculation_missing", "Обязательный расчёт отсутствует")
            else:
                add_critical("calculation_mismatch", "Расчёт не совпал с эталоном")
        add_check(
            "calculation_accuracy",
            matched_calculations == len(required_calculations),
            detail=f"{matched_calculations}/{len(required_calculations)}",
        )
    else:
        calculation_accuracy = None

    # 8. Abstention.
    warnings_union = _warning_union(turn)
    expected_warnings = [
        str(item) for item in checks.get("expected_warnings", []) or []
    ]
    reason_codes = [
        str(item) for item in expected.get("reason_codes", []) or []
    ]
    if expected_status == "abstain":
        declared_reasons = set(expected_warnings + reason_codes)
        reason_ok = (
            bool(warnings_union & declared_reasons)
            if declared_reasons
            else status == "abstain"
        )
        abstention_accuracy = 1.0 if status_ok and reason_ok else 0.0
        if not reason_ok:
            add_critical(
                "abstain_reason_missing",
                "Abstain без ожидаемой причины",
            )
        add_check("abstention_accuracy", abstention_accuracy == 1.0)
    else:
        abstention_accuracy = None

    if expected_warnings:
        missing_warnings = sorted(set(expected_warnings) - warnings_union)
        if missing_warnings:
            warnings.append("expected_warning_missing")
            add_check(
                "expected_warnings",
                False,
                detail=", ".join(missing_warnings),
            )
        else:
            add_check("expected_warnings", True)

    # 9. Наблюдаемое поведение инструментов.
    tools = _tool_names(turn)
    forbidden_tools = [
        tool for tool in behavior.get("must_not_use_tools", []) or [] if tool in tools
    ]
    if forbidden_tools:
        add_critical(
            "forbidden_tool",
            "Вызваны запрещённые инструменты: " + ", ".join(forbidden_tools),
        )
    required_tools = list(behavior.get("must_use_tools", []) or [])
    missing_tools = [tool for tool in required_tools if tool not in tools]
    strict_tools = bool(behavior.get("must_use_tools_strict"))
    if missing_tools and strict_tools:
        add_critical(
            "required_tool_missing",
            "Не вызваны обязательные инструменты: " + ", ".join(missing_tools),
        )
    elif missing_tools:
        warnings.append("alternative_tool_path")
    search_calls = sum(1 for name in tools if name == "search_reports")
    max_search_calls = behavior.get("max_search_calls")
    if isinstance(max_search_calls, int) and search_calls > max_search_calls:
        if behavior.get("strict_search_budget"):
            add_critical(
                "excess_search_calls",
                f"{search_calls} search_reports при лимите {max_search_calls}",
            )
        else:
            warnings.append("excess_search_calls")
    max_tool_calls = behavior.get("max_tool_calls")
    if isinstance(max_tool_calls, int) and len(tools) > max_tool_calls:
        if behavior.get("strict_search_budget"):
            add_critical(
                "excess_tool_calls",
                f"{len(tools)} tool calls при лимите {max_tool_calls}",
            )
        else:
            warnings.append("excess_tool_calls")
    add_check("must_not_use_tools", not forbidden_tools)
    add_check(
        "must_use_tools",
        not missing_tools,
        mandatory=strict_tools,
    )

    # 10. Structural changes — только если capability реально наблюдаема.
    # Пустой список считается доказательством отсутствия изменений только при
    # доступном инструменте структуры. При явной недоступности это
    # not_measured, а не провал. Если capability доступна и эталон требует
    # изменение, его отсутствие — critical error: иначе тест проходит без
    # структурного результата.
    required_structural = [
        item for item in expected.get("required_structural_changes", []) or []
        if isinstance(item, Mapping)
    ]
    if required_structural:
        observed = turn.get("structural_changes")
        structure_unavailable = "structure_unavailable" in warnings_union
        scope_limited = "structure_scope_limited" in warnings_union
        if observed is None or (
            not observed and (structure_unavailable or scope_limited)
        ):
            not_measured.append("structural_changes")
        else:
            observed_rows = (
                [item for item in observed if isinstance(item, Mapping)]
                if isinstance(observed, list)
                else []
            )
            missing_required = [
                required
                for required in required_structural
                if not any(
                    _structural_match(required, item) for item in observed_rows
                )
            ]
            if missing_required and scope_limited:
                # Инструмент покрыл не все годы: отсутствие изменения в
                # доступной части не доказывает его отсутствие целиком.
                not_measured.append("structural_changes")
            else:
                if missing_required:
                    descriptors = [
                        "/".join(
                            str(required.get(key))
                            for key in ("type", "section", "from_doc", "to_doc")
                            if required.get(key)
                        )
                        for required in missing_required
                    ]
                    add_critical(
                        "structural_changes_missing",
                        "Обязательное структурное изменение не найдено: "
                        + "; ".join(descriptors),
                    )
                add_check(
                    "structural_changes",
                    not missing_required,
                    detail=(
                        f"наблюдалось изменений: {len(observed_rows)}"
                        if observed_rows
                        else "structural_changes пуст"
                    ),
                )

    # 11. Числовые утверждения без источника.
    if checks.get("must_not_invent_values"):
        citation_numbers = set()
        for citation in citations:
            quote = citation.get("quote")
            if isinstance(quote, str):
                citation_numbers.update(numbers_in_text(quote))
            # Короткая цитата — выдержка, а не весь подтверждающий текст:
            # число может стоять в другой части того же cited-чанка.
            chunk_id = citation.get("chunk_id")
            if chunk_store is not None and isinstance(chunk_id, str):
                chunk = chunk_store.get(chunk_id)
                if isinstance(chunk, Mapping):
                    citation_numbers.update(
                        numbers_in_text(str(chunk.get("text") or ""))
                    )
        for value in text_numbers:
            if value in fact_years:
                continue
            if any(str(year) == str(int(value)) for year in fact_years if value.is_integer()):
                continue
            if citation_numbers and not any(
                _within(value, expected, 0.01) for expected in citation_numbers
            ) and not required_facts and not required_calculations:
                warnings.append("numeric_claim_unverified")
                break

    automatic_checks = [
        row for row in checks_rows if row["mandatory"]
    ]
    passed = not critical and all(row["pass"] for row in automatic_checks)
    if evidence_coverage is not None:
        min_coverage = (spec.get("gold_evidence") or {}).get("min_coverage", 1.0)
        if evidence_coverage < float(min_coverage or 0):
            passed = False
            warnings.append("evidence_coverage_below_min")

    return {
        "pass": passed,
        "critical_errors": critical,
        "warnings": list(dict.fromkeys(warnings)),
        "metrics": {
            "status_accuracy": 1.0 if status_ok else 0.0,
            "fact_accuracy": fact_accuracy,
            "citation_validity": citation_validity,
            "citation_coverage": citation_coverage,
            "calculation_accuracy": calculation_accuracy,
            "abstention_accuracy": abstention_accuracy,
            "required_elements": (
                1.0 if not missing_mentions else 0.0
            ) if checks.get("must_mention") else None,
        },
        "checks": checks_rows,
        "facts": fact_details,
        "not_measured": list(dict.fromkeys(not_measured)),
    }


def evaluate_answer(
    test: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    chunk_store: Mapping[str, Mapping[str, Any]] | None = None,
    retrieval_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Оценить одиночный тест или все turn'ы multi-turn сценария."""
    turns = trace.get("turns")
    if not isinstance(turns, list) or not turns:
        turns = [trace]
    test_turns = test.get("turns") if isinstance(test.get("turns"), list) else None
    if not test_turns:
        result = _evaluate_turn(
            test,
            turns[0],
            chunk_store=chunk_store,
            evidence_coverage=(retrieval_metrics or {}).get("evidence_coverage"),
        )
        result["turn_index"] = 0
        result["turns"] = [result]
        return result

    results: list[dict[str, Any]] = []
    by_index = {
        int(turn.get("turn_index", index)): turn
        for index, turn in enumerate(turns)
    }
    for index, test_turn in enumerate(test_turns):
        if not isinstance(test_turn, Mapping):
            continue
        trace_turn = by_index.get(index)
        if trace_turn is None:
            results.append(
                {
                    "pass": False,
                    "critical_errors": [
                        {"code": "turn_not_executed", "message": f"turn {index} не выполнен"}
                    ],
                    "warnings": [],
                    "metrics": {},
                    "checks": [],
                    "facts": [],
                    "not_measured": ["turn"],
                    "turn_index": index,
                }
            )
            continue
        result = _evaluate_turn(
            test_turn,
            trace_turn,
            chunk_store=chunk_store,
            evidence_coverage=None,
        )
        result["turn_index"] = index
        results.append(result)

    critical = [
        {**error, "turn_index": row["turn_index"]}
        for row in results
        for error in row.get("critical_errors", [])
    ]
    warnings = list(
        dict.fromkeys(
            warning for row in results for warning in row.get("warnings", [])
        )
    )
    metrics = {
        key: aggregate_mean([row.get("metrics", {}).get(key) for row in results])
        for key in (
            "status_accuracy",
            "fact_accuracy",
            "citation_validity",
            "citation_coverage",
            "calculation_accuracy",
            "abstention_accuracy",
            "required_elements",
        )
    }
    return {
        "pass": all(row.get("pass") for row in results),
        "critical_errors": critical,
        "warnings": warnings,
        "metrics": metrics,
        "checks": [row for item in results for row in item.get("checks", [])],
        "facts": [row for item in results for row in item.get("facts", [])],
        "not_measured": sorted(
            {value for row in results for value in row.get("not_measured", [])}
        ),
        "turns": results,
    }
