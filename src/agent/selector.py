"""Селектор действий: несколько осмысленных поисков вместо одного top-k.

Селектор не вызывает инструменты напрямую. Он только предлагает следующее
действие, а существующий StateGraph исполняет его через ``ToolNode``.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

from src.core.record import chunk_text as _chunk_text
from src.core.record import chunk_year as _chunk_year

from src.agent.coverage import (
    CoverageCell,
    CoverageReport,
    build_coverage,
    extract_metrics,
    metric_label,
    normalize_metric,
)
from src.agent.state import AgentAction, AgentState, QueryPlan
from src.retrieval.hybrid_retriever import SearchFilters

DEFAULT_MAX_SEARCHES = 2
DEFAULT_MAX_STRUCTURE_DOCS = 5
#: Для сравнения без числовых метрик каждый год ищется отдельно и с запасом
#: top-k: иначе RRF/разнообразие внутри года вытесняют нужный раздел.
COMPARISON_TOP_K = 10

#: Роутер может вернуть аспект каноническим ключом (``mortgage_portfolio``).
#: Такой запрос бесполезен для русскоязычного индекса, поэтому ключи
#: переводятся в формулировки отчёта. Неизвестные ключи не подменяются:
#: лучше исходный вопрос, чем выдуманный термин.
FINANCIAL_RESULTS_QUERY = (
    "ключевые финансовые результаты чистая прибыль "
    "рентабельность капитала достаточность капитала CET1"
)
STRATEGY_PRIORITIES_QUERY = (
    "Стратегия 2026 приоритеты человекоцентричность "
    "искусственный интеллект цифровая трансформация"
)
CREDIT_RISK_QUERY = "управление кредитным риском стоимость риска COR"
DIGITAL_LENDING_QUERY = (
    "цифровое кредитование цифровой кредитный процесс доля онлайн-выдач"
)

ASPECT_LABELS = {
    "net_profit": "чистая прибыль",
    "revenue": "выручка",
    "operating_income": "операционная прибыль",
    "roe": "рентабельность капитала",
    "roa": "рентабельность активов",
    "assets": "активы",
    "capital": "капитал",
    "margin": "маржа",
    "market_share": "доля на рынке",
    "mortgage_portfolio": "ипотечный портфель",
    "mortgage_market_share": "доля на рынке ипотечного кредитования",
    "credit_portfolio": "кредитный портфель",
    "corporate_credit_portfolio": "корпоративный кредитный портфель",
    "retail_credit_portfolio": "розничный кредитный портфель",
    "corporate_loan_portfolio": "корпоративный кредитный портфель",
    "retail_loan_portfolio": "розничный кредитный портфель",
    "corporate_loan_growth": "рост корпоративного кредитного портфеля",
    "retail_loan_growth": "рост розничного кредитного портфеля",
    "credit_risk_management": CREDIT_RISK_QUERY,
    "cost_of_risk": "стоимость риска COR",
    "digital_lending": DIGITAL_LENDING_QUERY,
    "digital_transformation": "цифровая трансформация",
    "cet1": "достаточность базового капитала CET1",
    # «Финансовые результаты» — не одна метрика, а кластер KPI отчёта.
    # Один запрос только по этой фразе попадает в оглавление; перечисление
    # самих показателей возвращает страницы с числами.
    "financial_results": FINANCIAL_RESULTS_QUERY,
    "strategy_priorities": STRATEGY_PRIORITIES_QUERY,
    "risk_management_section": "управление рисками климатические риски ESG",
    "strategy_priorities_2026": STRATEGY_PRIORITIES_QUERY,
}


_SECTION_FILTER_STOPWORDS = (
    "раздел",
    "разделе",
    "раздела",
    "глава",
    "главе",
    "часть",
    "части",
    "обзор",
    "описание",
)


def plan_section_path(value: Any) -> str | None:
    """Правдоподобный путь раздела из плана роутера.

    Роутер иногда возвращает не путь таксономии, а описание темы
    («раздел об управлении рисками»). Такой текст нельзя применять как
    ``section_path_prefix``: он не совпадает ни с одним путём корпуса и
    запускает section-fallback, который вытесняет нужные разделы. Свободное
    описание остаётся лишь смысловой подсказкой вопроса.
    """
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" /")
    if not text:
        return None
    if "/" in text:
        return text
    words = text.split()
    if len(words) > 3 or len(text) > 60:
        return None
    first = words[0].casefold()
    if first.startswith(_SECTION_FILTER_STOPWORDS):
        return None
    return text


def _aspect_label(value: str) -> str | None:
    """Поисковая формулировка аспекта или ``None``, если её нет."""
    key = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    label = ASPECT_LABELS.get(key)
    if label:
        return label
    # Роутер часто приписывает к каноническому ключу год: ``net_profit_2022``.
    # Год уже учитывается фильтром/целевым годом, поэтому ключ нормализуется.
    stripped = re.sub(r"_(?:18|19|20|21)\d{2}$", "", key)
    if stripped != key:
        label = ASPECT_LABELS.get(stripped)
        if label:
            return label
    # Русская формулировка уже пригодна для индекса.
    if re.search(r"[а-яё]", value, re.I) and len(value.strip()) > 3:
        return _expand_aspect_query(value.strip())
    return None


def _expand_aspect_query(label: str) -> str:
    """Смысловое расширение русской формулировки аспекта.

    Роутер отвечает то каноническими ключами, то свободным текстом
    («финансовые показатели 2023», «приоритеты Стратегии 2026»). Тематический
    кластер отчёта служит лучшим адресным запросом, чем общая фраза, которая
    встречается в оглавлении.
    """
    text = label.casefold()
    if "финансов" in text and (
        "показател" in text or "результат" in text
    ):
        return FINANCIAL_RESULTS_QUERY
    if "стратег" in text and ("приоритет" in text or "2026" in text):
        return STRATEGY_PRIORITIES_QUERY
    if "риск" in text and (
        "кредитн" in text or "управлен" in text or "стоимост" in text
    ):
        return CREDIT_RISK_QUERY
    if "цифров" in text and (
        "кредитован" in text or "выдач" in text or "онлайн" in text
    ):
        return DIGITAL_LENDING_QUERY
    return label


_WORD = re.compile(r"[0-9a-zа-я]+")


def _label_words(label: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(label.casefold().replace("ё", "е"))
        if len(word) > 2
    ]


#: Слова, которые не описывают тему: аспект «отчёт за 2023 год» не поможет
#: адресному поиску и не должен вытеснять содержательный аспект.
_GENERIC_ASPECT_WORDS = frozenset(
    {
        "отчет", "отчёт", "год", "году", "года", "годы", "годов",
        "период", "периода", "данные", "данным", "значение", "значения",
        "отчете", "отчёте", "отчетность", "отчётность",
    }
)

#: Названия эмитента/документа не являются показателем: запрос «Группа
#: Сбербанк 2022 год» не уточняет тему и вытесняет содержательный аспект.
_ENTITY_NAME_WORDS = frozenset(
    {
        "группа", "группы", "группе", "сбербанк", "сбербанка", "сбер",
        "банк", "банка", "банке", "пао", "компания", "компании", "отчет",
        "отчёт", "год", "году", "года",
    }
)


def _is_generic_label(label: str) -> bool:
    words = _label_words(label)
    if not words:
        return True
    if all(
        word in _GENERIC_ASPECT_WORDS or word.isdigit() for word in words
    ):
        return True
    if all(word in _ENTITY_NAME_WORDS or word.isdigit() for word in words):
        return True
    return False


def _label_coverage(
    label: str, evidence: Mapping[str, Any], *, year: int | None = None
) -> float:
    """Доля значимых слов аспекта, найденных в содержательном evidence."""
    words = _label_words(label)
    if not words:
        return 1.0
    tokens: set[str] = set()
    for chunk in (evidence or {}).values():
        if year is not None and _chunk_year(chunk) != year:
            continue
        if _is_navigation_chunk(chunk):
            continue
        tokens.update(
            token
            for token in _WORD.findall(_chunk_text(chunk).casefold().replace("ё", "е"))
            if len(token) > 2
        )
    matched = sum(
        1 for word in words if any(_same_word(word, token) for token in tokens)
    )
    return matched / len(words)


def _same_word(left: str, right: str) -> bool:
    """Словоформы одного термина: ``риски``/``рисками``, ``портфель``/``портфеля``."""
    if left == right:
        return True
    prefix = 0
    for first, second in zip(left, right):
        if first != second:
            break
        prefix += 1
    return prefix >= 4 and prefix >= min(len(left), len(right)) - 3


def _phrase_in_text(label: str, text: str) -> bool:
    """Аспект встречается в тексте как связная фраза, а не набор слов."""
    words = _label_words(label)
    if not words:
        return True
    tokens = [
        word
        for word in _WORD.findall(text.casefold().replace("ё", "е"))
        if len(word) > 2
    ]
    width = len(words)
    for start in range(max(len(tokens) - width + 1, 0)):
        if all(
            _same_word(words[offset], tokens[start + offset])
            for offset in range(width)
        ):
            return True
    return False


def _label_in_evidence(
    label: str, evidence: Mapping[str, Any], *, year: int | None = None
) -> bool:
    """Аспект подтверждён содержательным чанком (не оглавлением).

    В корпусе есть таблицы-оглавления, где перечислены все названия разделов.
    Формальное совпадение фразы в таком чанке не означает, что содержание
    раздела найдено: поиск по аспекту после этого не выполнялся бы.
    """
    for chunk in (evidence or {}).values():
        if year is not None and _chunk_year(chunk) != year:
            continue
        if _is_navigation_chunk(chunk):
            continue
        if _phrase_in_text(label, _chunk_text(chunk)):
            return True
    return False


_NAVIGATION_HINT = re.compile(
    r"(?i)(?:отчет по рискам\s+\d+|оглавление\b|"
    r"содержание\s+(?:отчета|годового\s+отчета))"
)
_NAVIGATION_NUMBER = re.compile(r"\b\d{2,4}\b")


def _is_navigation_chunk(chunk: Any) -> bool:
    """Чанк похож на оглавление: заголовок содержания и много номеров страниц."""
    text = _chunk_text(chunk)
    if not _NAVIGATION_HINT.search(text):
        return False
    return len(_NAVIGATION_NUMBER.findall(text)) >= 8


def _plan_aspect_labels(plan: QueryPlan) -> list[str]:
    """Уникальные поисковые формулировки аспектов/сущностей плана."""
    labels: list[str] = []
    seen: set[str] = set()
    aspects = getattr(plan, "aspects", ()) or ()
    entities = getattr(plan, "entities", ()) or ()
    for value in [*aspects, *entities]:
        label = _aspect_label(value)
        if not label or _is_generic_label(label):
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def aspect_search_candidates(
    state: AgentState, plan: QueryPlan
) -> list[tuple[str, int | None]]:
    """Аспекты запроса, которых ещё нет в evidence, с целевым годом.

    Адресный поиск по формулировке аспекта — не замена широкому поиску, а
    добивка: длинный вопрос размывает эмбеддинг, и нужный раздел проигрывает
    RRF/разнообразию. Один поиск на аспект, не больше двух за turn.
    """
    if state.get("search_calls_used", 0) >= DEFAULT_MAX_SEARCHES:
        return []
    labels = _plan_aspect_labels(plan)
    if not labels:
        return []
    if not (
        getattr(plan, "requires_comparison", False)
        or len(labels) >= 2
        or getattr(plan, "required_metrics", ())
    ):
        return []
    attempts = list(state.get("search_attempts", []))
    if not any(item.get("kind") == "broad" for item in attempts):
        return []
    attempted = _attempted_metric_years(state)
    evidence = state.get("evidence") or {}
    years = _target_years(state, plan)
    candidates: list[tuple[str, int | None]] = []
    scored: list[tuple[float, int, str, int | None]] = []
    for order, label in enumerate(labels[:3]):
        # Год, где тема уже подтверждена содержательным чанком, повторно не
        # ищем; для сравнительного вопроса цель — год, где аспекта ещё нет.
        if years:
            missing_years = [
                year
                for year in years
                if not _label_in_evidence(label, evidence, year=year)
            ]
        else:
            missing_years = [None] if not _label_in_evidence(label, evidence) else []
        for year in missing_years:
            if (normalize_metric(label) or label, year) in attempted:
                continue
            scored.append(
                (_label_coverage(label, evidence), order, label, year)
            )
            break
    # Первым ищем аспект, который покрыт слабее всего: иначе повторный поиск
    # уже частично найденной темы тратит бюджет, а вторая тема вопроса
    # остаётся без адресного запроса.
    scored.sort(key=lambda item: (item[0], item[1]))
    for _score, _order, label, year in scored[:2]:
        candidates.append((label, year))
    return candidates



_YEAR_TOKEN = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_PERIOD_ONLY_WORDS = frozenset(
    {
        "за", "в", "во", "на", "по", "год", "году", "года", "годы", "годов",
        "лет", "период", "периода", "диапазон", "покажи", "показать",
        "уточни", "уточните", "пожалуйста", "данные", "этот",
    }
)


def _is_period_fragment(question: str) -> bool:
    """Короткий follow-up, состоящий почти только из периода («За 2023 год»)."""
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", question or "")
    if not tokens:
        return False
    content = [
        token
        for token in tokens
        if not _YEAR_TOKEN.fullmatch(token)
        and token.casefold() not in _PERIOD_ONLY_WORDS
        and not token.isdigit()
    ]
    return len(content) <= 1 and bool(_YEAR_TOKEN.search(question))


def requested_years(value: Any) -> set[int]:
    """Годы из ``SearchFilters`` или их словарного представления."""
    if isinstance(value, SearchFilters):
        if value.years:
            return set(value.years)
        start = value.year_from
        end = value.year_to
    elif isinstance(value, Mapping):
        years = value.get("years")
        if isinstance(years, (list, tuple, set)):
            return {
                int(year)
                for year in years
                if isinstance(year, int) and not isinstance(year, bool)
            }
        start = value.get("year_from")
        end = value.get("year_to")
        if not isinstance(start, int) or isinstance(start, bool):
            start = None
        if not isinstance(end, int) or isinstance(end, bool):
            end = None
    else:
        return set()
    if start is None and end is None:
        return set()
    start = start if start is not None else end
    end = end if end is not None else start
    assert start is not None and end is not None
    return set(range(min(start, end), max(start, end) + 1))


def _target_years(state: AgentState, plan: QueryPlan) -> list[int]:
    years = set(state.get("expected_years", []))
    if not years:
        years = set(plan.years)
    if not years and plan.year_from is not None and plan.year_to is not None:
        years = set(range(plan.year_from, plan.year_to + 1))
    if not years and plan.intent in {"structural", "temporal", "multi_hop"}:
        years.update(state.get("available_documents", {}).keys())
    if not years:
        years.update(
            requested_years(
                state.get("effective_filters") or SearchFilters()
            )
        )
    return sorted(years)


def _attempted_years(state: AgentState) -> set[int]:
    years: set[int] = set()
    for attempt in state.get("search_attempts", []):
        if not isinstance(attempt, dict):
            continue
        year = attempt.get("year")
        if isinstance(year, int) and not isinstance(year, bool):
            years.add(year)
    return years


def _attempted_metric_years(state: AgentState) -> set[tuple[str, int]]:
    """Пары ``(metric, year)``, по которым адресный поиск уже выполнялся."""
    pairs: set[tuple[str, int]] = set()
    for attempt in state.get("search_attempts", []):
        if not isinstance(attempt, dict):
            continue
        metric = attempt.get("metric")
        year = attempt.get("year")
        if not isinstance(metric, str) or not metric:
            continue
        if not isinstance(year, int) or isinstance(year, bool):
            continue
        pairs.add((normalize_metric(metric) or metric, year))
    return pairs


def _report_from_mapping(value: dict[str, Any]) -> CoverageReport | None:
    """Восстанавливает отчёт из ``to_dict()`` (например, после сериализации)."""
    try:
        cells = [
            CoverageCell(
                metric=str(cell["metric"]),
                year=cell.get("year"),
                status=cell["status"],
                chunk_ids=list(cell.get("chunk_ids") or []),
            )
            for cell in value.get("cells", [])
            if isinstance(cell, dict)
        ]
        return CoverageReport(
            status=value["status"],
            expected_years=list(value.get("expected_years", [])),
            covered_years=list(value.get("covered_years", [])),
            missing_years=list(value.get("missing_years", [])),
            cells=cells,
            required_metrics=list(value.get("required_metrics", [])),
            metrics={
                str(metric): {
                    str(key): str(status)
                    for key, status in statuses.items()
                }
                for metric, statuses in (value.get("metrics") or {}).items()
                if isinstance(statuses, dict)
            },
            missing_metrics=list(value.get("missing_metrics", [])),
            missing_metric_years=[
                dict(item)
                for item in value.get("missing_metric_years", [])
                if isinstance(item, dict)
            ],
            distinct_docs=int(value.get("distinct_docs", 0) or 0),
            distinct_sections=int(value.get("distinct_sections", 0) or 0),
            notes=[str(item) for item in value.get("notes", [])],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _coverage_for_state(
    state: AgentState, plan: QueryPlan
) -> CoverageReport | None:
    """Использует готовый coverage, иначе строит его из собранных evidence."""
    value = state.get("coverage")
    if isinstance(value, CoverageReport):
        return value
    if isinstance(value, dict):
        report = _report_from_mapping(value)
        if report is not None:
            return report
    evidence = state.get("evidence") or {}
    if not evidence:
        # Без evidence матрица была бы полностью not_found и не давала бы
        # полезной информации; сохраняем прежний year-level fallback.
        return None
    metrics = list(plan.required_metrics) or extract_metrics(
        state.get("question", "")
    )
    available = set(state.get("available_documents", {}).keys())
    return build_coverage(
        _target_years(state, plan),
        metrics,
        evidence,
        available or None,
    )


def _effective_question(state: AgentState) -> str:
    """Поисковый вопрос, в котором короткий follow-up получает тему из истории.

    «За 2023 год.» — период без предмета запроса. Такой фрагмент нельзя
    отправлять в индекс как есть: поиск вернёт произвольные чанки 2023 года.
    Тема берётся из последнего пользовательского вопроса в истории диалога;
    сама формулировка пользователя при этом не переписывается в ответе.
    """
    question = " ".join(str(state.get("question") or "").split())
    if not _is_period_fragment(question):
        return question
    for item in reversed(list(state.get("history") or [])):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        previous = " ".join(str(item.get("content") or "").split())
        if previous and not _is_period_fragment(previous):
            return f"{previous} {question}".strip()
        break
    return question


def _broad_queries(
    state: AgentState, plan: QueryPlan, *, max_queries: int = 3
) -> list[str]:
    """Q1: original + aspects для сложного запроса, иначе rewritten/[question]."""
    question = _effective_question(state)
    aspects = [
        item.strip() for item in plan.aspects if item.strip()
    ]
    complex_query = (
        plan.needs_rewriting
        or plan.requires_comparison
        or len(plan.required_metrics) >= 2
    )
    if aspects and complex_query:
        candidates = [question]
        # Канонические ключи роутера («mortgage_portfolio») переводятся в
        # русские формулировки: иначе дополнительный запрос не находит ничего.
        candidates.extend(
            label
            for label in (_aspect_label(aspect) for aspect in aspects)
            if label
        )
        queries: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            queries.append(candidate)
            if len(queries) >= max_queries:
                break
        return queries
    rewritten = [
        item
        for item in state.get("rewritten_queries", [])
        if isinstance(item, str) and item.strip()
    ]
    return rewritten or ([question] if question else [])


def _query_for_metric_year(
    question: str, metric: str, year: int
) -> str:
    parts = [question]
    label = metric_label(metric)
    if label.casefold() not in question.casefold():
        parts.append(label)
    if str(year) not in question:
        parts.append(f"{year} год")
    return " ".join(part for part in parts if part)


def _query_for_year(
    question: str,
    year: int,
    metrics: list[str] | None = None,
) -> str:
    parts = [question]
    if str(year) not in question:
        parts.append(f"{year} год")
    for metric in metrics or []:
        label = metric_label(metric)
        if label.casefold() not in question.casefold():
            parts.append(label)
    return " ".join(part for part in parts if part)


def content_attempt_key(item: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
    """Стабильный ключ попытки адресного поиска по разделу."""
    years = item.get("years")
    if isinstance(years, (list, tuple, set)):
        normalized = tuple(
            sorted(
                int(year)
                for year in years
                if isinstance(year, int) and not isinstance(year, bool)
            )
        )
    else:
        year = item.get("year")
        normalized = (
            (year,)
            if isinstance(year, int) and not isinstance(year, bool)
            else ()
        )
    return normalized, item.get("section_path")


#: Если в evidence меньше этого числа чанков раздела, раздел дочитывается
#: адресным поиском по ``section_path``. Один-два чанка — это заголовок и
#: начало раздела, а не его содержание.
SECTION_COMPLETION_MIN_CHUNKS = 3


def _section_content_target(
    state: AgentState,
    checked: set[tuple[tuple[int, ...], Any]],
    max_searches: int,
) -> tuple[int, str] | None:
    """Раздел, который стоит дочитать по section_path (поздний год первым)."""
    if state.get("search_calls_used", 0) >= max_searches:
        return None
    allowed_years = requested_years(
        state.get("effective_filters") or SearchFilters()
    )
    evidence = state.get("evidence") or {}
    groups: dict[tuple[int, str], set[str]] = {}
    for chunk_id, chunk in evidence.items():
        if not isinstance(chunk, Mapping):
            continue
        year = _chunk_year(chunk)
        section = chunk.get("section_path")
        if year is None or not isinstance(section, str) or not section.strip():
            continue
        if allowed_years and year not in allowed_years:
            # Фильтр года сузить нельзя: (2022, раздел-2023) не даст хитов.
            continue
        groups.setdefault((year, section.strip()), set()).add(str(chunk_id))
    for (year, section), chunk_ids in sorted(
        groups.items(), key=lambda item: (-item[0][0], item[0][1])
    ):
        if len(chunk_ids) >= SECTION_COMPLETION_MIN_CHUNKS:
            continue
        if ((year,), section) in checked:
            continue
        return year, section
    return None


class PlanAwareSelector:
    """Использует QueryPlan и уже собранные evidence для следующего шага."""

    def __init__(
        self,
        *,
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_structure_docs: int = DEFAULT_MAX_STRUCTURE_DOCS,
        max_queries: int = 3,
        coverage_enabled: bool = True,
        max_coverage_searches: int = 1,
    ) -> None:
        if min(max_searches, max_structure_docs) < 1:
            raise ValueError("selector limits must be positive")
        if max_queries < 1 or max_coverage_searches < 1:
            raise ValueError("selector query limits must be positive")
        self.max_searches = max_searches
        self.max_structure_docs = max_structure_docs
        self.max_queries = max_queries
        self.coverage_enabled = coverage_enabled
        self.max_coverage_searches = max_coverage_searches

    def choose(self, state: AgentState) -> AgentAction | None:
        """Выбрать следующий шаг по плану и собранным evidence."""
        plan = state.get("query_plan") or QueryPlan()
        if plan.intent == "structural":
            return self._choose_structural(state, plan)
        return self._choose_search_or_graph(state, plan)

    @staticmethod
    def _coverage_searches_used(attempts: list[dict[str, Any]]) -> int:
        """Сколько адресных gap-поисков уже сделано (broad/graph не считаются)."""
        return sum(
            1
            for item in attempts
            if isinstance(item, dict)
            and item.get("kind") in {"aspect", "metric_year", "year"}
        )

    def _search_action(
        self,
        query: str,
        filters: SearchFilters,
        reason_code: str,
        *,
        queries: list[str] | None = None,
        metric: str | None = None,
        use_graph: bool = False,
        top_k: int = 5,
        rerank_per_year: bool | None = None,
    ) -> AgentAction:
        arguments: dict[str, Any] = {
            "query": query,
            "filters": filters,
            "retrieval_mode": "hybrid",
            # Граф — надстройка над широким поиском; адресные поиски
            # (metric×year, конкретный год) остаются без него.
            "use_graph": use_graph,
            "top_k": top_k,
        }
        if queries:
            arguments["queries"] = queries
        if metric:
            arguments["metric"] = metric
        if rerank_per_year is not None:
            arguments["rerank_per_year"] = rerank_per_year
        return AgentAction("search_reports", arguments, reason_code)

    def _choose_search_or_graph(
        self, state: AgentState, plan: QueryPlan
    ) -> AgentAction | None:
        attempts = list(state.get("search_attempts", []))
        searches = state.get("search_calls_used", 0)
        if searches >= self.max_searches:
            return None

        # Сравнение годов без числовых метрик: каждый год ищется отдельным
        # вызовом с расширенным top-k. Веерный поиск с фиксированным
        # per_year_k забирает лишь первые 2 хита года, и нужный раздел,
        # вытесненный diversity/RRF, теряется до генерации ответа.
        target_years = _target_years(state, plan)
        comparison_without_metrics = (
            plan.intent != "structural"
            and plan.requires_comparison
            and not plan.required_metrics
            and len(target_years) > 1
        )
        if comparison_without_metrics:
            covered = set(state.get("covered_years", []))
            attempted = _attempted_years(state)
            effective = state.get("effective_filters") or SearchFilters()
            requested = requested_years(effective)
            for year in target_years:
                if year in covered or year in attempted:
                    continue
                if requested and year not in requested:
                    # Год из истории нельзя искать: жёсткий фильтр сузит
                    # выдачу обратно к запрошенным годам.
                    continue
                filters = replace(
                    effective,
                    years={year},
                    year_from=None,
                    year_to=None,
                )
                return self._search_action(
                    _query_for_year(
                        _effective_question(state), year, plan.required_metrics
                    ),
                    filters,
                    "fact_lookup",
                    top_k=COMPARISON_TOP_K,
                    # Per-year сравнение раньше шло без реранкера: RRF внутри
                    # года вытеснял нужный раздел. Флаг опционален и
                    # применяется только backend'ом, который его поддержит.
                    rerank_per_year=True,
                )

        # Comparison-запрос с метриками решается несколькими циклами: широкий
        # поиск, затем по одному адресному поиску на каждую пару
        # (метрика, год), которая ещё не покрыта. Бюджет циклов — по числу
        # лет, а не один gap-поиск.
        comparison_with_metrics = (
            plan.requires_comparison
            and bool(plan.required_metrics)
            and len(target_years) > 1
        )

        # Широкий гибридный поиск — основной первый шаг. Граф подключается
        # после него и только как надстройка/добивка пропусков: сущность из
        # плана (например, метрика `net_profit`) может отсутствовать в графе
        # и не должна обнулять весь поиск.
        if not any(item.get("kind") == "broad" for item in attempts):
            filters = state.get("effective_filters") or SearchFilters()
            return self._search_action(
                _effective_question(state),
                filters,
                "fact_lookup",
                queries=_broad_queries(
                    state, plan, max_queries=self.max_queries
                ),
                use_graph=plan.needs_graph,
            )

        if (
            plan.needs_graph
            and plan.entities
            and not state.get("graph_evidence_manifest")
            and not any(item.get("kind") == "graph" for item in attempts)
        ):
            return AgentAction(
                "get_entities",
                {"entity_names": plan.entities},
                "explore_entities",
            )

        # Адресная добивка по аспектам, которых нет в уже собранном evidence.
        # Coverage можно отключить; число gap-поисков ограничено отдельно от
        # общего бюджета (spec_agent_enhancement §4.8).
        if not self.coverage_enabled:
            return None
        coverage_budget = self.max_coverage_searches
        if comparison_with_metrics:
            coverage_budget = max(coverage_budget, len(target_years))
        if self._coverage_searches_used(attempts) >= coverage_budget:
            return None
        if not comparison_with_metrics:
            for label, year in aspect_search_candidates(state, plan):
                filters = state.get("effective_filters") or SearchFilters()
                if year is not None:
                    filters = replace(
                        filters,
                        years={year},
                        year_from=None,
                        year_to=None,
                    )
                query = label if year is None else f"{label} {year} год"
                return self._search_action(
                    query,
                    filters,
                    "fact_lookup",
                    metric=label,
                )

        coverage = _coverage_for_state(state, plan)

        # 1. Адресный поиск по конкретной недостающей ячейке (metric, year).
        if coverage is not None:
            attempted_pairs = _attempted_metric_years(state)
            metric_gap = next(
                (
                    cell
                    for cell in sorted(
                        coverage.missing_cells(),
                        key=lambda item: (
                            item.year if item.year is not None else 0,
                            item.metric,
                        ),
                    )
                    if cell.status == "not_found"
                    and cell.metric
                    and cell.year is not None
                    and (cell.metric, cell.year) not in attempted_pairs
                ),
                None,
            )
            if metric_gap is not None and searches < self.max_searches:
                year = metric_gap.year
                if year is None:  # pragma: no cover - отфильтровано выше
                    return None
                filters = replace(
                    state.get("effective_filters") or SearchFilters(),
                    years={year},
                    year_from=None,
                    year_to=None,
                )
                return self._search_action(
                    _query_for_metric_year(
                        state.get("question", ""),
                        metric_gap.metric,
                        year,
                    ),
                    filters,
                    "fact_lookup",
                    metric=metric_gap.metric,
                )

        # 2. Fallback прежнего уровня: искать только недостающие годы.
        expected = _target_years(state, plan)
        covered = set(state.get("covered_years", []))
        attempted = _attempted_years(state)
        available = set(state.get("available_documents", {}).keys())
        missing = [
            year
            for year in expected
            if year not in covered
            and year not in attempted
            and (not available or year in available)
            and not (
                coverage is not None and coverage.is_year_absent(year)
            )
        ]
        if missing and searches < self.max_searches:
            year = missing[0]
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={year},
                year_from=None,
                year_to=None,
            )
            return self._search_action(
                _query_for_year(
                    state.get("question", ""),
                    year,
                    plan.required_metrics,
                ),
                filters,
                "fact_lookup",
            )
        return None

    def _choose_structural(
        self, state: AgentState, plan: QueryPlan
    ) -> AgentAction | None:
        attempts = list(state.get("search_attempts", []))
        snapshots = state.get("structure_snapshots", {})
        documents = state.get("available_documents", {})
        years = _target_years(state, plan)[-self.max_structure_docs :]

        for year in years:
            if year in snapshots:
                continue
            doc_ids = list(documents.get(year) or [])
            allowed = (
                state.get("effective_filters") or SearchFilters()
            ).allowed_doc_ids
            if allowed:
                narrowed = [doc_id for doc_id in doc_ids if doc_id in allowed]
                doc_ids = narrowed or sorted(allowed)
            if not doc_ids:
                doc_ids = sorted(allowed or [])
            if not doc_ids:
                continue
            if any(
                item.get("kind") == "structure"
                and item.get("year") == year
                for item in attempts
            ):
                continue
            return AgentAction(
                "get_structure_tree",
                {"doc_id": doc_ids[0], "max_depth": 4},
                "clarify_structure",
            )

        if not state.get("structure_comparison_complete"):
            # Сравнение выполняет assess_evidence, когда все доступные
            # структуры уже собраны.
            return None

        changes = [
            change
            for change in state.get("structural_changes", [])
            if isinstance(change, dict)
            and change.get("change_type") != "stable"
        ]
        stable_matches = [
            match
            for match in state.get("section_matches", [])
            if isinstance(match, dict)
            and match.get("change_type") == "stable"
        ]
        checked = {
            content_attempt_key(item)
            for item in attempts
            if item.get("kind") == "structure_content"
        }

        if not changes and not stable_matches:
            if not any(
                item.get("kind") == "structure_content"
                for item in attempts
            ):
                filters = state.get("effective_filters") or SearchFilters()
                section = plan_section_path(plan.section_filter)
                if section:
                    filters = replace(
                        filters, section_path_prefix=section
                    )
                return self._search_action(
                    state.get("question", ""),
                    filters,
                    "inspect_evidence",
                )
            # Первый content-поиск по нескольким годам отдаёт всего
            # ``per_year_k`` хитов на год. Для сравнения содержания раздела
            # этого мало: продолжение раздела (например, климатические риски)
            # остаётся за пределами top-k. Дочитываем раздел адресно по его
            # section_path, начиная с более позднего года.
            section_target = _section_content_target(state, checked, self.max_searches)
            if section_target is not None:
                year, section_path = section_target
                filters = replace(
                    state.get("effective_filters") or SearchFilters(),
                    years={year},
                    year_from=None,
                    year_to=None,
                    section_path_prefix=section_path,
                )
                return self._search_action(
                    state.get("question", ""),
                    filters,
                    "inspect_evidence",
                )
            return None

        searches = state.get("search_calls_used", 0)
        for change in changes:
            if searches >= self.max_searches:
                break
            if change.get("change_type") == "removed":
                target_year = change.get("from_year")
                section = change.get("from_section")
            else:
                target_year = change.get("to_year") or change.get("from_year")
                section = change.get("to_section") or change.get("from_section")
            if not isinstance(target_year, int):
                continue
            if not isinstance(section, dict):
                continue
            section_path = section.get("section_path")
            title = section.get("title")
            keys = {((target_year,), section_path)}
            if change.get("detail") == "content_changed":
                years = {
                    year
                    for year in (
                        change.get("from_year"),
                        change.get("to_year"),
                    )
                    if isinstance(year, int)
                }
                if years:
                    keys.add((tuple(sorted(years)), section_path))
            if keys & checked:
                continue
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={target_year},
                year_from=None,
                year_to=None,
            )
            if isinstance(section_path, str) and section_path:
                filters = replace(filters, section_path_prefix=section_path)
            query = (
                f"{title} {state.get('question', '')}"
                if isinstance(title, str) and title
                else state.get("question", "")
            )
            return self._search_action(
                query,
                filters,
                "inspect_evidence",
            )

        # Даже у структурно стабильного раздела содержание могло измениться.
        for match in stable_matches:
            if searches >= self.max_searches:
                break
            from_year = match.get("from_year")
            to_year = match.get("to_year")
            section = match.get("to_section") or match.get("from_section")
            if not isinstance(from_year, int) or not isinstance(to_year, int):
                continue
            if not isinstance(section, dict):
                continue
            section_path = section.get("section_path")
            title = section.get("title")
            key = ((from_year, to_year), section_path)
            if key in checked:
                continue
            filters = replace(
                state.get("effective_filters") or SearchFilters(),
                years={from_year, to_year},
                year_from=None,
                year_to=None,
            )
            if isinstance(section_path, str) and section_path:
                filters = replace(filters, section_path_prefix=section_path)
            query = (
                f"{title} {state.get('question', '')}"
                if isinstance(title, str) and title
                else state.get("question", "")
            )
            return self._search_action(
                query,
                filters,
                "inspect_evidence",
            )
        return None
