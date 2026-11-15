"""Начальный Streamlit-чат для контракта сервиса ReportAgent."""

from __future__ import annotations

import importlib
import re
import uuid
from collections.abc import Mapping
from typing import Any

try:
    from .citations import render_citations
except ImportError:  # Streamlit может исполнять этот файл напрямую.
    from citations import render_citations


AVAILABLE_ISSUERS = ("ПАО Сбербанк", "БПС-Сбербанк")
_ALLOWED_TOOLS = {"search_reports", "get_structure_tree", "get_entities", "get_chunk"}
_SECRET_KEY = re.compile(r"(api.?key|secret|token|password|credential)", re.IGNORECASE)

#: Подписи ячеек coverage. ``not_found`` — «не найдено поиском», а не
#: «показателя нет в отчёте»; ``year_absent`` нельзя показывать как 0/пусто.
_COVERAGE_STATUS_STYLES: dict[str, dict[str, str]] = {
    "covered": {"label": "найдено", "icon": "✅"},
    "not_found": {"label": "не найдено поиском", "icon": "🔍"},
    "year_absent": {"label": "года нет в корпусе", "icon": "🚫"},
}
_COVERAGE_UNKNOWN_STYLE = {"label": "нет данных", "icon": "—"}

_COVERAGE_REPORT_STYLES: dict[str, dict[str, str]] = {
    "not_required": {
        "label": "Проверка покрытия не требовалась",
        "tone": "info",
        "icon": "ℹ️",
    },
    "complete": {"label": "Покрытие полное", "tone": "success", "icon": "✅"},
    "partial": {"label": "Покрытие частичное", "tone": "warning", "icon": "⚠️"},
    "insufficient": {
        "label": "Недостаточно доказательств",
        "tone": "error",
        "icon": "⛔",
    },
}

_VERIFICATION_STATUS_STYLES: dict[str, dict[str, str]] = {
    "ok": {"label": "Проверка пройдена", "tone": "success", "icon": "✅"},
    "warning": {"label": "Есть предупреждения", "tone": "warning", "icon": "⚠️"},
    "failed": {"label": "Проверка не пройдена", "tone": "error", "icon": "⛔"},
}
_FATAL_SEVERITIES = {"fatal", "error", "critical"}

_STAGE_ORDER = ("dense", "bm25", "graph", "rrf", "rerank", "returned", "context")
_STAGE_LABELS = {
    "dense": "Dense (векторный)",
    "bm25": "BM25 (лексический)",
    "graph": "Graph",
    "rrf": "RRF (слияние)",
    "rerank": "Rerank",
    "returned": "Возвращено",
    "context": "Context (pack_context)",
}
_STAGE_REASONS = {
    "not_requested": "не запрашивался",
    "failed": "ошибка этапа",
    "error": "ошибка этапа",
    "fallback": "использован fallback",
    "upstream_failed": "предыдущий этап недоступен",
    "disabled": "этап отключён",
    "not_configured": "этап не настроен",
    "no_candidates": "нет кандидатов",
    "no_index": "нет индекса",
    "timeout": "таймаут",
    "unavailable": "недоступен",
}
_STAGE_STATUS_LABELS = {
    "measured": "измерен",
    "empty": "измерен, без результатов",
    "unavailable": "не измерялся",
    "count_only": "только счётчик",
    "unknown": "нет данных",
}
_STAGE_STATUS_ICONS = {
    "measured": "✅",
    "empty": "⚪",
    "unavailable": "⛔",
    "count_only": "•",
    "unknown": "—",
}

_STRUCTURAL_CHANGE_LABELS = {
    "stable": "Без изменений",
    "added": "Добавлен",
    "removed": "Не обнаружен в более позднем отчёте",
    "renamed": "Переименован",
    "restructured": "Перемещён / изменён уровень",
    "expanded": "Объём вырос",
    "reduced": "Объём сократился",
}
_STRUCTURAL_DETAIL_LABELS = {
    "section_not_found_in_later_report": "не найден в более позднем отчёте",
    "section_not_found_in_earlier_report": "не найден в более раннем отчёте",
    "content_changed": "содержание изменилось",
    "verified_by_llm": "подтверждено проверкой",
    # Backend сворачивает split/merge в ``restructured``; UI не выдаёт это
    # за отдельную поддерживаемую категорию.
    "section_split": "раздел перестроен",
    "section_merged": "раздел перестроен",
}


def new_conversation_id() -> str:
    """Создаёт непрозрачный уникальный ID изолированного диалога."""
    return str(uuid.uuid4())


def service_options(service: Any) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Годы и doc_id из фактического индекса, без списка годов в коде.

    Если backend не умеет сообщать покрытие, UI оставляет фильтры пустыми:
    это честнее, чем показывать годы, которых может не быть в корпусе.
    """
    backend = getattr(getattr(service, "tools", None), "search_backend", None)
    years: tuple[int, ...] = ()
    doc_ids: tuple[str, ...] = ()

    method = getattr(backend, "available_years", None)
    if callable(method):
        try:
            years = tuple(sorted({int(year) for year in method()}))
        except (TypeError, ValueError):
            years = ()

    method = getattr(backend, "available_doc_ids", None)
    if callable(method):
        try:
            doc_ids = tuple(sorted({str(item) for item in method() if item}))
        except (TypeError, ValueError):
            doc_ids = ()

    return years, doc_ids


def normalize_filters(
    years: Any = None,
    issuers: Any = None,
    doc_ids: Any = None,
) -> dict[str, list[Any]]:
    """Приводит видимые выборки UI к словарю фильтров AgentService."""
    normalized: dict[str, set[Any]] = {}
    for key, values in (("years", years), ("issuers", issuers), ("allowed_doc_ids", doc_ids)):
        if values is None:
            continue
        if isinstance(values, (str, int)):
            values = [values]
        selected = {value for value in values if value not in (None, "")}
        if selected:
            normalized[key] = selected
    return normalized


def build_service_filters(ui_filters: dict[str, Any], filters_factory: Any = None) -> Any:
    """Собирает SearchFilters через внедрённую фабрику модели, если она есть.

    Fallback-словарь использует точные имена полей из spec_retrieval_engine.md,
    поэтому headless-заглушки работают ещё до появления backend-модели.
    """
    if not ui_filters:
        return None
    if callable(filters_factory):
        return filters_factory(**ui_filters)
    return ui_filters


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


_MISSING = object()


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    """Сначала читает поле события напрямую, затем из его ``data``."""
    if isinstance(event, Mapping):
        direct = event.get(key, _MISSING)
        data = event.get("data", _MISSING)
    else:
        direct = getattr(event, key, _MISSING)
        data = getattr(event, "data", _MISSING)
    if direct is not _MISSING:
        return direct
    if isinstance(data, Mapping) and key in data:
        return data[key]
    return default


def event_result(event: Any) -> Any:
    """Возвращает терминальный AgentRunResult без прохода через безопасные сводки."""
    return _event_get(event, "result")


def _event_name(event: Any) -> str:
    name = (
        _event_get(event, "event_type")
        or _event_get(event, "type")
        or _event_get(event, "event")
        or _event_get(event, "kind")
    )
    if name is None:
        name = event.__class__.__name__
    if hasattr(name, "value"):
        name = name.value
    return str(name).lower().replace("-", "_")


def _safe_text(value: Any, limit: int = 240) -> str:
    """Ограничивает текст и скрывает значения, похожие на учётные данные."""
    if not isinstance(value, str):
        return ""
    value = re.sub(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", value)
    return value[:limit]


def safe_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Оставляет только короткие и не секретные аргументы для видимой трассы."""
    if not isinstance(arguments, Mapping):
        return {}
    safe: dict[str, Any] = {}
    visible_keys = {
        "query", "filters", "year", "years", "issuer", "issuers", "doc_id", "doc_ids",
        "allowed_doc_ids",
        "section_path", "retrieval_mode", "top_k", "max_depth", "max_hops",
        "entity_names", "chunk_id",
    }
    for key, value in arguments.items():
        name = str(key)
        if name.lower() not in visible_keys or _SECRET_KEY.search(name):
            continue
        if name.lower() in {"query"}:
            safe[name] = "[запрос скрыт]"
        elif isinstance(value, Mapping):
            safe[name] = safe_tool_arguments(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            safe[name] = [
                item if isinstance(item, (int, float, bool)) else _safe_text(str(item), 60)
                for item in list(value)[:5]
            ]
        elif isinstance(value, (str, int, float, bool)):
            safe[name] = _safe_text(str(value), 80) if isinstance(value, str) else value
    return safe


def summarize_event(event: Any) -> dict[str, Any]:
    """Превращает одно AgentEvent в компактную безопасную для UI сводку."""
    name = _event_name(event)
    summary: dict[str, Any] = {"event": name}
    if name == "run_started":
        request_id = _event_get(event, "request_id")
        if request_id is not None:
            summary["request_id"] = _safe_text(str(request_id), 100)
        summary["stage"] = "Запрос принят"
    elif name == "analysis_completed":
        summary["stage"] = "Анализ запроса завершён"
    elif name == "tool_started":
        tool = _event_get(event, "tool_name")
        if tool in _ALLOWED_TOOLS:
            summary.update(tool_name=tool, arguments=safe_tool_arguments(_event_get(event, "short_arguments", {})))
            summary["stage"] = f"Вызов инструмента: {tool}"
    elif name == "tool_completed":
        tool = _event_get(event, "tool_name")
        if tool in _ALLOWED_TOOLS:
            summary.update(
                tool_name=tool,
                result_count=_nonnegative_number(_event_get(event, "result_count")),
                duration_ms=_nonnegative_number(_event_get(event, "duration_ms")),
                status=_safe_text(str(_event_get(event, "status", "unknown")), 40),
            )
    elif name == "evidence_updated":
        summary["stage"] = "Свидетельства обновлены"
    elif name == "answer_started":
        summary["stage"] = "Подготовка ответа"
    elif name == "answer_delta":
        summary["delta"] = _safe_text(_event_get(event, "text", ""), 4000)
    elif name == "run_completed":
        summary["stage"] = "Обработка завершена"
        summary["answer_status"] = _safe_text(str(_event_get(event, "answer_status", "")), 40)
        summary["duration_ms"] = _nonnegative_number(_event_get(event, "duration_ms"))
    elif name == "run_failed":
        summary["stage"] = "Обработка не выполнена"
        summary["error_code"] = _safe_text(str(_event_get(event, "error_code", "")), 80)
        summary["user_message"] = _safe_text(str(_event_get(event, "user_message", "")), 240)
    return summary


def _nonnegative_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value


def _resolve_service_factory() -> Any:
    """Загружает единственный документированный хук интеграции, если пакет агента есть."""
    try:
        module = importlib.import_module("src.agent.service")
    except ImportError:
        return None
    factory = getattr(module, "create_agent_service", None)
    return factory if callable(factory) else None


def _make_service(agent_service: Any, service_factory: Any) -> Any:
    if agent_service is not None:
        return agent_service
    factory = service_factory if service_factory is not None else _resolve_service_factory()
    if callable(factory):
        try:
            return factory()
        except Exception:
            # Не показываем текст исключения backend: он может содержать секреты.
            return None
    return None


def _answer_payload(value: Any) -> Any:
    """Раскрывает только обычные контейнеры result/answer из заявленного контракта."""
    if value is None:
        return None
    nested = _get(value, "result")
    if nested is not None:
        value = nested
    # У самого Answer есть status/citations. Сохраняем этот объект, а не
    # разворачиваем его текстовое поле `answer`.
    if _get(value, "status") in {"answer", "abstain"} and _get(value, "citations") is not None:
        return value
    answer = _get(value, "answer")
    return answer if answer is not None else value


def _render_answer(st: Any, answer: Any, streamed_text: str = "", *, include_text: bool = True) -> None:
    payload = _answer_payload(answer)
    if isinstance(payload, str):
        status = "answer"
        text = payload
        citations = []
        warnings = []
    else:
        status = _get(payload, "status", "answer")
        text = _get(payload, "answer", "")
        citations = _get(payload, "citations", []) or []
        warnings = _get(payload, "warnings", []) or []
    if status not in {"answer", "abstain"}:
        status = "answer"
    if status == "abstain":
        citations = []
    st.markdown(f"**Статус: `{status}`**")
    if status == "abstain":
        st.info("Агент воздержался от ответа на основании доступных свидетельств.")
    elif include_text and isinstance(text, str) and text:
        st.text(text)
    elif include_text and streamed_text:
        st.text(streamed_text)
    elif include_text:
        st.info("Завершённый текст ответа не предоставлен AgentService.")
    render_citations(st, citations)
    _render_warnings(st, warnings)


def _render_warnings(st: Any, warnings: Any) -> None:
    if isinstance(warnings, str):
        warnings = [warnings]
    if warnings:
        with st.expander("Предупреждения"):
            for warning in warnings:
                st.warning(_safe_text(str(warning), 300))


def _string_list(value: Any) -> list[str]:
    """Список строк без выдуманных элементов; одиночная строка тоже допустима."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _year_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def coverage_status_style(status: Any) -> dict[str, Any]:
    """Подпись и иконка статуса ячейки coverage, включая неизвестный статус."""
    key = status.casefold() if isinstance(status, str) else ""
    style = _COVERAGE_STATUS_STYLES.get(key)
    known = style is not None
    if style is None:
        style = _COVERAGE_UNKNOWN_STYLE
    return {
        "status": key or None,
        "known": known,
        "label": style["label"],
        "icon": style["icon"],
        "display": f"{style['icon']} {style['label']}",
    }


def coverage_report_style(status: Any) -> dict[str, Any] | None:
    """Стиль баннера ``CoverageReport.status``; неизвестный статус не рисуем."""
    key = status.casefold() if isinstance(status, str) else ""
    style = _COVERAGE_REPORT_STYLES.get(key)
    if style is None:
        return None
    return {"status": key, **style}


def coverage_view(coverage: Any) -> dict[str, Any] | None:
    """Нормализует ``diagnostics["coverage"]`` в строки metric × year.

    UI не вычисляет покрытие: берётся матрица ``metrics``, а при её отсутствии —
    плоский список ``cells``. Неизвестная ячейка остаётся «нет данных», а не
    превращается в ``not_found`` или ноль.
    """
    if not isinstance(coverage, Mapping) or not coverage:
        return None

    raw_matrix = coverage.get("metrics")
    matrix = raw_matrix if isinstance(raw_matrix, Mapping) else {}
    raw_cells = coverage.get("cells")
    cells = (
        [cell for cell in raw_cells if isinstance(cell, Mapping)]
        if isinstance(raw_cells, (list, tuple))
        else []
    )

    years: set[int] = set()
    metric_order: list[str] = []
    metric_labels: dict[str, str] = {}
    cell_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    evidence_cells: list[dict[str, Any]] = []
    metricless_cells = False

    for cell in cells:
        metric = cell.get("metric")
        metric = metric if isinstance(metric, str) else ""
        label = cell.get("metric_label")
        if metric and isinstance(label, str) and label:
            metric_labels.setdefault(metric, label)
        status = cell.get("status")
        status = status if isinstance(status, str) else None
        chunk_ids = _string_list(cell.get("chunk_ids"))
        year = _year_value(cell.get("year"))

        if metric and metric not in metric_order:
            metric_order.append(metric)
        if not metric:
            metricless_cells = True
        if year is not None:
            years.add(year)
            cell_by_key.setdefault(
                (metric, year),
                {"status": status, "chunk_ids": chunk_ids},
            )
        if chunk_ids:
            evidence_cells.append(
                {
                    "metric": metric,
                    "metric_label": (
                        metric_labels.get(metric)
                        or metric
                        or "Покрытие по годам"
                    ),
                    "year": year,
                    "chunk_ids": chunk_ids,
                }
            )

    matrix_years: dict[str, dict[int, str]] = {}
    matrix_summary: dict[str, str] = {}
    for metric, per_year in matrix.items():
        if not isinstance(metric, str) or not isinstance(per_year, Mapping):
            continue
        if metric not in metric_order:
            metric_order.append(metric)
        normalized: dict[int, str] = {}
        for key, status in per_year.items():
            if key == "*":
                if isinstance(status, str):
                    matrix_summary[metric] = status
                continue
            year = _year_value(key)
            if year is None or not isinstance(status, str):
                continue
            normalized[year] = status
            years.add(year)
        matrix_years[metric] = normalized

    if not metric_order and metricless_cells:
        metric_order = [""]

    def status_for(metric: str, year: int) -> Any:
        status = matrix_years.get(metric, {}).get(year)
        if status is not None:
            return status
        cell = cell_by_key.get((metric, year))
        return cell.get("status") if cell else None

    sorted_years = sorted(years)
    metric_rows: list[dict[str, Any]] = []
    for metric in metric_order:
        per_year: list[dict[str, Any]] = []
        for year in sorted_years:
            style = coverage_status_style(status_for(metric, year))
            cell = cell_by_key.get((metric, year)) or {}
            per_year.append(
                {
                    "year": year,
                    "status": style["status"],
                    "known": style["known"],
                    "label": style["label"],
                    "icon": style["icon"],
                    "display": style["display"],
                    "chunk_ids": list(cell.get("chunk_ids", [])),
                }
            )
        summary = matrix_summary.get(metric)
        summary_style = coverage_status_style(summary)
        metric_rows.append(
            {
                "metric": metric,
                "metric_label": (
                    metric_labels.get(metric)
                    or metric
                    or "Покрытие по годам"
                ),
                "cells": per_year,
                "summary": summary_style["status"],
                "summary_label": (
                    summary_style["label"] if summary_style["known"] else None
                ),
                "summary_display": (
                    summary_style["display"] if summary_style["known"] else None
                ),
            }
        )

    missing_metric_years: list[dict[str, Any]] = []
    raw_missing_cells = coverage.get("missing_metric_years")
    if isinstance(raw_missing_cells, (list, tuple)):
        for item in raw_missing_cells:
            if not isinstance(item, Mapping):
                continue
            metric = item.get("metric")
            metric = metric if isinstance(metric, str) else ""
            label = item.get("metric_label")
            style = coverage_status_style(item.get("status"))
            missing_metric_years.append(
                {
                    "metric": metric,
                    "metric_label": (
                        label
                        if isinstance(label, str) and label
                        else metric_labels.get(metric, metric)
                    ),
                    "year": _year_value(item.get("year")),
                    "status": style["status"],
                    "status_label": style["label"],
                    "icon": style["icon"],
                }
            )

    missing_metrics: list[dict[str, str]] = []
    raw_missing_metrics = coverage.get("missing_metrics")
    if isinstance(raw_missing_metrics, (list, tuple)):
        for metric in raw_missing_metrics:
            if isinstance(metric, str) and metric:
                missing_metrics.append(
                    {
                        "metric": metric,
                        "metric_label": metric_labels.get(metric, metric),
                    }
                )

    raw_notes = coverage.get("notes")
    notes = (
        [note for note in raw_notes if isinstance(note, str) and note]
        if isinstance(raw_notes, (list, tuple))
        else []
    )

    raw_missing_years = coverage.get("missing_years")
    missing_years = sorted(
        {
            year
            for year in (
                _year_value(value)
                for value in (
                    raw_missing_years
                    if isinstance(raw_missing_years, (list, tuple))
                    else []
                )
            )
            if year is not None
        }
    )

    status = coverage.get("status")
    report_style = coverage_report_style(status)
    return {
        "status": report_style["status"] if report_style else None,
        "status_label": report_style["label"] if report_style else None,
        "status_tone": report_style["tone"] if report_style else None,
        "status_icon": report_style["icon"] if report_style else None,
        "years": sorted_years,
        "metrics": metric_rows,
        "notes": notes,
        "missing_metrics": missing_metrics,
        "missing_metric_years": missing_metric_years,
        "missing_years": missing_years,
        "evidence_cells": evidence_cells,
        "distinct_docs": _nonnegative_int(coverage.get("distinct_docs")),
        "distinct_sections": _nonnegative_int(coverage.get("distinct_sections")),
    }


def verification_view(verification: Any) -> dict[str, Any] | None:
    """Строки проверки ответа без traceback и значений, похожих на секреты."""
    if not isinstance(verification, Mapping) or not verification:
        return None

    raw_issues = verification.get("issues")
    issues: list[dict[str, Any]] = []
    if isinstance(raw_issues, (list, tuple)):
        for item in raw_issues:
            if not isinstance(item, Mapping):
                continue
            raw_severity = str(item.get("severity", "warning")).casefold()
            fatal = raw_severity in _FATAL_SEVERITIES
            chunk_ids = _string_list(item.get("chunk_ids"))
            if not chunk_ids:
                chunk_ids = _string_list(item.get("chunk_id"))
            fact_index = item.get("fact_index")
            issues.append(
                {
                    "code": str(item.get("code") or "verification_issue"),
                    "severity": "fatal" if fatal else "warning",
                    "severity_label": "критично" if fatal else "предупреждение",
                    "message": _safe_text(str(item.get("message") or ""), 300),
                    "chunk_ids": chunk_ids,
                    "fact_index": (
                        fact_index
                        if isinstance(fact_index, int)
                        and not isinstance(fact_index, bool)
                        else None
                    ),
                    "fatal": fatal,
                }
            )

    status = verification.get("status")
    status = status.casefold() if isinstance(status, str) else ""
    if status not in _VERIFICATION_STATUS_STYLES:
        status = (
            "failed"
            if any(issue["fatal"] for issue in issues)
            else "warning" if issues else "ok"
        )
    style = _VERIFICATION_STATUS_STYLES[status]
    fatal_count = sum(1 for issue in issues if issue["fatal"])
    return {
        "status": status,
        "status_label": style["label"],
        "tone": style["tone"],
        "icon": style["icon"],
        "issues": issues,
        "fatal_count": fatal_count,
        "warning_count": len(issues) - fatal_count,
        "supported_citations": _string_list(
            verification.get("supported_citations")
        ),
        "unsupported_facts": [
            item
            for item in (
                verification.get("unsupported_facts")
                if isinstance(
                    verification.get("unsupported_facts"), (list, tuple)
                )
                else []
            )
            if isinstance(item, int) and not isinstance(item, bool)
        ],
    }


def retrieval_stage_rows(call: Any) -> list[dict[str, Any]]:
    """Этапы одного retrieval-вызова: ``None``/``available=False`` — не 0."""
    if not isinstance(call, Mapping):
        return []
    stages = call.get("stages")
    stages = stages if isinstance(stages, Mapping) else {}
    available = call.get("available")
    available = available if isinstance(available, Mapping) else {}
    unavailable = call.get("unavailable")
    unavailable = unavailable if isinstance(unavailable, Mapping) else {}
    fallback = call.get("fallback")
    fallback = fallback if isinstance(fallback, Mapping) else {}
    counts = call.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}

    names: list[str] = []
    for name in _STAGE_ORDER:
        if name in stages or name in available or name in unavailable:
            names.append(name)
    for block in (stages, available, unavailable):
        for name in block:
            if isinstance(name, str) and name not in names:
                names.append(name)
    if not names:
        for name in counts:
            if isinstance(name, str) and name not in names:
                names.append(name)

    rows: list[dict[str, Any]] = []
    for name in names:
        has_stage = name in stages
        raw_items = stages.get(name, _MISSING)
        if has_stage and raw_items is None:
            is_available: bool | None = False
        elif name in available:
            is_available = bool(available.get(name))
        elif has_stage:
            is_available = True
        else:
            is_available = None

        reason = unavailable.get(name)
        reason = reason if isinstance(reason, str) and reason else None
        reason_label = _STAGE_REASONS.get(reason, reason) if reason else None
        count = (
            len(raw_items)
            if isinstance(raw_items, (list, tuple))
            else _nonnegative_int(counts.get(name))
        )

        raw_fallback = fallback.get(name)
        if raw_fallback is None and name == "rerank":
            raw_fallback = fallback.get("reranker")
        fallback_flag = bool(raw_fallback)

        if is_available is False:
            status = "unavailable"
            count = None
            count_label = "не измерялся"
            if reason_label:
                count_label += f" ({reason_label})"
        elif isinstance(raw_items, (list, tuple)):
            status = "empty" if not raw_items else "measured"
            count_label = (
                "0 — этап выполнен без результатов"
                if not raw_items
                else str(len(raw_items))
            )
        elif count is not None:
            status = "count_only"
            count_label = f"{count} (счётчик трассы)"
        else:
            status = "unknown"
            count_label = "нет данных"
        if fallback_flag and status != "unavailable":
            count_label += " · fallback"

        rows.append(
            {
                "name": name,
                "label": _STAGE_LABELS.get(name, name),
                "status": status,
                "status_label": _STAGE_STATUS_LABELS[status],
                "icon": _STAGE_STATUS_ICONS[status],
                "available": is_available,
                "count": count,
                "count_label": count_label,
                "reason": reason,
                "reason_label": reason_label,
                "fallback": fallback_flag,
            }
        )
    return rows


def retrieval_counters(call: Any) -> dict[str, Any]:
    """Плоские счётчики retrieval для совместимости с прежней диагностикой."""
    if not isinstance(call, Mapping):
        call = {}
    counts = call.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}

    def counter(flat: str, nested: str | None = None) -> int | None:
        value = _nonnegative_int(call.get(flat))
        if value is None and nested is not None:
            value = _nonnegative_int(counts.get(nested))
        return value

    return {
        "retrieval_mode": (
            call.get("retrieval_mode")
            if isinstance(call.get("retrieval_mode"), str)
            else "hybrid"
        ),
        "query_count": counter("query_count"),
        "fused_candidates": counter("fused_candidates", "rrf"),
        "returned": counter("returned", "returned"),
        "distinct_docs": counter("distinct_docs"),
        "distinct_sections": counter("distinct_sections"),
        "distinct_years": counter("distinct_years"),
        "counts": {
            "dense": counter("dense_candidates", "dense"),
            "bm25": counter("bm25_candidates", "bm25"),
            "graph": counter("graph_candidates", "graph"),
            "rrf": counter("fused_candidates", "rrf"),
            "rerank": counter("rerank_candidates", "rerank"),
            "returned": counter("returned", "returned"),
        },
    }


def retrieval_call_rows(calls: Any) -> list[dict[str, Any]]:
    """Безопасные строки ``diagnostics["retrieval_calls"]`` для expander."""
    if not isinstance(calls, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        call_index = call.get("call_index")
        rows.append(
            {
                "call_index": (
                    call_index
                    if isinstance(call_index, int)
                    and not isinstance(call_index, bool)
                    else index
                ),
                "tool_name": (
                    call.get("tool_name")
                    if isinstance(call.get("tool_name"), str)
                    else "search_reports"
                ),
                "retrieval_mode": retrieval_counters(call)["retrieval_mode"],
                "top_k": _nonnegative_int(call.get("top_k")),
                "status": (
                    call.get("status")
                    if isinstance(call.get("status"), str)
                    else None
                ),
                "counters": retrieval_counters(call),
                "stages": retrieval_stage_rows(call),
                "fallback": sorted(
                    str(name)
                    for name, value in (
                        call.get("fallback")
                        if isinstance(call.get("fallback"), Mapping)
                        else {}
                    ).items()
                    if value and isinstance(name, str)
                ),
                "unavailable": {
                    str(name): str(reason)
                    for name, reason in (
                        call.get("unavailable")
                        if isinstance(call.get("unavailable"), Mapping)
                        else {}
                    ).items()
                    if isinstance(name, str) and isinstance(reason, str)
                },
                "warnings": _string_list(call.get("warnings")),
            }
        )
    return rows


def context_chunk_view(payload: Any) -> dict[str, Any]:
    """Различает «контекст не зафиксирован» и измеренный пустой контекст."""
    diagnostics = payload if isinstance(payload, Mapping) else {}
    raw_ids = diagnostics.get("context_chunk_ids")
    source = diagnostics.get("context_source")
    source = source if isinstance(source, str) and source else None
    if raw_ids is None:
        return {
            "available": False,
            "count": None,
            "chunk_ids": [],
            "source": source,
            "label": "не зафиксирован",
        }
    chunk_ids = _string_list(raw_ids)
    return {
        "available": True,
        "count": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "source": source,
        "label": f"{len(chunk_ids)} chunk id",
    }


def graph_evidence_view(payload: Any) -> dict[str, Any] | None:
    """Graph-only evidence без статуса подтверждённого факта."""
    if not isinstance(payload, Mapping):
        return None
    chunk_ids = _string_list(payload.get("graph_evidence_manifest"))

    calls = payload.get("retrieval_calls")
    graph_stage: dict[str, Any] | None = None
    if isinstance(calls, (list, tuple)):
        for call in calls:
            for row in retrieval_stage_rows(call):
                if row["name"] == "graph":
                    graph_stage = row
    if not chunk_ids and graph_stage is None:
        return None

    evidence_ids = set(_string_list(payload.get("evidence_manifest")))
    entries = [
        {
            "chunk_id": chunk_id,
            "confirmed": chunk_id in evidence_ids,
            "label": (
                "подтверждено найденным chunk"
                if chunk_id in evidence_ids
                else "graph-only, не подтверждено до открытия chunk"
            ),
        }
        for chunk_id in chunk_ids
    ]
    unconfirmed = [entry for entry in entries if not entry["confirmed"]]
    if unconfirmed:
        warning = (
            "Graph-only evidence: не подтверждено до открытия chunk. "
            "Не считать граф доказанным фактом."
        )
        tone = "warning"
    elif entries:
        warning = (
            "Graph evidence подтверждён найденными chunk-ами; "
            "всё равно откройте источник."
        )
        tone = "info"
    else:
        warning = "Graph-этап не вернул graph-only chunk id."
        tone = "info"

    return {
        "chunk_ids": chunk_ids,
        "entries": entries,
        "count": len(chunk_ids),
        "confirmed_count": len(entries) - len(unconfirmed),
        "unconfirmed_count": len(unconfirmed),
        "warning": warning,
        "tone": tone,
        "stage": graph_stage,
    }


def structural_change_rows(changes: Any) -> list[dict[str, Any]]:
    """Строки ``structural_changes`` без выдуманных split/merge категорий."""
    if not isinstance(changes, (list, tuple)):
        return []
    rows: list[dict[str, Any]] = []
    for item in changes:
        if not isinstance(item, Mapping):
            continue
        from_section = (
            item.get("from_section")
            if isinstance(item.get("from_section"), Mapping)
            else {}
        )
        to_section = (
            item.get("to_section")
            if isinstance(item.get("to_section"), Mapping)
            else {}
        )
        title = (
            to_section.get("title")
            or from_section.get("title")
            or "раздел"
        )
        section_path = (
            to_section.get("section_path")
            or from_section.get("section_path")
        )
        level = (
            to_section.get("level")
            if isinstance(to_section.get("level"), int)
            and not isinstance(to_section.get("level"), bool)
            else from_section.get("level")
        )
        page_start = (
            to_section.get("page_start")
            if isinstance(to_section.get("page_start"), int)
            and not isinstance(to_section.get("page_start"), bool)
            else from_section.get("page_start")
        )
        page_end = (
            to_section.get("page_end")
            if isinstance(to_section.get("page_end"), int)
            and not isinstance(to_section.get("page_end"), bool)
            else from_section.get("page_end")
        )
        change_type = item.get("change_type")
        change_type = (
            change_type.strip().casefold()
            if isinstance(change_type, str) and change_type.strip()
            else ""
        )
        label = _STRUCTURAL_CHANGE_LABELS.get(change_type)
        if label is None:
            label = change_type or "изменение"
        evidence = _string_list(item.get("evidence_chunk_ids"))
        if not evidence:
            evidence = _string_list(item.get("chunk_ids"))
        detail = item.get("detail")
        detail = detail if isinstance(detail, str) else None
        if detail is not None:
            detail = _STRUCTURAL_DETAIL_LABELS.get(detail.casefold(), detail)
        rows.append(
            {
                "from_year": _year_value(item.get("from_year")),
                "to_year": _year_value(item.get("to_year")),
                "section_title": str(title),
                "section_path": section_path if isinstance(section_path, str) else None,
                "level": (
                    level
                    if isinstance(level, int) and not isinstance(level, bool)
                    else None
                ),
                "page_start": (
                    page_start
                    if isinstance(page_start, int) and not isinstance(page_start, bool)
                    else None
                ),
                "page_end": (
                    page_end
                    if isinstance(page_end, int) and not isinstance(page_end, bool)
                    else None
                ),
                "change_type": change_type or None,
                "change_label": label,
                "detail": detail if isinstance(detail, str) else None,
                "score": (
                    float(item.get("score"))
                    if isinstance(item.get("score"), (int, float))
                    and not isinstance(item.get("score"), bool)
                    else None
                ),
                "evidence_chunk_ids": evidence,
                "document_id": (
                    item.get("document_id")
                    if isinstance(item.get("document_id"), str)
                    else None
                ),
                "year": _year_value(item.get("year")),
            }
        )
    return rows


def analysis_payload(value: Any) -> dict[str, Any]:
    """Безопасный снимок diagnostics для отрисовки и истории сообщений.

    Хранит идентификаторы и сводки, но не полный текст chunk'ов. Повторный
    вызов для уже собранного payload идемпотентен (маркер ``kind``).
    """
    if value is None:
        return {}
    if isinstance(value, Mapping) and value.get("kind") == "analysis":
        return dict(value)

    diagnostics = _get(value, "diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    raw_answer = _get(value, "answer")
    answer = (
        raw_answer
        if raw_answer is not None and not isinstance(raw_answer, str)
        else value
    )

    raw_legacy = _get(value, "retrieval_diagnostics")
    legacy = list(raw_legacy) if isinstance(raw_legacy, (list, tuple)) else []
    raw_calls = diagnostics.get("retrieval_calls")
    has_calls = isinstance(raw_calls, (list, tuple))
    calls = list(raw_calls) if has_calls else legacy

    verification = diagnostics.get("verification")
    if not isinstance(verification, Mapping):
        verification = _get(answer, "verification")
    if not isinstance(verification, Mapping):
        verification = _get(value, "verification")

    raw_changes = _get(value, "structural_changes")
    if not isinstance(raw_changes, (list, tuple)):
        raw_changes = diagnostics.get("structural_changes")
    changes = list(raw_changes) if isinstance(raw_changes, (list, tuple)) else []

    raw_calculations = _get(value, "calculations")
    if not isinstance(raw_calculations, (list, tuple)):
        raw_calculations = _get(answer, "calculations")
    calculations = (
        list(raw_calculations)
        if isinstance(raw_calculations, (list, tuple))
        else []
    )

    coverage = diagnostics.get("coverage")
    query_plan = diagnostics.get("query_plan")
    raw_attempts = diagnostics.get("search_attempts")
    attempts = list(raw_attempts) if isinstance(raw_attempts, (list, tuple)) else []
    raw_context_ids = diagnostics.get("context_chunk_ids")
    context_source = diagnostics.get("context_source")

    return {
        "kind": "analysis",
        "calculations": [
            dict(item) for item in calculations if isinstance(item, Mapping)
        ],
        "structural_changes": [
            dict(item) for item in changes if isinstance(item, Mapping)
        ],
        "verification": dict(verification) if isinstance(verification, Mapping) else None,
        "coverage": dict(coverage) if isinstance(coverage, Mapping) else None,
        "retrieval_calls": [
            dict(item) for item in calls if isinstance(item, Mapping)
        ],
        "legacy_retrieval": (
            [] if has_calls else [dict(item) for item in legacy if isinstance(item, Mapping)]
        ),
        "query_plan": dict(query_plan) if isinstance(query_plan, Mapping) else None,
        "search_attempts": [
            dict(item) for item in attempts if isinstance(item, Mapping)
        ],
        "context_chunk_ids": (
            _string_list(raw_context_ids) if raw_context_ids is not None else None
        ),
        "context_source": (
            context_source if isinstance(context_source, str) else None
        ),
        "graph_evidence_manifest": _string_list(
            diagnostics.get("graph_evidence_manifest")
        ),
        "evidence_manifest": _string_list(diagnostics.get("evidence_manifest")),
    }


def _render_analysis(st: Any, result: Any) -> None:
    """Показывает coverage, verification, структуру и диагностику retrieval."""
    payload = analysis_payload(result)
    if not payload:
        return
    _render_calculations(st, payload.get("calculations"))
    _render_coverage(st, payload.get("coverage"))
    _render_verification(st, payload.get("verification"))
    _render_structural_changes(st, payload.get("structural_changes"))
    _render_retrieval_diagnostics(st, payload)
    _render_graph_evidence(st, payload)


def _render_calculations(st: Any, calculations: Any) -> None:
    if not isinstance(calculations, (list, tuple)) or not calculations:
        return
    with st.expander("Расчёты по фактам"):
        st.caption("Это расчёт по подтверждённым фактам, а не значение из отчёта.")
        for item in calculations:
            if not isinstance(item, Mapping):
                continue
            st.write(
                f"{item.get('metric', 'Показатель')}: "
                f"{item.get('from_year')}→{item.get('to_year')}, "
                f"{item.get('absolute_change')} "
                f"{item.get('unit', '')}".strip()
            )


def _render_coverage(st: Any, coverage: Any) -> None:
    view = coverage_view(coverage)
    if view is None:
        return
    tone = view.get("status_tone")
    if tone:
        renderer = getattr(st, tone, None)
        if callable(renderer):
            renderer(f"{view['status_icon']} {view['status_label']}")

    metric_rows = view.get("metrics") or []
    years = view.get("years") or []
    if metric_rows and years:
        has_summary = any(row.get("summary_display") for row in metric_rows)
        table: list[dict[str, Any]] = []
        for row in metric_rows:
            item = {"Показатель": row["metric_label"]}
            for cell in row["cells"]:
                item[str(cell["year"])] = cell["display"]
            if has_summary:
                item["Итого"] = row.get("summary_display") or "— нет данных"
            table.append(item)
        st.dataframe(table, hide_index=True)
        st.caption(
            "✅ найдено · 🔍 не найдено поиском · "
            "🚫 года нет в корпусе · — нет данных"
        )

    if view.get("missing_metrics"):
        st.caption(
            "Не закрыты показатели: "
            + ", ".join(
                item["metric_label"] for item in view["missing_metrics"]
            )
        )
    if view.get("missing_metric_years"):
        with st.expander("Незакрытые ячейки показателей"):
            for item in view["missing_metric_years"]:
                year = item.get("year")
                where = f" за {year}" if year is not None else ""
                st.caption(
                    f"{item['icon']} {item['metric_label']}{where}: "
                    f"{item['status_label']}"
                )
    if view.get("notes"):
        with st.expander("Замечания покрытия"):
            for note in view["notes"]:
                st.caption(f"• {note}")
    if view.get("evidence_cells"):
        with st.expander("Evidence по ячейкам покрытия"):
            for item in view["evidence_cells"]:
                year = item.get("year")
                where = f" · {year}" if year is not None else ""
                st.caption(
                    f"{item['metric_label']}{where}: "
                    + ", ".join(item["chunk_ids"])
                )
    meta: list[str] = []
    if view.get("distinct_docs") is not None:
        meta.append(f"документов: {view['distinct_docs']}")
    if view.get("distinct_sections") is not None:
        meta.append(f"разделов: {view['distinct_sections']}")
    if view.get("missing_years"):
        meta.append("нет лет: " + ", ".join(str(y) for y in view["missing_years"]))
    if meta:
        st.caption(" · ".join(meta))


def _render_verification(st: Any, verification: Any) -> None:
    view = verification_view(verification)
    if view is None:
        return
    if view["status"] == "failed":
        st.error(
            "Проверка ответа не пройдена: в ответе есть критичные "
            "несоответствия источникам."
        )
    with st.expander(
        "Проверка ответа",
        expanded=view["status"] == "failed",
    ):
        st.caption(
            f"{view['icon']} {view['status_label']} · "
            f"критичных: {view['fatal_count']} · "
            f"предупреждений: {view['warning_count']}"
        )
        if not view["issues"]:
            st.caption("Замечаний нет.")
        for issue in view["issues"]:
            line = (
                f"{issue['severity_label']}: `{issue['code']}` — "
                f"{issue['message']}"
            )
            if issue["fatal"]:
                st.error(line)
            else:
                st.warning(line)
            details: list[str] = []
            if issue["fact_index"] is not None:
                details.append(f"факт #{issue['fact_index']}")
            if issue["chunk_ids"]:
                details.append("chunk: " + ", ".join(issue["chunk_ids"]))
            if details:
                st.caption(" · ".join(details))
        if view["unsupported_facts"]:
            st.caption(
                "Неподтверждённые факты: "
                + ", ".join(f"#{index}" for index in view["unsupported_facts"])
            )


def _render_structural_changes(st: Any, changes: Any) -> None:
    rows = structural_change_rows(changes)
    if not rows:
        return
    with st.expander("Изменения структуры"):
        st.caption(
            "Сравнение попарно по годам; «не обнаружен» не равно «удалён». "
            "Разделение и объединение разделов анализатор не подтверждает."
        )
        for row in rows:
            st.markdown(
                f"**{row['section_title']}** — {row['change_label']}"
            )
            details: list[str] = []
            if row["from_year"] is not None or row["to_year"] is not None:
                details.append(f"{row['from_year']} → {row['to_year']}")
            if row["section_path"]:
                details.append(row["section_path"])
            if row["level"] is not None:
                details.append(f"уровень {row['level']}")
            if row["page_start"] is not None or row["page_end"] is not None:
                details.append(f"стр. {row['page_start']}–{row['page_end']}")
            if row["detail"]:
                details.append(row["detail"])
            if details:
                st.caption(" · ".join(details))
            if row["evidence_chunk_ids"]:
                st.caption(
                    "Evidence: " + ", ".join(row["evidence_chunk_ids"])
                )


def _retrieval_counters_text(counters: Mapping[str, Any]) -> str:
    def value(key: str) -> str:
        item = counters.get(key)
        return "—" if item is None else str(item)

    return (
        f"Режим: {value('retrieval_mode')}; "
        f"запросов: {value('query_count')}; "
        f"кандидатов: {value('fused_candidates')}; "
        f"возвращено: {value('returned')}; "
        f"документов: {value('distinct_docs')}; "
        f"разделов: {value('distinct_sections')}; "
        f"лет: {value('distinct_years')}"
    )


def _render_retrieval_diagnostics(st: Any, payload: Mapping[str, Any]) -> None:
    calls = retrieval_call_rows(payload.get("retrieval_calls"))
    legacy = payload.get("legacy_retrieval")
    legacy = legacy if isinstance(legacy, (list, tuple)) else []
    context = context_chunk_view(payload)
    if not calls and not legacy and not context["available"]:
        return

    with st.expander("Диагностика retrieval"):
        plan = payload.get("query_plan")
        if isinstance(plan, Mapping):
            parts: list[str] = []
            if plan.get("intent"):
                parts.append(f"intent: {plan['intent']}")
            if plan.get("answer_mode"):
                parts.append(f"режим ответа: {plan['answer_mode']}")
            if isinstance(plan.get("years"), (list, tuple)) and plan["years"]:
                parts.append("годы: " + ", ".join(str(y) for y in plan["years"]))
            if (
                isinstance(plan.get("required_metrics"), (list, tuple))
                and plan["required_metrics"]
            ):
                parts.append(
                    "метрики: " + ", ".join(str(m) for m in plan["required_metrics"])
                )
            if isinstance(plan.get("aspects"), (list, tuple)) and plan["aspects"]:
                parts.append("аспекты: " + ", ".join(str(a) for a in plan["aspects"]))
            if parts:
                st.caption("План запроса: " + "; ".join(parts))

        if not calls and legacy:
            st.caption(
                "Старый формат диагностики без stages; "
                "показаны сохранённые счётчики."
            )
            for item in legacy:
                if isinstance(item, Mapping):
                    st.caption(_retrieval_counters_text(retrieval_counters(item)))

        for row in calls:
            st.markdown(
                f"**Вызов {row['call_index'] + 1}: `{row['tool_name']}`**"
            )
            header: list[str] = [
                f"режим: {row['retrieval_mode']}",
                f"top_k: {row['top_k'] if row['top_k'] is not None else '—'}",
            ]
            if row["status"]:
                header.append(f"статус: {row['status']}")
            st.caption(" · ".join(header))
            st.caption(_retrieval_counters_text(row["counters"]))
            if row["stages"]:
                st.dataframe(
                    [
                        {
                            "Этап": stage["label"],
                            "Статус": (
                                f"{stage['icon']} {stage['status_label']}"
                            ),
                            "Результат": stage["count_label"],
                        }
                        for stage in row["stages"]
                    ],
                    hide_index=True,
                )
            if row["fallback"]:
                st.caption("Fallback: " + ", ".join(row["fallback"]))
            for warning in row["warnings"]:
                st.caption(f"⚠️ {_safe_text(warning, 200)}")

        if context["available"]:
            source = context["source"] or "evidence_manifest"
            st.caption(
                f"Контекст ответа: {context['label']} "
                f"(источник: {source})"
            )
            if context["chunk_ids"]:
                with st.expander("Chunk IDs контекста"):
                    for chunk_id in context["chunk_ids"][:100]:
                        st.caption(chunk_id)
                    if len(context["chunk_ids"]) > 100:
                        st.caption(
                            f"… ещё {len(context['chunk_ids']) - 100}"
                        )
        else:
            source = context["source"] or "evidence_manifest"
            st.caption(
                "Контекст ответа не зафиксирован "
                f"(источник: {source})."
            )


def _render_graph_evidence(st: Any, payload: Mapping[str, Any]) -> None:
    view = graph_evidence_view(payload)
    if view is None:
        return
    with st.expander("Graph evidence"):
        if view["tone"] == "warning":
            st.warning(view["warning"])
        elif view["chunk_ids"]:
            st.info(view["warning"])
        else:
            st.caption(view["warning"])
        stage = view.get("stage")
        if isinstance(stage, Mapping):
            st.caption(
                f"Graph-этап: {stage['icon']} {stage['status_label']} · "
                f"{stage['count_label']}"
            )
        for entry in view["entries"]:
            st.caption(f"{entry['chunk_id']} — {entry['label']}")


def _render_tool_trace(st: Any, trace: list[dict[str, Any]]) -> None:
    if not trace:
        return
    with st.expander("Трасса инструментов"):
        for item in trace:
            label = item.get("tool_name", "Инструмент")
            fields = []
            if "arguments" in item:
                fields.append(f"Аргументы: {item['arguments']}")
            if item.get("result_count") is not None:
                fields.append(f"Результатов: {item['result_count']}")
            if item.get("duration_ms") is not None:
                fields.append(f"Длительность: {item['duration_ms']} мс")
            if item.get("status"):
                fields.append(f"Статус: {item['status']}")
            st.write(f"**{label}** — {'; '.join(fields)}")


def _call_agent(service: Any, query: str, conversation_id: str, filters: Any, st: Any) -> Any:
    trace: list[dict[str, Any]] = []
    warnings: list[Any] = []
    request_id = None
    stream_method = getattr(service, "stream", None)
    ask_method = getattr(service, "ask", None)

    if callable(stream_method):
        streamed_text = ""
        final_answer = None
        final_result = None
        failed = False
        completed = False
        status_context = st.status("Обработка запроса…", expanded=True)
        try:
            events = stream_method(query, conversation_id=conversation_id, filters=filters or None)
            def visible_deltas():
                nonlocal streamed_text, final_answer, final_result, failed, completed, request_id
                for event in events:
                    summary = summarize_event(event)
                    if summary.get("request_id"):
                        request_id = summary["request_id"]
                    if summary.get("stage"):
                        status_context.update(label=summary["stage"])
                    if summary.get("event") == "tool_started":
                        trace.append(summary)
                    elif summary.get("event") == "tool_completed":
                        matching = next((item for item in reversed(trace)
                                         if item.get("tool_name") == summary.get("tool_name")
                                         and "status" not in item), None)
                        if matching is None:
                            trace.append(summary)
                        else:
                            matching.update({k: v for k, v in summary.items() if k != "event"})
                    elif summary.get("event") == "answer_delta":
                        delta = summary.get("delta", "")
                        streamed_text += delta
                        if delta:
                            yield delta
                    elif summary.get("event") == "run_failed":
                        failed = True
                        message = summary.get("user_message")
                        if message:
                            warnings.append(message)
                    elif summary.get("event") == "run_completed":
                        completed = True
                        candidate = event_result(event) or _event_get(event, "answer")
                        final_result = candidate
                        final_answer = _answer_payload(candidate)
                        if final_answer is None:
                            answer_status = summary.get("answer_status")
                            if answer_status in {"answer", "abstain"}:
                                final_answer = {
                                    "status": answer_status,
                                    "answer": streamed_text if answer_status == "answer" else "",
                                    "citations": [],
                                    "warnings": [],
                                }
                    event_warnings = _event_get(event, "warnings")
                    if isinstance(event_warnings, (list, tuple)):
                        warnings.extend(event_warnings)

            # Инкрементальную отрисовку ведёт Streamlit. Fallback сохраняет ту же
            # семантику событий для маленьких тестовых заглушек/старых версий.
            if callable(getattr(st, "write_stream", None)):
                st.write_stream(visible_deltas())
            else:
                list(visible_deltas())
        except Exception:
            failed = True

        if failed:
            status_context.update(label="Запрос не выполнен", state="error")
        elif completed:
            status_context.update(label="Обработка завершена", state="complete")
        else:
            status_context.update(label="Поток завершился без итогового результата", state="error")
        _render_tool_trace(st, trace)
        if request_id:
            st.caption(f"Request ID: {request_id}")
        if failed:
            st.error("Не удалось завершить запрос. Подробности доступны в серверных логах.")
            _render_warnings(st, warnings)
            return None
        if final_answer is not None:
            _render_answer(st, final_answer, streamed_text, include_text=not bool(streamed_text))
            analysis = analysis_payload(final_result or final_answer)
            _render_analysis(st, analysis)
        elif streamed_text:
            st.warning("Поток завершился без итогового результата; ответ не подтверждён.")
        elif callable(ask_method):
            return _call_ask(service, ask_method, query, conversation_id, filters, st)
        else:
            st.warning("AgentService не вернул итоговый Answer.")
        _render_warnings(st, warnings)
        return {
            "answer": final_answer,
            "tool_trace": trace,
            "request_id": request_id,
            "warnings": warnings,
            "analysis": analysis_payload(final_result or final_answer),
        } if final_answer is not None else None

    if callable(ask_method):
        return _call_ask(service, ask_method, query, conversation_id, filters, st)
    else:
        st.error("AgentService не реализует stream() или ask() из спецификации.")
        return None


def _call_ask(service: Any, ask_method: Any, query: str, conversation_id: str,
              filters: Any, st: Any) -> Any:
    try:
        result = ask_method(query, conversation_id=conversation_id, filters=filters or None)
    except Exception:
        st.error("Не удалось завершить запрос. Подробности доступны в серверных логах.")
        return None
    request_id = _get(result, "request_id")
    if request_id:
        st.caption(f"Request ID: {_safe_text(str(request_id), 100)}")
    _render_tool_trace(st, _ask_trace(result))
    _render_answer(st, result)
    analysis = analysis_payload(result)
    _render_analysis(st, analysis)
    return {
        "answer": _answer_payload(result),
        "tool_trace": _ask_trace(result),
        "request_id": _get(result, "request_id"),
        "analysis": analysis,
    }


def _ask_trace(result: Any) -> list[dict[str, Any]]:
    tool_events = _get(result, "tool_events", []) or []
    trace = []
    for event in tool_events:
        tool = _get(event, "tool_name")
        if tool not in _ALLOWED_TOOLS:
            continue
        trace.append({
            "tool_name": tool,
            "arguments": safe_tool_arguments(_get(event, "short_arguments", _get(event, "arguments", {}))),
            "result_count": _nonnegative_number(_get(event, "result_count")),
            "duration_ms": _nonnegative_number(_get(event, "duration_ms")),
            "status": _safe_text(str(_get(event, "status", "unknown")), 40),
        })
    return trace


def main(
    agent_service: Any = None,
    *,
    service_factory: Any = None,
    filters_factory: Any = None,
    st_module: Any = None,
) -> None:
    """Отрисовывает приложение; зависимости внедряются для headless-тестов."""
    if st_module is None:
        try:
            import streamlit as st_module
        except ImportError:
            raise RuntimeError("Streamlit is required to launch the ReportAgent UI.") from None
    st = st_module
    st.set_page_config(page_title="ReportAgent", page_icon="📄", layout="wide")
    st.title("ReportAgent — отчёты Сбербанка")

    service = _make_service(agent_service, service_factory)
    if service is None:
        st.error(
            "Backend не настроен: AgentService недоступен. "
            "Подключите фабрику `src.agent.service.create_agent_service`."
        )

    available_years, available_doc_ids = service_options(service)

    state = st.session_state
    if "conversation_id" not in state:
        state.conversation_id = new_conversation_id()
    if "messages" not in state:
        state.messages = []

    with st.sidebar:
        st.subheader("Фильтры документов")
        years = st.multiselect("Годы", available_years, default=[])
        issuers = st.multiselect("Issuer", AVAILABLE_ISSUERS, default=[])
        doc_ids = st.multiselect("Документы", available_doc_ids, default=[])
        if st.button("Новый диалог"):
            state.conversation_id = new_conversation_id()
            state.messages = []
            if hasattr(st, "rerun"):
                st.rerun()
            return
        st.caption(f"Conversation ID: {state.conversation_id}")

    filters = normalize_filters(years, issuers, doc_ids)
    st.caption(f"Активные фильтры: {filters or 'не заданы'}")
    service_filters = build_service_filters(filters, filters_factory)
    for message in state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("answer") is not None:
                _render_answer(st, message["answer"])
                _render_analysis(st, message.get("analysis"))
                _render_tool_trace(st, message.get("tool_trace", []))
                _render_warnings(st, message.get("warnings", []))
                if message.get("request_id"):
                    st.caption(f"Request ID: {message['request_id']}")
            else:
                st.text(message["content"])

    query = st.chat_input("Задайте вопрос по годовому отчёту")
    if query:
        state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.text(query)
        with st.chat_message("assistant"):
            if service is None:
                st.info("Ответы не генерируются, пока backend не подключён.")
                return
            response = _call_agent(service, query, state.conversation_id, service_filters, st)
            if response:
                state.messages.append({"role": "assistant", **response})
        # Память диалога backend изолирована по conversation_id и принадлежит
        # AgentService; UI хранит только видимую историю чата.


if __name__ == "__main__":
    main()
