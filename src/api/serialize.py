"""Сериализация dataclass-контрактов ядра в публичный JSON adapter'а.

Правила §7.10: поля не пересобираются и не пересчитываются, ``calculations``,
``structural_changes``, ``coverage``, ``verification`` и ``effective_filters``
сохраняются полностью. Adapter не добавляет событий, которых не было в
backend, кроме ``message_id``, heartbeat, ``run_cancelled`` и ``resync``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from src.retrieval.hybrid_retriever import SearchFilters

CONTRACT_VERSION = "1"
TERMINAL_EVENT_TYPES = frozenset({"run_completed", "run_failed", "run_cancelled"})

#: Аргументы инструментов, безопасные для панели «Ход исследования».
#: query/queries/chunk_id/filters не публикуются.
_SAFE_TOOL_ARGUMENTS = frozenset(
    {
        "doc_id",
        "section_path",
        "max_depth",
        "max_hops",
        "retrieval_mode",
        "top_k",
        "use_graph",
    }
)


def utc_now_iso() -> str:
    """Текущее время в формате ISO 8601 UTC с секундной точностью."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def to_utc_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def jsonable(value: Any) -> Any:
    """Приводит значение к JSON-безопасному виду без изменения семантики."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict())
        except Exception:  # noqa: BLE001 — сериализация не должна ронять запрос
            return None
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return getattr(value, "__dict__", {}) or {}


def filters_to_public(value: Any) -> dict[str, Any]:
    """``SearchFilters``/``FilterSet`` → публичный FilterSet без пустых полей."""
    data = _as_mapping(value)
    if not data:
        return {}
    result: dict[str, Any] = {}
    years = data.get("years")
    if years:
        try:
            result["years"] = sorted({int(year) for year in years})
        except (TypeError, ValueError):
            pass
    for key in ("year_from", "year_to"):
        item = data.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            result[key] = item
    for key in ("issuers", "allowed_doc_ids", "kinds"):
        items = data.get(key)
        if items:
            values = sorted(str(item) for item in items if str(item))
            if values:
                result[key] = values
    return result


def _filter_set_data(filters: Any) -> dict[str, Any]:
    data = _as_mapping(filters)
    if not data and filters is not None:
        # pydantic-модель
        dump = getattr(filters, "model_dump", None)
        if callable(dump):
            data = dump(exclude_none=True)
    return {key: value for key, value in data.items() if value not in (None, [], ())}


def filter_set_to_search(filters: Any) -> SearchFilters | None:
    """Публичные фильтры → ``SearchFilters``. Пустые значения не передаются."""
    data = _filter_set_data(filters)
    if not data:
        return None
    kwargs: dict[str, Any] = {}
    if data.get("years"):
        kwargs["years"] = {int(year) for year in data["years"]}
    if data.get("year_from") is not None:
        kwargs["year_from"] = int(data["year_from"])
    if data.get("year_to") is not None:
        kwargs["year_to"] = int(data["year_to"])
    if data.get("issuers"):
        kwargs["issuers"] = {str(item) for item in data["issuers"]}
    if data.get("allowed_doc_ids"):
        kwargs["allowed_doc_ids"] = {
            str(item) for item in data["allowed_doc_ids"]
        }
    if data.get("kinds"):
        kwargs["kinds"] = {str(item) for item in data["kinds"]}
    return SearchFilters(**kwargs).normalized()


def _section_segments(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    elif isinstance(value, str):
        raw = value.split("/")
    else:
        raw = []
    return [item.strip() for item in raw if item and item.strip()]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def citation_to_evidence(citation: Any, metadata: Any = None) -> dict[str, Any]:
    """``Citation`` → ``EvidenceItem`` по таблице §5.3."""
    chunk_meta = metadata if isinstance(metadata, Mapping) else None
    chunk_id = str(_field(citation, "chunk_id", "") or "")
    segments = _section_segments(_field(citation, "section_path"))
    doc_id = _field(citation, "doc_id")
    if doc_id is None and chunk_meta is not None:
        doc_id = chunk_meta.get("doc_id")
    year = _field(citation, "year")
    if year is None and chunk_meta is not None:
        year = chunk_meta.get("year")
    if isinstance(year, bool) or not isinstance(year, int):
        year = None
    issuer = _field(citation, "issuer")
    if issuer is None and chunk_meta is not None:
        issuer = chunk_meta.get("issuer")
    kind = chunk_meta.get("kind") if chunk_meta is not None else None
    source_type = kind if kind in {"text", "table"} else None
    page_start = _positive_int(_field(citation, "page_pdf"))
    page_end = None
    if chunk_meta is not None:
        page_end = _positive_int(chunk_meta.get("page_end"))
    quote = _field(citation, "quote")
    return {
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "document_id": str(doc_id) if doc_id is not None else None,
        "document_title": (
            str(chunk_meta.get("title"))
            if chunk_meta is not None and chunk_meta.get("title")
            else None
        ),
        "issuer": str(issuer) if issuer is not None else None,
        "year": year,
        "section_title": segments[-1] if segments else None,
        "section_path": segments or None,
        "page_start": page_start,
        "page_end": page_end,
        "page_label": (
            str(_field(citation, "page_label"))
            if _field(citation, "page_label") is not None
            else None
        ),
        "excerpt": str(quote) if quote is not None else "",
        "source_type": source_type,
        "has_full_text": chunk_meta is not None and bool(chunk_meta),
    }


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result


def analysis_from_result(result: Any) -> dict[str, Any]:
    """``AgentRunResult``/``Answer`` → ``AnalysisResult`` без потери полей."""
    answer = _field(result, "answer", result)
    diagnostics = _field(result, "diagnostics", {}) or {}
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    calculations = _field(result, "calculations", None)
    if not calculations:
        calculations = _field(answer, "calculations", []) or []
    coverage = diagnostics.get("coverage")
    if coverage is None:
        coverage = _field(result, "coverage", None)
    verification = _field(answer, "verification", None)
    if verification is None:
        verification = diagnostics.get("verification")
    structural = _field(result, "structural_changes", []) or []
    return {
        "calculations": jsonable(calculations),
        "facts": jsonable(_field(answer, "facts", []) or []),
        "coverage": jsonable(coverage) if isinstance(coverage, Mapping) else None,
        "structural_changes": jsonable(structural),
        "verification": jsonable(verification) if verification else None,
    }


def result_status(result: Any) -> str:
    status = _field(result, "status", "completed")
    if status == "failed":
        return "failed"
    answer_status = _field(_field(result, "answer"), "status", "abstain")
    if status == "abstained" or answer_status == "abstain":
        return "abstained"
    return "completed"


def result_message_fields(result: Any, metadata_for: Any = None) -> dict[str, Any]:
    """Терминальный ``AgentRunResult`` → поля сообщения и ``answer_json``."""
    answer = _field(result, "answer")
    citations = list(_field(answer, "citations", []) or [])
    evidence = [
        citation_to_evidence(
            citation,
            metadata_for(_field(citation, "chunk_id")) if metadata_for else None,
        )
        for citation in citations
    ]
    warnings = _unique_strings(
        list(_field(result, "warnings", []) or [])
        + list(_field(answer, "warnings", []) or [])
    )
    return {
        "content": str(_field(answer, "answer", "") or ""),
        "status": result_status(result),
        "evidence": evidence,
        "analysis": analysis_from_result(result),
        "effective_filters": filters_to_public(
            _field(result, "effective_filters")
        ),
        "warnings": warnings,
    }


def answer_json_from_fields(fields: Mapping[str, Any]) -> str:
    payload = {
        "evidence": fields.get("evidence", []),
        "analysis": fields.get("analysis"),
        "effective_filters": fields.get("effective_filters") or None,
        "warnings": fields.get("warnings", []),
    }
    return json.dumps(payload, ensure_ascii=False)


def _parse_json(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def message_from_stored(row: Mapping[str, Any]) -> dict[str, Any]:
    """Строка ``messages`` → публичный ``ChatMessage``."""
    answer = _parse_json(row.get("answer_json"))
    if not isinstance(answer, Mapping):
        answer = {}
    filters = _parse_json(row.get("filters_json"))
    if not isinstance(filters, Mapping):
        filters = answer.get("effective_filters")
        filters = filters if isinstance(filters, Mapping) else {}
    error = _parse_json(row.get("error_json"))
    analysis = answer.get("analysis")
    return {
        "id": row.get("id"),
        "research_id": row.get("research_id"),
        "role": row.get("role"),
        "content": row.get("content") or "",
        "created_at": row.get("created_at"),
        "status": row.get("status") or "failed",
        "request_id": row.get("request_id"),
        "effective_filters": dict(filters) if filters else None,
        "evidence": answer.get("evidence") if isinstance(
            answer.get("evidence"), list
        ) else [],
        "analysis": analysis if isinstance(analysis, Mapping) else None,
        "warnings": answer.get("warnings") if isinstance(
            answer.get("warnings"), list
        ) else [],
        "error": dict(error) if isinstance(error, Mapping) else None,
    }


def safe_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Только безопасные короткие аргументы; query/filters/chunk_id скрыты."""
    if not isinstance(arguments, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        name = str(key)
        if name not in _SAFE_TOOL_ARGUMENTS:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            safe[name] = value
        elif isinstance(value, str):
            safe[name] = value[:120]
    return safe


_RETRYABLE_ERROR_CODES = frozenset(
    {"agent_run_failed", "backend_unavailable", "internal_error", "interrupted"}
)


def agent_event_payload(
    event: Any,
    *,
    request_id: str,
    message_id: str,
) -> dict[str, Any] | None:
    """Одно ``AgentEvent`` → публичный конверт события §4.2."""
    kind = str(_field(event, "event_type", "") or "")
    data = _field(event, "data", {}) or {}
    if not isinstance(data, Mapping):
        data = {}

    if kind == "run_started":
        return {"type": kind, "request_id": request_id, "message_id": message_id}
    if kind == "analysis_completed":
        payload: dict[str, Any] = {"type": kind, "request_id": request_id}
        intent = data.get("intent")
        if isinstance(intent, str) and intent:
            payload["intent"] = intent
        payload["effective_filters"] = filters_to_public(
            data.get("effective_filters")
        )
        return payload
    if kind == "tool_started":
        payload = {
            "type": kind,
            "request_id": request_id,
            "tool_name": str(data.get("tool_name") or ""),
        }
        short = safe_tool_arguments(data.get("short_arguments"))
        if short:
            payload["short_arguments"] = short
        return payload
    if kind == "tool_completed":
        payload = {
            "type": kind,
            "request_id": request_id,
            "tool_name": str(data.get("tool_name") or ""),
            "status": str(data.get("status") or "unknown"),
        }
        count = data.get("result_count")
        if isinstance(count, int) and not isinstance(count, bool):
            payload["result_count"] = count
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            payload["duration_ms"] = round(float(duration), 3)
        return payload
    if kind == "answer_started":
        return {"type": kind, "request_id": request_id}
    if kind == "answer_delta":
        delta = data.get("delta")
        if not isinstance(delta, str):
            return None
        return {"type": kind, "request_id": request_id, "delta": delta}
    if kind == "evidence_updated":
        evidence = data.get("evidence")
        if not isinstance(evidence, list):
            return None
        return {
            "type": kind,
            "request_id": request_id,
            "evidence": jsonable(evidence),
        }
    if kind == "run_completed":
        return {
            "type": kind,
            "request_id": request_id,
            "answer_status": str(data.get("answer_status") or "answer"),
            "duration_ms": round(float(data.get("duration_ms") or 0.0), 3),
        }
    if kind == "run_failed":
        code = str(data.get("error_code") or "agent_run_failed")
        retryable = data.get("retryable")
        return {
            "type": kind,
            "request_id": request_id,
            "code": code,
            "message": str(
                data.get("user_message")
                or "Не удалось безопасно завершить обработку запроса."
            ),
            "retryable": (
                retryable
                if isinstance(retryable, bool)
                else code in _RETRYABLE_ERROR_CODES
            ),
        }
    if kind == "run_cancelled":
        return {"type": kind, "request_id": request_id}
    # Внутренние события backend'а не публикуются под чужим именем.
    return None


def error_info(code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"code": code, "message": message, "retryable": bool(retryable)}
