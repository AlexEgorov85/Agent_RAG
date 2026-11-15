"""Нормализация наблюдаемой трассы агента без prompt и chain-of-thought.

Trace собирается только из публичного ``AgentRunResult``: ``answer``,
``tool_events``, ``effective_filters`` и ``diagnostics``. Отдельные поля
результата не являются вторым источником истины. Если этап retrieval не
измерялся, в trace остаётся ``null``/``available=false``, а не выдуманный хит.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

TRACE_SCHEMA_VERSION = "1.1"
MAX_TRACE_BYTES = 5 * 1024 * 1024
MAX_QUOTE_CHARS = 500
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "headers",
        "password",
        "secret",
        "token",
        "access_token",
        "prompt",
        "system_prompt",
        "chain_of_thought",
        "reasoning",
        "messages",
    }
)
_SECRET_TEXT_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=]\s*\S+|"
    r"access[_-]?token\s*[:=]\s*\S+|bearer\s+[A-Za-z0-9._~+/=-]{8,})"
)
_FILTER_FIELDS = (
    "years",
    "year_from",
    "year_to",
    "report_periods",
    "issuers",
    "kinds",
    "section_path_prefix",
    "breadcrumb_contains",
    "max_section_level",
    "allowed_doc_ids",
)


def utc_now() -> str:
    """ISO 8601 UTC без микросекунд — как в контракте trace."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_value(value: Any) -> Any:
    """Привести dataclass/set/Path к JSON-совместимому виду."""
    if is_dataclass(value):
        return _redact(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _SECRET_KEYS or any(
                marker in normalized for marker in ("api_key", "secret", "password")
            ):
                cleaned[str(key)] = "[REDACTED]"
            else:
                cleaned[str(key)] = _redact(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_TEXT_RE.sub("[REDACTED]", value)
    return value


def normalize_filters(filters: Any) -> dict[str, Any]:
    """Нормализовать ``SearchFilters``/dict в JSON-безопасный словарь."""
    if filters is None:
        return {}
    values: dict[str, Any] = {}
    for name in _FILTER_FIELDS:
        value = _value(filters, name, None)
        if value is None:
            continue
        if isinstance(value, set):
            value = sorted(value)
        elif isinstance(value, (tuple, list)):
            value = list(value)
        if value in ([], "", set()):
            continue
        values[name] = _json_value(value)
    return values


def _citation(item: Any) -> dict[str, Any]:
    quote = _value(item, "quote")
    if isinstance(quote, str):
        quote = _SECRET_TEXT_RE.sub("[REDACTED]", quote)
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS]
    return {
        "chunk_id": _value(item, "chunk_id"),
        "doc_id": _value(item, "doc_id"),
        "year": _value(item, "year"),
        "issuer": _value(item, "issuer"),
        "page_pdf": _value(item, "page_pdf"),
        "page_label": _value(item, "page_label"),
        "section_path": _value(item, "section_path"),
        "quote": quote,
    }


def _answer(answer: Any) -> dict[str, Any]:
    if answer is None:
        return {
            "status": None,
            "text": None,
            "citations": [],
            "facts": [],
            "warnings": [],
            "calculations": [],
            "verification": None,
        }
    return {
        "status": _value(answer, "status"),
        "text": _value(answer, "answer", ""),
        "citations": [_citation(item) for item in (_value(answer, "citations", []) or [])],
        "facts": _json_value(list(_value(answer, "facts", []) or [])),
        "warnings": [str(item) for item in (_value(answer, "warnings", []) or [])],
        "calculations": _json_value(list(_value(answer, "calculations", []) or [])),
        "verification": _redact(_json_value(_value(answer, "verification"))),
    }


def _tool_events(result: Any) -> list[dict[str, Any]]:
    events = []
    for event in _value(result, "tool_events", []) or []:
        arguments = _value(event, "arguments", {}) or {}
        if isinstance(arguments, Mapping) and "filters" in arguments:
            arguments = dict(arguments)
            arguments["filters"] = normalize_filters(arguments.get("filters"))
        events.append(
            {
                "tool_name": _value(event, "tool_name"),
                "arguments": _redact(_json_value(arguments)),
                "status": _value(event, "status"),
                "result_count": _value(event, "result_count", 0),
                "duration_ms": _value(event, "duration_ms", 0.0),
                "warning": _value(event, "warning"),
            }
        )
    return events


def _retrieval_calls(result: Any, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Собирает retrieval-calls из ``diagnostics["retrieval_calls"]``.

    ``raw_calls`` уже содержат идентичность вызова (query/queries/top_k/filters),
    поэтому они — источник истины; событие инструмента используется только как
    fallback и сопоставляется по ``tool_name`` + порядку. Иначе появление
    ``get_entities`` между двумя ``search_reports`` сдвигало бы индексы и
    подменяло запросы.
    """
    diagnostics = _value(result, "diagnostics", {}) or {}
    raw_calls = diagnostics.get("retrieval_calls", []) if isinstance(diagnostics, Mapping) else []
    if not isinstance(raw_calls, list):
        return []
    events_by_tool: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        name = event.get("tool_name")
        if isinstance(name, str):
            events_by_tool.setdefault(name, []).append(event)
    positions: dict[str, int] = {}
    calls: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, Mapping):
            continue
        tool_name = raw.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = "search_reports"
        position = positions.get(tool_name, 0)
        positions[tool_name] = position + 1
        tool_events = events_by_tool.get(tool_name, [])
        event = tool_events[position] if position < len(tool_events) else None
        arguments = event.get("arguments", {}) if event is not None else {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        raw_filters = raw.get("filters")
        filters = (
            raw_filters
            if isinstance(raw_filters, Mapping)
            else arguments.get("filters")
        )
        query = raw.get("query")
        if not isinstance(query, str):
            fallback_query = arguments.get("query")
            query = fallback_query if isinstance(fallback_query, str) else None
        raw_queries = raw.get("queries")
        if isinstance(raw_queries, (list, tuple)):
            queries = [str(item) for item in raw_queries]
        else:
            fallback_queries = arguments.get("queries")
            if isinstance(fallback_queries, (list, tuple)):
                queries = [str(item) for item in fallback_queries]
            elif query:
                queries = [query]
            else:
                queries = []
        top_k = raw.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            fallback_top_k = arguments.get("top_k")
            top_k = fallback_top_k if isinstance(fallback_top_k, int) else None
        retrieval_mode = raw.get("retrieval_mode") or (
            arguments.get("retrieval_mode")
            if isinstance(arguments, Mapping)
            else None
        )
        call: dict[str, Any] = {
            "call_index": index,
            "tool_name": tool_name,
            "query": query if tool_name == "search_reports" else None,
            "queries": queries if tool_name == "search_reports" else None,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "filters": normalize_filters(filters),
            "status": (
                raw.get("status")
                or (event.get("status") if event is not None else None)
                or "completed"
            ),
            "warnings": (
                [str(item) for item in (raw.get("warnings") or [])]
                if isinstance(raw.get("warnings"), list)
                else []
            ),
            "counts": _redact(_json_value(raw.get("counts", {}))),
            "stages": {
                str(name): (
                    None
                    if items is None
                    else [_json_value(item) for item in items]
                )
                for name, items in (
                    raw.get("stages")
                    if isinstance(raw.get("stages"), Mapping)
                    else {}
                ).items()
            },
            "available": _redact(_json_value(raw.get("available", {}))),
            "fallback": _redact(_json_value(raw.get("fallback", {}))),
            "unavailable": _redact(_json_value(raw.get("unavailable", {}))),
        }
        if tool_name == "get_entities":
            names = raw.get("entity_names")
            if not isinstance(names, (list, tuple)):
                fallback_names = arguments.get("entity_names")
                names = (
                    fallback_names
                    if isinstance(fallback_names, (list, tuple))
                    else []
                )
            call["entity_names"] = [str(item) for item in names]
        calls.append(call)
    return calls


def _years(diagnostics: Any, key: str) -> list[int] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    coverage = diagnostics.get("coverage")
    if isinstance(coverage, Mapping) and key in coverage:
        value = coverage.get(key)
    else:
        value = diagnostics.get(key)
    if value is None:
        return None
    return sorted(int(year) for year in value if isinstance(year, int))


def build_turn_trace(
    result: Any,
    *,
    question: str,
    turn_index: int = 0,
    status: str | None = None,
    errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Собрать один turn trace из ``AgentRunResult``."""
    diagnostics = _value(result, "diagnostics", {}) or {}
    events = _tool_events(result)
    manifest = (
        diagnostics.get("evidence_manifest", [])
        if isinstance(diagnostics, Mapping)
        else []
    )
    if not isinstance(manifest, list):
        manifest = []
    context = (
        diagnostics.get("context_chunk_ids")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if context is None:
        context_ids = list(manifest)
        context_source = "evidence_manifest"
    else:
        context_ids = [str(item) for item in context]
        context_source = (
            diagnostics.get("context_source", "context")
            if isinstance(diagnostics, Mapping)
            else "context"
        )
    request_id = _value(result, "request_id")
    turn_status = status or _value(result, "status", "completed")
    turn_errors = list(errors or [])
    if turn_status == "failed" and not turn_errors:
        turn_errors.append(
            {"code": "agent_run_failed", "message": "Агент завершил turn со статусом failed"}
        )
    timings = _json_value(dict(_value(result, "timings", {}) or {}))
    return {
        "turn_index": turn_index,
        "question": question,
        "request_id": request_id,
        "status": turn_status,
        "query_plan": _redact(_json_value(
            diagnostics.get("query_plan") if isinstance(diagnostics, Mapping) else None
        )),
        "effective_filters": normalize_filters(_value(result, "effective_filters", None)),
        "tool_events": events,
        "search_attempts": _redact(_json_value(
            diagnostics.get("search_attempts", [])
            if isinstance(diagnostics, Mapping)
            else []
        )),
        "retrieval_calls": _retrieval_calls(result, events),
        "evidence_manifest": [str(item) for item in manifest],
        "context_chunk_ids": context_ids,
        "context_source": context_source,
        "covered_years": _years(diagnostics, "covered_years"),
        "missing_years": _years(diagnostics, "missing_years"),
        "answer": _answer(_value(result, "answer")),
        "run_warnings": [str(item) for item in (_value(result, "warnings", []) or [])],
        "structural_changes": _json_value(
            _value(result, "structural_changes", []) or []
        ),
        "timings": timings,
        "errors": turn_errors,
    }


def build_trace(
    *,
    run_id: str,
    test: Mapping[str, Any],
    turns: list[dict[str, Any]],
    retrieval_mode: str,
    attempt_index: int,
    conversation_id: str,
    started_at: str,
    duration_ms: float,
    status: str,
    git_sha: str,
    git_dirty: bool,
    dataset_sha256: str,
    corpus_sha256: str | None,
    corpus_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
    errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Собрать trace одного результата ``(test_id, mode, attempt)``."""
    all_errors = list(errors or [])
    trace = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "test_id": test.get("id"),
        "retrieval_mode": retrieval_mode,
        "attempt_index": attempt_index,
        "conversation_id": conversation_id,
        "started_at": started_at,
        "duration_ms": duration_ms,
        "status": status,
        "git_sha": git_sha,
        "git_dirty": bool(git_dirty),
        "dataset_sha256": dataset_sha256,
        "corpus_sha256": corpus_sha256,
        "corpus_fingerprint": corpus_fingerprint,
        "config_fingerprint": config_fingerprint,
        "turns": turns,
        "errors": all_errors,
    }
    return truncate_trace(trace)


def truncate_trace(
    trace: dict[str, Any], *, max_bytes: int = MAX_TRACE_BYTES
) -> dict[str, Any]:
    """Ограничить размер trace, не теряя факт обрезки.

    При превышении лимита списки этапов усекаются, а в trace ставится
    ``truncated=true``. Полные stage lists остаются в ``retrieval.csv``.
    """
    encoded = json.dumps(trace, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) <= max_bytes:
        trace.setdefault("truncated", False)
        return trace
    for turn in trace.get("turns", []) or []:
        for call in turn.get("retrieval_calls", []) or []:
            stages = call.get("stages")
            if not isinstance(stages, Mapping):
                continue
            for name, items in list(stages.items()):
                if isinstance(items, list) and len(items) > 100:
                    stages[name] = items[:100]
        answer = turn.get("answer")
        if isinstance(answer, Mapping) and isinstance(answer.get("text"), str):
            answer["text"] = answer["text"][:20_000]
    trace["truncated"] = True
    return trace


def trace_filename(test_id: str, retrieval_mode: str, attempt_index: int = 0) -> str:
    """Имя trace по контракту §6.1."""
    if attempt_index:
        return f"{test_id}.{retrieval_mode}.attempt_{attempt_index}.json"
    return f"{test_id}.{retrieval_mode}.json"


def trace_path(test_id: str, retrieval_mode: str, attempt_index: int = 0) -> str:
    """Относительная ссылка для ``cases.jsonl``."""
    return f"traces/{trace_filename(test_id, retrieval_mode, attempt_index)}"
