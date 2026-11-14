"""Initial Streamlit chat UI for the ReportAgent service contract."""

from __future__ import annotations

import importlib
import re
import uuid
from collections.abc import Mapping
from typing import Any

try:
    from .citations import render_citations
except ImportError:  # Streamlit may execute this file directly.
    from citations import render_citations


AVAILABLE_YEARS = (2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024)
AVAILABLE_ISSUERS = ("ПАО Сбербанк", "БПС-Сбербанк (Беларусь)")
AVAILABLE_DOC_IDS = tuple(f"sber_{year}" for year in AVAILABLE_YEARS)
_ALLOWED_TOOLS = {"search_reports", "get_structure_tree", "get_entities", "get_chunk"}
_SECRET_KEY = re.compile(r"(api.?key|secret|token|password|credential)", re.IGNORECASE)


def new_conversation_id() -> str:
    """Create an opaque, unique identifier for an isolated conversation."""
    return str(uuid.uuid4())


def normalize_filters(
    years: Any = None,
    issuers: Any = None,
    doc_ids: Any = None,
) -> dict[str, list[Any]]:
    """Normalize visible UI selections into the AgentService filters mapping."""
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
    """Build SearchFilters via an injected project model factory when available.

    The fallback mapping uses the exact field names from spec_retrieval_engine.md,
    allowing headless fakes before the backend model has landed.
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
    """Read an event field directly first, then from its data payload."""
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
    """Return the terminal AgentRunResult without passing it through safe summaries."""
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
    """Bound text and redact common credential-shaped values."""
    if not isinstance(value, str):
        return ""
    value = re.sub(r"(?i)(api[_ -]?key|token|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", value)
    return value[:limit]


def safe_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Keep only short, non-secret tool arguments suitable for a visible trace."""
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
    """Convert one AgentEvent into a small UI-safe summary."""
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
    """Load the one documented integration hook, if the agent package is present."""
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
            # Do not expose backend exception text, which may contain sensitive values.
            return None
    return None


def _answer_payload(value: Any) -> Any:
    """Unwrap only conventional result/answer containers from the stated contract."""
    if value is None:
        return None
    nested = _get(value, "result")
    if nested is not None:
        value = nested
    # An Answer itself has status/citations. Preserve that object rather than
    # unwrapping its `answer` text field.
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
        failed = False
        completed = False
        status_context = st.status("Обработка запроса…", expanded=True)
        try:
            events = stream_method(query, conversation_id=conversation_id, filters=filters or None)
            def visible_deltas():
                nonlocal streamed_text, final_answer, failed, completed, request_id
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

            # Streamlit owns incremental rendering. The fallback keeps the same
            # event semantics for small injected test doubles/older installations.
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
    return {
        "answer": _answer_payload(result),
        "tool_trace": _ask_trace(result),
        "request_id": _get(result, "request_id"),
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
    """Render the app; dependencies are injectable for headless tests."""
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

    state = st.session_state
    if "conversation_id" not in state:
        state.conversation_id = new_conversation_id()
    if "messages" not in state:
        state.messages = []

    with st.sidebar:
        st.subheader("Фильтры документов")
        years = st.multiselect("Годы", AVAILABLE_YEARS, default=[])
        issuers = st.multiselect("Issuer", AVAILABLE_ISSUERS, default=[])
        doc_ids = st.multiselect("Документы", AVAILABLE_DOC_IDS, default=[])
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
        # Backend conversation memory is isolated by conversation_id and owned by
        # AgentService; the UI stores only the visible chat transcript.


if __name__ == "__main__":
    main()
