"""Публичный синхронный API AgentService и безопасные потоковые события.

``create_agent_service`` — точка входа для UI: фабрика корпуса/API импортируется
лениво, поэтому импорт этого модуля не читает env и не строит индексы.
"""

from __future__ import annotations

import time
import uuid
import inspect
import logging
import re
from dataclasses import asdict, is_dataclass, replace
from queue import Queue
from threading import Lock, Thread, local
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver

from src.agent.agent_graph import (
    MAX_SEARCH_CALLS,
    MAX_TOOL_CALLS,
    build_agent_graph,
)
from src.agent.smalltalk import smalltalk_reply
from src.agent.state import (
    AgentEvent,
    AgentRunResult,
    Answer,
    Citation,
    QueryPlan,
)
from src.agent.tools import AgentTools
from src.agent.verification import verify_answer
from src.engine.calculations import format_calculations, summarize_facts
from src.engine.prompts import format_structural_changes
from src.retrieval.hybrid_retriever import SearchFilters

HISTORY_TURNS = 10
HISTORY_CONTENT_LIMIT = 4000
#: Сколько подтверждённых citation предыдущего хода переносить в следующий.
#: Ограничение защищает контекст генератора от разрастания диалога.
MAX_CARRIED_EVIDENCE = 6

#: Навигационный структурный ответ без content evidence: цитаты могут быть
#: пустыми, но это обязано быть явно помечено.
STRUCTURAL_CLAIM_WARNING = "structural_claim_without_citation"
_HISTORY_SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+"),
     r"\1=[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
)


class Answerer(Protocol):
    def generate(
        self,
        question: str,
        evidence: Mapping[str, Any],
        evidence_manifest: set[str],
        *,
        repair: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Any: ...


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


#: Инструменты, которые считаются retrieval-вызовами для trace.
RETRIEVAL_TOOL_NAMES = frozenset({"search_reports", "get_entities"})


def _coverage_payload(value: Any) -> dict[str, Any] | None:
    """Сериализует CoverageReport через его ``to_dict`` или совместимый dict."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:  # noqa: BLE001 — diagnostics не должны ронять запуск
            data = None
        if isinstance(data, Mapping):
            return dict(data)
    if is_dataclass(value):
        return asdict(value)
    return None


def _filters_payload(value: Any) -> dict[str, Any] | None:
    """JSON-безопасное представление фильтров из ToolEvent.arguments."""
    if value is None:
        return None
    if isinstance(value, SearchFilters):
        data: Mapping[str, Any] = vars(value)
    elif isinstance(value, Mapping):
        data = value
    else:
        return None
    payload: dict[str, Any] = {}
    for key, item in data.items():
        if isinstance(item, (set, frozenset)):
            payload[str(key)] = sorted(item)
        elif isinstance(item, (list, tuple)):
            payload[str(key)] = list(item)
        else:
            payload[str(key)] = item
    return payload


def _trace_payload(value: Any) -> dict[str, Any] | None:
    """Приводит RetrievalTrace (объект/словарь/asdict) к контракту §4.3.

    ``to_dict`` предпочтителен: он добавляет ``counts``. Для совместимости
    принимается и ``asdict`` старого ToolEvent; тогда ``counts`` вычисляется из
    тех же плоских счётчиков, а не выдумывается.
    """
    if value is None:
        return None
    data: Mapping[str, Any] | None = None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            candidate = to_dict()
        except Exception:  # noqa: BLE001
            candidate = None
        if isinstance(candidate, Mapping):
            data = candidate
    if data is None and is_dataclass(value) and not isinstance(value, type):
        data = asdict(value)
    if data is None and isinstance(value, Mapping):
        data = value
    if not isinstance(data, Mapping):
        return None

    payload = dict(data)
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        counts = {
            "dense": payload.get("dense_candidates", 0),
            "bm25": payload.get("bm25_candidates", 0),
            "rrf": payload.get("fused_candidates", 0),
            "rerank": payload.get("rerank_candidates", 0),
            "returned": payload.get("returned", 0),
        }
    else:
        counts = dict(counts)
    if "graph_candidates" in payload and "graph" not in counts:
        # ``asdict(GraphRetrievalTrace)`` не содержит сводного блока ``graph``;
        # счётчик восстанавливается из того же поля, что и в ``to_dict``.
        counts["graph"] = payload.get("graph_candidates", 0)
    payload["counts"] = counts
    stages = payload.get("stages")
    if isinstance(stages, Mapping):
        payload["stages"] = {
            str(name): (
                None
                if items is None
                else [dict(item) for item in items if isinstance(item, Mapping)]
            )
            for name, items in stages.items()
        }
    else:
        # Недоступный этап — это отсутствие данных, а не синтетический список.
        payload["stages"] = {}
    for key in ("available", "fallback", "unavailable"):
        block = payload.get(key)
        payload[key] = dict(block) if isinstance(block, Mapping) else {}
    return payload


def _event_warnings(event: Any) -> list[str]:
    """Предупреждения одного tool call в стабильном порядке без дублей."""
    warnings: list[str] = []
    diagnostics = getattr(event, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        raw = diagnostics.get("warnings")
        if isinstance(raw, (list, tuple)):
            warnings.extend(str(item) for item in raw if str(item))
    warning = getattr(event, "warning", None)
    if warning:
        warnings.append(str(warning))
    return list(dict.fromkeys(warnings))


def _retrieval_calls(tool_events: list[Any]) -> list[dict[str, Any]]:
    """Собирает ``diagnostics["retrieval_calls"]`` по контракту §4.1/§4.3.

    ``call_index`` — порядковый номер retrieval-вызова (``search_reports`` или
    ``get_entities``). В вызов не попадают prompt, chain-of-thought и секреты:
    только аргументы инструмента, статус и сериализованный trace.
    """
    calls: list[dict[str, Any]] = []
    for event in tool_events:
        name = getattr(event, "tool_name", None)
        if name not in RETRIEVAL_TOOL_NAMES:
            continue
        arguments = getattr(event, "arguments", None)
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        trace = _trace_payload(getattr(event, "diagnostics", None))
        query = arguments.get("query")
        queries = arguments.get("queries")
        if isinstance(queries, (list, tuple)):
            queries = [str(item) for item in queries]
        elif isinstance(query, str) and name == "search_reports":
            queries = [query]
        else:
            queries = None
        call: dict[str, Any] = {
            "call_index": len(calls),
            "tool_name": name,
            "query": query if name == "search_reports" else None,
            "queries": queries if name == "search_reports" else None,
            "retrieval_mode": (
                arguments.get("retrieval_mode")
                or (trace or {}).get("retrieval_mode")
                or "hybrid"
            ),
            "top_k": arguments.get("top_k"),
            "filters": _filters_payload(arguments.get("filters")),
            "status": getattr(event, "status", None),
            "warnings": _event_warnings(event),
            "counts": {},
            "stages": {},
            "available": {},
            "fallback": {},
            "unavailable": {},
        }
        if name == "get_entities":
            call["entity_names"] = [
                str(item) for item in (arguments.get("entity_names") or [])
            ]
        if trace is not None:
            # Trace поля не перетирают идентичность вызова (query/top_k/filters).
            for key, item in trace.items():
                if key == "retrieval_mode":
                    continue
                call[key] = item
        calls.append(call)
    return calls


def _safe_warning(value: Any) -> str:
    normalized = str(value).casefold()
    for code in (
        "missing_year",
        "missing_metric",
        "no_evidence_for_period",
        "no_evidence",
        "low_confidence",
        "scope_mismatch",
        "graph_unavailable",
        "issuer_mismatch",
        "retrieval_fallback",
        "incomplete_period",
        "structure_unverified",
        "structure_unavailable",
        "structure_scope_limited",
        "structure_content_unverified",
        "structural_claim_without_citation",
        "conflicting_values",
        "unit_mismatch",
        "query_rewrite_fallback",
        "query_analysis_fallback",
        "verification_failed",
        "verification_warning",
        "verification_retry",
        "generation_unavailable",
        "citation_validation_failed",
    ):
        if code in normalized:
            return code
    return "answer_warning"


def _history_content(value: str) -> str:
    content = value[:HISTORY_CONTENT_LIMIT]
    for pattern, replacement in _HISTORY_SECRET_PATTERNS:
        content = pattern.sub(replacement, content)
    return content


def _coverage_field(coverage: Any, name: str) -> Any:
    """Читает поле ``CoverageReport`` или его dict-представления."""
    if isinstance(coverage, Mapping):
        return coverage.get(name)
    return getattr(coverage, name, None)


def _as_years(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted(
        {
            year
            for year in value
            if isinstance(year, int) and not isinstance(year, bool)
        }
    )


def _years_ru(years: list[int]) -> str:
    return ", ".join(str(year) for year in years)


def missing_evidence_answer(state: Mapping[str, Any]) -> Answer:
    """Abstain с причиной: нет отчёта за год или нет показателя в отчёте.

    ``spec_retrieval_engine.md`` §4.2/§6 требует отличать отсутствующий документ
    от отсутствующего показателя. Общий ``no_evidence`` скрывает эту разницу,
    поэтому причина выводится из coverage.
    """
    coverage = state.get("coverage")
    missing_years = _as_years(_coverage_field(coverage, "missing_years"))
    if not missing_years:
        missing_years = _as_years(state.get("missing_years"))
    if missing_years:
        years = _years_ru(missing_years)
        suffix = "год" if len(missing_years) == 1 else "годы"
        return Answer(
            "abstain",
            (
                f"Отчёт за {years} {suffix} отсутствует в доступном корпусе, "
                "поэтому подтвердить показатель по этому периоду нельзя."
            ),
            warnings=["no_evidence_for_period"],
        )

    cells = _coverage_field(coverage, "cells") or []
    not_found: dict[str, list[int]] = {}
    for cell in cells:
        if _coverage_field(cell, "status") != "not_found":
            continue
        year = _coverage_field(cell, "year")
        label = _coverage_field(cell, "metric_label") or _coverage_field(
            cell, "metric"
        )
        if (
            isinstance(year, int)
            and not isinstance(year, bool)
            and isinstance(label, str)
            and label.strip()
        ):
            not_found.setdefault(label.strip(), []).append(year)
    if not_found:
        parts = [
            f"«{label}» за {_years_ru(sorted(set(years)))}"
            for label, years in not_found.items()
        ]
        return Answer(
            "abstain",
            (
                "В доступных отчётах не найден показатель "
                + "; ".join(parts)
                + ". Значение не интерполируется."
            ),
            warnings=["no_evidence", "missing_metric"],
        )

    # Период не задан или coverage недоступен — прежний безопасный отказ.
    return Answer(
        "abstain",
        "В доступных материалах не найдено подтверждение для ответа.",
        warnings=["no_evidence"],
    )


#: Логгер структурных этапов агента. INFO по умолчанию не выводится без
#: настроенного logging, поэтому тесты остаются тихими.
logger = logging.getLogger(__name__)

#: Белый список полей структурного лога. Всё остальное (question/evidence/
#: chunk text/prompt/секреты) отбрасывается ещё до форматирования.
LOG_FIELD_NAMES = frozenset(
    {
        "event",
        "request_id",
        "status",
        "answer_status",
        "intent",
        "tool_sequence",
        "coverage_status",
        "verification_status",
        "duration_ms",
        "tool_calls",
        "search_calls",
        "warnings",
        "scenario",
        "error_code",
    }
)
_LOG_TEXT_LIMIT = 160
_LOG_ITEM_LIMIT = 16


def _log_scalar(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    text = str(value)[:_LOG_TEXT_LIMIT]
    for pattern, replacement in _HISTORY_SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def safe_log_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Оставляет только безопасные структурные поля без текста чанков/prompt.

    Списки (например, ``tool_sequence``) обрезаются по длине, значения —
    по длине строки; вложенные словари целиком отбрасываются, чтобы тело
    evidence не могло попасть в лог.
    """
    payload: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in LOG_FIELD_NAMES:
            continue
        if isinstance(value, Mapping):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            payload[key] = [
                _log_scalar(item) for item in list(value)[:_LOG_ITEM_LIMIT]
            ]
        else:
            payload[key] = _log_scalar(value)
    return payload


def _log_safe(fields: Mapping[str, Any]) -> None:
    """Пишет структурную запись; сбой логирования не влияет на ответ."""
    try:
        logger.info(
            str(fields.get("event", "agent_event")),
            extra={"agent": safe_log_fields(fields)},
        )
    except Exception:  # noqa: BLE001 — логи не должны ломать пользовательский ход
        return


def log_run_summary(result: AgentRunResult) -> None:
    """Логирует завершение хода: request_id, этапы, coverage и длительность."""
    diagnostics = result.diagnostics if isinstance(result.diagnostics, Mapping) else {}
    plan = diagnostics.get("query_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    coverage = diagnostics.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    verification = diagnostics.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    _log_safe(
        {
            "event": "agent_run_completed",
            "request_id": result.request_id,
            "status": result.status,
            "answer_status": result.answer.status,
            "intent": plan.get("intent"),
            "tool_sequence": [
                f"{event.tool_name}:{event.status}"
                for event in result.tool_events
            ],
            "coverage_status": coverage.get("status"),
            "verification_status": verification.get("status"),
            "duration_ms": round(
                float(result.timings.get("total_ms", 0.0) or 0.0), 1
            ),
            "tool_calls": len(result.tool_events),
            "search_calls": sum(
                1
                for event in result.tool_events
                if event.tool_name == "search_reports"
            ),
            "warnings": list(result.warnings),
        }
    )


def _answer_from(value: Any) -> Answer:
    if isinstance(value, Answer):
        return Answer(
            status=value.status,
            answer=value.answer,
            citations=list(value.citations),
            facts=list(value.facts),
            warnings=[_safe_warning(item) for item in value.warnings],
            calculations=list(value.calculations),
            verification=value.verification,
        )
    citations = []
    for item in _value(value, "citations", []) or []:
        citations.append(
            item if isinstance(item, Citation) else Citation(
                chunk_id=str(_value(item, "chunk_id", "")),
                doc_id=_value(item, "doc_id"),
                year=_value(item, "year"),
                issuer=_value(item, "issuer"),
                page_pdf=_value(item, "page_pdf"),
                page_label=_value(item, "page_label"),
                section_path=_value(item, "section_path"),
                quote=_value(item, "quote"),
            )
        )
    status = _value(value, "status", "abstain")
    if status not in {"answer", "abstain"}:
        status = "abstain"
    return Answer(
        status=status,
        answer=str(_value(value, "answer", "")),
        citations=citations,
        facts=list(_value(value, "facts", []) or []),
        warnings=[_safe_warning(item) for item in (_value(value, "warnings", []) or [])],
        calculations=list(_value(value, "calculations", []) or []),
        verification=_value(value, "verification"),
    )


class AgentService:
    """Оркестрирует переданный набор инструментов и необязательный генератор ответа.

    Движок поиска, модель и сетевой клиент сами не создаются. На каждый ход
    берётся свежее состояние; ключ checkpoint LangGraph содержит
    ``conversation_id``, чтобы изолировать беседы при наличии saver.
    Терминальное событие ``run_completed`` содержит полный ``AgentRunResult``
    в ``AgentEvent.result`` для отрисовки проверенных цитат и статуса.
    """

    def __init__(
        self,
        tools: AgentTools,
        *,
        answerer: Answerer | Any | None = None,
        router: Any = None,
        action_selector: Any = None,
        checkpointer: Any = None,
        documents_by_year: Mapping[int, list[str]] | None = None,
        structure_analyzer: Any = None,
        query_rewriter: Any = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_search_calls: int = MAX_SEARCH_CALLS,
        verification_enabled: bool = True,
    ) -> None:
        if max_tool_calls < 1 or max_search_calls < 1:
            raise ValueError("agent budgets must be positive")
        self.tools = tools
        self.answerer = answerer
        self.router = router
        self.action_selector = action_selector
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.verification_enabled = verification_enabled
        self.documents_by_year = {
            int(year): [str(doc_id) for doc_id in docs]
            for year, docs in (documents_by_year or {}).items()
        }
        self.structure_analyzer = structure_analyzer
        self.query_rewriter = query_rewriter
        self.max_tool_calls = max_tool_calls
        self.max_search_calls = max_search_calls
        self._history_lock = Lock()
        self._conversation_history: dict[str, list[dict[str, str]]] = {}
        #: Подтверждённые citation предыдущего хода. Follow-up («а каким он был
        #: в 2022 году?») должен видеть те же источники: история хранит только
        #: текст ответа, а цитировать текст истории нельзя.
        self._conversation_evidence: dict[str, dict[str, dict[str, Any]]] = {}
        # Контекст последней генерации хранится thread-local: граф одного
        # запроса исполняется в своём потоке, поэтому параллельные беседы не
        # перетирают диагностику друг друга.
        self._context_state = local()

    def _set_context_chunk_ids(self, value: list[str] | None) -> None:
        self._context_state.chunk_ids = value

    def context_chunk_ids(self) -> list[str] | None:
        """ID чанков, фактически упакованных генератором в последнем ходе потока."""
        value = getattr(self._context_state, "chunk_ids", None)
        return list(value) if isinstance(value, list) else None

    def _capture_context_chunk_ids(self) -> None:
        """Читает у генератора список упакованных чанков, если он его раскрывает."""
        candidate: Any = None
        try:
            candidate = getattr(self.answerer, "last_context_chunk_ids", None)
        except Exception:  # noqa: BLE001 — произвольный answerer не обязан его иметь
            candidate = None
        if isinstance(candidate, (list, tuple)) and all(
            isinstance(item, str) for item in candidate
        ):
            self._set_context_chunk_ids(list(candidate))
        else:
            self._set_context_chunk_ids(None)

    def _history_snapshot(self, conversation_id: str) -> list[dict[str, str]]:
        with self._history_lock:
            history = self._conversation_history.get(conversation_id, [])
            return [{"role": item["role"], "content": item["content"]} for item in history]

    def _previous_evidence(self, conversation_id: str) -> dict[str, dict[str, Any]]:
        with self._history_lock:
            evidence = self._conversation_evidence.get(conversation_id, {})
            return {chunk_id: dict(chunk) for chunk_id, chunk in evidence.items()}

    def _remember_evidence(
        self,
        conversation_id: str,
        evidence: Mapping[str, Any],
        answer: Answer,
    ) -> None:
        """Сохранить подтверждённые цитатами чанки для следующего хода."""
        cited = [
            citation.chunk_id
            for citation in answer.citations
            if isinstance(citation.chunk_id, str) and citation.chunk_id
        ]
        remembered: dict[str, dict[str, Any]] = {}
        for chunk_id in cited[:MAX_CARRIED_EVIDENCE]:
            chunk = evidence.get(chunk_id)
            if isinstance(chunk, Mapping):
                remembered[chunk_id] = dict(chunk)
        if not remembered:
            return
        with self._history_lock:
            self._conversation_evidence[conversation_id] = remembered

    def _record_turn(
        self, conversation_id: str, question: str, answer: Answer
    ) -> None:
        turn = [
            {"role": "user", "content": _history_content(question)},
            {
                "role": "assistant",
                "content": _history_content(answer.answer),
            },
        ]
        with self._history_lock:
            history = self._conversation_history.setdefault(conversation_id, [])
            history.extend(turn)
            del history[:-HISTORY_TURNS * 2]

    def ask(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: SearchFilters | None = None,
    ) -> AgentRunResult:
        events = list(self.stream(query, conversation_id=conversation_id, filters=filters))
        if events and events[-1].result is not None:
            return events[-1].result
        # Доставка внутренних событий не должна мешать результату: если граф
        # не смог отработать, возвращаем безопасный результат со статусом failed.
        request_id = events[0].request_id if events else str(uuid.uuid4())
        return AgentRunResult(
            request_id=request_id,
            conversation_id=conversation_id,
            answer=Answer("abstain", "Не удалось безопасно завершить обработку запроса."),
            tool_events=[],
            effective_filters=filters or SearchFilters(),
            warnings=["agent_run_failed"],
            timings={},
            status="failed",
        )

    def stream(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: SearchFilters | None = None,
    ) -> Iterator[AgentEvent]:
        request_id = str(uuid.uuid4())
        events: Queue[AgentEvent | None] = Queue()
        history = self._history_snapshot(conversation_id)
        carried_evidence = self._previous_evidence(conversation_id)

        def emit(kind: str, data: dict[str, Any]) -> None:
            events.put(AgentEvent(kind, request_id, data))

        def run() -> None:
            started = time.perf_counter()
            self._set_context_chunk_ids(None)
            _log_safe(
                {"event": "agent_run_started", "request_id": request_id}
            )
            events.put(AgentEvent("run_started", request_id, {}))
            greeting = smalltalk_reply(query)
            if greeting is not None:
                # Служебная реплика вне корпуса: retrieval и генератор не
                # запускаются, чтобы не выдавать «нет доказательств» на
                # «Привет!». Ход не пишется в историю диалога: он не содержит
                # фактов, на которые можно сослаться в следующем вопросе.
                timings = {"total_ms": (time.perf_counter() - started) * 1000}
                result = AgentRunResult(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    answer=Answer("answer", greeting),
                    tool_events=[],
                    effective_filters=filters or SearchFilters(),
                    warnings=[],
                    timings=timings,
                    status="completed",
                    diagnostics={"smalltalk": True},
                )
                log_run_summary(result)
                events.put(
                    AgentEvent(
                        "run_completed",
                        request_id,
                        {
                            "answer_status": "answer",
                            "duration_ms": timings["total_ms"],
                        },
                        result=result,
                    )
                )
                events.put(None)
                return
            initial = {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "question": query,
                "history": history,
                "filters": filters or SearchFilters(),
                "tool_events": [],
                "retrieval_results": [],
                # Подтверждённые источники прошлого хода доступны как
                # evidence и могут цитироваться: follow-up не теряет контекст.
                "evidence_manifest": set(carried_evidence),
                "search_evidence_manifest": set(carried_evidence),
                "graph_evidence_manifest": set(),
                "evidence": dict(carried_evidence),
                "answer": None,
                "clarification": None,
                "status": "running",
                "tool_calls_used": 0,
                "search_calls_used": 0,
                "answer_repairs_used": 0,
                "warnings": [],
                "timings": {},
                "unknown_action_fallback_used": False,
                "expected_years": [],
                "covered_years": [],
                "missing_years": [],
                "coverage": None,
                "rewritten_queries": [],
                "search_attempts": [],
                "available_documents": self.documents_by_year,
                "structure_documents": {},
                "structure_snapshots": {},
                "section_matches": [],
                "structural_changes": [],
                "content_changes": [],
                "structure_comparison_complete": False,
                "structure_content_checked": False,
            }
            state: dict[str, Any] = {}
            try:
                graph = build_agent_graph(
                    self.tools,
                    router=self.router,
                    action_selector=self.action_selector,
                    checkpointer=self.checkpointer,
                    on_event=emit,
                    answer_generator=self._generate_answer,
                    answer_validator=self._valid_answer,
                    structure_analyzer=self.structure_analyzer,
                    query_rewriter=self.query_rewriter,
                    max_tool_calls=self.max_tool_calls,
                    max_search_calls=self.max_search_calls,
                )
                config = {
                    "configurable": {"thread_id": f"{conversation_id}:{request_id}"}
                }
                state = graph.invoke(initial, config=config)
                answer = state["answer"]
                status = "abstained" if answer.status == "abstain" else "completed"
                timings = dict(state.get("timings", {}))
                timings["total_ms"] = (time.perf_counter() - started) * 1000
                tool_events = list(state.get("tool_events", []))
                plan = state.get("query_plan")
                context_ids = self.context_chunk_ids()
                diagnostics = {
                    "query_plan": (
                        asdict(plan)
                        if is_dataclass(plan)
                        else plan if isinstance(plan, Mapping) else None
                    ),
                    "search_attempts": list(
                        state.get("search_attempts", [])
                    ),
                    "retrieval_calls": _retrieval_calls(tool_events),
                    # Primary evidence: чанки, подтверждённые поиском/get_chunk.
                    "evidence_manifest": sorted(
                        state.get("evidence_manifest", set())
                    ),
                    # Graph-only свидетельства до подтверждения через get_chunk (P7).
                    "graph_evidence_manifest": sorted(
                        state.get("graph_evidence_manifest", set())
                    ),
                    "context_chunk_ids": context_ids,
                    "context_source": (
                        "context" if context_ids is not None
                        else "evidence_manifest"
                    ),
                    "coverage": _coverage_payload(state.get("coverage")),
                    "verification": answer.verification,
                    "structural_changes": list(
                        state.get("structural_changes", [])
                    ),
                }
                result = AgentRunResult(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    answer=answer,
                    tool_events=tool_events,
                    effective_filters=state.get(
                        "effective_filters", filters or SearchFilters()
                    ),
                    warnings=list(dict.fromkeys(state.get("warnings", []))),
                    timings=timings,
                    status=status,
                    structural_changes=list(
                        state.get("structural_changes", [])
                    ),
                    calculations=list(answer.calculations),
                    diagnostics=diagnostics,
                )
            except Exception:
                # Не допускаем утечки текста исключений, traceback и тел ответов.
                result = AgentRunResult(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    answer=Answer(
                        "abstain",
                        "Не удалось безопасно завершить обработку запроса.",
                        warnings=["agent_run_failed"],
                    ),
                    tool_events=[],
                    effective_filters=filters or SearchFilters(),
                    warnings=["agent_run_failed"],
                    timings={"total_ms": (time.perf_counter() - started) * 1000},
                    status="failed",
                )
            self._record_turn(conversation_id, query, result.answer)
            self._remember_evidence(
                conversation_id,
                state.get("evidence", {}),
                result.answer,
            )
            log_run_summary(result)
            events.put(
                AgentEvent(
                    "run_failed" if result.status == "failed" else "run_completed",
                    request_id,
                    {
                        "answer_status": result.answer.status,
                        "duration_ms": result.timings["total_ms"],
                        **(
                            {
                                "error_code": "agent_run_failed",
                                "user_message": "Не удалось безопасно завершить обработку запроса.",
                            }
                            if result.status == "failed"
                            else {}
                        ),
                    },
                    result=result,
                )
            )
            events.put(None)

        Thread(target=run, name=f"agent-{request_id}", daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event

    def _structural_answer(
        self, state: Mapping[str, Any], plan: QueryPlan
    ) -> tuple[Answer, list[str]] | None:
        """Навигационный ответ по diff разделов без content evidence.

        Возвращает ``None``, если это не структурный вопрос или изменений нет:
        тогда действует прежний abstain/no_evidence.
        """
        if plan.intent != "structural":
            return None
        changes = [
            item
            for item in (state.get("structural_changes") or [])
            if isinstance(item, Mapping)
            and item.get("change_type") not in {None, "stable"}
        ]
        if not changes:
            return None
        text = format_structural_changes(
            changes,
            missing_years=[
                year
                for year in (state.get("missing_years") or [])
                if isinstance(year, int) and not isinstance(year, bool)
            ],
        )
        if not text:
            return None
        warning = STRUCTURAL_CLAIM_WARNING
        answer = Answer("answer", text, warnings=[warning])
        if not self.verification_enabled:
            return answer, [warning]
        verification = verify_answer(
            answer,
            state.get("evidence") or {},
            plan,
            state.get("coverage"),
            structure_complete=bool(
                state.get("structure_comparison_complete", False)
            ),
        )
        if verification.status == "failed":
            # Навигационный ответ без цитат не может провалить проверку
            # цитат/чисел; если это случилось — безопасный отказ.
            return Answer(
                "abstain",
                "Ответ не прошёл проверку источников и чисел.",
                warnings=["verification_failed"],
            ), [warning, "verification_failed"]
        if verification.status == "warning":
            answer = replace(
                answer,
                warnings=list(
                    dict.fromkeys(
                        list(answer.warnings) + ["verification_warning"]
                    )
                ),
            )
        return replace(answer, verification=verification.to_dict()), [warning]

    def _generate_answer(self, state: dict[str, Any]) -> tuple[Answer, list[str]]:
        # Контекст прошлого вызова не должен протечь в диагностику текущего:
        # если генератор не будет вызван, останется fallback evidence_manifest.
        self._set_context_chunk_ids(None)
        plan = state.get("query_plan")
        if not isinstance(plan, QueryPlan):
            plan = QueryPlan()
        manifest = set(state.get("evidence_manifest", set()))
        evidence = dict(state.get("evidence", {}))
        warnings: list[str] = []
        if not manifest:
            # Structure-aware fallback: diff разделов — валидный ответ сам по
            # себе, даже если content-поиск ничего не нашёл. Без content
            # evidence цитаты не выдумываются, а ответ помечается warning'ом.
            structural = self._structural_answer(state, plan)
            if structural is not None:
                return structural
            if any(item == "graph_unavailable" for item in state.get("warnings", [])):
                warnings.append("graph_unavailable")
            answer = missing_evidence_answer(state)
            return (
                replace(
                    answer,
                    warnings=list(dict.fromkeys(answer.warnings + warnings)),
                ),
                warnings,
            )
        if self.answerer is None:
            return Answer(
                "abstain",
                "Ответ не сформирован: генератор ответа не настроен.",
                warnings=["generation_unavailable"],
            ), warnings

        def generate(
            *, repair: bool = False, diagnostics: Mapping[str, Any] | None = None
        ) -> Any:
            target = (
                self.answerer.generate
                if hasattr(self.answerer, "generate")
                else self.answerer
            )
            kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(target).parameters.values()
                supports_kwargs = any(
                    item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters
                )
                by_name = {item.name: item for item in parameters}
                if supports_kwargs or (
                    "repair" in by_name
                    and by_name["repair"].kind
                    != inspect.Parameter.POSITIONAL_ONLY
                ):
                    kwargs["repair"] = repair
                if repair and diagnostics is not None and (
                    supports_kwargs
                    or (
                        "diagnostics" in by_name
                        and by_name["diagnostics"].kind
                        != inspect.Parameter.POSITIONAL_ONLY
                    )
                ):
                    kwargs["diagnostics"] = dict(diagnostics)
                if supports_kwargs or (
                    "history" in by_name
                    and by_name["history"].kind != inspect.Parameter.POSITIONAL_ONLY
                ):
                    kwargs["history"] = [
                        {"role": item["role"], "content": item["content"]}
                        for item in state.get("history", [])
                    ]
                plan = state.get("query_plan") or QueryPlan()
                for key, value in (
                    ("answer_mode", plan.answer_mode),
                    (
                        "structural_changes",
                        list(state.get("structural_changes", [])),
                    ),
                    (
                        "section_matches",
                        list(state.get("section_matches", [])),
                    ),
                    ("missing_years", list(state.get("missing_years", []))),
                    ("coverage", state.get("coverage")),
                ):
                    if supports_kwargs or key in by_name:
                        kwargs[key] = value
            except (TypeError, ValueError):
                # Старые/простые callable по-прежнему получают только исходные
                # три позиционных аргумента.
                pass
            answer = target(state["question"], evidence, manifest, **kwargs)
            # Генератор мог упаковать только часть evidence из-за бюджета;
            # фиксируем именно фактически упакованные chunk_id.
            self._capture_context_chunk_ids()
            return answer

        try:
            answer = _answer_from(generate(repair=False))
        except Exception:
            return Answer(
                "abstain",
                "Ответ не удалось безопасно сформировать.",
                warnings=["generation_unavailable"],
            ), warnings
        if not self._valid_answer(answer, manifest):
            warnings.append("citation_validation_retry")
            diagnostics = self._answer_repair_diagnostics(answer, manifest)
            try:
                answer = _answer_from(generate(repair=True, diagnostics=diagnostics))
            except Exception:
                answer = Answer(
                    "abstain",
                    "Ответ не прошёл проверку источников.",
                    warnings=["citation_validation_failed"],
                )
            if not self._valid_answer(answer, manifest):
                return Answer(
                    "abstain",
                    "Ответ не прошёл проверку источников.",
                    warnings=["citation_validation_failed"],
                ), warnings

        def finalize(current: Answer) -> Answer:
            if current.status != "answer":
                return current
            calculations = summarize_facts(current.facts)
            if not calculations:
                return current
            rendered = format_calculations(calculations)
            text = current.answer.rstrip()
            if rendered:
                text = f"{text}\n\n{rendered}" if text else rendered
            return replace(
                current,
                answer=text,
                calculations=calculations,
            )

        answer = finalize(answer)
        if not self.verification_enabled:
            return answer, warnings
        coverage = state.get("coverage")
        structure_complete = bool(
            state.get("structure_comparison_complete", True)
        )
        verification = verify_answer(
            answer,
            evidence,
            plan,
            coverage,
            structure_complete=structure_complete,
        )
        if verification.status == "failed":
            warnings.append("verification_retry")
            try:
                repaired = _answer_from(
                    generate(
                        repair=True,
                        diagnostics=verification.to_dict(),
                    )
                )
            except Exception:
                repaired = Answer(
                    "abstain",
                    "Ответ не прошёл проверку.",
                    warnings=["verification_failed"],
                )
            if self._valid_answer(repaired, manifest):
                candidate = finalize(repaired)
                candidate_verification = verify_answer(
                    candidate,
                    evidence,
                    plan,
                    coverage,
                    structure_complete=structure_complete,
                )
                if candidate_verification.status == "failed":
                    answer = Answer(
                        "abstain",
                        "Ответ не прошёл проверку источников и чисел.",
                        warnings=["verification_failed"],
                        verification=candidate_verification.to_dict(),
                    )
                    verification = candidate_verification
                else:
                    answer = candidate
                    verification = candidate_verification
            else:
                answer = Answer(
                    "abstain",
                    "Ответ не прошёл проверку источников и чисел.",
                    warnings=["verification_failed"],
                )
        answer = replace(answer, verification=verification.to_dict())
        if verification.status == "warning":
            answer = replace(
                answer,
                warnings=list(
                    dict.fromkeys(
                        list(answer.warnings) + ["verification_warning"]
                    )
                ),
            )
        return answer, warnings

    @classmethod
    def _answer_repair_diagnostics(
        cls, answer: Answer, manifest: set[str]
    ) -> dict[str, Any]:
        return {
            "reason": "answer_schema_or_citation_validation_failed",
            "invalid_citation_chunk_ids": sorted(
                {
                    str(citation.chunk_id)
                    for citation in answer.citations
                    if isinstance(citation.chunk_id, str)
                    and (
                        not citation.chunk_id
                        or citation.chunk_id not in manifest
                    )
                }
            ),
            "evidence_chunk_count": len(manifest),
        }

    @staticmethod
    def _valid_answer(answer: Answer, manifest: set[str]) -> bool:
        if answer.status == "abstain":
            return not answer.citations
        if not answer.answer.strip():
            return False
        if not answer.citations:
            # Навигационный структурный ответ по diff разделов: цитаты не
            # выдумываются, но пустой citations допустим только с явным
            # маркером и без фактов.
            return (
                STRUCTURAL_CLAIM_WARNING in answer.warnings
                and not answer.facts
            )
        for citation in answer.citations:
            if not citation.chunk_id or citation.chunk_id not in manifest:
                return False
            if citation.year is not None and (
                isinstance(citation.year, bool) or not isinstance(citation.year, int)
            ):
                return False
            if citation.page_pdf is not None and (
                isinstance(citation.page_pdf, bool)
                or not isinstance(citation.page_pdf, int)
                or citation.page_pdf < 1
            ):
                return False
            if citation.page_label is not None and not isinstance(
                citation.page_label, str
            ):
                return False
            if any(
                field is not None and not isinstance(field, str)
                for field in (
                    citation.doc_id,
                    citation.issuer,
                    citation.section_path,
                    citation.quote,
                )
            ):
                return False
        cited = {citation.chunk_id for citation in answer.citations}
        for fact in answer.facts:
            ids = fact.get("chunk_id") if isinstance(fact, Mapping) else None
            ids = ids if isinstance(ids, list) else [ids]
            if not ids or any(chunk_id not in cited for chunk_id in ids):
                return False
        return True


def create_agent_service(**kwargs: Any) -> "AgentService":
    """Собрать готовый сервис из локального корпуса (ленивый импорт фабрики).

    Args:
        **kwargs: Параметры :func:`src.agent.factory.create_agent_service`.

    Returns:
        Настроенный ``AgentService``.

    Raises:
        RuntimeError: корпус не собран или не задан ключ API.
    """
    from src.agent.factory import create_agent_service as build

    return build(**kwargs)
