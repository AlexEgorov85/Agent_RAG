"""Публичный синхронный API AgentService и безопасные потоковые события.

``create_agent_service`` — точка входа для UI: фабрика корпуса/API импортируется
лениво, поэтому импорт этого модуля не читает env и не строит индексы.
"""

from __future__ import annotations

import time
import uuid
import inspect
import re
from dataclasses import replace
from queue import Queue
from threading import Lock, Thread
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver

from src.agent.agent_graph import build_agent_graph
from src.agent.state import (
    AgentEvent,
    AgentRunResult,
    Answer,
    Citation,
    QueryPlan,
)
from src.agent.tools import AgentTools
from src.engine.calculations import format_calculations, summarize_facts
from src.retrieval.hybrid_retriever import SearchFilters

HISTORY_TURNS = 10
HISTORY_CONTENT_LIMIT = 4000
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


def _safe_warning(value: Any) -> str:
    normalized = str(value).casefold()
    for code in (
        "missing_year",
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
        "conflicting_values",
        "unit_mismatch",
        "query_rewrite_fallback",
        "query_analysis_fallback",
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


def _answer_from(value: Any) -> Answer:
    if isinstance(value, Answer):
        return Answer(
            status=value.status,
            answer=value.answer,
            citations=list(value.citations),
            facts=list(value.facts),
            warnings=[_safe_warning(item) for item in value.warnings],
            calculations=list(value.calculations),
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
    ) -> None:
        self.tools = tools
        self.answerer = answerer
        self.router = router
        self.action_selector = action_selector
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.documents_by_year = {
            int(year): [str(doc_id) for doc_id in docs]
            for year, docs in (documents_by_year or {}).items()
        }
        self.structure_analyzer = structure_analyzer
        self.query_rewriter = query_rewriter
        self._history_lock = Lock()
        self._conversation_history: dict[str, list[dict[str, str]]] = {}

    def _history_snapshot(self, conversation_id: str) -> list[dict[str, str]]:
        with self._history_lock:
            history = self._conversation_history.get(conversation_id, [])
            return [{"role": item["role"], "content": item["content"]} for item in history]

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

        def emit(kind: str, data: dict[str, Any]) -> None:
            events.put(AgentEvent(kind, request_id, data))

        def run() -> None:
            started = time.perf_counter()
            events.put(AgentEvent("run_started", request_id, {}))
            initial = {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "question": query,
                "history": history,
                "filters": filters or SearchFilters(),
                "tool_events": [],
                "retrieval_results": [],
                "evidence_manifest": set(),
                "search_evidence_manifest": set(),
                "graph_evidence_manifest": set(),
                "evidence": {},
                "answer": None,
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
                )
                config = {
                    "configurable": {"thread_id": f"{conversation_id}:{request_id}"}
                }
                state = graph.invoke(initial, config=config)
                answer = state["answer"]
                status = "abstained" if answer.status == "abstain" else "completed"
                timings = dict(state.get("timings", {}))
                timings["total_ms"] = (time.perf_counter() - started) * 1000
                result = AgentRunResult(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    answer=answer,
                    tool_events=list(state.get("tool_events", [])),
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
                    retrieval_diagnostics=[
                        event.diagnostics
                        for event in state.get("tool_events", [])
                        if event.diagnostics
                    ],
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

    def _generate_answer(self, state: dict[str, Any]) -> tuple[Answer, list[str]]:
        manifest = set(state.get("evidence_manifest", set()))
        evidence = dict(state.get("evidence", {}))
        warnings: list[str] = []
        if not manifest:
            reason = "no_evidence"
            if any(item == "graph_unavailable" for item in state.get("warnings", [])):
                warnings.append("graph_unavailable")
            return Answer(
                "abstain",
                "В доступных материалах не найдено подтверждение для ответа.",
                warnings=[reason],
            ), warnings
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
                ):
                    if supports_kwargs or key in by_name:
                        kwargs[key] = value
            except (TypeError, ValueError):
                # Старые/простые callable по-прежнему получают только исходные
                # три позиционных аргумента.
                pass
            return target(state["question"], evidence, manifest, **kwargs)

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

        if answer.status == "answer":
            calculations = summarize_facts(answer.facts)
            if calculations:
                rendered = format_calculations(calculations)
                text = answer.answer.rstrip()
                if rendered:
                    text = f"{text}\n\n{rendered}" if text else rendered
                answer = replace(
                    answer,
                    answer=text,
                    calculations=calculations,
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
        if not answer.answer.strip() or not answer.citations:
            return False
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
