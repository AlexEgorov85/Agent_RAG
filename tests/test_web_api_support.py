"""Общие fake-объекты для офлайн-тестов HTTP adapter'а.

Сеть, LLM и реальный корпус не используются. Fake ``AgentService`` повторяет
контракт ``stream``: события + терминальный ``AgentRunResult``.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from fastapi.testclient import TestClient

from src.agent.state import AgentEvent, AgentRunResult, Answer, Citation, ToolEvent
from src.api.app import create_app
from src.api.config import ApiConfig
from src.api.services import CorpusInfo, Services, StaticAgentProvider
from src.retrieval.hybrid_retriever import SearchFilters

CLIENT_MESSAGE_ID = "4d1c6c1e-6f1b-4d1f-9f1e-1234567890ab"


def make_config(tmp_path: Path, **overrides: Any) -> ApiConfig:
    values: dict[str, Any] = {
        "db_path": tmp_path / "api.sqlite3",
        "source_registry_path": tmp_path / "missing_registry.json",
        "llm_lock_path": tmp_path / "llm.lock",
        "heartbeat_seconds": 0.05,
        "replay_buffer_events": 100,
        "replay_ttl_seconds": 600,
        "page_size_default": 50,
        "page_size_max": 200,
        "rate_limit_per_minute": 10_000,
        "corpus_name": "Test corpus",
    }
    values.update(overrides)
    return ApiConfig(**values)


def default_corpus() -> CorpusInfo:
    return CorpusInfo(
        name="Test corpus",
        years=[2023, 2024],
        documents=[
            {
                "doc_id": "sber_2023",
                "year": 2023,
                "issuer": "ПАО Сбербанк",
                "title": None,
            },
            {
                "doc_id": "sber_2024",
                "year": 2024,
                "issuer": "ПАО Сбербанк",
                "title": None,
            },
        ],
        issuers=["ПАО Сбербанк"],
        years_available=True,
    )


def make_citation(
    chunk_id: str = "sber_2023:00001:c0:aaa",
    *,
    doc_id: str = "sber_2023",
    year: int = 2023,
    issuer: str = "ПАО Сбербанк",
    page_pdf: int = 12,
    page_label: str = "12",
    section_path: str = "Стратегия/Ключевые показатели",
    quote: str = "Фрагмент отчёта",
) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        doc_id=doc_id,
        year=year,
        issuer=issuer,
        page_pdf=page_pdf,
        page_label=page_label,
        section_path=section_path,
        quote=quote,
    )


def make_result(
    *,
    citations: list[Citation] | None = None,
    answer_text: str = "Ответ, подтверждённый источником [1].",
    answer_status: str = "answer",
    status: str = "completed",
    warnings: list[str] | None = None,
    answer_warnings: list[str] | None = None,
    facts: list[dict[str, Any]] | None = None,
    calculations: list[dict[str, Any]] | None = None,
    verification: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    structural_changes: list[dict[str, Any]] | None = None,
    effective_filters: SearchFilters | None = None,
) -> AgentRunResult:
    citation_list = citations if citations is not None else [make_citation()]
    if facts is None:
        facts = (
            [
                {
                    "metric": "revenue",
                    "year": 2023,
                    "value": 100,
                    "unit": "млрд руб.",
                    "chunk_id": citation_list[0].chunk_id,
                }
            ]
            if citation_list
            else []
        )
    calculations = calculations if calculations is not None else [
        {"metric": "revenue", "direction": "increase", "absolute_change": 10}
    ]
    answer = Answer(
        answer_status,  # type: ignore[arg-type]
        answer_text,
        citation_list,
        facts=facts,
        warnings=answer_warnings or [],
        calculations=calculations,
        verification=verification,
    )
    return AgentRunResult(
        request_id="backend-request",
        conversation_id="backend-conversation",
        answer=answer,
        tool_events=[
            ToolEvent("search_reports", {"query": "[redacted]"}, "completed", 1, 5.0)
        ],
        effective_filters=effective_filters or SearchFilters(years={2023}),
        warnings=warnings or [],
        timings={"total_ms": 12.5},
        status=status,  # type: ignore[arg-type]
        structural_changes=structural_changes or [],
        calculations=calculations,
        diagnostics={
            "coverage": coverage,
            "verification": verification,
            "retrieval_calls": [],
        },
    )


class FakeChunkBackend:
    def __init__(self, chunks: dict[str, dict[str, Any]] | None = None) -> None:
        self.chunks = dict(chunks or {})
        self.requested: list[str] = []

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        self.requested.append(chunk_id)
        if chunk_id not in self.chunks:
            raise KeyError(chunk_id)
        return dict(self.chunks[chunk_id])


class FakeAgentService:
    """Немедленный поток событий; считает вызовы ``stream``."""

    def __init__(
        self,
        result: AgentRunResult | None = None,
        *,
        chunk_backend: FakeChunkBackend | None = None,
        block: bool = False,
    ) -> None:
        self.result = result or make_result()
        self.calls: list[dict[str, Any]] = []
        self.tools = SimpleNamespace(chunk_backend=chunk_backend)
        self.documents_by_year = {2023: ["sber_2023"], 2024: ["sber_2024"]}
        self._started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.cancel_calls: list[str] = []
        self.block = block
        if not block:
            self.release.set()

    @property
    def started(self) -> threading.Event:
        return self._started

    def _events(self, request_id: str) -> Iterator[AgentEvent]:
        yield AgentEvent("run_started", request_id, {})
        yield AgentEvent(
            "analysis_completed",
            request_id,
            {"intent": "factoid", "effective_filters": SearchFilters(years={2023})},
        )
        yield AgentEvent(
            "tool_started",
            request_id,
            {
                "tool_name": "search_reports",
                "short_arguments": {
                    "query": "секретный поисковый запрос",
                    "filters": {"years": [2023]},
                    "top_k": 5,
                },
            },
        )
        yield AgentEvent(
            "tool_completed",
            request_id,
            {
                "tool_name": "search_reports",
                "status": "completed",
                "result_count": 3,
                "duration_ms": 5.0,
            },
        )
        yield AgentEvent("answer_started", request_id, {})
        yield AgentEvent(
            "run_completed",
            request_id,
            {"answer_status": self.result.answer.status, "duration_ms": 12.5},
            result=self.result,
        )

    def stream(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: SearchFilters | None = None,
    ) -> Iterator[AgentEvent]:
        self.calls.append(
            {
                "query": query,
                "conversation_id": conversation_id,
                "filters": filters,
            }
        )
        request_id = f"backend-{len(self.calls)}"
        if not self.block:
            yield from self._events(request_id)
            return
        # Отдаём run_started, затем ждём отмены/освобождения.
        events = self._events(request_id)
        yield next(events)
        self._started.set()
        self.release.wait(10)
        if self.cancelled.is_set():
            return
        for event in events:
            yield event


class CancellableAgentService(FakeAgentService):
    """Blocking fake с поддержкой ``cancel`` (аддитивный backend-контракт)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["block"] = True
        super().__init__(*args, **kwargs)

    def cancel(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)
        self.cancelled.set()
        self.release.set()


class EmptyAgentService:
    """Поток без терминального события: adapter обязан закрыть запрос."""

    def __init__(self) -> None:
        self.tools = SimpleNamespace(chunk_backend=None)
        self.documents_by_year = {2023: ["sber_2023"]}

    def stream(self, query, *, conversation_id, filters=None):
        yield AgentEvent("run_started", "backend-empty", {})


class MissingResultService:
    """Терминальное событие без ``result`` — безопасный run_failed."""

    def __init__(self) -> None:
        self.tools = SimpleNamespace(chunk_backend=None)
        self.documents_by_year = {2023: ["sber_2023"]}

    def stream(self, query, *, conversation_id, filters=None):
        yield AgentEvent("run_started", "backend-missing", {})
        yield AgentEvent(
            "run_completed", "backend-missing", {"answer_status": "answer"}
        )


class AskOnlyService:
    """Backend без ``stream``: adapter использует ``ask`` по §10."""

    def __init__(self, result: AgentRunResult | None = None) -> None:
        self.result = result or make_result()
        self.calls: list[dict[str, Any]] = []
        self.tools = SimpleNamespace(chunk_backend=None)
        self.documents_by_year = {2023: ["sber_2023"]}

    def ask(self, query, *, conversation_id, filters=None):
        self.calls.append(
            {
                "query": query,
                "conversation_id": conversation_id,
                "filters": filters,
            }
        )
        return self.result


def make_services(
    config: ApiConfig,
    service: Any,
    *,
    corpus: CorpusInfo | None = None,
) -> Services:
    return Services(
        config,
        agent_provider=StaticAgentProvider(service),
        corpus=corpus if corpus is not None else default_corpus(),
    )


@contextmanager
def api_client(
    tmp_path: Path,
    service: Any,
    *,
    config: ApiConfig | None = None,
    corpus: CorpusInfo | None = None,
) -> Iterator[TestClient]:
    cfg = config or make_config(tmp_path)
    services = make_services(cfg, service, corpus=corpus)
    app = create_app(config=cfg, services=services)
    with TestClient(app) as client:
        yield client


def create_research(client: TestClient, title: str = "Исследование") -> dict[str, Any]:
    response = client.post("/api/research", json={"title": title})
    assert response.status_code == 201, response.text
    return response.json()


def send_message(
    client: TestClient,
    research_id: str,
    *,
    content: str = "Какая выручка за 2023 год?",
    client_message_id: str = CLIENT_MESSAGE_ID,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": content,
        "client_message_id": client_message_id,
    }
    if filters is not None:
        payload["filters"] = filters
    response = client.post(
        f"/api/research/{research_id}/messages", json=payload
    )
    assert response.status_code == 202, response.text
    return response.json()


def parse_sse(text: str) -> list[dict[str, Any]]:
    """Разобрать SSE-поток на события и heartbeat-комментарии."""
    events: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event: dict[str, Any] = {
            "id": None,
            "event": None,
            "data": None,
            "comment": None,
        }
        for line in block.splitlines():
            if line.startswith(":"):
                event["comment"] = line
            elif line.startswith("id: "):
                event["id"] = int(line[4:])
            elif line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[6:])
        if event["event"] or event["comment"]:
            events.append(event)
    return events


def read_sse(
    client: TestClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Any, list[dict[str, Any]], str]:
    with client.stream("GET", url, headers=headers or {}) as response:
        text = "".join(response.iter_text())
    return response, parse_sse(text), text


def wait_for(predicate: Any, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
