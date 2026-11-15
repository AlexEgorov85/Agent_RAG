"""Ленивый singleton ``AgentService``, описание корпуса и capability-флаги.

Модуль не импортирует retrieval/agent на уровне импорта: тяжёлая сборка
происходит при первом обращении, а не при ``import src.api.app``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.api.config import ApiConfig
from src.api.errors import BackendUnavailable
from src.api.execution import ExecutionGate, PriorityGate

#: Честные флаги первого релиза. ``answer_delta``/``evidence_updated`` backend
#: сейчас не публикует, поэтому streaming-возможности выключены.
CAPABILITIES: dict[str, Any] = {
    "contract_version": "1",
    "token_streaming": False,
    "incremental_evidence": False,
    "coverage": True,
    "calculations": True,
    "structure_comparison": True,
    "structure_content_comparison": True,
    "graph_panel": False,
    "pdf_page_links": False,
    "evaluation_ui": False,
    "evaluation_runs": False,
    "evaluation_sse": False,
    "evaluation_cancel": False,
    "retrieval_modes": ["hybrid", "dense", "keyword"],
    "judge": False,
    "max_repeat": 5,
    "max_queued": 3,
}


class AgentProvider:
    """Создаёт ``AgentService`` ровно один раз и не раскрывает текст ошибки."""

    def __init__(
        self,
        factory: Callable[[], Any] | None = None,
        *,
        retry_after_s: float = 30.0,
    ) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._service: Any = None
        self._failed_until = 0.0
        self._retry_after_s = max(0.0, float(retry_after_s))

    def set_service(self, service: Any) -> None:
        with self._lock:
            self._service = service
            self._failed_until = 0.0

    def get(self) -> Any:
        with self._lock:
            if self._service is not None:
                return self._service
            if self._failed_until and time.monotonic() < self._failed_until:
                raise BackendUnavailable()
            try:
                factory = self._factory
                if factory is None:
                    from src.agent.service import create_agent_service

                    factory = create_agent_service
                self._service = factory()
            except Exception:  # noqa: BLE001 — текст исключения наружу не отдаём
                # Временная ошибка старта (нет ключа/сети) не должна требовать
                # перезапуска API: после паузы пробуем собрать backend снова.
                self._failed_until = time.monotonic() + self._retry_after_s
                raise BackendUnavailable() from None
            return self._service

    def available(self) -> bool:
        try:
            self.get()
        except BackendUnavailable:
            return False
        return True


class StaticAgentProvider:
    """Провайдер для тестов и внедрения готового сервиса."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def get(self) -> Any:
        return self.service

    def set_service(self, service: Any) -> None:
        self.service = service

    def available(self) -> bool:
        return self.service is not None


@dataclass(frozen=True)
class CorpusInfo:
    name: str = "Локальный корпус"
    years: list[int] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    issuers: list[str] = field(default_factory=list)
    years_available: bool = False

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "years": list(self.years),
            "documents": [dict(item) for item in self.documents],
            "issuers": list(self.issuers),
            "years_available": self.years_available,
        }


def _registry_entries(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in sources:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            result[doc_id] = item
    return result


def build_corpus_info(service: Any, config: ApiConfig) -> CorpusInfo:
    """Описание корпуса из уже загруженного индекса, без чтения чанков заново."""
    documents_by_year = getattr(service, "documents_by_year", None) or {}
    registry = _registry_entries(config.source_registry_path)
    years: list[int] = []
    documents: list[dict[str, Any]] = []
    issuers: set[str] = set()
    for year in sorted(documents_by_year):
        try:
            normalized_year = int(year)
        except (TypeError, ValueError):
            continue
        years.append(normalized_year)
        for doc_id in sorted(documents_by_year.get(year, [])):
            entry = registry.get(str(doc_id), {})
            issuer = entry.get("issuer")
            if not isinstance(issuer, str) or not issuer:
                issuer = None
            if issuer:
                issuers.add(issuer)
            documents.append(
                {
                    "doc_id": str(doc_id),
                    "year": normalized_year,
                    "issuer": issuer,
                    "title": None,
                }
            )
    return CorpusInfo(
        name=config.corpus_name,
        years=years,
        documents=documents,
        issuers=sorted(issuers),
        years_available=bool(years),
    )


class Services:
    """Контейнер зависимостей adapter'а (без второй оркестрации)."""

    def __init__(
        self,
        config: ApiConfig,
        *,
        agent_provider: Any = None,
        corpus: CorpusInfo | None = None,
    ) -> None:
        self.config = config
        self.agent_provider = agent_provider or AgentProvider()
        self._priority_gate = PriorityGate()
        self._chat_gate = ExecutionGate(
            self.agent_provider.get,
            priority="chat",
            gate=self._priority_gate,
            lock_path=self.config.llm_lock_path,
        )
        self._evaluation_gate = ExecutionGate(
            self.agent_provider.get,
            priority="evaluation",
            gate=self._priority_gate,
            lock_path=self.config.llm_lock_path,
        )
        self._corpus = corpus
        self._corpus_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self._registry: dict[str, dict[str, Any]] | None = None
        self._rate_lock = threading.Lock()
        self._rate_hits: dict[str, deque[float]] = {}

    def _document_registry(self) -> dict[str, dict[str, Any]]:
        with self._registry_lock:
            if self._registry is None:
                self._registry = _registry_entries(
                    self.config.source_registry_path
                )
            return self._registry

    @property
    def agent(self) -> Any:
        """Chat-прокси: process-wide gate + межпроцессный LLM-lock."""
        # Резолвим backend сразу, чтобы health/404 не маскировались прокси.
        self.agent_provider.get()
        return self._chat_gate

    def evaluation_agent(self) -> Any:
        """Evaluation-прокси с приоритетом ниже chat."""
        self.agent_provider.get()
        return self._evaluation_gate

    def set_agent(self, service: Any) -> None:
        self.agent_provider.set_service(service)

    @property
    def corpus(self) -> CorpusInfo:
        with self._corpus_lock:
            if self._corpus is None:
                self._corpus = build_corpus_info(self.agent, self.config)
            return self._corpus

    def chunk_metadata(self, chunk_id: Any) -> dict[str, Any] | None:
        """Метаданные chunk из существующего ``ChunkBackend``, не из файловой системы."""
        if not isinstance(chunk_id, str) or not chunk_id:
            return None
        service = self.agent
        tools = getattr(service, "tools", None)
        backend = getattr(tools, "chunk_backend", None)
        if backend is None:
            return None
        try:
            if hasattr(backend, "get_chunk"):
                chunk = backend.get_chunk(chunk_id)
            else:
                chunk = backend(chunk_id)
        except Exception:  # noqa: BLE001 — неизвестный chunk = 404, без деталей
            return None
        if isinstance(chunk, dict):
            result = dict(chunk)
            # Issuer отсутствует в chunk-артефактах, но есть в source_registry:
            # это метаданные backend'а, а не догадка из текста.
            doc_id = result.get("doc_id")
            entry = (
                self._document_registry().get(str(doc_id))
                if isinstance(doc_id, str)
                else None
            )
            if entry is not None:
                if not result.get("issuer") and isinstance(
                    entry.get("issuer"), str
                ):
                    result["issuer"] = entry["issuer"]
                if result.get("year") is None and isinstance(
                    entry.get("year"), int
                ):
                    result["year"] = entry["year"]
            return result
        return None

    def pdf_page_links_available(self) -> bool:
        """Реально ли доступен ``GET /api/documents/{doc_id}/file``.

        Импорт ленивый: ``src.api.routes.documents`` использует ``Services``,
        поэтому модульный импорт на уровне файла дал бы цикл. Любая ошибка
        реестра трактуется как «возможности нет», а не 500.
        """
        try:
            from src.api.routes.documents import pdf_page_links_available
        except Exception:  # noqa: BLE001 — capability не должна ронять /health
            return False
        try:
            return bool(pdf_page_links_available(self.config))
        except Exception:  # noqa: BLE001 — битый реестр = флаг false
            return False

    def capabilities(self) -> dict[str, Any]:
        capabilities = dict(CAPABILITIES)
        capabilities["retrieval_modes"] = list(self.config.capabilities_modes())
        # PDF-ссылки включаются только при валидном реестре и доступном файле.
        capabilities["pdf_page_links"] = self.pdf_page_links_available()
        # Evaluation-флаги управляются конфигурацией, чат-флаги не меняются.
        enabled = bool(self.config.eval_ui_enabled)
        capabilities["evaluation_ui"] = enabled
        capabilities["evaluation_runs"] = enabled
        capabilities["evaluation_sse"] = False
        capabilities["evaluation_cancel"] = enabled
        capabilities["judge"] = bool(self.config.judge_available())
        capabilities["max_repeat"] = 5
        capabilities["max_queued"] = int(self.config.eval_queue_size)
        return capabilities

    def rate_limited(self, owner_id: str) -> bool:
        limit = self.config.rate_limit_per_minute
        if limit <= 0:
            return False
        now = time.monotonic()
        with self._rate_lock:
            hits = self._rate_hits.setdefault(owner_id, deque())
            while hits and now - hits[0] > 60.0:
                hits.popleft()
            if len(hits) >= limit:
                return True
            hits.append(now)
            return False
