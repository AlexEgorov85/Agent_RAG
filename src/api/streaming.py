"""SSE-pump поверх ``AgentService.stream`` без второго workflow.

Синхронный итератор ядра исполняется в отдельном потоке; подписчики SSE ждут
события через ``threading.Condition``, а маршрут использует ``asyncio.to_thread``,
поэтому event loop не блокируется. Буфер ограничен и поддерживает replay по
``Last-Event-ID``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from src.api import serialize
from src.api.config import ApiConfig
from src.api.services import Services
from src.api.store import Store

SSE_TERMINAL_TYPES = serialize.TERMINAL_EVENT_TYPES


@dataclass(frozen=True)
class BufferedEvent:
    """Событие с ``sequence`` для replay."""
    sequence: int
    payload: dict[str, Any]

    @property
    def event_type(self) -> str:
        """Тип события."""
        return str(self.payload.get("type") or "")

    @property
    def is_terminal(self) -> bool:
        """Терминальное ли событие."""
        return self.event_type in SSE_TERMINAL_TYPES


class EventChannel:
    """Буфер событий одного ``request_id`` с ожиданием и replay."""

    def __init__(self, request_id: str, *, max_events: int) -> None:
        self.request_id = request_id
        self._max_events = max(1, int(max_events))
        self._condition = threading.Condition()
        self._buffer: deque[BufferedEvent] = deque()
        self._sequence = 0
        self._closed = False
        self._cancel_requested = False
        self._backend_request_id: str | None = None
        self._finished_at: float | None = None

    def publish(self, payload: Mapping[str, Any]) -> BufferedEvent | None:
        """Добавить событие в буфер."""
        with self._condition:
            if self._closed:
                return None
            self._sequence += 1
            envelope = dict(payload)
            envelope.setdefault("contract_version", serialize.CONTRACT_VERSION)
            envelope["sequence"] = self._sequence
            record = BufferedEvent(self._sequence, envelope)
            self._buffer.append(record)
            while len(self._buffer) > self._max_events:
                self._buffer.popleft()
            if record.is_terminal:
                self._finished_at = time.monotonic()
            self._condition.notify_all()
            return record

    def wait_after(
        self, sequence: int, timeout: float
    ) -> tuple[list[BufferedEvent], bool]:
        """Вернуть события с ``sequence`` больше указанного (или heartbeat timeout)."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                pending = [
                    item for item in self._buffer if item.sequence > sequence
                ]
                if pending:
                    return pending, self._closed
                if self._closed:
                    return [], True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], False
                self._condition.wait(remaining)

    def close(self) -> None:
        """Закрыть канал."""
        with self._condition:
            self._closed = True
            if self._finished_at is None:
                self._finished_at = time.monotonic()
            self._condition.notify_all()

    def mark_cancelled(self) -> None:
        """Пометить отмену запроса."""
        with self._condition:
            self._cancel_requested = True
            self._condition.notify_all()

    def set_backend_request_id(self, request_id: str) -> None:
        """Запомнить ID запроса backend'а."""
        with self._condition:
            if self._backend_request_id is None:
                self._backend_request_id = request_id

    @property
    def backend_request_id(self) -> str | None:
        """ID запроса backend'а."""
        return self._backend_request_id

    @property
    def cancel_requested(self) -> bool:
        """Запрошена ли отмена."""
        return self._cancel_requested

    @property
    def closed(self) -> bool:
        """Закрыт ли канал."""
        return self._closed

    @property
    def finished_at(self) -> float | None:
        """Время завершения канала."""
        return self._finished_at

    @property
    def first_sequence(self) -> int:
        """``sequence`` первого события в буфере."""
        with self._condition:
            return self._buffer[0].sequence if self._buffer else 0


class EventHub:
    """Реестр активных запросов: запуск, replay-буфер и отмена."""

    def __init__(
        self,
        store: Store,
        services: Services,
        config: ApiConfig,
    ) -> None:
        self.store = store
        self.services = services
        self.config = config
        self._lock = threading.Lock()
        self._channels: dict[str, EventChannel] = {}

    # ------------------------------------------------------------------ channel
    def open(self, request_id: str) -> EventChannel:
        """Создать канал для запроса."""
        with self._lock:
            self._purge_locked()
            channel = self._channels.get(request_id)
            if channel is None:
                channel = EventChannel(
                    request_id, max_events=self.config.replay_buffer_events
                )
                self._channels[request_id] = channel
            return channel

    def get(self, request_id: str) -> EventChannel | None:
        """Канал по ``request_id``."""
        with self._lock:
            return self._channels.get(request_id)

    def _purge_locked(self) -> None:
        ttl = max(1, int(self.config.replay_ttl_seconds))
        now = time.monotonic()
        for key, channel in list(self._channels.items()):
            finished = channel.finished_at
            if channel.closed and finished is not None and now - finished > ttl:
                self._channels.pop(key, None)

    # --------------------------------------------------------------------- run
    def start(
        self,
        *,
        request_id: str,
        research_id: str,
        assistant_message_id: str,
        query: str,
        filters: Any,
        service: Any,
    ) -> None:
        """Запустить выполнение запроса."""
        channel = self.open(request_id)
        self.store.mark_request_running(request_id)
        thread = threading.Thread(
            target=self._pump,
            kwargs={
                "channel": channel,
                "research_id": research_id,
                "assistant_message_id": assistant_message_id,
                "query": query,
                "filters": filters,
                "service": service,
            },
            name=f"api-{request_id}",
            daemon=True,
        )
        thread.start()

    def _pump(
        self,
        *,
        channel: EventChannel,
        research_id: str,
        assistant_message_id: str,
        query: str,
        filters: Any,
        service: Any,
    ) -> None:
        request_id = channel.request_id
        terminal = False
        try:
            stream = getattr(service, "stream", None)
            supports = getattr(service, "supports_stream", None)
            if supports is False or not callable(stream):
                self._run_ask(
                    channel,
                    research_id=research_id,
                    assistant_message_id=assistant_message_id,
                    query=query,
                    filters=filters,
                    service=service,
                )
                terminal = True
                return
            iterator = stream(
                query,
                conversation_id=research_id,
                filters=filters,
            )
            for event in iterator:
                if channel.closed or channel.cancel_requested:
                    break
                backend_request_id = getattr(event, "request_id", None)
                if isinstance(backend_request_id, str) and backend_request_id:
                    channel.set_backend_request_id(backend_request_id)
                kind = str(getattr(event, "event_type", "") or "")
                if kind == "run_completed":
                    message = self._persist_completed(
                        channel, research_id, assistant_message_id, event
                    )
                    if message is None:
                        # Отмена уже закрыла поток; иначе backend не передал
                        # результат — завершаем запрос безопасной ошибкой.
                        if not channel.closed and not channel.cancel_requested:
                            self._fail_safely(
                                channel,
                                research_id,
                                assistant_message_id,
                                code="internal_error",
                                message="Поток завершился без результата.",
                                retryable=True,
                            )
                        terminal = True
                        break
                    payload = serialize.agent_event_payload(
                        event,
                        request_id=request_id,
                        message_id=assistant_message_id,
                    )
                    if payload is not None:
                        payload["message"] = message
                        channel.publish(payload)
                    terminal = True
                    break
                if kind == "run_failed":
                    self._persist_failed(
                        channel, research_id, assistant_message_id, event
                    )
                    payload = serialize.agent_event_payload(
                        event,
                        request_id=request_id,
                        message_id=assistant_message_id,
                    )
                    if payload is not None:
                        channel.publish(payload)
                    terminal = True
                    break
                payload = serialize.agent_event_payload(
                    event,
                    request_id=request_id,
                    message_id=assistant_message_id,
                )
                if payload is not None:
                    channel.publish(payload)
        except Exception:  # noqa: BLE001 — наружу только безопасный код
            if not channel.closed:
                self._fail_safely(
                    channel,
                    research_id,
                    assistant_message_id,
                    code="internal_error",
                    message="Внутренняя ошибка adapter'а.",
                    retryable=False,
                )
            terminal = True
        finally:
            if not terminal and not channel.closed:
                self._fail_safely(
                    channel,
                    research_id,
                    assistant_message_id,
                    code="internal_error",
                    message="Поток выполнения завершился без результата.",
                    retryable=True,
                )
            channel.close()

    # ---------------------------------------------------------------- terminal
    def _run_ask(
        self,
        channel: EventChannel,
        *,
        research_id: str,
        assistant_message_id: str,
        query: str,
        filters: Any,
        service: Any,
    ) -> None:
        """Fallback §10: backend без ``stream`` → ``ask`` + 2 события."""
        if channel.closed or channel.cancel_requested:
            return
        channel.publish(
            {
                "type": "run_started",
                "request_id": channel.request_id,
                "message_id": assistant_message_id,
            }
        )
        result = service.ask(
            query,
            conversation_id=research_id,
            filters=filters,
        )
        if getattr(result, "status", None) == "failed":
            self._fail_safely(
                channel,
                research_id,
                assistant_message_id,
                code="agent_run_failed",
                message="Не удалось безопасно завершить обработку запроса.",
                retryable=True,
            )
            return
        answer = getattr(result, "answer", None)
        timings = getattr(result, "timings", {}) or {}
        event = SimpleNamespace(
            event_type="run_completed",
            request_id=channel.request_id,
            data={
                "answer_status": getattr(answer, "status", "abstain"),
                "duration_ms": timings.get("total_ms", 0.0),
            },
            result=result,
        )
        message = self._persist_completed(
            channel, research_id, assistant_message_id, event
        )
        if message is None:
            if not channel.closed and not channel.cancel_requested:
                self._fail_safely(
                    channel,
                    research_id,
                    assistant_message_id,
                    code="internal_error",
                    message="Поток завершился без результата.",
                    retryable=True,
                )
            return
        payload = serialize.agent_event_payload(
            event,
            request_id=channel.request_id,
            message_id=assistant_message_id,
        )
        if payload is not None:
            payload["message"] = message
            channel.publish(payload)

    def _persist_completed(
        self,
        channel: EventChannel,
        research_id: str,
        message_id: str,
        event: Any,
    ) -> dict[str, Any] | None:
        if self.store.request_status(channel.request_id) == "cancelled":
            return None
        result = getattr(event, "result", None)
        if result is None:
            return None
        fields = serialize.result_message_fields(
            result, self.services.chunk_metadata
        )
        answer_json = serialize.answer_json_from_fields(fields)
        filters_json = (
            json.dumps(fields["effective_filters"], ensure_ascii=False)
            if fields.get("effective_filters")
            else None
        )
        self.store.update_message(
            message_id,
            content=str(fields["content"]),
            status=str(fields["status"]),
            answer_json=answer_json,
            filters_json=filters_json,
        )
        request_status = (
            fields["status"]
            if fields["status"] in {"completed", "abstained", "failed"}
            else "failed"
        )
        self.store.finish_request(channel.request_id, request_status)
        self.store.touch_research(research_id)
        row = self.store.get_message_internal(message_id)
        return serialize.message_from_stored(row) if row is not None else None

    def _persist_failed(
        self,
        channel: EventChannel,
        research_id: str,
        message_id: str,
        event: Any,
    ) -> None:
        if self.store.request_status(channel.request_id) == "cancelled":
            return
        payload = serialize.agent_event_payload(
            event,
            request_id=channel.request_id,
            message_id=message_id,
        ) or {}
        error = serialize.error_info(
            str(payload.get("code") or "agent_run_failed"),
            str(payload.get("message") or "Запрос завершился с ошибкой."),
            bool(payload.get("retryable")),
        )
        self._write_failure(channel, research_id, message_id, error)

    def _fail_safely(
        self,
        channel: EventChannel,
        research_id: str,
        message_id: str,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        payload = {
            "type": "run_failed",
            "request_id": channel.request_id,
            "code": code,
            "message": message,
            "retryable": retryable,
        }
        if self.store.request_status(channel.request_id) != "cancelled":
            self._write_failure(
                channel,
                research_id,
                message_id,
                serialize.error_info(code, message, retryable),
            )
            channel.publish(payload)

    def _write_failure(
        self,
        channel: EventChannel,
        research_id: str,
        message_id: str,
        error: dict[str, Any],
    ) -> None:
        self.store.update_message(
            message_id,
            content="",
            status="failed",
            error_json=json.dumps(error, ensure_ascii=False),
        )
        self.store.finish_request(
            channel.request_id, "failed", error_code=str(error["code"])
        )
        self.store.touch_research(research_id)

    # ------------------------------------------------------------------- cancel
    def cancel(self, request_id: str, service: Any) -> tuple[str, bool]:
        """Вернуть ``(status, cancel_supported)``.

        ``not_supported`` означает, что backend не имеет ``AgentService.cancel``:
        adapter не имитирует отмену.
        """
        status = self.store.request_status(request_id)
        if status is None:
            return "not_found", False
        supported = callable(getattr(service, "cancel", None))
        if status in {"completed", "abstained", "failed", "cancelled"}:
            return "already_finished", supported
        if not supported:
            return "not_supported", False
        if not self.store.cancel_request(request_id):
            return "already_finished", True
        channel = self.get(request_id)
        target = request_id
        if channel is not None:
            channel.mark_cancelled()
            target = channel.backend_request_id or request_id
            channel.publish(
                {"type": "run_cancelled", "request_id": request_id}
            )
            channel.close()
        try:
            service.cancel(target)
        except Exception:  # noqa: BLE001 — ответ всё равно отбрасывается
            pass
        return "cancelled", True

    def shutdown(self) -> None:
        """Остановить все каналы."""
        with self._lock:
            channels = list(self._channels.values())
            self._channels.clear()
        for channel in channels:
            channel.close()
