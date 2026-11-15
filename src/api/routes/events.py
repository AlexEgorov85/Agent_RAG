"""REST-статус запроса, SSE-поток и отмена."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from src.api import serialize
from src.api.config import ApiConfig
from src.api.errors import (
    ApiError,
    BackendUnavailable,
    request_not_found,
)
from src.api.schemas import (
    CancelResponse,
    ChatMessage,
    ErrorInfo,
    RequestState,
)
from src.api.services import Services
from src.api.store import Store
from src.api.streaming import BufferedEvent, EventChannel, EventHub

from .deps import Owner, get_config, get_hub, get_services, get_store

router = APIRouter()

_TERMINAL_REQUEST_STATUSES = {
    "completed",
    "abstained",
    "failed",
    "cancelled",
}


def _parse_last_event_id(value: str | None) -> int:
    if not value:
        return 0
    try:
        return max(0, int(value.strip()))
    except (TypeError, ValueError):
        return 0


def _event_frame(record: BufferedEvent) -> str:
    body = json.dumps(
        record.payload, ensure_ascii=False, separators=(",", ":")
    )
    return f"id: {record.sequence}\nevent: agent\ndata: {body}\n\n"


def _resync_frame(reason: str, request_id: str) -> str:
    body = json.dumps(
        {"type": "resync", "reason": reason, "request_id": request_id},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: resync\ndata: {body}\n\n"


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


async def _stream_channel(
    channel: EventChannel,
    *,
    request_id: str,
    last_event_id: int,
    heartbeat: float,
    max_event_bytes: int | None = None,
):
    first = channel.first_sequence
    if last_event_id and first and last_event_id < first - 1:
        yield _resync_frame("replay_window_expired", request_id)
        return
    last = last_event_id
    while True:
        batch, closed = await asyncio.to_thread(
            channel.wait_after, last, heartbeat
        )
        if batch:
            for record in batch:
                frame = _event_frame(record)
                if (
                    max_event_bytes is not None
                    and len(frame.encode("utf-8")) > max_event_bytes
                ):
                    # Событие не влезает в лимит: UI восстановит сообщение по REST.
                    yield _resync_frame("event_too_large", request_id)
                    return
                yield frame
                last = record.sequence
                if record.is_terminal:
                    return
            continue
        if closed:
            return
        yield ": keep-alive\n\n"


async def _resync_only(reason: str, request_id: str):
    yield _resync_frame(reason, request_id)


def _message_from_request_row(row: dict[str, Any]) -> ChatMessage | None:
    if not row.get("message_id"):
        return None
    payload = {
        "id": row["message_id"],
        "research_id": row["research_id"],
        "role": row.get("role") or "assistant",
        "content": row.get("content") or "",
        "created_at": row.get("created_at"),
        "status": row.get("message_status") or "failed",
        "request_id": row.get("message_request_id") or row.get("request_id"),
        "answer_json": row.get("answer_json"),
        "filters_json": row.get("filters_json"),
        "error_json": row.get("error_json"),
    }
    return ChatMessage(**serialize.message_from_stored(payload))


def _error_from_row(row: dict[str, Any]) -> ErrorInfo | None:
    raw = row.get("error_json")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("code"):
            return ErrorInfo(
                code=str(data["code"]),
                message=str(data.get("message") or "Запрос завершился с ошибкой."),
                retryable=bool(data.get("retryable")),
            )
    if row.get("error_code"):
        return ErrorInfo(
            code=str(row["error_code"]),
            message="Запрос завершился с ошибкой.",
            retryable=str(row["error_code"]) in {"interrupted", "agent_run_failed"},
        )
    return None


@router.get(
    "/requests/{request_id}",
    response_model=RequestState,
    tags=["requests"],
)
def get_request_state(
    request_id: str,
    owner: Owner,
    store: Store = Depends(get_store),
) -> RequestState:
    row = store.get_request(owner, request_id)
    if row is None:
        raise request_not_found(request_id)
    status = str(row["request_status"])
    message = None
    if status in _TERMINAL_REQUEST_STATUSES:
        message = _message_from_request_row(row)
    return RequestState(
        request_id=request_id,
        status=status,
        message_id=str(row["assistant_message_id"]),
        message=message,
        error=_error_from_row(row),
    )


@router.get("/requests/{request_id}/events", tags=["requests"])
async def stream_events(
    request_id: str,
    owner: Owner,
    request: Request,
    last_event_id_header: str | None = Header(
        default=None, alias="Last-Event-ID"
    ),
    config: ApiConfig = Depends(get_config),
    store: Store = Depends(get_store),
    hub: EventHub = Depends(get_hub),
) -> StreamingResponse:
    row = store.get_request(owner, request_id)
    if row is None:
        # Не раскрываем существование чужого/просроченного запроса.
        raise ApiError(
            410,
            "request_not_replayable",
            "События запроса недоступны.",
            request_id=request_id,
        )
    channel = hub.get(request_id)
    last_event_id = _parse_last_event_id(last_event_id_header)
    if channel is None:
        reason = (
            "request_finished"
            if str(row["request_status"]) in _TERMINAL_REQUEST_STATUSES
            else "events_unavailable"
        )
        generator = _resync_only(reason, request_id)
    else:
        generator = _stream_channel(
            channel,
            request_id=request_id,
            last_event_id=last_event_id,
            heartbeat=max(0.05, float(config.heartbeat_seconds)),
            max_event_bytes=config.max_event_bytes,
        )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@router.post(
    "/requests/{request_id}/cancel",
    response_model=CancelResponse,
    response_model_exclude_none=True,
    tags=["requests"],
)
def cancel_request(
    request_id: str,
    owner: Owner,
    store: Store = Depends(get_store),
    services: Services = Depends(get_services),
    hub: EventHub = Depends(get_hub),
) -> CancelResponse:
    row = store.get_request(owner, request_id)
    if row is None:
        raise request_not_found(request_id)
    terminal = str(row["request_status"]) in _TERMINAL_REQUEST_STATUSES
    try:
        service = services.agent
    except BackendUnavailable:
        if terminal:
            return CancelResponse(
                request_id=request_id,
                status="already_finished",
                cancel_supported=False,
            )
        raise
    status, supported = hub.cancel(request_id, service)
    if status == "not_found":
        raise request_not_found(request_id)
    if status not in {"cancelled", "not_supported", "already_finished"}:
        raise ApiError(
            500, "internal_error", "Не удалось обработать отмену."
        )
    return CancelResponse(
        request_id=request_id,
        status=status,
        cancel_supported=supported,
        code="cancel_not_supported" if status == "not_supported" else None,
    )
