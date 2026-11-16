"""История сообщений и отправка вопроса в ``AgentService``."""

from __future__ import annotations

import json
import sqlite3
import uuid

from fastapi import APIRouter, Depends, Query

from src.api import serialize
from src.api.config import ApiConfig
from src.api.errors import (
    ApiError,
    RequestInProgress,
    not_found,
    validation_error,
)
from src.api.schemas import (
    ChatMessage,
    MessageAccepted,
    MessageCreate,
    MessageListResponse,
    ResearchRef,
)
from src.api.services import Services
from src.api.store import (
    ActiveRequestError,
    ResearchNotFoundError,
    Store,
)
from src.api.streaming import EventHub

from .deps import Owner, get_config, get_hub, get_services, get_store

router = APIRouter()


def _accepted(
    store: Store,
    owner: str,
    research_id: str,
    user_row: dict,
    request_row: dict,
) -> MessageAccepted:
    research = store.get_research(owner, research_id)
    request_id = str(request_row["id"])
    return MessageAccepted(
        request_id=request_id,
        user_message=ChatMessage(**serialize.message_from_stored(user_row)),
        assistant_message_id=str(request_row["assistant_message_id"]),
        events_url=f"/api/requests/{request_id}/events",
        research=ResearchRef(
            id=research_id,
            updated_at=(research or {}).get("updated_at") or "",
        ),
    )


def _existing_accepted(
    store: Store, owner: str, research_id: str, user_row: dict
) -> MessageAccepted:
    request_id = user_row.get("request_id")
    request_row = (
        store.get_request(owner, str(request_id)) if request_id else None
    )
    if request_row is None:
        raise ApiError(
            500,
            "internal_error",
            "Состояние запроса не найдено.",
        )
    return _accepted(
        store,
        owner,
        research_id,
        user_row,
        {
            "id": request_row["request_id"],
            "assistant_message_id": request_row["assistant_message_id"],
        },
    )


@router.get(
    "/research/{research_id}/messages",
    response_model=MessageListResponse,
    tags=["messages"],
)
def list_messages(
    research_id: str,
    owner: Owner,
    limit: int | None = Query(default=None, ge=1),
    before: str | None = Query(default=None),
    config: ApiConfig = Depends(get_config),
    store: Store = Depends(get_store),
) -> MessageListResponse:
    """История сообщений исследования."""
    page_size = config.page_size_default if limit is None else limit
    if page_size > config.page_size_max:
        raise validation_error(
            f"limit не может превышать {config.page_size_max}."
        )
    try:
        rows = store.list_messages(
            owner, research_id, limit=page_size, before=before
        )
    except KeyError:
        raise not_found("Сообщение не найдено.") from None
    if rows is None:
        raise not_found("Диалог не найден.")
    items = [ChatMessage(**serialize.message_from_stored(row)) for row in rows]
    next_before = items[0].id if len(items) >= page_size and items else None
    return MessageListResponse(items=items, next_before=next_before)


@router.post(
    "/research/{research_id}/messages",
    response_model=MessageAccepted,
    status_code=202,
    tags=["messages"],
)
def send_message(
    research_id: str,
    payload: MessageCreate,
    owner: Owner,
    config: ApiConfig = Depends(get_config),
    store: Store = Depends(get_store),
    services: Services = Depends(get_services),
    hub: EventHub = Depends(get_hub),
) -> MessageAccepted:
    """Принять вопрос и запустить асинхронный ход."""
    if store.get_research(owner, research_id) is None:
        raise not_found("Диалог не найден.")

    client_message_id = str(payload.client_message_id)
    existing = store.find_client_message(
        owner, research_id, client_message_id
    )
    if existing is not None:
        return _existing_accepted(store, owner, research_id, existing)

    content = payload.content
    if len(content) > config.max_message_chars:
        raise ApiError(
            413,
            "payload_too_large",
            "Сообщение превышает допустимый размер.",
        )
    if services.rate_limited(owner):
        raise ApiError(
            429, "rate_limited", "Превышен лимит запросов.", retryable=True
        )

    try:
        search_filters = serialize.filter_set_to_search(payload.filters)
    except (TypeError, ValueError):
        raise validation_error("Некорректные фильтры запроса.") from None

    # Корпус/LLM недоступны — не создаём «висящее» сообщение.
    service = services.agent

    user_message_id = str(uuid.uuid4())
    assistant_message_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    filters_json = (
        json.dumps(
            serialize.filters_to_public(search_filters), ensure_ascii=False
        )
        if search_filters is not None
        else None
    )
    try:
        outcome = store.start_turn(
            owner_id=owner,
            research_id=research_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            request_id=request_id,
            content=content,
            client_message_id=client_message_id,
            filters_json=filters_json,
        )
    except ResearchNotFoundError:
        raise not_found("Диалог не найден.") from None
    except ActiveRequestError as error:
        raise RequestInProgress(error.active_request_id) from None
    except sqlite3.IntegrityError:
        # Гонка двух одинаковых POST: второй видит уже созданное сообщение.
        existing = store.find_client_message(
            owner, research_id, client_message_id
        )
        if existing is None:
            raise
        return _existing_accepted(store, owner, research_id, existing)

    if outcome["duplicate"]:
        return _existing_accepted(store, owner, research_id, outcome["user_message"])

    hub.start(
        request_id=request_id,
        research_id=research_id,
        assistant_message_id=assistant_message_id,
        query=content,
        filters=search_filters,
        service=service,
    )
    return _accepted(
        store,
        owner,
        research_id,
        outcome["user_message"],
        outcome["request"],
    )
