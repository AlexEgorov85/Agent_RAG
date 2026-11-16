"""CRUD исследований. Владелец проверяется на каждой операции (404 для чужого)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from src.api.config import ApiConfig
from src.api.errors import not_found, validation_error
from src.api.schemas import (
    ResearchCreate,
    ResearchListResponse,
    ResearchSummary,
    ResearchUpdate,
)
from src.api.store import Store

from .deps import Owner, get_config, get_store

router = APIRouter()
DEFAULT_RESEARCH_TITLE = "Новый диалог"


@router.post(
    "/research",
    response_model=ResearchSummary,
    status_code=201,
    tags=["research"],
)
def create_research(
    owner: Owner,
    payload: ResearchCreate | None = None,
    config: ApiConfig = Depends(get_config),
    store: Store = Depends(get_store),
) -> ResearchSummary:
    """Создать исследование."""
    title = (payload.title if payload else None) or DEFAULT_RESEARCH_TITLE
    if len(title) > config.request_title_limit:
        raise validation_error("Слишком длинное название диалога.")
    row = store.create_research(owner, title=title)
    return ResearchSummary(**row)


@router.get(
    "/research", response_model=ResearchListResponse, tags=["research"]
)
def list_research(
    owner: Owner,
    q: str | None = Query(default=None, max_length=500),
    limit: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
    config: ApiConfig = Depends(get_config),
    store: Store = Depends(get_store),
) -> ResearchListResponse:
    """Список исследований владельца."""
    page_size = config.page_size_default if limit is None else limit
    if page_size > config.page_size_max:
        raise validation_error(
            f"limit не может превышать {config.page_size_max}."
        )
    items, total = store.list_researches(
        owner, query=q, limit=page_size, offset=offset
    )
    next_offset = (
        offset + len(items) if offset + len(items) < total else None
    )
    return ResearchListResponse(
        items=[ResearchSummary(**item) for item in items],
        total=total,
        next_offset=next_offset,
    )


@router.get(
    "/research/{research_id}",
    response_model=ResearchSummary,
    tags=["research"],
)
def get_research(
    research_id: str,
    owner: Owner,
    store: Store = Depends(get_store),
) -> ResearchSummary:
    """Исследование по ID."""
    row = store.get_research(owner, research_id)
    if row is None:
        raise not_found("Диалог не найден.")
    return ResearchSummary(**row)


@router.patch(
    "/research/{research_id}",
    response_model=ResearchSummary,
    tags=["research"],
)
def rename_research(
    research_id: str,
    payload: ResearchUpdate,
    owner: Owner,
    store: Store = Depends(get_store),
) -> ResearchSummary:
    """Переименовать исследование."""
    row = store.update_research_title(owner, research_id, payload.title)
    if row is None:
        raise not_found("Диалог не найден.")
    return ResearchSummary(**row)


@router.delete(
    "/research/{research_id}", status_code=204, tags=["research"]
)
def delete_research(
    research_id: str,
    owner: Owner,
    store: Store = Depends(get_store),
) -> Response:
    """Удалить исследование."""
    if not store.delete_research(owner, research_id):
        raise not_found("Диалог не найден.")
    return Response(status_code=204)
