"""Служебные endpoint'ы: health, capabilities, corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from src.api.errors import BackendUnavailable
from src.api.schemas import (
    CapabilitiesResponse,
    CorpusResponse,
    HealthResponse,
)
from src.api.services import Services
from src.api.store import Store

from .deps import Owner, get_services, get_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(
    owner: Owner,
    response: Response,
    store: Store = Depends(get_store),
    services: Services = Depends(get_services),
) -> HealthResponse:
    """Проверка живости API и backend'а."""
    database_ok = store.ping()
    agent_available = False
    corpus_available = False
    try:
        services.agent
        agent_available = True
        corpus_available = services.corpus.years_available
    except BackendUnavailable:
        agent_available = False
    status = "ok" if database_ok and agent_available else "unavailable"
    if status != "ok":
        response.status_code = 503
    return HealthResponse(
        status=status,
        contract_version=services.config.contract_version,
        database="ok" if database_ok else "error",
        corpus_available=corpus_available,
        agent_available=agent_available,
    )


@router.get(
    "/capabilities", response_model=CapabilitiesResponse, tags=["system"]
)
def capabilities(
    owner: Owner,
    services: Services = Depends(get_services),
) -> CapabilitiesResponse:
    """Возможности API для UI."""
    return CapabilitiesResponse(**services.capabilities())


@router.get("/corpus", response_model=CorpusResponse, tags=["system"])
def corpus(
    owner: Owner,
    services: Services = Depends(get_services),
) -> CorpusResponse:
    """Описание загруженного корпуса."""
    return CorpusResponse(**services.corpus.to_public())
