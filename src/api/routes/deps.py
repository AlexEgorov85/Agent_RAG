"""Общие зависимости маршрутов adapter'а."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Response

from src.api.auth import config_from, require_owner
from src.api.bootstrap import ensure_app_state
from src.api.config import ApiConfig
from src.api.services import Services
from src.api.store import Store
from src.api.streaming import EventHub


def _owner(request: Request, response: Response) -> str:
    ensure_app_state(request.app)
    return require_owner(request, response)


Owner = Annotated[str, Depends(_owner)]


def get_config(request: Request) -> ApiConfig:
    ensure_app_state(request.app)
    return config_from(request)


def get_store(request: Request) -> Store:
    ensure_app_state(request.app)
    return request.app.state.store


def get_services(request: Request) -> Services:
    ensure_app_state(request.app)
    return request.app.state.services


def get_hub(request: Request) -> EventHub:
    ensure_app_state(request.app)
    return request.app.state.hub
