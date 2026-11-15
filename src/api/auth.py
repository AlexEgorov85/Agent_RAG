"""Режим ``local``: один владелец, HttpOnly cookie, same-origin проверка.

Cookie не содержит секретов и не является токеном провайдера. EventSource
использует ту же cookie (заголовки в URL не передаются).
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import urlsplit

from fastapi import Request, Response

from src.api.config import ApiConfig
from src.api.errors import ApiError

_OWNER_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def new_owner_id() -> str:
    return uuid.uuid4().hex


def config_from(request: Request) -> ApiConfig:
    config = getattr(request.app.state, "config", None)
    return config if isinstance(config, ApiConfig) else ApiConfig()


def owner_from_cookie(request: Request, config: ApiConfig) -> str | None:
    raw = request.cookies.get(config.cookie_name)
    if isinstance(raw, str) and _OWNER_PATTERN.fullmatch(raw):
        return raw
    return None


def _origin_allowed(origin: str, request: Request, config: ApiConfig) -> bool:
    if origin in config.cors_origins:
        return True
    parsed = urlsplit(origin)
    host = request.headers.get("host")
    # Production работает same-origin: Origin должен совпадать с Host.
    return bool(host) and parsed.netloc == host


def check_csrf(request: Request, config: ApiConfig) -> None:
    if request.method.upper() not in _UNSAFE_METHODS:
        return
    origin = request.headers.get("origin")
    if not origin:
        return
    if not _origin_allowed(origin, request, config):
        raise ApiError(
            403,
            "csrf_rejected",
            "Запрос отклонён: недопустимый Origin.",
        )


def require_owner(request: Request, response: Response) -> str:
    """FastAPI-зависимость: вернуть owner_id и выставить cookie при первом визите."""
    config = config_from(request)
    check_csrf(request, config)
    owner = owner_from_cookie(request, config)
    if owner is None:
        owner = new_owner_id()
        response.set_cookie(
            key=config.cookie_name,
            value=owner,
            max_age=config.session_max_age_seconds,
            httponly=True,
            samesite="lax",
            secure=config.cookie_secure,
            path="/",
        )
    request.state.owner_id = owner
    return owner
