"""Интроспекция вызываемых объектов без импорта их реализаций."""

from __future__ import annotations

import inspect
from typing import Any


def supports_keyword(target: Any, keyword: str) -> bool:
    """``True``, если ``target`` принимает ``keyword`` явно или через ``**kwargs``."""
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or (
            item.name == keyword
            and item.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for item in parameters
    )
