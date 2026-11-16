"""Общие контракты retrieval-слоя."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankResult:
    """Переранжированные хиты и диагностика использования fallback на RRF."""

    hits: list[dict[str, Any]] = field(default_factory=list)
    used_fallback: bool = False
    attempts: int = 1
    warning: str | None = None
