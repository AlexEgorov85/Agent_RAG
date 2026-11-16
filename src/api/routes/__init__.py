"""Сборка общих маршрутов ``/api``; evaluation и documents подключаются в ``app``."""

from __future__ import annotations

from fastapi import APIRouter

from . import events, evidence, messages, research, system

router = APIRouter()
router.include_router(system.router)
router.include_router(research.router)
router.include_router(messages.router)
router.include_router(events.router)
router.include_router(evidence.router)
