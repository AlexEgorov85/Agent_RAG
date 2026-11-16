"""Ленивая инициализация состояния приложения.

Обычно состояние создаётся в lifespan. Функция также позволяет работать
``TestClient`` без явного контекста: БД открывается при первом запросе, а не при
импорте модуля. ``AgentService`` по-прежнему создаётся только при обращении к
backend'у, не здесь.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI

from src.api.config import load_api_config
from src.api.services import Services
from src.api.store import Store
from src.api.streaming import EventHub


def ensure_app_state(app: FastAPI) -> None:
    """Лениво создать store/services/hub для приложения."""
    lock = getattr(app.state, "_bootstrap_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state._bootstrap_lock = lock
    with lock:
        config = getattr(app.state, "config", None)
        if config is None and getattr(app.state, "services", None) is not None:
            config = app.state.services.config
        if config is None:
            config = load_api_config()
        app.state.config = config

        store = getattr(app.state, "store", None)
        if store is None:
            store = Store(config.db_path)
            app.state.store = store
        if not getattr(app.state, "_store_initialized", False):
            store.init()
            app.state._store_initialized = True

        services = getattr(app.state, "services", None)
        if services is None:
            services = Services(config)
            app.state.services = services

        hub = getattr(app.state, "hub", None)
        if hub is None:
            hub = EventHub(store, services, config)
            app.state.hub = hub
