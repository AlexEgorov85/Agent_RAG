"""HTTP/SSE adapter React WebUI над существующим AgentService.

Пакет не заменяет RAG-ядро: он принимает HTTP-запросы, хранит историю
исследований, сериализует события и передаёт работу ``src.agent``.
"""

from __future__ import annotations

__all__ = ["create_app"]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "create_app":
        from src.api.app import create_app

        return create_app
    raise AttributeError(name)
