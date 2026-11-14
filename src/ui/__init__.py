"""Пакет Streamlit-UI для агента отчётов.

Импорт пакета не требует установленного Streamlit.
"""

from __future__ import annotations

from typing import Any


def main(*args: Any, **kwargs: Any) -> None:
    """Лениво загружает и запускает UI."""
    from .app import main as run_app

    run_app(*args, **kwargs)


__all__ = ["main"]
