"""Streamlit UI package for the report agent.

Importing this package does not require Streamlit to be installed.
"""

from __future__ import annotations

from typing import Any


def main(*args: Any, **kwargs: Any) -> None:
    """Load and run the UI lazily."""
    from .app import main as run_app

    run_app(*args, **kwargs)


__all__ = ["main"]
