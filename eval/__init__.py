"""Benchmark / evaluation контур Agent_RAG.

Канонический запуск: ``python -m eval.runner --suite smoke``.
Пакет не управляет агентом: он наблюдает публичный ``AgentService.ask()``,
строит trace, считает детерминированные метрики и пишет артефакты запуска.
"""

from __future__ import annotations

from eval.dataset import DatasetBundle, DatasetError, validate_dataset

__all__ = ["DatasetBundle", "DatasetError", "validate_dataset"]
__version__ = "1.1"
