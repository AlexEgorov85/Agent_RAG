"""Транспортные DTO запросов и ответов Evaluation API (docs/specs/evaluation-webui.md §8).

Ответы собираются из уже посчитанных артефактов (``summary.json``/
``cases.jsonl``/trace). Модели только описывают форму данных: метрики здесь не
вычисляются. Бизнес-валидация кодов (``invalid_suite`` и т.п.) выполняется в
роутере, чтобы клиент получал контрактный ``code``, а не общий
``validation_error``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRequestModel(BaseModel):
    """Базовый DTO evaluation API с игнорированием лишних полей."""
    model_config = ConfigDict(extra="ignore")


class EvaluationRunCreate(EvaluationRequestModel):
    """Запрос на создание evaluation-прогона."""
    suite: str = "smoke"
    split: str | None = None
    retrieval_mode: str = "hybrid"
    judge: bool = False
    repeat: int = 1
    notes: str | None = Field(default=None, max_length=500)
    client_request_id: str | None = Field(default=None, max_length=128)
    confirm_holdout: bool = False

    def cleaned_notes(self) -> str | None:
        """Нормализованные заметки без лишних пробелов."""
        if self.notes is None:
            return None
        text = " ".join(self.notes.split())
        return text or None
