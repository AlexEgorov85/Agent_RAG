"""Транспортные модели Evaluation API (specs/evaluation-webui.md §8).

Только DTO запросов: ответы собираются из уже посчитанных артефактов
(``summary.json``/``cases.jsonl``/trace), поэтому метрики здесь не описываются
и не пересчитываются. Bизнес-валидация кодов (``invalid_suite`` и т.п.)
выполняется в роутере, чтобы клиент получал контрактный ``code``, а не общий
``validation_error``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class EvaluationRunCreate(EvaluationRequestModel):
    suite: str = "smoke"
    split: str | None = None
    retrieval_mode: str = "hybrid"
    judge: bool = False
    repeat: int = 1
    notes: str | None = Field(default=None, max_length=500)
    client_request_id: str | None = Field(default=None, max_length=128)
    confirm_holdout: bool = False

    def cleaned_notes(self) -> str | None:
        if self.notes is None:
            return None
        text = " ".join(self.notes.split())
        return text or None


class EvaluationCancelResponse(EvaluationRequestModel):
    id: str
    status: str
    cancel_supported: bool = True


class EvaluationCompareMeta(EvaluationRequestModel):
    id: str
    status: str
    suite: str
    split: str
    retrieval_mode: str
    created_at: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    dataset_sha256: str | None = None
    corpus_content_sha256: str | None = None
    corpus_fingerprint: str | None = None
    llm_model: str | None = None


class EvaluationMetricDelta(EvaluationRequestModel):
    key: str
    base: float | int | None = None
    candidate: float | int | None = None
    delta: float | int | None = None


class EvaluationCompareResponse(EvaluationRequestModel):
    comparable: bool
    reasons: list[str] = Field(default_factory=list)
    version_differences: list[str] = Field(default_factory=list)
    base: EvaluationCompareMeta
    candidate: EvaluationCompareMeta
    metrics: list[EvaluationMetricDelta] = Field(default_factory=list)
    regressions: list[dict[str, Any]] = Field(default_factory=list)
    improvements: list[dict[str, Any]] = Field(default_factory=list)
    test_changes: list[dict[str, Any]] = Field(default_factory=list)
