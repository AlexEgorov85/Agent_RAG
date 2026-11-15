"""Опциональный LLM-судья: интерфейс и безопасная заглушка.

Судья вторичен: deterministic critical errors и проверки ответа остаются
главными. В offline-тестах и при ``--no-judge`` используется ``NullJudge``,
который не вызывает LLM и возвращает ``None`` для rubric. Невалидный JSON
судьи должен давать ``judge_error``, а не нулевую оценку.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

RUBRIC_CRITERIA: tuple[str, ...] = (
    "factual_accuracy",
    "completeness",
    "sources",
    "analysis",
    "constraints",
)
LEGACY_CRITERIA: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "answer_correctness",
)


class Judge(Protocol):
    """Минимальный контракт судьи; реализация не входит в benchmark-core."""

    def evaluate(self, case: Mapping[str, Any]) -> Mapping[str, Any] | None: ...


@dataclass
class NullJudge:
    """Заглушка: судья выключен, rubric и judge-метрики не измерялись."""

    reason: str = "judge_disabled"
    model: str | None = None

    def evaluate(self, case: Mapping[str, Any]) -> None:
        return None


@dataclass
class JudgeResult:
    """Результат судьи или явная отметка недоступности."""

    rubric_total: float | None = None
    scores: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    rationale: str | None = None
    judge_model: str | None = None
    error: str | None = None
    not_measured: list[str] = field(default_factory=list)


def evaluate_optional(
    judge: Judge | None,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Вызвать судью, если он передан; иначе вернуть ``not_measured``."""
    if judge is None or isinstance(judge, NullJudge):
        return {
            "available": False,
            "rubric_total": None,
            "scores": {},
            "issues": [],
            "error": None,
            "not_measured": ["judge", "rubric_mean"],
        }
    try:
        payload = judge.evaluate(case)
    except Exception:
        return {
            "available": False,
            "rubric_total": None,
            "scores": {},
            "issues": [],
            "error": "judge_error",
            "not_measured": ["judge", "rubric_mean"],
        }
    if not isinstance(payload, Mapping):
        return {
            "available": False,
            "rubric_total": None,
            "scores": {},
            "issues": [],
            "error": "judge_error",
            "not_measured": ["judge", "rubric_mean"],
        }
    scores: dict[str, int] = {}
    for criterion in RUBRIC_CRITERIA:
        value = payload.get(criterion)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2:
            return {
                "available": False,
                "rubric_total": None,
                "scores": {},
                "issues": [],
                "error": "judge_error",
                "not_measured": ["judge", "rubric_mean"],
            }
        scores[criterion] = value
    total = float(sum(scores.values()))
    return {
        "available": True,
        "rubric_total": total,
        "scores": scores,
        "issues": [str(item) for item in payload.get("issues", []) or []],
        "rationale": payload.get("rationale"),
        "judge_model": payload.get("judge_model"),
        "error": None,
        "not_measured": [],
    }
