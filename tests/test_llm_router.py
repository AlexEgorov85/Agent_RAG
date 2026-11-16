"""Тесты LLM-роутера: план, fallback и защита от выдуманных годов."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.agent.llm_router import LLMRouter
from src.agent.simple_router import SimpleRouter
from src.retrieval.hybrid_retriever import SearchFilters


class FakeClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0

        def create(**kwargs: Any) -> Any:
            self.calls += 1
            content = (
                payload
                if isinstance(payload, str)
                else json.dumps(payload, ensure_ascii=False)
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ]
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )


YEARS = [2019, 2020, 2022, 2023, 2024]


def test_llm_adds_structural_intent_and_section() -> None:
    client = FakeClient(
        {
            "intent": "structural",
            "years": [],
            "year_from": None,
            "year_to": None,
            "section_filter": "Управление рисками",
            "entities": ["Сбербанк"],
            "needs_rewriting": True,
            "needs_graph": False,
            "answer_mode": "comparison",
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze(
        "Как менялись подходы к безопасности?",
        SearchFilters(),
    )

    assert plan.intent == "structural"
    assert plan.section_filter == "Управление рисками"
    assert plan.entities == ["Сбербанк"]
    assert plan.years == YEARS
    assert plan.answer_mode == "comparison"


def test_explicit_period_wins_over_llm_hallucination() -> None:
    client = FakeClient(
        {
            "intent": "temporal",
            "years": [1999],
            "year_from": 1999,
            "year_to": 1999,
            "answer_mode": "comparison",
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze(
        "Как изменилась прибыль за последние 5 лет?",
        SearchFilters(),
    )

    assert plan.years == [2020, 2021, 2022, 2023, 2024]


def test_llm_years_are_used_when_rules_find_no_period() -> None:
    client = FakeClient(
        {
            "intent": "temporal",
            "years": [2022, 2023],
            "answer_mode": "comparison",
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze("Покажи показатели", SearchFilters())

    assert plan.intent == "temporal"
    assert plan.years == [2022, 2023]


def test_invalid_json_falls_back_to_deterministic_plan() -> None:
    client = FakeClient("not json")
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze(
        "Какая прибыль за 2023 год?", SearchFilters()
    )

    assert plan.years == [2023]
    assert plan.answer_mode == "concise"


def test_llm_years_adjacent_to_corpus_are_ignored() -> None:
    client = FakeClient(
        {
            "intent": "temporal",
            "years": [2018, 2025],
            "answer_mode": "comparison",
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze("Покажи показатели", SearchFilters())

    assert plan.years == []


def test_llm_years_outside_corpus_window_are_ignored() -> None:
    client = FakeClient(
        {
            "intent": "temporal",
            "years": [1900, 2999],
            "answer_mode": "comparison",
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze(
        "Покажи показатели", SearchFilters()
    )

    assert plan.years == []
    assert plan.intent == "factoid"


def test_llm_does_not_invent_metrics_not_mentioned_in_question() -> None:
    """Модель не должна расширять простой factoid до всего словаря метрик."""
    client = FakeClient(
        {
            "intent": "factoid",
            "years": [2023],
            "required_metrics": [
                "net_profit",
                "revenue",
                "roe",
                "roa",
                "assets",
            ],
        }
    )
    router = LLMRouter(
        client,
        "fake",
        YEARS,
        fallback=SimpleRouter(YEARS),
    )

    plan = router.analyze(
        "Какая чистая прибыль Сбербанка за 2023 год?", SearchFilters()
    )

    assert plan.required_metrics == ["net_profit"]
