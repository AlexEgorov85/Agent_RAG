"""Тесты безопасного перефразирования запроса."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.retrieval.rewrite import QueryRewriter


class FakeClient:
    def __init__(self, payload: Any) -> None:
        def create(**kwargs: Any) -> Any:
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


def test_returns_valid_variants_without_new_years_or_amounts() -> None:
    client = FakeClient(
        {
            "queries": [
                "динамика чистой прибыли за последние 5 лет",
                "чистая прибыль 2024 года",
                "прибыль 2024 и 2025 годов",
            ]
        }
    )
    rewriter = QueryRewriter(client, "fake")

    variants = rewriter.rewrite(
        "Как изменилась чистая прибыль за последние 5 лет?"
    )

    assert variants == ["динамика чистой прибыли за последние 5 лет"]


def test_invalid_json_returns_empty_list() -> None:
    rewriter = QueryRewriter(FakeClient("not json"), "fake")

    assert rewriter.rewrite("Как менялась прибыль?") == []


def test_original_query_is_never_returned_as_variant() -> None:
    client = FakeClient({"queries": ["Как менялась прибыль?"]})
    rewriter = QueryRewriter(client, "fake")

    assert rewriter.rewrite("Как менялась прибыль?") == []
