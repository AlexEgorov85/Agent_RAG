"""Offline tests for the explicit API smoke-test entry point."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.smoke_api import (
    SmokeFailure,
    _safe_reranker_failure_category,
    main,
    run_smoke,
)


class _Embeddings:
    def create(self, *, model: str, input: list[str]) -> dict[str, Any]:
        return {
            "data": [
                {"index": index, "embedding": [float(len(text)), 1.0]}
                for index, text in enumerate(input)
            ]
        }


class _Completions:
    def create(self, **kwargs: Any) -> Any:
        candidate_payload = kwargs["messages"][1]["content"]
        # The fake returns scores for the two fixed smoke candidates.
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"scores":['
                            '{"chunk_id":"smoke:relevant","score":3},'
                            '{"chunk_id":"smoke:other","score":0}'
                            ']}'
                        )
                    )
                )
            ]
        )


@pytest.fixture
def fake_client() -> Any:
    return SimpleNamespace(
        embeddings=SimpleNamespace(create=_Embeddings().create),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_Completions().create)
        ),
    )


def test_run_smoke_uses_injected_client_without_environment(fake_client) -> None:
    result = run_smoke(
        fake_client,
        embedding_model="fake-embed",
        reranker_model="fake-rerank",
    )

    assert result.embedding_dimensions == 2
    assert result.reranked_chunk_ids == ("smoke:relevant", "smoke:other")


def test_live_requests_require_explicit_opt_in(capsys) -> None:
    assert main([]) == 2
    assert "--live" in capsys.readouterr().out


def test_run_smoke_requires_at_least_one_check(fake_client) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_smoke(
            fake_client,
            embedding_model=None,
            reranker_model=None,
            check_embeddings=False,
            check_reranker=False,
        )


@pytest.mark.parametrize(
    ("warning", "expected"),
    [
        (
            "LLM reranker failed: Error code: 400 - secret response body",
            "provider rejected request (HTTP 400)",
        ),
        (
            "LLM reranker failed: reranker returned invalid JSON with private payload",
            "provider response did not match the reranker schema",
        ),
        ("private details", "request or response failure"),
    ],
)
def test_reranker_failure_classification_does_not_expose_details(
    warning: str, expected: str
) -> None:
    category = _safe_reranker_failure_category(warning)

    assert category == expected
    assert "private" not in category
    assert "secret" not in category


def test_run_smoke_reports_sanitized_reranker_fallback() -> None:
    class FailedCompletions:
        def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("Error code: 400 - private provider response")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FailedCompletions(),
        )
    )

    with pytest.raises(SmokeFailure) as error:
        run_smoke(
            client,
            embedding_model=None,
            reranker_model="fake",
            check_embeddings=False,
        )

    assert error.value.stage == "reranker"
    assert error.value.category == "provider rejected request (HTTP 400)"
    assert "private" not in str(error.value)
