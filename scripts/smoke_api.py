"""Explicit smoke test for the configured embeddings and reranker APIs.

Run with ``python -m scripts.smoke_api --live``. The script makes real API
requests only when ``--live`` is supplied and never prints credentials.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import sys
from typing import Any

from src.core.embeddings import CourseEmbeddings
from src.retrieval.reranker import LLMReranker


@dataclass(frozen=True)
class SmokeResult:
    """Safe summary of completed smoke checks; contains no request secrets."""

    embedding_dimensions: int | None
    reranked_chunk_ids: tuple[str, ...] | None


class SmokeFailure(RuntimeError):
    """A deliberately sanitized smoke-test failure category."""

    def __init__(self, stage: str, category: str) -> None:
        super().__init__(category)
        self.stage = stage
        self.category = category


def _safe_reranker_failure_category(warning: str | None) -> str:
    """Classify reranker failure without exposing provider response bodies."""
    message = warning or ""
    status = re.search(r"(?:Error code:\s*|HTTP\s+)(\d{3})", message, re.IGNORECASE)
    if status:
        return f"provider rejected request (HTTP {status.group(1)})"
    if any(
        marker in message.casefold()
        for marker in ("invalid json", "scores", "chunk_id", "response must")
    ):
        return "provider response did not match the reranker schema"
    return "request or response failure"


def run_smoke(
    client: Any,
    *,
    embedding_model: str | None,
    reranker_model: str | None,
    query_prefix: str = "query:",
    check_embeddings: bool = True,
    check_reranker: bool = True,
) -> SmokeResult:
    """Run selected checks using an injected OpenAI-compatible client.

    This function is independently testable with a fake client and does not
    load environment variables or create a client.
    """
    if not check_embeddings and not check_reranker:
        raise ValueError("enable at least one API check")

    embedding_dimensions: int | None = None
    reranked_ids: tuple[str, ...] | None = None

    if check_embeddings:
        if not embedding_model:
            raise ValueError("EMBED_MODEL is required for the embeddings smoke test")
        adapter = CourseEmbeddings(
            client=client,
            model=embedding_model,
            query_prefix=query_prefix,
        )
        vector = adapter.embed_query("Smoke test: annual report retrieval")
        if not vector:
            raise RuntimeError("embeddings API returned an empty vector")
        embedding_dimensions = len(vector)

    if check_reranker:
        if not reranker_model:
            raise ValueError("RERANKER_MODEL is required for the reranker smoke test")
        reranker = LLMReranker(client=client, model=reranker_model)
        result = reranker.rerank(
            "Which chunk mentions operating risk controls?",
            [
                {
                    "chunk_id": "smoke:relevant",
                    "text": "The report describes operating risk controls and monitoring.",
                },
                {
                    "chunk_id": "smoke:other",
                    "text": "The report lists unrelated office locations.",
                },
            ],
        )
        if result.used_fallback:
            raise SmokeFailure(
                "reranker",
                _safe_reranker_failure_category(result.warning),
            )
        reranked_ids = tuple(
            str(hit.get("chunk_id", hit.get("metadata", {}).get("chunk_id", "")))
            for hit in result.hits
        )
        if set(reranked_ids) != {"smoke:relevant", "smoke:other"}:
            raise SmokeFailure("reranker", "unexpected candidate set")

    return SmokeResult(
        embedding_dimensions=embedding_dimensions,
        reranked_chunk_ids=reranked_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser; live API use requires explicit opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow real API requests (one embedding request and/or reranker call).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip the embeddings endpoint check.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip the reranker chat-completions check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load local env settings and run the explicitly requested smoke checks."""
    args = build_parser().parse_args(argv)
    if not args.live:
        print("No API request was made. Pass --live to run the smoke test.")
        return 2
    check_embeddings = not args.no_embeddings
    check_reranker = not args.no_reranker
    if not check_embeddings and not check_reranker:
        print("Select at least one check (embeddings or reranker).", file=sys.stderr)
        return 2

    try:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("LLM_API_KEY and LLM_BASE_URL must be configured")

        client = OpenAI(api_key=api_key, base_url=base_url)
        result = run_smoke(
            client,
            embedding_model=os.getenv("EMBED_MODEL"),
            reranker_model=os.getenv("RERANKER_MODEL"),
            query_prefix=os.getenv("EMBED_QUERY_PREFIX", "query:"),
            check_embeddings=check_embeddings,
            check_reranker=check_reranker,
        )
    except SmokeFailure as exc:
        print(
            f"API smoke test failed at {exc.stage}: {exc.category}.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - never print provider payloads/secrets
        print(f"API smoke test failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    if result.embedding_dimensions is not None:
        print(f"Embeddings API: OK ({result.embedding_dimensions} dimensions)")
    if result.reranked_chunk_ids is not None:
        print(f"Reranker API: OK ({len(result.reranked_chunk_ids)} candidates)")
    return 0


if __name__ == "__main__":  # pragma: no cover - covered through main/run_smoke
    raise SystemExit(main())
