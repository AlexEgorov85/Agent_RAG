"""Явный smoke-тест настроенных API эмбеддингов и реранкера.

Запуск: ``python -m scripts.smoke_api --live``. Реальные запросы к API
выполняются только с ``--live``; учётные данные никогда не печатаются.
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
    """Безопасная сводка выполненных проверок; не содержит секретов запроса."""

    embedding_dimensions: int | None
    reranked_chunk_ids: tuple[str, ...] | None


class SmokeFailure(RuntimeError):
    """Намеренно очищенная категория сбоя smoke-теста."""

    def __init__(self, stage: str, category: str) -> None:
        super().__init__(category)
        self.stage = stage
        self.category = category


def _safe_reranker_failure_category(warning: str | None) -> str:
    """Классифицирует сбой реранкера, не раскрывая тела ответов провайдера."""
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
    """Выполняет выбранные проверки через переданный OpenAI-совместимый клиент.

    Функцию можно независимо тестировать с заглушкой клиента; она не читает env
    и не создаёт клиент сама.
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
    """Создаёт парсер CLI; реальные запросы требуют явного ``--live``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Разрешить реальные запросы к API (один запрос эмбеддингов и/или вызов реранкера).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Пропустить проверку эндпоинта эмбеддингов.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Пропустить проверку chat-completions реранкера.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Загружает локальные env-настройки и выполняет явно запрошенные проверки."""
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
    except Exception as exc:  # noqa: BLE001 — никогда не печатаем тела ответов/секреты
        print(f"API smoke test failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    if result.embedding_dimensions is not None:
        print(f"Embeddings API: OK ({result.embedding_dimensions} dimensions)")
    if result.reranked_chunk_ids is not None:
        print(f"Reranker API: OK ({len(result.reranked_chunk_ids)} candidates)")
    return 0


if __name__ == "__main__":  # pragma: no cover — покрыто через main/run_smoke
    raise SystemExit(main())
