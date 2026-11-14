"""Бэкенд RAG: поиск → упаковка целыми чанками → обоснованный ответ."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.agent.state import Answer, Citation
from src.engine.context import pack_context
from src.engine.generator import (
    DEFAULT_MAX_CONTEXT_CHARS,
    GENERATION_UNAVAILABLE,
    GENERATION_UNAVAILABLE_TEXT,
    AnswerGenerator,
)
from src.retrieval.hybrid_retriever import SearchFilters

_LOGGER = logging.getLogger(__name__)

NO_EVIDENCE_TEXT = "В доступных материалах не найдено подтверждение для ответа."
CITATION_FAILED_TEXT = "Ответ не прошёл проверку источников."


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_citation(item: Any) -> Citation | None:
    if isinstance(item, Citation):
        return item
    if not isinstance(item, Mapping):
        return None
    chunk_id = item.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        return None
    return Citation(
        chunk_id=chunk_id,
        doc_id=item.get("doc_id") if isinstance(item.get("doc_id"), str) else None,
        year=item.get("year")
        if isinstance(item.get("year"), int) and not isinstance(item.get("year"), bool)
        else None,
        issuer=item.get("issuer") if isinstance(item.get("issuer"), str) else None,
        page_pdf=item.get("page_pdf")
        if isinstance(item.get("page_pdf"), int)
        and not isinstance(item.get("page_pdf"), bool)
        else None,
        page_label=item.get("page_label")
        if isinstance(item.get("page_label"), str)
        else None,
        section_path=item.get("section_path")
        if isinstance(item.get("section_path"), str)
        else None,
        quote=item.get("quote") if isinstance(item.get("quote"), str) else None,
    )


def _coerce_answer(value: Any) -> Answer | None:
    if isinstance(value, Answer):
        return value
    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    if status not in {"answer", "abstain"}:
        return None
    answer_text = value.get("answer")
    if not isinstance(answer_text, str):
        return None
    citations: list[Citation] = []
    for item in value.get("citations", []) or []:
        citation = _coerce_citation(item)
        if citation is None:
            return None
        citations.append(citation)
    facts = value.get("facts", []) or []
    warnings = value.get("warnings", []) or []
    if not isinstance(facts, list) or not isinstance(warnings, list):
        return None
    return Answer(
        status,
        answer_text,
        citations=citations,
        facts=[dict(item) for item in facts if isinstance(item, Mapping)],
        warnings=[item for item in warnings if isinstance(item, str)],
    )


def _safe_abstain(text: str, warning: str) -> Answer:
    return Answer("abstain", text, warnings=[warning])


class RAGEngine:
    """Связывает переданные retriever и answerer, не создавая их сам.

    ``conversation_id`` принимается для совместимости API с ``AgentService``.
    Этот backend-компонент не хранит историю диалога; для разрешения уточнений
    вызывающий передаёт историю напрямую в ``AnswerGenerator``.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        *,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_tokens: int | None = None,
    ) -> None:
        if retriever is None:
            raise ValueError("retriever is required")
        if generator is None:
            raise ValueError("generator is required")
        if isinstance(max_context_chars, bool) or not isinstance(
            max_context_chars, int
        ):
            raise ValueError("max_context_chars must be a non-negative integer")
        if max_context_chars < 0:
            raise ValueError("max_context_chars must be a non-negative integer")
        if max_context_tokens is not None:
            if isinstance(max_context_tokens, bool) or not isinstance(
                max_context_tokens, int
            ):
                raise ValueError(
                    "max_context_tokens must be a non-negative integer or None")
            if max_context_tokens < 0:
                raise ValueError(
                    "max_context_tokens must be a non-negative integer or None")
        self.retriever = retriever
        self.generator = generator
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens

    def retrieve(
        self, query: str, *, filters: SearchFilters | None = None
    ) -> Any:
        """Только делегирует поиск; генератор никогда не вызывает."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        return self.retriever.retrieve(query, filters=filters)

    def ask(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        conversation_id: str | None = None,
    ) -> Answer:
        """Ищет, упаковывает контекст и формирует ответ или информативный abstain."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        del conversation_id  # Принят для совместимости API; история не хранится.

        result = self.retrieve(question, filters=filters)
        hits = getattr(result, "hits", None) or []
        packed = pack_context(
            hits,
            max_chars=self.max_context_chars,
            max_tokens=self.max_context_tokens,
        )
        _LOGGER.info(
            "rag context packed: used_chars=%d max_chars=%d "
            "used_tokens=%d max_tokens=%s included=%d skipped=%d",
            packed.manifest.used_chars,
            packed.manifest.max_chars,
            packed.manifest.used_tokens,
            packed.manifest.max_tokens,
            len(packed.manifest.included),
            len(packed.manifest.skipped),
        )
        if not packed.records:
            # Не позволяем модели дополнять пустой поиск своими знаниями:
            # сразу отказываем локально и вовсе не вызываем провайдера.
            return _safe_abstain(NO_EVIDENCE_TEXT, "no_evidence")

        evidence = packed.evidence
        manifest = set(evidence)
        try:
            raw_answer = self.generator.generate(question, evidence, manifest)
        except Exception:
            return _safe_abstain(GENERATION_UNAVAILABLE_TEXT, GENERATION_UNAVAILABLE)

        answer = _coerce_answer(raw_answer)
        if answer is None:
            return _safe_abstain(GENERATION_UNAVAILABLE_TEXT, GENERATION_UNAVAILABLE)
        return self._validated(answer, manifest)

    @staticmethod
    def _validated(answer: Answer, manifest: set[str]) -> Answer:
        if answer.status == "abstain":
            if answer.citations:
                return _safe_abstain(CITATION_FAILED_TEXT, "citation_validation_failed")
            return Answer(
                "abstain",
                answer.answer,
                facts=[],
                warnings=list(answer.warnings),
            )
        if not answer.answer.strip() or not answer.citations:
            return _safe_abstain(CITATION_FAILED_TEXT, "citation_validation_failed")
        for citation in answer.citations:
            if not citation.chunk_id or citation.chunk_id not in manifest:
                return _safe_abstain(
                    CITATION_FAILED_TEXT, "citation_validation_failed"
                )
        return answer


def build_engine(
    retriever: Any,
    generator: Any | None = None,
    *,
    client: Any | None = None,
    model: str | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_context_tokens: int | None = None,
) -> RAGEngine:
    """Собирает ``RAGEngine`` из уже созданных компонентов.

    Передайте готовый ``generator`` либо настроенный OpenAI-клиент вместе с
    ``model``. Переменные окружения, клиенты, индексы и сетевые вызовы
    неявно не создаются.
    """
    if generator is not None:
        if client is not None or model is not None:
            raise ValueError("provide either generator or client/model, not both")
        return RAGEngine(
            retriever, generator,
            max_context_chars=max_context_chars,
            max_context_tokens=max_context_tokens,
        )
    if client is None or model is None:
        raise ValueError("generator or client and model are required")
    return RAGEngine(
        retriever,
        AnswerGenerator(
            client,
            model,
            max_context_chars=max_context_chars,
            max_context_tokens=max_context_tokens,
        ),
        max_context_chars=max_context_chars,
        max_context_tokens=max_context_tokens,
    )
