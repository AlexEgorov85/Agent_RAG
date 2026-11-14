"""OpenAI-compatible answer generator with strict citation validation.

The generator can be injected into ``AgentService`` (as its ``Answerer``) or
used by ``RAGEngine``. Provider failures, invalid JSON, and schema violations
are retried once by default and then converted into a safe ``abstain`` answer.
Raw provider text is never copied into warnings or user-visible text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.agent.state import Answer, Citation
from src.engine.context import pack_context
from src.engine.prompts import (
    MAX_QUOTE_CHARS,
    SYSTEM_PROMPT,
    build_user_prompt,
)

DEFAULT_MAX_TOKENS = 1_024
DEFAULT_MAX_CONTEXT_CHARS = 16_000
GENERATION_UNAVAILABLE = "generation_unavailable"
NO_EVIDENCE_TEXT = "В доступных материалах не найдено подтверждение для ответа."
GENERATION_UNAVAILABLE_TEXT = "Ответ не удалось безопасно сформировать."
ABSTAIN_FALLBACK_TEXT = "В доступных источниках недостаточно подтверждений для ответа."

#: Warning values that may be shown to the UI. Model-provided strings outside
#: this set are replaced, so a provider cannot smuggle prompt/body text into an
#: answer warning.
SAFE_WARNING_CODES = frozenset(
    {
        "missing_year",
        "no_evidence",
        "low_confidence",
        "scope_mismatch",
        "issuer_mismatch",
        "citation_outside_manifest",
    }
)


class _SchemaError(ValueError):
    """Internal marker for an answer that violates the JSON contract."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_content(response: Any) -> str:
    """Extract message text from an OpenAI-compatible response or fake."""
    try:
        choice = _get(response, "choices")[0]
        message = _get(choice, "message")
        content = _get(message, "content")
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise _SchemaError("completion did not contain a message") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text")
            if isinstance(item, Mapping)
            else getattr(item, "text", None)
            for item in content
        ]
        joined = "".join(part for part in parts if isinstance(part, str))
        if joined:
            return joined
    raise _SchemaError("completion message content must be text")


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced is not None:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _SchemaError("answer is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise _SchemaError("answer JSON must be an object")
    return payload


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _safe_warnings(values: Sequence[Any]) -> list[str]:
    warnings: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        code = value if value in SAFE_WARNING_CODES else "answer_warning"
        if code not in warnings:
            warnings.append(code)
    return warnings


class AnswerGenerator:
    """Generate a grounded ``Answer`` with an injected chat client.

    ``max_retries`` is the number of *additional* attempts after the first one.
    Citations outside ``evidence_manifest`` are dropped (documented choice);
    if that leaves an ``answer`` without any citation, validation fails and is
    retried. An abstain response must have empty citations and facts.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 1,
        *,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be a non-negative integer")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
            raise ValueError("max_context_chars must be a non-negative integer")
        if max_context_chars < 0:
            raise ValueError("max_context_chars must be a non-negative integer")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_context_chars = max_context_chars

    def generate(
        self,
        question: str,
        evidence: Mapping[str, Any],
        evidence_manifest: set[str] | Sequence[str],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        repair: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Return a validated answer or a safe abstain; never raise provider text."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        manifest = {
            item
            for item in evidence_manifest
            if isinstance(item, str) and item
        }
        if not isinstance(evidence, Mapping) or not manifest:
            return Answer("abstain", NO_EVIDENCE_TEXT, warnings=["no_evidence"])

        # Only manifest-authorized chunks may enter the prompt. Rebuild each
        # hit so a missing inner chunk_id cannot hide the manifest key.
        allowed_hits: list[dict[str, Any]] = []
        for chunk_id, record in evidence.items():
            if chunk_id not in manifest or not isinstance(chunk_id, str):
                continue
            if isinstance(record, Mapping):
                hit = dict(record)
            else:
                inner = getattr(record, "chunk", None)
                if not isinstance(inner, Mapping):
                    continue
                hit = dict(inner)
            hit.setdefault("chunk_id", chunk_id)
            allowed_hits.append(hit)
        if not allowed_hits:
            return Answer("abstain", NO_EVIDENCE_TEXT, warnings=["no_evidence"])

        packed = pack_context(allowed_hits, max_chars=self.max_context_chars)
        if not packed.records:
            return Answer("abstain", NO_EVIDENCE_TEXT, warnings=["no_evidence"])

        user_prompt = build_user_prompt(
            question,
            packed.records,
            history=history,
            repair=repair or diagnostics is not None,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        completions = self.client.chat.completions
        create = getattr(completions, "create", completions)
        for _attempt in range(self.max_retries + 1):
            try:
                response = create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_tokens,
                )
                payload = _parse_json(_response_content(response))
                return self._to_answer(payload, packed.evidence, manifest)
            except Exception:
                # Deliberately do not retain or surface provider/exception text.
                continue
        return Answer(
            "abstain",
            GENERATION_UNAVAILABLE_TEXT,
            warnings=[GENERATION_UNAVAILABLE],
        )

    @staticmethod
    def _citation(
        chunk_id: str,
        quote: str | None,
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> Citation:
        source = evidence.get(chunk_id, {})
        return Citation(
            chunk_id=chunk_id,
            doc_id=_optional_str(source.get("doc_id")),
            year=_optional_int(source.get("year")),
            issuer=_optional_str(source.get("issuer")),
            page_pdf=_optional_int(source.get("page_pdf")),
            page_label=_optional_str(source.get("page_label")),
            section_path=_optional_str(source.get("section_path")),
            quote=quote or None,
        )

    @classmethod
    def _to_answer(
        cls,
        payload: Mapping[str, Any],
        evidence: Mapping[str, Mapping[str, Any]],
        manifest: set[str],
    ) -> Answer:
        status = payload.get("status")
        if status not in {"answer", "abstain"}:
            raise _SchemaError("status must be answer or abstain")
        answer_text = payload.get("answer")
        if not isinstance(answer_text, str):
            raise _SchemaError("answer must be a string")

        raw_citations = payload.get("citations", [])
        if not isinstance(raw_citations, list):
            raise _SchemaError("citations must be a list")
        raw_facts = payload.get("facts", [])
        if not isinstance(raw_facts, list):
            raise _SchemaError("facts must be a list")
        raw_warnings = payload.get("warnings", [])
        if raw_warnings is None:
            raw_warnings = []
        if not isinstance(raw_warnings, list):
            raise _SchemaError("warnings must be a list")

        if status == "abstain":
            if raw_citations or raw_facts:
                raise _SchemaError("abstain must not cite evidence or state facts")
            warnings = _safe_warnings(raw_warnings) or ["no_evidence"]
            return Answer(
                "abstain",
                answer_text.strip() or ABSTAIN_FALLBACK_TEXT,
                warnings=warnings,
            )

        if not answer_text.strip():
            raise _SchemaError("answer text must not be empty")

        citations: list[Citation] = []
        seen: set[str] = set()
        dropped = False
        for item in raw_citations:
            if not isinstance(item, Mapping):
                raise _SchemaError("each citation must be an object")
            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise _SchemaError("citation chunk_id must be a non-empty string")
            if chunk_id not in manifest:
                # Documented choice: ignore an invented/out-of-context citation
                # rather than relaundering it into a valid-looking reference.
                dropped = True
                continue
            if chunk_id in seen:
                continue
            quote = item.get("quote")
            if quote is not None and not isinstance(quote, str):
                raise _SchemaError("citation quote must be a string")
            if isinstance(quote, str) and len(quote) > MAX_QUOTE_CHARS:
                raise _SchemaError("citation quote is too long")
            citations.append(cls._citation(chunk_id, quote, evidence))
            seen.add(chunk_id)
        if not citations:
            raise _SchemaError("answer requires at least one valid citation")

        cited = {citation.chunk_id for citation in citations}
        facts: list[dict[str, Any]] = []
        for item in raw_facts:
            if not isinstance(item, Mapping):
                raise _SchemaError("each fact must be an object")
            chunk_id = item.get("chunk_id")
            references = chunk_id if isinstance(chunk_id, list) else [chunk_id]
            if not references or any(
                not isinstance(reference, str) or reference not in cited
                for reference in references
            ):
                raise _SchemaError("every fact must reference a returned citation")
            claim = item.get("claim")
            if claim is not None and (not isinstance(claim, str) or not claim.strip()):
                raise _SchemaError("fact claim must be a non-empty string")
            facts.append(dict(item))

        warnings = _safe_warnings(raw_warnings)
        if dropped and "citation_outside_manifest" not in warnings:
            warnings.append("citation_outside_manifest")
        return Answer(
            "answer",
            answer_text.strip(),
            citations=citations,
            facts=facts,
            warnings=warnings,
        )
