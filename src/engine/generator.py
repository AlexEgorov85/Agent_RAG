"""Генератор ответов для OpenAI-совместимого API со строгой проверкой цитат.

Генератор можно внедрить в ``AgentService`` как ``Answerer`` или использовать
в ``RAGEngine``. Сбои провайдера, невалидный JSON и нарушения схемы по умолчанию
повторяются один раз, после чего превращаются в безопасный ответ ``abstain``.
Сырой текст провайдера никогда не попадает в предупреждения или ответ.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from threading import local
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

#: Коды предупреждений, которые можно показывать в UI. Строки модели вне этого
#: набора заменяются, поэтому провайдер не может протащить текст промпта/ответа
#: в предупреждение.
SAFE_WARNING_CODES = frozenset(
    {
        "missing_year",
        "no_evidence",
        "low_confidence",
        "scope_mismatch",
        "issuer_mismatch",
        "incomplete_period",
        "structure_unverified",
        "structure_unavailable",
        "structure_scope_limited",
        "structure_content_unverified",
        "structural_claim_without_citation",
        "conflicting_values",
        "unit_mismatch",
        "citation_outside_manifest",
        "fact_not_supported",
        "quote_truncated",
    }
)


class _SchemaError(ValueError):
    """Внутренний маркер ответа, нарушившего JSON-контракт."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_content(response: Any) -> str:
    """Извлекает текст сообщения из OpenAI-совместимого ответа или заглушки."""
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


#: Числовой токен факта: «1 508,6», «25,3%», «0,8», «(1 234)».
_FACT_NUMBER = re.compile(
    r"[-−–—]?"
    r"(?:\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)*)"
)


def _numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for match in _FACT_NUMBER.finditer(str(text or "")):
        raw = (
            match.group(0)
            .replace("\u00a0", "")
            .replace(" ", "")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(",", ".")
        )
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def _value_in_source(value_text: str, source_text: str) -> bool:
    """Число факта встречается в тексте цитируемого источника.

    Сверка намеренно консервативная и не заменяет ``verify_answer``: она
    отбрасывает факт, число которого модель взяла не из citation, чтобы
    необоснованный факт не ронял весь ответ на повторной генерации.
    """
    numbers = _numeric_values(value_text)
    if not numbers:
        return True
    source_numbers = _numeric_values(source_text)
    return any(
        abs(value - candidate) <= max(0.005, abs(candidate) * 1e-6)
        for value in numbers
        for candidate in source_numbers
    )


def _source_text(evidence: Mapping[str, Any], chunk_id: str) -> str:
    source = evidence.get(chunk_id)
    if not isinstance(source, Mapping):
        return ""
    return f"{source.get('text') or ''} {source.get('html') or ''}"


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
    """Формирует обоснованный ``Answer`` через переданный чат-клиент.

    ``max_retries`` — число *дополнительных* попыток после первой. Цитаты вне
    ``evidence_manifest`` отбрасываются (осознанное решение); если после этого
    в ответе не осталось цитат, проверка не проходит и попытка повторяется.
    Ответ ``abstain`` обязан иметь пустые цитаты и факты.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 2,
        *,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_tokens: int | None = None,
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
        if max_context_tokens is not None:
            if isinstance(max_context_tokens, bool) or not isinstance(
                max_context_tokens, int
            ):
                raise ValueError(
                    "max_context_tokens must be a non-negative integer or None")
            if max_context_tokens < 0:
                raise ValueError(
                    "max_context_tokens must be a non-negative integer or None")

        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        # Диагностика упаковки последнего вызова в текущем потоке. Публичный
        # ``Answer`` не меняется: это наблюдаемость для AgentService.
        self._context_state = local()

    @property
    def last_context_chunk_ids(self) -> list[str] | None:
        """ID чанков, реально упакованных в контекст последнего ``generate``.

        ``None`` означает, что генератор в этом потоке ещё не вызывался (или
        вернулся до упаковки контекста). Пустой список — валидный результат:
        контекст был собран, но ни один чанк не поместился в бюджет.
        """
        value = getattr(self._context_state, "chunk_ids", None)
        return list(value) if isinstance(value, list) else None

    def _set_context_chunk_ids(self, value: list[str] | None) -> None:
        self._context_state.chunk_ids = value

    def generate(
        self,
        question: str,
        evidence: Mapping[str, Any],
        evidence_manifest: set[str] | Sequence[str],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        repair: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
        answer_mode: str = "concise",
        structural_changes: Sequence[Mapping[str, Any]] | None = None,
        section_matches: Sequence[Mapping[str, Any]] | None = None,
        missing_years: Sequence[int] | None = None,
        coverage: Any = None,
    ) -> Answer:
        """Возвращает проверенный ответ или безопасный abstain; текст провайдера не всплывает."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        # Сбрасываем диагностику прошлого вызова: список относится только к
        # текущему контексту.
        self._set_context_chunk_ids(None)
        manifest = {
            item
            for item in evidence_manifest
            if isinstance(item, str) and item
        }
        if not isinstance(evidence, Mapping) or not manifest:
            return Answer("abstain", NO_EVIDENCE_TEXT, warnings=["no_evidence"])

        # В промпт попадают только чанки из манифеста. Пересобираем хит, чтобы
        # отсутствующий внутренний chunk_id не скрыл ключ манифеста.
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

        packed = pack_context(
            allowed_hits,
            max_chars=self.max_context_chars,
            max_tokens=self.max_context_tokens,
        )
        self._set_context_chunk_ids(list(packed.context_chunk_ids))
        if not packed.records:
            return Answer("abstain", NO_EVIDENCE_TEXT, warnings=["no_evidence"])

        analysis_context: dict[str, Any] = {}
        structural_available = bool(structural_changes)
        if answer_mode in {"concise", "comparison", "outline"}:
            analysis_context["answer_mode"] = answer_mode
        if missing_years:
            analysis_context["missing_years"] = [
                int(year) for year in missing_years
            ]
        if structural_changes:
            analysis_context["structural_changes"] = [
                dict(item)
                for item in list(structural_changes)[:40]
                if isinstance(item, Mapping)
            ]
        if section_matches:
            analysis_context["section_matches"] = [
                dict(item)
                for item in list(section_matches)[:40]
                if isinstance(item, Mapping)
            ]
        if coverage is not None:
            to_dict = getattr(coverage, "to_dict", None)
            if callable(to_dict):
                analysis_context["coverage"] = to_dict()
            elif isinstance(coverage, Mapping):
                analysis_context["coverage"] = dict(coverage)

        user_prompt = build_user_prompt(
            question,
            packed.records,
            history=history,
            repair=repair or diagnostics is not None,
            analysis_context=analysis_context or None,
            repair_diagnostics=diagnostics,
        )
        # Повтор с тем же промптом при temperature=0 часто воспроизводит ту же
        # невалидную схему. Со второй попытки добавляем repair-инструкцию,
        # чтобы модель пересобрала ответ, а не повторила ошибку. Диагностика
        # передаётся только кодами/ID: см. sanitize_repair_diagnostics.
        retry_prompt = build_user_prompt(
            question,
            packed.records,
            history=history,
            repair=True,
            analysis_context=analysis_context or None,
            repair_diagnostics=diagnostics,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        completions = self.client.chat.completions
        create = getattr(completions, "create", completions)
        for attempt in range(self.max_retries + 1):
            attempt_messages = messages
            if attempt > 0 and not repair:
                attempt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": retry_prompt},
                ]
            try:
                response = create(
                    model=self.model,
                    messages=attempt_messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_tokens,
                )
                payload = _parse_json(_response_content(response))
                return self._to_answer(
                    payload,
                    packed.evidence,
                    manifest,
                    allow_uncited_structural=structural_available,
                )
            except Exception:
                # Намеренно не сохраняем и не показываем текст провайдера/исключения.
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
        *,
        allow_uncited_structural: bool = False,
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
        truncated_quote = False
        for item in raw_citations:
            if not isinstance(item, Mapping):
                raise _SchemaError("each citation must be an object")
            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise _SchemaError("citation chunk_id must be a non-empty string")
            if chunk_id not in manifest:
                # Осознанное решение: выдуманную/неуместную цитату игнорируем,
                # а не превращаем в ссылку, выглядящую допустимой.
                dropped = True
                continue
            if chunk_id in seen:
                continue
            quote = item.get("quote")
            if quote is not None and not isinstance(quote, str):
                raise _SchemaError("citation quote must be a string")
            if isinstance(quote, str) and len(quote) > MAX_QUOTE_CHARS:
                # Длинная цитата не должна ронять весь ответ: обрезаем её
                # до лимита и честно сообщаем об этом warning'ом.
                quote = quote[:MAX_QUOTE_CHARS].rstrip()
                truncated_quote = True
            citations.append(cls._citation(chunk_id, quote, evidence))
            seen.add(chunk_id)
        if not citations:
            # Навигационный структурный ответ: модель явно пометила, что
            # цитат по содержанию нет, и не привела фактов. Цитаты не
            # выдумываются; для остальных режимов ответ без цитат запрещён.
            if (
                allow_uncited_structural
                and not raw_facts
                and "structural_claim_without_citation" in raw_warnings
            ):
                warnings = _safe_warnings(raw_warnings)
                if "structural_claim_without_citation" not in warnings:
                    warnings.append("structural_claim_without_citation")
                return Answer(
                    "answer",
                    answer_text.strip(),
                    warnings=warnings,
                )
            raise _SchemaError("answer requires at least one valid citation")

        cited = {citation.chunk_id for citation in citations}
        facts: list[dict[str, Any]] = []
        dropped_fact = False
        for item in raw_facts:
            if not isinstance(item, Mapping):
                raise _SchemaError("each fact must be an object")
            chunk_id = item.get("chunk_id")
            references = chunk_id if isinstance(chunk_id, list) else [chunk_id]
            if not references or any(
                not isinstance(reference, str) or reference not in cited
                for reference in references
            ):
                # Факт, который ссылается на отброшенную/чужую citation, не
                # может быть подтверждён. Раньше это была SchemaError и весь
                # ответ уходил в повторную генерацию/abstain; безопаснее
                # сохранить подтверждённую часть и пометить потерю явно.
                dropped_fact = True
                continue
            claim = item.get("claim")
            if claim is not None and (not isinstance(claim, str) or not claim.strip()):
                raise _SchemaError("fact claim must be a non-empty string")
            value = item.get("value")
            if value is not None:
                source_text = " ".join(
                    _source_text(evidence, reference)
                    for reference in references
                    if isinstance(reference, str)
                )
                if source_text and not _value_in_source(str(value), source_text):
                    # Число не найдено в цитируемом чанке — факт необоснован.
                    dropped_fact = True
                    continue
            facts.append(dict(item))

        warnings = _safe_warnings(raw_warnings)
        if dropped and "citation_outside_manifest" not in warnings:
            warnings.append("citation_outside_manifest")
        if dropped_fact and "fact_not_supported" not in warnings:
            warnings.append("fact_not_supported")
        if truncated_quote and "quote_truncated" not in warnings:
            warnings.append("quote_truncated")
        return Answer(
            "answer",
            answer_text.strip(),
            citations=citations,
            facts=facts,
            warnings=warnings,
        )
