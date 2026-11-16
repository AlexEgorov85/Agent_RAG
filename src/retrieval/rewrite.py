"""Безопасное перефразирование поискового запроса.

Перефразы расширяют recall. Проверка отбрасывает варианты, добавляющие годы или
числа; произвольные компании и факты детектор не распознаёт, поэтому промпт
запрещает их отдельно. Оригинальный запрос всегда остаётся первым и не
заменяется.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_YEAR = re.compile(r"(?:18|19|20|21)\d{2}")
_NUMBER = re.compile(r"\d[\d \u00a0.,]*")
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

_SYSTEM_PROMPT = """Ты переформулируешь поисковый запрос по годовым отчётам.
Верни только JSON: {"queries": ["вариант 1", "вариант 2"]}.
Правила:
- не добавляй новые годы, суммы, компании, сущности или факты;
- не меняй смысл и период вопроса;
- не отвечай на вопрос;
- максимум два варианта, каждый короче и пригоден для поиска."""


def _content(response: Any) -> str:
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("completion has no message") from exc
    if isinstance(content, str):
        return content
    raise ValueError("completion content must be text")


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = _FENCE.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("rewrite must be an object")
    return payload


def _numbers(text: str) -> set[str]:
    return {
        match.group(0).replace(" ", "").replace("\u00a0", "").strip(".,")
        for match in _NUMBER.finditer(text)
    }


def _safe_variant(question: str, variant: str) -> bool:
    if not variant.strip():
        return False
    if _YEAR.findall(variant) != _YEAR.findall(question):
        # Множества, а не порядок: перефраз не должен добавлять/убирать годы.
        if set(_YEAR.findall(variant)) - set(_YEAR.findall(question)):
            return False
        if set(_YEAR.findall(question)) - set(_YEAR.findall(variant)):
            return False
    return not (_numbers(variant) - _numbers(question))


class QueryRewriter:
    """LLM-перефразирование с детерминированной проверкой допустимости."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        max_variants: int = 2,
        max_tokens: int = 256,
        temperature: float = 0,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        self.client = client
        self.model = model
        self.max_variants = max_variants
        self.max_tokens = max_tokens
        self.temperature = temperature

    def rewrite(
        self,
        question: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        """Вернуть оригинал и безопасные перефразирования."""
        del history
        if not isinstance(question, str) or not question.strip():
            return []
        try:
            completions = self.client.chat.completions
            create = getattr(completions, "create", completions)
            response = create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question}, ensure_ascii=False
                        ),
                    },
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
            )
            payload = _parse_json(_content(response))
        except Exception:
            return []

        raw = payload.get("queries")
        if not isinstance(raw, (list, tuple)):
            return []
        variants: list[str] = []
        seen = {question.casefold()}
        for item in raw:
            if not isinstance(item, str):
                continue
            variant = " ".join(item.split())
            if variant.casefold() in seen or not _safe_variant(question, variant):
                continue
            seen.add(variant.casefold())
            variants.append(variant)
            if len(variants) >= self.max_variants:
                break
        return variants
