"""LLM-планировщик поиска с детерминированным fallback.

Router не отвечает на вопрос и не добавляет факты. Он только классифицирует
задачу и предлагает ``QueryPlan``. Числовые годы из вопроса и относительные
периоды сначала извлекает ``SimpleRouter``; LLM дополняет план сущностями,
разделом и типом задачи, но не может расширить годы за пределы корпуса.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.agent.coverage import extract_metrics, normalize_metric
from src.agent.simple_router import SimpleRouter
from src.agent.state import QueryPlan
from src.retrieval.hybrid_retriever import SearchFilters

_INTENTS = {"factoid", "structural", "temporal", "multi_hop"}
_ANSWER_MODES = {"concise", "comparison", "outline"}
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

_SYSTEM_PROMPT = """Ты — планировщик поиска по годовым отчётам, а не генератор ответа.
Верни только один JSON-объект без markdown и пояснений:
{
  "intent": "factoid" | "structural" | "temporal" | "multi_hop",
  "years": [целые годы],
  "year_from": целый год | null,
  "year_to": целый год | null,
  "section_filter": "путь раздела" | null,
  "entities": ["сущность"],
  "needs_rewriting": true | false,
  "needs_graph": true | false,
  "answer_mode": "concise" | "comparison" | "outline",
  "aspects": ["до 4 поисковых аспектов"],
  "required_metrics": ["net_profit", "revenue", "roe", "roa", "assets", "capital", "margin"],
  "requires_comparison": true | false
}

Правила:
- Не отвечай на вопрос и не добавляй факты.
- Годы бери только из вопроса или из available_years.
- «Последние N лет» считай от максимального available_year, а не от текущей даты.
- Для диапазона используй year_from/year_to; для отдельных лет — years.
- structural — вопросы о структуре, разделах, оглавлении и изменениях содержания.
- multi_hop — вопросы, требующие связей между сущностями, фактами или разделами.
- answer_mode=comparison для сравнений и динамики по нескольким годам.
- required_metrics: только канонические id из списка
  (net_profit, revenue, operating_income, roe, roa, assets, capital, margin).
- aspects — до четырёх коротких аспектов поиска, без новых фактов.
- Если сомневаешься, верни безопасный factoid без выдуманных годов."""


def _response_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("completion has no message") from exc
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
    raise ValueError("completion content must be text")


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = _FENCE.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("plan must be an object")
    return payload


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool)
    ]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _plan_years(plan: QueryPlan) -> set[int]:
    if plan.years:
        return set(plan.years)
    if plan.year_from is None and plan.year_to is None:
        return set()
    start = plan.year_from if plan.year_from is not None else plan.year_to
    end = plan.year_to if plan.year_to is not None else plan.year_from
    if start is None or end is None:
        return set()
    return set(range(min(start, end), max(start, end) + 1))


class LLMRouter:
    """Структурированный LLM-роутер поверх детерминированного периода."""

    def __init__(
        self,
        client: Any,
        model: str,
        available_years: Sequence[int],
        *,
        fallback: SimpleRouter | None = None,
        max_tokens: int = 512,
        temperature: float = 0,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        self.client = client
        self.model = model
        self.available_years = tuple(sorted({int(year) for year in available_years}))
        self.fallback = fallback or SimpleRouter(self.available_years)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_year = self.available_years[0] if self.available_years else None
        self.max_year = self.available_years[-1] if self.available_years else None

    def analyze(
        self,
        question: str,
        filters: SearchFilters,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> QueryPlan:
        base = self.fallback.analyze(question, filters)
        try:
            payload = self._call(question, filters, history)
        except Exception:
            return base
        if not isinstance(payload, Mapping):
            return base
        return self._merge(base, payload, question)

    def _call(
        self,
        question: str,
        filters: SearchFilters,
        history: Sequence[Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "available_years": list(self.available_years),
            "min_year": self.min_year,
            "max_year": self.max_year,
            "explicit_filters": {
                "years": sorted(filters.years) if filters.years else None,
                "year_from": filters.year_from,
                "year_to": filters.year_to,
                "section_path_prefix": filters.section_path_prefix,
                "allowed_doc_ids": sorted(filters.allowed_doc_ids)
                if filters.allowed_doc_ids
                else None,
            },
            "recent_dialogue": [
                {"role": item.get("role"), "content": item.get("content")}
                for item in list(history or [])[-4:]
                if isinstance(item, Mapping)
            ],
        }
        completions = self.client.chat.completions
        create = getattr(completions, "create", completions)
        response = create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        return _parse_json(_response_content(response))

    def _in_window(self, year: int) -> bool:
        if self.min_year is None or self.max_year is None:
            return 1800 <= year <= 2199
        return self.min_year - 1 <= year <= self.max_year + 1

    def _valid_years(self, value: Any) -> list[int]:
        return sorted({year for year in _int_list(value) if self._in_window(year)})

    def _valid_range(self, value: Any) -> tuple[int | None, int | None]:
        if not isinstance(value, Mapping):
            return None, None
        start = value.get("year_from")
        end = value.get("year_to")
        if not isinstance(start, int) or isinstance(start, bool):
            start = None
        if not isinstance(end, int) or isinstance(end, bool):
            end = None
        if start is None and end is None:
            return None, None
        if start is None:
            start = end
        if end is None:
            end = start
        start, end = min(start, end), max(start, end)
        if self.min_year is not None:
            start = max(start, self.min_year)
        if self.max_year is not None:
            end = min(end, self.max_year)
        if start > end:
            return None, None
        return start, end

    def _merge(
        self,
        base: QueryPlan,
        payload: Mapping[str, Any],
        question: str = "",
    ) -> QueryPlan:
        intent = payload.get("intent")
        if intent not in _INTENTS:
            intent = base.intent
        if (
            intent == "factoid"
            and base.intent == "temporal"
            and _plan_years(base)
        ):
            # LLM не должна «упростить» уже распознанный период.
            intent = base.intent

        years = list(base.years)
        year_from, year_to = base.year_from, base.year_to
        raw_llm_years = _int_list(payload.get("years"))
        if not years and year_from is None and year_to is None:
            years = self._valid_years(payload.get("years"))
            if not years:
                year_from, year_to = self._valid_range(payload)
        if (
            not years
            and year_from is None
            and year_to is None
            and raw_llm_years
            and base.intent == "factoid"
        ):
            # Все годы плана оказались вне корпуса — безопасный fallback.
            return base

        section_filter = base.section_filter
        if section_filter is None:
            candidate = payload.get("section_filter")
            if isinstance(candidate, str) and candidate.strip():
                section_filter = candidate.strip()

        entities: list[str] = []
        seen: set[str] = set()
        for item in list(base.entities) + _str_list(payload.get("entities")):
            key = item.casefold()
            if key not in seen:
                seen.add(key)
                entities.append(item)

        needs_rewriting = base.needs_rewriting or (
            payload.get("needs_rewriting") is True
        )
        needs_graph = base.needs_graph or (
            payload.get("needs_graph") is True
        )

        answer_mode = base.answer_mode
        candidate_mode = payload.get("answer_mode")
        if candidate_mode in _ANSWER_MODES and answer_mode == "concise":
            answer_mode = candidate_mode

        effective_years = set(years)
        if not effective_years and year_from is not None and year_to is not None:
            effective_years = set(range(year_from, year_to + 1))
        if len(effective_years) > 1:
            answer_mode = "comparison"
        if intent == "structural" and len(effective_years) > 1:
            answer_mode = "comparison"

        aspects: list[str] = []
        for item in list(base.aspects) + _str_list(payload.get("aspects")):
            if item not in aspects:
                aspects.append(item)
        aspects = aspects[:4]

        # LLM не должен расширять набор показателей: для простого вопроса
        # «какая чистая прибыль» он нередко возвращает весь словарь метрик.
        # Берём метрики из вопроса/детерминированного fallback и добавляем
        # только те метрики LLM, которые реально упомянуты в вопросе.
        question_metrics = set(extract_metrics(question))
        metrics: list[str] = []
        for item in list(base.required_metrics) + _str_list(
            payload.get("required_metrics")
        ):
            metric = normalize_metric(item)
            if metric is None or metric in metrics:
                continue
            if metric in question_metrics or metric in base.required_metrics:
                metrics.append(metric)
        metrics = metrics[:5]

        requires_comparison = bool(
            base.requires_comparison
            or payload.get("requires_comparison") is True
            or len(effective_years) > 1
        )

        return QueryPlan(
            intent=intent,
            years=sorted(set(years)),
            year_from=year_from,
            year_to=year_to,
            section_filter=section_filter,
            entities=entities,
            needs_rewriting=needs_rewriting,
            needs_graph=needs_graph,
            answer_mode=answer_mode,
            aspects=aspects,
            required_metrics=metrics,
            requires_comparison=requires_comparison,
        )
