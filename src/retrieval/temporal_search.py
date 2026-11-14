"""Веерный поиск по годам поверх обычного retriever.

Агент по-прежнему делает один tool-call ``search_reports``. Внутри адаптер
выполняет отдельный поиск на каждый запрошенный год, поэтому сравнение не
зависит от того, попали ли разные годы в один top-k. Все годы берутся из
фильтра/корпуса, никаких фиксированных дат здесь нет.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.retrieval.hybrid_retriever import (
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)

_METRIC_WORDS = (
    "прибыль",
    "выручка",
    "roe",
    "roa",
    "актив",
    "капитал",
    "доход",
    "расход",
    "маржа",
)


def _supports_keyword(target: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or (
            item.name == keyword
            and item.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for item in parameters
    )


def _hit_id(hit: Any) -> str | None:
    if isinstance(hit, Mapping):
        value = hit.get("chunk_id")
    else:
        value = getattr(hit, "chunk_id", None)
    return value if isinstance(value, str) and value else None


class PerYearSearch:
    """Один внешний вызов, N внутренних поисков с фильтром по каждому году."""

    def __init__(
        self,
        retriever: Any,
        *,
        per_year_k: int = 2,
        max_years: int = 12,
        max_period_years: int = 30,
        rerank_per_year: bool = False,
    ) -> None:
        if per_year_k < 1:
            raise ValueError("per_year_k must be positive")
        if max_years < 1:
            raise ValueError("max_years must be positive")
        if max_period_years < 1:
            raise ValueError("max_period_years must be positive")
        self.retriever = retriever
        self.per_year_k = per_year_k
        self.max_years = max_years
        self.max_period_years = max_period_years
        self.rerank_per_year = rerank_per_year
        retrieve = getattr(retriever, "retrieve", None)
        self._supports_reranker_flag = _supports_keyword(
            retrieve, "use_reranker"
        )
        self._supports_top_k = _supports_keyword(retrieve, "top_k")
        self._supports_mode = _supports_keyword(retrieve, "retrieval_mode")

    def __getattr__(self, name: str) -> Any:
        # Сохраняем доступ к .reranker/.top_k и другим полям обёрнутого retriever.
        retriever = self.__dict__.get("retriever")
        if retriever is None:
            raise AttributeError(name)
        return getattr(retriever, name)

    def _available_years(self) -> set[int]:
        for name in ("available_years", "_available_years"):
            method = getattr(self.retriever, name, None)
            if not callable(method):
                continue
            try:
                return {int(year) for year in method()}
            except (TypeError, ValueError):
                continue
        return set()

    def _retrieve(
        self,
        query: str,
        filters: SearchFilters,
        *,
        use_reranker: bool,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
    ) -> Any:
        kwargs: dict[str, Any] = {"filters": filters}
        if self._supports_reranker_flag:
            kwargs["use_reranker"] = use_reranker
        if top_k is not None and self._supports_top_k:
            kwargs["top_k"] = top_k
        if self._supports_mode:
            kwargs["retrieval_mode"] = retrieval_mode
        return self.retriever.retrieve(query, **kwargs)

    def _retrieve_year(
        self,
        query: str,
        filters: SearchFilters,
        *,
        metric_question: bool,
        top_k: int | None,
        retrieval_mode: str,
    ) -> Any:
        use_reranker = self.rerank_per_year
        if metric_question:
            table_filters = replace(filters, kinds={"table"})
            result = self._retrieve(
                query,
                table_filters,
                use_reranker=use_reranker,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            table_hits = getattr(result, "hits", None)
            if table_hits is None and isinstance(result, (list, tuple)):
                table_hits = result
            if table_hits:
                return result
        return self._retrieve(
            query,
            filters,
            use_reranker=use_reranker,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
    ) -> RetrievalResult:
        effective = (filters or SearchFilters()).normalized()
        requested = sorted(effective.requested_years())

        if not requested:
            return self._retrieve(
                query,
                effective,
                use_reranker=True,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )

        truncated = False
        if len(requested) > self.max_period_years:
            requested = requested[-self.max_period_years:]
            truncated = True

        available = self._available_years()
        if not available:
            # Без метаданных индекса считаем запрошенные годы существующими:
            # решение о missing_years остаётся за конкретным retriever.
            available = set(requested)

        present = [year for year in requested if year in available]
        if len(present) <= 1:
            return self._retrieve(
                query,
                effective,
                use_reranker=True,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )

        if len(present) > self.max_years:
            present = present[-self.max_years:]
            truncated = True

        missing: set[int] = {
            year for year in requested if year not in available
        }
        per_year = self.per_year_k if len(present) <= 6 else 1
        metric_question = effective.kinds is None and any(
            word in query.casefold() for word in _METRIC_WORDS
        )

        hits: list[Any] = []
        warnings: list[str] = []
        seen: set[str] = set()
        trace = RetrievalTrace()

        for year in present:
            year_filters = replace(
                effective,
                years={year},
                year_from=None,
                year_to=None,
            )
            result = self._retrieve_year(
                query,
                year_filters,
                metric_question=metric_question,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            warnings.extend(getattr(result, "warnings", []) or [])
            missing.update(
                int(year)
                for year in (getattr(result, "missing_years", []) or [])
            )

            sub_trace = getattr(result, "trace", None)
            if isinstance(sub_trace, RetrievalTrace):
                trace.dense_failed = trace.dense_failed or sub_trace.dense_failed
                trace.bm25_failed = trace.bm25_failed or sub_trace.bm25_failed
                trace.reranker_used_fallback = (
                    trace.reranker_used_fallback
                    or sub_trace.reranker_used_fallback
                )
                trace.fused_candidates += sub_trace.fused_candidates
                trace.rerank_candidates += sub_trace.rerank_candidates
                trace.dense_error_type = (
                    trace.dense_error_type or sub_trace.dense_error_type
                )
                trace.bm25_error_type = (
                    trace.bm25_error_type or sub_trace.bm25_error_type
                )
                trace.reranker_error_type = (
                    trace.reranker_error_type or sub_trace.reranker_error_type
                )
                if sub_trace.reranker_status == "fallback":
                    trace.reranker_status = "fallback"
                elif trace.reranker_status == "not_used":
                    trace.reranker_status = sub_trace.reranker_status

            result_hits = getattr(result, "hits", None)
            if result_hits is None and isinstance(result, (list, tuple)):
                result_hits = result
            for hit in list(result_hits or [])[:per_year]:
                chunk_id = _hit_id(hit)
                if chunk_id is None or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                hits.append(hit)

        missing_years = sorted(missing)
        if missing_years:
            warnings.append(
                "missing years in corpus: "
                + ", ".join(map(str, missing_years))
            )
        if truncated:
            warnings.append("period truncated to available years")

        trace.returned = len(hits)
        return RetrievalResult(
            hits=hits,
            warnings=list(dict.fromkeys(warnings)),
            missing_years=missing_years,
            effective_filters=effective,
            trace=trace,
        )
