"""Веерный поиск по годам поверх обычного retriever.

Агент по-прежнему делает один tool-call ``search_reports``. Внутри адаптер
выполняет отдельный поиск на каждый запрошенный год, поэтому сравнение не
зависит от того, попали ли разные годы в один top-k. Все годы берутся из
фильтра/корпуса, никаких фиксированных дат здесь нет.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Collection, Mapping, Sequence
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


def _hit_field(hit: Any, name: str) -> Any:
    """Читает поле чанка у объекта хита, вложенного словаря или плоского словаря."""
    if isinstance(hit, Mapping):
        chunk = hit.get("chunk")
    else:
        chunk = getattr(hit, "chunk", None)
    if isinstance(chunk, Mapping) and chunk.get(name) is not None:
        return chunk.get(name)
    if isinstance(hit, Mapping):
        return hit.get(name)
    return getattr(hit, name, None)


def _distinct(hits: Sequence[Any], name: str) -> int:
    return len({
        value for value in (_hit_field(hit, name) for hit in hits)
        if value is not None
    })


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
        documents_by_year: Mapping[int, Sequence[str]] | None = None,
        noncomparable_doc_ids: Collection[str] | None = None,
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
        self.documents_by_year = {
            int(year): tuple(str(doc) for doc in docs)
            for year, docs in (documents_by_year or {}).items()
        }
        self.noncomparable_doc_ids = frozenset(noncomparable_doc_ids or ())
        retrieve = getattr(retriever, "retrieve", None)
        self._supports_reranker_flag = _supports_keyword(
            retrieve, "use_reranker"
        )
        self._supports_top_k = _supports_keyword(retrieve, "top_k")
        self._supports_mode = _supports_keyword(retrieve, "retrieval_mode")
        self._supports_queries = _supports_keyword(retrieve, "queries")

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

    def _excluded_years(
        self,
        query: str,
        filters: SearchFilters,
        years: Sequence[int],
    ) -> set[int]:
        """Годы, которые нельзя сравнивать с остальными по умолчанию.

        Отчёт за 2015 год — БПС-Сбербанк (Беларусь), а не ПАО Сбербанк:
        такой год исключается из многочисленного сравнения, если пользователь
        не запросил его явно (год в тексте или ``allowed_doc_ids``).
        """
        if not self.noncomparable_doc_ids or len(years) <= 1:
            return set()
        explicit_docs = set(filters.allowed_doc_ids or ())
        explicit_years = {
            int(match) for match in re.findall(r"(?<!\d)(20\d{2})(?!\d)", query)
        }
        excluded: set[int] = set()
        for year in years:
            docs = set(self.documents_by_year.get(int(year), ()))
            if not docs or not docs <= self.noncomparable_doc_ids:
                continue
            if docs & explicit_docs or int(year) in explicit_years:
                continue
            excluded.add(int(year))
        return excluded

    def _retrieve(
        self,
        query: str,
        filters: SearchFilters,
        *,
        use_reranker: bool,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"filters": filters}
        if self._supports_reranker_flag:
            kwargs["use_reranker"] = use_reranker
        if top_k is not None and self._supports_top_k:
            kwargs["top_k"] = top_k
        if self._supports_mode:
            kwargs["retrieval_mode"] = retrieval_mode
        if queries and self._supports_queries:
            kwargs["queries"] = list(queries)
        return self.retriever.retrieve(query, **kwargs)

    def _retrieve_year(
        self,
        query: str,
        filters: SearchFilters,
        *,
        metric_question: bool,
        top_k: int | None,
        retrieval_mode: str,
        use_reranker: bool,
        queries: Sequence[str] | None = None,
    ) -> Any:
        if metric_question:
            table_filters = replace(filters, kinds={"table"})
            result = self._retrieve(
                query,
                table_filters,
                use_reranker=use_reranker,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                queries=queries,
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
            queries=queries,
        )

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
        rerank_per_year: bool | None = None,
    ) -> RetrievalResult:
        """Веерный поиск; ``rerank_per_year`` переопределяет настройку на вызов.

        ``False`` по умолчанию экономит вызовы реранкера, ``True`` включает
        реранкер для каждого года (сравнение per-year идёт без него). ``None``
        означает «как сконфигурировано в конструкторе».
        """
        effective = (filters or SearchFilters()).normalized()
        requested = sorted(effective.requested_years())
        use_reranker = (
            self.rerank_per_year
            if rerank_per_year is None
            else bool(rerank_per_year)
        )
        # Одиночный поиск по умолчанию идёт с реранкером; явный False
        # вызывающего уважается и здесь.
        single_use_reranker = (
            True if rerank_per_year is None else use_reranker
        )

        if not requested:
            return self._retrieve(
                query,
                effective,
                use_reranker=single_use_reranker,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                queries=queries,
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
                use_reranker=single_use_reranker,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                queries=queries,
            )

        if len(present) > self.max_years:
            present = present[-self.max_years:]
            truncated = True

        warnings: list[str] = []
        excluded_years = self._excluded_years(query, effective, present)
        if excluded_years:
            present = [year for year in present if year not in excluded_years]
            requested = [year for year in requested if year not in excluded_years]
            effective = replace(
                effective,
                years=set(requested),
                year_from=None,
                year_to=None,
            )
            warnings.append(
                "incomparable_year_excluded: "
                + ", ".join(map(str, sorted(excluded_years)))
            )

        missing: set[int] = {
            year for year in requested if year not in available
        }
        per_year = self.per_year_k if len(present) <= 6 else 1
        metric_question = effective.kinds is None and any(
            word in query.casefold() for word in _METRIC_WORDS
        )

        hits: list[Any] = []
        seen: set[str] = set()
        trace = RetrievalTrace()
        trace.retrieval_mode = retrieval_mode
        merged_stages: dict[str, list[dict[str, Any]]] = {}
        merged_available: dict[str, bool] = {}
        merged_reasons: dict[str, str] = {}
        merged_fallback: dict[str, bool] = {}
        query_variants = [
            variant for variant in (queries or [])
            if isinstance(variant, str) and variant.strip()
        ]
        if query_variants and self._supports_queries:
            trace.query_count = 1 + len({
                variant.casefold() for variant in query_variants
                if variant.casefold() != query.casefold()
            })
        else:
            trace.query_count = 1

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
                use_reranker=use_reranker,
                queries=queries,
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
                trace.dense_candidates += sub_trace.dense_candidates
                trace.bm25_candidates += sub_trace.bm25_candidates
                trace.query_count = max(
                    trace.query_count, sub_trace.query_count
                )
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
                for name, fallback_flag in sub_trace.fallback.items():
                    merged_fallback[name] = (
                        merged_fallback.get(name, False) or bool(fallback_flag)
                    )
                for name, items in sub_trace.stages.items():
                    available_flag = sub_trace.available.get(
                        name, items is not None
                    )
                    merged_available[name] = (
                        merged_available.get(name, False) or available_flag
                    )
                    if items is None:
                        merged_stages.setdefault(name, [])
                        merged_reasons.setdefault(
                            name,
                            sub_trace.unavailable.get(name, "unavailable"),
                        )
                        continue
                    # Порядок «год за годом» сохраняется; дубли схлопываются.
                    merged_stages.setdefault(name, []).extend(
                        dict(item) for item in items
                    )

            result_hits = getattr(result, "hits", None)
            if result_hits is None and isinstance(result, (list, tuple)):
                result_hits = result
            for hit in list(result_hits or [])[:per_year]:
                chunk_id = _hit_id(hit)
                if chunk_id is None or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                hits.append(hit)

        limit = None if top_k is None else max(0, int(top_k))
        if limit is not None:
            hits = hits[:limit]

        for name, items in merged_stages.items():
            if merged_available.get(name, False):
                trace.record_stage(name, items, sort_by_rank=False)
            else:
                trace.mark_unavailable(
                    name, merged_reasons.get(name, "unavailable")
                )
        trace.fallback.update(merged_fallback)

        missing_years = sorted(missing)
        if missing_years:
            warnings.append(
                "missing years in corpus: "
                + ", ".join(map(str, missing_years))
            )
        if truncated:
            warnings.append("period truncated to available years")

        trace.returned = len(hits)
        trace.distinct_docs = _distinct(hits, "doc_id")
        trace.distinct_sections = _distinct(hits, "section_path")
        trace.distinct_years = _distinct(hits, "year")
        return RetrievalResult(
            hits=hits,
            warnings=list(dict.fromkeys(warnings)),
            missing_years=missing_years,
            effective_filters=effective,
            trace=trace,
        )
