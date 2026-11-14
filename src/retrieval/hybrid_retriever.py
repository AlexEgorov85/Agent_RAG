"""Гибридный поиск BM25 + dense с RRF-слиянием и опциональным LLM-реранкингом."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RRF_CONSTANT = 60


@dataclass(frozen=True)
class RerankResult:
    """Контракт результата реранкера; ему соответствуют и заглушки, и реальные реранкеры."""

    hits: list[dict[str, Any]] = field(default_factory=list)
    used_fallback: bool = False
    attempts: int = 1
    warning: str | None = None
DEFAULT_CANDIDATES = 20
DEFAULT_RERANK_POOL = 10
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class SearchFilters:
    """Жёсткие ограничения по метаданным/структуре до отбора top-k."""

    years: set[int] | None = None
    year_from: int | None = None
    year_to: int | None = None
    report_periods: set[str] | None = None
    issuers: set[str] | None = None
    kinds: set[str] | None = None
    section_path_prefix: str | None = None
    breadcrumb_contains: str | None = None
    max_section_level: int | None = None
    allowed_doc_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if (self.year_from is not None and self.year_to is not None
                and self.year_from > self.year_to):
            raise ValueError("year_from must be less than or equal to year_to")
        if self.years is not None and any(
                isinstance(year, bool) or not isinstance(year, int)
                for year in self.years):
            raise ValueError("years must contain integers")

    def normalized(self) -> "SearchFilters":
        """Возвращает проверенную копию с множествами, приведёнными к типу ``set``."""
        from dataclasses import replace

        return replace(
            self,
            years=None if self.years is None else {int(year) for year in self.years},
            report_periods=(None if self.report_periods is None
                            else {str(period) for period in self.report_periods}),
            issuers=None if self.issuers is None else set(self.issuers),
            kinds=None if self.kinds is None else set(self.kinds),
            allowed_doc_ids=(None if self.allowed_doc_ids is None
                             else set(self.allowed_doc_ids)),
            section_path_prefix=(self.section_path_prefix.strip()
                                 if isinstance(self.section_path_prefix, str)
                                 else self.section_path_prefix),
            breadcrumb_contains=(self.breadcrumb_contains.strip()
                                 if isinstance(self.breadcrumb_contains, str)
                                 else self.breadcrumb_contains),
        )

    def requested_years(self) -> set[int]:
        """Годы, явно запрошенные точным набором или диапазоном."""
        if self.years:
            return set(self.years)
        if self.year_from is not None or self.year_to is not None:
            start = self.year_from if self.year_from is not None else self.year_to
            end = self.year_to if self.year_to is not None else self.year_from
            return set(range(int(start), int(end) + 1))
        return set()


@dataclass(frozen=True)
class RetrievalHit:
    """Один слитый кандидат с независимой диагностикой по каналам."""

    chunk_id: str
    chunk: Mapping[str, Any]
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


@dataclass
class RetrievalTrace:
    """Счётчики и флаги сбоев для наблюдаемости."""

    fused_candidates: int = 0
    rerank_candidates: int = 0
    returned: int = 0
    retrieval_mode: str = "hybrid"
    query_count: int = 1
    dense_candidates: int = 0
    bm25_candidates: int = 0
    diversity_removed: int = 0
    distinct_docs: int = 0
    distinct_sections: int = 0
    distinct_years: int = 0
    dense_failed: bool = False
    bm25_failed: bool = False
    dense_error_type: str | None = None
    bm25_error_type: str | None = None
    reranker_used_fallback: bool = False
    reranker_status: str = "not_used"
    reranker_error_type: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Хиты поиска, предупреждения, действующие фильтры и трейс."""

    hits: list[RetrievalHit]
    warnings: list[str] = field(default_factory=list)
    missing_years: list[int] = field(default_factory=list)
    effective_filters: SearchFilters = field(default_factory=SearchFilters)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


def _value(chunk: Mapping[str, Any], key: str) -> Any:
    value = chunk.get(key)
    if value is None:
        metadata = chunk.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata.get(key)
    return value


def _segments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.strip("/ ").split("/") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _chunk_id(chunk: Mapping[str, Any]) -> str:
    value = _value(chunk, "chunk_id")
    if not isinstance(value, str) or not value:
        raise ValueError("chunk must have a non-empty chunk_id")
    return value


def _matches(chunk: Mapping[str, Any], filters: SearchFilters) -> bool:
    year = _value(chunk, "year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    if filters.years is not None and year not in filters.years:
        return False
    if filters.year_from is not None and (year is None or year < filters.year_from):
        return False
    if filters.year_to is not None and (year is None or year > filters.year_to):
        return False
    if filters.report_periods is not None:
        if str(_value(chunk, "report_period")) not in filters.report_periods:
            return False
    if filters.issuers is not None:
        if _value(chunk, "issuer") not in filters.issuers:
            return False
    if filters.kinds is not None:
        if _value(chunk, "kind") not in filters.kinds:
            return False
    if filters.allowed_doc_ids is not None:
        if _value(chunk, "doc_id") not in filters.allowed_doc_ids:
            return False

    path_segments = [segment.casefold() for segment in
                     _segments(_value(chunk, "section_path"))]
    if filters.section_path_prefix:
        prefix = [segment.casefold() for segment in
                  _segments(filters.section_path_prefix)]
        if not prefix or path_segments[:len(prefix)] != prefix:
            return False
    if filters.breadcrumb_contains:
        target = filters.breadcrumb_contains.strip().casefold()
        breadcrumbs = [segment.strip().casefold() for segment in
                       _segments(_value(chunk, "breadcrumbs"))]
        if target not in breadcrumbs:
            return False
    if filters.max_section_level is not None:
        level = _value(chunk, "section_level")
        if level is None:
            level = len(_segments(_value(chunk, "breadcrumbs")))
        try:
            level = int(level)
        except (TypeError, ValueError):
            return False
        if level > filters.max_section_level:
            return False
    return True


def _edit_distance(left: str, right: str) -> int:
    """Расстояние Левенштейна для ограниченного исправления ID кандидатов."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _repair_candidate_ids(returned: Sequence[str],
                          expected: set[str]) -> list[str] | None:
    """Сопоставляет почти совпавшие ID провайдера (одна опечатка) реальным кандидатам.

    Реранкер не должен добавлять/удалять чанки; опечатка в один символ внутри
    длинного ``chunk_id`` исправляется, только если однозначно указывает на
    одного кандидата.
    """
    used: set[str] = set()
    repaired: list[str] = []
    for chunk_id in returned:
        if chunk_id in expected and chunk_id not in used:
            used.add(chunk_id)
            repaired.append(chunk_id)
            continue
        matches = [candidate for candidate in expected
                   if candidate not in used
                   and _edit_distance(chunk_id, candidate) <= 1]
        if len(matches) != 1:
            return None
        used.add(matches[0])
        repaired.append(matches[0])
    return repaired if used == expected else None


def _strip_scores(chunk: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {"score", "dense_score", "bm25_score", "rrf_score",
                "rerank_score", "rerank_rank"}
    return {key: value for key, value in chunk.items() if key not in reserved}


class HybridRetriever:
    """BM25 + dense → RRF → опциональный реранкер, с мягкой деградацией."""

    def __init__(self, dense_index: Any, bm25_index: Any, embedder: Any, *,
                 reranker: Any = None, candidates: int = DEFAULT_CANDIDATES,
                 rerank_pool: int = DEFAULT_RERANK_POOL,
                 top_k: int = DEFAULT_TOP_K,
                 diversity: bool = False) -> None:
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_pool = rerank_pool
        self.top_k = top_k
        self.diversity = diversity

    @staticmethod
    def _diversify(
        selected: list[dict[str, Any]],
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Сначала по одному хиту на раздел/документ, затем остальные."""
        if limit <= 0:
            return [], len(selected)
        primary: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        seen_sections: set[str] = set()
        seen_docs: set[str] = set()
        for item in selected:
            chunk = item.get("chunk") if isinstance(item, dict) else None
            chunk = chunk if isinstance(chunk, Mapping) else item
            section = _value(chunk, "section_path") or _value(
                chunk, "breadcrumb_contains"
            )
            doc_id = _value(chunk, "doc_id")
            key = str(section) if section else ""
            doc_key = str(doc_id) if doc_id else ""
            if key and key in seen_sections:
                rest.append(item)
                continue
            if doc_key and doc_key in seen_docs and not key:
                rest.append(item)
                continue
            if key:
                seen_sections.add(key)
            if doc_key:
                seen_docs.add(doc_key)
            primary.append(item)
        ordered = primary + rest
        return ordered[:limit], len(rest)

    def _chunks(self) -> list[Mapping[str, Any]]:
        chunks = getattr(self.dense_index, "chunks", None)
        if chunks is None:
            chunks = getattr(self.bm25_index, "chunks", [])
        return list(chunks or [])

    def _available_years(self) -> set[int]:
        years: set[int] = set()
        for chunk in self._chunks():
            try:
                years.add(int(_value(chunk, "year")))
            except (TypeError, ValueError):
                continue
        return years

    def available_years(self) -> set[int]:
        """Годы, реально представленные в индексе (без хардкода корпуса)."""
        return self._available_years()

    def available_doc_ids(self) -> set[str]:
        """doc_id, реально представленные в индексе."""
        doc_ids: set[str] = set()
        for chunk in self._chunks():
            value = _value(chunk, "doc_id")
            if isinstance(value, str) and value:
                doc_ids.add(value)
        return doc_ids

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Возвращает top-k слитых кандидатов и диагностирует отсутствующие годы."""
        if retrieval_mode not in {"hybrid", "dense", "keyword"}:
            raise ValueError("invalid retrieval_mode")
        effective = filters or SearchFilters()
        warnings: list[str] = []
        trace = RetrievalTrace()
        trace.retrieval_mode = retrieval_mode
        eligible = (lambda chunk: _matches(chunk, effective))

        query_list = [query]
        seen_queries = {query.casefold()}
        for variant in queries or []:
            if not isinstance(variant, str) or not variant.strip():
                continue
            key = variant.casefold()
            if key not in seen_queries:
                seen_queries.add(key)
                query_list.append(variant)
        trace.query_count = len(query_list)

        rankings: list[tuple[str, list[Mapping[str, Any]]]] = []
        if retrieval_mode in {"hybrid", "dense"}:
            try:
                for variant in query_list:
                    query_vector = self.embedder.embed_query(variant)
                    hits = list(self.dense_index.search(
                        query_vector, k=self.candidates, eligible=eligible))
                    trace.dense_candidates += len(hits)
                    rankings.append(("dense", hits))
            except Exception as exc:  # noqa: BLE001 — один канал не должен блокировать базовый поиск
                trace.dense_failed = True
                trace.dense_error_type = type(exc).__name__
                warnings.append(f"Dense retrieval failed ({type(exc).__name__})")
        if retrieval_mode in {"hybrid", "keyword"}:
            try:
                for variant in query_list:
                    hits = list(self.bm25_index.search(
                        variant, k=self.candidates, eligible=eligible))
                    trace.bm25_candidates += len(hits)
                    rankings.append(("bm25", hits))
            except Exception as exc:  # noqa: BLE001
                trace.bm25_failed = True
                trace.bm25_error_type = type(exc).__name__
                warnings.append(f"BM25 retrieval failed ({type(exc).__name__})")

        fused: dict[str, dict[str, Any]] = {}
        first_seen: dict[str, int] = {}
        counter = 0
        for channel, hits in rankings:
            seen: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                chunk_id = _chunk_id(hit)
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "chunk": _strip_scores(hit),
                        "dense_rank": None, "bm25_rank": None,
                        "dense_score": None, "bm25_score": None,
                        "rrf_score": 0.0,
                    }
                    first_seen[chunk_id] = counter
                    counter += 1
                record = fused[chunk_id]
                best_rank = record[f"{channel}_rank"]
                if best_rank is None or rank < best_rank:
                    record[f"{channel}_rank"] = rank
                score = float(hit.get("score", 0.0))
                best_score = record[f"{channel}_score"]
                if best_score is None or score > best_score:
                    record[f"{channel}_score"] = score
                record["rrf_score"] += 1 / (RRF_CONSTANT + rank)

        order = sorted(fused, key=lambda cid: (-fused[cid]["rrf_score"],
                                               first_seen[cid]))
        candidates = [
            {
                **fused[chunk_id]["chunk"],
                "chunk_id": chunk_id,
                "text": fused[chunk_id]["chunk"].get("text", ""),
                "rrf_score": fused[chunk_id]["rrf_score"],
                "dense_rank": fused[chunk_id]["dense_rank"],
                "bm25_rank": fused[chunk_id]["bm25_rank"],
                "dense_score": fused[chunk_id]["dense_score"],
                "bm25_score": fused[chunk_id]["bm25_score"],
            }
            for chunk_id in order[:self.rerank_pool]
        ]
        trace.fused_candidates = len(candidates)
        trace.rerank_candidates = len(candidates)

        selected = candidates
        if not use_reranker:
            trace.reranker_status = "disabled"
        elif self.reranker is not None and candidates:
            try:
                result = self.reranker.rerank(query, candidates)
                if not hasattr(result, "hits"):
                    raise ValueError("reranker must return a result with hits")
                if getattr(result, "used_fallback", False):
                    trace.reranker_used_fallback = True
                    trace.reranker_status = "fallback"
                    warnings.append("LLM reranker fallback; preserving RRF order")
                else:
                    expected = {candidate["chunk_id"] for candidate in candidates}
                    returned = [hit.get("chunk_id") for hit in result.hits]
                    if any(not isinstance(item, str) or not item
                           for item in returned):
                        raise ValueError("reranker changed the candidate set")
                    if len(returned) != len(expected) or set(returned) != expected:
                        repaired = _repair_candidate_ids(returned, expected)
                        if repaired is None:
                            raise ValueError("reranker changed the candidate set")
                        result = RerankResult(
                            hits=[{**dict(hit), "chunk_id": corrected}
                                  for hit, corrected in zip(result.hits,
                                                            repaired)],
                            used_fallback=False,
                            attempts=getattr(result, "attempts", 1))
                        warnings.append("reranker chunk ids repaired")
                    if any(not isinstance(hit, dict) for hit in result.hits):
                        raise ValueError("reranker hits must be mappings")
                    selected = [dict(hit) for hit in result.hits]
                    trace.reranker_status = "ok"
            except Exception as exc:  # noqa: BLE001 — реранкер необязателен
                trace.reranker_used_fallback = True
                trace.reranker_status = "fallback"
                trace.reranker_error_type = type(exc).__name__
                warnings.append("LLM reranker fallback; preserving RRF order")
        elif self.reranker is not None:
            trace.reranker_status = "empty"

        limit = self.top_k if top_k is None else int(top_k)
        if self.diversity:
            selected, removed = self._diversify(selected, limit)
            trace.diversity_removed = removed
        else:
            selected = selected[:limit]
        hits: list[RetrievalHit] = []
        for position, item in enumerate(selected, start=1):
            chunk_id = item["chunk_id"]
            base = next((candidate for candidate in candidates
                         if candidate["chunk_id"] == chunk_id), None)
            if base is None:
                continue
            hits.append(RetrievalHit(
                chunk_id=chunk_id,
                chunk=_strip_scores(base),
                dense_rank=base["dense_rank"],
                bm25_rank=base["bm25_rank"],
                dense_score=base["dense_score"],
                bm25_score=base["bm25_score"],
                rrf_score=base["rrf_score"],
                rerank_score=item.get("rerank_score"),
                rerank_rank=item.get("rerank_rank", position
                                     if "rerank_score" in item else None),
            ))
        trace.returned = len(hits)
        trace.distinct_docs = len({
            _value(hit.chunk, "doc_id") for hit in hits
            if _value(hit.chunk, "doc_id") is not None
        })
        trace.distinct_sections = len({
            _value(hit.chunk, "section_path") for hit in hits
            if _value(hit.chunk, "section_path") is not None
        })
        trace.distinct_years = len({
            _value(hit.chunk, "year") for hit in hits
            if _value(hit.chunk, "year") is not None
        })

        requested = effective.requested_years()
        missing = sorted(requested - self._available_years())
        if missing:
            warnings.append(
                "missing years in corpus: " + ", ".join(map(str, missing)))
        return RetrievalResult(hits=hits, warnings=list(dict.fromkeys(warnings)),
                               missing_years=missing,
                               effective_filters=effective, trace=trace)
