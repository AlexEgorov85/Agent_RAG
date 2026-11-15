"""Гибридный поиск BM25 + dense с RRF-слиянием и опциональным LLM-реранкингом."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

RRF_CONSTANT = 60

#: Явное предупреждение: жёсткий ``section_path_prefix`` не дал ни одного хита,
#: поэтому выполнен поиск без него. Оно не означает, что хиты точно из
#: запрошенного раздела.
SECTION_FILTER_FALLBACK = "section_filter_fallback"

#: Раздел в корпусе есть, но чанков за запрошенные годы в нём нет. Fallback в
#: этом случае не выполняется: чужие разделы не подменяют пустой ответ.
SECTION_YEAR_ABSENT = "section_year_absent"

#: Имена этапов retrieval для trace. ``graph`` заполняет графовый ретривер,
#: ``context`` — упаковщик контекста; базовый гибридный поиск их не выдумывает.
STAGE_NAMES: tuple[str, ...] = (
    "dense",
    "bm25",
    "graph",
    "rrf",
    "rerank",
    "returned",
    "context",
)


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


def _score(value: Any) -> float | None:
    """Приводит технический score к ``float``, не выдавая его за вероятность."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank(value: Any, position: int) -> int:
    if isinstance(value, bool):
        return position
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return position


def _normalise_stage(
    items: Sequence[Mapping[str, Any]],
    *,
    sort_by_rank: bool = True,
) -> list[dict[str, Any]]:
    """Схлопывает дубли по ``chunk_id`` и нумерует ранги с 1.

    ``sort_by_rank=True`` упорядочивает элементы по лучшему рангу (типично для
    одного канала с несколькими формулировками); ``sort_by_rank=False``
    сохраняет порядок встречи — это нужно, когда веерный поиск складывает
    этапы разных лет.
    """
    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, item in enumerate(items or ()):
        if not isinstance(item, Mapping):
            continue
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        rank = _rank(item.get("rank"), position + 1)
        score = _score(item.get("score"))
        if chunk_id not in entries:
            entries[chunk_id] = {"rank": rank, "score": score, "order": position}
            order.append(chunk_id)
            continue
        current = entries[chunk_id]
        if rank < current["rank"]:
            current["rank"] = rank
            current["score"] = score
    if sort_by_rank:
        order.sort(key=lambda cid: (entries[cid]["rank"], entries[cid]["order"]))
    return [
        {"chunk_id": chunk_id, "rank": rank, "score": entries[chunk_id]["score"]}
        for rank, chunk_id in enumerate(order, start=1)
    ]


@dataclass
class RetrievalTrace:
    """Счётчики, этапы и флаги сбоев для наблюдаемости.

    Этапы хранятся двумя согласованными словарями:

    - ``stages[name]`` — список ``{"chunk_id", "rank", "score"}`` или ``None``;
    - ``available[name]`` — ``True``, если этап реально измерен.

    Пустой список означает «этап выполнен, результатов 0», а ``None`` вместе с
    ``available[name] is False`` — «этап не выполнялся/недоступен». Причина
    недоступности лежит в ``unavailable[name]`` (``not_requested``, ``failed``,
    ``fallback``, ...). ``score`` — технический ранг/скор, не вероятность
    истинности.

    Поля ``section_filter_*`` описывают жёсткий фильтр раздела:

    - ``section_filter_status``: ``not_requested`` (фильтра не было),
      ``applied`` (фильтр сработал), ``year_absent`` (раздел есть в корпусе,
      но не содержит чанков за запрошенные годы; выдача пустая, fallback не
      выполнялся), ``fallback_title`` (фильтр не дал хитов; хиты взяты по
      совпадению названия раздела, потому что пути в structure и чанках
      расходятся), ``fallback`` (фильтр не дал хитов, название раздела в путях
      не найдено — поиск без section-фильтра), ``fallback_empty`` (fallback
      тоже ничего не нашёл);
    - ``section_filter_prefix`` — запрошенный префикс;
    - ``section_filter_strict_hits`` — сколько хитов дал строгий поиск;
    - ``section_filter_corpus_matches`` — сколько чанков индекса имеют такой
      ``section_path`` без учёта остальных фильтров (0 = такой таксономии в
      корпусе нет);
    - ``section_filter_title`` — название раздела (последний сегмент пути),
      по которому искался soft-эквивалент;
    - ``section_filter_title_hits`` — сколько fallback-хитов совпало с
      названием раздела на уровне сегмента ``section_path``;
    - ``section_filter_fallback_query`` — запрос fallback-поиска (с названием
      раздела).

    В режиме fallback счётчики этапов/каналов описывают именно fallback-поиск,
    потому что он вернул хиты; строгая попытка видна по
    ``section_filter_strict_hits`` и в ``warnings`` результата.
    """

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
    section_filter_status: str = "not_requested"
    section_filter_prefix: str | None = None
    section_filter_strict_hits: int = 0
    section_filter_corpus_matches: int = 0
    section_filter_title: str | None = None
    section_filter_title_hits: int = 0
    section_filter_fallback_query: str | None = None
    stages: dict[str, list[dict[str, Any]] | None] = field(default_factory=dict)
    available: dict[str, bool] = field(default_factory=dict)
    fallback: dict[str, bool] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    def record_stage(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]] | None,
        *,
        sort_by_rank: bool = True,
    ) -> None:
        """Публикует измеренный этап; ``None`` помечает этап недоступным."""
        if items is None:
            self.mark_unavailable(name)
            return
        self.stages[name] = _normalise_stage(items, sort_by_rank=sort_by_rank)
        self.available[name] = True
        self.unavailable.pop(name, None)

    def mark_unavailable(self, name: str, reason: str = "unavailable") -> None:
        """Помечает этап как недоступный, не подменяя его пустым списком."""
        self.stages[name] = None
        self.available[name] = False
        self.unavailable[name] = reason

    def is_stage_available(self, name: str) -> bool:
        return bool(self.available.get(name, False))

    def stage_items(self, name: str) -> list[dict[str, Any]] | None:
        return self.stages.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Представление trace для ``retrieval_calls``/UI без новых счётчиков.

        Плоские счётчики сохранены для существующего UI, ``counts`` повторяет
        их по контракту ``spec_benchmark.md`` §4.3.
        """
        return {
            "retrieval_mode": self.retrieval_mode,
            "query_count": self.query_count,
            "fused_candidates": self.fused_candidates,
            "rerank_candidates": self.rerank_candidates,
            "returned": self.returned,
            "dense_candidates": self.dense_candidates,
            "bm25_candidates": self.bm25_candidates,
            "diversity_removed": self.diversity_removed,
            "distinct_docs": self.distinct_docs,
            "distinct_sections": self.distinct_sections,
            "distinct_years": self.distinct_years,
            "counts": {
                "dense": self.dense_candidates,
                "bm25": self.bm25_candidates,
                "rrf": self.fused_candidates,
                "rerank": self.rerank_candidates,
                "returned": self.returned,
            },
            "stages": {
                name: (None if items is None else [dict(item) for item in items])
                for name, items in self.stages.items()
            },
            "available": dict(self.available),
            "fallback": dict(self.fallback),
            "unavailable": dict(self.unavailable),
        }


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


def _section_fallback_query(query: str, prefix: str) -> str:
    """Добавляет в запрос название раздела (последний сегмент пути).

    ``data/structure`` и ``section_path`` чанков могут использовать разные
    таксономии, поэтому название раздела — единственный переносимый признак:
    префикс пути остаётся только в фильтре и в диагностике, а поиск без
    фильтра получает title, чтобы ранжирование не потеряло тему раздела.
    Запрос не меняется, если название в нём уже есть.
    """
    segments = _segments(prefix)
    if not segments or not query.strip():
        return query
    title = segments[-1]
    if title.casefold() in query.casefold():
        return query
    return f"{title} {query}".strip()


def _section_title(prefix: str) -> str:
    segments = _segments(prefix)
    return segments[-1] if segments else ""


def _title_in_path(chunk: Mapping[str, Any], title: str) -> bool:
    """Название раздела встречается в ``section_path`` как отдельный сегмент.

    Это мягкое соответствие: путь чанка может содержать лишних предков
    (``Сбер в цифрах и фактах/Сбер сегодня``), но лист — тот же раздел.
    """
    if not title:
        return False
    target = title.casefold()
    return target in {
        segment.casefold() for segment in _segments(_value(chunk, "section_path"))
    }


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
                 diversity: bool = False,
                 max_per_doc: int = 2,
                 max_per_section: int = 1) -> None:
        if max_per_doc < 1 or max_per_section < 1:
            raise ValueError("diversity limits must be positive")
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_pool = rerank_pool
        self.top_k = top_k
        self.diversity = diversity
        self.max_per_doc = max_per_doc
        self.max_per_section = max_per_section

    @staticmethod
    def _diversify(
        selected: list[dict[str, Any]],
        limit: int,
        *,
        max_per_doc: int = 2,
        max_per_section: int = 1,
    ) -> tuple[list[dict[str, Any]], int]:
        """Мягкое разнообразие: дубли отодвигаются, но recall не теряется.

        Первый хит сохраняется всегда; затем идут хиты в пределах лимитов на
        раздел/документ; оставшиеся добираются в исходном порядке до ``limit``.
        """
        if limit <= 0:
            return [], len(selected)
        kept: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        section_counts: dict[str, int] = {}
        doc_counts: dict[str, int] = {}
        for index, item in enumerate(selected):
            chunk = item.get("chunk") if isinstance(item, dict) else None
            chunk = chunk if isinstance(chunk, Mapping) else item
            section = _value(chunk, "section_path") or _value(
                chunk, "breadcrumb_contains"
            )
            doc_id = _value(chunk, "doc_id")
            key = str(section) if section else ""
            doc_key = str(doc_id) if doc_id else ""
            section_ok = not key or section_counts.get(key, 0) < max_per_section
            doc_ok = not doc_key or doc_counts.get(doc_key, 0) < max_per_doc
            if index == 0 or (section_ok and doc_ok):
                kept.append(item)
                if key:
                    section_counts[key] = section_counts.get(key, 0) + 1
                if doc_key:
                    doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            else:
                deferred.append(item)
        ordered = kept + deferred
        return ordered[:limit], len(deferred)

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

    def _section_prefix_matches(self, prefix: str) -> int:
        """Сколько чанков индекса лежат под ``prefix`` (без прочих фильтров).

        Ноль означает, что такой таксономии в корпусе нет: ``data/structure`` и
        ``section_path`` чанков расходятся. Значение больше нуля при нулевом
        строгом результате говорит, что путь в корпусе есть, но его отсекли
        другие фильтры (год/документ/kind).
        """
        needle = [segment.casefold() for segment in _segments(prefix)]
        if not needle:
            return 0
        matches = 0
        for chunk in self._chunks():
            path = [
                segment.casefold()
                for segment in _segments(_value(chunk, "section_path"))
            ]
            if path[: len(needle)] == needle:
                matches += 1
        return matches

    def _section_years(self, prefix: str) -> set[int]:
        """Годы, реально представленные чанками под ``prefix``."""
        needle = [segment.casefold() for segment in _segments(prefix)]
        if not needle:
            return set()
        years: set[int] = set()
        for chunk in self._chunks():
            path = [
                segment.casefold()
                for segment in _segments(_value(chunk, "section_path"))
            ]
            if path[: len(needle)] != needle:
                continue
            try:
                years.add(int(_value(chunk, "year")))
            except (TypeError, ValueError):
                continue
        return years

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
        """Возвращает top-k слитых кандидатов и диагностирует отсутствующие годы.

        ``section_path_prefix`` — жёсткий фильтр. Если он задан, но строгий
        поиск не дал ни одного хита:

        * раздел есть в корпусе, а запрошенных лет в нём нет — возвращается
          пустая выдача с предупреждением ``section_year_absent``. Чужие
          разделы не подставляются: пустой ответ честнее ложного;
        * раздела нет в корпусе (расхождение таксономии ``data/structure`` и
          ``section_path`` чанков) — выполняется явный fallback: тот же поиск
          без section-фильтра, но с названием раздела в запросе. Если название
          раздела (лист префикса) найдено в ``section_path`` чанков, остаются
          только такие хиты (title-mapping), иначе берутся все хиты поиска без
          фильтра.

        Fallback никогда не включается для запросов без ``section_path_prefix``
        и не выполняется, если строгий поиск дал хиты; он всегда помечен
        предупреждением ``section_filter_fallback`` и полями
        ``trace.section_filter_*``. Хиты fallback остаются обычными хитами:
        фильтр раздела к ним не применялся, и это видно в диагностике.
        """
        effective = (filters or SearchFilters()).normalized()
        strict = self._retrieve_once(
            query,
            filters=effective,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            queries=queries,
        )
        prefix = effective.section_path_prefix
        if not prefix:
            # Обычный factoid/temporal-поиск: fallback не нужен и не выполняется.
            return strict
        if strict.hits:
            strict.trace.section_filter_status = "applied"
            strict.trace.section_filter_prefix = prefix
            strict.trace.section_filter_strict_hits = len(strict.hits)
            strict.trace.section_filter_corpus_matches = (
                self._section_prefix_matches(prefix)
            )
            return strict
        requested_years = effective.requested_years()
        corpus_matches = self._section_prefix_matches(prefix)
        channels_failed = strict.trace.dense_failed and strict.trace.bm25_failed
        if requested_years and corpus_matches and not channels_failed:
            section_years = self._section_years(prefix)
            if not (requested_years & section_years):
                # Раздел в корпусе есть, но года в нём нет: fallback вернул бы
                # хиты из других разделов и выдал бы их за нужный.
                return self._retrieve_section_year_absent(
                    strict=strict,
                    filters=effective,
                    prefix=prefix,
                    section_matches=corpus_matches,
                    section_years=section_years,
                )
        return self._retrieve_section_fallback(
            query,
            strict=strict,
            filters=effective,
            prefix=prefix,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            queries=queries,
        )

    def _retrieve_section_year_absent(
        self,
        *,
        strict: RetrievalResult,
        filters: SearchFilters,
        prefix: str,
        section_matches: int,
        section_years: set[int],
    ) -> RetrievalResult:
        """Раздел есть, но запрошенных лет в нём нет: пустая честная выдача.

        Fallback здесь запрещён: поиск без section-фильтра вернул бы чанки
        других разделов, и они выглядели бы ответом по запрошенному разделу.
        Предупреждение ``section_year_absent`` и ``missing_years`` явно
        показывают, каких лет не хватает.
        """
        requested = filters.requested_years()
        absent_years = sorted(requested - section_years)
        trace = strict.trace
        trace.section_filter_status = "year_absent"
        trace.section_filter_prefix = prefix
        trace.section_filter_strict_hits = len(strict.hits)
        trace.section_filter_corpus_matches = section_matches
        trace.section_filter_title = _section_title(prefix) or None
        warnings = list(dict.fromkeys(strict.warnings))
        warnings.append(SECTION_YEAR_ABSENT)
        warnings.append(
            "section exists in corpus but has no chunks for years: "
            + ", ".join(map(str, absent_years))
        )
        return RetrievalResult(
            hits=[],
            warnings=list(dict.fromkeys(warnings)),
            missing_years=sorted(set(strict.missing_years) | set(absent_years)),
            effective_filters=filters,
            trace=trace,
        )

    def _retrieve_section_fallback(
        self,
        query: str,
        *,
        strict: RetrievalResult,
        filters: SearchFilters,
        prefix: str,
        top_k: int | None,
        retrieval_mode: str,
        use_reranker: bool,
        queries: Sequence[str] | None,
    ) -> RetrievalResult:
        """Поиск без section-фильтра с явной пометкой в warnings/trace.

        Две ступени, обе честно помечены:

        1. ``fallback_title`` — если в путях чанков есть тот же раздел по
           названию (лист ``section_path``), остаются только такие хиты: это
           mapping ``structure.title`` → ``section_path`` при расхождении
           таксономий;
        2. ``fallback`` — иначе берутся все хиты поиска без section-фильтра.

        Хиты не выдаются за точное попадание в префикс: warning
        ``section_filter_fallback`` и ``trace.section_filter_*`` фиксируют, что
        жёсткий фильтр раздела не сработал.
        """
        fallback_query = _section_fallback_query(query, prefix)
        fallback_filters = replace(filters, section_path_prefix=None)
        fallback = self._retrieve_once(
            fallback_query,
            filters=fallback_filters,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            queries=queries,
        )
        title = _section_title(prefix)
        mapped = [
            hit for hit in fallback.hits if _title_in_path(hit.chunk, title)
        ]
        hits = mapped or fallback.hits
        trace = fallback.trace
        if mapped:
            # Отфильтровали часть fallback-хитов: этап ``returned`` и
            # разнообразие пересчитываются по фактическому результату, иначе
            # trace описывал бы чанки, которых нет в выдаче.
            trace.returned = len(hits)
            trace.stages["returned"] = [
                {
                    "chunk_id": hit.chunk_id,
                    "rank": position,
                    "score": (
                        hit.rerank_score
                        if hit.rerank_score is not None
                        else hit.rrf_score
                    ),
                }
                for position, hit in enumerate(hits, start=1)
            ]
            trace.available["returned"] = True
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
        trace.section_filter_prefix = prefix
        trace.section_filter_strict_hits = len(strict.hits)
        trace.section_filter_corpus_matches = self._section_prefix_matches(prefix)
        trace.section_filter_title = title or None
        trace.section_filter_title_hits = len(mapped)
        trace.section_filter_fallback_query = fallback_query
        if mapped:
            trace.section_filter_status = "fallback_title"
        elif hits:
            trace.section_filter_status = "fallback"
        else:
            trace.section_filter_status = "fallback_empty"
        trace.fallback["section_filter"] = True
        # section_filter — не измеренный этап retrieval, а снятое ограничение.
        trace.available["section_filter"] = False
        trace.unavailable["section_filter"] = "fallback"
        warnings = list(dict.fromkeys([*strict.warnings, *fallback.warnings]))
        warnings.append(SECTION_FILTER_FALLBACK)
        if mapped:
            detail = (
                "section filter matched no chunks; section_path matched by "
                f"title: {title[:160]}"
            )
        else:
            detail = (
                "section filter matched no chunks; searched without "
                f"section_path_prefix: {prefix[:160]}"
            )
        warnings.append(detail)
        return RetrievalResult(
            hits=hits,
            warnings=list(dict.fromkeys(warnings)),
            missing_years=fallback.missing_years,
            effective_filters=fallback_filters,
            trace=trace,
        )

    def _retrieve_once(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Один поиск: BM25 + dense → RRF → опциональный реранкер."""
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
        channel_entries: dict[str, dict[str, dict[str, Any]]] = {
            "dense": {},
            "bm25": {},
        }
        channel_order: dict[str, list[str]] = {"dense": [], "bm25": []}

        def collect(channel: str, hits: Sequence[Mapping[str, Any]]) -> None:
            """Копит лучший ранг/score канала по всем формулировкам."""
            entries = channel_entries[channel]
            for rank, hit in enumerate(hits, start=1):
                try:
                    chunk_id = _chunk_id(hit)
                except ValueError:
                    continue
                score = _score(hit.get("score"))
                if chunk_id not in entries:
                    channel_order[channel].append(chunk_id)
                    entries[chunk_id] = {"rank": rank, "score": score}
                elif rank < entries[chunk_id]["rank"]:
                    entries[chunk_id] = {"rank": rank, "score": score}

        def channel_items(channel: str) -> list[dict[str, Any]]:
            return [
                {
                    "chunk_id": chunk_id,
                    "rank": channel_entries[channel][chunk_id]["rank"],
                    "score": channel_entries[channel][chunk_id]["score"],
                }
                for chunk_id in channel_order[channel]
            ]

        dense_requested = retrieval_mode in {"hybrid", "dense"}
        bm25_requested = retrieval_mode in {"hybrid", "keyword"}
        if dense_requested:
            try:
                for variant in query_list:
                    query_vector = self.embedder.embed_query(variant)
                    hits = list(self.dense_index.search(
                        query_vector, k=self.candidates, eligible=eligible))
                    trace.dense_candidates += len(hits)
                    collect("dense", hits)
                    rankings.append(("dense", hits))
            except Exception as exc:  # noqa: BLE001 — один канал не должен блокировать базовый поиск
                trace.dense_failed = True
                trace.dense_error_type = type(exc).__name__
                warnings.append(f"Dense retrieval failed ({type(exc).__name__})")
                trace.mark_unavailable("dense", "failed")
            else:
                trace.record_stage("dense", channel_items("dense"))
        else:
            # В keyword-режиме dense-канал не вызывается и не заполняется.
            trace.mark_unavailable("dense", "not_requested")
        if bm25_requested:
            try:
                for variant in query_list:
                    hits = list(self.bm25_index.search(
                        variant, k=self.candidates, eligible=eligible))
                    trace.bm25_candidates += len(hits)
                    collect("bm25", hits)
                    rankings.append(("bm25", hits))
            except Exception as exc:  # noqa: BLE001
                trace.bm25_failed = True
                trace.bm25_error_type = type(exc).__name__
                warnings.append(f"BM25 retrieval failed ({type(exc).__name__})")
                trace.mark_unavailable("bm25", "failed")
            else:
                trace.record_stage("bm25", channel_items("bm25"))
        else:
            # В dense-режиме BM25 не вызывается и не заполняется.
            trace.mark_unavailable("bm25", "not_requested")
        trace.fallback["dense"] = trace.dense_failed
        trace.fallback["bm25"] = trace.bm25_failed

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
        failed_channels = [
            failed
            for requested, failed in (
                (dense_requested, trace.dense_failed),
                (bm25_requested, trace.bm25_failed),
            )
            if requested
        ]
        if failed_channels and all(failed_channels):
            # Слияние формально выполнилось, но все входы упали: 0 — не результат.
            trace.mark_unavailable("rrf", "upstream_failed")
        else:
            trace.record_stage("rrf", [
                {
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": fused[chunk_id]["rrf_score"],
                }
                for rank, chunk_id in enumerate(order, start=1)
            ])
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
            trace.mark_unavailable("rerank", "disabled")
        elif self.reranker is not None and candidates:
            try:
                result = self.reranker.rerank(query, candidates)
                if not hasattr(result, "hits"):
                    raise ValueError("reranker must return a result with hits")
                if getattr(result, "used_fallback", False):
                    trace.reranker_used_fallback = True
                    trace.reranker_status = "fallback"
                    # RRF-порядок сохраняется; недоступный этап не выдумываем.
                    trace.mark_unavailable("rerank", "fallback")
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
                    trace.record_stage("rerank", [
                        {
                            "chunk_id": item["chunk_id"],
                            "rank": position,
                            "score": _score(item.get("rerank_score")),
                        }
                        for position, item in enumerate(selected, start=1)
                    ])
            except Exception as exc:  # noqa: BLE001 — реранкер необязателен
                trace.reranker_used_fallback = True
                trace.reranker_status = "fallback"
                trace.reranker_error_type = type(exc).__name__
                trace.mark_unavailable("rerank", "error")
                warnings.append("LLM reranker fallback; preserving RRF order")
        elif self.reranker is not None:
            trace.reranker_status = "empty"
            trace.mark_unavailable("rerank", "no_candidates")
        else:
            trace.mark_unavailable("rerank", "not_configured")
        trace.fallback["reranker"] = trace.reranker_used_fallback

        limit = self.top_k if top_k is None else int(top_k)
        if self.diversity:
            selected, removed = self._diversify(
                selected,
                limit,
                max_per_doc=self.max_per_doc,
                max_per_section=self.max_per_section,
            )
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
        trace.record_stage("returned", [
            {
                "chunk_id": hit.chunk_id,
                "rank": position,
                "score": (hit.rerank_score if hit.rerank_score is not None
                          else hit.rrf_score),
            }
            for position, hit in enumerate(hits, start=1)
        ])
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
