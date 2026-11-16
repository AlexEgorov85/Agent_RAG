"""Обёртка graph-augmented поиска вокруг существующего гибридного ретривера.

Обёртка намеренно не меняет ``HybridRetriever``. Сначала выполняется базовый
поиск, затем отфильтрованные чанки-свидетельства из графа добавляются как
дополнительный список ранжирования:

* базовые хиты сохраняют порядок, оценки и диагностику;
* только графовые чанки добавляются после всех базовых, с сортировкой по
  ``graph_score`` по убыванию, затем по ``chunk_id``;
* дубликаты отбрасываются: чанк, уже возвращённый базой, не добавляется дважды.

Помимо сущностного канала (`MENTIONS`/`RELATED_TO`) обёртка умеет структурное
расширение: верхние базовые хиты дополняются соседями по разделу (`CONTAINS`) и
по странице — не больше ``structure_budget`` на сид. Такие кандидаты получают
``graph_score=STRUCTURE_GRAPH_SCORE`` (ниже любой сущностной находки), поэтому
могут лишь пополнить пул перед реранжированием, но не вытеснить доказанное.

Поддерживаются две политики отбора:

* **по количеству** (по умолчанию): ``top_k`` — жёсткий потолок, при равенстве
  слот всегда достаётся базовому хиту;
* **по важности** (``reranker=...``): базовые и графовые кандидаты реранжируются
  вместе, затем остаются все хиты с ``rerank_score`` не ниже
  ``relevance_threshold``. ``top_k`` остаётся страховочным потолком, поэтому
  ответ может содержать меньше хитов, если релевантных мало.

Без реранкера графовые хиты имеют ``rerank_score=None`` (и
``rerank_rank=None``). Их ``rrf_score`` — синтетическая документированная оценка
``graph_weight / (RRF_CONSTANT + graph_rank)``, сопоставимая с вкладом базового
RRF, хотя чанк не проходил через RRF.

Любой сбой графа (нет/битый граф, нет сущностей, устаревший чанк-свидетельство)
деградирует до предупреждения. В этом случае возвращается только базовый
результат, что соответствует fallback из спеки: «нет графового индекса →
гибридный поиск без графа».

Трейс (``spec_benchmark.md`` §4.3) дополняется этапом ``graph``:

* ``stages["graph"]`` — ``chunk_id``, ``rank``, ``score`` (графовый скор),
  ``path``, ``evidence_chunk_ids`` и ``matched_entities``; дубликаты
  схлопываются по ``chunk_id``, ранги начинаются с 1;
* ``available["graph"]``/``unavailable["graph"]`` — этап либо измерен, либо
  недоступен с причиной: ``no_index``, ``disabled``, ``mode_not_supported``,
  ``graph_failed``, ``no_candidates``;
* счётчики ``graph_candidates``, ``graph_added`` (и ``graph_added_chunk_ids``),
  ``graph_duplicates``, ``graph_matched_entities``, ``graph_failed``.

Provenance свидетельств разделено явно: ``evidence_manifest(result)``
возвращает только чанки, подтверждённые базовым поиском (primary). Графовый
чанк, которого не было в базовой выдаче, попадает в
``graph_evidence_manifest(result)``; его можно цитировать только после
получения через поиск/``get_chunk`` (P7). ``graph_added=0`` — валидный
результат, а не ошибка.

Граф работает только поверх ``retrieval_mode="hybrid"``: в ``dense``/``keyword``
этап отключается с предупреждением ``mode_not_supported``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import networkx as nx

from src.graph.graph_models import GraphRetrievalResult
from src.retrieval.bm25 import tokenize as _tokenize
from src.retrieval.graph_retriever import (
    resolve_entity_ids,
    retrieve_graph,
    structural_neighbors,
)
from src.retrieval.hybrid_retriever import (
    RRF_CONSTANT,
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)
from src.retrieval.reranker import MAX_CANDIDATES as RERANKER_MAX_CANDIDATES

#: Оценки реранкера — калиброванные целые 0..3; значение не ниже
#: ``DEFAULT_RELEVANCE_THRESHOLD`` считается релевантным свидетельством.
DEFAULT_RELEVANCE_THRESHOLD = 2

#: Графовый скор структурных соседей. Ниже любой сущностной находки
#: (прямое упоминание — 1.0, глубина 1 — 0.5, глубина 2 — 0.33), поэтому
#: структурный кандидат никогда не вытесняет доказанный сущностный.
STRUCTURE_GRAPH_SCORE = 0.25

#: ``entity_matcher(query, graph)`` или короткая форма ``entity_matcher(query)``.
#: Может вернуть ID узлов-сущностей или имена/псевдонимы; обёртка разрешает оба.
EntityMatcher = Callable[..., Iterable[str]]

#: Очень частые слова, встречающиеся почти в любом вопросе. Если считать их
#: псевдонимами сущностей (например, «Банк» у «ПАО Сбербанк»), они тянут те же
#: свидетельства в несвязанные запросы, поэтому матчер по умолчанию их игнорирует.
GENERIC_ENTITY_NAMES = frozenset({
    "банк", "банка", "банку", "банке", "банки", "банков",
    "группа", "группы", "группе", "группу", "компания", "компании",
    "рынок", "рынка", "рынке", "отчет", "отчета", "год", "года", "году",
})
MIN_ENTITY_NAME_LENGTH = 3


def normalize_entity_text(value: Any) -> str:
    """Приводит текст к нижнему регистру и схлопывает пробелы для сопоставления имён."""
    return " ".join(str(value).split()).casefold()


def _lemma_text(value: Any) -> str:
    """Нормализует текст в лемматизированные токены (с откатом к поверхностным).

    Общий с BM25 токенизатор делает сопоставление сущностей согласованным с
    keyword-поиском: ``ключевой ставке`` и ``ключевая ставка`` дают одну
    последовательность лемм, поэтому падежные формы русского запроса совпадают
    с сущностями графа.
    """
    return " ".join(_tokenize(str(value)))


def _matchable_name(value: Any) -> str:
    """Возвращает нормализованное имя для сопоставления с запросом (может быть пустым)."""
    normalized = _lemma_text(value)
    if len(normalized) < MIN_ENTITY_NAME_LENGTH:
        return ""
    if normalized in GENERIC_ENTITY_NAMES:
        return ""
    return normalized


def default_entity_matcher(query: str, graph: nx.MultiDiGraph) -> list[str]:
    """Возвращает отсортированные ID узлов ``Entity``, чьё имя/псевдоним есть в ``query``.

    Сопоставление не зависит от регистра, пробелы схлопываются, токены
    лемматизируются общим токенизатором BM25, поэтому падежные формы вроде
    ``ключевой ставке`` совпадают с ``ключевая ставка``. Общие слова (``банк``,
    ``группа`` и т.п.) игнорируются; если одно имя входит в другое, остаётся
    только более длинное (конкретное), чтобы ``Сбербанк`` не совпадал ещё и с
    коротким псевдонимом ``Сбер``.
    """
    normalized_query = _lemma_text(query)
    if not normalized_query:
        return []

    matched: list[tuple[str, str]] = []  # (ID узла-сущности, совпавшее имя)
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("type") != "Entity":
            continue
        names: list[Any] = [attrs.get("canonical_name") or ""]
        aliases = attrs.get("aliases") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        names.extend(aliases)
        candidates = [
            name
            for name in (_matchable_name(item) for item in names)
            if name and name in normalized_query
        ]
        if candidates:
            matched.append((str(node_id), max(candidates, key=len)))

    specific = [
        (node_id, name)
        for node_id, name in matched
        if not any(
            name != other and name in other for _, other in matched
        )
    ]
    return sorted({node_id for node_id, _ in specific})


@dataclass
class GraphRetrievalTrace(RetrievalTrace):
    """Трейс базового поиска, расширенный диагностикой графового этапа.

    Класс наследуется от :class:`RetrievalTrace`, поэтому совместим со всеми
    потребителями, которые читают только базовые счётчики и флаги сбоев.
    Новые поля не дублируют базовые: ``graph_added_chunk_ids`` хранит сами
    идентификаторы добавленных графовых чанков (для provenance), тогда как
    ``graph_added`` остаётся счётчиком.
    """

    graph_matched_entities: list[str] = field(default_factory=list)
    graph_candidates: int = 0
    graph_added: int = 0
    graph_duplicates: int = 0
    graph_skipped_missing_chunks: int = 0
    graph_skipped_filtered: int = 0
    graph_failed: bool = False
    graph_error_type: str | None = None
    graph_depth: int = 0
    graph_results: list[dict[str, Any]] = field(default_factory=list)
    graph_added_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Добавляет к общему trace графовые счётчики и provenance.

        ``counts["graph"]`` — число графовых кандидатов; отдельный блок
        ``graph`` агрегирует диагностику, которую читают UI/benchmark
        (``graph_added``, matched entities, добавленные ``chunk_id``).
        """
        data = super().to_dict()
        data["counts"]["graph"] = self.graph_candidates
        data["graph"] = {
            "candidates": self.graph_candidates,
            "added": self.graph_added,
            "added_chunk_ids": list(self.graph_added_chunk_ids),
            "duplicates": self.graph_duplicates,
            "skipped_missing_chunks": self.graph_skipped_missing_chunks,
            "skipped_filtered": self.graph_skipped_filtered,
            "failed": self.graph_failed,
            "error_type": self.graph_error_type,
            "depth": self.graph_depth,
            "matched_entities": list(self.graph_matched_entities),
        }
        return data


#: Причины, по которым этап ``graph`` не измерен (``available=False``).
GRAPH_UNAVAILABLE_HINTS: dict[str, str] = {
    "no_index": "graph index/artifact is absent; using hybrid retrieval only",
    "disabled": "graph augmentation is disabled (graph_max_results=0)",
    "mode_not_supported": (
        "graph runs only over the hybrid base; dense/keyword disable it"
    ),
    "graph_failed": "graph backend failed; using the hybrid result",
    "no_candidates": "graph found no evidence chunks; using the hybrid result",
}


def graph_unavailable_warning(reason: str, *, error_type: str | None = None) -> str:
    """Собирает единое предупреждение ``graph_unavailable: <reason>``."""
    hint = GRAPH_UNAVAILABLE_HINTS.get(reason, "graph stage is unavailable")
    detail = f"; {error_type}" if error_type else ""
    return f"graph_unavailable: {reason} ({hint}{detail})"


def _copy_trace(base_trace: Any) -> GraphRetrievalTrace:
    """Копирует объявленные поля базового трейса в графовый.

    Словари этапов/доступности копируются отдельно: иначе запись этапа
    ``graph`` изменила бы базовый trace, который может переиспользоваться
    вызывающим кодом.
    """
    trace = GraphRetrievalTrace()
    if base_trace is None:
        return trace
    for item in fields(RetrievalTrace):
        if hasattr(base_trace, item.name):
            setattr(trace, item.name, getattr(base_trace, item.name))
    if isinstance(base_trace, GraphRetrievalTrace):
        for item in fields(GraphRetrievalTrace):
            setattr(trace, item.name, getattr(base_trace, item.name))
    trace.stages = {
        name: None if items is None else [dict(entry) for entry in items]
        for name, items in (trace.stages or {}).items()
    }
    trace.available = dict(trace.available or {})
    trace.fallback = dict(trace.fallback or {})
    trace.unavailable = dict(trace.unavailable or {})
    trace.graph_matched_entities = list(trace.graph_matched_entities or [])
    trace.graph_results = [dict(item) for item in (trace.graph_results or [])]
    trace.graph_added_chunk_ids = list(trace.graph_added_chunk_ids or [])
    return trace


def _accepts_keyword(target: Any, keyword: str) -> bool:
    """Проверяет, принимает ли ``target`` именованный аргумент ``keyword``."""
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or (
            item.name == keyword
            and item.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for item in parameters.values()
    )


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    """Читает поле результата поиска из объекта или словаря."""
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _base_hits(result: Any) -> list[RetrievalHit]:
    """Приводит базовый результат к объектам ``RetrievalHit`` без потери данных."""
    raw = _result_field(result, "hits")
    if raw is None and isinstance(result, (list, tuple)):
        raw = result
    if raw is None:
        raise TypeError("base.retrieve() must return hits or a RetrievalResult")
    return [_as_hit(item) for item in raw]


def _unique_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """Схлопывает свидетельства по ``chunk_id``, сохраняя первое вхождение."""
    seen: set[str] = set()
    unique: list[RetrievalHit] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        unique.append(hit)
    return unique


def _as_hit(item: Any) -> RetrievalHit:
    """Возвращает ``item`` без изменений или приводит словарь/объект к ``RetrievalHit``."""
    if isinstance(item, RetrievalHit):
        return item
    if isinstance(item, Mapping):
        chunk_id = item.get("chunk_id")
        if chunk_id is None:
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping):
                chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("base hit must have a non-empty chunk_id")
        chunk = item.get("chunk")
        if not isinstance(chunk, Mapping):
            chunk = {
                key: value for key, value in item.items()
                if key not in {"chunk_id", "chunk", "dense_rank", "bm25_rank",
                               "dense_score", "bm25_score", "rrf_score",
                               "rerank_score", "rerank_rank"}
            }
        return RetrievalHit(
            chunk_id=chunk_id,
            chunk=chunk,
            dense_rank=item.get("dense_rank"),
            bm25_rank=item.get("bm25_rank"),
            dense_score=item.get("dense_score"),
            bm25_score=item.get("bm25_score"),
            rrf_score=item.get("rrf_score"),
            rerank_score=item.get("rerank_score"),
            rerank_rank=item.get("rerank_rank"),
        )
    chunk_id = getattr(item, "chunk_id", None)
    chunk = getattr(item, "chunk", None)
    if not isinstance(chunk_id, str) or not chunk_id or not isinstance(chunk, Mapping):
        raise ValueError("base hit must expose chunk_id and a chunk mapping")
    return RetrievalHit(
        chunk_id=chunk_id,
        chunk=chunk,
        dense_rank=getattr(item, "dense_rank", None),
        bm25_rank=getattr(item, "bm25_rank", None),
        dense_score=getattr(item, "dense_score", None),
        bm25_score=getattr(item, "bm25_score", None),
        rrf_score=getattr(item, "rrf_score", None),
        rerank_score=getattr(item, "rerank_score", None),
        rerank_rank=getattr(item, "rerank_rank", None),
    )


def _field(chunk: Mapping[str, Any], name: str) -> Any:
    """Читает поле чанка, при необходимости из его ``metadata``."""
    if name in chunk:
        return chunk[name]
    metadata = chunk.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


def _passes_extra_filters(chunk: Mapping[str, Any], filters: SearchFilters) -> bool:
    """Применяет поля ``SearchFilters``, которые ``retrieve_graph`` не видит.

    ``retrieve_graph`` уже ограничивает документ/год/issuer/раздел на узлах
    чанков. ``report_period`` и ``kind`` — метаданные чанка, поэтому проверяются
    здесь, прежде чем графовый кандидат попадёт в результат.
    """
    if filters.report_periods is not None:
        if str(_field(chunk, "report_period")) not in filters.report_periods:
            return False
    if filters.kinds is not None:
        if _field(chunk, "kind") not in filters.kinds:
            return False
    return True


def _distinct_chunk_values(hits: Iterable[RetrievalHit], name: str) -> int:
    """Число различных непустых значений поля чанка в финальной выдаче."""
    values = {
        _field(hit.chunk, name)
        for hit in hits
        if isinstance(hit.chunk, Mapping)
    }
    return len({value for value in values if value is not None})


def _graph_score(item: GraphRetrievalResult) -> float:
    """Возвращает конечную графовую оценку; некорректные значения уходят в конец."""
    try:
        value = float(item.graph_score)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _call_entity_matcher(
    matcher: EntityMatcher, query: str, graph: nx.MultiDiGraph
) -> Iterable[str]:
    """Вызывает матчер с одним или двумя аргументами, не скрывая его реальных ошибок."""
    try:
        signature = inspect.signature(matcher)
    except (TypeError, ValueError):
        return matcher(query, graph)
    positional = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    ]
    accepts_graph = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in positional
    ) or len(positional) >= 2
    return matcher(query, graph) if accepts_graph else matcher(query)


class GraphAugmentedRetriever:
    """Добавляет графовые чанки-свидетельства к упорядоченным хитам базового ретривера.

    Args:
        base: Объект с ``retrieve(query, filters=..., top_k=...)``, возвращающий
            ``RetrievalResult`` (или совместимый объект с ``hits``).
        graph: ``networkx.MultiDiGraph`` из ``src.graph.graph_builder``.
            ``None`` или пустой граф эквивалентны отсутствию graph-индекса:
            возвращается базовый результат с ``unavailable["graph"]="no_index"``.
        chunk_store: Отображение ``chunk_id -> словарь чанка`` для материализации
            графовых свидетельств. Отсутствующие ID пропускаются, не выдумываются.
        entity_matcher: Необязательный callable ``(query, graph) -> entity IDs`` или
            ``(query) -> entity IDs/names``. По умолчанию канонические имена и
            псевдонимы сопоставляются с нормализованным запросом.
        graph_depth: Максимальная глубина обхода графа (по умолчанию 2, как в спеке).
        graph_max_results: Максимум графовых кандидатов до слияния.
        graph_weight: Вес синтетического вклада графового списка в RRF.
        include_relation_evidence: При ``False`` (по умолчанию) добавляются только
            чанки, напрямую упоминающие найденные сущности. Это не даёт через
            ``RELATED_TO`` притянуть несвязанные свидетельства о соседних
            сущностях; ``True`` включает их.
        reranker: Необязательный объект с ``rerank(query, hits) -> RerankResult``
            (существующий ``LLMReranker``). Если задан, базовые и графовые
            кандидаты реранжируются вместе и могут вытеснять друг друга. Без него
            обёртка только добавляет хиты и считает по ``top_k``.
        relevance_threshold: Минимальный ``rerank_score`` (0..3), сохраняемый в
            режиме важности. ``None`` отключает фильтрацию (остаётся только
            потолок). Требует ``reranker``.
        max_candidates: Верхняя граница пула кандидатов для реранкера; по
            умолчанию — лимит самого реранкера.
        base_candidate_k: Сколько базовых кандидатов запрашивать, когда обёртка
            сама реранжирует. По умолчанию ``max_candidates``, чтобы решал
            бюджет, а не произвольный top-5.
        structure_budget: Сколько структурных соседей добавлять на один
            базовый хит (0 — канал выключен). Соседи берутся по рёбрам
            ``CONTAINS`` раздела и по странице, ближайшие первыми.
        structure_seed_k: Сколько верхних базовых хитов служат сидами для
            структурного расширения.
        structure_max_results: Потолок структурных кандидатов на запрос.
    """

    def __init__(
        self,
        base: Any,
        graph: nx.MultiDiGraph | None,
        chunk_store: Mapping[str, Mapping[str, Any]] | None,
        *,
        entity_matcher: EntityMatcher | None = None,
        graph_depth: int = 2,
        graph_max_results: int = 10,
        graph_weight: float = 0.5,
        include_relation_evidence: bool = False,
        reranker: Any | None = None,
        relevance_threshold: int | None = None,
        max_candidates: int = RERANKER_MAX_CANDIDATES,
        base_candidate_k: int | None = None,
        structure_budget: int = 0,
        structure_seed_k: int = 5,
        structure_max_results: int = 8,
    ) -> None:
        if graph_depth < 0:
            raise ValueError("graph_depth must be non-negative")
        if graph_max_results < 0:
            raise ValueError("graph_max_results must be non-negative")
        if structure_budget < 0:
            raise ValueError("structure_budget must be non-negative")
        if structure_seed_k < 0:
            raise ValueError("structure_seed_k must be non-negative")
        if structure_max_results < 0:
            raise ValueError("structure_max_results must be non-negative")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError("max_candidates must be a positive integer")
        if base_candidate_k is not None:
            if (
                isinstance(base_candidate_k, bool)
                or not isinstance(base_candidate_k, int)
                or base_candidate_k < 0
            ):
                raise ValueError(
                    "base_candidate_k must be a non-negative integer or None")
        if relevance_threshold is not None:
            if reranker is None:
                raise ValueError("relevance_threshold requires a reranker")
            if (
                isinstance(relevance_threshold, bool)
                or not isinstance(relevance_threshold, int)
                or not 0 <= relevance_threshold <= 3
            ):
                raise ValueError("relevance_threshold must be an integer from 0 to 3")
        if (
            isinstance(graph_weight, bool)
            or not isinstance(graph_weight, (int, float))
            or not math.isfinite(float(graph_weight))
            or graph_weight < 0
        ):
            raise ValueError("graph_weight must be a finite non-negative number")
        self.base = base
        self.graph = graph
        self.chunk_store = chunk_store
        self.entity_matcher = entity_matcher
        self.graph_depth = int(graph_depth)
        self.graph_max_results = int(graph_max_results)
        self.graph_weight = float(graph_weight)
        self.include_relation_evidence = bool(include_relation_evidence)
        self.reranker = reranker
        self.relevance_threshold = relevance_threshold
        self.max_candidates = int(max_candidates)
        self.structure_budget = int(structure_budget)
        self.structure_seed_k = int(structure_seed_k)
        self.structure_max_results = int(structure_max_results)
        if base_candidate_k is None:
            self.base_candidate_k = self.max_candidates if reranker is not None else 0
        else:
            self.base_candidate_k = int(base_candidate_k)
        base_retrieve = getattr(base, "retrieve", None)
        self._base_accepts_mode = _accepts_keyword(base_retrieve, "retrieval_mode")
        self._base_accepts_queries = _accepts_keyword(base_retrieve, "queries")

    def _graph_is_empty(self) -> bool:
        """Отличает отсутствующий/пустой граф от сломанного backend'а."""
        if self.graph is None:
            return True
        try:
            return self.graph.number_of_nodes() == 0
        except Exception:  # noqa: BLE001 — объект может не быть графом вовсе
            return False

    def _graph_unavailable_reason(self, retrieval_mode: str) -> str | None:
        """Возвращает причину отключения графа до его вызова, если она есть."""
        if retrieval_mode != "hybrid":
            # Граф добавляет свидетельства только к гибридной базе.
            return "mode_not_supported"
        if self.graph_max_results == 0:
            return "disabled"
        if self._graph_is_empty():
            return "no_index"
        return None

    def _retrieve_base(
        self,
        query: str,
        *,
        filters: SearchFilters | None,
        top_k: int | None,
        retrieval_mode: str,
        queries: Sequence[str] | None,
    ) -> Any:
        """Вызывает базовый поиск, пробрасывая только поддерживаемые аргументы."""
        kwargs: dict[str, Any] = {"filters": filters}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if self._base_accepts_mode:
            kwargs["retrieval_mode"] = retrieval_mode
        if queries and self._base_accepts_queries:
            kwargs["queries"] = list(queries)
        return self.base.retrieve(query, **kwargs)

    def _match_entity_ids(self, query: str) -> list[str]:
        """Разрешает упоминания запроса в существующие ID узлов-сущностей."""
        matcher = self.entity_matcher or default_entity_matcher
        values = _call_entity_matcher(matcher, query, self.graph)
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]

        entity_ids: set[str] = set()
        for value in values:
            text = str(value)
            if not text:
                continue
            if text in self.graph and self.graph.nodes[text].get("type") == "Entity":
                entity_ids.add(text)
            else:
                entity_ids.update(resolve_entity_ids(self.graph, [text]))
        return sorted(entity_ids)

    def _load_evidence(
        self,
        results: Iterable[GraphRetrievalResult],
        filters: SearchFilters,
        trace: GraphRetrievalTrace,
    ) -> list[tuple[GraphRetrievalResult, dict[str, Any]]]:
        """Загружает только существующие и подходящие под фильтры чанки-свидетельства."""
        loaded: dict[str, tuple[GraphRetrievalResult, dict[str, Any]]] = {}
        for item in results:
            chunk_id = str(getattr(item, "chunk_id", "") or "")
            if not chunk_id:
                trace.graph_skipped_missing_chunks += 1
                continue
            try:
                chunk = self.chunk_store.get(chunk_id)
            except AttributeError:
                chunk = None
            if not isinstance(chunk, Mapping):
                trace.graph_skipped_missing_chunks += 1
                continue
            if not _passes_extra_filters(chunk, filters):
                trace.graph_skipped_filtered += 1
                continue
            previous = loaded.get(chunk_id)
            if previous is None or _graph_score(item) > _graph_score(previous[0]):
                loaded[chunk_id] = (item, dict(chunk))

        return sorted(
            loaded.values(),
            key=lambda pair: (-_graph_score(pair[0]), pair[0].chunk_id),
        )

    def _structure_candidates(
        self,
        base_hits: Sequence[RetrievalHit],
        filters: SearchFilters,
        *,
        excluded: set[str],
    ) -> list[GraphRetrievalResult]:
        """Структурные соседи верхних базовых хитов (раздел + страница).

        Канал отвечает на «что рядом с этим свидетельством» и не требует
        совпадения сущностей, поэтому работает и на запросах без имён. Скор
        намеренно ниже любого сущностного: структурный кандидат добавляется к
        пулу, а не вытесняет доказанный.
        """
        if self.structure_budget <= 0 or self.structure_max_results <= 0:
            return []
        seeds = [
            str(hit.chunk_id) for hit in base_hits[: self.structure_seed_k]
            if str(getattr(hit, "chunk_id", "") or "")
        ]
        if not seeds:
            return []
        seen = set(excluded)
        results: list[GraphRetrievalResult] = []
        for seed in seeds:
            neighbours = structural_neighbors(
                self.graph,
                seed,
                budget=self.structure_budget,
                doc_ids=filters.allowed_doc_ids,
                years=filters.requested_years() or None,
                issuers=filters.issuers,
                section_path=filters.section_path_prefix,
            )
            for neighbour in neighbours:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                results.append(GraphRetrievalResult(
                    chunk_id=neighbour,
                    graph_score=STRUCTURE_GRAPH_SCORE,
                    matched_entities=(),
                    path=self._structure_path(seed, neighbour),
                    evidence_chunk_ids=(seed, neighbour),
                ))
                if len(results) >= self.structure_max_results:
                    return results
        return results

    def _structure_path(self, seed: str, neighbour: str) -> tuple[str, ...]:
        """Путь-объяснение: сид → общий раздел (если есть) → сосед."""
        seed_node = f"chunk:{seed}"
        sections = sorted(
            parent for parent in self.graph.predecessors(seed_node)
            if self.graph.nodes[parent].get("type") == "Section"
        )
        return (seed_node, *sections, f"chunk:{neighbour}")

    @staticmethod
    def _diagnostics(results: Iterable[GraphRetrievalResult]) -> list[dict[str, Any]]:
        """Возвращает JSON-безопасные объяснения графовых кандидатов для трейса."""
        return [
            {
                "chunk_id": str(item.chunk_id),
                "graph_score": _graph_score(item),
                "matched_entities": list(item.matched_entities),
                "path": list(item.path),
                "evidence_chunk_ids": list(item.evidence_chunk_ids),
            }
            for item in results
        ]

    @staticmethod
    def _graph_stage_entries(
        results: Iterable[GraphRetrievalResult],
    ) -> list[dict[str, Any]]:
        """Собирает этап ``graph``: лучший скор на ``chunk_id`` и путь-объяснение.

        ``record_stage`` из базового trace оставляет только ``chunk_id/rank/score``,
        поэтому этап публикуется здесь же: для graph важны ``path`` и
        ``evidence_chunk_ids`` (``spec_benchmark.md`` §4.3).
        """
        entries: dict[str, dict[str, Any]] = {}
        first_seen: dict[str, int] = {}
        for position, item in enumerate(results):
            chunk_id = str(getattr(item, "chunk_id", "") or "")
            if not chunk_id:
                continue
            score = _graph_score(item)
            if chunk_id not in entries:
                first_seen[chunk_id] = position
                entries[chunk_id] = {
                    "chunk_id": chunk_id,
                    "rank": position + 1,
                    "score": score,
                    "path": list(item.path),
                    "evidence_chunk_ids": list(item.evidence_chunk_ids),
                    "matched_entities": list(item.matched_entities),
                }
            elif score > entries[chunk_id]["score"]:
                entries[chunk_id]["score"] = score
                entries[chunk_id]["path"] = list(item.path)
                entries[chunk_id]["evidence_chunk_ids"] = list(
                    item.evidence_chunk_ids)
        ordered = sorted(
            entries.values(),
            key=lambda entry: (-entry["score"], first_seen[entry["chunk_id"]]),
        )
        for rank, entry in enumerate(ordered, start=1):
            entry["rank"] = rank
        return ordered

    def _publish_graph_stage(
        self,
        trace: GraphRetrievalTrace,
        results: Iterable[GraphRetrievalResult],
    ) -> None:
        """Помечает этап ``graph`` измеренным (возможно, с пустым списком)."""
        trace.stages["graph"] = self._graph_stage_entries(results)
        trace.available["graph"] = True
        trace.unavailable.pop("graph", None)

    @staticmethod
    def _candidate_pool(
        base_hits: list[RetrievalHit],
        graph_hits: list[RetrievalHit],
        limit: int,
    ) -> list[RetrievalHit]:
        """Чередует базовые и графовые кандидаты, чтобы ни одна сторона не голодала."""
        pool: list[RetrievalHit] = []
        seen: set[str] = set()
        left = right = 0
        while len(pool) < limit and (left < len(base_hits) or right < len(graph_hits)):
            if left < len(base_hits):
                hit = base_hits[left]
                left += 1
                if hit.chunk_id not in seen:
                    pool.append(hit)
                    seen.add(hit.chunk_id)
            if len(pool) < limit and right < len(graph_hits):
                hit = graph_hits[right]
                right += 1
                if hit.chunk_id not in seen:
                    pool.append(hit)
                    seen.add(hit.chunk_id)
        return pool

    def _rerank_merged(
        self,
        query: str,
        hits: list[RetrievalHit],
        trace: GraphRetrievalTrace,
        warnings: list[str],
    ) -> list[RetrievalHit]:
        """Реранжирует базовые и графовые кандидаты вместе (режим важности)."""
        payloads = []
        for hit in hits:
            chunk = hit.chunk if isinstance(hit.chunk, Mapping) else {}
            text = chunk.get("text", "")
            payloads.append({
                "chunk_id": hit.chunk_id,
                "text": text if isinstance(text, str) else str(text),
            })
        try:
            result = self.reranker.rerank(query, payloads)
        except Exception as exc:  # noqa: BLE001 — поиск не должен блокироваться
            trace.reranker_status = "error"
            trace.reranker_error_type = type(exc).__name__
            warnings.append(
                "Graph reranker failed; using merged RRF order "
                f"({trace.reranker_error_type})"
            )
            return hits

        used_fallback = bool(getattr(result, "used_fallback", False))
        trace.reranker_used_fallback = used_fallback
        trace.reranker_status = "fallback" if used_fallback else "used"
        note = getattr(result, "warning", None)
        if note:
            warnings.append(str(note))

        by_id = {hit.chunk_id: hit for hit in hits}
        reranked: list[RetrievalHit] = []
        for entry in getattr(result, "hits", []) or []:
            chunk_id = str(entry.get("chunk_id", ""))
            original = by_id.get(chunk_id)
            if original is None:
                continue
            reranked.append(replace(
                original,
                rerank_score=entry.get("rerank_score"),
                rerank_rank=entry.get("rerank_rank"),
            ))
        # Сохраняем кандидатов, которых реранкер не вернул, вместо тихой потери.
        seen = {hit.chunk_id for hit in reranked}
        reranked.extend(hit for hit in hits if hit.chunk_id not in seen)
        return reranked

    def retrieve(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        retrieval_mode: str = "hybrid",
        queries: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Возвращает базовые хиты, за которыми идут новые графовые чанки-свидетельства.

        ``retrieval_mode``/``queries`` пробрасываются в базовый retriever, если
        он их поддерживает. Граф работает только поверх ``hybrid``: в
        ``dense``/``keyword`` этап отключается с ``mode_not_supported``.
        Любой сбой графа деградирует до базового результата, не роняя поиск.
        """
        if top_k is not None and (isinstance(top_k, bool) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if retrieval_mode not in {"hybrid", "dense", "keyword"}:
            raise ValueError("invalid retrieval_mode")

        # В режиме важности запрашиваем у базы более широкий пул, чтобы решали
        # реранкер и бюджет токенов, а не произвольное фиксированное число.
        base_top_k = top_k
        if self.reranker is not None:
            base_top_k = max(
                int(top_k) if top_k is not None else 0,
                self.base_candidate_k,
            )
        base_result = self._retrieve_base(
            query,
            filters=filters,
            top_k=base_top_k,
            retrieval_mode=retrieval_mode,
            queries=queries,
        )
        hits = _unique_hits(_base_hits(base_result))
        warnings = list(_result_field(base_result, "warnings") or [])
        missing_years = list(_result_field(base_result, "missing_years") or [])
        trace = _copy_trace(_result_field(base_result, "trace"))
        # Режим запроса — это то, что просил вызывающий; baseline мог его не
        # поддерживать, но graph-этап отключается именно по запрошенному режиму.
        trace.retrieval_mode = retrieval_mode

        effective = filters if isinstance(filters, SearchFilters) else _result_field(
            base_result, "effective_filters")
        if not isinstance(effective, SearchFilters):
            effective = SearchFilters()

        base_ids = {hit.chunk_id for hit in hits}
        graph_only: list[RetrievalHit] = []
        unavailable_reason = self._graph_unavailable_reason(retrieval_mode)
        if unavailable_reason is not None:
            trace.mark_unavailable("graph", unavailable_reason)
            warnings.append(graph_unavailable_warning(unavailable_reason))
        else:
            try:
                trace.graph_depth = self.graph_depth
                entity_ids = self._match_entity_ids(query)
                trace.graph_matched_entities = list(entity_ids)
                graph_results: list[GraphRetrievalResult] = []
                if entity_ids:
                    graph_results = list(retrieve_graph(
                        self.graph,
                        entity_ids,
                        doc_ids=effective.allowed_doc_ids,
                        years=effective.requested_years() or None,
                        issuers=effective.issuers,
                        section_path=effective.section_path_prefix,
                        max_depth=self.graph_depth,
                        max_results=self.graph_max_results,
                        include_relation_evidence=self.include_relation_evidence,
                    ))
                if self.structure_budget > 0:
                    # Структурный канал не зависит от найденных сущностей: он
                    # расширяет верхние базовые хиты по разделу и странице.
                    graph_results.extend(self._structure_candidates(
                        hits,
                        effective,
                        excluded=base_ids | {item.chunk_id for item in graph_results},
                    ))
                trace.graph_candidates = len(graph_results)
                trace.graph_results = self._diagnostics(graph_results)
                if graph_results:
                    self._publish_graph_stage(trace, graph_results)
                else:
                    # Нет кандидатов — это не измеренный «0 результатов», а
                    # отсутствие graph-свидетельств для данного запроса.
                    trace.mark_unavailable("graph", "no_candidates")

                loaded = self._load_evidence(graph_results, effective, trace)
                for rank, (item, chunk) in enumerate(loaded, start=1):
                    if item.chunk_id in base_ids:
                        trace.graph_duplicates += 1
                        continue
                    graph_only.append(RetrievalHit(
                        chunk_id=item.chunk_id,
                        chunk=chunk,
                        rrf_score=self.graph_weight / (RRF_CONSTANT + rank),
                        rerank_score=None,
                        rerank_rank=None,
                    ))
            except Exception as exc:  # noqa: BLE001 - graph must never block baseline
                trace.graph_failed = True
                trace.graph_error_type = type(exc).__name__
                trace.graph_candidates = 0
                trace.graph_results = []
                graph_only = []
                trace.mark_unavailable("graph", "graph_failed")
                warnings.append("graph_candidates=0")
                warnings.append(graph_unavailable_warning(
                    "graph_failed", error_type=trace.graph_error_type))

        # fallback["graph"] фиксирует, что результат собран без graph-этапа.
        trace.fallback["graph"] = not trace.is_stage_available("graph")

        merged = list(hits) + graph_only
        graph_only_ids = {hit.chunk_id for hit in graph_only}

        if self.reranker is not None and merged:
            pool = self._candidate_pool(hits, graph_only, self.max_candidates)
            merged = self._rerank_merged(query, pool, trace, warnings)
            if self.relevance_threshold is not None:
                if any(hit.rerank_score is None for hit in merged):
                    # Оценки из fallback/пропущенные несопоставимы с порогом;
                    # оставляем их, чтобы не терять свидетельства.
                    kept = [
                        hit for hit in merged
                        if hit.rerank_score is None
                        or hit.rerank_score >= self.relevance_threshold
                    ]
                    if len(kept) != len(merged):
                        warnings.append(
                            "relevance_threshold applied partially "
                            "(some candidates were not scored)"
                        )
                    merged = kept
                else:
                    merged = [
                        hit for hit in merged
                        if hit.rerank_score >= self.relevance_threshold
                    ]
                threshold = self.relevance_threshold
                if not merged:
                    warnings.append(
                        f"relevance_threshold={threshold} kept no hits"
                    )
                else:
                    warnings.append(
                        f"relevance_threshold={threshold} kept {len(merged)} hits"
                    )

        if top_k is not None:
            merged = merged[:int(top_k)]
        merged = _unique_hits(merged)
        trace.graph_added_chunk_ids = [
            hit.chunk_id for hit in merged if hit.chunk_id in graph_only_ids
        ]
        trace.graph_added = len(trace.graph_added_chunk_ids)
        trace.returned = len(merged)
        # Обёртка меняет финальный top-k, поэтому этап „returned“ и
        # разнообразие пересчитываются по фактическому результату.
        trace.stages["returned"] = [
            {
                "chunk_id": hit.chunk_id,
                "rank": position,
                "score": (hit.rerank_score if hit.rerank_score is not None
                          else hit.rrf_score),
            }
            for position, hit in enumerate(merged, start=1)
        ]
        trace.available["returned"] = True
        trace.unavailable.pop("returned", None)
        trace.distinct_docs = _distinct_chunk_values(merged, "doc_id")
        trace.distinct_sections = _distinct_chunk_values(merged, "section_path")
        trace.distinct_years = _distinct_chunk_values(merged, "year")

        if not trace.graph_failed:
            warnings.append(f"graph_candidates={trace.graph_candidates}")
            if not trace.graph_matched_entities:
                warnings.append("graph_matched_entities=0")
            if trace.graph_duplicates:
                warnings.append(f"graph_duplicates={trace.graph_duplicates}")
            if trace.graph_skipped_missing_chunks:
                warnings.append(
                    "graph_skipped_missing_chunks="
                    f"{trace.graph_skipped_missing_chunks}"
                )
            if trace.graph_skipped_filtered:
                warnings.append(
                    f"graph_skipped_filtered={trace.graph_skipped_filtered}"
                )
            warnings.append(f"graph_added={trace.graph_added}")

        return RetrievalResult(
            hits=merged,
            warnings=list(dict.fromkeys(warnings)),
            missing_years=missing_years,
            effective_filters=effective,
            trace=trace,
        )

    # --- provenance -------------------------------------------------------
    #
    # ``retrieve`` возвращает graph-кандидатов вместе с базовыми хитами, но
    # итоговый evidence manifest должен содержать только подтверждённые
    # primary чанки. Графовый чанк, которого не было в базовой выдаче, живёт
    # в ``graph_evidence_manifest`` до получения через поиск/``get_chunk``
    # (P7, AC-P7-3/AC-P7-4).

    @staticmethod
    def _graph_only_ids(result: Any) -> set[str]:
        """ID графовых чанков, вошедших в результат помимо базовой выдачи."""
        values = getattr(_result_field(result, "trace"),
                         "graph_added_chunk_ids", None)
        if not isinstance(values, (list, tuple, set, frozenset)):
            return set()
        return {str(value) for value in values if str(value)}

    def primary_evidence_hits(self, result: Any) -> list[RetrievalHit]:
        """Хиты, подтверждённые базовым поиском; их можно цитировать сразу."""
        graph_only = self._graph_only_ids(result)
        return [
            hit for hit in _unique_hits(_base_hits(result))
            if hit.chunk_id not in graph_only
        ]

    def graph_evidence_hits(self, result: Any) -> list[RetrievalHit]:
        """Хиты, найденные только графом; цитировать можно после ``get_chunk``."""
        graph_only = self._graph_only_ids(result)
        if not graph_only:
            return []
        return [
            hit for hit in _unique_hits(_base_hits(result))
            if hit.chunk_id in graph_only
        ]

    def evidence_manifest(self, result: Any) -> list[str]:
        """Evidence manifest из подтверждённых primary ``chunk_id`` без дублей."""
        return [hit.chunk_id for hit in self.primary_evidence_hits(result)]

    def graph_evidence_manifest(self, result: Any) -> list[str]:
        """``chunk_id`` graph-only свидетельств, ожидающих подтверждения."""
        return [hit.chunk_id for hit in self.graph_evidence_hits(result)]
