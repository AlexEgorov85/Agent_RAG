"""Чистые метрики benchmark: recall, MRR, nDCG, coverage и loss attribution.

Все функции принимают уже нормализованные данные и не ходят в retrieval/LLM.
Если метрика не измерялась (нет gold-групп, нет этапа), возвращается ``None``,
а не ``0.0``: ноль — это измеренное отсутствие попадания, ``None`` — отсутствие
измерения.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def gold_groups(gold_evidence: Any) -> list[dict[str, Any]]:
    """Привести ``gold_evidence`` к списку групп с непустым ``any_of``."""
    if not isinstance(gold_evidence, Mapping):
        return []
    groups = gold_evidence.get("groups")
    if not isinstance(groups, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            continue
        alternatives = group.get("any_of")
        if not isinstance(alternatives, list):
            continue
        chunk_ids = [
            str(chunk_id)
            for chunk_id in alternatives
            if isinstance(chunk_id, str) and chunk_id
        ]
        if not chunk_ids:
            continue
        normalized.append(
            {
                "id": str(group.get("id") or f"group_{index}"),
                "any_of": list(dict.fromkeys(chunk_ids)),
                "section": group.get("section"),
            }
        )
    return normalized


def unique_ids(chunk_ids: Iterable[Any]) -> list[str]:
    """Уникальные chunk_id с сохранением порядка."""
    result: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        value = str(chunk_id)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def covered_groups(groups: Sequence[Mapping[str, Any]], chunk_ids: Iterable[Any]) -> set[int]:
    """Индексы групп, покрытых хотя бы одним альтернативным chunk_id."""
    ranked = set(unique_ids(chunk_ids))
    return {
        index
        for index, group in enumerate(groups)
        if set(group.get("any_of", [])) & ranked
    }


def coverage_ratio(covered: int, total: int) -> float | None:
    """Знаменатель coverage: пустой набор не превращается в ноль."""
    if total <= 0:
        return None
    return covered / total


def recall_at_k(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[Any],
    k: int,
) -> float | None:
    """Доля gold-групп, покрытых первыми ``k`` уникальными чанками."""
    if not groups:
        return None
    if k < 1:
        raise ValueError("k must be positive")
    ranked = unique_ids(ranked_ids)[:k]
    return coverage_ratio(len(covered_groups(groups, ranked)), len(groups))


def first_relevant_rank(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[Any],
) -> int | None:
    """Ранг первого чанка, покрывающего любую gold-группу; 1-based."""
    ranked = unique_ids(ranked_ids)
    return next(
        (
            rank
            for rank, chunk_id in enumerate(ranked, start=1)
            if any(chunk_id in group.get("any_of", []) for group in groups)
        ),
        None,
    )


def mrr(groups: Sequence[Mapping[str, Any]], ranked_ids: Sequence[Any]) -> float | None:
    """Mean reciprocal rank первого доказательства."""
    if not groups:
        return None
    rank = first_relevant_rank(groups, ranked_ids)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(
    groups: Sequence[Mapping[str, Any]],
    ranked_ids: Sequence[Any],
    k: int = 5,
) -> float | None:
    """Бинарный nDCG@k по gold-группам.

    Группа даёт gain один раз: первый чанк из ``any_of`` покрывает её, повторные
    альтернативы не увеличивают DCG. Это соответствует semantics
    ``gold_evidence.groups``, где альтернативы равнозначны.
    """
    if not groups:
        return None
    if k < 1:
        raise ValueError("k must be positive")
    ranked = unique_ids(ranked_ids)[:k]
    covered: set[int] = set()
    dcg = 0.0
    for position, chunk_id in enumerate(ranked, start=1):
        newly_covered = {
            index
            for index, group in enumerate(groups)
            if index not in covered and chunk_id in group.get("any_of", [])
        }
        if newly_covered:
            dcg += 1.0 / math.log2(position + 1)
            covered.update(newly_covered)
    ideal_hits = min(k, len(groups))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    if idcg == 0:
        return None
    return dcg / idcg


def evidence_coverage(
    groups: Sequence[Mapping[str, Any]],
    manifest_ids: Iterable[Any] | None,
) -> float | None:
    """Доля gold-групп в финальном ``evidence_manifest``."""
    if not groups:
        return None
    if manifest_ids is None:
        return None
    return coverage_ratio(len(covered_groups(groups, manifest_ids)), len(groups))


def context_coverage(
    groups: Sequence[Mapping[str, Any]],
    context_ids: Iterable[Any] | None,
    *,
    context_measured: bool = True,
) -> float | None:
    """Доля gold-групп в контексте; ``None``, если контекст не наблюдался."""
    if not groups or not context_measured or context_ids is None:
        return None
    return coverage_ratio(len(covered_groups(groups, context_ids)), len(groups))


def group_loss(
    group: Mapping[str, Any],
    *,
    stages: Mapping[str, Sequence[str] | None],
    available: Mapping[str, bool] | None,
    context_ids: Sequence[str] | None,
    cited_ids: Sequence[str] | None,
    known_chunk_ids: set[str] | None = None,
    context_measured: bool = True,
) -> str:
    """Определить этап потери одной gold-группы.

    Возможные значения: ``not_in_corpus``, ``never_retrieved``,
    ``lost_at_fusion``, ``lost_at_rerank``, ``lost_at_context``, ``not_cited``,
    ``ok`` и ``not_measured`` (если нужные этапы недоступны).

    Классификация идёт от поздних этапов к ранним: если gold-чанк дошёл до
    ``returned``/``context``, потеря на fusion/rerank не заявляется, даже если
    промежуточный этап не был измерен.
    """
    alternatives = {str(chunk_id) for chunk_id in group.get("any_of", [])}
    if not alternatives:
        return "not_measured"
    if known_chunk_ids is not None and not (alternatives & known_chunk_ids):
        return "not_in_corpus"

    availability = dict(available or {})
    stage_sets = {
        name: None if items is None else set(str(item) for item in items)
        for name, items in stages.items()
    }

    def measured(name: str) -> bool:
        return bool(
            availability.get(name, stage_sets.get(name) is not None)
        )

    def present(name: str) -> bool:
        values = stage_sets.get(name)
        return bool(values is not None and values & alternatives)

    measured_stages = [
        name
        for name in ("dense", "bm25", "graph", "rrf", "rerank", "returned")
        if measured(name)
    ]
    if measured_stages and not any(present(name) for name in measured_stages):
        return "never_retrieved"
    if not measured_stages:
        return "not_measured"

    # Поздние этапы важнее ранних: если фрагмент дошёл до них, он не потерян
    # на fusion/rerank.
    if measured("returned") and not present("returned"):
        if measured("rerank") and present("rerank"):
            return "lost_at_rerank"
        if measured("rrf") and present("rrf"):
            return "lost_at_rerank"
        if (measured("dense") and present("dense")) or (
            measured("bm25") and present("bm25")
        ):
            return "lost_at_fusion"
        return "not_measured"

    context_available = context_measured and context_ids is not None
    if context_available and present("returned") and not (
        alternatives & set(str(item) for item in context_ids)
    ):
        return "lost_at_context"

    if measured("rerank") and not present("rerank"):
        if measured("rrf") and present("rrf"):
            return "lost_at_rerank"
        if not present("returned"):
            return "lost_at_rerank"
    if measured("rrf") and not present("rrf"):
        if present("dense") or present("bm25"):
            return "lost_at_fusion"

    if cited_ids is not None:
        cited = set(str(item) for item in cited_ids)
        context_or_returned = (
            set(str(item) for item in context_ids or [])
            if context_available
            else stage_sets.get("returned") or set()
        )
        if alternatives & context_or_returned and not (alternatives & cited):
            return "not_cited"
    return "ok"


LOSS_PRECEDENCE: tuple[str, ...] = (
    "not_in_corpus",
    "never_retrieved",
    "lost_at_fusion",
    "lost_at_rerank",
    "lost_at_context",
    "not_cited",
    "ok",
)


def worst_loss(codes: Iterable[str]) -> str | None:
    """Самый ранний по цепочке этап потери среди групп."""
    values = [code for code in codes if code in LOSS_PRECEDENCE]
    if not values:
        return None
    return min(values, key=LOSS_PRECEDENCE.index)


def loss_attribution(
    groups: Sequence[Mapping[str, Any]],
    *,
    stages: Mapping[str, Sequence[str] | None],
    available: Mapping[str, bool] | None = None,
    context_ids: Sequence[str] | None = None,
    cited_ids: Sequence[str] | None = None,
    known_chunk_ids: set[str] | None = None,
    context_measured: bool = True,
) -> dict[str, Any]:
    """Loss attribution по всем gold-группам и общий ``loss_stage``."""
    if not groups:
        return {
            "loss_stage": None,
            "groups": [],
            "not_measured": ["loss_stage"],
        }
    group_rows = []
    for group in groups:
        code = group_loss(
            group,
            stages=stages,
            available=available,
            context_ids=context_ids,
            cited_ids=cited_ids,
            known_chunk_ids=known_chunk_ids,
            context_measured=context_measured,
        )
        group_rows.append({"group": group.get("id"), "loss_stage": code})
    codes = [row["loss_stage"] for row in group_rows]
    measured = [code for code in codes if code != "not_measured"]
    loss_stage = worst_loss(measured) if measured else None
    return {
        "loss_stage": loss_stage,
        "groups": group_rows,
        "not_measured": [] if measured else ["loss_stage"],
    }


def stage_chunk_ids(
    retrieval_calls: Sequence[Mapping[str, Any]],
    stage: str,
    *,
    call_index: int | None = None,
) -> list[str]:
    """Chunk_id выбранного этапа; дубликаты схлопываются по порядку."""
    calls = (
        [retrieval_calls[call_index]]
        if call_index is not None and 0 <= call_index < len(retrieval_calls)
        else list(retrieval_calls)
    )
    ids: list[str] = []
    for call in calls:
        stages = call.get("stages") if isinstance(call, Mapping) else None
        if not isinstance(stages, Mapping):
            continue
        items = stages.get(stage)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping) and item.get("chunk_id"):
                ids.append(str(item["chunk_id"]))
    return unique_ids(ids)


def stage_available(
    retrieval_calls: Sequence[Mapping[str, Any]],
    stage: str,
    *,
    call_index: int | None = None,
) -> bool:
    """Был ли этап измерен хотя бы в одном call."""
    calls = (
        [retrieval_calls[call_index]]
        if call_index is not None and 0 <= call_index < len(retrieval_calls)
        else list(retrieval_calls)
    )
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        available = call.get("available")
        if isinstance(available, Mapping) and stage in available:
            if available.get(stage):
                return True
            continue
        stages = call.get("stages")
        if isinstance(stages, Mapping) and isinstance(stages.get(stage), list):
            return True
    return False


def aggregate_mean(values: Iterable[float | int | None]) -> float | None:
    """Macro mean по измеренным значениям; пустой список даёт ``None``."""
    measured = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not measured:
        return None
    return sum(measured) / len(measured)
