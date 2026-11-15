"""Evaluator retrieval: Recall@5/10, MRR, coverage и loss attribution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from eval.metrics import (
    aggregate_mean,
    context_coverage,
    evidence_coverage,
    gold_groups,
    loss_attribution,
    mrr,
    ndcg_at_k,
    recall_at_k,
    stage_available,
    stage_chunk_ids,
    unique_ids,
    worst_loss,
)

_NOT_MEASURED_RETRIEVAL = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "retrieval_coverage",
    "loss_stage",
)


def _turn_traces(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    turns = trace.get("turns")
    if isinstance(turns, list) and turns:
        return [turn for turn in turns if isinstance(turn, Mapping)]
    return [trace]


def _completed_calls(turns: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for turn in turns:
        for call in turn.get("retrieval_calls", []) or []:
            if not isinstance(call, Mapping):
                continue
            if call.get("status") not in (None, "completed"):
                continue
            calls.append(call)
    return calls


def _stage_union(
    calls: Sequence[Mapping[str, Any]], stage: str
) -> tuple[list[str] | None, bool]:
    """Объединение этапа по всем calls и флаг его доступности."""
    measured = False
    ids: list[str] = []
    for call in calls:
        stages = call.get("stages")
        if not isinstance(stages, Mapping):
            continue
        items = stages.get(stage)
        if isinstance(items, list):
            measured = True
            for item in items:
                if isinstance(item, Mapping) and item.get("chunk_id"):
                    ids.append(str(item["chunk_id"]))
    return (unique_ids(ids) if measured else None), measured


def _first_returned_call(calls: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for call in calls:
        if stage_available([call], "returned"):
            return call
    return None


def _evaluate_one(
    gold: Mapping[str, Any],
    turn: Mapping[str, Any],
    *,
    known_chunk_ids: set[str] | None,
) -> dict[str, Any]:
    groups = gold_groups(gold)
    calls = [
        call
        for call in turn.get("retrieval_calls", []) or []
        if isinstance(call, Mapping) and call.get("status") in (None, "completed")
    ]
    not_measured: list[str] = []
    if not groups:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_5": None,
            "retrieval_coverage": None,
            "evidence_coverage": None,
            "context_coverage": None,
            "loss_stage": None,
            "loss_groups": [],
            "not_measured": ["gold_evidence"],
        }
    if not calls:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_5": None,
            "retrieval_coverage": None,
            "evidence_coverage": evidence_coverage(
                groups, turn.get("evidence_manifest")
            ),
            "context_coverage": None,
            "loss_stage": None,
            "loss_groups": [],
            "not_measured": list(_NOT_MEASURED_RETRIEVAL),
        }

    first = _first_returned_call(calls)
    if first is None:
        recall_5 = recall_10 = mrr_value = ndcg_5 = None
        not_measured.extend(["recall_at_5", "recall_at_10", "mrr", "ndcg_at_5"])
    else:
        returned = stage_chunk_ids([first], "returned")
        recall_5 = recall_at_k(groups, returned, 5)
        recall_10 = recall_at_k(groups, returned, 10)
        mrr_value = mrr(groups, returned)
        ndcg_5 = ndcg_at_k(groups, returned, 5)

    all_returned: list[str] = []
    returned_measured = False
    for call in calls:
        if stage_available([call], "returned"):
            returned_measured = True
            all_returned.extend(stage_chunk_ids([call], "returned"))
    retrieval_cov = (
        evidence_coverage(groups, unique_ids(all_returned))
        if returned_measured
        else None
    )
    if not returned_measured:
        not_measured.append("retrieval_coverage")

    manifest = turn.get("evidence_manifest")
    if not isinstance(manifest, list):
        manifest = None
    evidence_cov = evidence_coverage(groups, manifest)
    if evidence_cov is None:
        not_measured.append("evidence_coverage")

    context_source = turn.get("context_source")
    context_ids = turn.get("context_chunk_ids")
    context_measured = (
        context_source != "evidence_manifest" and isinstance(context_ids, list)
    )
    context_cov = context_coverage(
        groups, context_ids, context_measured=context_measured
    )
    if context_cov is None:
        not_measured.append("context_coverage")

    stages: dict[str, list[str] | None] = {}
    available: dict[str, bool] = {}
    for stage in ("dense", "bm25", "graph", "rrf", "rerank", "returned"):
        ids, measured = _stage_union(calls, stage)
        stages[stage] = ids
        available[stage] = measured

    citations = turn.get("answer", {}).get("citations", [])
    cited_ids = [
        str(citation.get("chunk_id"))
        for citation in citations or []
        if isinstance(citation, Mapping) and citation.get("chunk_id")
    ]
    loss = loss_attribution(
        groups,
        stages=stages,
        available=available,
        context_ids=context_ids if context_measured else None,
        cited_ids=cited_ids,
        known_chunk_ids=known_chunk_ids,
        context_measured=context_measured,
    )
    if loss.get("loss_stage") is None:
        not_measured.extend(loss.get("not_measured", []))
    return {
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "mrr": mrr_value,
        "ndcg_at_5": ndcg_5,
        "retrieval_coverage": retrieval_cov,
        "evidence_coverage": evidence_cov,
        "context_coverage": context_cov,
        "loss_stage": loss.get("loss_stage"),
        "loss_groups": loss.get("groups", []),
        "not_measured": list(dict.fromkeys(not_measured)),
    }


def evaluate_retrieval(
    test: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    known_chunk_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Оценить retrieval одного теста (для multi-turn — по turn'ам).

    Основные Recall/MRR считаются по первому результативному
    ``search_reports``; coverage — по всем вызовам. Пустые gold-группы дают
    ``None`` и ``not_measured``, а не нулевую оценку.
    """
    turns = _turn_traces(trace)
    test_turns = test.get("turns") if isinstance(test.get("turns"), list) else None
    per_turn: list[dict[str, Any]] = []
    if test_turns:
        by_index = {
            int(turn.get("turn_index", index)): turn
            for index, turn in enumerate(turns)
        }
        for index, test_turn in enumerate(test_turns):
            if not isinstance(test_turn, Mapping):
                continue
            gold = test_turn.get("gold_evidence")
            if not isinstance(gold, Mapping) or not gold_groups(gold):
                continue
            trace_turn = by_index.get(index)
            if trace_turn is None:
                continue
            row = _evaluate_one(
                gold,
                trace_turn,
                known_chunk_ids=known_chunk_ids,
            )
            row["turn_index"] = index
            per_turn.append(row)
    else:
        gold = test.get("gold_evidence")
        if isinstance(gold, Mapping) and gold_groups(gold):
            row = _evaluate_one(
                gold,
                turns[0],
                known_chunk_ids=known_chunk_ids,
            )
            row["turn_index"] = 0
            per_turn.append(row)

    if not per_turn:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg_at_5": None,
            "retrieval_coverage": None,
            "evidence_coverage": None,
            "context_coverage": None,
            "loss_stage": None,
            "loss_groups": [],
            "groups": 0,
            "best_call": None,
            "all_calls": [],
            "not_measured": list(_NOT_MEASURED_RETRIEVAL) + ["gold_evidence"],
        }

    def mean_metric(key: str) -> float | None:
        return aggregate_mean([row.get(key) for row in per_turn])

    loss_codes = [row.get("loss_stage") for row in per_turn if row.get("loss_stage")]
    not_measured = sorted(
        {
            item
            for row in per_turn
            for item in row.get("not_measured", [])
        }
    )
    calls = _completed_calls(turns)
    best_call = None
    if calls:
        best_call = {
            "call_index": _first_returned_call(calls).get("call_index")
            if _first_returned_call(calls) is not None
            else 0,
            "returned": stage_chunk_ids(
                [_first_returned_call(calls)] if _first_returned_call(calls) else [],
                "returned",
            ),
        }
    result = {
        "recall_at_5": mean_metric("recall_at_5"),
        "recall_at_10": mean_metric("recall_at_10"),
        "mrr": mean_metric("mrr"),
        "ndcg_at_5": mean_metric("ndcg_at_5"),
        "retrieval_coverage": mean_metric("retrieval_coverage"),
        "evidence_coverage": mean_metric("evidence_coverage"),
        "context_coverage": mean_metric("context_coverage"),
        "loss_stage": worst_loss(loss_codes),
        "loss_groups": [
            item for row in per_turn for item in row.get("loss_groups", [])
        ],
        "groups": sum(1 for row in per_turn if row.get("loss_groups") is not None),
        "turns": per_turn,
        "best_call": best_call,
        "all_calls": [
            {
                "call_index": call.get("call_index"),
                "returned": stage_chunk_ids([call], "returned"),
                "available": call.get("available", {}),
            }
            for call in calls
        ],
        "not_measured": not_measured,
    }
    for key in (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "retrieval_coverage",
        "evidence_coverage",
        "context_coverage",
        "loss_stage",
    ):
        if result[key] is None and key not in result["not_measured"]:
            result["not_measured"].append(key)
    return result
