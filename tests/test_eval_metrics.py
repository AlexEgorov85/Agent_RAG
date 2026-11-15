"""Офлайн-тесты retrieval-метрик, nDCG, coverage и loss attribution."""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    aggregate_mean,
    context_coverage,
    coverage_ratio,
    evidence_coverage,
    gold_groups,
    loss_attribution,
    mrr,
    ndcg_at_k,
    recall_at_k,
    stage_available,
    stage_chunk_ids,
    worst_loss,
)

GROUPS = [
    {"id": "profit", "any_of": ["a", "b"]},
    {"id": "roe", "any_of": ["c"]},
]


def dcg(gains: list[int]) -> float:
    return sum(
        gain / math.log2(position + 1)
        for position, gain in enumerate(gains, start=1)
    )


def test_gold_groups_filters_empty_alternatives():
    groups = gold_groups(
        {
            "groups": [
                {"id": "ok", "any_of": ["a", None, "b"]},
                {"id": "bad", "any_of": []},
                "not-a-group",
            ]
        }
    )
    assert groups == [{"id": "ok", "any_of": ["a", "b"], "section": None}]
    assert len(groups) == 1


def test_recall_at_k_uses_groups_and_alternatives():
    assert recall_at_k(GROUPS, ["b", "c"], 5) == 1.0
    assert recall_at_k(GROUPS, ["b", "x"], 5) == 0.5
    assert recall_at_k(GROUPS, ["x", "y", "z", "u", "v"], 5) == 0.0
    # Один и тот же gold chunk не удваивает покрытие.
    assert recall_at_k(GROUPS, ["a", "b", "c"], 5) == 1.0


def test_recall_empty_gold_is_none_not_zero():
    assert recall_at_k([], ["a"], 5) is None
    assert mrr([], ["a"]) is None
    assert ndcg_at_k([], ["a"], 5) is None
    assert evidence_coverage([], ["a"]) is None
    assert context_coverage([], ["a"]) is None


def test_recall_at_5_differs_from_recall_at_10():
    ranked = ["x", "y", "z", "u", "v", "a"]
    assert recall_at_k(GROUPS, ranked, 5) == 0.0
    assert recall_at_k(GROUPS, ranked, 10) == 0.5
    assert mrr(GROUPS, ranked) == pytest.approx(1 / 6)


def test_mrr_first_relevant_rank():
    assert mrr(GROUPS, ["b", "a", "c"]) == 1.0
    assert mrr(GROUPS, ["x", "c"]) == 0.5
    assert mrr(GROUPS, ["x", "y"]) == 0.0


def test_ndcg_at_5_known_values():
    # Оба gold-группы на первых двух позициях: ideal ranking.
    assert ndcg_at_k(GROUPS, ["b", "c"], 5) == pytest.approx(1.0)
    # Первая группа на позиции 2.
    expected = dcg([0, 1, 0]) / dcg([1, 1, 0])
    assert ndcg_at_k(GROUPS, ["x", "b"], 5) == pytest.approx(expected)
    # Позиция 6 вне окна nDCG@5.
    assert ndcg_at_k(GROUPS, ["x", "y", "z", "u", "v", "b"], 5) == 0.0


def test_ndcg_alternative_chunks_do_not_double_count_group():
    groups = [{"id": "g1", "any_of": ["a", "b"]}, {"id": "g2", "any_of": ["c"]}]
    expected = dcg([1, 0]) / dcg([1, 1])
    assert ndcg_at_k(groups, ["a", "b"], 5) == pytest.approx(expected)
    assert ndcg_at_k(groups, ["a", "b", "c"], 5) > expected


def test_coverage_ratio_and_evidence_context_denominator():
    assert coverage_ratio(0, 0) is None
    assert coverage_ratio(1, 2) == 0.5
    assert evidence_coverage(GROUPS, ["a", "x"]) == 0.5
    assert context_coverage(GROUPS, ["c"], context_measured=False) is None
    assert context_coverage(GROUPS, ["c"], context_measured=True) == 0.5


def test_aggregate_mean_keeps_none_out_and_empty_gives_none():
    assert aggregate_mean([None, None]) is None
    assert aggregate_mean([1.0, None, 0.0]) == 0.5
    assert aggregate_mean([]) is None


def test_loss_attribution_all_stages():
    group = {"id": "g", "any_of": ["a"]}
    known = {"a"}

    def diagnose(**kwargs):
        base = {
            "stages": {},
            "available": {},
            "context_ids": ["a"],
            "cited_ids": ["a"],
            "known_chunk_ids": known,
            "context_measured": True,
        }
        base.update(kwargs)
        return loss_attribution([group], **base)["loss_stage"]

    assert diagnose(known_chunk_ids={"other"}) == "not_in_corpus"
    assert diagnose(
        stages={"dense": [], "bm25": []},
        available={"dense": True, "bm25": True, "rrf": True},
        context_ids=None,
        cited_ids=None,
    ) == "never_retrieved"
    assert diagnose(
        stages={"dense": ["a"], "bm25": [], "rrf": []},
        available={"dense": True, "bm25": True, "rrf": True},
        context_ids=None,
        cited_ids=None,
    ) == "lost_at_fusion"
    assert diagnose(
        stages={"dense": ["a"], "bm25": [], "rrf": ["a"], "rerank": []},
        available={"dense": True, "bm25": True, "rrf": True, "rerank": True},
        context_ids=None,
        cited_ids=None,
    ) == "lost_at_rerank"
    assert diagnose(
        stages={"dense": ["a"], "bm25": [], "rrf": ["a"], "rerank": ["a"], "returned": ["a"]},
        available={"dense": True, "bm25": True, "rrf": True, "rerank": True, "returned": True},
        context_ids=[],
        cited_ids=[],
    ) == "lost_at_context"
    assert diagnose(
        stages={"dense": [], "bm25": [], "rrf": ["a"], "rerank": ["a"], "returned": ["a"]},
        available={"dense": True, "bm25": True, "rrf": True, "rerank": True, "returned": True},
        context_ids=["a"],
        cited_ids=[],
    ) == "not_cited"
    assert diagnose(
        stages={"dense": [], "bm25": [], "rrf": ["a"], "rerank": ["a"], "returned": ["a"]},
        available={"dense": True, "bm25": True, "rrf": True, "rerank": True, "returned": True},
        context_ids=["a"],
        cited_ids=["a"],
    ) == "ok"


def test_loss_attribution_not_measured_without_stages_or_gold():
    result = loss_attribution([{"id": "g", "any_of": ["a"]}], stages={})
    assert result["loss_stage"] in {None, "ok"}
    empty = loss_attribution([], stages={})
    assert empty["loss_stage"] is None
    assert "loss_stage" in empty["not_measured"]


def test_worst_loss_precedence():
    assert worst_loss(["ok", "lost_at_rerank", "never_retrieved"]) == "never_retrieved"
    assert worst_loss(["ok", "not_cited"]) == "not_cited"
    assert worst_loss([]) is None


def test_stage_helpers_dedupe_and_detect_unavailable():
    calls = [
        {
            "stages": {"returned": [{"chunk_id": "a", "rank": 1}, {"chunk_id": "a", "rank": 2}]},
            "available": {"returned": True},
        },
        {
            "stages": {"returned": None, "dense": []},
            "available": {"returned": False, "dense": True},
        },
    ]
    assert stage_chunk_ids(calls, "returned") == ["a"]
    assert stage_available(calls, "returned") is True
    assert stage_available(calls, "dense") is True
    assert stage_available(calls, "rerank") is False


def test_mention_present_accepts_russian_word_forms():
    from eval.evaluators.answer import mention_present

    text = "Размер ипотечного портфеля вырос, климатические риски учтены, ИИ внедрён."
    assert mention_present(text, "ипотека") is True
    assert mention_present(text, "климат") is True
    assert mention_present(text, "ипотечный портфель") is True
    assert mention_present(text, "портфель ценных бумаг") is False
    # Аббревиатура не должна находиться внутри другого слова.
    assert mention_present("иной подход", "ИИ") is False


def test_fact_match_compares_equivalent_units():
    from eval.evaluators.answer import _fact_match

    required = {
        "metric": "net_profit",
        "metric_aliases": ["чистая прибыль"],
        "year": 2023,
        "value": 1508.6,
        "unit": "млрд руб.",
        "tolerance": 0.01,
    }
    citations = [{"chunk_id": "c1"}]

    matched = _fact_match(
        required,
        {
            "text": "прибыль 1,5 трлн руб.",
            "facts": [
                {
                    "metric": "Чистая прибыль",
                    "year": 2023,
                    "value": "1,5 трлн руб.",
                    "unit": "руб.",
                    "chunk_id": "c1",
                }
            ],
        },
        citations,
    )
    assert matched["matched"] is True

    wrong = _fact_match(
        required,
        {
            "text": "прибыль 1,9 трлн руб.",
            "facts": [
                {
                    "metric": "Чистая прибыль",
                    "year": 2023,
                    "value": "1,9 трлн руб.",
                    "unit": "руб.",
                    "chunk_id": "c1",
                }
            ],
        },
        citations,
    )
    assert wrong["matched"] is False


def _answer_case(
    *,
    checks: dict | None = None,
    required_structural: list[dict] | None = None,
    quote: str = "Чистая прибыль за 2023 год составила 1 508,6 млрд рублей.",
    run_warnings: list[str] | None = None,
    structural_changes: list[dict] | None = None,
) -> tuple[dict, dict]:
    test = {
        "expected_status": "answer",
        "expected": {
            "required_facts": [],
            "required_years": [2023],
            "required_documents": [],
            "required_structural_changes": required_structural or [],
        },
        "checks": {
            "must_mention": [],
            "must_not_invent_values": False,
            **(checks or {}),
        },
        "gold_evidence": {"min_coverage": 0.0, "groups": []},
    }
    trace = {
        "answer": {
            "status": "answer",
            "text": "Ответ за 2023 год.",
            "facts": [],
            "calculations": [],
            "citations": [
                {"chunk_id": "c1", "year": 2023, "quote": quote}
            ],
        },
        "evidence_manifest": ["c1"],
        "context_chunk_ids": ["c1"],
        "context_source": "context",
        "tool_events": [],
        "query_plan": {"years": [2023]},
        "effective_filters": {"years": [2023]},
        "run_warnings": run_warnings or [],
        "structural_changes": structural_changes if structural_changes is not None else [],
    }
    return test, trace


CHUNK_STORE = {
    "c1": {
        "chunk_id": "c1",
        "doc_id": "sber_2023",
        "year": 2023,
        "text": "Чистая прибыль за 2023 год составила 1 508,6 млрд рублей.",
    }
}


def test_strict_quotes_turns_mismatch_into_critical_error():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        checks={"strict_quotes": True}, quote="Такого текста нет в чанке"
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is False
    assert any(
        error["code"] == "citation_quote_mismatch"
        for error in result["critical_errors"]
    )
    quote_check = next(
        row for row in result["checks"] if row["name"] == "citation_quotes"
    )
    assert quote_check["mandatory"] is True
    assert quote_check["pass"] is False


def test_non_strict_quotes_mismatch_is_only_a_warning():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        checks={"strict_quotes": False}, quote="Такого текста нет в чанке"
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is True, result["critical_errors"]
    assert "citation_quote_mismatch" in result["warnings"]
    assert not result["critical_errors"]
    quote_check = next(
        row for row in result["checks"] if row["name"] == "citation_quotes"
    )
    assert quote_check["mandatory"] is False
    assert quote_check["pass"] is False


def test_matching_quote_passes_strict_check():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(checks={"strict_quotes": True})

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is True, result["critical_errors"]
    quote_check = next(
        row for row in result["checks"] if row["name"] == "citation_quotes"
    )
    assert quote_check["pass"] is True


def test_required_structural_change_absent_is_critical_when_capability_available():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        required_structural=[
            {
                "type": "restructured",
                "section": "Управление рисками",
                "from_doc": "sber_2022",
                "to_doc": "sber_2023",
            }
        ],
        structural_changes=[],
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is False
    assert "structural_changes" not in result["not_measured"]
    assert any(
        error["code"] == "structural_changes_missing"
        for error in result["critical_errors"]
    )
    check = next(row for row in result["checks"] if row["name"] == "structural_changes")
    assert check["pass"] is False


def test_required_structural_change_is_not_measured_when_structure_unavailable():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        required_structural=[
            {"type": "restructured", "section": "Управление рисками"}
        ],
        structural_changes=[],
        run_warnings=["structure_unavailable"],
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is True, result["critical_errors"]
    assert "structural_changes" in result["not_measured"]
    assert not any(
        error["code"] == "structural_changes_missing"
        for error in result["critical_errors"]
    )


def test_required_structural_change_matches_service_schema():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        required_structural=[
            {
                "type": "restructured",
                "section": "Управление рисками",
                "from_doc": "sber_2022",
                "to_doc": "sber_2023",
            }
        ],
        structural_changes=[
            {
                "change_type": "restructured",
                "from_year": 2022,
                "to_year": 2023,
                "from_section": {
                    "title": "Управление рисками",
                    "section_path": "Сбер в цифрах/Управление рисками",
                },
                "to_section": {
                    "title": "Управление рисками в 2023 году",
                    "section_path": "Сбер в цифрах/Управление рисками в 2023 году",
                },
            }
        ],
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is True, result["critical_errors"]
    check = next(row for row in result["checks"] if row["name"] == "structural_changes")
    assert check["pass"] is True


def test_required_structural_change_wrong_type_is_critical():
    from eval.evaluators.answer import evaluate_answer

    test, trace = _answer_case(
        required_structural=[
            {"type": "restructured", "section": "Управление рисками"}
        ],
        structural_changes=[
            {
                "change_type": "added",
                "to_section": {"title": "Управление рисками"},
            }
        ],
    )

    result = evaluate_answer(test, trace, chunk_store=CHUNK_STORE)

    assert result["pass"] is False
    assert any(
        error["code"] == "structural_changes_missing"
        for error in result["critical_errors"]
    )

