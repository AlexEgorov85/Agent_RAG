from __future__ import annotations

from src.agent.state import AgentRunResult, Answer, Citation
from src.retrieval.hybrid_retriever import SearchFilters
from src.ui.app import (
    analysis_payload,
    context_chunk_view,
    coverage_report_style,
    coverage_status_style,
    coverage_view,
    graph_evidence_view,
    retrieval_call_rows,
    retrieval_counters,
    retrieval_stage_rows,
    structural_change_rows,
    verification_view,
)


def _coverage_payload() -> dict:
    return {
        "status": "partial",
        "expected_years": [2020, 2021, 2022],
        "covered_years": [2020],
        "missing_years": [2021],
        "uncovered_years": [2021, 2022],
        "required_metrics": ["net_profit"],
        "cells": [
            {
                "metric": "net_profit",
                "metric_label": "Чистая прибыль",
                "year": 2020,
                "status": "covered",
                "chunk_ids": ["sber_2020:c1"],
            },
            {
                "metric": "net_profit",
                "metric_label": "Чистая прибыль",
                "year": 2021,
                "status": "year_absent",
                "chunk_ids": [],
            },
            {
                "metric": "net_profit",
                "metric_label": "Чистая прибыль",
                "year": 2022,
                "status": "not_found",
                "chunk_ids": [],
            },
        ],
        "metrics": {
            "net_profit": {
                "2020": "covered",
                "2021": "year_absent",
                "2022": "not_found",
                "*": "not_found",
            }
        },
        "missing_metrics": ["net_profit"],
        "missing_metric_years": [
            {
                "metric": "net_profit",
                "metric_label": "Чистая прибыль",
                "year": 2022,
                "status": "not_found",
            }
        ],
        "notes": ["2021: года нет в корпусе"],
        "distinct_docs": 2,
        "distinct_sections": 3,
    }


def test_coverage_keeps_year_absent_distinct_from_not_found() -> None:
    view = coverage_view(_coverage_payload())

    assert view is not None
    assert view["status"] == "partial"
    assert view["status_label"] == "Покрытие частичное"
    assert view["status_tone"] == "warning"
    assert view["years"] == [2020, 2021, 2022]

    metric = view["metrics"][0]
    assert metric["metric_label"] == "Чистая прибыль"
    cells = {cell["year"]: cell for cell in metric["cells"]}

    assert cells[2021]["status"] == "year_absent"
    assert "года нет" in cells[2021]["label"]
    # year_absent не должен выглядеть как 0 или пустая ячейка.
    assert cells[2021]["display"].strip() != ""
    assert cells[2021]["display"] != "0"

    assert cells[2022]["status"] == "not_found"
    assert "не найдено поиском" in cells[2022]["label"]
    assert "нет в отчёте" not in cells[2022]["label"]
    assert cells[2021]["display"] != cells[2022]["display"]

    assert metric["summary"] == "not_found"
    assert view["evidence_cells"] == [
        {
            "metric": "net_profit",
            "metric_label": "Чистая прибыль",
            "year": 2020,
            "chunk_ids": ["sber_2020:c1"],
        }
    ]
    assert view["notes"] == ["2021: года нет в корпусе"]
    assert view["missing_metrics"][0]["metric"] == "net_profit"
    assert view["missing_metric_years"][0]["status_label"] == "не найдено поиском"
    assert view["missing_years"] == [2021]
    assert view["distinct_docs"] == 2


def test_coverage_is_hidden_when_backend_sends_nothing() -> None:
    assert coverage_view(None) is None
    assert coverage_view({}) is None
    assert coverage_view([]) is None
    assert coverage_view("coverage") is None


def test_coverage_falls_back_to_cells_when_matrix_is_missing() -> None:
    view = coverage_view(
        {
            "status": "insufficient",
            "cells": [
                {
                    "metric": "revenue",
                    "metric_label": "Выручка",
                    "year": 2023,
                    "status": "not_found",
                    "chunk_ids": [],
                },
                {
                    "metric": "revenue",
                    "metric_label": "Выручка",
                    "year": 2022,
                    "status": "covered",
                    "chunk_ids": ["sber_2022:c9"],
                },
            ],
            "notes": ["нет собранных свидетельств"],
            "missing_metric_years": [
                {"metric": "revenue", "year": 2023, "status": "not_found"}
            ],
        }
    )

    assert view is not None
    assert view["status_tone"] == "error"
    assert view["years"] == [2022, 2023]
    assert view["metrics"][0]["metric"] == "revenue"
    assert view["metrics"][0]["metric_label"] == "Выручка"
    cells = {cell["year"]: cell for cell in view["metrics"][0]["cells"]}
    assert cells[2022]["status"] == "covered"
    assert cells[2023]["status"] == "not_found"
    assert view["metrics"][0]["summary"] is None
    assert view["notes"] == ["нет собранных свидетельств"]
    assert view["missing_metric_years"][0]["metric_label"] == "Выручка"


def test_coverage_status_styles_and_report_banners() -> None:
    assert coverage_status_style("covered")["icon"] == "✅"
    assert coverage_status_style("not_found")["label"] == "не найдено поиском"
    assert coverage_status_style("year_absent")["label"] == "года нет в корпусе"
    unknown = coverage_status_style("unknown_status")
    assert unknown["known"] is False
    assert unknown["label"] == "нет данных"

    assert coverage_report_style("complete")["tone"] == "success"
    assert coverage_report_style("partial")["tone"] == "warning"
    assert coverage_report_style("insufficient")["tone"] == "error"
    assert coverage_report_style("not_required")["label"] == (
        "Проверка покрытия не требовалась"
    )
    assert coverage_report_style("unknown_status") is None


def test_verification_rows_map_severity_and_failed_status() -> None:
    view = verification_view(
        {
            "status": "failed",
            "issues": [
                {
                    "code": "fact_value_not_in_source",
                    "severity": "fatal",
                    "message": "Значение факта не найдено в источнике.",
                    "chunk_ids": ["sber_2023:c1"],
                    "fact_index": 1,
                },
                {
                    "code": "gap_not_disclosed",
                    "severity": "warning",
                    "message": "Известные пробелы не раскрыты.",
                },
                {
                    "code": "legacy_error",
                    "severity": "error",
                    "message": "Старый alias критичной ошибки.",
                },
            ],
        }
    )

    assert view is not None
    assert view["status"] == "failed"
    assert view["tone"] == "error"
    assert view["fatal_count"] == 2
    assert view["warning_count"] == 1
    assert view["issues"][0]["severity_label"] == "критично"
    assert view["issues"][1]["severity_label"] == "предупреждение"
    # ``error`` читается как устаревший alias ``fatal``.
    assert view["issues"][2]["severity"] == "fatal"
    assert view["issues"][0]["chunk_ids"] == ["sber_2023:c1"]
    assert view["issues"][0]["fact_index"] == 1


def test_verification_is_absent_or_sanitized() -> None:
    assert verification_view(None) is None
    assert verification_view({}) is None

    view = verification_view(
        {
            "status": "warning",
            "issues": [
                {
                    "code": "safe",
                    "severity": "warning",
                    "message": "api_key=super-secret-value",
                }
            ],
        }
    )
    assert view is not None
    assert "super-secret-value" not in view["issues"][0]["message"]
    assert "[redacted]" in view["issues"][0]["message"]


def test_stage_rows_distinguish_unavailable_from_measured_zero() -> None:
    call = {
        "stages": {
            "dense": [],
            "bm25": [{"chunk_id": "a", "rank": 1, "score": 0.5}],
            "rrf": None,
        },
        "available": {"dense": True, "bm25": True, "rrf": False},
        "unavailable": {"rrf": "upstream_failed"},
        "fallback": {"bm25": True, "rrf": True},
        "counts": {"dense": 0, "bm25": 1, "rrf": 0, "returned": 1},
    }
    rows = {row["name"]: row for row in retrieval_stage_rows(call)}

    assert rows["dense"]["status"] == "empty"
    assert rows["dense"]["available"] is True
    assert rows["dense"]["count"] == 0
    assert "0" in rows["dense"]["count_label"]
    assert "не измерялся" not in rows["dense"]["count_label"]

    assert rows["bm25"]["status"] == "measured"
    assert rows["bm25"]["count"] == 1
    assert rows["bm25"]["fallback"] is True

    assert rows["rrf"]["status"] == "unavailable"
    assert rows["rrf"]["available"] is False
    # Недоступный этап не превращается в измеренный ноль.
    assert rows["rrf"]["count"] is None
    assert rows["rrf"]["count_label"].startswith("не измерялся")
    assert rows["rrf"]["reason"] == "upstream_failed"
    assert rows["rrf"]["fallback"] is True


def test_stage_rows_and_counters_keep_legacy_counts() -> None:
    rows = retrieval_stage_rows({"counts": {"dense": 3, "returned": 2}})
    by_name = {row["name"]: row for row in rows}

    assert by_name["dense"]["status"] == "count_only"
    assert by_name["dense"]["count"] == 3
    assert "3" in by_name["dense"]["count_label"]
    assert by_name["returned"]["count"] == 2

    counters = retrieval_counters(
        {
            "retrieval_mode": "hybrid",
            "query_count": 2,
            "fused_candidates": 7,
            "returned": 4,
            "distinct_docs": 3,
            "distinct_sections": 5,
            "distinct_years": 2,
            "counts": {"dense": 9, "bm25": 8, "rrf": 7, "rerank": 4},
        }
    )
    assert counters["retrieval_mode"] == "hybrid"
    assert counters["query_count"] == 2
    assert counters["fused_candidates"] == 7
    assert counters["returned"] == 4
    assert counters["distinct_docs"] == 3
    assert counters["distinct_sections"] == 5
    assert counters["distinct_years"] == 2
    assert counters["counts"]["dense"] == 9


def test_retrieval_call_rows_expose_stages_and_fallback() -> None:
    rows = retrieval_call_rows(
        [
            {
                "call_index": 0,
                "tool_name": "search_reports",
                "retrieval_mode": "hybrid",
                "top_k": 10,
                "status": "ok",
                "stages": {"dense": None, "rrf": None},
                "available": {"dense": False, "rrf": False},
                "unavailable": {"dense": "failed", "rrf": "upstream_failed"},
                "fallback": {"dense": True, "rrf": True},
                "warnings": ["retrieval_fallback"],
                "counts": {"returned": 0},
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["call_index"] == 0
    assert row["tool_name"] == "search_reports"
    assert row["retrieval_mode"] == "hybrid"
    assert row["top_k"] == 10
    assert row["fallback"] == ["dense", "rrf"]
    assert row["warnings"] == ["retrieval_fallback"]
    assert {stage["name"] for stage in row["stages"]} == {"dense", "rrf"}
    assert all(stage["status"] == "unavailable" for stage in row["stages"])


def test_graph_evidence_is_never_presented_as_confirmed_fact() -> None:
    call = {
        "stages": {"graph": None},
        "available": {"graph": False},
        "unavailable": {"graph": "no_index"},
        "counts": {"graph": 0},
    }
    payload = {
        "graph_evidence_manifest": ["sber_2023:g1"],
        "evidence_manifest": [],
        "retrieval_calls": [call],
    }
    view = graph_evidence_view(payload)

    assert view is not None
    assert view["chunk_ids"] == ["sber_2023:g1"]
    assert view["unconfirmed_count"] == 1
    assert view["entries"][0]["confirmed"] is False
    assert "не подтверждено" in view["warning"]
    assert "graph-only" in view["entries"][0]["label"]
    assert "не подтверждено до открытия chunk" in view["entries"][0]["label"]
    assert view["stage"]["name"] == "graph"
    assert view["stage"]["status"] == "unavailable"

    assert graph_evidence_view({}) is None
    assert graph_evidence_view({"graph_evidence_manifest": []}) is None


def test_graph_evidence_marks_confirmed_chunks() -> None:
    view = graph_evidence_view(
        {
            "graph_evidence_manifest": ["sber_2023:g1"],
            "evidence_manifest": ["sber_2023:g1"],
        }
    )

    assert view is not None
    assert view["unconfirmed_count"] == 0
    assert view["confirmed_count"] == 1
    assert view["entries"][0]["confirmed"] is True
    assert "не подтверждено" not in view["warning"]


def test_graph_stage_without_ids_still_renders_status() -> None:
    view = graph_evidence_view(
        {
            "retrieval_calls": [
                {
                    "stages": {"graph": []},
                    "available": {"graph": True},
                    "counts": {"graph": 0},
                }
            ]
        }
    )

    assert view is not None
    assert view["chunk_ids"] == []
    assert view["stage"]["status"] == "empty"


def test_structural_change_rows_use_only_backend_categories() -> None:
    rows = structural_change_rows(
        [
            {
                "from_year": 2022,
                "to_year": 2023,
                "from_section": None,
                "to_section": {
                    "title": "ESG",
                    "section_path": "ESG",
                    "level": 1,
                    "page_start": 10,
                    "page_end": 14,
                },
                "change_type": "added",
                "detail": "section_not_found_in_earlier_report",
                "evidence_chunk_ids": ["sber_2023:c7"],
            },
            {
                "from_year": 2022,
                "to_year": 2023,
                "from_section": {"title": "Корпоративное управление"},
                "to_section": None,
                "change_type": "removed",
                "detail": "section_not_found_in_later_report",
            },
        ]
    )

    assert rows[0]["section_title"] == "ESG"
    assert rows[0]["change_label"] == "Добавлен"
    assert rows[0]["from_year"] == 2022
    assert rows[0]["to_year"] == 2023
    assert rows[0]["page_start"] == 10
    assert rows[0]["evidence_chunk_ids"] == ["sber_2023:c7"]
    assert rows[1]["change_label"] == "Не обнаружен в более позднем отчёте"
    # UI не выдумывает split/merge: таких подписей нет.
    assert all(
        "split" not in row["change_label"].casefold()
        and "merge" not in row["change_label"].casefold()
        for row in rows
    )

    assert structural_change_rows([]) == []
    assert structural_change_rows(None) == []

    split = structural_change_rows(
        [
            {
                "change_type": "restructured",
                "detail": "section_split",
                "from_section": {"title": "Риски"},
            }
        ]
    )
    assert split[0]["change_label"] == "Перемещён / изменён уровень"
    assert "section_split" not in split[0]["detail"]
    assert split[0]["detail"] == "раздел перестроен"


def test_analysis_payload_hides_missing_data_and_is_idempotent() -> None:
    result = AgentRunResult(
        request_id="request-1",
        conversation_id="conversation-1",
        answer=Answer(
            status="answer",
            answer="Ответ",
            citations=[Citation(chunk_id="sber_2023:c1", year=2023)],
            verification={"status": "ok", "issues": []},
        ),
        tool_events=[],
        effective_filters=SearchFilters(),
        warnings=[],
        timings={"total_ms": 12.0},
        diagnostics={},
    )

    payload = analysis_payload(result)
    assert payload["coverage"] is None
    assert payload["retrieval_calls"] == []
    assert payload["structural_changes"] == []
    assert payload["verification"] == {"status": "ok", "issues": []}
    assert payload["context_chunk_ids"] is None
    assert analysis_payload(payload) == payload


def test_analysis_payload_uses_diagnostics_and_legacy_fallback() -> None:
    result = AgentRunResult(
        request_id="request-2",
        conversation_id="conversation-1",
        answer=Answer(status="answer", answer="Ответ"),
        tool_events=[],
        effective_filters=SearchFilters(),
        warnings=[],
        timings={},
        diagnostics={
            "coverage": _coverage_payload(),
            "verification": {"status": "warning", "issues": []},
            "retrieval_calls": [
                {
                    "call_index": 0,
                    "tool_name": "search_reports",
                    "stages": {"dense": []},
                    "available": {"dense": True},
                    "counts": {"dense": 0},
                }
            ],
            "context_chunk_ids": ["sber_2023:c1"],
            "context_source": "context",
            "graph_evidence_manifest": ["sber_2023:g1"],
            "evidence_manifest": ["sber_2023:c1"],
            "structural_changes": [{"change_type": "added"}],
            "query_plan": {"intent": "temporal", "years": [2022, 2023]},
            "search_attempts": [{"kind": "year", "year": 2023, "result_count": 4}],
        },
    )

    payload = analysis_payload(result)
    assert payload["coverage"]["status"] == "partial"
    assert payload["verification"]["status"] == "warning"
    assert len(payload["retrieval_calls"]) == 1
    assert payload["legacy_retrieval"] == []
    assert payload["context_chunk_ids"] == ["sber_2023:c1"]
    assert payload["context_source"] == "context"
    assert payload["graph_evidence_manifest"] == ["sber_2023:g1"]
    assert payload["query_plan"]["intent"] == "temporal"
    assert payload["search_attempts"][0]["year"] == 2023

    legacy = analysis_payload(
        {
            "retrieval_diagnostics": [
                {
                    "retrieval_mode": "hybrid",
                    "query_count": 1,
                    "fused_candidates": 4,
                    "returned": 2,
                }
            ]
        }
    )
    assert legacy["retrieval_calls"] == [
        {"retrieval_mode": "hybrid", "query_count": 1, "fused_candidates": 4, "returned": 2}
    ]
    assert len(legacy["legacy_retrieval"]) == 1


def test_context_chunk_view_distinguishes_missing_and_empty() -> None:
    missing = context_chunk_view({})
    assert missing["available"] is False
    assert missing["count"] is None
    assert missing["label"] == "не зафиксирован"

    empty = context_chunk_view(
        {"context_chunk_ids": [], "context_source": "context"}
    )
    assert empty["available"] is True
    assert empty["count"] == 0
    assert empty["source"] == "context"

    filled = context_chunk_view(
        {"context_chunk_ids": ["a", "b"], "context_source": "evidence_manifest"}
    )
    assert filled["chunk_ids"] == ["a", "b"]
    assert filled["count"] == 2
