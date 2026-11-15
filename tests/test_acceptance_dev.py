"""Тесты acceptance-скрипта на fake service (без сети/LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import acceptance_dev as acceptance
from src.agent.state import AgentRunResult, Answer, Citation, ToolEvent
from src.retrieval.hybrid_retriever import SearchFilters


def _run(
    *,
    status: str = "completed",
    answer_status: str = "answer",
    citations: list[Citation] | None = None,
    tools: list[ToolEvent] | None = None,
    coverage_status: str = "complete",
    structural_changes: list[dict[str, Any]] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        request_id="req-fake",
        conversation_id="fake",
        answer=Answer(
            answer_status,
            "Тестовый ответ." if answer_status == "answer" else "Нет данных.",
            citations or [],
        ),
        tool_events=tools or [],
        effective_filters=SearchFilters(years={2023}),
        warnings=[],
        timings={"total_ms": 5.0},
        status=status,
        structural_changes=structural_changes or [],
        diagnostics={
            "query_plan": {"intent": "factoid"},
            "coverage": {
                "status": coverage_status,
                "missing_years": [],
                "covered_years": [2023],
            },
            "verification": {"status": "ok", "issues": []},
        },
    )


class FakeService:
    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results

    def ask(
        self, query: str, *, conversation_id: str, filters: Any = None
    ) -> Any:
        del conversation_id, filters
        result = self.results.get(query)
        if isinstance(result, BaseException):
            raise result
        return result


def _factoid_scenario() -> acceptance.Scenario:
    return acceptance.Scenario(
        name="factoid_2023",
        intent="factoid",
        question="Какая чистая прибыль за 2023?",
        expectation="answer + citation 2023",
    )


def _abstain_scenario() -> acceptance.Scenario:
    return acceptance.Scenario(
        name="abstain_2021",
        intent="abstain",
        question="Какая чистая прибыль за 2021?",
        expectation="abstain без citations",
    )


def test_run_scenarios_marks_pass_and_failure_honestly() -> None:
    service = FakeService(
        {
            "Какая чистая прибыль за 2023?": _run(
                citations=[Citation("c2023", year=2023)]
            ),
            "Какая чистая прибыль за 2021?": _run(
                citations=[Citation("c2022", year=2022)]
            ),
        }
    )

    records = acceptance.run_scenarios(
        service, [_factoid_scenario(), _abstain_scenario()]
    )

    assert [item["outcome"] for item in records] == ["passed", "failed"]
    assert records[0]["citations"][0]["year"] == 2023
    assert "expected_abstain" in records[1]["failures"]
    assert "abstain_must_not_cite" in records[1]["failures"]


def test_run_scenarios_records_exception_as_error_without_secrets() -> None:
    service = FakeService(
        {
            "Какая чистая прибыль за 2023?": RuntimeError(
                "provider failed: api_key=sk-supersecret123456"
            )
        }
    )

    records = acceptance.run_scenarios(service, [_factoid_scenario()])

    assert records[0]["outcome"] == "error"
    assert records[0]["error_code"] == "RuntimeError"
    assert "supersecret" not in records[0]["error_detail"]
    assert "[REDACTED]" in records[0]["error_detail"]


def test_structural_scenario_requires_two_structure_tools() -> None:
    scenario = acceptance.Scenario(
        name="structural_2016_2023",
        intent="structural",
        question="Как изменилась структура?",
        expectation="два дерева + изменения",
    )
    service = FakeService(
        {
            scenario.question: _run(
                tools=[
                    ToolEvent(
                        "get_structure_tree", {}, "completed", result_count=4
                    ),
                    ToolEvent(
                        "get_structure_tree", {}, "completed", result_count=4
                    ),
                ],
                structural_changes=[{"change_type": "renamed"}],
                citations=[Citation("c2023", year=2023)],
            )
        }
    )

    records = acceptance.run_scenarios(service, [scenario])

    assert records[0]["outcome"] == "passed"
    assert len(records[0]["structural_changes"]) == 1


def test_structural_changes_without_answer_are_failed() -> None:
    scenario = acceptance.Scenario(
        name="structural_2016_2023",
        intent="structural",
        question="Как изменилась структура?",
        expectation="ответ с изменениями",
    )
    service = FakeService(
        {
            scenario.question: _run(
                answer_status="abstain",
                tools=[
                    ToolEvent(
                        "get_structure_tree", {}, "completed", result_count=4
                    ),
                    ToolEvent(
                        "get_structure_tree", {}, "completed", result_count=4
                    ),
                ],
                structural_changes=[{"change_type": "renamed"}],
            )
        }
    )

    records = acceptance.run_scenarios(service, [scenario])

    assert records[0]["outcome"] == "failed"
    assert "expected_answer" in records[0]["failures"]


def test_report_roundtrip_writes_json_and_markdown(tmp_path: Path) -> None:
    service = FakeService(
        {
            "Какая чистая прибыль за 2023?": _run(
                citations=[Citation("c2023", year=2023)]
            )
        }
    )
    records = acceptance.run_scenarios(service, [_factoid_scenario()])
    report = acceptance.build_report(records, chunks_dir=tmp_path / "chunks")

    json_path, markdown_path = acceptance.write_report(report, tmp_path)

    assert json_path.name == "acceptance_report.json"
    assert markdown_path.name == "acceptance_report.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 1
    assert payload["scenarios"][0]["coverage"]["status"] == "complete"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "factoid_2023" in markdown
    assert "Coverage" in markdown
    assert "Verification" in markdown


def test_main_writes_skipped_report_without_llm_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SimpleNamespace(
        llm_api_key="",
        cache_dir=tmp_path / "cache",
        model="fake",
        base_url="http://127.0.0.1:9/v1",
        rerank_model=None,
        max_tool_calls=4,
        max_search_calls=2,
        per_year_k=2,
        max_years=12,
        max_structure_docs=5,
        llm_router_enabled=True,
        llm_rewrite_enabled=True,
        llm_structure_verify=True,
        rerank_enabled=True,
        top_k=5,
    )
    import src.core.rag_config as rag_config

    monkeypatch.setattr(rag_config, "load_rag_settings", lambda: settings)

    exit_code = acceptance.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--chunks-dir",
            str(tmp_path / "missing-chunks"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (tmp_path / "out" / "acceptance_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["summary"]["skipped"] == 4
    assert payload["summary"]["passed"] == 0
    assert payload["skipped_reason"] == "llm_api_key_missing"
    assert {
        item["outcome"] for item in payload["scenarios"]
    } == {"skipped"}
    assert all(
        item["skip_reason"] == "llm_api_key_missing"
        for item in payload["scenarios"]
    )
    assert (tmp_path / "out" / "acceptance_report.md").is_file()


def test_main_rejects_unknown_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.core.rag_config as rag_config

    monkeypatch.setattr(
        rag_config, "load_rag_settings", lambda: SimpleNamespace()
    )

    exit_code = acceptance.main(
        ["--output-dir", str(tmp_path), "--scenario", "unknown"]
    )

    assert exit_code == 2
