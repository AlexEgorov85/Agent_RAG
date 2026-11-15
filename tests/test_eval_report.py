"""Отчёты benchmark: HTML и сравнение запусков (offline)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.report import (
    main,
    render_comparison_html,
    render_html,
    write_report,
)


def _summary(run_id: str, *, pass_rate: float = 0.5) -> dict:
    return {
        "run_id": run_id,
        "suite": "smoke",
        "split": "dev",
        "git": {"sha": "abcdef1", "dirty": False},
        "dataset": {"version": "v1", "sha256": "a" * 64},
        "corpus": {"chunk_count": 10, "content_sha256": "b" * 64},
        "coverage": {
            "tests_selected": 2,
            "executed": 2,
            "passed": 1,
            "failed": 1,
        },
        "pass_rate": pass_rate,
        "execution_coverage": 1.0,
        "groups": {"retrieval": {"recall_at_10": 0.8}},
        "problem_tests": [{"id": "easy_001", "reasons": ["missing_citation"]}],
        "efficiency": {"mean_tool_calls": 2.0, "llm_concurrency": 1},
        "limitations": ["<script>alert(1)</script>"],
    }


def test_write_report_creates_html_without_raw_script(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    path = write_report(run_dir, _summary("run_1"))

    assert path.name == "report.md"
    html_text = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html_text
    assert "run_1" in html_text
    assert "&lt;script&gt;" in html_text
    assert "<script>" not in html_text


def test_report_main_compare_writes_md_and_html(tmp_path: Path) -> None:
    results = tmp_path / "results"
    base = results / "base"
    candidate = results / "candidate"
    for run_dir, run_id in ((base, "base"), (candidate, "candidate")):
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(_summary(run_id)), encoding="utf-8"
        )

    code = main(["--compare", "base", "candidate", "--results-dir", str(results)])

    assert code == 0
    assert (candidate / "comparison.md").exists()
    comparison_html = (candidate / "comparison.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in comparison_html
    assert "pass_rate" in comparison_html


def test_comparison_html_marks_incomparable() -> None:
    text = render_comparison_html(
        {
            "comparable": False,
            "reasons": ["dataset_sha256_mismatch"],
            "metrics": [{"key": "pass_rate", "base": 0.1, "candidate": 0.2, "delta": None}],
        }
    )

    assert "not_comparable" in text
    assert "dataset_sha256_mismatch" in text


def test_render_html_contains_metrics() -> None:
    text = render_html(_summary("run_2"))

    assert "Pass rate" in text
    assert "recall_at_10" in text


def test_render_html_and_markdown_show_warnings_and_failed_checks() -> None:
    summary = _summary("run_3")
    summary["warnings"] = [
        {"code": "citation_quote_mismatch", "count": 1, "test_ids": ["easy_001"]}
    ]
    summary["failed_checks"] = [
        {
            "name": "citation_quotes",
            "mandatory": False,
            "count": 1,
            "test_ids": ["easy_001"],
            "details": ["1/1: chunk-1"],
        }
    ]

    from eval.report import render_markdown

    markdown = render_markdown(summary)
    html_text = render_html(summary)

    assert "## Warnings" in markdown
    assert "citation_quote_mismatch" in markdown
    assert "## Failed checks" in markdown
    assert "citation_quotes" in markdown
    assert "Warnings" in html_text
    assert "citation_quote_mismatch" in html_text
    assert "Failed checks" in html_text
    assert "1/1: chunk-1" in html_text
