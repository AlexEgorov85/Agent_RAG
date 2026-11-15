"""Markdown/JSON/HTML отчёт benchmark и вспомогательная атомарная запись.

Отчёт читает уже посчитанный ``summary.json`` и не пересчитывает метрики.
Неизмеренная метрика печатается как «—», а не как ноль.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DASH = "—"


def atomic_write_text(path: Path, text: str) -> None:
    """Записать файл атомарно: временный файл рядом + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _format_value(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return DASH
    if isinstance(value, bool):
        return "да" if value else "нет"
    if percent and isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _metric_row(name: str, value: Any, *, percent: bool = False) -> str:
    return f"| {name} | {_format_value(value, percent=percent)} |"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Собрать краткий Markdown-отчёт для PR/CI и локального просмотра."""
    dataset = summary.get("dataset") or {}
    corpus = summary.get("corpus") or {}
    coverage = summary.get("coverage") or {}
    groups = summary.get("groups") or {}
    retrieval = groups.get("retrieval") or {}
    answer = groups.get("answer") or {}
    reasoning = groups.get("reasoning") or {}
    dialogue = groups.get("dialogue") or {}
    lines: list[str] = []
    lines.append(f"# Benchmark {summary.get('run_id', DASH)}")
    lines.append("")
    lines.append(
        "Run: {run} | Dataset: {dataset} ({sha}) | Git: {git}{dirty}".format(
            run=summary.get("run_id", DASH),
            dataset=dataset.get("version", DASH),
            sha=_short(dataset.get("sha256")),
            git=summary.get("git", {}).get("sha", DASH),
            dirty=" (dirty)" if summary.get("git", {}).get("dirty") else "",
        )
    )
    lines.append(
        "Suite: {suite} | Split: {split} | Mode: {mode} | Judge: {judge}".format(
            suite=summary.get("suite", DASH),
            split=summary.get("split", DASH),
            mode=summary.get("config", {}).get("retrieval_mode", DASH),
            judge="on" if summary.get("judge") else "off",
        )
    )
    lines.append(
        "Corpus: {count} chunks | content {sha} | index {fingerprint} | graph {graph}".format(
            count=_format_value(corpus.get("chunk_count")),
            sha=_short(corpus.get("content_sha256")),
            fingerprint=_short(corpus.get("index_fingerprint")),
            graph=_short(corpus.get("graph_fingerprint")),
        )
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    for key, label in (
        ("tests_selected", "Выбрано"),
        ("tests_eligible", "Допущено (eligible)"),
        ("executed", "Выполнено"),
        ("passed", "Прошло"),
        ("failed", "Провалено"),
        ("skipped", "Пропущено"),
        ("unsupported", "Неподдержано"),
        ("timeouts", "Таймауты"),
        ("errors", "Ошибки runner"),
        ("judge_errors", "Ошибки судьи"),
    ):
        lines.append(_metric_row(label, coverage.get(key)))
    lines.append(_metric_row("Pass rate", summary.get("pass_rate"), percent=True))
    lines.append(
        _metric_row("Execution coverage", summary.get("execution_coverage"), percent=True)
    )
    lines.append(_metric_row("Pass@1", summary.get("pass_at_1"), percent=True))
    lines.append(_metric_row("Повторов на пару", summary.get("repeat")))
    # При repeat=1 stability = null: отчёт не подменяет её pass@1.
    lines.append(_metric_row("Stability", summary.get("stability"), percent=True))
    lines.append("")
    lines.append("## Метрики по группам")
    lines.append("")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---:|")
    for label, key in (
        ("Retrieval Recall@5", "recall_at_5"),
        ("Retrieval Recall@10", "recall_at_10"),
        ("Retrieval MRR", "mrr"),
        ("Retrieval nDCG@5", "ndcg_at_5"),
        ("Retrieval coverage", "retrieval_coverage"),
        ("Evidence coverage", "evidence_coverage"),
        ("Context coverage", "context_coverage"),
    ):
        lines.append(_metric_row(label, retrieval.get(key)))
    for label, key in (
        ("Answer automatic pass", "automatic_pass_rate"),
        ("Fact accuracy", "fact_accuracy"),
        ("Citation validity", "citation_validity"),
        ("Citation coverage", "citation_coverage"),
        ("Calculation accuracy", "calculation_accuracy"),
        ("Abstention accuracy", "abstention_accuracy"),
        ("Status accuracy", "status_accuracy"),
        ("Rubric mean", "rubric_mean"),
    ):
        lines.append(_metric_row(label, answer.get(key)))
    lines.append(_metric_row("Reasoning pass", reasoning.get("pass_rate")))
    lines.append(_metric_row("Dialogue success", dialogue.get("scenario_success_rate")))
    lines.append("")
    lines.append("## По сложности")
    lines.append("")
    lines.append("| Bucket | Pass rate | Eligible | Executed |")
    lines.append("|---|---:|---:|---:|")
    for name, row in (summary.get("by_difficulty") or {}).items():
        row = row if isinstance(row, Mapping) else {}
        lines.append(
            f"| {name} | {_format_value(row.get('pass_rate'), percent=True)} "
            f"| {_format_value(row.get('eligible'))} "
            f"| {_format_value(row.get('executed'))} |"
        )
    lines.append("")
    if summary.get("by_mode"):
        lines.append("## По режимам")
        lines.append("")
        lines.append("| Режим | Pass rate | Eligible | Executed |")
        lines.append("|---|---:|---:|---:|")
        for name, row in (summary.get("by_mode") or {}).items():
            row = row if isinstance(row, Mapping) else {}
            lines.append(
                f"| {name} | {_format_value(row.get('pass_rate'), percent=True)} "
                f"| {_format_value(row.get('eligible'))} "
                f"| {_format_value(row.get('executed'))} |"
            )
        lines.append("")
    critical = summary.get("critical_errors") or []
    lines.append("## Critical errors")
    lines.append("")
    if critical:
        lines.append("| Код | Кол-во | Тесты |")
        lines.append("|---|---:|---|")
        for item in critical:
            lines.append(
                f"| {item.get('code')} | {item.get('count')} | "
                f"{', '.join(item.get('test_ids', []) or [])} |"
            )
    else:
        lines.append("Нет critical errors.")
    lines.append("")
    warnings = summary.get("warnings") or []
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        lines.append("| Код | Кол-во | Тесты |")
        lines.append("|---|---:|---|")
        for item in warnings:
            lines.append(
                f"| {item.get('code')} | {item.get('count')} | "
                f"{', '.join(str(value) for value in item.get('test_ids', []) or [])} |"
            )
    else:
        lines.append("Предупреждений нет.")
    lines.append("")
    failed_checks = summary.get("failed_checks") or []
    lines.append("## Failed checks")
    lines.append("")
    if failed_checks:
        lines.append("| Check | Mandatory | Кол-во | Тесты | Детали |")
        lines.append("|---|---|---:|---|---|")
        for item in failed_checks:
            details = "; ".join(
                str(value) for value in item.get("details", []) or [] if value
            )
            lines.append(
                f"| {item.get('name')} | "
                f"{_format_value(item.get('mandatory'))} | "
                f"{item.get('count')} | "
                f"{', '.join(str(value) for value in item.get('test_ids', []) or [])} | "
                f"{details or DASH} |"
            )
    else:
        lines.append("Проваленных checks нет.")
    lines.append("")
    lines.append("## Problem tests")
    lines.append("")
    problems = summary.get("problem_tests") or []
    if problems:
        lines.append("| Тест | Причины |")
        lines.append("|---|---|")
        for item in problems:
            lines.append(
                f"| {item.get('id')} | {', '.join(item.get('reasons', []) or [])} |"
            )
    else:
        lines.append("Проблемных тестов нет.")
    lines.append("")
    lines.append("## Эффективность")
    lines.append("")
    efficiency = summary.get("efficiency") or {}
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    lines.append(_metric_row("Средние tool calls", efficiency.get("mean_tool_calls")))
    lines.append(_metric_row("Средние search calls", efficiency.get("mean_search_calls")))
    lines.append(_metric_row("Средняя latency, мс", efficiency.get("mean_latency_ms")))
    lines.append(_metric_row("LLM calls", efficiency.get("llm_calls")))
    lines.append(_metric_row("LLM concurrency", efficiency.get("llm_concurrency")))
    lines.append("")
    lines.append("## Ограничения")
    lines.append("")
    limitations = summary.get("limitations") or []
    if limitations:
        for item in limitations:
            lines.append(f"- {item}")
    else:
        lines.append("Ограничения не зафиксированы.")
    lines.append("")
    return "\n".join(lines)


def _html_page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:72rem}"
        "table{border-collapse:collapse;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
        "th{background:#f2f2f2}code{background:#f6f6f6;padding:.1rem .25rem}"
        "</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


def _html_table(rows: Sequence[tuple[str, Any]], *, percent: bool = False) -> str:
    cells = "".join(
        f"<tr><th>{html.escape(str(label))}</th>"
        f"<td>{html.escape(_format_value(value, percent=percent))}</td></tr>"
        for label, value in rows
    )
    return f"<table>{cells}</table>"


def render_html(summary: Mapping[str, Any]) -> str:
    """Самодостаточный HTML-отчёт без внешних ресурсов и секретов."""
    dataset = summary.get("dataset") or {}
    corpus = summary.get("corpus") or {}
    groups = summary.get("groups") or {}
    coverage = summary.get("coverage") or {}
    efficiency = summary.get("efficiency") or {}
    run_id = html.escape(str(summary.get("run_id", DASH)))
    parts = [
        f"<h1>Benchmark {run_id}</h1>",
        "<p>"
        f"Dataset: {html.escape(str(dataset.get('version', DASH)))} "
        f"({html.escape(_short(dataset.get('sha256')))}) | "
        f"Git: {html.escape(str((summary.get('git') or {}).get('sha', DASH)))} | "
        f"Suite: {html.escape(str(summary.get('suite', DASH)))} | "
        f"Split: {html.escape(str(summary.get('split', DASH)))}"
        "</p>",
        "<p>"
        f"Corpus: {html.escape(_format_value(corpus.get('chunk_count')))} chunks | "
        f"content {html.escape(_short(corpus.get('content_sha256')))}"
        "</p>",
        "<h2>Coverage</h2>",
        _html_table(
            [
                ("Выбрано", coverage.get("tests_selected")),
                ("Выполнено", coverage.get("executed")),
                ("Прошло", coverage.get("passed")),
                ("Провалено", coverage.get("failed")),
                ("Пропущено", coverage.get("skipped")),
                ("Таймауты", coverage.get("timeouts")),
                ("Ошибки runner", coverage.get("errors")),
                ("Pass rate", summary.get("pass_rate")),
                ("Execution coverage", summary.get("execution_coverage")),
                ("Повторов на пару", summary.get("repeat")),
                ("Pass@1", summary.get("pass_at_1")),
                ("Stability", summary.get("stability")),
            ]
        ),
        "<h2>Метрики по группам</h2>",
    ]
    for group_name, metrics in groups.items():
        if not isinstance(metrics, Mapping):
            continue
        parts.append(f"<h3>{html.escape(str(group_name))}</h3>")
        parts.append(
            _html_table(
                [
                    (str(key), value)
                    for key, value in metrics.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                ]
            )
        )
    problems = summary.get("problem_tests") or []
    parts.append("<h2>Проблемные тесты</h2>")
    if problems:
        cells = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('id', DASH)))}</td>"
            f"<td>{html.escape(', '.join(item.get('reasons', []) or []))}</td>"
            "</tr>"
            for item in problems
        )
        parts.append(
            "<table><tr><th>Тест</th><th>Причины</th></tr>"
            f"{cells}</table>"
        )
    else:
        parts.append("<p>Проблемных тестов нет.</p>")
    warnings = summary.get("warnings") or []
    parts.append("<h2>Warnings</h2>")
    if warnings:
        cells = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('code', DASH)))}</td>"
            f"<td>{html.escape(str(item.get('count', DASH)))}</td>"
            f"<td>{html.escape(', '.join(str(value) for value in item.get('test_ids', []) or []))}</td>"
            "</tr>"
            for item in warnings
        )
        parts.append(
            "<table><tr><th>Код</th><th>Кол-во</th><th>Тесты</th></tr>"
            f"{cells}</table>"
        )
    else:
        parts.append("<p>Предупреждений нет.</p>")
    failed_checks = summary.get("failed_checks") or []
    parts.append("<h2>Failed checks</h2>")
    if failed_checks:
        cells = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('name', DASH)))}</td>"
            f"<td>{html.escape(_format_value(item.get('mandatory')))}</td>"
            f"<td>{html.escape(str(item.get('count', DASH)))}</td>"
            f"<td>{html.escape(', '.join(str(value) for value in item.get('test_ids', []) or []))}</td>"
            f"<td>{html.escape('; '.join(str(value) for value in item.get('details', []) or [] if value) or DASH)}</td>"
            "</tr>"
            for item in failed_checks
        )
        parts.append(
            "<table><tr><th>Check</th><th>Mandatory</th><th>Кол-во</th>"
            f"<th>Тесты</th><th>Детали</th></tr>{cells}</table>"
        )
    else:
        parts.append("<p>Проваленных checks нет.</p>")
    parts.append("<h2>Эффективность</h2>")
    parts.append(
        _html_table(
            [
                ("Средние tool calls", efficiency.get("mean_tool_calls")),
                ("Средние search calls", efficiency.get("mean_search_calls")),
                ("Средняя latency, мс", efficiency.get("mean_latency_ms")),
                ("LLM concurrency", efficiency.get("llm_concurrency")),
            ]
        )
    )
    limitations = summary.get("limitations") or []
    parts.append("<h2>Ограничения</h2>")
    if limitations:
        parts.append(
            "<ul>"
            + "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations)
            + "</ul>"
        )
    else:
        parts.append("<p>Ограничения не зафиксированы.</p>")
    return _html_page(f"Benchmark {run_id}", "\n".join(parts))


def render_comparison_html(comparison: Mapping[str, Any]) -> str:
    parts = ["<h1>Сравнение запусков</h1>"]
    if not comparison.get("comparable"):
        reasons = ", ".join(comparison.get("reasons", []) or [])
        parts.append(
            f"<p><strong>not_comparable</strong>: дельты не выдаются. "
            f"Причины: {html.escape(reasons)}</p>"
        )
    metrics = comparison.get("metrics") or []
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('key', DASH)))}</td>"
        f"<td>{html.escape(_format_value(item.get('base')))}</td>"
        f"<td>{html.escape(_format_value(item.get('candidate')))}</td>"
        f"<td>{html.escape(_format_value(item.get('delta')))}</td>"
        "</tr>"
        for item in metrics
    )
    parts.append(
        "<table><tr><th>Метрика</th><th>До</th><th>После</th><th>Δ</th></tr>"
        f"{rows}</table>"
    )
    return _html_page("Сравнение запусков", "\n".join(parts))


def _short(value: Any, length: int = 12) -> str:
    if not isinstance(value, str) or not value:
        return DASH
    return value if len(value) <= length else value[:length]


def render_json(summary: Mapping[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2) + "\n"


def write_report(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    """Записать ``report.md``, ``report.json`` и ``report.html`` атомарно."""
    path = run_dir / "report.md"
    atomic_write_text(path, render_markdown(summary))
    atomic_write_text(run_dir / "report.json", render_json(summary))
    atomic_write_text(run_dir / "report.html", render_html(summary))
    return path


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json не найден: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json должен содержать объект")
    return payload


def latest_run(results_dir: Path = RESULTS_DIR) -> Path | None:
    candidates = [
        path
        for path in Path(results_dir).iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    ] if Path(results_dir).exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "summary.json").stat().st_mtime)


def compare_summaries(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Сопоставимость и дельты по §8.3; метрика ``None`` не превращается в ноль."""
    base_dataset = (base.get("dataset") or {}).get("sha256")
    candidate_dataset = (candidate.get("dataset") or {}).get("sha256")
    base_corpus = (base.get("corpus") or {}).get("content_sha256")
    candidate_corpus = (candidate.get("corpus") or {}).get("content_sha256")
    reasons: list[str] = []
    if base_dataset != candidate_dataset:
        reasons.append("dataset_sha256_mismatch")
    if base_corpus != candidate_corpus:
        reasons.append("corpus_content_sha256_mismatch")
    if (base.get("suite"), base.get("split")) != (
        candidate.get("suite"),
        candidate.get("split"),
    ):
        reasons.append("suite_or_split_mismatch")
    comparable = not reasons
    metrics = []
    for key in ("pass_rate", "execution_coverage"):
        before = base.get(key)
        after = candidate.get(key)
        delta = (
            after - before
            if comparable and isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
        metrics.append({"key": key, "base": before, "candidate": after, "delta": delta})
    for key in ("recall_at_10", "evidence_coverage"):
        before = ((base.get("groups") or {}).get("retrieval") or {}).get(key)
        after = ((candidate.get("groups") or {}).get("retrieval") or {}).get(key)
        delta = (
            after - before
            if comparable and isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
        metrics.append({"key": key, "base": before, "candidate": after, "delta": delta})
    return {
        "comparable": comparable,
        "reasons": reasons,
        "base": {"id": base.get("run_id"), "git_sha": (base.get("git") or {}).get("sha")},
        "candidate": {
            "id": candidate.get("run_id"),
            "git_sha": (candidate.get("git") or {}).get("sha"),
        },
        "metrics": metrics,
    }


def render_comparison(comparison: Mapping[str, Any]) -> str:
    lines = ["# Сравнение запусков", ""]
    if not comparison.get("comparable"):
        lines.append(
            "**not_comparable**: дельты не выдаются. Причины: "
            + ", ".join(comparison.get("reasons", []))
        )
        lines.append("")
    lines.append("| Метрика | До | После | Δ |")
    lines.append("|---|---:|---:|---:|")
    for item in comparison.get("metrics", []) or []:
        lines.append(
            f"| {item.get('key')} | {_format_value(item.get('base'))} "
            f"| {_format_value(item.get('candidate'))} "
            f"| {_format_value(item.get('delta'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_run(value: str, results_dir: Path) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    candidate = Path(results_dir) / value
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Запуск не найден: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Отчёт benchmark Agent_RAG")
    parser.add_argument("--run-dir", default=None, help="Каталог конкретного запуска")
    parser.add_argument("--latest", action="store_true", help="Последний завершённый запуск")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "CANDIDATE"))
    parser.add_argument("--out", default=None, help="Куда сохранить comparison.md")
    args = parser.parse_args(argv)

    if args.compare:
        results_dir = Path(args.results_dir)
        base_dir = _resolve_run(args.compare[0], results_dir)
        candidate_dir = _resolve_run(args.compare[1], results_dir)
        base = _load_summary(base_dir)
        candidate = _load_summary(candidate_dir)
        comparison = compare_summaries(base, candidate)
        text = render_comparison(comparison)
        html_text = render_comparison_html(comparison)
        out = Path(args.out) if args.out else candidate_dir / "comparison.md"
        atomic_write_text(out, text)
        atomic_write_text(out.with_suffix(".html"), html_text)
        print(text)
        return 0

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.results_dir))
    if run_dir is None:
        print("Запуски не найдены. Сначала: python -m eval.runner --suite smoke")
        return 2
    summary = _load_summary(run_dir)
    # Артефакты отчёта создаются рядом с summary.json (§6.1).
    atomic_write_text(run_dir / "report.md", render_markdown(summary))
    atomic_write_text(run_dir / "report.html", render_html(summary))
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
