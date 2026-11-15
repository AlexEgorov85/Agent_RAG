# -*- coding: utf-8 -*-
"""Acceptance-сценарии на готовом dev-корпусе (без парсинга PDF).

Скрипт собирает реальный ``AgentService`` через фабрику и прогоняет четыре
сценария: factoid / temporal / structural / abstain. Результаты (JSON+Markdown)
пишутся в ``data/dev/acceptance/``. Если ключ/LLM недоступны, сценарии честно
помечаются ``skipped``/``error`` — фейковый ответ не подставляется.

Запуск::

    .venv\\Scripts\\python.exe scripts/acceptance_dev.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DEFAULT_OUTPUT_DIR = PROJECT / "data" / "dev" / "acceptance"
DEFAULT_CHUNKS_DIR = PROJECT / "data" / "dev" / "chunks"
DEFAULT_STRUCTURE_DIR = PROJECT / "data" / "structure"
DEFAULT_GRAPH_DIR = PROJECT / "data" / "graph"

_REPORT_SCENARIOS = "acceptance_report.json"
_REPORT_MARKDOWN = "acceptance_report.md"


@dataclass(frozen=True)
class Scenario:
    """Один acceptance-сценарий с явным ожиданием и годами."""

    name: str
    intent: str
    question: str
    expectation: str
    #: Явные годы сценария (как фильтр пользователя). Пусто — годы из вопроса.
    years: tuple[int, ...] = ()


def acceptance_scenarios() -> list[Scenario]:
    """Четыре сценария на годах dev-корпуса 2016/2017/2022/2023.

    2021 в корпусе отсутствует, поэтому abstain-сценарий проверяет честный
    отказ, а не «похожий» ответ из другого года. Годы задаются явным
    фильтром: acceptance проверяет пайплайн, а не угадывание периода.
    """
    return [
        Scenario(
            name="factoid_2023",
            intent="factoid",
            question="Какая чистая прибыль Сбербанка за 2023 год?",
            expectation="answer со ссылкой на источник 2023 года",
            years=(2023,),
        ),
        Scenario(
            name="temporal_2016_2017",
            intent="temporal",
            question=(
                "Сравни чистую прибыль Сбербанка за 2016 и 2017 годы."
            ),
            expectation="answer с цитатами за 2016 и 2017 годы",
            years=(2016, 2017),
        ),
        Scenario(
            name="structural_2016_2023",
            intent="structural",
            question=(
                "Как изменилась структура разделов годового отчёта "
                "Сбербанка между 2016 и 2023 годами?"
            ),
            expectation=(
                "≥2 успешных get_structure_tree и найденные структурные "
                "изменения либо подтверждённый ответ"
            ),
            years=(2016, 2023),
        ),
        Scenario(
            name="abstain_2021",
            intent="abstain",
            question="Какая чистая прибыль Сбербанка за 2021 год?",
            expectation="abstain без citations: 2021 года нет в dev-корпусе",
            years=(2021,),
        ),
    ]


_SECRET_PATTERNS = (
    (r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+",
     r"\1=[REDACTED]"),
    (r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]"),
    (r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]"),
)


def _scrub(value: Any, limit: int = 400) -> str:
    """Короткий безопасный текст: без ключей/токенов и без огромных полей."""
    text = str(value)[:limit]
    for pattern, replacement in _SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _scrub(value)


def _citation_payload(citation: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(citation, "chunk_id", None),
        "doc_id": getattr(citation, "doc_id", None),
        "year": getattr(citation, "year", None),
        "page_pdf": getattr(citation, "page_pdf", None),
        "page_label": getattr(citation, "page_label", None),
        "section_path": getattr(citation, "section_path", None),
        "quote": (
            _scrub(citation.quote, 240)
            if getattr(citation, "quote", None)
            else None
        ),
    }


def _tool_sequence(run: Any) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for event in getattr(run, "tool_events", []) or []:
        sequence.append(
            {
                "tool": getattr(event, "tool_name", None),
                "status": getattr(event, "status", None),
                "result_count": getattr(event, "result_count", None),
                "duration_ms": round(
                    float(getattr(event, "duration_ms", 0.0) or 0.0), 1
                ),
                "warning": getattr(event, "warning", None),
            }
        )
    return sequence


def _coverage_payload(diagnostics: Mapping[str, Any]) -> dict[str, Any] | None:
    coverage = diagnostics.get("coverage")
    if not isinstance(coverage, Mapping):
        return None
    # CoverageReport.to_dict содержит только статусы/годы/metric id, без
    # текста чанков; сохраняем его как есть для разбора.
    return _json_safe(coverage)


def _verification_payload(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any] | None:
    verification = diagnostics.get("verification")
    return _json_safe(verification) if isinstance(verification, Mapping) else None


def _structural_payload(run: Any) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for item in getattr(run, "structural_changes", []) or []:
        if not isinstance(item, Mapping):
            continue
        section = item.get("to_section") or item.get("from_section") or {}
        section = section if isinstance(section, Mapping) else {}
        changes.append(
            {
                "change_type": item.get("change_type"),
                "from_year": item.get("from_year"),
                "to_year": item.get("to_year"),
                "section": section.get("title") or section.get("section_path"),
                "detail": item.get("detail"),
            }
        )
    return changes[:20]


def _evaluate(scenario: Scenario, record: Mapping[str, Any]) -> list[str]:
    """Возвращает коды провалов; пустой список — сценарий пройден."""
    failures: list[str] = []
    if record.get("run_status") == "failed":
        failures.append("run_failed")
    if record.get("error_code"):
        failures.append(f"error:{record['error_code']}")

    answer_status = record.get("answer_status")
    citations = record.get("citations") or []
    cited_years = {
        item.get("year")
        for item in citations
        if isinstance(item, Mapping)
    }
    tools = record.get("tool_sequence") or []

    if scenario.name == "factoid_2023":
        if answer_status != "answer":
            failures.append("expected_answer")
        if 2023 not in cited_years:
            failures.append("expected_2023_citation")
    elif scenario.name == "temporal_2016_2017":
        if answer_status != "answer":
            failures.append("expected_answer")
        if not {2016, 2017}.issubset(cited_years):
            failures.append("expected_2016_2017_citations")
    elif scenario.name == "structural_2016_2023":
        structure_ok = sum(
            1
            for item in tools
            if item.get("tool") == "get_structure_tree"
            and item.get("status") == "completed"
            and (item.get("result_count") or 0) > 0
        )
        if structure_ok < 2:
            failures.append("expected_two_structure_trees")
        if not record.get("structural_changes"):
            failures.append("expected_structural_changes")
        # Внутренние изменения бесполезны, если ответ их не подтвердил:
        # acceptance проверяет пользовательский результат, а не только анализ.
        if answer_status != "answer":
            failures.append("expected_answer")
        if not citations:
            failures.append("expected_citations")
    elif scenario.name == "abstain_2021":
        if answer_status != "abstain":
            failures.append("expected_abstain")
        if citations:
            failures.append("abstain_must_not_cite")
    return failures


def _record_from_run(
    scenario: Scenario,
    run: Any,
    *,
    elapsed_ms: float,
) -> dict[str, Any]:
    diagnostics = getattr(run, "diagnostics", {}) or {}
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    plan = diagnostics.get("query_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    answer = getattr(run, "answer", None)
    citations = [
        _citation_payload(item)
        for item in (getattr(answer, "citations", []) or [])
    ]
    timings = {
        str(key): round(float(value), 1)
        for key, value in (getattr(run, "timings", {}) or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "scenario": scenario.name,
        "expected_intent": scenario.intent,
        "question": scenario.question,
        "expectation": scenario.expectation,
        "years": list(scenario.years),
        "run_status": getattr(run, "status", None),
        "answer_status": getattr(answer, "status", None),
        "answer": getattr(answer, "answer", ""),
        "intent": plan.get("intent"),
        "citations": citations,
        "warnings": list(getattr(run, "warnings", []) or []),
        "coverage": _coverage_payload(diagnostics),
        "verification": _verification_payload(diagnostics),
        "tool_sequence": _tool_sequence(run),
        "structural_changes": _structural_payload(run),
        "timings": {**timings, "measured_ms": round(elapsed_ms, 1)},
        "request_id": getattr(run, "request_id", None),
    }
    failures = _evaluate(scenario, record)
    record["failures"] = failures
    record["outcome"] = "failed" if failures else "passed"
    return record


def _skipped_record(scenario: Scenario, reason: str) -> dict[str, Any]:
    return {
        "scenario": scenario.name,
        "expected_intent": scenario.intent,
        "question": scenario.question,
        "expectation": scenario.expectation,
        "years": list(scenario.years),
        "outcome": "skipped",
        "skip_reason": reason,
        "failures": [f"skipped:{reason}"],
    }


def _error_record(
    scenario: Scenario,
    exc: BaseException,
    *,
    elapsed_ms: float,
) -> dict[str, Any]:
    record = {
        "scenario": scenario.name,
        "expected_intent": scenario.intent,
        "question": scenario.question,
        "expectation": scenario.expectation,
        "years": list(scenario.years),
        "outcome": "error",
        "error_code": type(exc).__name__,
        "error_detail": _scrub(exc, 300),
        "timings": {"measured_ms": round(elapsed_ms, 1)},
        "failures": [f"error:{type(exc).__name__}"],
    }
    return record


def run_scenarios(
    service: Any,
    scenarios: Sequence[Scenario] | None = None,
) -> list[dict[str, Any]]:
    """Прогнать сценарии через ``service.ask``; без сети и без подмены ответов."""
    from src.retrieval.hybrid_retriever import SearchFilters

    records: list[dict[str, Any]] = []
    for scenario in scenarios or acceptance_scenarios():
        started = time.perf_counter()
        filters = (
            SearchFilters(years=set(scenario.years)) if scenario.years else None
        )
        try:
            run = service.ask(
                scenario.question,
                conversation_id=f"acceptance-{scenario.name}",
                filters=filters,
            )
        except Exception as exc:  # noqa: BLE001 — провал сценария не должен скрываться
            records.append(
                _error_record(
                    scenario,
                    exc,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            )
            continue
        records.append(
            _record_from_run(
                scenario,
                run,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return records


def _settings_payload(settings: Any) -> dict[str, Any]:
    keys = (
        "model",
        "base_url",
        "rerank_model",
        "max_tool_calls",
        "max_search_calls",
        "per_year_k",
        "max_years",
        "max_structure_docs",
        "llm_router_enabled",
        "llm_rewrite_enabled",
        "llm_structure_verify",
        "rerank_enabled",
        "top_k",
    )
    return {key: getattr(settings, key, None) for key in keys}


def _corpus_payload(chunks_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunks_dir": str(chunks_dir),
        "documents": [],
        "years": [],
        "chunks": 0,
    }
    try:
        from src.core.corpus import load_chunks

        chunks = load_chunks(chunks_dir)
    except Exception as exc:  # noqa: BLE001 — метаданные не должны ронять отчёт
        payload["error"] = type(exc).__name__
        return payload
    payload["chunks"] = len(chunks)
    payload["documents"] = sorted(
        {str(chunk.get("doc_id")) for chunk in chunks if chunk.get("doc_id")}
    )
    payload["years"] = sorted(
        {
            int(chunk["year"])
            for chunk in chunks
            if isinstance(chunk.get("year"), int)
            and not isinstance(chunk.get("year"), bool)
        }
    )
    return payload


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    settings: Any | None = None,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    """Собрать отчёт с честной сводкой passed/failed/skipped/error."""
    outcomes = [str(item.get("outcome")) for item in records]
    summary = {
        "total": len(records),
        "passed": outcomes.count("passed"),
        "failed": outcomes.count("failed"),
        "skipped": outcomes.count("skipped"),
        "errors": outcomes.count("error"),
    }
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": _corpus_payload(chunks_dir),
        "settings": _settings_payload(settings) if settings is not None else None,
        "skipped_reason": skipped_reason,
        "summary": summary,
        "scenarios": [_json_safe(item) for item in records],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Markdown-отчёт: сводка + детали каждого сценария."""
    summary = report.get("summary") or {}
    corpus = report.get("corpus") or {}
    settings = report.get("settings") or {}
    lines = [
        "# Acceptance report — dev corpus",
        "",
        f"- Сгенерировано: `{report.get('generated_at')}`",
        f"- Корпус: `{corpus.get('chunks_dir')}`",
        (
            f"- Чанков: {corpus.get('chunks')} | документы: "
            f"{', '.join(corpus.get('documents') or []) or '—'} | годы: "
            f"{', '.join(str(year) for year in corpus.get('years') or []) or '—'}"
        ),
    ]
    if settings:
        lines.append(
            "- Лимиты: "
            f"MAX_TOOL_CALLS={settings.get('max_tool_calls')}, "
            f"MAX_SEARCH_CALLS={settings.get('max_search_calls')}, "
            f"PER_YEAR_K={settings.get('per_year_k')}, "
            f"MAX_YEARS={settings.get('max_years')}, "
            f"MAX_STRUCTURE_DOCS={settings.get('max_structure_docs')}"
        )
        lines.append(
            "- Флаги LLM: "
            f"router={settings.get('llm_router_enabled')}, "
            f"rewrite={settings.get('llm_rewrite_enabled')}, "
            f"structure_verify={settings.get('llm_structure_verify')}, "
            f"rerank={settings.get('rerank_enabled')}"
        )
    if report.get("skipped_reason"):
        lines.append(f"- Причина skip: `{report['skipped_reason']}`")
    lines += [
        "",
        "## Summary",
        "",
        (
            f"passed={summary.get('passed')} | failed={summary.get('failed')} | "
            f"skipped={summary.get('skipped')} | errors={summary.get('errors')}"
        ),
        "",
        "| scenario | outcome | intent | run | answer |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report.get("scenarios") or []:
        lines.append(
            "| {scenario} | {outcome} | {intent} | {run} | {answer} |".format(
                scenario=item.get("scenario"),
                outcome=item.get("outcome"),
                intent=item.get("intent") or item.get("expected_intent") or "—",
                run=item.get("run_status") or "—",
                answer=item.get("answer_status") or "—",
            )
        )
    lines.append("")
    for item in report.get("scenarios") or []:
        lines += [
            f"## {item.get('scenario')}",
            "",
            f"- Вопрос: {item.get('question')}",
            f"- Ожидание: {item.get('expectation')}",
            f"- Outcome: **{item.get('outcome')}**",
        ]
        if item.get("skip_reason"):
            lines.append(f"- Skip: `{item['skip_reason']}`")
        if item.get("error_code"):
            lines.append(
                f"- Error: `{item.get('error_code')}` — {item.get('error_detail') or '—'}"
            )
        if item.get("run_status") is not None:
            lines.append(
                f"- Run status: `{item.get('run_status')}` | answer: "
                f"`{item.get('answer_status')}` | intent: "
                f"`{item.get('intent') or '—'}`"
            )
            lines.append("")
            lines.append("Ответ:")
            lines.append("")
            answer_text = item.get("answer") or "—"
            lines.extend(f"> {line}" for line in str(answer_text).splitlines() or ["—"])
            lines.append("")
            citations = item.get("citations") or []
            if citations:
                lines.append("Citations:")
                lines.append("")
                lines.append("| chunk_id | year | doc_id | page | section |")
                lines.append("| --- | --- | --- | --- | --- |")
                for citation in citations:
                    lines.append(
                        "| {chunk} | {year} | {doc} | {page} | {section} |".format(
                            chunk=citation.get("chunk_id"),
                            year=citation.get("year"),
                            doc=citation.get("doc_id"),
                            page=citation.get("page_pdf")
                            or citation.get("page_label")
                            or "—",
                            section=citation.get("section_path") or "—",
                        )
                    )
                lines.append("")
            else:
                lines += ["Citations: —", ""]
            lines.append(
                f"Coverage: `{((item.get('coverage') or {}).get('status')) or '—'}`"
                f" | missing_years: "
                f"{(item.get('coverage') or {}).get('missing_years') or []}"
            )
            verification = item.get("verification") or {}
            issues = [
                issue.get("code")
                for issue in verification.get("issues") or []
                if isinstance(issue, Mapping)
            ]
            lines.append(
                f"Verification: `{verification.get('status') or '—'}`"
                f" | issues: {issues or []}"
            )
            lines.append(
                f"Warnings: {', '.join(item.get('warnings') or []) or '—'}"
            )
            if item.get("structural_changes"):
                lines.append(
                    "Structural changes: "
                    f"{len(item['structural_changes'])}"
                )
            tools = item.get("tool_sequence") or []
            lines.append(
                "Tools: "
                + (
                    " → ".join(
                        f"{tool.get('tool')}:{tool.get('status')}"
                        f"({tool.get('result_count')})"
                        for tool in tools
                    )
                    or "—"
                )
            )
            timings = item.get("timings") or {}
            lines.append(
                "Timings: "
                + ", ".join(
                    f"{key}={value}ms" for key, value in timings.items()
                )
            )
        if item.get("failures"):
            lines.append(
                f"- Failures: {', '.join(str(code) for code in item['failures'])}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Записать JSON и Markdown; возвращает пути файлов."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / _REPORT_SCENARIOS
    markdown_path = output_dir / _REPORT_MARKDOWN
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _skip_reason(settings: Any, chunks_dir: Path) -> str | None:
    key = str(getattr(settings, "llm_api_key", "") or "").strip()
    if not key:
        return "llm_api_key_missing"
    if "ВСТАВИТЬ" in key.upper():
        return "llm_api_key_placeholder"
    if not chunks_dir.is_dir():
        return "corpus_missing"
    return None


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT / path)


def _build_service(settings: Any, args: argparse.Namespace) -> Any:
    from src.agent.factory import create_agent_service

    cache_dir = Path(getattr(settings, "cache_dir", "data/dev/cache"))
    cache_dir = _project_path(cache_dir)
    return create_agent_service(
        settings=settings,
        chunks_dir=_project_path(args.chunks_dir),
        structure_dir=_project_path(args.structure_dir),
        graph_dir=_project_path(args.graph_dir),
        cache_dir=cache_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acceptance-сценарии Structure-Aware Graph RAG на dev-корпусе"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR))
    parser.add_argument("--structure-dir", default=str(DEFAULT_STRUCTURE_DIR))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Прогнать только выбранный сценарий (можно несколько раз)",
    )
    args = parser.parse_args(argv)

    from src.core.rag_config import load_rag_settings

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env")
    except Exception:  # noqa: BLE001 — .env опционален
        pass

    settings = load_rag_settings()
    chunks_dir = _project_path(args.chunks_dir)
    output_dir = _project_path(args.output_dir)
    scenarios = acceptance_scenarios()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [item for item in scenarios if item.name in wanted]
        if not scenarios:
            print(f"Неизвестные сценарии: {sorted(wanted)}")
            return 2

    skip_reason = _skip_reason(settings, chunks_dir)
    records: list[dict[str, Any]]
    if skip_reason is not None:
        records = [_skipped_record(item, skip_reason) for item in scenarios]
    else:
        try:
            service = _build_service(settings, args)
        except Exception as exc:  # noqa: BLE001 — честный skip вместо фейка
            reason = f"service_unavailable:{type(exc).__name__}"
            print(f"Service недоступен: {reason} ({_scrub(exc, 300)})")
            records = [ _skipped_record(item, reason) for item in scenarios ]
        else:
            records = run_scenarios(service, scenarios)

    report = build_report(
        records,
        settings=settings,
        chunks_dir=chunks_dir,
        skipped_reason=skip_reason,
    )
    json_path, markdown_path = write_report(report, output_dir)
    summary = report["summary"]
    print(
        "Acceptance: "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"skipped={summary['skipped']} errors={summary['errors']}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    if summary["failed"] or summary["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
