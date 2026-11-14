"""Интеграция Temporal Structure Analysis в существующий AgentService."""

from __future__ import annotations

from typing import Any

from src.agent.selector import PlanAwareSelector
from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation
from src.agent.tools import AgentTools


def section(
    title: str,
    *,
    parent: str | None = None,
    start: int = 10,
    end: int = 12,
) -> dict[str, Any]:
    path = f"{parent}/{title}" if parent else title
    return {
        "title": title,
        "section_path": path,
        "breadcrumbs": [parent, title] if parent else [title],
        "level": 2 if parent else 1,
        "page_start": start,
        "page_end": end,
    }


SNAPSHOTS = {
    2022: [
        section("Стратегия"),
        section("Цифровая трансформация", parent="Стратегия"),
        section("Управление рисками"),
    ],
    2023: [
        section("Стратегия"),
        section(
            "Цифровая трансформация и технологии",
            parent="Стратегия",
            start=10,
            end=18,
        ),
        section("Система управления рисками"),
        section("Кибербезопасность", parent="Управление рисками"),
    ],
}


def structure_backend(
    doc_id: str, *, section_path: str | None = None, max_depth: int = 4
) -> dict[str, Any]:
    year = int(doc_id[-4:])
    return {
        "doc_id": doc_id,
        "section_path": section_path,
        "max_depth": max_depth,
        "sections": SNAPSHOTS[year],
    }


def search_backend(query: str, **kwargs: Any) -> dict[str, Any]:
    filters = kwargs["filters"]
    year = next(iter(filters.years or {2023}))
    path = filters.section_path_prefix or "Раздел"
    return {
        "hits": [
            {
                "chunk_id": f"c{year}",
                "chunk": {
                    "chunk_id": f"c{year}",
                    "year": year,
                    "doc_id": f"doc{year}",
                    "section_path": path,
                    "text": f"Содержание раздела за {year} год.",
                },
            }
        ],
        "warnings": [],
    }


class StructureAnswerer:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}

    def generate(
        self,
        question: str,
        evidence: Any,
        evidence_manifest: set[str],
        **kwargs: Any,
    ) -> Answer:
        self.context = dict(kwargs)
        chunk_id = sorted(evidence_manifest)[0]
        chunk = evidence[chunk_id]
        return Answer(
            "answer",
            "Структура и содержание разделов изменились.",
            citations=[
                Citation(
                    chunk_id=chunk_id,
                    year=chunk.get("year"),
                    section_path=chunk.get("section_path"),
                )
            ],
        )


def test_structure_analysis_uses_existing_tools_and_agent_loop() -> None:
    answerer = StructureAnswerer()
    service = AgentService(
        AgentTools(
            search_backend=search_backend,
            structure_backend=structure_backend,
        ),
        answerer=answerer,
        router=SimpleRouter({2022, 2023}),
        action_selector=PlanAwareSelector(),
        documents_by_year={2022: ["doc2022"], 2023: ["doc2023"]},
    )

    result = service.ask(
        "Как менялась структура отчётов с 2022 по 2023 год?",
        conversation_id="structure",
    )

    assert result.answer.status == "answer", (
        result.answer,
        result.warnings,
        [(event.tool_name, event.result_count, event.status) for event in result.tool_events],
    )
    assert result.structural_changes
    assert answerer.context.get("structural_changes")
    names = [event.tool_name for event in result.tool_events]
    assert names.count("get_structure_tree") == 2
    assert "search_reports" in names
    assert result.answer.citations
