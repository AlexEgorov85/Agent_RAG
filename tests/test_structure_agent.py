"""Интеграция Temporal Structure Analysis в существующий AgentService."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.agent.selector import PlanAwareSelector
from src.agent.service import AgentService
from src.agent.simple_router import SimpleRouter
from src.agent.state import Answer, Citation
from src.agent.tools import AgentTools
from src.engine.generator import AnswerGenerator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.temporal_search import PerYearSearch


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
    # Content evidence есть: навигационный warning не нужен.
    assert "structural_claim_without_citation" not in result.answer.warnings
    assert "structural_claim_without_citation" not in result.warnings


class SectionTaxonomyIndex:
    """Индекс с «другой» таксономией section_path, чем у get_structure_tree."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def search(self, query: Any, *, k: int, eligible: Any) -> list[dict[str, Any]]:
        return [dict(chunk) for chunk in self.chunks if eligible(chunk)][:k]


class SectionTaxonomyEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_structural_content_search_falls_back_when_section_paths_differ() -> None:
    """Пути из structure не совпадают с section_path чанков: fallback даёт evidence."""
    chunks: list[dict[str, Any]] = []
    for year, paths in {
        2022: ["01. Стратегия", "02. Управление рисками"],
        2023: [
            "01. Стратегия",
            "02. Управление рисками",
            "03. Кибербезопасность",
        ],
    }.items():
        for index, path in enumerate(paths):
            chunks.append(
                {
                    "chunk_id": f"c{year}-{index}",
                    "doc_id": f"doc{year}",
                    "year": year,
                    "kind": "text",
                    "section_path": path,
                    "text": f"Содержание раздела «{path}» за {year} год.",
                }
            )
    retriever = HybridRetriever(
        SectionTaxonomyIndex(chunks),
        SectionTaxonomyIndex(chunks),
        SectionTaxonomyEmbedder(),
        top_k=3,
    )
    service = AgentService(
        AgentTools(
            search_backend=PerYearSearch(retriever, per_year_k=1),
            structure_backend=structure_backend,
        ),
        answerer=StructureAnswerer(),
        router=SimpleRouter({2022, 2023}),
        action_selector=PlanAwareSelector(),
        documents_by_year={2022: ["doc2022"], 2023: ["doc2023"]},
    )

    result = service.ask(
        "Как менялась структура отчётов с 2022 по 2023 год?",
        conversation_id="structure-fallback",
    )

    assert result.answer.status == "answer", (result.answer, result.warnings)
    assert result.answer.citations
    # На уровне run generic-проекция warnings; точная причина — в diagnostics события.
    assert "retrieval_fallback" in result.warnings
    searches = [
        event
        for event in result.tool_events
        if event.tool_name == "search_reports"
        and event.status == "completed"
    ]
    assert searches
    assert all((event.result_count or 0) > 0 for event in searches)
    assert any(
        (event.diagnostics or {}).get("section_filter_status") == "fallback"
        and (event.diagnostics or {}).get("fallback", {}).get("section_filter") is True
        for event in searches
    )
    assert {citation.year for citation in result.answer.citations} <= {2022, 2023}


class EmptySearch:
    """Content-поиск ничего не находит: структура есть, evidence нет."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(
        self,
        query: str,
        *,
        filters: Any,
        retrieval_mode: str = "hybrid",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append((query, filters, retrieval_mode, top_k))
        return {"hits": [], "warnings": []}


def test_structural_diff_without_content_evidence_answers_navigationally() -> None:
    service = AgentService(
        AgentTools(
            search_backend=EmptySearch(),
            structure_backend=structure_backend,
        ),
        # Генератор не нужен: ответ строится детерминированно по diff.
        answerer=None,
        router=SimpleRouter({2022, 2023}),
        action_selector=PlanAwareSelector(),
        documents_by_year={2022: ["doc2022"], 2023: ["doc2023"]},
    )

    result = service.ask(
        "Как менялась структура отчётов с 2022 по 2023 год?",
        conversation_id="structure-no-evidence",
    )

    assert result.answer.status == "answer", (
        result.answer,
        result.warnings,
        result.diagnostics,
    )
    assert result.structural_changes
    assert "structural_claim_without_citation" in result.answer.warnings
    assert "structural_claim_without_citation" in result.warnings
    # Цитаты не выдумываются при отсутствии content evidence.
    assert result.answer.citations == []
    assert result.answer.facts == []
    text = result.answer.answer
    assert "структур" in text.casefold()
    assert "2022" in text and "2023" in text
    assert "Цифровая трансформация" in text
    assert "Кибербезопасность" in text
    # Это навигация, а не утверждение содержания.
    assert "содержани" in text.casefold()


def test_structural_without_changes_keeps_previous_abstain() -> None:
    stable = {
        2022: [section("Стратегия"), section("Управление рисками")],
        2023: [section("Стратегия"), section("Управление рисками")],
    }

    def stable_structure(
        doc_id: str, *, section_path: str | None = None, max_depth: int = 4
    ) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "section_path": section_path,
            "max_depth": max_depth,
            "sections": stable[int(doc_id[-4:])],
        }

    service = AgentService(
        AgentTools(
            search_backend=EmptySearch(),
            structure_backend=stable_structure,
        ),
        answerer=None,
        router=SimpleRouter({2022, 2023}),
        action_selector=PlanAwareSelector(),
        documents_by_year={2022: ["doc2022"], 2023: ["doc2023"]},
    )

    result = service.ask(
        "Как менялась структура отчётов с 2022 по 2023 год?",
        conversation_id="structure-stable",
    )

    assert result.answer.status == "abstain"
    assert result.structural_changes == []
    assert "no_evidence" in result.answer.warnings
    assert result.answer.citations == []


class NavigationCompletions:
    """Модель отвечает навигационно, без citations, с явным warning."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        payload = {
            "status": "answer",
            "answer": "Структура изменилась: раздел добавлен.",
            "citations": [],
            "facts": [],
            "warnings": ["structural_claim_without_citation"],
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload, ensure_ascii=False)
                    )
                )
            ]
        )


def _navigation_generator() -> tuple[AnswerGenerator, NavigationCompletions]:
    completions = NavigationCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return AnswerGenerator(client, "fake-model"), completions


EVIDENCE = {"c1": {"chunk_id": "c1", "year": 2023, "text": "Текст раздела."}}


def test_generator_accepts_uncited_navigation_answer_for_structural_diff() -> None:
    generator, completions = _navigation_generator()

    answer = generator.generate(
        "Как изменилась структура?",
        EVIDENCE,
        {"c1"},
        structural_changes=[
            {"change_type": "added", "from_year": 2022, "to_year": 2023}
        ],
    )

    assert answer.status == "answer"
    assert answer.citations == []
    assert "structural_claim_without_citation" in answer.warnings
    assert completions.calls == 1


def test_generator_still_requires_citations_without_structural_diff() -> None:
    generator, completions = _navigation_generator()

    answer = generator.generate(
        "Какая чистая прибыль за 2023 год?",
        EVIDENCE,
        {"c1"},
    )

    assert answer.status == "abstain"
    assert answer.citations == []
    # Без structural diff ответ без цитат не принимается, попытка повторяется.
    assert completions.calls == 3
