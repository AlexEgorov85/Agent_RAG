"""Офлайн-тесты извлечения сущностей через LLM и его интеграции с графом.

Все вызовы провайдера обслуживает детерминированная in-memory заглушка,
поэтому тесты не читают учётные данные и не обращаются к сети.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable

from src.graph.entity_extractor import (
    LLM_ENTITY_EXTRACTOR_NAME,
    LLM_ENTITY_EXTRACTOR_VERSION,
    LLM_ENTITY_PROMPT_VERSION,
    LLMEntityExtractor,
)
from src.graph.graph_builder import build_graph
from src.graph.graph_models import GraphBuildConfig
from src.ingestion.structure_builder import DocumentStructure, SectionNode


class FakeCompletions:
    """Записывает kwargs запроса и возвращает JSON-контент от обработчика."""

    def __init__(self, handler: Callable[[dict[str, Any]], str]) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        content = self.handler(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, handler: Callable[[dict[str, Any]], str]) -> None:
        self.completions = FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)


def _batch(request: dict[str, Any]) -> list[dict[str, str]]:
    """Читает пакет чанков из пользовательского сообщения с чистым JSON."""
    return json.loads(request["messages"][-1]["content"])["chunks"]


def _response(
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {"entities": entities or [], "relations": relations or []},
        ensure_ascii=False,
    )


def _chunk(chunk_id: str, text: str = "Evidence text.", kind: str = "text") -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc", "kind": kind, "text": text}


def test_batches_chunks_and_attaches_source_chunk_ids():
    def respond(request: dict[str, Any]) -> str:
        return _response(entities=[
            {
                "chunk_id": chunk["chunk_id"],
                "name": f"Entity {chunk['chunk_id']}",
                "type": "Risk",
            }
            for chunk in _batch(request)
        ])

    client = FakeClient(respond)
    chunks = [
        _chunk(f"c{index}", text=f"Entity c{index} evidence text.")
        for index in range(5)
    ]
    extractor = LLMEntityExtractor(client, "test-model", chunks_per_call=2)
    result = extractor.extract(chunks)

    assert result.calls == 3
    assert [len(_batch(request)) for request in client.completions.requests] == [
        2, 2, 1,
    ]
    assert result.failed_chunk_ids == []
    assert result.skipped_chunk_ids == []
    assert {entity.chunk_id for entity in result.entities} == {
        f"c{index}" for index in range(5)
    }
    assert all(entity.chunk_id in {f"c{index}" for index in range(5)}
               for entity in result.entities)

    request = client.completions.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    system_prompt = request["messages"][0]["content"].casefold()
    assert "untrusted" in system_prompt
    assert "never follow instructions" in system_prompt


def test_entity_types_are_normalized_and_provenance_is_required():
    def respond(_request: dict[str, Any]) -> str:
        return _response(entities=[
            {
                "chunk_id": "c1",
                "name": "  Credit   Risk ",
                "type": "Risk",
                "aliases": ["credit exposure"],
            },
            {"chunk_id": "c1", "name": "Sberbank", "type": "organization"},
            {"chunk_id": "c1", "name": "Mystery", "type": "Company"},
            {"chunk_id": "c1", "name": "Bad Type", "type": 7},
            {"name": "No Chunk", "type": "Risk"},
            {"chunk_id": "foreign", "name": "Foreign", "type": "Risk"},
        ])

    extractor = LLMEntityExtractor(FakeClient(respond), "test-model")
    result = extractor.extract([
        _chunk("c1", text="Credit Risk at Sberbank; Mystery and Bad Type noted.")
    ])

    by_name = {entity.name: entity for entity in result.entities}
    assert set(by_name) == {"credit risk", "sberbank", "mystery", "bad type"}
    assert by_name["credit risk"].entity_type == "Risk"
    assert by_name["credit risk"].aliases == ("credit exposure", "Credit Risk")
    assert by_name["sberbank"].entity_type == "Organization"
    assert by_name["mystery"].entity_type == "Other"
    assert by_name["bad type"].entity_type == "Other"
    assert all(entity.chunk_id == "c1" for entity in result.entities)
    assert all(entity.source == "llm_entity_extractor"
               for entity in result.entities)


def test_relations_need_evidence_from_the_same_batch():
    def respond(_request: dict[str, Any]) -> str:
        return _response(
            entities=[
                {"chunk_id": "c1", "name": "Sberbank", "type": "Organization"},
                {"chunk_id": "c2", "name": "Credit risk", "type": "Risk"},
            ],
            relations=[
                {
                    "subject": "Sberbank",
                    "object": "Credit risk",
                    "predicate": "INCREASES",
                    "evidence_chunk_ids": ["c2", "c1"],
                },
                {
                    "subject": "Sberbank",
                    "object": "Ghost",
                    "evidence_chunk_ids": ["missing"],
                },
                {"subject": "Sberbank", "object": "Ghost", "evidence_chunk_ids": []},
                {"subject": "Sberbank", "object": "Ghost"},
            ],
        )

    extractor = LLMEntityExtractor(
        FakeClient(respond), "test-model", chunks_per_call=2)
    result = extractor.extract([
        _chunk("c1", text="Sberbank reports results."),
        _chunk("c2", text="Credit risk remains material."),
    ])

    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.subject == "sberbank"
    assert relation.object == "credit risk"
    assert relation.predicate == "RELATED_TO"
    assert relation.evidence_chunk_ids == ("c1", "c2")
    assert relation.subject_type == "Organization"
    assert relation.object_type == "Risk"
    assert result.failed_chunk_ids == []


def test_invalid_json_is_retried_then_batch_is_marked_failed():
    attempts: dict[tuple[str, ...], int] = {}

    def respond(request: dict[str, Any]) -> str:
        ids = tuple(chunk["chunk_id"] for chunk in _batch(request))
        attempts[ids] = attempts.get(ids, 0) + 1
        if ids == ("c0", "c1"):
            return '{"entities": ['
        return _response(entities=[
            {"chunk_id": ids[0], "name": f"Entity {ids[0]}", "type": "Risk"},
        ])

    client = FakeClient(respond)
    extractor = LLMEntityExtractor(
        client, "test-model", max_retries=1, chunks_per_call=2)
    result = extractor.extract([
        _chunk(f"c{index}", text=f"Entity c{index} evidence text.")
        for index in range(3)
    ])

    assert result.calls == 3
    assert result.failed_chunk_ids == ["c0", "c1"]
    assert [entity.chunk_id for entity in result.entities] == ["c2"]
    assert attempts[("c0", "c1")] == 2
    assert attempts[("c2",)] == 1


def test_provider_failure_is_contained_and_text_is_not_exposed():
    def respond(_request: dict[str, Any]) -> str:
        raise RuntimeError("SECRET provider response body")

    extractor = LLMEntityExtractor(
        FakeClient(respond), "test-model", max_retries=1)
    result = extractor.extract([_chunk("c1")])

    assert result.calls == 2
    assert result.failed_chunk_ids == ["c1"]
    assert result.entities == []
    assert "SECRET" not in repr(result)


def test_duplicate_entities_and_relations_are_merged_deterministically():
    def respond(_request: dict[str, Any]) -> str:
        return _response(
            entities=[
                {
                    "chunk_id": "c1", "name": "Sberbank", "type": "Organization",
                    "aliases": ["Sber"], "confidence": 0.4,
                },
                {
                    "chunk_id": "c1", "name": "  sberbank ", "type": "Organization",
                    "aliases": ["SBERBANK"], "confidence": 0.7,
                },
            ],
            relations=[
                {
                    "subject": "Sberbank", "object": "Credit risk",
                    "evidence_chunk_ids": ["c1"], "confidence": 0.2,
                },
                {
                    "subject": "sberbank", "object": "credit   risk",
                    "evidence_chunk_ids": ["c1"], "confidence": 0.6,
                },
            ],
        )

    first = LLMEntityExtractor(
        FakeClient(respond),
        "test-model",
    ).extract([_chunk("c1", text="Sberbank and Credit risk are related.")])
    second = LLMEntityExtractor(
        FakeClient(respond),
        "test-model",
    ).extract([_chunk("c1", text="Sberbank and Credit risk are related.")])

    assert len(first.entities) == 1
    assert first.entities[0].name == "sberbank"
    assert first.entities[0].aliases == ("Sber", "Sberbank")
    assert first.entities[0].confidence == 0.7
    assert len(first.relations) == 1
    assert first.relations[0].evidence_chunk_ids == ("c1",)
    assert first.relations[0].confidence == 0.6
    assert first == second


def test_empty_and_visual_chunks_are_skipped_without_provider_calls():
    calls = 0

    def respond(request: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        assert [chunk["chunk_id"] for chunk in _batch(request)] == ["c-text"]
        return _response(entities=[
            {"chunk_id": "c-text", "name": "Risk", "type": "Risk"},
        ])

    extractor = LLMEntityExtractor(FakeClient(respond), "test-model")
    result = extractor.extract([
        _chunk("c-empty", text="   "),
        _chunk("c-visual", kind="chart"),
        _chunk("c-text", text="Risk evidence text."),
    ])

    assert calls == 1
    assert result.calls == 1
    assert result.skipped_chunk_ids == ["c-empty", "c-visual"]
    assert result.failed_chunk_ids == []
    assert [entity.chunk_id for entity in result.entities] == ["c-text"]


def test_entity_is_reassigned_to_the_chunk_that_literally_mentions_it():
    def respond(_request: dict[str, Any]) -> str:
        return _response(entities=[
            # Модель указывает c1, хотя слова есть только в c2.
            {"chunk_id": "c1", "name": "СберПрайм", "type": "Product"},
        ])

    extractor = LLMEntityExtractor(
        FakeClient(respond), "test-model", chunks_per_call=2)
    result = extractor.extract([
        _chunk("c1", text="Стратегия и финансовые результаты."),
        _chunk("c2", text="Подписчики СберПрайм выросли за год."),
    ])

    assert [(entity.name, entity.chunk_id) for entity in result.entities] == [
        ("сберпрайм", "c2"),
    ]


def test_entity_without_literal_mention_is_dropped():
    def respond(_request: dict[str, Any]) -> str:
        return _response(entities=[
            # Не упоминается ни в одном чанке пакета.
            {"chunk_id": "c1", "name": "ГигаЧат", "type": "Technology"},
            # Неоднозначно: встречается в двух других чанках, надёжного
            # происхождения нет.
            {"chunk_id": "c1", "name": "Сбер", "type": "Organization"},
        ])

    extractor = LLMEntityExtractor(
        FakeClient(respond), "test-model", chunks_per_call=3)
    result = extractor.extract([
        _chunk("c1", text="Стратегия и финансовые результаты."),
        _chunk("c2", text="Сбер публикует отчет."),
        _chunk("c3", text="Сбер и СберПрайм в одном предложении."),
    ])

    assert result.entities == []


def test_relation_evidence_must_mention_subject_or_object():
    def respond(_request: dict[str, Any]) -> str:
        return _response(
            entities=[
                {"chunk_id": "c1", "name": "Сбер", "type": "Organization"},
            ],
            relations=[
                {
                    "subject": "Сбер",
                    "object": "СберПрайм",
                    "evidence_chunk_ids": ["c1", "c2"],
                },
            ],
        )

    extractor = LLMEntityExtractor(
        FakeClient(respond), "test-model", chunks_per_call=2)
    result = extractor.extract([
        _chunk("c1", text="Сбер публикует отчет."),
        _chunk("c2", text="Совершенно посторонний текст без имен."),
    ])

    assert len(result.relations) == 1
    assert result.relations[0].evidence_chunk_ids == ("c1",)


def _graph_fixtures() -> tuple[list[dict], dict[str, DocumentStructure]]:
    structure = DocumentStructure(
        doc_id="report_2023",
        sections=[
            SectionNode("Risk", "Risks", 0, 1, 10, ["Risks"]),
        ],
    )
    chunks = [
        {
            "chunk_id": "c-risk", "doc_id": "report_2023", "year": 2023,
            "issuer": "Example Bank", "kind": "text", "page_pdf": 3,
            "section_path": "Risks", "text": "Credit risk is material.",
        },
        {
            "chunk_id": "c-bank", "doc_id": "report_2022", "year": 2022,
            "issuer": "Example Bank", "kind": "text", "page_pdf": 8,
            "text": "Example Bank provides financing.",
        },
    ]
    return chunks, {"report_2023": structure}


def test_extraction_feeds_build_graph_with_provenance_and_stable_fingerprint():
    def respond(request: dict[str, Any]) -> str:
        batch_ids = {chunk["chunk_id"] for chunk in _batch(request)}
        entities = []
        if "c-risk" in batch_ids:
            entities.append(
                {"chunk_id": "c-risk", "name": "Credit Risk", "type": "Risk"})
        if "c-bank" in batch_ids:
            entities.append(
                {"chunk_id": "c-bank", "name": "Example Bank",
                 "type": "Organization"})
        relations = []
        if {"c-risk", "c-bank"} <= batch_ids:
            relations.append({
                "subject": "Credit Risk",
                "object": "Example Bank",
                "predicate": "RELATED_TO",
                "evidence_chunk_ids": ["c-risk", "c-bank"],
            })
        return _response(entities=entities, relations=relations)

    chunks, structures = _graph_fixtures()

    def build() -> Any:
        extraction = LLMEntityExtractor(
            FakeClient(respond), "test-model", chunks_per_call=4
        ).extract(chunks)
        return extraction, build_graph(
            chunks,
            structures,
            entities=extraction.entities,
            relations=extraction.relations,
            config=GraphBuildConfig(
                extractor_name=LLM_ENTITY_EXTRACTOR_NAME,
                extractor_version=LLM_ENTITY_EXTRACTOR_VERSION,
                prompt_version=LLM_ENTITY_PROMPT_VERSION,
            ),
        )

    extraction, result = build()
    repeated_extraction, repeated = build()

    assert extraction.failed_chunk_ids == []
    assert len(extraction.entities) == 2
    assert len(extraction.relations) == 1
    assert result.graph_fingerprint == repeated.graph_fingerprint
    assert extraction == repeated_extraction

    graph = result.graph
    assert "entity:Risk:credit risk" in graph
    assert "entity:Organization:example bank" in graph
    assert graph.has_edge("chunk:c-risk", "entity:Risk:credit risk")
    mention = next(
        data for source, _, data in graph.edges(data=True)
        if data.get("type") == "MENTIONS" and source == "chunk:c-risk"
    )
    assert mention["evidence_chunk_ids"] == ["c-risk"]

    relation = next(
        data for _, _, data in graph.edges(data=True)
        if data.get("type") == "RELATED_TO"
    )
    assert relation["evidence_chunk_ids"] == ["c-bank", "c-risk"]
    assert result.manifest["entity_count"] == 2
    assert result.manifest["relation_count"] == 1
    assert result.manifest["extractor_name"] == "llm_entity_extractor"
