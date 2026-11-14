"""Офлайн-тесты фабрики AgentService на локальном корпусе."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent.factory import create_agent_service
from src.agent.service import create_agent_service as service_factory
from src.agent.tools import ToolUnavailable
from src.core.rag_config import RAGSettings


class FakeEmbeddingsEndpoint:
    def create(self, *, model: str, input: list[str]) -> dict[str, Any]:
        return {
            "data": [
                {"index": index, "embedding": [float(len(text)), 1.0]}
                for index, text in enumerate(input)
            ]
        }


class FakeCompletionsEndpoint:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = {
            "status": "answer",
            "answer": "Чистая прибыль составила 1 508,6 млрд рублей.",
            "citations": [{"chunk_id": self.chunk_id, "quote": "1 508,6"}],
            "facts": [],
            "warnings": [],
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


@pytest.fixture
def fake_client() -> Any:
    return SimpleNamespace(
        embeddings=SimpleNamespace(create=FakeEmbeddingsEndpoint().create),
        chat=SimpleNamespace(
            completions=FakeCompletionsEndpoint("sber_2023:00001:c0").create
        ),
    )


def _chunk() -> dict[str, Any]:
    return {
        "chunk_id": "sber_2023:00001:c0",
        "evidence_id": "sber_2023:p008",
        "doc_id": "sber_2023",
        "title": "Годовой отчёт",
        "page_pdf": 8,
        "page_label": "7",
        "year": 2023,
        "report_period": "2023",
        "kind": "text",
        "element_ids": ["sber_2023:00001"],
        "text": "По итогам 2023 года чистая прибыль составила 1 508,6 млрд рублей.",
        "html": None,
        "source_url": None,
        "source_sha256": "a" * 64,
        "section_path": "Обзор результатов",
        "breadcrumbs": ["Обзор результатов"],
        "section_level": 0,
        "parent_section": None,
    }


def _settings(tmp_path: Path) -> RAGSettings:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "sber_2023_chunks.json").write_text(
        json.dumps([_chunk()], ensure_ascii=False), encoding="utf-8"
    )
    return RAGSettings(
        llm_api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        model="fake-chat",
        embeddings_model="fake-embed",
        chunks_dir=chunks_dir,
        cache_dir=tmp_path / "cache",
    )


def test_factory_builds_service_and_answers_with_citation(
    tmp_path: Path, fake_client: Any
) -> None:
    service = create_agent_service(
        settings=_settings(tmp_path), client=fake_client, use_reranker=False
    )

    result = service.ask("Какая чистая прибыль за 2023 год?", conversation_id="c1")

    assert result.answer.status == "answer", (
        result.answer.answer,
        result.answer.warnings,
        result.warnings,
        [(e.tool_name, e.status, e.result_count) for e in result.tool_events],
    )
    assert [c.chunk_id for c in result.answer.citations] == ["sber_2023:00001:c0"]
    assert result.answer.citations[0].page_pdf == 8
    assert [event.tool_name for event in result.tool_events] == ["search_reports"]
    # Для этого детерминированного теста реранкер явно отключён.
    assert service.tools.search_backend.reranker is None


def test_reranker_is_enabled_by_default_and_can_be_disabled(
    tmp_path: Path, fake_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    default_service = create_agent_service(settings=settings, client=fake_client)
    assert default_service.tools.search_backend.reranker is not None

    disabled_service = create_agent_service(
        settings=settings, client=fake_client, use_reranker=False
    )
    assert disabled_service.tools.search_backend.reranker is None

    monkeypatch.setenv("RERANK_ENABLED", "0")
    env_disabled = create_agent_service(settings=settings, client=fake_client)
    assert env_disabled.tools.search_backend.reranker is None


def test_factory_requires_a_built_corpus(tmp_path: Path, fake_client: Any) -> None:
    settings = RAGSettings(
        llm_api_key="test",
        chunks_dir=tmp_path / "missing",
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="Корпус не найден"):
        create_agent_service(settings=settings, client=fake_client)


def test_local_tools_read_structure_and_chunks(
    tmp_path: Path, fake_client: Any
) -> None:
    settings = _settings(tmp_path)
    structure_dir = tmp_path / "structure"
    structure_dir.mkdir()
    (structure_dir / "sber_2023_sections.json").write_text(
        json.dumps(
            {
                "doc_id": "sber_2023",
                "sections": [
                    {
                        "id": "Обзор",
                        "title": "Обзор результатов",
                        "level": 0,
                        "page_start": 8,
                        "page_end": 10,
                        "breadcrumbs": ["Обзор результатов"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = create_agent_service(
        settings=settings, client=fake_client, structure_dir=structure_dir
    )

    tree = service.tools.invoke(
        "get_structure_tree", {"doc_id": "sber_2023", "max_depth": 2}
    )
    chunk = service.tools.invoke(
        "get_chunk", {"chunk_id": "sber_2023:00001:c0"}
    )

    assert [section["title"] for section in tree["sections"]] == [
        "Обзор результатов"
    ]
    assert chunk["page_pdf"] == 8
    with pytest.raises(ToolUnavailable):
        service.tools.invoke("get_chunk", {"chunk_id": "unknown"})
    with pytest.raises(ToolUnavailable):
        service.tools.invoke("get_structure_tree", {"doc_id": "sber_1999"})


def test_service_module_exposes_lazy_factory(tmp_path: Path, fake_client: Any) -> None:
    service = service_factory(settings=_settings(tmp_path), client=fake_client)

    assert service.ask("вопрос", conversation_id="c2").answer.status == "answer"
