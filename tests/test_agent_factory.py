"""Офлайн-тесты фабрики AgentService на локальном корпусе."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent.factory import create_agent_service
from src.agent.llm_router import LLMRouter
from src.agent.service import create_agent_service as service_factory
from src.agent.tools import ToolUnavailable
from src.core.rag_config import RAGSettings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.temporal_search import PerYearSearch


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
        indexes_dir=tmp_path / "indexes",
    )


def test_factory_builds_service_and_answers_with_citation(
    tmp_path: Path, fake_client: Any
) -> None:
    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
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
    backend = service.tools.search_backend
    # Temporal fan-out собран фабрикой поверх hybrid retriever; фактологический
    # вопрос с одним годом делегируется в retriever напрямую.
    assert isinstance(backend, PerYearSearch)
    assert isinstance(backend.retriever, HybridRetriever)
    assert backend.per_year_k == 2
    assert backend.max_years == 12
    assert backend.reranker is None
    assert isinstance(service.router, LLMRouter)
    assert service.router.available_years == (2023,)
    assert result.retrieval_diagnostics
    assert result.retrieval_diagnostics[0]["retrieval_mode"] == "hybrid"
    coverage = result.diagnostics["coverage"]
    assert coverage["status"] == "complete"
    # diagnostics["coverage"] — сериализованный CoverageReport из спеки §4.6,
    # а не asdict с потерей производных полей.
    for key in (
        "metrics",
        "missing_metrics",
        "missing_metric_years",
        "distinct_docs",
        "distinct_sections",
        "notes",
    ):
        assert key in coverage
    assert result.diagnostics["verification"]["status"] == "ok"


def test_factory_persists_and_reuses_faiss_index(
    tmp_path: Path, fake_client: Any
) -> None:
    pytest.importorskip("faiss")
    from src.core.faiss_index import FaissDenseIndex

    settings = _settings(tmp_path)
    first = create_agent_service(
        settings=settings,
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
    )
    dense = first.tools.search_backend.retriever.dense_index
    assert isinstance(dense, FaissDenseIndex)
    index_dirs = list((tmp_path / "indexes").iterdir())
    assert len(index_dirs) == 1
    assert (index_dirs[0] / "index.faiss").is_file()

    second = create_agent_service(
        settings=settings,
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
    )
    assert isinstance(second.tools.search_backend.retriever.dense_index, FaissDenseIndex)
    assert len(list((tmp_path / "indexes").iterdir())) == 1


def test_reranker_is_enabled_by_default_and_can_be_disabled(
    tmp_path: Path, fake_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    default_service = create_agent_service(
        settings=settings, client=fake_client, graph_dir=tmp_path / "graph"
    )
    assert default_service.tools.search_backend.reranker is not None

    disabled_service = create_agent_service(
        settings=settings,
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
    )
    assert disabled_service.tools.search_backend.reranker is None

    monkeypatch.setenv("RERANK_ENABLED", "0")
    env_disabled = create_agent_service(
        settings=settings, client=fake_client, graph_dir=tmp_path / "graph"
    )
    assert env_disabled.tools.search_backend.reranker is None


def test_factory_wires_temporal_limits_from_settings(
    tmp_path: Path, fake_client: Any
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"per_year_k": 3, "max_years": 4}
    )
    service = create_agent_service(
        settings=settings,
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
    )

    backend = service.tools.search_backend
    assert isinstance(backend, PerYearSearch)
    assert backend.per_year_k == 3
    assert backend.max_years == 4

    # Factoid с одним годом не разветвляется и отвечает как раньше.
    result = service.ask(
        "Какая чистая прибыль за 2023 год?", conversation_id="temporal-limits"
    )
    assert result.answer.status == "answer"
    assert [event.tool_name for event in result.tool_events] == ["search_reports"]
    assert result.answer.citations[0].chunk_id == "sber_2023:00001:c0"


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
        settings=settings,
        client=fake_client,
        structure_dir=structure_dir,
        graph_dir=tmp_path / "graph",
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
    service = service_factory(
        settings=_settings(tmp_path),
        client=fake_client,
        graph_dir=tmp_path / "graph",
    )

    assert service.ask("вопрос", conversation_id="c2").answer.status == "answer"


def _write_graph_artifact(graph_dir: Path, chunk_id: str) -> Path:
    from src.graph.graph_builder import build_graph, save_graph
    from src.graph.graph_models import EntityRecord

    result = build_graph(
        [_chunk()],
        {},
        entities=[
            EntityRecord(
                "Credit Risk", "Risk", chunk_id, ("credit risk",), "v1"
            )
        ],
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    return save_graph(result, graph_dir / "graph.json")


def test_factory_builds_graph_backend_from_artifact(
    tmp_path: Path, fake_client: Any
) -> None:
    from src.retrieval.graph_hybrid import GraphAugmentedRetriever
    from src.retrieval.hybrid_retriever import SearchFilters

    graph_dir = tmp_path / "graph"
    chunk_id = "sber_2023:00001:c0"
    _write_graph_artifact(graph_dir, chunk_id)

    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=graph_dir,
    )

    assert isinstance(service.tools.graph_search_backend, GraphAugmentedRetriever)
    # Graph-артефакт загружается один раз и питает оба backend'а.
    assert service.tools.entity_backend is not None
    entities = service.tools.invoke(
        "get_entities",
        {
            "entity_names": ["Credit Risk"],
            "filters": SearchFilters(),
            "max_hops": 1,
        },
    )
    assert chunk_id in entities["evidence_chunk_ids"]

    result = service.tools.invoke(
        "search_reports",
        {
            "query": "credit risk",
            "filters": SearchFilters(),
            "retrieval_mode": "hybrid",
            "use_graph": True,
        },
    )
    assert result.trace.available["graph"] is True
    graph_ids = set(service.tools.graph_search_backend.evidence_manifest(result))
    graph_ids.update(
        service.tools.graph_search_backend.graph_evidence_manifest(result)
    )
    assert chunk_id in graph_ids


def test_factory_without_graph_artifact_keeps_graph_optional(
    tmp_path: Path, fake_client: Any
) -> None:
    from src.retrieval.hybrid_retriever import SearchFilters

    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "missing-graph",
    )

    assert service.tools.graph_search_backend is None
    assert service.tools.entity_backend is None
    with pytest.raises(ToolUnavailable):
        service.tools.invoke(
            "get_entities",
            {
                "entity_names": ["Credit Risk"],
                "filters": SearchFilters(),
                "max_hops": 1,
            },
        )

    # use_graph без артефакта — обычный поиск с явным warning, а не падение.
    result = service.tools.invoke(
        "search_reports",
        {
            "query": "чистая прибыль",
            "filters": SearchFilters(),
            "use_graph": True,
        },
    )
    assert "graph_unavailable" in result.warnings


def _registry_record(doc_id: str = "sber_2023", **overrides: Any) -> dict[str, Any]:
    record = {
        "doc_id": doc_id,
        "file_path": f"data/raw/{doc_id}.pdf",
        "source_sha256": "a" * 64,
        "year": 2023,
        "report_period": "2023",
        "issuer": "ПАО Сбербанк",
        "country": "Россия",
        "source_status": "comparable",
        "inclusion_note": "Годовой отчёт ПАО Сбербанк.",
    }
    record.update(overrides)
    return record


def _write_registry(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_factory_enriches_chunks_with_registry_issuer(
    tmp_path: Path, fake_client: Any
) -> None:
    from src.retrieval.hybrid_retriever import SearchFilters

    registry = _write_registry(
        tmp_path / "source_registry.json", [_registry_record()]
    )
    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
        source_registry_path=registry,
    )

    chunk = service.tools.chunk_backend("sber_2023:00001:c0")
    assert chunk["issuer"] == "ПАО Сбербанк"
    assert chunk["country"] == "Россия"
    assert chunk["source_status"] == "comparable"

    # Фильтр по эмитенту больше не отбрасывает весь корпус.
    result = service.tools.invoke(
        "search_reports",
        {
            "query": "чистая прибыль",
            "filters": SearchFilters(issuers={"ПАО Сбербанк"}),
        },
    )
    assert [hit.chunk_id for hit in result.hits] == ["sber_2023:00001:c0"]


def test_factory_without_registry_keeps_chunks_usable(
    tmp_path: Path, fake_client: Any
) -> None:
    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=tmp_path / "graph",
        source_registry_path=tmp_path / "missing_registry.json",
    )

    chunk = service.tools.chunk_backend("sber_2023:00001:c0")
    assert "issuer" not in chunk
    answer = service.ask("Какая чистая прибыль за 2023 год?", conversation_id="c1")
    assert answer.answer.status == "answer"


def _write_manifested_graph(
    graph_dir: Path, chunk_id: str, *, status: str, created_at: str
):
    from src.graph.graph_builder import build_graph, save_graph
    from src.graph.graph_models import EntityRecord

    chunk = dict(_chunk())
    chunk["chunk_id"] = chunk_id
    result = build_graph(
        [chunk],
        {},
        entities=[
            EntityRecord("Credit Risk", "Risk", chunk_id, ("credit risk",), "v1")
        ],
    )
    artifact = save_graph(result, graph_dir / f"{result.graph_fingerprint}.json")
    (graph_dir / f"{result.graph_fingerprint}_manifest.json").write_text(
        json.dumps(
            {
                "graph_fingerprint": result.graph_fingerprint,
                "status": status,
                "created_at": created_at,
                "chunk_count": 1,
                "failed_chunk_ids": [] if status == "complete" else ["missing"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result, artifact


def test_factory_prefers_complete_graph_over_newer_incomplete(
    tmp_path: Path, fake_client: Any
) -> None:
    from src.agent.factory import _load_graph_artifact

    graph_dir = tmp_path / "graph"
    complete, _ = _write_manifested_graph(
        graph_dir,
        "sber_2023:00001:c0",
        status="complete",
        created_at="2026-11-14T10:00:00+00:00",
    )
    _write_manifested_graph(
        graph_dir,
        "sber_2023:00001:c1",
        status="incomplete",
        created_at="2026-11-14T20:00:00+00:00",
    )

    loaded = _load_graph_artifact(graph_dir)
    assert loaded is not None
    assert loaded.graph_fingerprint == complete.graph_fingerprint

    service = create_agent_service(
        settings=_settings(tmp_path),
        client=fake_client,
        use_reranker=False,
        graph_dir=graph_dir,
    )
    assert service.tools.graph_search_backend is not None


def test_load_graph_artifact_ignores_only_incomplete_index(tmp_path: Path) -> None:
    from src.agent.factory import _load_graph_artifact

    graph_dir = tmp_path / "graph"
    _write_manifested_graph(
        graph_dir,
        "sber_2023:00001:c0",
        status="incomplete",
        created_at="2026-11-14T20:00:00+00:00",
    )

    assert _load_graph_artifact(graph_dir) is None


def test_factory_passes_shared_reranker_to_graph_backend(
    tmp_path: Path, fake_client: Any
) -> None:
    graph_dir = tmp_path / "graph"
    _write_graph_artifact(graph_dir, "sber_2023:00001:c0")
    settings = _settings(tmp_path)

    service = create_agent_service(
        settings=settings,
        client=fake_client,
        use_reranker=True,
        graph_dir=graph_dir,
    )

    backend = service.tools.graph_search_backend
    assert backend is not None
    assert backend.reranker is not None
    assert backend.base_candidate_k == settings.rerank_pool
    assert backend.max_candidates == settings.candidates
