"""Evidence endpoint: метаданные, таблица, 404 и отсутствие path traversal."""

from __future__ import annotations

import json

from test_web_api_support import (
    FakeAgentService,
    FakeChunkBackend,
    api_client,
    create_research,
    make_config,
    read_sse,
    send_message,
)

TEXT_CHUNK = {
    "chunk_id": "sber_2023:00001:c0:aaa",
    "doc_id": "sber_2023",
    "title": "Годовой отчёт 2023",
    "year": 2023,
    "page_pdf": 12,
    "page_label": "12",
    "kind": "text",
    "text": "Полный текст фрагмента.",
    "html": None,
    "section_path": "Стратегия/Ключевые показатели",
    "breadcrumbs": ["Стратегия", "Ключевые показатели"],
}
TABLE_CHUNK = {
    "chunk_id": "sber_2023:00002:c0:bbb",
    "doc_id": "sber_2023",
    "title": "Годовой отчёт 2023",
    "year": 2023,
    "page_pdf": 20,
    "page_label": "20",
    "kind": "table",
    "text": "Выручка | 100",
    "html": "<table><tr><td>Выручка</td><td>100</td></tr></table>",
    "section_path": "Финансовые результаты",
    "breadcrumbs": ["Финансовые результаты"],
}


def _service() -> FakeAgentService:
    backend = FakeChunkBackend(
        {
            TEXT_CHUNK["chunk_id"]: TEXT_CHUNK,
            TABLE_CHUNK["chunk_id"]: TABLE_CHUNK,
        }
    )
    return FakeAgentService(chunk_backend=backend)


def test_evidence_returns_text_metadata_and_cache_header(tmp_path) -> None:
    with api_client(tmp_path, _service()) as client:
        response = client.get(f"/api/evidence/{TEXT_CHUNK['chunk_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "chunk_id": TEXT_CHUNK["chunk_id"],
        "document_id": "sber_2023",
        "document_title": "Годовой отчёт 2023",
        "issuer": None,
        "year": 2023,
        "section_path": ["Стратегия", "Ключевые показатели"],
        "page_start": 12,
        "page_end": None,
        "page_label": "12",
        "source_type": "text",
        "text": "Полный текст фрагмента.",
        "html": None,
    }
    assert response.headers["cache-control"] == "private, max-age=300"


def test_evidence_returns_table_html(tmp_path) -> None:
    with api_client(tmp_path, _service()) as client:
        body = client.get(f"/api/evidence/{TABLE_CHUNK['chunk_id']}").json()
    assert body["source_type"] == "table"
    assert body["html"] == TABLE_CHUNK["html"]
    assert body["text"] == TABLE_CHUNK["text"]


def test_evidence_unknown_id_is_404(tmp_path) -> None:
    with api_client(tmp_path, _service()) as client:
        response = client.get("/api/evidence/unknown-chunk")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_evidence_issuer_comes_from_source_registry(tmp_path) -> None:
    registry_path = tmp_path / "source_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "doc_id": "sber_2023",
                        "year": 2023,
                        "issuer": "ПАО Сбербанк",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = make_config(tmp_path, source_registry_path=registry_path)
    with api_client(tmp_path, _service(), config=config) as client:
        body = client.get(f"/api/evidence/{TEXT_CHUNK['chunk_id']}").json()
    assert body["issuer"] == "ПАО Сбербанк"


def test_evidence_rejects_backend_id_mismatch(tmp_path) -> None:
    class WrongBackend:
        def get_chunk(self, chunk_id):
            return {**TEXT_CHUNK, "chunk_id": "another-chunk"}

    service = FakeAgentService(chunk_backend=WrongBackend())
    with api_client(tmp_path, service) as client:
        response = client.get("/api/evidence/requested-chunk")
    assert response.status_code == 404


def test_evidence_does_not_allow_path_traversal(tmp_path) -> None:
    service = _service()
    with api_client(tmp_path, service) as client:
        response = client.get("/api/evidence/..%2F..%2Fsecret.txt")
    assert response.status_code == 404
    assert "secret.txt" not in response.text


def test_citation_becomes_evidence_item_with_full_text_flag(tmp_path) -> None:
    service = _service()
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        response, events, _ = read_sse(client, accepted["events_url"])
        assert response.status_code == 200
        completed = [
            item
            for item in events
            if (item.get("data") or {}).get("type") == "run_completed"
        ][-1]
        item = completed["data"]["message"]["evidence"][0]
    assert item["document_title"] == "Годовой отчёт 2023"
    assert item["source_type"] == "text"
    assert item["has_full_text"] is True
    assert item["section_title"] == "Ключевые показатели"
    assert item["section_path"] == ["Стратегия", "Ключевые показатели"]
