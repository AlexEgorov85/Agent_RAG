"""Тесты PDF-endpoint'а ``GET /api/documents/{doc_id}/file`` (§7.2).

Проверяются: отдача PDF с корректными заголовками, 404 для неизвестного
документа и отсутствующего файла, защита от path traversal, Range/ETag и
честный capability-флаг ``pdf_page_links``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from test_web_api_support import FakeAgentService, api_client, make_config

PDF_BYTES = (
    b"%PDF-1.7\n"
    b"% test fixture, not a real report\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
)

SECRET_BYTES = b"not a pdf, must never be served"


def make_source_entry(
    *,
    doc_id: str = "sber_2023",
    file_path: str | Path,
    sha256: str | None = None,
    year: int = 2023,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "file_path": str(file_path),
        "source_sha256": sha256 or hashlib.sha256(PDF_BYTES).hexdigest(),
        "year": year,
        "report_period": str(year),
        "issuer": "ПАО Сбербанк",
        "country": "Россия",
        "source_status": "comparable",
        "inclusion_note": "Включён как годовой отчёт ПАО Сбербанк.",
    }


def write_registry(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> Path:
    path = tmp_path / "source_registry.json"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "sources": entries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def make_pdf(tmp_path: Path, name: str = "sber_2023.pdf") -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / name
    pdf_path.write_bytes(PDF_BYTES)
    return pdf_path


def config_for(tmp_path: Path, registry_path: Path):
    return make_config(tmp_path, source_registry_path=registry_path)


def test_document_file_returns_pdf_with_inline_disposition(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get("/api/documents/sber_2023/file")

    assert response.status_code == 200
    assert response.content == PDF_BYTES
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline;")
    assert 'filename="sber_2023.pdf"' in response.headers["content-disposition"]
    cache_control = response.headers["cache-control"]
    assert "private" in cache_control
    assert "max-age=" in cache_control
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["etag"]


def test_document_file_supports_range_and_conditional_get(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        ranged = client.get(
            "/api/documents/sber_2023/file",
            headers={"Range": "bytes=0-3"},
        )
        assert ranged.status_code == 206
        assert ranged.content == PDF_BYTES[:4]
        assert ranged.headers["content-range"] == f"bytes 0-3/{len(PDF_BYTES)}"

        full = client.get("/api/documents/sber_2023/file")
        etag = full.headers["etag"]
        cached = client.get(
            "/api/documents/sber_2023/file",
            headers={"If-None-Match": etag},
        )
        assert cached.status_code == 304
        assert cached.headers["etag"] == etag
        assert cached.content == b""


def test_unknown_doc_id_returns_not_found(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get("/api/documents/sber_2099/file")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "sber_2099" not in response.text
    assert str(tmp_path) not in response.text


def test_missing_file_returns_not_found(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    missing = raw_dir / "deleted.pdf"
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=missing)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get("/api/documents/sber_2023/file")
        capabilities = client.get("/api/capabilities").json()

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "deleted.pdf" not in response.text
    assert capabilities["pdf_page_links"] is False


def test_path_outside_raw_directory_is_rejected(tmp_path) -> None:
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(SECRET_BYTES)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=secret)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get("/api/documents/sber_2023/file")
        capabilities = client.get("/api/capabilities").json()

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert SECRET_BYTES not in response.content
    assert "secret.pdf" not in response.text
    assert capabilities["pdf_page_links"] is False


def test_relative_traversal_in_registry_is_rejected(tmp_path) -> None:
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(SECRET_BYTES)
    registry = write_registry(
        tmp_path,
        [make_source_entry(file_path="raw/../../secret.pdf")],
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get("/api/documents/sber_2023/file")

    assert response.status_code == 404
    assert SECRET_BYTES not in response.content
    assert "secret.pdf" not in response.text


def test_query_parameters_cannot_override_file_path(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(SECRET_BYTES)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        response = client.get(
            "/api/documents/sber_2023/file",
            params={"path": str(secret), "file": "../../secret.pdf"},
        )

    assert response.status_code == 200
    assert response.content == PDF_BYTES
    assert SECRET_BYTES not in response.content


def test_traversal_doc_id_is_not_found(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        for doc_id in ("..", "..%2F..%2Fsecret", "sber_2023%2F..%2F.."):
            response = client.get(f"/api/documents/{doc_id}/file")
            assert response.status_code == 404, doc_id
            assert response.json()["error"]["code"] == "not_found"


def test_capability_enabled_only_for_valid_registry(tmp_path) -> None:
    pdf_path = make_pdf(tmp_path)
    registry = write_registry(
        tmp_path, [make_source_entry(file_path=pdf_path)]
    )

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        capabilities = client.get("/api/capabilities").json()

    assert capabilities["pdf_page_links"] is True
    # Существующие флаги не ломаются.
    assert capabilities["coverage"] is True
    assert capabilities["calculations"] is True
    assert capabilities["graph_panel"] is False
    assert capabilities["token_streaming"] is False


def test_capability_disabled_without_registry_file(tmp_path) -> None:
    with api_client(
        tmp_path,
        FakeAgentService(),
        config=make_config(tmp_path),
    ) as client:
        capabilities = client.get("/api/capabilities").json()

    assert capabilities["pdf_page_links"] is False


def test_capability_disabled_for_corrupted_registry(tmp_path) -> None:
    registry = tmp_path / "source_registry.json"
    registry.write_text("{not json", encoding="utf-8")

    with api_client(
        tmp_path,
        FakeAgentService(),
        config=config_for(tmp_path, registry),
    ) as client:
        capabilities = client.get("/api/capabilities").json()
        response = client.get("/api/documents/sber_2023/file")

    assert capabilities["pdf_page_links"] is False
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
