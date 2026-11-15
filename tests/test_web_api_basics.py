"""Contract-тесты служебных endpoint'ов и отсутствия секретов."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.errors import BackendUnavailable
from src.api.services import Services
from test_web_api_support import (
    FakeAgentService,
    api_client,
    default_corpus,
    make_config,
    make_services,
)


def test_import_and_create_do_not_touch_database_until_lifespan(tmp_path) -> None:
    config = make_config(tmp_path)
    app = create_app(config=config, services=make_services(config, FakeAgentService()))
    assert not config.db_path.exists()

    with TestClient(app) as client:
        assert config.db_path.exists()
        response = client.get("/api/health")
        assert response.status_code == 200
        cookie = response.headers.get("set-cookie", "")
        assert "rag_owner=" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie


def test_requests_work_without_explicit_lifespan_context(tmp_path) -> None:
    config = make_config(tmp_path)
    app = create_app(config=config, services=make_services(config, FakeAgentService()))
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert config.db_path.exists()


def test_health_capabilities_and_corpus(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "contract_version": "1",
            "database": "ok",
            "corpus_available": True,
            "agent_available": True,
        }

        capabilities = client.get("/api/capabilities").json()
        assert capabilities["contract_version"] == "1"
        assert capabilities["token_streaming"] is False
        assert capabilities["incremental_evidence"] is False
        assert capabilities["coverage"] is True
        assert capabilities["calculations"] is True
        assert capabilities["structure_comparison"] is True
        assert capabilities["structure_content_comparison"] is True
        assert capabilities["graph_panel"] is False
        assert capabilities["pdf_page_links"] is False
        assert capabilities["evaluation_ui"] is False
        assert capabilities["evaluation_sse"] is False
        assert capabilities["evaluation_cancel"] is False
        assert capabilities["retrieval_modes"] == ["hybrid", "dense", "keyword"]
        assert capabilities["judge"] is False

        corpus = client.get("/api/corpus").json()
        assert corpus == default_corpus().to_public()


def test_health_returns_503_without_leaking_backend_error(tmp_path) -> None:
    class FailingProvider:
        def get(self):
            raise BackendUnavailable()

        def available(self) -> bool:
            return False

    config = make_config(tmp_path)
    services = Services(config, agent_provider=FailingProvider())
    app = create_app(config=config, services=services)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["agent_available"] is False
    text = response.text.casefold()
    assert "traceback" not in text
    assert "secret" not in text
    assert "api_key" not in text


def test_errors_are_unified_and_do_not_echo_payload(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        missing = client.get("/api/research/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"

        invalid = client.post(
            "/api/research",
            json={"title": "x", "private_token": "secret-value"},
        )
        # Неизвестные поля игнорируются; в ответе нет присланного секрета.
        assert invalid.status_code == 201
        assert "secret-value" not in invalid.text

        bad_route = client.get("/api/evaluation/runs")
        assert bad_route.status_code == 404
        # Раздел Evaluation отключён конфигурацией: честный код, не пустой 404.
        assert bad_route.json()["error"]["code"] == "evaluation_disabled"
        assert "traceback" not in bad_route.text.casefold()


def test_validation_errors_use_contract_shape(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = client.post("/api/research", json={"title": "Test"}).json()
        response = client.post(
            f"/api/research/{research['id']}/messages",
            json={
                "content": "Вопрос",
                "client_message_id": "not-a-uuid",
                "filters": {"years": ["2023"]},
            },
        )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "traceback" not in response.text.casefold()
    assert all(set(field) == {"loc", "msg", "type"} for field in error["details"]["fields"])


def test_foreign_origin_post_is_rejected(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        rejected = client.post(
            "/api/research",
            json={"title": "Test"},
            headers={"Origin": "http://evil.example"},
        )
        assert rejected.status_code == 403
        assert rejected.json()["error"]["code"] == "csrf_rejected"

        allowed = client.post(
            "/api/research",
            json={"title": "Test"},
            headers={"Origin": "http://localhost:5173"},
        )
        assert allowed.status_code == 201
        assert (
            allowed.headers.get("access-control-allow-origin")
            == "http://localhost:5173"
        )


def test_spa_is_not_served_by_default(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        assert client.get("/").status_code == 404


def test_spa_is_served_from_dist_and_falls_back_to_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><div id="root">SPA shell</div>', encoding="utf-8"
    )
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    config = make_config(tmp_path, serve_spa=True, spa_dist_dir=dist)

    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "SPA shell" in root.text
        assert root.headers["content-type"].startswith("text/html")

        # Клиентский маршрут отдаёт index.html, а не 404.
        evaluation = client.get("/evaluation")
        assert evaluation.status_code == 200
        assert "SPA shell" in evaluation.text

        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "console.log" in asset.text

        # API не перекрывается SPA fallback.
        health = client.get("/api/health")
        assert health.status_code == 200
        unknown = client.get("/api/does-not-exist")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "not_found"
        assert "SPA shell" not in unknown.text


def test_spa_flag_without_dist_keeps_api_only(tmp_path) -> None:
    config = make_config(
        tmp_path, serve_spa=True, spa_dist_dir=tmp_path / "missing-dist"
    )
    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        assert client.get("/").status_code == 404
        assert client.get("/api/health").status_code == 200
