"""CRUD исследований, поиск, каскадное удаление и изоляция владельца."""

from __future__ import annotations

import sqlite3

from test_web_api_support import (
    CLIENT_MESSAGE_ID,
    FakeAgentService,
    api_client,
    create_research,
    send_message,
)


def test_research_crud_search_and_rename(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        first = create_research(client, "  Динамика прибыли  ")
        assert first["title"] == "Динамика прибыли"
        assert first["last_status"] == "idle"
        assert first["messages_count"] == 0

        second = create_research(client, "Структура отчёта")
        send_message(
            client,
            second["id"],
            content="Найди чистую прибыль",
            client_message_id=CLIENT_MESSAGE_ID,
        )

        listing = client.get("/api/research").json()
        assert listing["total"] == 2
        assert listing["items"][0]["id"] == second["id"]
        assert listing["items"][0]["last_message_preview"] == "Найди чистую прибыль"
        assert listing["items"][0]["last_status"] in {"running", "completed"}
        assert listing["items"][0]["messages_count"] == 2

        # Поиск по тексту сообщения (case-insensitive, включая кириллицу).
        found = client.get("/api/research", params={"q": "ПРИБЫЛЬ"}).json()
        assert found["total"] == 1
        assert found["items"][0]["id"] == second["id"]

        by_title = client.get("/api/research", params={"q": "структура"}).json()
        assert by_title["total"] == 1
        assert by_title["items"][0]["id"] == second["id"]

        renamed = client.patch(
            f"/api/research/{first['id']}", json={"title": "  Новое имя  "}
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Новое имя"

        invalid = client.patch(
            f"/api/research/{first['id']}", json={"title": "   "}
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"


def test_delete_research_is_cascading(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        send_message(client, research["id"])
        assert client.get(
            f"/api/research/{research['id']}/messages"
        ).json()["items"]

        deleted = client.delete(f"/api/research/{research['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/api/research/{research['id']}").status_code == 404
        assert client.get("/api/research").json()["total"] == 0

        config = client.app.state.config
        connection = sqlite3.connect(config.db_path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
        finally:
            connection.close()


def test_foreign_research_is_404_not_403(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        send_message(client, research["id"])
        owner = _owner_of(client)

        client.cookies.clear()
        client.cookies.set("rag_owner", "another-owner")

        assert client.get(f"/api/research/{research['id']}").status_code == 404
        assert (
            client.patch(
                f"/api/research/{research['id']}", json={"title": "Hijack"}
            ).status_code
            == 404
        )
        assert (
            client.delete(f"/api/research/{research['id']}").status_code == 404
        )
        assert (
            client.get(f"/api/research/{research['id']}/messages").status_code
            == 404
        )
        response = client.post(
            f"/api/research/{research['id']}/messages",
            json={"content": "Чужой вопрос", "client_message_id": CLIENT_MESSAGE_ID},
        )
        assert response.status_code == 404
        # Данные владельца не пострадали.
        client.cookies.clear()
        client.cookies.set("rag_owner", owner)
        assert client.get(f"/api/research/{research['id']}").status_code == 200


def _owner_of(client) -> str:
    for item in client.cookies.jar:
        if item.name == "rag_owner":
            return item.value
    raise AssertionError("owner cookie is missing")
