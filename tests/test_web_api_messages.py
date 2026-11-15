"""Сообщения: идемпотентность, 409, фильтры, история и лимиты."""

from __future__ import annotations

from test_web_api_support import (
    CLIENT_MESSAGE_ID,
    FakeAgentService,
    api_client,
    create_research,
    make_config,
    send_message,
    wait_for,
)


def test_client_message_id_is_idempotent(tmp_path) -> None:
    service = FakeAgentService()
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        first = send_message(client, research["id"])
        second = send_message(client, research["id"])

        assert first["request_id"] == second["request_id"]
        assert first["user_message"]["id"] == second["user_message"]["id"]
        assert len(service.calls) == 1

        assert wait_for(
            lambda: client.get(f"/api/research/{research['id']}").json()[
                "last_status"
            ]
            == "completed"
        )
        messages = client.get(
            f"/api/research/{research['id']}/messages"
        ).json()["items"]
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[1]["status"] == "completed"


def test_second_active_request_returns_409(tmp_path) -> None:
    service = FakeAgentService(block=True)
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        first = send_message(client, research["id"])
        assert wait_for(service.started.is_set)

        response = client.post(
            f"/api/research/{research['id']}/messages",
            json={
                "content": "Второй вопрос",
                "client_message_id": "11111111-1111-4111-8111-111111111111",
            },
        )
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "request_in_progress"
        assert error["retryable"] is True
        assert error["details"]["active_request_id"] == first["request_id"]
        service.release.set()


def test_filters_are_passed_and_empty_values_are_omitted(tmp_path) -> None:
    service = FakeAgentService()
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        send_message(
            client,
            research["id"],
            filters={
                "years": [2023, 2023],
                "issuers": ["ПАО Сбербанк"],
                "allowed_doc_ids": ["sber_2023"],
                "kinds": ["text"],
            },
        )
        assert wait_for(lambda: len(service.calls) >= 1)
        filters = service.calls[0]["filters"]
        assert filters.years == {2023}
        assert filters.issuers == {"ПАО Сбербанк"}
        assert filters.allowed_doc_ids == {"sber_2023"}
        assert filters.kinds == {"text"}

        second = create_research(client, "Пустые фильтры")
        send_message(
            client,
            second["id"],
            client_message_id="22222222-2222-4222-8222-222222222222",
            filters={"years": [], "issuers": [], "allowed_doc_ids": []},
        )
        assert wait_for(lambda: len(service.calls) >= 2)
        assert service.calls[1]["filters"] is None


def test_invalid_filters_and_content_are_validation_errors(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)

        bad_type = client.post(
            f"/api/research/{research['id']}/messages",
            json={
                "content": "Вопрос",
                "client_message_id": CLIENT_MESSAGE_ID,
                "filters": {"years": [True]},
            },
        )
        assert bad_type.status_code == 422
        assert bad_type.json()["error"]["code"] == "validation_error"

        blank = client.post(
            f"/api/research/{research['id']}/messages",
            json={"content": "   ", "client_message_id": CLIENT_MESSAGE_ID},
        )
        assert blank.status_code == 422

        missing_id = client.post(
            f"/api/research/{research['id']}/messages",
            json={"content": "Вопрос"},
        )
        assert missing_id.status_code == 422


def test_message_size_limit_returns_413(tmp_path) -> None:
    config = make_config(tmp_path, max_message_chars=10)
    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        research = create_research(client)
        response = client.post(
            f"/api/research/{research['id']}/messages",
            json={
                "content": "x" * 11,
                "client_message_id": CLIENT_MESSAGE_ID,
            },
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_rate_limit_is_configurable(tmp_path) -> None:
    config = make_config(tmp_path, rate_limit_per_minute=1)
    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        research = create_research(client)
        send_message(client, research["id"])
        response = client.post(
            f"/api/research/{research['id']}/messages",
            json={
                "content": "Второй вопрос",
                "client_message_id": "33333333-3333-4333-8333-333333333333",
            },
        )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"


def test_user_message_is_saved_before_agent_finishes(tmp_path) -> None:
    service = FakeAgentService(block=True)
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        send_message(client, research["id"])
        assert wait_for(service.started.is_set)

        items = client.get(
            f"/api/research/{research['id']}/messages"
        ).json()["items"]
        assert items[0]["role"] == "user"
        assert items[0]["status"] == "completed"
        assert items[1]["role"] == "assistant"
        assert items[1]["status"] == "streaming"
        assert items[1]["content"] == ""
        service.release.set()


def test_messages_paginate_without_gaps(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        send_message(client, research["id"])
        wait_for(
            lambda: client.get(f"/api/research/{research['id']}").json()[
                "last_status"
            ]
            in {"completed", "abstained", "failed"}
        )
        page = client.get(
            f"/api/research/{research['id']}/messages", params={"limit": 1}
        ).json()
        assert len(page["items"]) == 1
        assert page["next_before"] == page["items"][0]["id"]

        earlier = client.get(
            f"/api/research/{research['id']}/messages",
            params={"limit": 1, "before": page["next_before"]},
        ).json()
        assert len(earlier["items"]) == 1
        assert earlier["items"][0]["role"] == "user"
        assert earlier["items"][0]["id"] != page["items"][0]["id"]
