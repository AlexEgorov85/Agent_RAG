"""SQLite-хранилище: схема, WAL, каскад, идемпотентность и interrupted."""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.api.store import (
    ActiveRequestError,
    ResearchNotFoundError,
    Store,
)


def _store(tmp_path) -> Store:
    store = Store(tmp_path / "api.sqlite3")
    store.init()
    return store


def _start(store: Store, owner: str = "owner-a", research_id: str = "research-1"):
    store.create_research(owner, title="Research", research_id=research_id)
    return store.start_turn(
        owner_id=owner,
        research_id=research_id,
        user_message_id="user-1",
        assistant_message_id="assistant-1",
        request_id="request-1",
        content="Вопрос",
        client_message_id="client-1",
        filters_json=json.dumps({"years": [2023]}),
    )


def test_schema_wal_and_foreign_keys(tmp_path) -> None:
    store = _store(tmp_path)
    connection = sqlite3.connect(store.path)
    try:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        connection.close()
    assert mode.casefold() == "wal"
    assert {"researches", "messages", "requests"} <= tables
    assert "ux_messages_client" in indexes

    connection = store._connect()
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_start_turn_is_idempotent_and_blocks_second_active_request(tmp_path) -> None:
    store = _store(tmp_path)
    outcome = _start(store)
    assert outcome["duplicate"] is False
    assert outcome["assistant_message"]["filters_json"] == '{"years": [2023]}'

    duplicate = store.start_turn(
        owner_id="owner-a",
        research_id="research-1",
        user_message_id="user-2",
        assistant_message_id="assistant-2",
        request_id="request-2",
        content="Повтор",
        client_message_id="client-1",
        filters_json=None,
    )
    assert duplicate["duplicate"] is True
    assert duplicate["request"]["id"] == "request-1"

    with pytest.raises(ActiveRequestError) as error:
        store.start_turn(
            owner_id="owner-a",
            research_id="research-1",
            user_message_id="user-3",
            assistant_message_id="assistant-3",
            request_id="request-3",
            content="Второй",
            client_message_id="client-2",
            filters_json=None,
        )
    assert error.value.active_request_id == "request-1"


def test_foreign_research_cannot_be_used(tmp_path) -> None:
    store = _store(tmp_path)
    _start(store)
    with pytest.raises(ResearchNotFoundError):
        store.start_turn(
            owner_id="owner-b",
            research_id="research-1",
            user_message_id="user-2",
            assistant_message_id="assistant-2",
            request_id="request-2",
            content="Чужой",
            client_message_id="client-2",
            filters_json=None,
        )
    assert store.get_research("owner-b", "research-1") is None


def test_delete_is_cascading(tmp_path) -> None:
    store = _store(tmp_path)
    _start(store)
    assert store.delete_research("owner-a", "research-1") is True
    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    finally:
        connection.close()


def test_recover_interrupted_marks_running_failed(tmp_path) -> None:
    path = tmp_path / "api.sqlite3"
    store = Store(path)
    store.init()
    _start(store)
    assert store.request_status("request-1") == "queued"

    restarted = Store(path)
    restarted.init()

    request = restarted.get_request_internal("request-1")
    assert request["status"] == "failed"
    assert request["error_code"] == "interrupted"
    message = restarted.get_message_internal("assistant-1")
    assert message["status"] == "failed"
    error = json.loads(message["error_json"])
    assert error["code"] == "interrupted"
    research = restarted.get_research("owner-a", "research-1")
    assert research["last_status"] == "failed"


def test_cancel_request_clears_answer_and_is_terminal(tmp_path) -> None:
    store = _store(tmp_path)
    _start(store)
    store.update_message(
        "assistant-1",
        content="partial",
        status="streaming",
        answer_json='{"evidence": []}',
    )
    assert store.cancel_request("request-1") is True
    request = store.get_request_internal("request-1")
    assert request["status"] == "cancelled"
    message = store.get_message_internal("assistant-1")
    assert message["status"] == "cancelled"
    assert message["content"] == ""
    assert message["answer_json"] is None
    assert store.cancel_request("request-1") is False


def test_list_researches_is_owner_scoped_and_sorted(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_research("owner-a", title="First", research_id="r-1")
    store.create_research("owner-a", title="Second", research_id="r-2")
    store.create_research("owner-b", title="Foreign", research_id="r-3")
    items, total = store.list_researches("owner-a", limit=50)
    assert total == 2
    assert {item["id"] for item in items} == {"r-1", "r-2"}
    assert store.list_researches("owner-a", query="second")[1] == 1
    assert store.list_researches("owner-a", query="FOREIGN")[1] == 0
