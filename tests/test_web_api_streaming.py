"""SSE: sequence, Last-Event-ID, heartbeat, terminal и отмена."""

from __future__ import annotations

import asyncio

from src.api.routes.events import _stream_channel  # noqa: PLC2701
from src.api.streaming import EventChannel
from test_web_api_support import (
    AskOnlyService,
    CancellableAgentService,
    EmptyAgentService,
    FakeAgentService,
    MissingResultService,
    api_client,
    create_research,
    make_config,
    read_sse,
    send_message,
    wait_for,
)


def test_sse_sequence_terminal_and_message_payload(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        response, events, text = read_sse(
            client, accepted["events_url"]
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["x-accel-buffering"] == "no"

        agent_events = [item for item in events if item["event"] == "agent"]
        assert [item["id"] for item in agent_events] == [1, 2, 3, 4, 5, 6]
        types = [item["data"]["type"] for item in agent_events]
        assert types == [
            "run_started",
            "analysis_completed",
            "tool_started",
            "tool_completed",
            "answer_started",
            "run_completed",
        ]
        assert all(
            item["data"]["request_id"] == accepted["request_id"]
            for item in agent_events
        )
        assert all(
            item["data"]["sequence"] == item["id"]
            and item["data"]["contract_version"] == "1"
            for item in agent_events
        )
        started = agent_events[0]["data"]
        assert started["message_id"] == accepted["assistant_message_id"]
        assert agent_events[1]["data"]["effective_filters"] == {"years": [2023]}
        assert agent_events[2]["data"]["short_arguments"] == {"top_k": 5}
        assert "секретный" not in text
        completed = agent_events[-1]["data"]
        assert completed["answer_status"] == "answer"
        message = completed["message"]
        assert message["id"] == accepted["assistant_message_id"]
        assert message["status"] == "completed"
        assert message["analysis"]["calculations"]
        assert message["evidence"][0]["chunk_id"] == "sber_2023:00001:c0:aaa"


def test_last_event_id_replays_only_new_events(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        read_sse(client, accepted["events_url"])

        _, replayed, _ = read_sse(
            client,
            accepted["events_url"],
            headers={"Last-Event-ID": "3"},
        )
        agent_events = [item for item in replayed if item["event"] == "agent"]
        assert [item["id"] for item in agent_events] == [4, 5, 6]
        assert agent_events[-1]["data"]["type"] == "run_completed"

        _, nothing, _ = read_sse(
            client,
            accepted["events_url"],
            headers={"Last-Event-ID": "6"},
        )
        assert [item for item in nothing if item["event"] == "agent"] == []


def test_replay_window_expired_returns_resync(tmp_path) -> None:
    config = make_config(tmp_path, replay_buffer_events=2)
    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        read_sse(client, accepted["events_url"])

        _, events, _ = read_sse(
            client,
            accepted["events_url"],
            headers={"Last-Event-ID": "1"},
        )
    assert events
    assert events[0]["event"] == "resync"
    assert events[0]["data"]["reason"] == "replay_window_expired"
    assert events[0]["data"]["type"] == "resync"


def test_oversized_event_returns_resync(tmp_path) -> None:
    config = make_config(tmp_path, max_event_bytes=50)
    with api_client(tmp_path, FakeAgentService(), config=config) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        _, events, _ = read_sse(client, accepted["events_url"])
    assert events
    assert events[0]["event"] == "resync"
    assert events[0]["data"]["reason"] == "event_too_large"


def test_unknown_request_events_return_410(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        response = client.get("/api/requests/unknown/events")
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "request_not_replayable"


def test_missing_channel_returns_resync_for_saved_request(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        read_sse(client, accepted["events_url"])
        client.app.state.hub.shutdown()

        _, events, _ = read_sse(client, accepted["events_url"])
    assert events[0]["event"] == "resync"
    assert events[0]["data"]["reason"] == "request_finished"


def test_heartbeat_is_emitted_while_no_events_arrive() -> None:
    channel = EventChannel("request-heartbeat", max_events=10)

    async def first_frame() -> str:
        generator = _stream_channel(
            channel,
            request_id="request-heartbeat",
            last_event_id=0,
            heartbeat=0.02,
        )
        try:
            return await asyncio.wait_for(anext(generator), timeout=1)
        finally:
            await generator.aclose()

    try:
        assert asyncio.run(first_frame()) == ": keep-alive\n\n"
    finally:
        channel.close()


def test_incomplete_backend_stream_gets_terminal_run_failed(tmp_path) -> None:
    with api_client(tmp_path, EmptyAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        _, events, text = read_sse(client, accepted["events_url"])

        agent_events = [item for item in events if item["event"] == "agent"]
        types = [item["data"]["type"] for item in agent_events]
        assert types[-1] == "run_failed"
        assert agent_events[-1]["data"]["code"] == "internal_error"
        assert "traceback" not in text.casefold()

        state = client.get(f"/api/requests/{accepted['request_id']}").json()
        assert state["status"] == "failed"
        assert state["message"]["status"] == "failed"
        assert state["error"]["code"] == "internal_error"


def test_run_completed_without_result_becomes_safe_terminal_error(tmp_path) -> None:
    with api_client(tmp_path, MissingResultService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        _, events, _ = read_sse(client, accepted["events_url"])
        agent_events = [item for item in events if item["event"] == "agent"]
        assert agent_events[-1]["data"]["type"] == "run_failed"
        state = client.get(f"/api/requests/{accepted['request_id']}").json()
    assert state["status"] == "failed"


def test_ask_only_backend_publishes_started_and_completed(tmp_path) -> None:
    service = AskOnlyService()
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        _, events, _ = read_sse(client, accepted["events_url"])
        agent_events = [item for item in events if item["event"] == "agent"]
        assert [item["data"]["type"] for item in agent_events] == [
            "run_started",
            "run_completed",
        ]
        assert agent_events[-1]["data"]["message"]["status"] == "completed"
        assert len(service.calls) == 1
        assert service.calls[0]["conversation_id"] == research["id"]


def test_cancel_not_supported_does_not_fake_cancellation(tmp_path) -> None:
    service = FakeAgentService(block=True)
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        assert wait_for(service.started.is_set)

        response = client.post(
            f"/api/requests/{accepted['request_id']}/cancel"
        )
        assert response.status_code == 200
        assert response.json() == {
            "request_id": accepted["request_id"],
            "status": "not_supported",
            "cancel_supported": False,
            "code": "cancel_not_supported",
        }

        state = client.get(f"/api/requests/{accepted['request_id']}").json()
        assert state["status"] == "running"
        assert state["message"] is None
        service.release.set()


def test_cancel_supported_persists_cancelled_without_answer(tmp_path) -> None:
    service = CancellableAgentService()
    with api_client(tmp_path, service) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        assert wait_for(service.started.is_set)

        response = client.post(
            f"/api/requests/{accepted['request_id']}/cancel"
        )
        assert response.status_code == 200
        assert response.json() == {
            "request_id": accepted["request_id"],
            "status": "cancelled",
            "cancel_supported": True,
        }
        assert service.cancel_calls == ["backend-1"]

        _, events, _ = read_sse(client, accepted["events_url"])
        agent_events = [item for item in events if item["event"] == "agent"]
        assert agent_events[-1]["data"]["type"] == "run_cancelled"
        assert "run_completed" not in [
            item["data"]["type"] for item in agent_events
        ]

        state = client.get(f"/api/requests/{accepted['request_id']}").json()
        assert state["status"] == "cancelled"
        assert state["message"]["status"] == "cancelled"
        assert state["message"]["content"] == ""
        assert state["message"]["analysis"] is None
        assert state["message"]["evidence"] == []
        assert state["error"]["code"] == "cancelled"

        again = client.post(
            f"/api/requests/{accepted['request_id']}/cancel"
        )
        assert again.json()["status"] == "already_finished"


def test_cancel_after_completion_is_noop(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        read_sse(client, accepted["events_url"])
        response = client.post(
            f"/api/requests/{accepted['request_id']}/cancel"
        )
    assert response.status_code == 200
    assert response.json()["status"] == "already_finished"


def test_foreign_request_is_not_visible(tmp_path) -> None:
    with api_client(tmp_path, FakeAgentService()) as client:
        research = create_research(client)
        accepted = send_message(client, research["id"])
        client.cookies.clear()
        client.cookies.set("rag_owner", "another-owner")

        status_response = client.get(
            f"/api/requests/{accepted['request_id']}"
        )
        assert status_response.status_code == 404
        assert status_response.json()["error"]["code"] == "request_not_found"

        events_response = client.get(
            f"/api/requests/{accepted['request_id']}/events"
        )
        assert events_response.status_code == 410
        assert (
            events_response.json()["error"]["code"]
            == "request_not_replayable"
        )

        cancel_response = client.post(
            f"/api/requests/{accepted['request_id']}/cancel"
        )
        assert cancel_response.status_code == 404
