"""Тесты межпроцессного ``LLMExecutionLock`` (без реальных LLM)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from src.core.llm_lock import LLMBusy, LLMExecutionLock


def _lock(path: Path, **kwargs) -> LLMExecutionLock:
    options = {
        "timeout_s": 0.5,
        "stale_s": 60.0,
        "poll_interval_s": 0.01,
    }
    options.update(kwargs)
    return LLMExecutionLock(path=path, **options)


def test_lock_is_exclusive_and_no_wait_raises(tmp_path: Path) -> None:
    path = tmp_path / "llm.lock"
    first = _lock(path)
    second = _lock(path)

    first.acquire()
    try:
        assert path.exists()
        with pytest.raises(LLMBusy):
            second.acquire(wait=False)
    finally:
        first.release()

    second.acquire(wait=False)
    second.release()
    assert not path.exists()


def test_stale_lock_with_dead_pid_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "llm.lock"
    path.write_text(
        json.dumps(
            {
                "pid": 0,
                "token": "dead",
                "kind": "cli",
                "acquired_at": "2000-01-01T00:00:00Z",
                "heartbeat_at": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    lock = _lock(path)
    warnings = lock.acquire(wait=False)
    try:
        assert "stale_llm_lock_recovered" in warnings
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
    finally:
        lock.release()


def test_heartbeat_updates_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "llm.lock"
    lock = _lock(path)
    lock.acquire()
    try:
        before = json.loads(path.read_text(encoding="utf-8"))["heartbeat_at"]
        time.sleep(0.01)
        lock.heartbeat()
        after = json.loads(path.read_text(encoding="utf-8"))["heartbeat_at"]
        assert after >= before
    finally:
        lock.release()


class _Service:
    def ask(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "ok"


def test_runner_no_wait_turns_busy_lock_into_llm_busy(tmp_path: Path) -> None:
    from eval.runner import _ask_with_timeout

    path = tmp_path / "llm.lock"
    holder = _lock(path)
    holder.acquire()
    try:
        with pytest.raises(LLMBusy):
            _ask_with_timeout(
                _Service(),
                "question",
                conversation_id="c1",
                filters=None,
                timeout=None,
                llm_lock=_lock(path),
                wait_for_llm=False,
            )
    finally:
        holder.release()


def test_runner_cli_returns_code_5_when_lock_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eval.runner as runner

    def busy(**_kwargs):
        raise LLMBusy("lock busy")

    monkeypatch.setattr(runner, "run_suite", busy)
    assert runner.main(["--suite", "smoke"]) == 5


class _SlowService:
    def ask(self, *args, **kwargs):  # noqa: ANN002, ANN003
        time.sleep(5)
        return "late"


def test_timed_out_turn_releases_lock(tmp_path: Path) -> None:
    from eval.runner import TurnTimeout, _ask_with_timeout

    path = tmp_path / "llm.lock"
    lock = _lock(path, timeout_s=0.2)

    with pytest.raises(TurnTimeout):
        _ask_with_timeout(
            _SlowService(),
            "question",
            conversation_id="c1",
            filters=None,
            timeout=0.05,
            llm_lock=lock,
        )

    assert not path.exists()
    # Следующий turn не ждёт abandoned-поток.
    result = _ask_with_timeout(
        _Service(),
        "question",
        conversation_id="c2",
        filters=None,
        timeout=1.0,
        llm_lock=_lock(path, timeout_s=0.2),
    )
    assert result == "ok"
    assert not path.exists()
