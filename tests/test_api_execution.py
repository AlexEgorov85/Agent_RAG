"""ExecutionGate: один активный LLM-turn и освобождение lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from src.api.execution import ExecutionGate, PriorityGate


class RecordingService:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = threading.Lock()

    def ask(self, *args: Any, **kwargs: Any) -> str:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return "ok"


def test_gate_serializes_calls_and_releases_lock(tmp_path: Path) -> None:
    service = RecordingService()
    lock_path = tmp_path / "llm.lock"
    gate = ExecutionGate(
        lambda: service, priority="chat", gate=PriorityGate(), lock_path=lock_path
    )

    threads = [
        threading.Thread(target=lambda: gate.ask("question"))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert service.calls == 3
    assert service.max_active == 1
    assert not lock_path.exists()


def test_gate_supports_stream_reflects_backend(tmp_path: Path) -> None:
    service = RecordingService()
    gate = ExecutionGate(lambda: service, lock_path=tmp_path / "llm.lock")
    assert gate.supports_stream is False

    class Streaming(RecordingService):
        def stream(self, *args: Any, **kwargs: Any):
            yield "event"

    gate = ExecutionGate(lambda: Streaming(), lock_path=tmp_path / "llm.lock")
    assert gate.supports_stream is True
    assert list(gate.stream("question")) == ["event"]
    assert not (tmp_path / "llm.lock").exists()
