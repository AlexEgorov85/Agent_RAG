"""Process-wide ``ExecutionGate`` и приоритет chat > evaluation.

Внутри API одновременно работает один ``AgentService``. Все вызовы
``ask/stream`` проходят через gate, а между процессами — ещё и через
``LLMExecutionLock`` (``docs/specs/evaluation-webui.md`` §9.2).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from src.core.llm_lock import LLMExecutionLock

Priority = Literal["chat", "evaluation"]


class PriorityGate:
    """Один активныйturn; chat имеет приоритет над evaluation."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._chat_waiting = 0

    @contextmanager
    def hold(self, priority: Priority) -> Iterator[None]:
        with self._condition:
            if priority == "chat":
                self._chat_waiting += 1
            try:
                while self._active or (
                    priority == "evaluation" and self._chat_waiting > 0
                ):
                    self._condition.wait()
                self._active = True
            finally:
                if priority == "chat":
                    self._chat_waiting -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


class ExecutionGate:
    """Прокси ``AgentService``: gate + межпроцессный LLM lock на один turn."""

    def __init__(
        self,
        provider: Callable[[], Any],
        *,
        priority: Priority = "chat",
        gate: PriorityGate | None = None,
        lock_path: Path | str | None = None,
        lock_timeout_s: float | None = None,
    ) -> None:
        self._provider = provider
        self.priority = priority
        self._gate = gate or PriorityGate()
        self._lock_path = Path(lock_path) if lock_path else None
        self._lock_timeout_s = lock_timeout_s

    def _lock(self) -> LLMExecutionLock:
        options: dict[str, Any] = {
            "kind": self.priority,
            "timeout_s": self._lock_timeout_s,
        }
        if self._lock_path is not None:
            options["path"] = self._lock_path
        return LLMExecutionLock(**options)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._gate.hold(self.priority):
            with self._lock().held(wait=True):
                return getattr(self._provider(), method)(*args, **kwargs)

    def ask(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("ask", *args, **kwargs)

    @property
    def supports_stream(self) -> bool:
        """Прокси не должен выдавать себя за backend с ``stream``."""
        return callable(getattr(self._provider(), "stream", None))

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Держит gate/lock всё время итерации, а не только до первого yield."""
        iterator = None

        def generate() -> Iterator[Any]:
            nonlocal iterator
            with self._gate.hold(self.priority):
                with self._lock().held(wait=True):
                    iterator = getattr(self._provider(), "stream")(*args, **kwargs)
                    yield from iterator

        return generate()

    def __getattr__(self, name: str) -> Any:
        # Ленивое делегирование: gate не должен менять публичный контракт.
        return getattr(self._provider(), name)
