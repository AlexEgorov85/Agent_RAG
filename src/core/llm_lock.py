"""Межпроцессная сериализация LLM: ``LLMExecutionLock``.

Локальный контур не должен одновременно вызывать LLM из чата, API-worker'а
evaluation и CLI benchmark. Внутрипроцессный ``ExecutionGate`` этого не решает,
поэтому используется эксклюзивный файловый lock (``spec evaluation-webui.md``
§9.2, ``spec_benchmark.md`` §7.5).

Lock-файл — атомарно создаваемый JSON без prompt/секретов:

```json
{"pid": 123, "token": "...", "kind": "cli", "run_id": "...", "test_id": "...",
 "acquired_at": "...", "heartbeat_at": "..."}
```

Если процесс-владелец умер или heartbeat устарел, lock считается брошенным и
снимается с warning ``stale_llm_lock_recovered``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LOCK_PATH = Path("data/evaluation/llm.lock")
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_STALE_S = 900.0
HEARTBEAT_MIN_S = 5.0


class LLMBusy(RuntimeError):
    """Lock занят, а ожидание запрещено/истекло. CLI-код 5."""

    code = "llm_busy"


class LLMLockUnavailable(RuntimeError):
    """Файловая система не поддерживает эксклюзивный lock."""

    code = "llm_lock_unavailable"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = [
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_uint32,
            ]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                # ERROR_ACCESS_DENIED — процесс существует, но недоступен.
                return ctypes.get_last_error() == 5
            kernel32.CloseHandle(handle)
            return True
        except Exception:  # noqa: BLE001 — проверка не должна ломать lock
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def lock_path_from_env() -> Path:
    """Путь lock-файла из env."""
    raw = os.getenv("LLM_LOCK_PATH")
    return Path(raw) if raw and raw.strip() else DEFAULT_LOCK_PATH


@dataclass
class LLMExecutionLock:
    """Эксклюзивный файловый lock на один LLM-turn.

    ``wait=False`` превращает занятый lock в ``LLMBusy`` (CLI/API код
    ``llm_busy``) вместо ожидания. ``timeout_s`` и ``stale_s`` читаются из env,
    если не переданы явно.
    """

    path: Path = field(default_factory=lock_path_from_env)
    kind: str = "cli"
    run_id: str | None = None
    test_id: str | None = None
    timeout_s: float | None = None
    stale_s: float | None = None
    poll_interval_s: float = 0.2
    warnings: list[str] = field(default_factory=list)
    _held: bool = field(default=False, init=False, repr=False)
    _stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _heartbeat: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _token: str = field(
        default_factory=lambda: uuid.uuid4().hex, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.timeout_s = (
            float(self.timeout_s)
            if self.timeout_s is not None
            else _env_float("LLM_LOCK_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        )
        self.stale_s = (
            float(self.stale_s)
            if self.stale_s is not None
            else _env_float("LLM_LOCK_STALE_S", DEFAULT_STALE_S)
        )

    # ------------------------------------------------------------------ public
    def acquire(self, *, wait: bool = True) -> list[str]:
        """Взять lock или поднять ``LLMBusy``/``LLMLockUnavailable``."""
        deadline = time.monotonic() + max(0.0, self.timeout_s or 0.0)
        while True:
            try:
                self._create()
            except FileExistsError:
                if self._recover_stale():
                    continue
                if not wait:
                    raise LLMBusy("LLM lock занят другим процессом") from None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMBusy(
                        "LLM lock не освободился за отведённое время"
                    ) from None
                time.sleep(min(self.poll_interval_s, remaining))
            except OSError as exc:
                raise LLMLockUnavailable(
                    f"Не удалось создать LLM lock: {self.path}"
                ) from exc
            else:
                self._start_heartbeat()
                return list(self.warnings)

    def release(self) -> None:
        """Освободить принадлежащий нам lock."""
        self._stop_heartbeat()
        if not self._held:
            return
        try:
            payload = self._read()
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("token") == self._token:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
        self._held = False

    def heartbeat(self) -> None:
        """Обновить ``heartbeat_at``, сохранив владельца lock'а."""
        if not self._held:
            return
        payload = self._read()
        if not isinstance(payload, dict) or payload.get("token") != self._token:
            return
        payload["heartbeat_at"] = _utc_now()
        self._write(payload)

    @contextmanager
    def held(self, *, wait: bool = True) -> Iterator["LLMExecutionLock"]:
        """Контекстный менеджер удержания lock."""
        self.acquire(wait=wait)
        try:
            yield self
        finally:
            self.release()

    # ----------------------------------------------------------------- private
    def _metadata(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "pid": os.getpid(),
            "token": self._token,
            "kind": self.kind,
            "run_id": self.run_id,
            "test_id": self.test_id,
            "acquired_at": now,
            "heartbeat_at": now,
        }

    def _create(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL — атомарное «создать, только если нет» и на Windows, и на POSIX.
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            payload = json.dumps(
                self._metadata(), ensure_ascii=False, separators=(",", ":")
            )
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        self._held = True

    def _read(self) -> Any:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{self._token}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _recover_stale(self) -> bool:
        """Снять брошенный lock и вернуть ``True``, если можно повторить."""
        try:
            payload = self._read()
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            heartbeat = _parse_time(payload.get("heartbeat_at"))
            age = (
                (datetime.now(timezone.utc) - heartbeat).total_seconds()
                if heartbeat is not None
                else None
            )
            alive = _pid_alive(payload.get("pid"))
            stale = (not alive) or (age is not None and age > self.stale_s)
        else:
            # Битый/пустой файл: считаем брошенным, если он старше stale_s.
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                return False
            stale = age > self.stale_s
        if not stale:
            return False
        stale_path = self.path.with_name(
            f".{self.path.name}.stale-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            os.replace(self.path, stale_path)
        except OSError:
            return False
        try:
            stale_path.unlink(missing_ok=True)
        except OSError:
            pass
        if "stale_llm_lock_recovered" not in self.warnings:
            self.warnings.append("stale_llm_lock_recovered")
        return True

    def _start_heartbeat(self) -> None:
        interval = max(HEARTBEAT_MIN_S, (self.stale_s or DEFAULT_STALE_S) / 3.0)
        self._stop.clear()

        def beat() -> None:
            while not self._stop.wait(interval):
                try:
                    self.heartbeat()
                except (OSError, ValueError):
                    return

        self._heartbeat = threading.Thread(
            target=beat, name="llm-lock-heartbeat", daemon=True
        )
        self._heartbeat.start()

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        thread = self._heartbeat
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._heartbeat = None
