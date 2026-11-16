"""SQLite-хранилище исследований, сообщений и запросов (stdlib ``sqlite3``).

Без ORM. WAL, ``PRAGMA foreign_keys = ON``, миграции при старте. В БД попадают
только данные, которые уже вернул backend: полный текст chunk, HTML таблиц,
prompt и секреты не сохраняются.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from src.api.serialize import utc_now_iso

SCHEMA_VERSION = 1
ACTIVE_REQUEST_STATUSES = ("queued", "running")
TERMINAL_REQUEST_STATUSES = ("completed", "abstained", "failed", "cancelled")

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS researches (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  research_id TEXT NOT NULL REFERENCES researches(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL,
  request_id TEXT,
  client_message_id TEXT,
  answer_json TEXT,
  filters_json TEXT,
  error_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY,
  research_id TEXT NOT NULL REFERENCES researches(id) ON DELETE CASCADE,
  assistant_message_id TEXT NOT NULL,
  status TEXT NOT NULL,
  error_code TEXT,
  started_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_client
  ON messages(research_id, client_message_id)
  WHERE client_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_researches_owner_updated
  ON researches(owner_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS ix_messages_research
  ON messages(research_id);

CREATE INDEX IF NOT EXISTS ix_requests_research_status
  ON requests(research_id, status);

CREATE INDEX IF NOT EXISTS ix_requests_message
  ON requests(assistant_message_id);
"""

_INTERRUPTED_ERROR = {
    "code": "interrupted",
    "message": "Выполнение прервано перезапуском API.",
    "retryable": True,
}
_CANCELLED_ERROR = {
    "code": "cancelled",
    "message": "Запрос отменён.",
    "retryable": False,
}


class ResearchNotFoundError(LookupError):
    """Исследование не существует или принадлежит другому владельцу."""


class ActiveRequestError(RuntimeError):
    """У исследования уже есть активный запрос."""

    def __init__(self, active_request_id: str) -> None:
        super().__init__("request_in_progress")
        self.active_request_id = active_request_id


class Store:
    """Тонкая обёртка над SQLite: каждая операция — отдельное соединение."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.create_function("casefold", 1, _casefold)
        return connection

    def init(self) -> None:
        """Создать каталог, включить WAL и применить миграции."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                connection.executescript(_SCHEMA_V1)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        finally:
            connection.close()
        self.recover_interrupted()

    def ping(self) -> bool:
        """Проверить доступность SQLite."""
        try:
            connection = self._connect()
            try:
                connection.execute("SELECT 1").fetchone()
                return True
            finally:
                connection.close()
        except sqlite3.Error:
            return False

    # -------------------------------------------------------- interrupted runs
    def recover_interrupted(self) -> int:
        """``running``/``queued`` после перезапуска → ``failed(interrupted)``."""
        now = utc_now_iso()
        connection = self._connect()
        recovered = 0
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id, research_id, assistant_message_id FROM requests "
                "WHERE status IN (?, ?)",
                ACTIVE_REQUEST_STATUSES,
            ).fetchall()
            payload = json.dumps(_INTERRUPTED_ERROR, ensure_ascii=False)
            for row in rows:
                connection.execute(
                    "UPDATE requests SET status = 'failed', error_code = 'interrupted', "
                    "finished_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                connection.execute(
                    "UPDATE messages SET status = 'failed', error_json = ? "
                    "WHERE id = ?",
                    (payload, row["assistant_message_id"]),
                )
                connection.execute(
                    "UPDATE researches SET updated_at = ? WHERE id = ?",
                    (now, row["research_id"]),
                )
                recovered += 1
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()
        return recovered

    # -------------------------------------------------------------- researches
    def create_research(
        self,
        owner_id: str,
        *,
        title: str,
        research_id: str | None = None,
    ) -> dict[str, Any]:
        """Создать исследование."""
        now = utc_now_iso()
        identifier = research_id or str(uuid.uuid4())
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO researches (id, owner_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (identifier, owner_id, title, now, now),
            )
            connection.commit()
        finally:
            connection.close()
        result = self.get_research(owner_id, identifier)
        assert result is not None
        return result

    def get_research(
        self, owner_id: str, research_id: str
    ) -> dict[str, Any] | None:
        """Исследование владельца по ID."""
        connection = self._connect()
        try:
            row = connection.execute(
                _RESEARCH_SELECT + " WHERE r.owner_id = ? AND r.id = ?",
                (owner_id, research_id),
            ).fetchone()
        finally:
            connection.close()
        return _research_row(row) if row is not None else None

    def list_researches(
        self,
        owner_id: str,
        *,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Исследования владельца."""
        search = (query or "").strip()
        pattern = f"%{_casefold(search)}%"
        where = (
            " WHERE r.owner_id = ? AND ("
            "? = '' OR casefold(r.title) LIKE ? OR EXISTS ("
            "SELECT 1 FROM messages m WHERE m.research_id = r.id "
            "AND casefold(m.content) LIKE ?))"
        )
        params = (owner_id, search, pattern, pattern)
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM researches r" + where, params
                ).fetchone()[0]
            )
            rows = connection.execute(
                _RESEARCH_SELECT
                + where
                + " ORDER BY r.updated_at DESC, r.rowid DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        finally:
            connection.close()
        return [_research_row(row) for row in rows], total

    def update_research_title(
        self, owner_id: str, research_id: str, title: str
    ) -> dict[str, Any] | None:
        """Переименовать исследование."""
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE researches SET title = ? WHERE id = ? AND owner_id = ?",
                (title, research_id, owner_id),
            )
            connection.commit()
            if cursor.rowcount == 0:
                return None
        finally:
            connection.close()
        return self.get_research(owner_id, research_id)

    def delete_research(self, owner_id: str, research_id: str) -> bool:
        """Удалить исследование."""
        connection = self._connect()
        try:
            cursor = connection.execute(
                "DELETE FROM researches WHERE id = ? AND owner_id = ?",
                (research_id, owner_id),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    def touch_research(self, research_id: str) -> None:
        """Обновить ``updated_at`` исследования."""
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE researches SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), research_id),
            )
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------------- turns
    def find_client_message(
        self, owner_id: str, research_id: str, client_message_id: str
    ) -> dict[str, Any] | None:
        """Найти сообщение по idempotency-ключу."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT m.* FROM messages m "
                "JOIN researches r ON r.id = m.research_id "
                "WHERE m.research_id = ? AND r.owner_id = ? "
                "AND m.client_message_id = ?",
                (research_id, owner_id, client_message_id),
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def start_turn(
        self,
        *,
        owner_id: str,
        research_id: str,
        user_message_id: str,
        assistant_message_id: str,
        request_id: str,
        content: str,
        client_message_id: str,
        filters_json: str | None,
    ) -> dict[str, Any]:
        """Атомарно создать пользовательское сообщение, assistant-заглушку и запрос."""
        now = utc_now_iso()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            research = connection.execute(
                "SELECT id FROM researches WHERE id = ? AND owner_id = ?",
                (research_id, owner_id),
            ).fetchone()
            if research is None:
                connection.rollback()
                raise ResearchNotFoundError(research_id)

            existing = connection.execute(
                "SELECT * FROM messages WHERE research_id = ? "
                "AND client_message_id = ?",
                (research_id, client_message_id),
            ).fetchone()
            if existing is not None:
                request = connection.execute(
                    "SELECT * FROM requests WHERE id = ?",
                    (existing["request_id"],),
                ).fetchone()
                connection.commit()
                return {
                    "duplicate": True,
                    "user_message": dict(existing),
                    "assistant_message": (
                        self._get_message_row(
                            connection, request["assistant_message_id"]
                        )
                        if request is not None
                        else None
                    ),
                    "request": dict(request) if request is not None else None,
                }

            active = connection.execute(
                "SELECT id FROM requests WHERE research_id = ? "
                "AND status IN (?, ?) ORDER BY rowid DESC LIMIT 1",
                (research_id, *ACTIVE_REQUEST_STATUSES),
            ).fetchone()
            if active is not None:
                connection.rollback()
                raise ActiveRequestError(str(active["id"]))

            connection.execute(
                "INSERT INTO messages (id, research_id, role, content, status, "
                "request_id, client_message_id, created_at) "
                "VALUES (?, ?, 'user', ?, 'completed', ?, ?, ?)",
                (user_message_id, research_id, content, request_id,
                 client_message_id, now),
            )
            connection.execute(
                "INSERT INTO messages (id, research_id, role, content, status, "
                "request_id, filters_json, created_at) "
                "VALUES (?, ?, 'assistant', '', 'streaming', ?, ?, ?)",
                (assistant_message_id, research_id, request_id,
                 filters_json, now),
            )
            connection.execute(
                "INSERT INTO requests (id, research_id, assistant_message_id, "
                "status, started_at) VALUES (?, ?, ?, 'queued', ?)",
                (request_id, research_id, assistant_message_id, now),
            )
            connection.execute(
                "UPDATE researches SET updated_at = ? WHERE id = ?",
                (now, research_id),
            )
            connection.commit()
            return {
                "duplicate": False,
                "user_message": self._get_message_row(connection, user_message_id),
                "assistant_message": self._get_message_row(
                    connection, assistant_message_id
                ),
                "request": self._get_request_row(connection, request_id),
            }
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _get_message_row(
        connection: sqlite3.Connection, message_id: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _get_request_row(
        connection: sqlite3.Connection, request_id: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def mark_request_running(self, request_id: str) -> None:
        """Перевести запрос в ``running``."""
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE requests SET status = 'running' WHERE id = ? "
                "AND status = 'queued'",
                (request_id,),
            )
            connection.commit()
        finally:
            connection.close()

    def finish_request(
        self, request_id: str, status: str, *, error_code: str | None = None
    ) -> bool:
        """Завершить запрос указанным статусом."""
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE requests SET status = ?, error_code = ?, finished_at = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (status, error_code, utc_now_iso(), request_id,
                 *ACTIVE_REQUEST_STATUSES),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    def request_status(self, request_id: str) -> str | None:
        """Статус запроса."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT status FROM requests WHERE id = ?", (request_id,)
            ).fetchone()
        finally:
            connection.close()
        return str(row["status"]) if row is not None else None

    def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        status: str | None = None,
        answer_json: str | None = None,
        filters_json: str | None = None,
        error_json: str | None = None,
    ) -> None:
        """Обновить поля сообщения."""
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("content", content),
            ("status", status),
            ("answer_json", answer_json),
            ("filters_json", filters_json),
            ("error_json", error_json),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(message_id)
        connection = self._connect()
        try:
            connection.execute(
                f"UPDATE messages SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            connection.commit()
        finally:
            connection.close()

    def cancel_request(self, request_id: str) -> bool:
        """Пометить активный запрос и его сообщение как отменённые."""
        error = dict(_CANCELLED_ERROR)
        payload = json.dumps(error, ensure_ascii=False)
        now = utc_now_iso()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id, research_id, assistant_message_id, status "
                "FROM requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if row is None or row["status"] not in ACTIVE_REQUEST_STATUSES:
                connection.rollback()
                return False
            connection.execute(
                "UPDATE requests SET status = 'cancelled', error_code = 'cancelled', "
                "finished_at = ? WHERE id = ?",
                (now, request_id),
            )
            connection.execute(
                "UPDATE messages SET status = 'cancelled', content = '', "
                "answer_json = NULL, error_json = ? WHERE id = ?",
                (payload, row["assistant_message_id"]),
            )
            connection.execute(
                "UPDATE researches SET updated_at = ? WHERE id = ?",
                (now, row["research_id"]),
            )
            connection.commit()
            return True
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()

    # ---------------------------------------------------------------- requests
    def get_request_internal(self, request_id: str) -> dict[str, Any] | None:
        """Запрос без проверки владельца."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM requests WHERE id = ?", (request_id,)
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def get_request(
        self, owner_id: str, request_id: str
    ) -> dict[str, Any] | None:
        """Запрос с его assistant-сообщением; чужая запись не видна."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT q.id AS request_id, q.research_id, "
                "q.assistant_message_id, q.status AS request_status, "
                "q.error_code, q.started_at, q.finished_at, "
                "m.id AS message_id, m.role, m.content, "
                "m.status AS message_status, m.request_id AS message_request_id, "
                "m.answer_json, m.filters_json, m.error_json, m.created_at "
                "FROM requests q "
                "LEFT JOIN messages m ON m.id = q.assistant_message_id "
                "JOIN researches r ON r.id = q.research_id "
                "WHERE q.id = ? AND r.owner_id = ?",
                (request_id, owner_id),
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    # ---------------------------------------------------------------- messages
    def get_message_internal(self, message_id: str) -> dict[str, Any] | None:
        """Сообщение без проверки владельца."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def list_messages(
        self,
        owner_id: str,
        research_id: str,
        *,
        limit: int = 50,
        before: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Сообщения исследования владельца."""
        connection = self._connect()
        try:
            research = connection.execute(
                "SELECT id FROM researches WHERE id = ? AND owner_id = ?",
                (research_id, owner_id),
            ).fetchone()
            if research is None:
                return None
            before_rowid: int | None = None
            if before:
                anchor = connection.execute(
                    "SELECT m.rowid AS rowid FROM messages m "
                    "WHERE m.id = ? AND m.research_id = ?",
                    (before, research_id),
                ).fetchone()
                if anchor is None:
                    raise KeyError(before)
                before_rowid = int(anchor["rowid"])
            if before_rowid is None:
                rows = connection.execute(
                    "SELECT * FROM messages WHERE research_id = ? "
                    "ORDER BY rowid DESC LIMIT ?",
                    (research_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM messages WHERE research_id = ? AND rowid < ? "
                    "ORDER BY rowid DESC LIMIT ?",
                    (research_id, before_rowid, limit),
                ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in reversed(rows)]


def _casefold(value: Any) -> str:
    return str(value or "").casefold()


_RESEARCH_SELECT = """
SELECT r.id, r.owner_id, r.title, r.created_at, r.updated_at,
  (SELECT COUNT(*) FROM messages m WHERE m.research_id = r.id) AS messages_count,
  (SELECT m.content FROM messages m WHERE m.research_id = r.id
     AND m.role = 'user' ORDER BY m.rowid ASC LIMIT 1) AS first_user_content,
  (SELECT m.status FROM messages m WHERE m.research_id = r.id
     AND m.role = 'assistant' ORDER BY m.rowid DESC LIMIT 1) AS last_message_status,
  (SELECT q.status FROM requests q WHERE q.research_id = r.id
     ORDER BY q.rowid DESC LIMIT 1) AS last_request_status
FROM researches r
"""


def _last_status(row: sqlite3.Row) -> str:
    request_status = row["last_request_status"]
    if request_status:
        return "running" if request_status in ACTIVE_REQUEST_STATUSES else str(
            request_status
        )
    message_status = row["last_message_status"]
    if message_status in {"streaming", "pending"}:
        return "running"
    if message_status in {
        "completed",
        "abstained",
        "failed",
        "cancelled",
    }:
        return str(message_status)
    return "idle"


def _research_row(row: sqlite3.Row) -> dict[str, Any]:
    preview = row["first_user_content"]
    if isinstance(preview, str) and preview:
        preview = " ".join(preview.split())
        if len(preview) > 160:
            preview = preview[:157].rstrip() + "..."
    else:
        preview = None
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_message_preview": preview,
        "last_status": _last_status(row),
        "messages_count": int(row["messages_count"] or 0),
    }
