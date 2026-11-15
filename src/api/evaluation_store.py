"""Evaluation storage + фоновый worker (stdlib ``sqlite3``, без ORM).

Отдельная БД ``data/evaluation/evaluation.sqlite3`` (``EVAL_DB_PATH``) не
смешивается с историей пользовательских диалогов. Полные traces и отчёты
остаются на диске в ``eval/results/{run_id}/``; SQLite хранит индекс, статусы,
summary и краткую запись ``cases.jsonl``.

API не пересчитывает метрики: ``summary.json`` и ``cases.jsonl`` читает worker
после завершения runner'а, а endpoint'ы только отдают сохранённые значения.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.api.config import ApiConfig
from src.api.serialize import utc_now_iso

SCHEMA_VERSION = 1

RUN_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "cancel_requested",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
)
ACTIVE_RUN_STATUSES: tuple[str, ...] = ("queued", "running", "cancel_requested")
TERMINAL_RUN_STATUSES: tuple[str, ...] = (
    "completed",
    "failed",
    "cancelled",
    "interrupted",
)
RETRIEVAL_MODES: tuple[str, ...] = ("hybrid", "dense", "keyword")

_SCHEDULED_STATUSES = ("queued", "running", "cancel_requested")

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
  id TEXT PRIMARY KEY,
  client_request_id TEXT UNIQUE,
  status TEXT NOT NULL,
  suite TEXT NOT NULL,
  split TEXT NOT NULL,
  retrieval_mode TEXT NOT NULL,
  judge INTEGER NOT NULL DEFAULT 0,
  repeat INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  progress_json TEXT,
  dataset_sha256 TEXT,
  corpus_content_sha256 TEXT,
  git_sha TEXT,
  config_json TEXT,
  summary_json TEXT,
  error_json TEXT,
  artifacts_dir TEXT,
  request_json TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_test_results (
  run_id TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  test_id TEXT NOT NULL,
  retrieval_mode TEXT NOT NULL,
  attempt_index INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  pass INTEGER NOT NULL,
  metrics_json TEXT,
  critical_json TEXT,
  trace_path TEXT,
  duration_ms REAL,
  suite TEXT,
  category TEXT,
  difficulty TEXT,
  split TEXT,
  trace_available INTEGER NOT NULL DEFAULT 0,
  critical_count INTEGER NOT NULL DEFAULT 0,
  case_json TEXT,
  PRIMARY KEY (run_id, test_id, retrieval_mode, attempt_index)
);

CREATE INDEX IF NOT EXISTS ix_eval_runs_created
  ON evaluation_runs(created_at DESC);

CREATE INDEX IF NOT EXISTS ix_eval_results_failed
  ON evaluation_test_results(run_id, retrieval_mode, pass);

CREATE INDEX IF NOT EXISTS ix_eval_results_test
  ON evaluation_test_results(test_id, run_id);
"""

_INTERRUPTED_ERROR = {
    "code": "interrupted",
    "message": "Выполнение прервано перезапуском API.",
    "retryable": True,
}
_CANCELLED_ERROR = {
    "code": "cancelled",
    "message": "Запуск отменён.",
    "retryable": False,
}
_CANCEL_TIMEOUT_ERROR = {
    "code": "cancel_timeout",
    "message": "Worker не остановился за отведённое время.",
    "retryable": True,
}


def utc_now() -> str:
    """ISO 8601 UTC с секундной точностью, как в контракте API."""
    return utc_now_iso()


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _loads(raw: Any, fallback: Any = None) -> Any:
    if raw is None:
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return fallback


def _as_bool_int(value: Any) -> int:
    return 1 if value else 0


class EvaluationStore:
    """Тонкая обёртка над SQLite: каждая операция — отдельное соединение."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def init(self) -> None:
        """Создать каталог, включить WAL, применить миграции, снять зависшие run."""
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if version < 1:
                    connection.executescript(_SCHEMA_V1)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            finally:
                connection.close()
            self._initialized = True
        self.recover_interrupted()

    def ping(self) -> bool:
        try:
            connection = self._connect()
            try:
                connection.execute("SELECT 1").fetchone()
                return True
            finally:
                connection.close()
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------- lifecycle
    def recover_interrupted(self) -> int:
        """``queued``/``running``/``cancel_requested`` после рестарта → ``interrupted``."""
        now = utc_now()
        payload = _dumps(_INTERRUPTED_ERROR)
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE evaluation_runs SET status = 'interrupted', "
                "finished_at = ?, error_json = ? WHERE status IN (?, ?, ?)",
                (now, payload, *_SCHEDULED_STATUSES),
            )
            connection.commit()
            return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def count_active(self) -> tuple[int, int]:
        """Вернуть ``(running_like, queued)`` без гонок между вызовами."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT "
                "SUM(CASE WHEN status IN ('running', 'cancel_requested') "
                "THEN 1 ELSE 0 END) AS running, "
                "SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued "
                "FROM evaluation_runs"
            ).fetchone()
        finally:
            connection.close()
        return int(row["running"] or 0), int(row["queued"] or 0)

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        *,
        run_id: str,
        suite: str,
        split: str,
        retrieval_mode: str,
        judge: bool,
        repeat: int,
        request: Mapping[str, Any] | None = None,
        notes: str | None = None,
        client_request_id: str | None = None,
        tests_total: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        progress = {
            "tests_total": int(tests_total or 0),
            "tests_completed": 0,
            "current_test_id": None,
            "current_turn": None,
            "percent": 0.0,
        }
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO evaluation_runs (id, client_request_id, status, suite, "
                "split, retrieval_mode, judge, repeat, created_at, progress_json, "
                "request_json, notes) "
                "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    client_request_id,
                    suite,
                    split,
                    retrieval_mode,
                    _as_bool_int(judge),
                    int(repeat),
                    now,
                    _dumps(progress),
                    _dumps(dict(request or {})),
                    notes,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        run = self.get_run(run_id)
        assert run is not None
        return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?", (run_id,)
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def find_by_client_request_id(
        self, client_request_id: str
    ) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE client_request_id = ?",
                (client_request_id,),
            ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def list_runs(
        self,
        *,
        suite: str | None = None,
        status: str | None = None,
        split: str | None = None,
        retrieval_mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("suite", suite),
            ("status", status),
            ("split", split),
            ("retrieval_mode", retrieval_mode),
        ):
            if value:
                where.append(f"{column} = ?")
                params.append(value)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM evaluation_runs" + clause, tuple(params)
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT * FROM evaluation_runs"
                + clause
                + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (*params, int(limit), int(offset)),
            ).fetchall()
        finally:
            connection.close()
        return [dict(row) for row in rows], total

    def last_completed_run(self, *, retrieval_mode: str | None = None) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            if retrieval_mode:
                row = connection.execute(
                    "SELECT * FROM evaluation_runs WHERE status = 'completed' "
                    "AND retrieval_mode = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (retrieval_mode,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM evaluation_runs WHERE status = 'completed' "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
        finally:
            connection.close()
        return dict(row) if row is not None else None

    def mark_running(self, run_id: str) -> bool:
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE evaluation_runs SET status = 'running', started_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (utc_now(), run_id),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    def set_progress(self, run_id: str, progress: Mapping[str, Any]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE evaluation_runs SET progress_json = ? "
                "WHERE id = ? AND status IN (?, ?, ?)",
                (_dumps(dict(progress)), run_id, *_SCHEDULED_STATUSES),
            )
            connection.commit()
        finally:
            connection.close()

    def request_cancel(self, run_id: str) -> str | None:
        """Запросить отмену. Возвращает новый статус или ``None``.

        ``queued`` → ``cancelled`` сразу; ``running`` → ``cancel_requested``;
        терминальный статус — ``None`` (вызывающий отдаёт 409).
        """
        now = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM evaluation_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            status = str(row["status"])
            if status == "queued":
                connection.execute(
                    "UPDATE evaluation_runs SET status = 'cancelled', finished_at = ?, "
                    "error_json = ? WHERE id = ? AND status = 'queued'",
                    (now, _dumps(_CANCELLED_ERROR), run_id),
                )
                connection.commit()
                return "cancelled"
            if status == "running":
                connection.execute(
                    "UPDATE evaluation_runs SET status = 'cancel_requested' "
                    "WHERE id = ? AND status = 'running'",
                    (run_id,),
                )
                connection.commit()
                return "cancel_requested"
            if status == "cancel_requested":
                connection.commit()
                return "cancel_requested"
            connection.rollback()
            return None
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_cancel_timeout(self, run_id: str) -> bool:
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE evaluation_runs SET status = 'failed', finished_at = ?, "
                "error_json = ? WHERE id = ? AND status = 'cancel_requested'",
                (utc_now(), _dumps(_CANCEL_TIMEOUT_ERROR), run_id),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        error: Mapping[str, Any] | None = None,
        artifacts_dir: str | None = None,
        summary: Mapping[str, Any] | None = None,
        dataset_sha256: str | None = None,
        corpus_content_sha256: str | None = None,
        git_sha: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> bool:
        """Перевести run в терминальный статус; назад переходы запрещены."""
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError(status)
        assignments = [
            "status = ?",
            "finished_at = ?",
            "error_json = ?",
        ]
        values: list[Any] = [
            status,
            utc_now(),
            _dumps(dict(error)) if error else None,
        ]
        for column, value in (
            ("artifacts_dir", artifacts_dir),
            ("summary_json", _dumps(dict(summary)) if summary is not None else None),
            ("dataset_sha256", dataset_sha256),
            ("corpus_content_sha256", corpus_content_sha256),
            ("git_sha", git_sha),
            ("config_json", _dumps(dict(config)) if config is not None else None),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        values.append(run_id)
        connection = self._connect()
        try:
            cursor = connection.execute(
                "UPDATE evaluation_runs SET "
                + ", ".join(assignments)
                + " WHERE id = ? AND status IN (?, ?, ?)",
                (*values, *_SCHEDULED_STATUSES),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()

    # --------------------------------------------------------------- results
    def update_artifacts(
        self,
        run_id: str,
        *,
        summary: Mapping[str, Any] | None = None,
        artifacts_dir: Path | str | None = None,
    ) -> None:
        """Сохранить summary/каталог артефактов, не меняя статус run'а."""
        assignments: list[str] = []
        values: list[Any] = []
        if summary is not None:
            assignments.append("summary_json = ?")
            values.append(_dumps(dict(summary)))
        if artifacts_dir is not None:
            assignments.append("artifacts_dir = ?")
            values.append(str(artifacts_dir))
        if not assignments:
            return
        values.append(run_id)
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE evaluation_runs SET " + ", ".join(assignments) + " WHERE id = ?",
                values,
            )
            connection.commit()
        finally:
            connection.close()

    def replace_results(
        self, run_id: str, cases: list[Mapping[str, Any]]
    ) -> int:
        """Идемпотентно записать ``cases.jsonl``: повтор обновляет строку."""
        rows = []
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            test_id = case.get("id")
            if not isinstance(test_id, str) or not test_id:
                continue
            mode = case.get("retrieval_mode") or "hybrid"
            attempt = case.get("attempt_index")
            if isinstance(attempt, bool) or not isinstance(attempt, int):
                attempt = 0
            critical = case.get("critical_errors") or []
            if not isinstance(critical, list):
                critical = []
            metrics = case.get("metrics")
            if not isinstance(metrics, Mapping):
                metrics = case.get("retrieval") or {}
            latency = (case.get("efficiency") or {}).get("latency_ms")
            rows.append(
                (
                    run_id,
                    test_id,
                    str(mode),
                    int(attempt),
                    str(case.get("status") or "completed"),
                    _as_bool_int(case.get("pass")),
                    _dumps(dict(metrics)),
                    _dumps(list(critical)),
                    case.get("trace_path"),
                    float(latency) if isinstance(latency, (int, float)) else None,
                    case.get("suite"),
                    case.get("category"),
                    case.get("difficulty"),
                    case.get("split"),
                    _as_bool_int(case.get("trace_available")),
                    len(critical),
                    _dumps(dict(case)),
                )
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO evaluation_test_results (run_id, test_id, "
                "retrieval_mode, attempt_index, status, pass, metrics_json, "
                "critical_json, trace_path, duration_ms, suite, category, "
                "difficulty, split, trace_available, critical_count, case_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, test_id, retrieval_mode, attempt_index) "
                "DO UPDATE SET status = excluded.status, pass = excluded.pass, "
                "metrics_json = excluded.metrics_json, "
                "critical_json = excluded.critical_json, "
                "trace_path = excluded.trace_path, duration_ms = excluded.duration_ms, "
                "suite = excluded.suite, category = excluded.category, "
                "difficulty = excluded.difficulty, split = excluded.split, "
                "trace_available = excluded.trace_available, "
                "critical_count = excluded.critical_count, "
                "case_json = excluded.case_json",
                rows,
            )
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()
        return len(rows)

    def get_result(
        self,
        run_id: str,
        test_id: str,
        retrieval_mode: str,
        attempt_index: int,
    ) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM evaluation_test_results WHERE run_id = ? "
                "AND test_id = ? AND retrieval_mode = ? AND attempt_index = ?",
                (run_id, test_id, retrieval_mode, int(attempt_index)),
            ).fetchone()
        finally:
            connection.close()
        return _result_row(row) if row is not None else None

    def result_modes(self, run_id: str, test_id: str) -> list[str]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT DISTINCT retrieval_mode FROM evaluation_test_results "
                "WHERE run_id = ? AND test_id = ? ORDER BY retrieval_mode",
                (run_id, test_id),
            ).fetchall()
        finally:
            connection.close()
        return [str(row["retrieval_mode"]) for row in rows]

    def list_results(
        self,
        run_id: str,
        *,
        status: str | None = None,
        passed: bool | None = None,
        critical: bool | None = None,
        difficulty: str | None = None,
        category: str | None = None,
        retrieval_mode: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where = ["run_id = ?"]
        params: list[Any] = [run_id]
        if status:
            where.append("status = ?")
            params.append(status)
        if passed is not None:
            where.append("pass = ?")
            params.append(_as_bool_int(passed))
        if critical is not None:
            where.append("critical_count > 0" if critical else "critical_count = 0")
        if difficulty:
            where.append("difficulty = ?")
            params.append(difficulty)
        if category:
            where.append("category = ?")
            params.append(category)
        if retrieval_mode:
            where.append("retrieval_mode = ?")
            params.append(retrieval_mode)
        clause = " WHERE " + " AND ".join(where)
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM evaluation_test_results" + clause,
                    tuple(params),
                ).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT * FROM evaluation_test_results"
                + clause
                + " ORDER BY test_id, retrieval_mode, attempt_index LIMIT ? OFFSET ?",
                (*params, int(limit), int(offset)),
            ).fetchall()
        finally:
            connection.close()
        return [_result_row(row) for row in rows], total

    def all_results(self, run_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM evaluation_test_results WHERE run_id = ? "
                "ORDER BY test_id, retrieval_mode, attempt_index",
                (run_id,),
            ).fetchall()
        finally:
            connection.close()
        return [_result_row(row) for row in rows]

    def last_results_for_run(
        self, run_id: str, retrieval_mode: str
    ) -> dict[str, dict[str, Any]]:
        """Последний attempt каждого теста в режиме — для каталога тестов."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM evaluation_test_results WHERE run_id = ? "
                "AND retrieval_mode = ? ORDER BY test_id, attempt_index DESC",
                (run_id, retrieval_mode),
            ).fetchall()
        finally:
            connection.close()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = _result_row(row)
            result.setdefault(str(payload["test_id"]), payload)
        return result

    def test_history(
        self, test_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT r.run_id, r.retrieval_mode, r.attempt_index, r.status, "
                "r.pass, r.metrics_json, r.critical_json, r.critical_count, "
                "r.case_json, r.trace_path, r.trace_available, "
                "q.created_at, q.suite, q.split "
                "FROM evaluation_test_results r "
                "JOIN evaluation_runs q ON q.id = r.run_id "
                "WHERE r.test_id = ? AND q.status = 'completed' "
                "ORDER BY q.created_at DESC, r.attempt_index DESC LIMIT ?",
                (test_id, int(limit)),
            ).fetchall()
        finally:
            connection.close()
        return [
            {
                "run_id": str(row["run_id"]),
                "created_at": row["created_at"],
                "suite": row["suite"],
                "split": row["split"],
                "retrieval_mode": row["retrieval_mode"],
                "attempt_index": int(row["attempt_index"] or 0),
                "status": row["status"],
                "pass": bool(row["pass"]),
                "metrics": _loads(row["metrics_json"], {}),
                "critical_errors": _loads(row["critical_json"], []),
                "trace_available": bool(row["trace_available"]),
            }
            for row in rows
        ]

    # -------------------------------------------------------------- retention
    def prune(self, keep: int) -> list[dict[str, Any]]:
        """Удалить старые завершённые run'ы; активные не трогаются."""
        if keep <= 0:
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT id, artifacts_dir FROM evaluation_runs "
                "WHERE status IN ('completed', 'failed', 'cancelled', 'interrupted') "
                "ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?",
                (int(keep),),
            ).fetchall()
            removed = [dict(row) for row in rows]
            if removed:
                connection.executemany(
                    "DELETE FROM evaluation_runs WHERE id = ? AND status IN "
                    "('completed', 'failed', 'cancelled', 'interrupted')",
                    [(row["id"],) for row in removed],
                )
                connection.commit()
        finally:
            connection.close()
        return removed


def _result_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["pass"] = bool(payload.get("pass"))
    payload["metrics"] = _loads(payload.pop("metrics_json", None), {})
    payload["critical_errors"] = _loads(payload.pop("critical_json", None), [])
    payload["trace_available"] = bool(payload.get("trace_available"))
    payload["case"] = _loads(payload.pop("case_json", None), {})
    payload["attempt_index"] = int(payload.get("attempt_index") or 0)
    return payload


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class _RunControl:
    cancel: threading.Event = field(default_factory=threading.Event)


#: Контракт runner'а: (request, out_dir, progress, cancel_event) -> путь артефактов.
Runner = Callable[..., Path]


def make_agent_runner(config: ApiConfig, services: Any = None) -> Runner:
    """Тонкая обёртка над ``eval.runner.run_suite`` — второй реализации нет."""

    def run(
        *,
        request: Mapping[str, Any],
        out_dir: Path,
        progress: Callable[[dict[str, Any]], None],
        cancel_event: threading.Event,
    ) -> Path:
        from eval.runner import run_suite

        fake = bool(config.eval_fake_service)
        service = None
        if not fake and services is not None:
            service = services.evaluation_agent()
        return run_suite(
            suite=str(request.get("suite") or "smoke"),
            split=str(request.get("split") or "dev"),
            retrieval_mode=str(request.get("retrieval_mode") or "hybrid"),
            judge=bool(request.get("judge")),
            repeat=int(request.get("repeat") or 1),
            progress_callback=progress,
            cancel_event=cancel_event,
            service=service,
            fake_service=fake,
            service_handles_lock=not fake,
            out_dir=Path(out_dir),
            timeout=float(config.eval_turn_timeout_s) or None,
        )

    return run


class EvaluationWorker:
    """Один фоновый исполнитель benchmark; LLM в HTTP-запросе не запускается."""

    def __init__(
        self,
        store: EvaluationStore,
        config: ApiConfig,
        *,
        runner: Runner | None = None,
        services: Any = None,
    ) -> None:
        self.store = store
        self.config = config
        self.services = services
        self._runner = runner or make_agent_runner(config, services)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eval-worker"
        )
        self._lock = threading.Lock()
        self._controls: dict[str, _RunControl] = {}
        self._progress: dict[str, dict[str, Any]] = {}
        self._closed = False

    # ------------------------------------------------------------------ queue
    def submit(self, run_id: str) -> None:
        """Поставить запуск в очередь; сама задача выполняется в потоке worker'а."""
        control = _RunControl()
        with self._lock:
            self._controls[run_id] = control
        self._executor.submit(self._execute, run_id, control)

    def cancel(self, run_id: str, *, status: str) -> None:
        """Сообщить worker'у о запросе отмены и запустить watchdog таймаута."""
        with self._lock:
            control = self._controls.get(run_id)
            self._progress.setdefault(run_id, {})
        if control is not None:
            control.cancel.set()
        if status == "cancel_requested":
            grace = max(float(self.config.eval_turn_timeout_s), 0.05) + 0.5
            timer = threading.Timer(grace, self._cancel_timeout, args=(run_id,))
            timer.daemon = True
            timer.start()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            controls = list(self._controls.values())
        for control in controls:
            control.cancel.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # --------------------------------------------------------------- internal
    def _cancel_timeout(self, run_id: str) -> None:
        self.store.mark_cancel_timeout(run_id)

    def _run_timeout(self, run_id: str, control: _RunControl) -> None:
        """Per-run timeout: не превращается в 0 и не оставляет вечный running."""
        current = (self.store.get_run(run_id) or {}).get("status")
        if current not in {"running", "cancel_requested"}:
            return
        control.cancel.set()
        self.store.finish_run(
            run_id,
            "failed",
            error={
                "code": "run_timeout",
                "message": "Запуск превысил per-run timeout.",
                "retryable": True,
            },
        )

    def _notify(self, run_id: str, event: Mapping[str, Any]) -> None:
        progress = self._progress.setdefault(run_id, {})
        kind = str(event.get("event") or "")
        total = event.get("total")
        if isinstance(total, int) and not isinstance(total, bool):
            progress["tests_total"] = total
        selected = event.get("tests_selected")
        if isinstance(selected, int) and not isinstance(selected, bool):
            progress["tests_total"] = selected
        if kind == "test_started":
            progress["current_test_id"] = event.get("test_id")
            attempt = event.get("attempt_index")
            progress["current_turn"] = (
                int(attempt) + 1
                if isinstance(attempt, int) and not isinstance(attempt, bool)
                else 1
            )
        if kind == "test_completed":
            completed = event.get("completed")
            if isinstance(completed, int) and not isinstance(completed, bool):
                progress["tests_completed"] = completed
            progress["current_test_id"] = None
            progress["current_turn"] = None
        total_value = progress.get("tests_total") or 0
        completed_value = progress.get("tests_completed") or 0
        progress["percent"] = (
            round(completed_value * 100.0 / total_value, 1) if total_value else 0.0
        )
        if kind in {"test_started", "test_completed", "run_started"}:
            self.store.set_progress(run_id, progress)

    def _execute(self, run_id: str, control: _RunControl) -> None:
        run = self.store.get_run(run_id)
        if run is None or run.get("status") != "queued":
            self._forget(run_id)
            return
        if not self.store.mark_running(run_id):
            self._forget(run_id)
            return
        request = _loads(run.get("request_json"), {}) or {}
        out_dir = Path(self.config.eval_results_dir)
        run_timer = threading.Timer(
            max(float(self.config.eval_run_timeout_s), 1.0),
            self._run_timeout,
            args=(run_id, control),
        )
        run_timer.daemon = True
        run_timer.start()
        try:
            final_dir = self._runner(
                request=request,
                out_dir=out_dir,
                progress=lambda event: self._notify(run_id, event),
                cancel_event=control.cancel,
            )
        except BaseException as exc:  # noqa: BLE001 — код ошибки наружу, без traceback
            self._fail(run_id, exc)
            self._forget(run_id)
            return
        finally:
            run_timer.cancel()

        imported = False
        if final_dir is not None:
            imported = self.import_artifacts(run_id, Path(final_dir))
        if control.cancel.is_set() or (self.store.get_run(run_id) or {}).get(
            "status"
        ) == "cancel_requested":
            self.store.finish_run(
                run_id, "cancelled", error=dict(_CANCELLED_ERROR)
            )
            self._prune()
            self._forget(run_id)
            return
        if not imported:
            self.store.finish_run(
                run_id,
                "failed",
                error={
                    "code": "artifact_import_error",
                    "message": "Runner не создал summary.json/cases.jsonl.",
                    "retryable": True,
                },
            )
            self._forget(run_id)
            return
        self.store.finish_run(run_id, "completed")
        self._prune()
        self._forget(run_id)

    def _fail(self, run_id: str, exc: BaseException) -> None:
        status = (self.store.get_run(run_id) or {}).get("status")
        if status == "cancel_requested":
            self.store.finish_run(
                run_id, "cancelled", error=dict(_CANCELLED_ERROR)
            )
            return
        code = getattr(exc, "code", None)
        if not isinstance(code, str) or not code:
            code = "runner_error"
        message = _safe_message(exc)
        self.store.finish_run(
            run_id,
            "failed",
            error={"code": code, "message": message, "retryable": True},
        )

    def _forget(self, run_id: str) -> None:
        with self._lock:
            self._controls.pop(run_id, None)
            self._progress.pop(run_id, None)

    def _prune(self) -> None:
        removed = self.store.prune(int(self.config.eval_retention_runs))
        for row in removed:
            artifacts = row.get("artifacts_dir")
            if not artifacts:
                continue
            path = Path(str(artifacts))
            # Удаляем только каталог внутри настроенного results_dir.
            try:
                root = Path(self.config.eval_results_dir).resolve()
                resolved = path.resolve()
                if resolved.is_dir() and root in resolved.parents:
                    import shutil

                    shutil.rmtree(resolved, ignore_errors=True)
            except OSError:
                continue

    # ---------------------------------------------------------------- import
    def import_artifacts(self, run_id: str, artifacts_dir: Path) -> bool:
        """Импортировать summary.json/cases.jsonl; traces остаются на диске."""
        summary_path = artifacts_dir / "summary.json"
        if not summary_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(summary, dict):
            return False
        cases: list[dict[str, Any]] = []
        cases_path = artifacts_dir / "cases.jsonl"
        if cases_path.exists():
            try:
                for line in cases_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        cases.append(payload)
            except (OSError, ValueError):
                return False
        try:
            self.store.replace_results(run_id, cases)
        except sqlite3.Error:
            return False
        dataset = summary.get("dataset") or {}
        corpus = summary.get("corpus") or {}
        git = summary.get("git") or {}
        config = summary.get("config") or {}
        current = (self.store.get_run(run_id) or {}).get("status")
        if current in {"cancel_requested", "cancelled", "failed", "interrupted"}:
            # Отмена/сбой во время выполнения: артефакты сохраняем, но
            # терминальный статус не переписываем на completed.
            self.store.update_artifacts(
                run_id, summary=summary, artifacts_dir=artifacts_dir
            )
            return True
        updated = self.store.finish_run(
            run_id,
            "completed",
            summary=summary,
            artifacts_dir=str(artifacts_dir),
            dataset_sha256=dataset.get("sha256"),
            corpus_content_sha256=corpus.get("content_sha256"),
            git_sha=git.get("sha"),
            config=config,
        )
        if not updated:
            # Статус уже изменён (например, watchdog отметил cancel_timeout):
            # summary сохраняем, но терминальный статус не переписываем.
            self.store.update_artifacts(run_id, summary=summary, artifacts_dir=artifacts_dir)
        return True


_SECRET_MARKERS = (
    "api_key",
    "api-key",
    "apikey",
    "authorization",
    "bearer ",
    "secret",
    "password",
    "access_token",
)


def _safe_message(exc: BaseException) -> str:
    """Короткое сообщение без ключей, prompt и traceback."""
    text = str(exc) or exc.__class__.__name__
    lowered = text.casefold()
    for marker in _SECRET_MARKERS:
        if marker in lowered:
            text = "Ошибка runner'а (детали скрыты)."
            break
    return text[:300]


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class EvaluationDatasetSnapshot:
    """Проверенный набор для каталога/определений тестов; без чтения корпуса."""

    available: bool
    tests: list[dict[str, Any]] = field(default_factory=list)
    suite_hashes: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    dataset_sha256: str | None = None
    expected_corpus_sha256: str | None = None
    error_code: str | None = None

    def test_by_id(self, test_id: str) -> dict[str, Any] | None:
        for test in self.tests:
            if test.get("id") == test_id:
                return test
        return None


class EvaluationRuntime:
    """Store + worker + dataset-кэш, привязанные к конфигурации приложения."""

    def __init__(
        self,
        config: ApiConfig,
        store: EvaluationStore,
        worker: EvaluationWorker,
        *,
        datasets_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.worker = worker
        self.datasets_dir = Path(datasets_dir or config.eval_datasets_dir)
        self._dataset_lock = threading.Lock()
        self._dataset: EvaluationDatasetSnapshot | None = None

    def dataset(self, *, refresh: bool = False) -> EvaluationDatasetSnapshot:
        with self._dataset_lock:
            if self._dataset is not None and not refresh:
                return self._dataset
            self._dataset = _load_dataset(self.datasets_dir)
            return self._dataset

    def shutdown(self) -> None:
        self.worker.shutdown()


def _load_dataset(datasets_dir: Path) -> EvaluationDatasetSnapshot:
    try:
        from eval.dataset import DatasetError, validate_dataset
    except Exception as exc:  # noqa: BLE001 — каталог без eval-пакета честно пуст
        return EvaluationDatasetSnapshot(
            available=False, error_code=type(exc).__name__
        )
    try:
        bundle = validate_dataset(
            datasets_dir, check_corpus=False, require_manifest=False
        )
    except DatasetError as exc:
        return EvaluationDatasetSnapshot(available=False, error_code=exc.code)
    except Exception:  # noqa: BLE001
        return EvaluationDatasetSnapshot(available=False, error_code="invalid_dataset")
    return EvaluationDatasetSnapshot(
        available=True,
        tests=list(bundle.tests),
        suite_hashes=dict(bundle.suite_hashes),
        manifest=dict(bundle.manifest or {}),
        dataset_sha256=bundle.dataset_sha256,
        expected_corpus_sha256=bundle.expected_corpus_sha256,
    )


def create_evaluation_runtime(
    config: ApiConfig,
    services: Any = None,
    *,
    store: EvaluationStore | None = None,
    runner: Runner | None = None,
    datasets_dir: Path | None = None,
) -> EvaluationRuntime:
    """Собрать storage/worker и применить миграции (recovery — в ``init``)."""
    eval_store = store or EvaluationStore(config.eval_db_path)
    eval_store.init()
    worker = EvaluationWorker(
        eval_store, config, runner=runner, services=services
    )
    return EvaluationRuntime(
        config,
        eval_store,
        worker,
        datasets_dir=datasets_dir or getattr(config, "eval_datasets_dir", None),
    )
