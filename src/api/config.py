"""Конфигурация HTTP adapter'а, читаемая из env только по запросу.

Импорт модуля не читает окружение и не создаёт каталогов: ``load_api_config``
вызывается в lifespan ASGI-приложения. Значения по умолчанию безопасны для
локального запуска и офлайн-тестов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "var" / "api.sqlite3"
DEFAULT_SOURCE_REGISTRY = PROJECT_ROOT / "data" / "source_registry.json"
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173",)
DEFAULT_EVAL_DB_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation.sqlite3"
DEFAULT_EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_EVAL_DATASETS_DIR = PROJECT_ROOT / "eval" / "datasets"


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None else str(value)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(
    env: Mapping[str, str], name: str, default: int, *, minimum: int = 1
) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_float(
    env: Mapping[str, str], name: str, default: float, *, minimum: float = 0.1
) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_origins(env: Mapping[str, str]) -> tuple[str, ...]:
    raw = env.get("API_CORS_ORIGINS")
    if raw is None:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(item.strip() for item in raw.split(",") if item.strip())
    return origins or DEFAULT_CORS_ORIGINS


@dataclass(frozen=True)
class ApiConfig:
    """Настройки adapter'а. Секретов здесь нет и быть не должно."""

    db_path: Path = DEFAULT_DB_PATH
    source_registry_path: Path = DEFAULT_SOURCE_REGISTRY
    corpus_name: str = "Локальный корпус"
    cookie_name: str = "rag_owner"
    cookie_secure: bool = False
    session_max_age_seconds: int = 365 * 24 * 3600
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    max_message_chars: int = 16_000
    max_event_bytes: int = 256 * 1024
    max_evidence_bytes: int = 1024 * 1024
    replay_buffer_events: int = 1_000
    replay_ttl_seconds: int = 600
    heartbeat_seconds: float = 15.0
    page_size_default: int = 50
    page_size_max: int = 200
    request_title_limit: int = 200
    rate_limit_per_minute: int = 120
    contract_version: str = "1"
    #: Same-origin SPA. По умолчанию выключено: API не обязан раздавать фронтенд,
    #: и dev-сервер Vite проксирует /api сам. Включается API_SERVE_SPA=1 и
    #: раздаёт собранный frontend/dist, если он существует.
    serve_spa: bool = False
    spa_dist_dir: Path = DEFAULT_FRONTEND_DIST
    # Evaluation / Benchmark (specs/evaluation-webui.md §12.1). Отключено по
    # умолчанию: раздел — developer-инструмент, включается EVAL_UI_ENABLED=1.
    #: Межпроцессный LLM-lock (CLI benchmark и API chat/evaluation).
    llm_lock_path: Path = Path("data/evaluation/llm.lock")
    eval_ui_enabled: bool = False
    eval_db_path: Path = DEFAULT_EVAL_DB_PATH
    eval_results_dir: Path = DEFAULT_EVAL_RESULTS_DIR
    eval_datasets_dir: Path = DEFAULT_EVAL_DATASETS_DIR
    eval_queue_size: int = 3
    eval_retention_runs: int = 50
    eval_turn_timeout_s: float = 300.0
    eval_run_timeout_s: float = 7_200.0
    #: Offline-режим для локальной проверки инфраструктуры: артефакты помечаются
    #: fake_service и не выдаются за качество реального агента.
    eval_fake_service: bool = False
    eval_judge_enabled: bool = False

    def capabilities_modes(self) -> tuple[str, ...]:
        """Retrieval-режимы, объявленные adapter'ом для benchmark-контура."""
        return ("hybrid", "dense", "keyword")

    def judge_available(self) -> bool:
        """LLM-судья по умолчанию не подключён (§8.3)."""
        return bool(self.eval_judge_enabled)

    def eval_cancel_available(self) -> bool:
        """Доступна ли отмена evaluation-прогонов."""
        return bool(self.eval_ui_enabled)

    def with_overrides(self, **kwargs: object) -> "ApiConfig":
        """Копия конфигурации для тестов без повторного чтения env."""
        return replace(self, **kwargs)


def load_api_config(env: Mapping[str, str] | None = None) -> ApiConfig:
    """Собрать конфигурацию adapter'а из переменных окружения.

    Неизвестные/некорректные значения мягко заменяются значениями по умолчанию:
    ошибка конфигурации не должна блокировать запуск health-проверки.
    """
    source = os.environ if env is None else env
    return ApiConfig(
        db_path=Path(_env(source, "API_DB_PATH", str(DEFAULT_DB_PATH))),
        source_registry_path=Path(
            _env(source, "API_SOURCE_REGISTRY_PATH", str(DEFAULT_SOURCE_REGISTRY))
        ),
        corpus_name=_env(source, "API_CORPUS_NAME", "Локальный корпус"),
        cookie_name=_env(source, "API_OWNER_COOKIE", "rag_owner"),
        cookie_secure=_env_bool(source, "API_COOKIE_SECURE", False),
        session_max_age_seconds=_env_int(
            source, "API_SESSION_MAX_AGE", 365 * 24 * 3600
        ),
        cors_origins=_env_origins(source),
        max_message_chars=_env_int(source, "API_MAX_MESSAGE_CHARS", 16_000),
        max_event_bytes=_env_int(source, "API_MAX_EVENT_BYTES", 256 * 1024),
        max_evidence_bytes=_env_int(source, "API_MAX_EVIDENCE_BYTES", 1024 * 1024),
        replay_buffer_events=_env_int(source, "API_REPLAY_BUFFER_EVENTS", 1_000),
        replay_ttl_seconds=_env_int(source, "API_REPLAY_TTL_SECONDS", 600),
        heartbeat_seconds=_env_float(source, "API_HEARTBEAT_SECONDS", 15.0),
        page_size_default=_env_int(source, "API_PAGE_SIZE_DEFAULT", 50),
        page_size_max=_env_int(source, "API_PAGE_SIZE_MAX", 200),
        rate_limit_per_minute=_env_int(source, "API_RATE_LIMIT_PER_MINUTE", 120),
        serve_spa=_env_bool(source, "API_SERVE_SPA", False),
        spa_dist_dir=Path(
            _env(source, "API_SPA_DIST_DIR", str(DEFAULT_FRONTEND_DIST))
        ),
        llm_lock_path=Path(
            _env(source, "LLM_LOCK_PATH", "data/evaluation/llm.lock")
        ),
        eval_ui_enabled=_env_bool(source, "EVAL_UI_ENABLED", False),
        eval_db_path=Path(
            _env(source, "EVAL_DB_PATH", str(DEFAULT_EVAL_DB_PATH))
        ),
        eval_results_dir=Path(
            _env(source, "EVAL_RESULTS_DIR", str(DEFAULT_EVAL_RESULTS_DIR))
        ),
        eval_datasets_dir=Path(
            _env(source, "EVAL_DATASETS_DIR", str(DEFAULT_EVAL_DATASETS_DIR))
        ),
        eval_queue_size=_env_int(source, "EVAL_QUEUE_SIZE", 3),
        eval_retention_runs=_env_int(source, "EVAL_RETENTION_RUNS", 50),
        eval_turn_timeout_s=_env_float(source, "EVAL_TURN_TIMEOUT_S", 300.0),
        eval_run_timeout_s=_env_float(source, "EVAL_RUN_TIMEOUT_S", 7_200.0),
        eval_fake_service=_env_bool(source, "EVAL_FAKE_SERVICE", False),
        eval_judge_enabled=_env_bool(source, "EVAL_JUDGE_ENABLED", False),
    )
