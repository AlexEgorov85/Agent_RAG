"""Конфигурация retrieval/генерации (курсовые Settings + наши лимиты)."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Курсовая инструкция для Giga-Embeddings-instruct (Занятие 3).
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a query, retrieve relevant passages\nQuery: "
)


class RAGSettings(BaseSettings):
    """Настройки Cloud.ru, retrieval-конвейера и dev-корпуса."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "CLOUD_RU_API_KEY"))
    base_url: str = Field(
        default="https://foundation-models.api.cloud.ru/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "BASE_URL"))
    model: str = Field(
        default="GigaChat/GigaChat-3-Pro",
        validation_alias=AliasChoices("LLM_MODEL", "MODEL"))
    embeddings_model: str = Field(
        default="ai-sage/Giga-Embeddings-instruct-3B",
        validation_alias=AliasChoices("EMBED_MODEL", "EMBEDDINGS_MODEL"))
    query_prefix: str = Field(
        default=DEFAULT_QUERY_PREFIX,
        validation_alias=AliasChoices("EMBED_QUERY_PREFIX", "QUERY_PREFIX"))
    rerank_model: str = Field(
        default="ai-sage/GigaChat3-10B-A1.8B",
        validation_alias=AliasChoices("RERANKER_MODEL", "RERANK_MODEL"))

    top_k: int = 5
    candidates: int = 20
    rerank_pool: int = 10
    context_chars: int = 16000
    #: Полное окно модели в токенах (для Cloud.ru GigaChat-3-Pro — 32k).
    context_window_tokens: int = 32768
    #: Резерв под ответ и системный промпт; контекст получает остаток.
    output_tokens: int = 2048
    context_reserve_tokens: int = 2048

    @property
    def context_tokens(self) -> int:
        """Сколько токенов реально доступно под упакованные чанки."""
        return max(
            0,
            self.context_window_tokens
            - self.output_tokens
            - self.context_reserve_tokens,
        )

    chunks_dir: Path = Path("data/dev/chunks")
    cache_dir: Path = Path("data/dev/cache")

    def client_kwargs(self) -> dict:
        """Параметры OpenAI-совместимого клиента Cloud.ru (курс, Занятие 1)."""
        return {"base_url": self.base_url, "timeout": 90, "max_retries": 2}


def load_rag_settings() -> RAGSettings:
    """Загрузить настройки из env/.env."""
    return RAGSettings()
