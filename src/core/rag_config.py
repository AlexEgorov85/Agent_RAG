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
    router_model: str = Field(
        default="",
        validation_alias=AliasChoices("ROUTER_MODEL", "LLM_ROUTER_MODEL"))

    top_k: int = 5
    candidates: int = 20
    rerank_pool: int = 10
    context_chars: int = 16000

    #: Безопасные бюджеты агента (``spec_agent.md`` §3.1): не более 4 tool
    #: calls и 2 поисков за turn. Это защитный механизм, а не способ
    #: управления исследованием; значения по умолчанию совпадают с
    #: константами ``src.agent.agent_graph``/``src.agent.selector``.
    max_tool_calls: int = Field(default=4, ge=1)
    max_search_calls: int = Field(default=2, ge=1)
    #: Лимиты TemporalSearch (per-year top-k и число рассматриваемых лет).
    per_year_k: int = Field(default=2, ge=1)
    max_years: int = Field(default=12, ge=1)
    #: Сколько документов/лет структурного анализа просматривать за прогон.
    max_structure_docs: int = Field(default=5, ge=1)

    #: Флаги и лимиты retrieval/агента (``spec_agent_enhancement.md`` §4.8).
    #: ``max_queries`` — потолок формулировок одного поиска (original + rewrites).
    max_queries: int = Field(default=3, ge=1)
    #: Мягкое разнообразие выдачи: лимиты на документ/раздел не снижают recall,
    #: а лишь отодвигают дубли (третий проход добирает top-k).
    diversity_enabled: bool = True
    max_per_doc: int = Field(default=2, ge=1)
    max_per_section: int = Field(default=1, ge=1)
    #: Coverage управляет добивкой пропусков, verification — проверкой ответа.
    coverage_enabled: bool = True
    max_coverage_searches: int = Field(default=1, ge=1)
    verification_enabled: bool = True

    #: Флаги LLM-подсистем. ``False`` — детерминированный fallback без сети.
    llm_router_enabled: bool = True
    llm_rewrite_enabled: bool = True
    llm_structure_verify: bool = True
    rerank_enabled: bool = True
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
    #: Постоянный FAISS-индекс эмбеддингов. ``False`` — старый in-memory
    #: ``DenseIndex`` (полезно для минимальных окружений без faiss-cpu).
    faiss_enabled: bool = True
    #: Сохранять собранный FAISS-индекс в ``indexes_dir``.
    faiss_persist: bool = True
    indexes_dir: Path = Path("data/indexes")

    def client_kwargs(self) -> dict:
        """Параметры OpenAI-совместимого клиента Cloud.ru (курс, Занятие 1)."""
        return {"base_url": self.base_url, "timeout": 90, "max_retries": 2}

    def temporal_search_limits(self) -> dict[str, int]:
        """Лимиты для ``TemporalSearch`` в виде kwargs конструктора."""
        return {"per_year_k": self.per_year_k, "max_years": self.max_years}


def load_rag_settings() -> RAGSettings:
    """Загрузить настройки из env/.env."""
    return RAGSettings()
