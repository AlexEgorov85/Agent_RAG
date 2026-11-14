"""Курсовой клиент GigaChat-3-Pro через Chat Completions Cloud.ru."""

from __future__ import annotations

from src.core.credentials import get_credentials
from src.core.rag_config import RAGSettings, load_rag_settings


def make_chat(settings: RAGSettings | None = None,
              credentials: str | None = None):
    """Создать LangChain-клиент GigaChat с курсовыми параметрами."""
    from langchain_openai import ChatOpenAI

    settings = settings or load_rag_settings()
    return ChatOpenAI(
        api_key=credentials or get_credentials(settings),
        model=settings.model,
        temperature=0.1,
        max_tokens=1400,
        use_responses_api=False,
        stream_usage=False,
        disabled_params={"parallel_tool_calls": None},
        **settings.client_kwargs(),
    )
