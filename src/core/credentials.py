"""Env-версия курсового ``get_credentials`` (Занятие 1), без Colab/getpass."""

from __future__ import annotations

import os

from src.core.rag_config import RAGSettings, load_rag_settings


def get_credentials(settings: RAGSettings | None = None) -> str:
    """Прочитать ключ Cloud.ru из настроек или переменных окружения.

    Args:
        settings: Настройки; None — ``.env``/env.

    Returns:
        Непустой API-ключ.

    Raises:
        RuntimeError: ключ не задан.
    """
    settings = settings or load_rag_settings()
    value = (settings.llm_api_key or os.environ.get("CLOUD_RU_API_KEY")
             or os.environ.get("LLM_API_KEY") or "").strip()
    if not value:
        raise RuntimeError(
            "Не задан LLM_API_KEY (Cloud.ru). Добавьте его в .env или "
            "переменную окружения LLM_API_KEY.")
    return value
