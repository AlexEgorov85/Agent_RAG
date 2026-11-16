"""Адаптер эмбеддингов для OpenAI-совместимого API Cloud.ru.

Префикс поиска получает только запрос, документы встраиваются как есть.
Адаптер переупорядочивает ответ по индексам и отбраковывает битые векторы,
чтобы плохой ответ провайдера не испортил кэш.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any


def make_embeddings_client(
    base_url: str | None = None,
    api_key: str | None = None,
    *,
    fallback_client: Any = None,
) -> Any:
    """Вернуть OpenAI-клиент для эмбеддингов.

    Если задан отдельный провайдер (``EMBED_BASE_URL``/``EMBED_API_KEY``),
    создаётся собственный клиент, а LLM остаётся на ``LLM_BASE_URL``. Без
    отдельного провайдера используется ``fallback_client`` или LLM-настройки.
    """
    base = (base_url or os.getenv("EMBED_BASE_URL")
            or os.getenv("EMBEDDINGS_BASE_URL") or "").strip()
    key = (api_key or os.getenv("EMBED_API_KEY")
           or os.getenv("NEURALDEEP_API_KEY") or "").strip()
    if not base and not key:
        if fallback_client is not None:
            return fallback_client
        base = (os.getenv("LLM_BASE_URL") or "").strip()
        key = (os.getenv("LLM_API_KEY") or "").strip()
        if not base or not key:
            raise RuntimeError(
                "Не заданы EMBED_BASE_URL/EMBED_API_KEY (или LLM_BASE_URL/"
                "LLM_API_KEY). Добавьте их в .env.")
    if not base:
        raise RuntimeError(
            "Не задан EMBED_BASE_URL для отдельного провайдера эмбеддингов.")
    if not key:
        raise RuntimeError(
            "Не задан EMBED_API_KEY для отдельного провайдера эмбеддингов.")
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base, timeout=90, max_retries=2)


def _response_data(response: Any) -> list[Any]:
    data = response.get("data") if isinstance(response, Mapping) else getattr(response, "data", None)
    if not isinstance(data, list):
        raise ValueError("embeddings response must contain a data list")
    return data


def _item_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class ReportEmbeddings:
    """Встраивает документы без префикса, а запросы — с префиксом поиска."""

    def __init__(self, client: Any, model: str, query_prefix: str = "query:") -> None:
        if client is None:
            raise ValueError("client is required")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(query_prefix, str):
            raise ValueError("query_prefix must be a string")
        self.client = client
        self.model = model
        self.query_prefix = query_prefix

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model, input=list(texts))
        data = _response_data(response)
        if len(data) != len(texts):
            raise ValueError("embeddings response count does not match input")
        ordered: list[tuple[int, list[float]]] = []
        for fallback_index, item in enumerate(data):
            index = _item_field(item, "index", fallback_index)
            embedding = _item_field(item, "embedding")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("embedding index must be an integer")
            if not isinstance(embedding, (list, tuple)):
                raise ValueError("embedding must be a numeric vector")
            vector: list[float] = []
            for value in embedding:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("embedding contains a non-numeric value")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("embedding contains a non-finite value")
                vector.append(number)
            if not vector:
                raise ValueError("embedding vector must not be empty")
            ordered.append((index, vector))
        ordered.sort(key=lambda item: item[0])
        if [index for index, _ in ordered] != list(range(len(texts))):
            raise ValueError("embedding indices are not a complete permutation")
        dimensions = {len(vector) for _, vector in ordered}
        if len(dimensions) != 1:
            raise ValueError("embedding dimensions differ")
        return [vector for _, vector in ordered]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Возвращает векторы документов без префикса поиска."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Возвращает один вектор запроса с настроенным префиксом поиска."""
        if not isinstance(text, str):
            raise TypeError("query must be a string")
        vectors = self._embed([self.query_prefix + text])
        return vectors[0]
