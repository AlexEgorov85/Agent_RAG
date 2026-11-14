"""Course embeddings adapter for OpenAI-compatible Cloud.ru endpoints.

Only queries receive the retrieval prefix; document chunks are embedded as-is.
The adapter reorders indexed responses and rejects malformed vectors so a bad
provider response cannot poison the cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _response_data(response: Any) -> list[Any]:
    data = response.get("data") if isinstance(response, Mapping) else getattr(response, "data", None)
    if not isinstance(data, list):
        raise ValueError("embeddings response must contain a data list")
    return data


def _item_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class CourseEmbeddings:
    """Embed documents unchanged and prefix queries for retrieval models."""

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
        """Return document vectors without a retrieval prefix."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Return one query vector with the configured retrieval prefix."""
        if not isinstance(text, str):
            raise TypeError("query must be a string")
        vectors = self._embed([self.query_prefix + text])
        return vectors[0]


def normalize(vector: Sequence[float]) -> list[float]:
    """Normalize a vector to unit length; zero vectors stay zero."""
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("vector must contain finite numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]
