"""Постоянный FAISS-индекс эмбеддингов чанков.

Индекс строится по нормализованным векторам и использует inner product,
что эквивалентно косинусному сходству. Артефакт состоит из двух файлов::

    data/indexes/{index_fingerprint}/index.faiss
    data/indexes/{index_fingerprint}/manifest.json

``index_fingerprint`` совпадает с fingerprint кэша эмбеддингов: он учитывает
текст чанков, модель, префикс запроса и SHA-256 исходных PDF. Загрузка
проверяет схему, размерность, количество и порядок ``chunk_id``; при любом
расхождении индекс считается непригодным и вызывающий код пересобирает его.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.dense_index import _chunk_id, _normalize

INDEX_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.json"
INDEX_TYPE = "faiss.IndexFlatIP"
METRIC = "cosine"

Eligibility = Callable[[Mapping[str, Any]], bool] | Iterable[int] | None


class FaissIndexError(RuntimeError):
    """Артефакт FAISS отсутствует, повреждён или не соответствует корпусу."""


def _faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise FaissIndexError(
            "faiss-cpu не установлен; установите `pip install faiss-cpu`"
        ) from exc
    return faiss


def _ordered_vectors(
    chunks: Sequence[Mapping[str, Any]],
    vectors: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Приводит векторы к порядку чанков и проверяет контракт."""
    if isinstance(vectors, Mapping):
        if len(vectors) != len(chunks):
            raise ValueError("vector count must match chunk count")
        try:
            ordered = [vectors[_chunk_id(chunk)] for chunk in chunks]
        except KeyError as exc:
            raise ValueError("vector count must match chunk count") from exc
    else:
        ordered = list(vectors)
        if len(ordered) != len(chunks):
            raise ValueError("vector count must match chunk count")

    normalized: list[list[float]] = []
    for values in ordered:
        vector = _normalize(values)
        if normalized and len(vector) != len(normalized[0]):
            raise ValueError("vector dimension mismatch")
        normalized.append(vector)
    return normalized


def _eligible_positions(
    chunks: Sequence[Mapping[str, Any]], eligible: Eligibility
) -> list[int] | None:
    if eligible is None:
        return None
    if callable(eligible):
        return [
            index for index, chunk in enumerate(chunks) if eligible(chunk)
        ]
    allowed = {int(index) for index in eligible}
    return [index for index in sorted(allowed) if 0 <= index < len(chunks)]


def _source_hashes(chunks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "")
        sha = chunk.get("source_sha256")
        if doc_id and isinstance(sha, str):
            hashes.setdefault(doc_id, sha)
    return dict(sorted(hashes.items()))


class FaissDenseIndex:
    """FAISS-индекс с тем же контрактом ``search``, что и ``DenseIndex``."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        vectors: Sequence[Sequence[float]] | Mapping[str, Sequence[float]] | None,
        *,
        index: Any | None = None,
    ) -> None:
        self.chunks: list[Mapping[str, Any]] = list(chunks)
        self.chunk_ids: list[str] = [_chunk_id(chunk) for chunk in self.chunks]
        if vectors is None:
            if index is None:
                raise ValueError("vectors are required to build a FAISS index")
            ordered: list[list[float]] = []
            self.dimension = int(getattr(index, "d", 0))
        else:
            ordered = _ordered_vectors(self.chunks, vectors)
            self.dimension = len(ordered[0]) if ordered else 0

        if index is None:
            if not ordered:
                self._index = None
            else:
                faiss = _faiss()
                matrix = self._matrix(ordered)
                flat = faiss.IndexFlatIP(self.dimension)
                flat.add(matrix)
                self._index = flat
        else:
            self._index = index
            if self.chunk_ids and getattr(index, "ntotal", 0) != len(self.chunk_ids):
                raise FaissIndexError("index size does not match chunk count")
            if self.chunk_ids and getattr(index, "d", 0) != self.dimension:
                raise FaissIndexError("index dimension does not match vectors")

    @staticmethod
    def _matrix(vectors: Sequence[Sequence[float]]) -> Any:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - numpy есть в зависимостях
            raise FaissIndexError("numpy is required for the FAISS index") from exc
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError("vectors must be a two-dimensional array")
        if not bool(np.isfinite(matrix).all()):
            raise ValueError("vectors must contain finite numbers")
        return matrix

    def _search_raw(self, query_vector: Sequence[float], k: int) -> tuple[list[float], list[int]]:
        if self._index is None:
            return [], []
        query = _normalize(query_vector)
        if len(query) != self.dimension:
            raise ValueError("query dimension does not match index dimension")
        matrix = self._matrix([query])
        scores, ids = self._index.search(matrix, int(k))
        return [float(value) for value in scores[0]], [int(value) for value in ids[0]]

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
        eligible: Eligibility = None,
    ) -> list[dict[str, Any]]:
        """Возвращает до ``k`` чанков по косинусному сходству.

        Если задан ``eligible``, фильтр применяется до обрезки до ``k``:
        FAISS возвращает весь индекс, после чего позиции фильтруются.
        """
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks:
            return []

        positions = _eligible_positions(self.chunks, eligible)
        search_k = len(self.chunks) if positions is not None else min(k, len(self.chunks))
        scores, ids = self._search_raw(query_vector, search_k)
        allowed = set(positions) if positions is not None else None
        scored: list[tuple[float, str, int]] = []
        for score, index in zip(scores, ids, strict=True):
            if index < 0 or index >= len(self.chunks):
                continue
            if allowed is not None and index not in allowed:
                continue
            scored.append((score, self.chunk_ids[index], index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**dict(self.chunks[index]), "score": score}
            for score, _chunk_id_value, index in scored[:k]
        ]

    def save(
        self,
        directory: Path | str,
        *,
        fingerprint: str,
        model: str,
        query_prefix: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Атомарно сохранить FAISS-индекс и manifest в ``directory``."""
        if self._index is None:
            raise FaissIndexError("cannot save an empty FAISS index")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss = _faiss()
        index_path = target / INDEX_FILENAME
        manifest_path = target / MANIFEST_FILENAME
        index_tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        try:
            # ``faiss.write_index`` принимает const char* и не открывает пути
            # с не-ASCII символами (например, кириллический профиль Windows).
            # Сериализуем индекс в bytes и пишем через pathlib.
            index_tmp.write_bytes(faiss.serialize_index(self._index).tobytes())
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "index_type": INDEX_TYPE,
                "metric": METRIC,
                "dimension": self.dimension,
                "chunk_count": len(self.chunk_ids),
                "chunk_ids": self.chunk_ids,
                "corpus_fingerprint": fingerprint,
                "embedding_model": model,
                "query_prefix": query_prefix,
                "source_hashes": _source_hashes(self.chunks),
                "created_at": datetime.now(timezone.utc).isoformat(),
                **dict(extra or {}),
            }
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            index_tmp.replace(index_path)
            manifest_tmp.replace(manifest_path)
        finally:
            for temporary in (index_tmp, manifest_tmp):
                if temporary.exists():
                    temporary.unlink()
        return index_path

    @classmethod
    def load(
        cls,
        directory: Path | str,
        chunks: Sequence[Mapping[str, Any]],
        *,
        fingerprint: str | None = None,
        model: str | None = None,
        strict: bool = False,
    ) -> "FaissDenseIndex | None":
        """Загрузить индекс, если manifest соответствует корпусу.

        При несовпадении возвращает ``None`` (или бросает ``FaissIndexError``
        при ``strict=True``), чтобы вызывающий код мог безопасно пересобрать
        индекс вместо падения.
        """
        target = Path(directory)
        index_path = target / INDEX_FILENAME
        manifest_path = target / MANIFEST_FILENAME
        try:
            if not index_path.is_file() or not manifest_path.is_file():
                if strict:
                    raise FaissIndexError("index artifact is missing")
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
                raise FaissIndexError("unsupported index schema")
            if manifest.get("metric") != METRIC or manifest.get("index_type") != INDEX_TYPE:
                raise FaissIndexError("unsupported index type or metric")
            chunk_ids = [_chunk_id(chunk) for chunk in chunks]
            if manifest.get("chunk_ids") != chunk_ids:
                raise FaissIndexError("index chunk_ids do not match corpus")
            if manifest.get("chunk_count") != len(chunk_ids):
                raise FaissIndexError("index chunk count does not match corpus")
            if fingerprint is not None and manifest.get("corpus_fingerprint") != fingerprint:
                raise FaissIndexError("index fingerprint does not match corpus")
            if model is not None and manifest.get("embedding_model") != model:
                raise FaissIndexError("index embedding model does not match")
            faiss = _faiss()
            try:
                import numpy as np
            except Exception as exc:  # pragma: no cover - numpy есть в зависимостях
                raise FaissIndexError("numpy is required for the FAISS index") from exc
            payload = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
            if payload.size == 0:
                raise FaissIndexError("index artifact is empty")
            raw = faiss.deserialize_index(payload)
            if raw.ntotal != len(chunk_ids):
                raise FaissIndexError("index size does not match corpus")
            dimension = manifest.get("dimension")
            if not isinstance(dimension, int) or dimension != raw.d:
                raise FaissIndexError("index dimension does not match manifest")
            return cls(chunks, None, index=raw)
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            json.JSONDecodeError,
            FaissIndexError,
        ) as exc:
            if strict:
                if isinstance(exc, FaissIndexError):
                    raise
                raise FaissIndexError(str(exc)) from exc
            warnings.warn(f"faiss_index_unavailable: {exc}", RuntimeWarning, stacklevel=2)
            return None

    @classmethod
    def load_or_build(
        cls,
        directory: Path | str,
        chunks: Sequence[Mapping[str, Any]],
        vectors: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
        *,
        fingerprint: str,
        model: str,
        query_prefix: str = "",
        persist: bool = True,
    ) -> "FaissDenseIndex":
        """Загрузить готовый индекс или собрать и (опционально) сохранить новый."""
        loaded = cls.load(
            directory, chunks, fingerprint=fingerprint, model=model, strict=False
        )
        if loaded is not None:
            return loaded
        index = cls(chunks, vectors)
        if persist:
            try:
                index.save(
                    directory,
                    fingerprint=fingerprint,
                    model=model,
                    query_prefix=query_prefix,
                )
            except (OSError, FaissIndexError) as exc:
                warnings.warn(
                    f"faiss_index_save_failed: {exc}", RuntimeWarning, stacklevel=2
                )
        return index
