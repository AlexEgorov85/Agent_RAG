"""Упаковка контекста целыми чанками с явным манифестом включённых и пропущенных.

Пакер никогда не обрезает чанк: он либо рендерится полностью, либо опускается.
Так строки таблиц не теряют год/единицу измерения, без которых цитату нельзя
проверить, а все ``chunk_id`` в промпте остаются разрешимыми.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: Консервативный запас, когда токенизатор недоступен: русский текст обычно
#: даёт 2.5–3.5 символа на токен для словарей типа cl100k.
CHARS_PER_TOKEN = 3

_ENCODER: Any = None
_ENCODER_FAILED = False

#: Поля, которые генератор должен видеть для каждого упакованного чанка.
CONTEXT_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "year",
    "issuer",
    "page_pdf",
    "page_label",
    "section_path",
    "kind",
    "text",
    "html",
)


@dataclass(frozen=True)
class ContextManifest:
    """Запись аудита одного вызова упаковки.

    ``included`` и ``skipped`` сохраняют порядок встречи. ``duplicates`` — это
    повторно встреченные ID, которые уже были упакованы. ``used_chars`` равен
    ``len(PackedContext.text)`` и не превышает ``max_chars``; ``used_tokens`` и
    ``max_tokens`` отражают дополнительный токенный бюджет, если он задан.
    """

    included: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    used_chars: int = 0
    max_chars: int = 0
    used_tokens: int = 0
    max_tokens: int | None = None

    @property
    def included_chunk_ids(self) -> list[str]:
        """Псевдоним с явным названием для кода, избегающего сокращений."""
        return list(self.included)

    @property
    def skipped_chunk_ids(self) -> list[str]:
        """Псевдоним, соответствующий :attr:`included_chunk_ids`."""
        return list(self.skipped)


@dataclass(frozen=True)
class PackedContext:
    """Отрисованный текст контекста, структурные записи и манифест."""

    records: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    manifest: ContextManifest = field(default_factory=ContextManifest)

    @property
    def chunk_ids(self) -> list[str]:
        return [record["chunk_id"] for record in self.records]

    @property
    def context_chunk_ids(self) -> list[str]:
        """ID чанков, реально упакованных в ``text`` (в порядке появления).

        Совпадает с :attr:`ContextManifest.included_chunk_ids`, но доступно
        прямо на результате упаковки для ``diagnostics["context_chunk_ids"]``.
        """
        return [record["chunk_id"] for record in self.records]

    @property
    def skipped_chunk_ids(self) -> list[str]:
        """ID чанков, которые не влезли в бюджет и были опущены целиком."""
        return list(self.manifest.skipped)

    @property
    def context_stage(self) -> list[dict[str, Any]]:
        """Этап ``context`` для retrieval trace: ID, ранг, без score."""
        return [
            {"chunk_id": record["chunk_id"], "rank": position, "score": None}
            for position, record in enumerate(self.records, start=1)
        ]

    @property
    def evidence(self) -> dict[str, dict[str, Any]]:
        """Словарь свидетельств по ``chunk_id``, пригодный для генератора ответа."""
        return {record["chunk_id"]: record for record in self.records}


def _field(data: Mapping[str, Any], name: str) -> Any:
    if name in data and data[name] is not None:
        return data[name]
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _hit_parts(hit: Any) -> tuple[Any, Mapping[str, Any] | None]:
    """Возвращает ID чанка с уровня хита и словарь с полями чанка."""
    if isinstance(hit, Mapping):
        chunk_id = hit.get("chunk_id")
        inner = hit.get("chunk")
        if isinstance(inner, Mapping):
            return chunk_id, inner
        return chunk_id, hit
    chunk_id = getattr(hit, "chunk_id", None)
    inner = getattr(hit, "chunk", None)
    if isinstance(inner, Mapping):
        return chunk_id, inner
    # Поддерживаем и простые объекты, отдающие поля чанка как атрибуты.
    attributes = {
        name: getattr(hit, name) for name in CONTEXT_FIELDS if hasattr(hit, name)
    }
    return chunk_id, attributes or None


def _normalise_hit(hit: Any) -> tuple[str, dict[str, Any]] | None:
    chunk_id, inner = _hit_parts(hit)
    if inner is None:
        return None
    if not isinstance(chunk_id, str) or not chunk_id:
        chunk_id = _field(inner, "chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        return None

    def value(name: str) -> Any:
        found = _field(inner, name)
        if found is None and isinstance(hit, Mapping) and hit is not inner:
            found = _field(hit, name)
        return found

    raw_text = value("text")
    text = "" if raw_text is None else raw_text if isinstance(raw_text, str) else str(raw_text)
    html = _optional_str(value("html"))
    record = {
        "chunk_id": chunk_id,
        "doc_id": _optional_str(value("doc_id")),
        "year": _optional_int(value("year")),
        "issuer": _optional_str(value("issuer")),
        "page_pdf": _optional_int(value("page_pdf")),
        "page_label": _optional_str(value("page_label")),
        "section_path": _optional_str(value("section_path")),
        "kind": _optional_str(value("kind")),
        "text": text,
        "html": html if html else None,
    }
    return chunk_id, record


def _render(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)


def count_tokens(text: str) -> int:
    """Приблизительно считает число токенов в ``text``.

    Использует кэшированный энкодер ``tiktoken`` cl100k, если он доступен,
    иначе — консервативную эвристику по символам. Точный токенизатор целевой
    модели может немного отличаться, поэтому вызывающий код сохраняет лимит по
    символам как страховку; для планирования бюджета оценки достаточно, и она
    детерминированно выделяет один и тот же бюджет.
    """
    global _ENCODER, _ENCODER_FAILED
    if not text:
        return 0
    if _ENCODER is None and not _ENCODER_FAILED:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 — необязательная зависимость
            _ENCODER_FAILED = True
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def pack_context(
    hits: Iterable[Any],
    max_chars: int = 16_000,
    *,
    max_tokens: int | None = None,
    token_counter: Any = None,
) -> PackedContext:
    """Упаковывает целые чанки в текст длиной не более ``max_chars``.

    ``max_tokens`` задаёт дополнительный бюджет в токенах (например,
    ``context_window - max_output - prompt_reserve``). Чанк включается, только
    если он целиком помещается и в символьный, и в токенный бюджет, поэтому
    число чанков вычисляется по фактическому размеру, а не фиксируется
    заранее. ``token_counter`` позволяет подставить точный токенизатор модели.

    ``hits`` может содержать объекты
    :class:`~src.retrieval.hybrid_retriever.RetrievalHit`, словари-хиты с
    вложенным ``chunk`` или плоские словари чанков. Чанки дедуплицируются по
    ``chunk_id``. Чанк, который не влезает в остаток бюджета, пропускается и
    попадает в манифест, но цикл продолжает проверять более мелкие чанки.
    Табличный ``html`` остаётся в той же записи, что и исходный ``text``.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError("max_chars must be a non-negative integer")
    if max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be a non-negative integer or None")
        if max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer or None")
    counter = token_counter or count_tokens

    records: list[dict[str, Any]] = []
    included: list[str] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    text = ""

    for hit in hits or ():
        normalised = _normalise_hit(hit)
        if normalised is None:
            continue
        chunk_id, record = normalised
        if chunk_id in seen:
            if chunk_id not in duplicates:
                duplicates.append(chunk_id)
            continue
        seen.add(chunk_id)

        block = _render(record)
        candidate = block if not text else f"{text}\n\n{block}"
        if len(candidate) > max_chars:
            skipped.append(chunk_id)
            continue
        if max_tokens is not None and counter(candidate) > max_tokens:
            skipped.append(chunk_id)
            continue
        text = candidate
        records.append(record)
        included.append(chunk_id)

    used_chars = len(text)
    used_tokens = counter(text) if max_tokens is not None else 0
    manifest = ContextManifest(
        included=included,
        skipped=skipped,
        duplicates=duplicates,
        used_chars=used_chars,
        max_chars=max_chars,
        used_tokens=used_tokens,
        max_tokens=max_tokens,
    )
    return PackedContext(records=records, text=text, manifest=manifest)
