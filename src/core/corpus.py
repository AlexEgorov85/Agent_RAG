"""Загрузка чанков из каталога артефактов (dev/канонический корпус)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List


def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Прочитать все ``*_chunks.json`` из каталога и отсортировать."""
    chunks: List[Dict[str, Any]] = []
    for path in sorted(Path(chunks_dir).glob("*_chunks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            chunks.extend(data)
    chunks.sort(key=lambda chunk: (chunk.get("doc_id", ""),
                                   int(chunk.get("page_pdf") or 0),
                                   chunk.get("chunk_id", "")))
    return chunks


#: Поля реестра, которые переносятся в чанк, если он их не содержит.
REGISTRY_CHUNK_FIELDS: tuple[str, ...] = (
    "issuer",
    "country",
    "source_status",
    "report_period",
)


def enrich_chunks_with_registry(
    chunks: Iterable[Dict[str, Any]],
    records: Iterable[Mapping[str, Any]],
) -> int:
    """Дополнить чанки метаданными реестра источников по ``doc_id``.

    Чанки, собранные из PDF, не содержат ``issuer``/``country``: эти поля
    задаются только реестром (``spec_target.md`` §3.1), поэтому фильтр
    ``issuers`` без такого обогащения отбрасывал бы весь корпус.

    Уже присутствующие в чанке значения не перезаписываются: метаданные
    артефакта считаются более точными, реестр закрывает только пропуски.

    Args:
        chunks: Чанки, изменяются на месте.
        records: Записи реестра (``SourceRecord`` или словари с ``doc_id``).

    Returns:
        Число чанков, которым добавлено хотя бы одно поле.
    """
    by_doc: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping):
            data: Mapping[str, Any] = record
        else:
            # SourceRecord — dataclass; slots-вариант читается по именам полей.
            data = getattr(record, "__dict__", None) or {
                name: getattr(record, name, None)
                for name in REGISTRY_CHUNK_FIELDS
            }
        doc_id = data.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        by_doc[doc_id] = data

    enriched = 0
    for chunk in chunks:
        record = by_doc.get(chunk.get("doc_id"))
        if record is None:
            continue
        touched = False
        for name in REGISTRY_CHUNK_FIELDS:
            value = record.get(name)
            if value in (None, "") or chunk.get(name) not in (None, ""):
                continue
            chunk[name] = value
            touched = True
        if touched:
            enriched += 1
    return enriched
