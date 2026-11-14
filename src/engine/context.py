"""Whole-chunk context packing with an explicit included/skipped manifest.

The packer never trims a chunk: a chunk is either rendered completely or
omitted. This protects table rows from losing the year/unit that makes a
citation verifiable, and keeps every chunk ID in the prompt resolvable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: Fields that must be visible to the generator for every packed chunk.
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
    """Audit record of one packing call.

    ``included`` and ``skipped`` keep encounter order. ``duplicates`` lists
    repeated chunk IDs that were already seen and therefore packed once.
    ``used_chars`` is exactly ``len(PackedContext.text)`` and never exceeds
    ``max_chars``.
    """

    included: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    used_chars: int = 0
    max_chars: int = 0

    @property
    def included_chunk_ids(self) -> list[str]:
        """Alias with an explicit unit for callers that avoid abbreviations."""
        return list(self.included)

    @property
    def skipped_chunk_ids(self) -> list[str]:
        """Alias matching :attr:`included_chunk_ids`."""
        return list(self.skipped)


@dataclass(frozen=True)
class PackedContext:
    """Rendered context text plus structured records and its manifest."""

    records: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    manifest: ContextManifest = field(default_factory=ContextManifest)

    @property
    def chunk_ids(self) -> list[str]:
        return [record["chunk_id"] for record in self.records]

    @property
    def evidence(self) -> dict[str, dict[str, Any]]:
        """Evidence mapping keyed by chunk ID, suitable for an answerer."""
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
    """Return the hit-level chunk ID and the mapping holding chunk fields."""
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
    # Tolerate plain objects exposing the chunk contract as attributes.
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


def pack_context(
    hits: Iterable[Any], max_chars: int = 16_000
) -> PackedContext:
    """Pack whole chunks into at most ``max_chars`` rendered characters.

    ``hits`` may contain :class:`~src.retrieval.hybrid_retriever.RetrievalHit`
    objects, mapping hits with a nested ``chunk``, or flat chunk mappings.
    Chunks are deduplicated by ``chunk_id``. A chunk whose rendered form does
    not fit the remaining budget is skipped and recorded in the manifest; the
    loop still considers later, smaller chunks. Table ``html`` stays in the
    same record as its source ``text``.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError("max_chars must be a non-negative integer")
    if max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")

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
        text = candidate
        records.append(record)
        included.append(chunk_id)

    used_chars = len(text)
    manifest = ContextManifest(
        included=included,
        skipped=skipped,
        duplicates=duplicates,
        used_chars=used_chars,
        max_chars=max_chars,
    )
    return PackedContext(records=records, text=text, manifest=manifest)
