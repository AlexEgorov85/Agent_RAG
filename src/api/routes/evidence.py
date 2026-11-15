"""Endpoint источника: chunk из существующего ``ChunkBackend`` ядра."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response

from src.api.config import ApiConfig
from src.api.errors import ApiError, not_found
from src.api.schemas import EvidenceDetail
from src.api.services import Services

from .deps import Owner, get_config, get_services

router = APIRouter()


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _section_path(chunk: dict[str, Any]) -> list[str] | None:
    breadcrumbs = chunk.get("breadcrumbs")
    if isinstance(breadcrumbs, list):
        segments = [
            str(item).strip()
            for item in breadcrumbs
            if isinstance(item, str) and item.strip()
        ]
        if segments:
            return segments
    raw = chunk.get("section_path")
    if isinstance(raw, str):
        segments = [item.strip() for item in raw.split("/") if item.strip()]
        return segments or None
    return None


@router.get(
    "/evidence/{chunk_id}",
    response_model=EvidenceDetail,
    tags=["evidence"],
)
def get_evidence(
    chunk_id: str,
    owner: Owner,
    response: Response,
    config: ApiConfig = Depends(get_config),
    services: Services = Depends(get_services),
) -> EvidenceDetail:
    chunk = services.chunk_metadata(chunk_id)
    if chunk is None:
        raise not_found("Источник не найден.")
    known_id = chunk.get("chunk_id")
    if known_id is not None and str(known_id) != chunk_id:
        raise not_found("Источник не найден.")

    text = chunk.get("text")
    text = text if isinstance(text, str) else ""
    kind = chunk.get("kind")
    html = chunk.get("html") if kind == "table" else None
    html = html if isinstance(html, str) else None
    size = len(text.encode("utf-8")) + (
        len(html.encode("utf-8")) if html else 0
    )
    if size > config.max_evidence_bytes:
        raise ApiError(
            413,
            "payload_too_large",
            "Источник превышает допустимый размер ответа.",
        )

    year = chunk.get("year")
    year = year if isinstance(year, int) and not isinstance(year, bool) else None
    issuer = chunk.get("issuer")
    title = chunk.get("title")
    response.headers["Cache-Control"] = "private, max-age=300"
    return EvidenceDetail(
        chunk_id=str(known_id or chunk_id),
        document_id=(
            str(chunk["doc_id"]) if chunk.get("doc_id") is not None else None
        ),
        document_title=str(title) if title else None,
        issuer=str(issuer) if issuer else None,
        year=year,
        section_path=_section_path(chunk),
        page_start=_positive_int(chunk.get("page_pdf")),
        page_end=_positive_int(chunk.get("page_end")),
        page_label=(
            str(chunk["page_label"])
            if chunk.get("page_label") is not None
            else None
        ),
        source_type=kind if kind in {"text", "table"} else None,
        text=text,
        html=html,
    )
