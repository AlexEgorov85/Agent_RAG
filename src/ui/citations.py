"""Чистые помощники отрисовки цитат без зависимости от Streamlit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def citation_metadata(citation: Any) -> dict[str, Any]:
    """Возвращает только метаданные цитаты, определённые контрактом Answer."""
    metadata = {
        "chunk_id": _value(citation, "chunk_id"),
        "doc_id": _value(citation, "doc_id"),
        "year": _value(citation, "year"),
        "issuer": _value(citation, "issuer"),
        "section_path": _value(citation, "section_path"),
        "page_pdf": _value(citation, "page_pdf"),
        "page_label": _value(citation, "page_label"),
    }
    return {key: value for key, value in metadata.items() if value is not None and value != ""}


def citation_quote(citation: Any) -> str | None:
    """Возвращает короткую цитату из самой ссылки, не подставляя чанк по догадке."""
    quote = _value(citation, "quote")
    if not isinstance(quote, str):
        return None
    quote = quote.strip()
    return quote or None


def render_citations(st: Any, citations: Any) -> None:
    """Отрисовывает цитаты Answer как метаданные и раскрываемые карточки с цитатой.

    Контракт сервиса агента даёт короткие фрагменты, а не резолвер хранилища
    чанков. Помощник намеренно не придумывает и не загружает полный текст чанка.
    """
    if not citations:
        return
    st.subheader("Источники")
    for index, citation in enumerate(citations, start=1):
        metadata = citation_metadata(citation)
        title_parts = [
            str(metadata.get("year", "")),
            str(metadata.get("issuer", "")),
            str(metadata.get("doc_id", "")),
        ]
        title = " · ".join(part for part in title_parts if part) or f"Источник {index}"
        with st.expander(title):
            for label, key in (
                ("Документ", "doc_id"),
                ("Год", "year"),
                ("Issuer", "issuer"),
                ("Раздел", "section_path"),
                ("PDF-страница", "page_pdf"),
                ("Метка страницы", "page_label"),
                ("Chunk ID", "chunk_id"),
            ):
                if key in metadata:
                    st.caption(f"{label}: {metadata[key]}")
            quote = citation_quote(citation)
            if quote:
                st.text(quote)
            else:
                st.caption("Цитата не предоставлена AgentService.")
