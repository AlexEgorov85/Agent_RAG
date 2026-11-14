"""Pure helpers for displaying citations without requiring Streamlit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def citation_metadata(citation: Any) -> dict[str, Any]:
    """Return only the citation metadata defined by the Answer contract."""
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
    """Return the short quote supplied on the citation, never an inferred chunk."""
    quote = _value(citation, "quote")
    if not isinstance(quote, str):
        return None
    quote = quote.strip()
    return quote or None


def render_citations(st: Any, citations: Any) -> None:
    """Render Answer citations as metadata and expandable quote cards.

    The agent service's contract supplies quote snippets, not a chunk-store
    resolver. This helper deliberately doesn't invent or fetch full chunk text.
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
