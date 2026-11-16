"""Чистые помощники отрисовки цитат без зависимости от Streamlit."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from src.core.record import field_value

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def _value(item: Any, key: str, default: Any = None) -> Any:
    return field_value(item, key, default)


def api_base_url(environ: Mapping[str, str] | None = None) -> str:
    """Возвращает базовый URL API без завершающего слэша.

    Значение берётся из ``API_BASE_URL`` во время вызова, а не при импорте:
    тесты и разные окружения могут переопределять его без перезапуска.
    """
    source = os.environ if environ is None else environ
    value = source.get("API_BASE_URL", "")
    if not isinstance(value, str):
        value = ""
    value = value.strip().rstrip("/")
    return value or DEFAULT_API_BASE_URL


def _page_number(value: Any) -> int | None:
    """Номер PDF-страницы для fragment; невалидное значение — без fragment."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() and int(text) > 0:
            return int(text)
    return None


def citation_pdf_link(
    citation: Any,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Строит внешнюю ссылку на PDF документа цитаты.

    ``doc_id`` обязателен: без него URL не выдумывается. ``page_pdf``
    необязателен: без него ссылка ведёт на документ без ``#page=``.
    """
    doc_id = _value(citation, "doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return {
            "available": False,
            "url": None,
            "label": "PDF недоступен",
            "page": None,
        }

    normalized_base = (
        base_url.strip().rstrip("/")
        if isinstance(base_url, str) and base_url.strip()
        else api_base_url()
    )
    page = _page_number(_value(citation, "page_pdf"))
    url = (
        f"{normalized_base}/api/documents/"
        f"{quote(doc_id.strip(), safe='')}/file"
    )
    if page is None:
        return {
            "available": True,
            "url": url,
            "label": "Открыть PDF",
            "page": None,
        }
    return {
        "available": True,
        "url": f"{url}#page={page}",
        "label": f"Открыть PDF, стр. {page}",
        "page": page,
    }


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
            link = citation_pdf_link(citation)
            if link["available"]:
                # Только внешняя ссылка; PDF-viewer внутри UI не имитируется.
                st.markdown(f"[{link['label']}]({link['url']})")
            else:
                st.caption(link["label"])
            quote = citation_quote(citation)
            if quote:
                st.text(quote)
            else:
                st.caption("Цитата не предоставлена AgentService.")
