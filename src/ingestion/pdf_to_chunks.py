"""Чанкинг нормализованных элементов PDF в контракт чанка К2 (Занятие 2).

Модуль переносит курсовой контракт ``chunk_record``/``make_chunks`` на наш
``RawElement``: тот же формат ``chunk_id``/``evidence_id``, сплиттер 800/100,
«таблица = один чанк» с HTML. Дополнительно чанки обогащаются структурой
(``chunk_enricher.py``), поэтому базовые поля К2 не меняются.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.core.models import RawElement

#: Курсовые обязательные поля чанка (Занятие 2) — НЕ МЕНЯЕМ.
REQUIRED_CHUNK_FIELDS = {
    "chunk_id", "evidence_id", "doc_id", "title",
    "page_pdf", "page_label", "year", "report_period",
    "kind", "element_ids", "text", "html",
    "source_url", "source_sha256",
}

#: Размер и перекрытие текстового чанка (К2: 800/100).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def make_splitter(chunk_size: int = CHUNK_SIZE,
                  chunk_overlap: int = CHUNK_OVERLAP) -> Any:
    """Создать курсовой ``RecursiveCharacterTextSplitter`` (800/100 по умолчанию).

    Args:
        chunk_size: Максимальная длина текстового чанка в символах.
        chunk_overlap: Перекрытие соседних чанков.

    Returns:
        Настроенный сплиттер LangChain.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def source_meta_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Собрать метаданные источника из имени PDF и его содержимого.

    Args:
        pdf_path: Путь к PDF-отчёту.

    Returns:
        Словарь с ``doc_id``/``title``/``year``/``report_period``/``url``/
        ``sha256`` (совместим с курсовым ``SOURCES``).
    """
    from src.ingestion.pdf_parser import compute_source_sha256, make_doc_id

    path = Path(pdf_path)
    doc_id = make_doc_id(path.name)
    year = int(doc_id.rsplit("_", 1)[-1])
    return {
        "doc_id": doc_id,
        "title": path.stem,
        "year": year,
        "report_period": str(year),
        "url": None,
        "sha256": compute_source_sha256(path),
    }


def element_records(elements: Sequence[RawElement],
                    source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Превратить ``RawElement`` в записи-элементы контракта К2.

    Пропускаются ``Image``/``PageBreak`` и пустые элементы; таблица обязана
    иметь текст (HTML без текста — ошибка корпуса, как в курсе).

    Args:
        elements: Нормализованные элементы документа (в порядке следования).
        source: Метаданные источника (``doc_id``, ``title``, ``year``,
            ``report_period``, ``url``, ``sha256``).

    Returns:
        Список записей с полями, достаточными для ``chunk_record``.

    Raises:
        ValueError: если у таблицы есть HTML, но нет текста, либо не осталось
            ни одной записи с текстом.
    """
    records: List[Dict[str, Any]] = []
    for element in elements:
        if element.type in {"Image", "PageBreak"}:
            continue
        text = (element.text or "").strip()
        html = element.html if element.type == "Table" else None
        if not text:
            if html:
                raise ValueError(
                    f"{source['doc_id']}, страница {element.page}: "
                    "таблица содержит HTML, но не содержит текста для поиска.")
            continue
        records.append({
            "element_id": element.element_id,
            "evidence_id": f"{source['doc_id']}:p{element.page:03d}",
            "doc_id": source["doc_id"],
            "title": source["title"],
            "page_pdf": element.page,
            "page_label": element.page_label,
            "year": int(source["year"]),
            "report_period": source["report_period"],
            "kind": "table" if element.type == "Table" else "text",
            "category": element.type,
            "text": text,
            "html": html,
            "source_url": source.get("url"),
            "source_sha256": source.get("sha256"),
        })
    if not records:
        raise ValueError(f"{source['doc_id']}: нет элементов с текстом.")
    return records


def chunk_record(records: Sequence[Mapping[str, Any]], text: str, kind: str,
                 part_number: int = 0) -> Dict[str, Any]:
    """Собрать один чанк контракта К2 из группы записей-элементов.

    Args:
        records: Записи, вошедшие в чанк (первая задаёт метаданные страницы).
        text: Текст чанка (для таблицы — её плоский текст).
        kind: ``"text"`` или ``"table"``.
        part_number: Порядковый номер части внутри страницы.

    Returns:
        Словарь чанка с курсовыми полями (``chunk_id`` c sha256-диджестом).
    """
    first = records[0]
    metadata_keys = [
        "evidence_id", "doc_id", "title", "page_pdf", "page_label", "year",
        "report_period", "source_url", "source_sha256",
    ]
    html = first["html"] if kind == "table" else None
    element_ids = [record["element_id"] for record in records]
    payload = json.dumps([text, html, element_ids], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return {
        **{key: first[key] for key in metadata_keys},
        "chunk_id": f"{first['element_id']}:c{part_number}:{digest}",
        "kind": kind,
        "element_ids": element_ids,
        "text": text,
        "html": html,
    }


def split_text_records(records: Sequence[Mapping[str, Any]], text: str,
                       splitter: Any) -> List[Dict[str, Any]]:
    """Разрезать группу текстовых записей сплиттером на чанки ``kind="text"``."""
    return [chunk_record(records, part, "text", part_number)
            for part_number, part in enumerate(splitter.split_text(text))]


def make_chunks(records: Sequence[Mapping[str, Any]],
                splitter: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Собрать чанки К2: группировка по странице, таблица = один чанк.

    Группа текста закрывается при смене страницы (``evidence_id``) или перед
    таблицей; каждая таблица становится отдельным чанком с сохранённым HTML.

    Args:
        records: Записи-элементы из ``element_records``.
        splitter: Сплиттер (None → ``make_splitter()`` 800/100).

    Returns:
        Список чанков в порядке документа.

    Raises:
        ValueError: если получились повторяющиеся ``chunk_id``.
    """
    splitter = splitter or make_splitter()
    chunks: List[Dict[str, Any]] = []
    pending_text: List[Mapping[str, Any]] = []
    for record in records:
        new_page = bool(pending_text) and (
            pending_text[-1]["evidence_id"] != record["evidence_id"])
        if new_page or record["kind"] == "table":
            chunks.extend(split_text_records(
                pending_text,
                "\n\n".join(item["text"] for item in pending_text),
                splitter))
            pending_text = []
        if record["kind"] == "table":
            chunks.append(chunk_record([record], record["text"], "table"))
        else:
            pending_text.append(record)
    chunks.extend(split_text_records(
        pending_text,
        "\n\n".join(item["text"] for item in pending_text),
        splitter))
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Получились повторяющиеся chunk_id.")
    return chunks


def build_chunks(elements: Sequence[RawElement], source: Mapping[str, Any],
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Полный цикл: ``RawElement`` → записи → чанки К2.

    Args:
        elements: Нормализованные элементы документа.
        source: Метаданные источника (``source_meta_from_pdf``).
        chunk_size: Размер текстового чанка.
        chunk_overlap: Перекрытие чанков.

    Returns:
        Список чанков контракта К2.
    """
    records = element_records(elements, source)
    return make_chunks(records, make_splitter(chunk_size, chunk_overlap))


def load_elements(elements_json: Path) -> List[RawElement]:
    """Прочитать нормализованные элементы из ``{doc_id}_elements.json``.

    Args:
        elements_json: Путь к артефакту парсера.

    Returns:
        Список ``RawElement``.

    Raises:
        FileNotFoundError: если артефакта нет.
    """
    path = Path(elements_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт парсера не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [RawElement.from_dict(item) for item in raw]


def save_chunks(chunks: Sequence[Mapping[str, Any]], doc_id: str,
                chunks_dir: Path) -> Path:
    """Атомарно сохранить ``{doc_id}_chunks.json``.

    Args:
        chunks: Чанки документа.
        doc_id: Идентификатор документа.
        chunks_dir: Каталог ``data/chunks`` (создаётся при отсутствии).

    Returns:
        Путь к записанному файлу.
    """
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    target = chunks_dir / f"{doc_id}_chunks.json"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json",
        dir=chunks_dir, delete=False)
    try:
        json.dump(list(chunks), handle, ensure_ascii=False, indent=1)
        handle.flush()
    finally:
        handle.close()
    os.replace(handle.name, target)
    return target
