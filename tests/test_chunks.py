"""Тесты чанкинга К2 и обогащения breadcrumbs (шаг 2.5 плана)."""

from __future__ import annotations

from typing import List

from src.core.models import RawElement
from src.ingestion.chunk_enricher import enrich_chunks, find_section_for_page
from src.ingestion.pdf_to_chunks import (
    REQUIRED_CHUNK_FIELDS,
    build_chunks,
    chunk_record,
    element_records,
    make_chunks,
    make_splitter,
    save_chunks,
)
from src.ingestion.structure_builder import build_document_structure

SOURCE = {
    "doc_id": "sber_2023",
    "title": "Сбер 2023",
    "year": 2023,
    "report_period": "2023",
    "url": None,
    "sha256": "deadbeef",
}


def raw(element_id: str, type_: str, text: str, page: int,
        html: str | None = None, page_label: str | None = None) -> RawElement:
    """Короткая фабрика нормализованных элементов."""
    return RawElement(element_id=element_id, doc_id="sber_2023", type=type_,
                      text=text, page=page, html=html, page_label=page_label)


def test_element_records_skips_images_and_empty() -> None:
    """Image/PageBreak и пустые элементы не попадают в записи; таблица несёт html."""
    elements = [
        raw("sber_2023:00000", "Title", "1. Управление рисками", 10),
        raw("sber_2023:00001", "Image", "", 10),
        raw("sber_2023:00002", "NarrativeText", "   ", 10),
        raw("sber_2023:00003", "Table", "активы 100", 11, html="<table>1</table>"),
    ]
    records = element_records(elements, SOURCE)

    assert [r["kind"] for r in records] == ["text", "table"]
    assert records[0]["evidence_id"] == "sber_2023:p010"
    assert records[1]["html"] == "<table>1</table>"
    assert records[0]["year"] == 2023
    # Таблица без текста, но с html — нарушение корпуса
    broken = [raw("sber_2023:00009", "Table", "", 12, html="<table>2</table>")]
    import pytest
    with pytest.raises(ValueError):
        element_records(broken, SOURCE)


def test_element_records_empty_document_raises() -> None:
    """Документ без текстовых элементов — ошибка, а не пустой корпус."""
    import pytest

    with pytest.raises(ValueError):
        element_records([raw("sber_2023:00000", "Image", "", 1)], SOURCE)


def test_make_chunks_table_single_chunk_and_page_boundary() -> None:
    """Таблица = один чанк с html; смена страницы закрывает текстовую группу."""
    elements = [
        raw("sber_2023:00000", "Title", "1. Раздел", 1),
        raw("sber_2023:00001", "NarrativeText", "Текст страницы один.", 1),
        raw("sber_2023:00002", "Table", "активы 100", 2, html="<table>1</table>"),
        raw("sber_2023:00003", "NarrativeText", "Текст страницы два.", 2),
    ]
    chunks = make_chunks(element_records(elements, SOURCE))

    assert all(REQUIRED_CHUNK_FIELDS <= set(chunk) for chunk in chunks)
    tables = [c for c in chunks if c["kind"] == "table"]
    assert len(tables) == 1 and tables[0]["html"] == "<table>1</table>"
    # Текст страницы 1 и 2 — разные evidence_id → разные чанки
    text_chunks = [c for c in chunks if c["kind"] == "text"]
    assert {c["evidence_id"] for c in text_chunks} == {
        "sber_2023:p001", "sber_2023:p002"}
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_chunk_record_digest_is_stable() -> None:
    """chunk_id детерминирован: одинаковые входные данные → одинаковый digest."""
    records = element_records(
        [raw("sber_2023:00000", "NarrativeText", "Абзац.", 1)], SOURCE)
    first = chunk_record(records, "Абзац.", "text")
    second = chunk_record(records, "Абзац.", "text")
    assert first["chunk_id"] == second["chunk_id"]
    assert first["chunk_id"].startswith("sber_2023:00000:c0:")


def test_build_chunks_respects_chunk_size() -> None:
    """Сплиттер режет длинный текст на части не больше chunk_size."""
    long_text = "Предложение про риски. " * 30
    elements = [raw("sber_2023:00000", "NarrativeText", long_text, 5)]
    chunks = build_chunks(elements, SOURCE, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)


def test_enrich_chunks_adds_structure_fields() -> None:
    """Чанк страницы получает breadcrumbs/section_path/level/parent_section."""
    structure = build_document_structure([
        raw("sber_2023:00000", "Title", "1. Управление рисками", 10),
        raw("sber_2023:00001", "Title", "1.1 Кредитный риск", 12),
        raw("sber_2023:00002", "Title", "2. Технологии", 20),
    ])
    elements = [
        raw("sber_2023:00010", "NarrativeText", "Текст до первого раздела.", 5),
        raw("sber_2023:00011", "NarrativeText", "Кредитный портфель вырос.", 13),
        raw("sber_2023:00012", "NarrativeText", "Технологический текст.", 25),
    ]
    chunks = enrich_chunks(make_chunks(element_records(elements, SOURCE)),
                           structure)

    assert "section_path" not in chunks[0]          # страница до заголовков
    assert chunks[1]["breadcrumbs"] == ["1. Управление рисками",
                                        "1.1 Кредитный риск"]
    assert chunks[1]["section_path"] == "1. Управление рисками/1.1 Кредитный риск"
    assert chunks[1]["section_level"] == 1
    assert chunks[1]["parent_section"] == "1. Управление рисками"
    assert chunks[2]["breadcrumbs"] == ["2. Технологии"]
    assert chunks[2]["parent_section"] is None


def test_find_section_for_page_uses_last_heading() -> None:
    """Действует последний заголовок с page_start <= страницы."""
    structure = build_document_structure([
        raw("sber_2023:00000", "Title", "1. Риски", 3),
        raw("sber_2023:00001", "Title", "2. Технологии", 10),
    ])
    assert find_section_for_page(structure, 1) is None
    assert find_section_for_page(structure, 3).title == "1. Риски"
    assert find_section_for_page(structure, 9).title == "1. Риски"
    assert find_section_for_page(structure, 30).title == "2. Технологии"


def test_save_chunks_roundtrip(tmp_path) -> None:
    """save_chunks пишет валидный JSON-список чанков."""
    import json

    chunks = make_chunks(element_records(
        [raw("sber_2023:00000", "NarrativeText", "Текст.", 1)], SOURCE))
    target = save_chunks(chunks, "sber_2023", tmp_path / "chunks")
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(saved, list) and len(saved) == len(chunks)
    assert REQUIRED_CHUNK_FIELDS <= set(saved[0])
