"""Тесты чтения встроенной структуры PDF (закладки и link-аннотации)."""

from __future__ import annotations

import pytest
from pypdf import PdfReader, PdfWriter

from src.core.models import RawElement
from src.ingestion.pdf_outline import (
    OutlineEntry,
    load_links,
    load_outline,
    outline_level_map,
    read_links,
    read_outline,
    save_links,
    save_outline,
)
from src.ingestion.structure_builder import (
    TocEntry,
    build_document_structure,
    normalize_section_name,
)


def _make_pdf(path, bookmarks):
    """Создать PDF из N пустых страниц и закладок ``(title, page_index, depth)``."""
    writer = PdfWriter()
    n_pages = max(page for _, page, _ in bookmarks) + 1 if bookmarks else 1
    for _ in range(n_pages):
        writer.add_blank_page(width=595, height=842)
    parents = {}
    for title, page_index, depth in bookmarks:
        parent = parents.get(depth - 1) if depth else None
        item = writer.add_outline_item(title, page_index, parent=parent)
        parents[depth] = item
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


def test_read_outline_levels_pages_and_junk_filter(tmp_path) -> None:
    """Закладки читаются с глубиной и страницами; `_GoBack`/`Par12` отфильтрованы."""
    pdf = _make_pdf(tmp_path / "book.pdf", [
        ("Сбер в цифрах и фактах", 0, 0),
        ("Сбер сегодня", 1, 1),
        ("_GoBack", 2, 1),
        ("Par763", 2, 1),
        ("Управление рисками", 3, 1),
    ])

    entries = read_outline(pdf)

    assert [(e.title, e.level, e.page) for e in entries] == [
        ("Сбер в цифрах и фактах", 0, 1),
        ("Сбер сегодня", 1, 2),
        ("Управление рисками", 1, 4),
    ]


def test_read_outline_without_bookmarks_returns_empty(tmp_path) -> None:
    """PDF без outline не даёт исключений — пустой список."""
    pdf = _make_pdf(tmp_path / "empty.pdf", [])
    assert read_outline(pdf) == []


def test_outline_level_map_normalizes_names(tmp_path) -> None:
    """Карта уровней использует ту же нормализацию, что и сопоставление Title."""
    entries = [
        OutlineEntry(title="Сбер в цифрах и\u00a0фактах", level=0, page=10),
        OutlineEntry(title="Сбер сегодня", level=1, page=11),
    ]
    mapping = outline_level_map(entries)

    assert mapping[normalize_section_name("Сбер в цифрах и фактах")] == 0
    assert mapping[normalize_section_name("СБЕР СЕГОДНЯ")] == 1


def test_outline_priority_over_toc_and_heuristic() -> None:
    """Уровень из закладки важнее TOC и эвристики по нумерации Title."""
    elements = [
        RawElement(element_id="sber_2023:00000", doc_id="sber_2023",
                   type="Title", text="1. Управление рисками", page=10),
        RawElement(element_id="sber_2023:00001", doc_id="sber_2023",
                   type="Title", text="Управление рисками в 2023 году", page=19),
    ]
    toc = [TocEntry(name="1. Управление рисками", page=10, level=0)]
    outline = [
        OutlineEntry(title="Сбер в цифрах и фактах", level=0, page=1),
        OutlineEntry(title="1. Управление рисками", level=1, page=10),
        OutlineEntry(title="Управление рисками в 2023 году", level=2, page=19),
    ]

    with_outline = build_document_structure(elements, toc=toc, outline=outline)
    without_outline = build_document_structure(elements, toc=toc)

    assert [s.level for s in with_outline.sections] == [1, 2]
    # Без закладок первый Title получает уровень из TOC, второй — без сигнала
    # считается «сестрой» (фолбэк-sibling), а не ложным ребёнком.
    assert without_outline.sections[0].level == 0
    assert without_outline.sections[1].level == 0


def test_outline_fuzzy_match_short_guard() -> None:
    """Fuzzy-матч по закладке работает, но не на именах короче 8 символов."""
    elements = [
        RawElement(element_id="sber_2024:00000", doc_id="sber_2024",
                   type="Title", text="Управление рисками", page=20),
        RawElement(element_id="sber_2024:00001", doc_id="sber_2024",
                   type="Title", text="Итоги", page=21),
    ]
    outline = [
        OutlineEntry(title="Управление рисками в 2024 году", level=1, page=20),
        OutlineEntry(title="Итоги года", level=0, page=21),
    ]

    structure = build_document_structure(elements, outline=outline)

    assert structure.sections[0].level == 1
    # guard: «Итоги» короче 8 символов — закладка не применяется, работает
    # фолбэк-sibling (без bbox глубина не выдумывается)
    assert structure.sections[1].level == 1


def test_outline_accepts_dicts() -> None:
    """`build_document_structure` принимает закладки-словари (без цикла импортов)."""
    elements = [
        RawElement(element_id="sber_2023:00000", doc_id="sber_2023",
                   type="Title", text="Сбер сегодня", page=11),
    ]
    structure = build_document_structure(
        elements, outline=[{"title": "Сбер сегодня", "level": 1}])
    assert structure.sections[0].level == 1


def test_read_links_empty_pdf(tmp_path) -> None:
    """PDF без link-аннотаций → пустой список."""
    pdf = _make_pdf(tmp_path / "plain.pdf", [])
    assert read_links(pdf) == []


def test_outline_links_roundtrip(tmp_path) -> None:
    """Артефакты outline/links атомарно сохраняются и читаются обратно."""
    entries = [OutlineEntry(title="Раздел", level=0, page=3)]
    links = [{"source_page": 1, "target_page": 3}]

    outline_path = save_outline(entries, "sber_2023", tmp_path)
    links_path = save_links(links, "sber_2023", tmp_path)

    assert outline_path.name == "sber_2023_outline.json"
    assert links_path.name == "sber_2023_links.json"
    assert load_outline(outline_path) == entries
    assert load_links(links_path) == links


def test_load_outline_missing_raises(tmp_path) -> None:
    """Отсутствие артефакта закладок — FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_outline(tmp_path / "nope.json")
