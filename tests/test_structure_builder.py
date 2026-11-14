"""Тесты structure_builder (шаг 2 плана): уровни по эвристике + построение дерева."""

from __future__ import annotations

from typing import List

from src.core.models import RawElement
from src.ingestion.structure_builder import (
    build_document_structure,
    infer_title_level,
)


def element(text: str, page: int, type_: str = "Title", doc_id: str = "sber_2023"):
    """Короткая фабрика элементов (type не-Title нужен для «пролёта» мимо)."""
    return RawElement(element_id=f"{doc_id}:0", doc_id=doc_id,
                      type=type_, text=text, page=page)


def test_infer_level_numbering() -> None:
    """Нумерация «1.», «1.1.», «1.1.1» — уровни 0/1/2 (самый частый случай отчётов)."""
    assert infer_title_level("1. Стратегический отчет") == 0
    assert infer_title_level("2.1 Обзор ключевых показателей") == 1
    assert infer_title_level("2.1.3. Киберриски") == 2


def test_infer_level_word_and_caps() -> None:
    """Словесные метки и ALL-CAPS — уровень 0; обычные заголовки — None."""
    assert infer_title_level("ГЛАВА 2. Управление рисками") == 0
    assert infer_title_level("ПРИЛОЖЕНИЕ 3. Сводные таблицы") == 0
    assert infer_title_level("СТРАТЕГИЧЕСКИЙ ОТЧЕТ") == 0
    assert infer_title_level("Сообщение Председателя Правления") is None
    assert infer_title_level("") is None


def test_build_tree_breadcrumbs_and_parent() -> None:
    """Дерево из трёх заголовков: breadcrumbs картеж по стеку, parent у вложенных."""
    elements: List[RawElement] = [
        element("1. Управление рисками", page=10),
        element("1.1 Кредитный риск", page=12),
        element("1.1.1 Портфель кредитов МСБ", page=15),
    ]
    structure = build_document_structure(elements)

    assert structure.doc_id == "sber_2023"
    lvl0, lvl1, lvl2 = structure.sections
    # Уровни и breadcrumbs строятся на стеке
    assert (lvl0.level, lvl1.level, lvl2.level) == (0, 1, 2)
    assert lvl0.breadcrumbs == ["1. Управление рисками"]
    assert lvl1.breadcrumbs == ["1. Управление рисками", "1.1 Кредитный риск"]
    assert lvl2.breadcrumbs == [
        "1. Управление рисками", "1.1 Кредитный риск",
        "1.1.1 Портфель кредитов МСБ"]
    # Родители/дети и section_path-«ключи»
    assert lvl1.parent_id == lvl0.id and lvl2.parent_id == lvl1.id
    assert lvl0.id.endswith("1. Управление рисками")
    assert lvl0.children_ids == [lvl1.id]


def test_stack_closes_on_same_level() -> None:
    """Второй «1.x» закрывает ветку: breadcrumbs у «1.2» не содержит «1.1»."""
    structure = build_document_structure([
        element("1. Риски", page=1),
        element("1.1 Кредитный", page=2),
        element("1.2 Рыночный", page=5),
    ])
    first, second = structure.sections[1], structure.sections[2]
    assert second.breadcrumbs == ["1. Риски", "1.2 Рыночный"]
    assert "1.1 Кредитный" not in second.breadcrumbs


def test_page_end_next_section_boundary() -> None:
    """page_end раздела = страница следующего (но не меньше page_start)."""
    structure = build_document_structure([
        element("1. Риски", page=1),
        element("1.1 Кредитный", page=2),
        element("1.2 Рыночный", page=5),
    ])
    assert structure.sections[0].page_end == 2
    assert structure.sections[1].page_end == 5
    # Последний раздел — page_end остаётся равен page_start (до конца документа)


def test_non_title_ignored_and_page_monotonic() -> None:
    """NarrativeText/Table не создают разделов; page_end не «падает»."""
    structure = build_document_structure([
        element("1. Риски", page=1),
        element("какой-то текст", page=2, type_="NarrativeText"),
        element("<table>", page=2, type_="Table"),
        element("2. Сводка", page=4),
    ])
    assert [s.title for s in structure.sections] == ["1. Риски", "2. Сводка"]
    assert all(s.page_end >= s.page_start for s in structure.sections)


def test_to_dict_serializable() -> None:
    """``to_dict`` пригоден для ``json.dump`` без дополнительной обработки."""
    import json

    structure = build_document_structure([element("1. Риски", page=1)])
    payload = structure.to_dict()
    json.dumps(payload, ensure_ascii=False)  # не падает
    assert payload["doc_id"] == "sber_2023"
    assert payload["sections"][0]["breadcrumbs"] == ["1. Риски"]


# ---------------------- Оглавление (правка уровней) ----------------------

def toc_element(text: str, page: int, type_: str = "Title"):
    """Фабрика элементов страницы-оглавления."""
    return element(text, page=page, type_=type_)


def test_parse_toc_entries_collects_numbered_lines() -> None:
    """Строки «имя + страница» после заголовка «Содержание» парсятся в TocEntry."""
    from src.ingestion.structure_builder import parse_toc_entries

    rows = [
        toc_element("Содержание", page=9),
        toc_element("Обзор результатов 23 27 58 76 82", page=9),
        toc_element("1. Стратегический отчет 23", page=9),
        toc_element("1.1 Обзор ключевых показателей 27", page=10),
        toc_element("Отчет по рискам 76", page=10),
        toc_element("какой-то текст без номера", page=11, type_="NarrativeText"),
        toc_element("тоже текст", page=12, type_="NarrativeText"),
    ]
    entries = parse_toc_entries(rows)
    assert [(e.name, e.page) for e in entries] == [
        ("Обзор результатов", 23),          # первое число из мульти-столбца
        ("1. Стратегический отчет", 23),
        ("1.1 Обзор ключевых показателей", 27),
        ("Отчет по рискам", 76),
    ]
    # Уровень по нумерации имени, где она есть
    assert entries[1].level == 0
    assert entries[2].level == 1
    assert entries[0].level is None


def test_parse_toc_stops_after_toc_block() -> None:
    """TOC заканчивается: две подряд не-TOC строки закрывают сбор."""
    from src.ingestion.structure_builder import parse_toc_entries

    rows = [
        toc_element("Содержание", page=3),
        toc_element("Обзор результатов 23", page=3),
        toc_element("Отчет по рискам 76", page=4),
        toc_element("Введение без номера", page=5, type_="NarrativeText"),
        toc_element("Ещё абзац", page=5, type_="NarrativeText"),
        toc_element("Новая часть 999", page=99, type_="NarrativeText"),
    ]
    assert len(parse_toc_entries(rows)) == 2


def test_scan_toc_text_dirs_merged_blocks() -> None:
    """Слитный Table-текст «имя число имя число» → пары; года и футеры тише."""
    from src.ingestion.structure_builder import _scan_toc_text

    entries = _scan_toc_text("Отчет по рискам 11 Сбер сегодня 104")
    assert [(e.name, e.page) for e in entries] == [
        ("Отчет по рискам", 11), ("Сбер сегодня", 104)]

    # Мульти-столбец чисел: вторые числа съедаются до новых слов
    entries = _scan_toc_text("Обзор результатов 23 27 58 76 82 Сеть продаж 76")
    assert [(e.name, e.page) for e in entries] == [
        ("Обзор результатов", 23), ("Сеть продаж", 76)]

    # Года («в 2023 году») остаются в имени, но НЕ становятся страницей
    entries = _scan_toc_text("Экосистема В2С в 2023 году 15")
    assert [(e.name, e.page) for e in entries] == [("Экосистема В2С в 2023 году", 15)]


def test_toc_level_overrides_heuristic() -> None:
    """Имя, совпавшее с записью TOC, берёт уровень из TOC вместо фолбэка."""
    from src.ingestion.structure_builder import build_document_structure, parse_toc_entries

    toc = parse_toc_entries([
        toc_element("Содержание", page=9),
        toc_element("Отчет по рискам 76", page=9),
        toc_element("1.1 Кредитный риск 80", page=9),
    ])
    elements = [
        element("Отчет по рискам", page=76),        # без нумерации: фолбэк дал бы 1
        element("1.1 Кредитный риск", page=80),
    ]
    structure = build_document_structure(elements, toc=toc)
    assert structure.sections[0].level == 0        # уровень пришёл из TOC
    assert structure.sections[1].level == 1


def test_normalize_section_name_hyphens_and_case() -> None:
    """Нормализация глушит регистр, пунктуацию и дефис-переносы строк."""
    from src.ingestion.structure_builder import normalize_section_name

    assert normalize_section_name("Управление рисками") == normalize_section_name(
        "управление РИСКАМИ")
    assert normalize_section_name("Финансовая\nотчет- ность") == \
        normalize_section_name("финансовая отчетность")


def test_junk_titles_filtered() -> None:
    """ФИО-строки советов и «строки-ссылки» не попадают в дерево разделов."""
    from src.ingestion.structure_builder import is_junk_title

    assert is_junk_title("1. ПРОКОПОВИЧ ПетР ПетРОВИЧ")
    assert is_junk_title("2. МАтЮШеВСКИй ВАСИлИй СтАНИСлАВОВИЧ")
    assert is_junk_title("Перейти")
    assert is_junk_title("")
    assert is_junk_title(">5,7%")
    assert is_junk_title("~800 млрд руб.")
    assert is_junk_title("х" * 200)
    assert is_junk_title("93 АС")          # милстоуны инфографики
    assert is_junk_title("116,5 ПБ")
    assert is_junk_title("3 PLATFORM V")
    assert is_junk_title("В 15 раз")
    assert is_junk_title("GRI 3-3")
    assert is_junk_title("Y")
    assert not is_junk_title("1. Управление рисками")
    assert not is_junk_title("Миссия и ценности")
