"""Тесты structure_builder (шаг 2 плана): уровни по эвристике + построение дерева."""

from __future__ import annotations

from typing import List

from src.core.models import RawElement
from src.ingestion.structure_builder import (
    build_document_structure,
    infer_title_level,
    is_junk_title,
    parse_toc_entries,
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


def test_junk_filters_infographic_and_cover() -> None:
    """GRI-метки, обложка «2) ...», строка-предложение — не разделы."""
    assert is_junk_title("GRI 2-1, 2-3")
    assert is_junk_title("2) БПС-СБЕРБАНК")
    assert is_junk_title("Кроме того, содержание Отчета соответствует:")
    assert not is_junk_title("Обзор рынка")


def test_toc_header_variants_and_html_rows() -> None:
    """«Оглавление» (2016/2017/2018) и HTML-таблица TOC дают записи."""
    elements = [
        element("Оглавление", page=1),
        RawElement(element_id="sber_2016:1", doc_id="sber_2016",
                   type="Table", page=1, text="",
                   html=("<table><tr><td>Портрет Группы</td><td>4</td></tr>"
                         "<tr><td>Стратегический отчет</td><td>22</td></tr>"
                         "</table>")),
    ]
    entries = parse_toc_entries(elements)
    assert [(entry.name, entry.page) for entry in entries] == [
        ("Портрет Группы", 4), ("Стратегический отчет", 22)]


def test_toc_combines_split_name_across_cells() -> None:
    """«Обращение ...» | «Правления .... 23» → одно имя + страница (2016)."""
    elements = [
        element("Содержание", page=1),
        RawElement(element_id="sber_2016:1", doc_id="sber_2016",
                   type="Table", page=1, text="",
                   html=("<table><tr><td>Обращение Президента, Председателя</td>"
                         "<td>Правления .... 23</td></tr></table>")),
    ]
    entries = parse_toc_entries(elements)
    assert [(e.name, e.page) for e in entries] == [
        ("Обращение Президента, Председателя Правления", 23)]


def test_toc_does_not_capture_body_pages() -> None:
    """Окно TOC — заголовок + 1 страница; тело документа не должно попадать."""
    elements = [
        element("Содержание", page=1),
        element("Достижения", page=5),
    ]
    assert parse_toc_entries(elements) == []


def test_toc_pages_resolved_from_body_titles() -> None:
    """2017: TOC без страниц сопоставляется с Title в теле по имени."""
    names = ["Портрет Группы", "Стратегический отчет", "Обзор результатов",
             "Корпоративное управление", "Финансовая результативность"]
    rows = "".join(f"<tr><td>{name}</td></tr>" for name in names)
    elements = [
        element("Оглавление", page=1),
        RawElement(element_id="sber_2017:toc", doc_id="sber_2017",
                   type="Table", page=1, text="",
                   html=f"<table>{rows}</table>"),
        *[element(name, page=10 + index) for index, name in enumerate(names)],
    ]
    structure = build_document_structure(elements)
    assert [(section.title, section.page_start)
            for section in structure.sections] == [
        (name, 10 + index) for index, name in enumerate(names)]


def test_toc_pages_are_not_sections() -> None:
    """Title на страницах оглавления не попадают в дерево разделов."""
    elements = [
        element("Содержание", page=1, type_="Header"),
        element("Портрет Группы", page=1),
        element("Стратегический отчет", page=6),
    ]
    structure = build_document_structure(elements)
    assert [section.title for section in structure.sections] == ["Стратегический отчет"]


def test_bbox_height_assigns_heading_levels() -> None:
    """Без outline/TOC крупный/средний/мелкий блок дают уровни 0/1/2."""
    def sized(text: str, height: float) -> RawElement:
        return RawElement(element_id=f"sber_2022:{text}", doc_id="sber_2022",
                          type="Title", text=text, page=10,
                          bbox=[0.0, 0.0, 1000.0, height],
                          page_width=1000, page_height=1000)

    elements = [sized("Часть первая", 300), sized("Глава вторая", 150),
                sized("Раздел третий", 50)]
    structure = build_document_structure(elements)
    assert [section.level for section in structure.sections] == [0, 1, 2]


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
    assert is_junk_title("Содержание")     # страница TOC — не раздел
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


def test_best_toc_level_fuzzy_substring() -> None:
    """Межколоночная склейка TOC («Кредитные рейтинги Обзор рынка») матчится подстрокой."""
    from src.ingestion.structure_builder import (_best_toc_level,
                                                normalize_section_name)

    toc_norm = [(normalize_section_name("Кредитные рейтинги Обзор рынка"), 0)]
    assert _best_toc_level("Обзор рынка", toc_norm) == 0
    assert _best_toc_level("Кредитные рейтинги Обзор рынка", toc_norm) == 0
    # Обратное вложение: Title длиннее TOC-имени
    assert _best_toc_level("Сбер сегодня и стратегия 2026", [("сбер сегодня", 0)]) == 0


def test_best_toc_level_short_names_guarded() -> None:
    """Короткие имена (<8 символов) не матчатся — иначе тянут случайные хвосты."""
    from src.ingestion.structure_builder import _best_toc_level

    assert _best_toc_level("ROE", [("roe и достаточность капитала", 1)]) is None
    assert _best_toc_level("Итоги", [("ключевые итоги года", 0)]) is None


def test_build_from_elements_file_and_save_sections(tmp_path) -> None:
    """Артефакт elements.json → DocumentStructure → sections.json (roundtrip)."""
    import json

    from src.ingestion.structure_builder import (
        build_from_elements_file,
        load_sections,
        save_sections,
    )

    elements = [
        element("1. Управление рисками", page=10),
        element("1.1 Кредитный риск", page=12),
        element("Перейти", page=12),          # junk — не должен попасть в дерево
        element("2. Технологии", page=20),
    ]
    elements_json = tmp_path / "sber_2023_elements.json"
    elements_json.write_text(
        json.dumps([e.to_dict() for e in elements], ensure_ascii=False),
        encoding="utf-8")

    structure = build_from_elements_file(elements_json)
    assert structure.doc_id == "sber_2023"
    assert [s.title for s in structure.sections] == [
        "1. Управление рисками", "1.1 Кредитный риск", "2. Технологии"]

    target = save_sections(structure, tmp_path / "structure")
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["doc_id"] == structure.doc_id
    assert len(saved["sections"]) == len(structure.sections)
    assert saved["sections"][1]["breadcrumbs"] == [
        "1. Управление рисками", "1.1 Кредитный риск"]

    # Обратное чтение артефакта даёт эквивалентное дерево
    restored = load_sections(target)
    assert restored.doc_id == structure.doc_id
    assert [s.title for s in restored.sections] == [
        s.title for s in structure.sections]
    assert restored.sections[1].parent_id == structure.sections[1].parent_id


def test_build_from_elements_file_missing_raises(tmp_path) -> None:
    """Нет артефакта — FileNotFoundError (CLI не должен молча глотать)."""
    import pytest

    from src.ingestion.structure_builder import build_from_elements_file

    with pytest.raises(FileNotFoundError):
        build_from_elements_file(tmp_path / "nope_elements.json")


# ------------------- 2016–2018: калибровка, части, плоские закладки -------------

def test_toc_printed_page_offset_calibrated_via_body_titles() -> None:
    """2016: печатные страницы TOC смещены на +3 относительно PDF."""
    rows = "".join(
        f"<tr><td>{name}</td><td>{printed}</td></tr>"
        for name, printed in [
            ("Сбербанк сегодня", 4), ("Достижения", 6),
            ("Обзор рынка", 26), ("Стратегия и ее реализация", 33),
            ("Управление рисками", 198),
        ])
    elements = [
        element("Оглавление", page=5),
        RawElement(element_id="sber_2016:toc", doc_id="sber_2016",
                   type="Table", page=5, text="",
                   html=f"<table>{rows}</table>"),
        # Тело: те же заголовки на PDF-страницах printed+3
        element("Сбербанк сегодня", page=7, doc_id="sber_2016"),
        element("Достижения", page=9, doc_id="sber_2016"),
        element("Обзор рынка", page=29, doc_id="sber_2016"),
        element("Стратегия и ее реализация", page=36, doc_id="sber_2016"),
        element("Управление рисками", page=201, doc_id="sber_2016"),
    ]
    structure = build_document_structure(
        elements, "sber_2016", toc=parse_toc_entries(elements))
    assert [(s.title, s.page_start) for s in structure.sections] == [
        ("Сбербанк сегодня", 7), ("Достижения", 9),
        ("Обзор рынка", 29), ("Стратегия и ее реализация", 36),
        ("Управление рисками", 201)]


def test_toc_group_becomes_parent_section() -> None:
    """Часть TOC («01. Портрет Группы») — level 0, её строки — level 1."""
    rows = "".join(
        f"<tr><td>{name}</td><td>{page}</td></tr>"
        for name, page in [
            ("Сбербанк сегодня", 4), ("Достижения", 6),
            ("Ключевые результаты", 8), ("Стратегия Группы", 10),
            ("Корпоративное управление", 12),
        ])
    elements = [
        element("Оглавление", page=3),
        element("01. Портрет Группы", page=3),
        RawElement(element_id="sber_2017:toc", doc_id="sber_2017",
                   type="Table", page=3, text="",
                   html=f"<table>{rows}</table>"),
        element("Сбербанк сегодня", page=4, doc_id="sber_2017"),
        element("Достижения", page=6, doc_id="sber_2017"),
        element("Ключевые результаты", page=8, doc_id="sber_2017"),
        element("Стратегия Группы", page=10, doc_id="sber_2017"),
        element("Корпоративное управление", page=12, doc_id="sber_2017"),
    ]
    structure = build_document_structure(
        elements, "sber_2017", toc=parse_toc_entries(elements))
    assert structure.sections[0].title == "01. Портрет Группы"
    assert structure.sections[0].level == 0
    assert [s.level for s in structure.sections[1:]] == [1] * 5
    assert structure.sections[1].breadcrumbs == [
        "01. Портрет Группы", "Сбербанк сегодня"]


def test_flat_outline_levels_inferred_from_numbering() -> None:
    """2018: закладки одного уровня → уровни по нумерации частей."""
    outline = [{"title": "1 Портрет Группы", "level": 0, "page": 5}]
    outline += [{"title": f"Раздел портрета {index}", "level": 0,
                 "page": 6 + index} for index in range(8)]
    outline += [{"title": "2 Стратегический отчет", "level": 0, "page": 20}]
    outline += [{"title": f"Раздел стратегии {index}", "level": 0,
                 "page": 21 + index} for index in range(8)]
    outline += [{"title": "3.1 Клиентский опыт", "level": 0, "page": 40},
                {"title": "Розничные клиенты", "level": 0, "page": 41}]
    structure = build_document_structure(
        [element("Обзор рынка", page=100)], "sber_2018", outline=outline)
    levels = {s.title: s.level for s in structure.sections}
    assert levels["1 Портрет Группы"] == 0
    assert levels["Раздел портрета 0"] == 1
    assert levels["2 Стратегический отчет"] == 0
    assert levels["3.1 Клиентский опыт"] == 1
    assert levels["Розничные клиенты"] == 2



def test_toc_body_page_after_toc_not_excluded() -> None:
    """Страницы тела сразу после TOC не считаются страницами оглавления."""
    elements = [
        element("Оглавление", page=5),
        RawElement(element_id="sber_2016:toc", doc_id="sber_2016",
                   type="Table", page=5, text="",
                   html="<table><tr><td>Раздел .......</td><td>7</td></tr>"
                        "<tr><td>Второй .......</td><td>9</td></tr></table>"),
        element("Первый раздел тела", page=6, doc_id="sber_2016"),
    ]
    from src.ingestion.structure_builder import _toc_page_numbers

    assert _toc_page_numbers(elements) == {5}
