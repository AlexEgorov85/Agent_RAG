"""Построение дерева разделов документа из parsed-элементов (шаг 2 плана).

Идея спеки (`spec_project.md` §5.1): ``Title``-элементы Unstructured собираются
в ``DocumentStructure`` — плоский список ``SectionNode`` с деревом по ``level``
и breadcrumbs вида ``["Управление рисками", "Операционные риски"]``.

Ключевая особенность (смоук парсера, tasks.md 7.3): на hi_res у ``Title``
``category_depth = None`` — Unstructured глубину заголовков не даёт. Поэтому
уровень считается собственной эвристикой ``infer_title_level``:

1. **Нумерация текста** (самый надёжный сигнал годовых отчётов):
   ``1.`` → уровень 0, ``1.1.`` → 1, ``1.1.1`` → 2;
2. **Словесные метки** («ГЛАВА 2», «РАЗДЕЛ I», «ПРИЛОЖЕНИЕ 3») → уровень 0;
3. **ALL-CAPS без цифр** → уровень 0 (типографический ход годовых отчётов:
   части отчёта набираются заглавными);
4. **Фолбэк** — «унаследованный уровень»: заголовки без сигналов продолжают
   разметку последнего виденного уровня, а не всегда один фиксированный.

Инварианты: поля ``RawElement`` не изменяются (парсер — источник истины);
уровень живёт только в ``SectionNode``. Контракт чанка не меняется:
``section_path``/``breadcrumbs`` — опциональные поля поверх обязательных полей К2
(см. `spec_project.md` §5.2).

Приоритет источников уровня (согласован): **PDF outline (закладки) → текст
«Содержание» → эвристика по Title**. Закладки читает ``pdf_outline.py``;
готовый список принимается через ``outline=``, а при необходимости
``read_outline_safely`` читает их лениво, поэтому цикла импортов нет.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.io import atomic_write_json

from src.core.models import RawElement

#: Словесные метки верхнего уровня: «ГЛАВА 2», «РАЗДЕЛ I», «ПРИЛОЖЕНИЕ 3»
_WORDS_PART_MARKERS = re.compile(
    r"^\s*(ГЛАВА|РАЗДЕЛ|ПРИЛОЖЕНИЕ|ЧАСТЬ|ОТДЕЛ)\s+([0-9]+|[IVXLC]+|[А-ЯЁ]{1,5})\b\.?",
    re.IGNORECASE,
)
#: Нумерация «N.», «N.M.», «N.M.N.»: ТОЧКА после первой группы обязательна
#: («1. Риски», «1.1 Кредитный», «1.1.1 Портфель»), иначе «93 АС» — тоже «номер»
_NUMBERED = re.compile(r"^\s*(\d{1,2}\.(?:\d{1,2}\.?){0,3})\s")


def _is_mostly_uppercase(text: str) -> bool:
    """True, если в тексте есть буквы и почти все буквы — заглавные."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper >= len(letters) - 1


def infer_title_level(text: str) -> Optional[int]:
    """Уровень заголовка по тексту (эвристика вместо отсутствующего depth).

    Args:
        text: Текст Title-элемента (обычно 1-3 строки, первая — сигнал).

    Returns:
        Уровень: 0 = верхний раздел, 1..N = подразделы;
        None, если ни одного сигнала не найдено (фолбэк «наследует уровень»).
    """
    text = (text or "").strip()
    if not text:
        return None

    match = _NUMBERED.match(text)
    if match is not None:
        # Уровень = число цифровых сегментов - 1: «1.» → 0, «2.1» → 1, «2.1.3.» → 2
        parts = re.findall(r"\d{1,2}", match.group(1))
        return max(0, min(len(parts) - 1, 3))

    if _WORDS_PART_MARKERS.match(text):
        return 0

    if _is_mostly_uppercase(text):
        return 0
    return None


@dataclass
class SectionNode:
    """Узел дерева разделов (spec_project.md §5.1).

    Attributes:
        id: Ключ раздела = конкатенация breadcrumbs (совпадает с
            ``section_path`` чанков — один единственный путь по строке).
        title: Текст заголовка (как в PDF).
        level: Глубина (0 = верхний раздел, 1..N = вложенные).
        page_start: Первая страница раздела (та, где стоит заголовок).
        page_end: Последняя страница раздела (страница следующего раздела).
        breadcrumbs: От корня до текущего, включая self.
        parent_id: Id родителя (None у уровня 0).
        children_ids: Id детей (заполняется при обходе).
    """

    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    breadcrumbs: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь (готова для ``json.dump(..., ensure_ascii=False)``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionNode":
        """Восстановить узел из словаря артефакта ``*_sections.json``."""
        known = {"id", "title", "level", "page_start", "page_end",
                 "breadcrumbs", "parent_id", "children_ids"}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DocumentStructure:
    """Плоский список разделов документа; дерево восстанавливается по parent_id.

    Attributes:
        doc_id: Идентификатор документа (как в реестре источников, ``sber_2015``).
        sections: Разделы в порядке появления в документе.
    """

    doc_id: str
    sections: List[SectionNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация (для артефакта ``{doc_id}_sections.json``)."""
        return {"doc_id": self.doc_id,
                "sections": [s.to_dict() for s in self.sections]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStructure":
        """Восстановить структуру из словаря артефакта ``*_sections.json``."""
        return cls(doc_id=data.get("doc_id", ""),
                   sections=[SectionNode.from_dict(item)
                             for item in data.get("sections", [])])


def section_path(breadcrumbs: Sequence[str]) -> str:
    """Из breadcrumbs делает строковый ключ: ``раздел/подраздел``."""
    return "/".join(breadcrumbs)


# ===================== Оглавление (второй источник уровней) ==============
#: Заголовок оглавления (варианты набора в разных годах)
_TOC_HEADER_RE = re.compile(
    r"^\s*(СОДЕРЖАНИ[ЕИЮ]|ОГЛАВЛЕНИ[ЕИЮ]|CONTENTS)\s*:?\s*$",
    re.IGNORECASE,
)
#: Максимальное «значение страницы» TOC (годы вида 2023 не считаются)
_TOC_MAX_PAGE = 2000
#: Колонтитулы/«дно» TOC, которые не должны становиться записями.
_TOC_JUNK = re.compile(
    r"^(?:www\.|sberbank\.com|годовой\s+отчет|сбербанк\s+—\s+годовой)",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_DOTTED_LEADERS_RE = re.compile(r"[.·…]{2,}")
#: OCR-лидеры точечных заполнителей: «или», «лилии», «иииии»
_OCR_LEADER_RE = re.compile(r"\b[ил]{2,}\b", re.IGNORECASE)
#: Слова с тремя и более одинаковыми буквами подряд — почти всегда OCR-шум
_OCR_REPEAT_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b", re.IGNORECASE)
#: Визуально одинаковые кириллица/латиница: «В2В» и «B2B» — один раздел.
_HOMOGLYPHS = str.maketrans({
    "В": "B", "С": "C", "Р": "P", "А": "A", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Т": "T", "Х": "X", "У": "Y",
    "в": "b", "с": "c", "р": "p", "а": "a", "е": "e", "к": "k",
    "м": "m", "н": "h", "о": "o", "т": "t", "х": "x", "у": "y",
    "ё": "е", "й": "и",
})


def _scan_toc_text(text: str) -> List[TocEntry]:
    """Жадный разбор слитного TOC-текста (когда unstructured отдает
    «Содержание» одним Table/Image-блоком без построчного разбиения).

    Имя копится из нечисловых токенов; отдельный числовой токен < 2000
    закрывает накопленное имя записью (name, страница). Соль-числа после
    закрывшего номера (суб-столбцы) отбрасываются до появления новых слов.

    Args:
        text: Сырой текст блока (со слитыми строками из 2+ колонок).

    Returns:
        ``TocEntry`` из потока (имена нормализуются по краям).
    """
    entries: List[TocEntry] = []
    name_tokens: List[str] = []
    for token in text.split():
        if token.isdigit() and int(token) < _TOC_MAX_PAGE:
            raw = " ".join(name_tokens).strip(" .-—–;")
            if len(raw) >= 2:
                entries.append(TocEntry(name=raw,
                                        page=int(token),
                                        level=infer_title_level(raw)))
            name_tokens = []
        else:
            name_tokens.append(token)
    return entries


def _clean_toc_line(text: str) -> str:
    """Очистить строку/ячейку TOC от HTML, точечных лидеров и мусора."""
    cleaned = _HTML_TAG_RE.sub(" ", text or "")
    cleaned = unescape(cleaned)
    cleaned = _DOTTED_LEADERS_RE.sub(" ", cleaned)
    cleaned = _OCR_LEADER_RE.sub(" ", cleaned)
    cleaned = _OCR_REPEAT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[\u00ad]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .-—–;:")


def _table_rows(html: str) -> List[List[str]]:
    """Разобрать ``text_as_html`` таблицы в список строк-ячеек (без тегов)."""
    rows: List[List[str]] = []
    for row_html in _HTML_ROW_RE.findall(html or ""):
        cells = [_clean_toc_line(cell)
                 for cell in _HTML_CELL_RE.findall(row_html)]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def _display_title(text: str) -> str:
    """Схлопнуть переносы/табы в названии раздела для отображения и id."""
    return re.sub(r"\s+", " ", text or "").strip()


def _looks_like_toc_name(name: str) -> bool:
    """Похоже ли имя на раздел, а не на OCR-мусор («еяа010-9 0», «млн»)."""
    stripped = (name or "").strip()
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", stripped)
    if len(letters) < 4:
        return False
    visible = [ch for ch in stripped if not ch.isspace()]
    if not visible or len(letters) / len(visible) < 0.45:
        return False
    return stripped.lower() not in {"млн", "тыс", "итого", "всего", "шт", "ед"}


def _similar_names(left: str, right: str) -> bool:
    """Похожи ли два имени (дедуп битых закладок и дублей OCR)."""
    left_key = normalize_section_name(left)
    right_key = normalize_section_name(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 8 and shorter in longer:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.68


def _very_similar_names(left: str, right: str) -> bool:
    """Сильно похожие имена (OCR-опечатки) — дедуп без привязки к странице."""
    left_key = normalize_section_name(left)
    right_key = normalize_section_name(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.88


def _strip_trailing_page(name: str) -> str:
    """Убрать хвостовой 1–3-значный номер страницы из имени."""
    return re.sub(r"\s+\d{1,3}$", "", name or "").strip()


def _add_toc_entry(entries: List["TocEntry"], name: str,
                   page: Optional[int]) -> None:
    """Добавить запись TOC с фильтрацией пустых/служебных строк.

    ``page=None`` допустим: в 2017 таблицы оглавления без номеров, страницу
    позже восстановим по совпадению с Title в теле документа.
    """
    name = _display_title(name)
    if len(name) < 2 or not _looks_like_toc_name(name):
        return
    if page is not None and not 1 <= int(page) < _TOC_MAX_PAGE:
        return
    if _TOC_JUNK.match(name):
        return
    entries.append(TocEntry(name=name,
                            page=int(page) if page is not None else None,
                            level=infer_title_level(name)))


def _consume_toc_line(entries: List["TocEntry"], state: Dict[str, Any],
                      line: str) -> None:
    """Обработать одну строку TOC, поддерживая «имя ↔ страница» через state.

    Поддерживаются три раскладки годовых отчётов:
    - ``Название .... 23`` (хвостовая страница);
    - ``23 НАЗВАНИЕ`` (ведущая страница, 2018/2019);
    - отдельная страница на строке (``7``) после имени (2015/2019).
    """
    line = _clean_toc_line(line)
    if not line or _TOC_JUNK.match(line):
        return

    if re.fullmatch(r"\d{1,3}", line):
        page = int(line)
        if state.get("name"):
            _add_toc_entry(entries, state["name"], page)
            state.clear()
        else:
            state["page"] = page
        return

    leading = re.match(r"^(\d{1,3})\s+(.{3,})$", line)
    if leading is not None:
        # «8 бизнес-моделъь 10»: первый номер — страница, второй — хвост
        # соседней колонки, его в имени быть не должно.
        _add_toc_entry(entries, _strip_trailing_page(leading.group(2)),
                       int(leading.group(1)))
        state.clear()
        return

    # «Название 23» и много-колоночный хвост «Название 23 27 58 76»:
    # берём первую страницу, остальные — суб-колонки/соседние разделы.
    trailing = re.match(r"^(.{2,}?)\s+(\d{1,3}(?:\s+\d{1,3})*)$", line)
    if trailing is not None:
        _add_toc_entry(entries, trailing.group(1),
                       int(trailing.group(2).split()[0]))
        state.clear()
        return

    # Строка без страницы: имя (возможно, первая часть многострочного имени).
    if state.get("page") is not None:
        _add_toc_entry(entries, line, state["page"])
        state.clear()
    elif state.get("name"):
        state["name"] = f"{state['name']} {line}"
    else:
        state["name"] = line


def _consume_toc_cells(entries: List["TocEntry"], state: Dict[str, Any],
                       cells: Sequence[str]) -> None:
    """Обработать ячейки одной строки TOC-таблицы."""
    if not cells:
        return
    if len(cells) == 1:
        _consume_toc_line(entries, state, cells[0])
        return
    if len(cells) == 2:
        left, right = cells
        if re.fullmatch(r"\d{1,3}", right):
            _add_toc_entry(entries, left, int(right))
            state.clear()
            return
        if re.fullmatch(r"\d{1,3}", left):
            _add_toc_entry(entries, right, int(left))
            state.clear()
            return
        # Разрыв имени по колонкам: «Обращение Президента, Председателя» |
        # «Правления .... 23» → одно имя + страница.
        trailing = re.match(r"^(.{2,}?)\s+(\d{1,3}(?:\s+\d{1,3})*)$", right)
        if trailing is not None and len(left) >= 3:
            combined = _display_title(f"{left} {trailing.group(1)}")
            _add_toc_entry(entries, combined,
                           int(trailing.group(2).split()[0]))
            state.clear()
            return
    for cell in cells:
        _consume_toc_line(entries, state, cell)


def _looks_like_toc_group(line: str) -> bool:
    """Похоже ли на заголовок части оглавления (не на строку «имя ... 23»)."""
    cleaned = _clean_toc_line(line)
    if not cleaned or len(cleaned) > 120:
        return False
    if not _looks_like_toc_name(cleaned):
        return False
    if _DOTTED_LEADERS_RE.search(line or ""):
        return False
    if re.search(r"\s\d{1,3}$", cleaned):
        return False
    return not _TOC_JUNK.match(cleaned)


#: Обрывки имён после межколоночной склейки: «Сбербанка и», «Прорывные ... в».
_DANGLING_END_RE = re.compile(
    r"(?:^|\s)(?:и|в|на|с|по|для|от|к|о|об|из|за|линии|ини|ии)$",
    re.IGNORECASE,
)
#: OCR-хвосты TOC-строк: «Структура персонала нии», «Достижения oo».
_TOC_TAIL_JUNK_RE = re.compile(
    r"\s+(?:линии|лии|ини|ии|нии|нк|oo|оо|о)$", re.IGNORECASE)
#: Висячий предлог/союз в конце имени: «Ключевые результаты и».
_TOC_TAIL_CONJ_RE = re.compile(
    r"\s+(?:и|в|на|с|по|для|от|к|о|об|из|за)$", re.IGNORECASE)



def _clean_toc_name(name: str) -> str:
    """Убрать OCR-хвосты и висячие союзы из имени строки оглавления."""
    cleaned = _display_title(name)
    cleaned = _TOC_TAIL_JUNK_RE.sub("", cleaned)
    cleaned = _TOC_TAIL_CONJ_RE.sub("", cleaned)
    return cleaned.strip(" .,:;-")


def _is_dangling_fragment(name: str) -> bool:
    """True для обрывка имени, который нельзя сделать разделом."""
    stripped = (name or "").strip()
    if not stripped:
        return True
    if stripped.endswith((",", ";", ":")):
        return True
    if re.match(r"^(?:и|в|на|с|по|для|от|к|о|об|из|за)\s",
                stripped, re.IGNORECASE):
        return True
    return bool(_DANGLING_END_RE.search(stripped))


#: Слова-обрывки TOC 2017: одиночные/парные хвосты и обрывы на предлоге.
_TOC_FRAGMENT_TAIL_ANY = {
    "сбербанка", "управления", "финансов", "отчетности",
}
_TOC_FRAGMENT_TAIL_SHORT = {"области", "повышении"}


def _is_fragment_tail(name: str) -> bool:
    """Короткое имя, оборванное на служебном слове («Цели в области»)."""
    words = (name or "").strip().split()
    if not words:
        return False
    last = words[-1].strip(".,:;").lower()
    if len(words) <= 2 and last in _TOC_FRAGMENT_TAIL_ANY:
        return True
    return len(words) <= 4 and last in _TOC_FRAGMENT_TAIL_SHORT




def _toc_entries_from_window(window: Sequence[RawElement]) -> List[TocEntry]:
    """Собрать записи TOC из окна элементов (от заголовка до +4 страниц)."""
    entries: List[TocEntry] = []
    state: Dict[str, Any] = {}
    group: Optional[str] = None
    pending_number: Optional[str] = None

    def _tag_new(before: int) -> None:
        """Проставить текущую часть новым записям окна."""
        for entry in entries[before:]:
            if entry.group is None and group is not None:
                entry.group = group

    for element in window:
        if element.html:
            before = len(entries)
            for cells in _table_rows(element.html):
                before_row = len(entries)
                _consume_toc_cells(entries, state, cells)
                # Строка таблицы 2017 часто без номера страницы (OCR потерял
                # лидер): сохраняем имя как кандидат, страницу восстановит
                # сопоставление с Title тела.
                if (len(entries) == before_row and state.get("name")
                        and state.get("page") is None):
                    name = state.pop("name")
                    _add_toc_entry(entries, name, None)
                    state.clear()
            _tag_new(before)
            state.clear()
            continue

        raw_text = element.text or ""
        cleaned = _clean_toc_line(raw_text)

        # «04.» отдельным блоком: номер относится к следующему заголовку части.
        if cleaned and re.fullmatch(r"\d{1,2}\.?", cleaned):
            pending_number = cleaned.rstrip(".")
            continue

        if element.type in {"Title", "ListItem", "UncategorizedText",
                            "Header"} and _looks_like_toc_group(raw_text):
            group = cleaned
            if pending_number is not None and not re.match(
                    r"^\d{1,2}\.", cleaned):
                group = f"{pending_number}. {cleaned}"
            pending_number = None
            state.clear()
            continue

        before = len(entries)
        for line in raw_text.splitlines():
            _consume_toc_line(entries, state, line)
        _tag_new(before)

    # Последнее имя без страницы (TOC без номеров, 2017) — тоже кандидат.
    if state.get("name") and state.get("page") is None:
        before = len(entries)
        _add_toc_entry(entries, state["name"], None)
        _tag_new(before)
    return entries



def _is_toc_header(text: str) -> bool:
    """True для заголовка «Содержание»/«Оглавление» (в т.ч. с OCR-мусором)."""
    stripped = (text or "").strip()
    if _TOC_HEADER_RE.match(stripped):
        return True
    return len(stripped) < 60 and bool(
        re.search(r"\b(СОДЕРЖАНИЕ|ОГЛАВЛЕНИЕ|CONTENTS)\b", stripped, re.IGNORECASE))


@dataclass
class TocEntry:
    """Запись оглавления: имя части и стартовая страница.

    Attributes:
        name: Название части (нормализуется по краям, без хвостовых точек).
        page: Начальная страница от TOC (первое число строки); None у заголовка.
        level: Уровень, вычисленный по нумерации имени («1.», «1.1.») или None.
        group: Имя части-родителя, если запись взята из таблицы под заголовком
            части TOC («Портрет Группы», «01. Портрет Группы»); None иначе.
    """

    name: str
    page: Optional[int] = None
    level: Optional[int] = None
    group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация (для артефактов и тестов)."""
        return asdict(self)


def parse_toc_entries(elements: Sequence[RawElement]) -> List[TocEntry]:
    """Разобрать страницы-«Содержание» в список ``TocEntry``.

    В годовых отчётах Сбер оглавление обычно стоит в первых страницах: заголовок
    «Содержание», затем строки вида ``Обзор результатов 23`` или
    ``1. Управление рисками 58``. Каждая строка при этом попадает отдельным
    элементом Unstructured — это и используется.

    Args:
        elements: Все элементы документа (или его начала) в порядке страниц.

    Returns:
        Записи оглавления (без самого заголовка «Содержание»); может быть
        пуста, если страницы-оглавления не найдены/не распознались.
    """
    entries: List[TocEntry] = []
    toc_pages = _toc_page_numbers(elements)
    for index, element in enumerate(elements):
        if not _is_toc_header(element.text):
            continue
        header_page = element.page
        window: List[RawElement] = []
        for following in elements[index + 1:]:
            if _is_toc_header(following.text):
                break
            # Окно — только реальные страницы оглавления (2017: одна стр. 3),
            # иначе тело попадает в TOC и даёт «разделы» из инфографики.
            if following.page not in toc_pages:
                break
            window.append(following)
        # Страницы самого оглавления не являются разделами. Номера страниц в
        # TOC — печатные, поэтому они могут быть МЕНЬШЕ PDF-страницы заголовка
        # (2016: «Сбербанк сегодня 4» при «Оглавлении» на 5-й странице). Такие
        # записи валидны; отбрасываем только совпадение с самой страницей TOC.
        entries.extend(
            entry for entry in _toc_entries_from_window(window)
            if entry.page is None
            or (entry.page >= 2 and entry.page != header_page))

    # Дедуп по нормализованному имени: TOC часто дублируется (Header + Title,
    # две колонки, повторный OCR одной страницы).
    by_key: Dict[str, TocEntry] = {}
    for entry in entries:
        key = normalize_section_name(entry.name)
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = entry
        elif existing.page is None and entry.page is not None:
            # Одна и та же строка встретилась без страницы и со страницей.
            by_key[key] = entry
    return list(by_key.values())


def toc_level_map(entries: Sequence[TocEntry]) -> Dict[str, int]:
    """Нормализованное имя → уровень (только для записей с нумерацией).

    Записи TOC без нумерации пропускаются: иначе все части получили бы level 0
    и затёрли более сильный bbox/outline-сигнал. Такие имена остаются только
    для page-навигации.
    """
    out: Dict[str, int] = {}
    for entry in entries:
        # TOC без явной нумерации НЕ даёт уровня: иначе все части станут 0 и
        # затрут более сильный bbox/outline-сигнал. Такие имена остаются для
        # page-навигации, но не для иерархии.
        if entry.level is None:
            continue
        out.setdefault(normalize_section_name(entry.name), entry.level)
    return out


def normalize_section_name(text: str) -> str:
    """Нормализация имени для сопоставления Title ↔ записи TOC.

    Схлопывает пробельные переносы, склеивает дефис-переносы («отчет- ность»),
    выкидывает пунктуацию, приводит к нижнему регистру.
    """
    collapsed = re.sub(r"[\s\u00ad]+", " ", text or "")
    collapsed = re.sub(r"([А-Яа-яЁёA-Za-z])- ", r"\1", collapsed)
    # Гомоглифы: OCR путает «В»/«B», «С»/«C» и т.п. Приводим к латинице,
    # чтобы «Экосистема В2В» и «Экосистема B2B» склеивались при дедупе.
    collapsed = collapsed.translate(_HOMOGLYPHS)
    collapsed = re.sub(r"[^0-9A-Za-zА-Яа-яЁё ]", " ", collapsed)
    return re.sub(r"\s+", " ", collapsed).strip().lower()


def is_junk_title(text: str) -> bool:
    """Проверка «фальшивых» заголовков, которые нельзя считать разделами.

    Из глаз-проверки 2015 hi_res:
    - строки списков членов советов/правлений (``1. ПРОКОПОВИЧ ПетР ПетРОВИЧ``)
      — детектор unstructured часто даёт им Title, но иерархию они рушат;
    - короткие «строки-ссылки» инфографики (``Перейти``, ``отчет``).
    """
    text = (text or "").strip()
    if not text or len(text) > 140:
        return True
    if text.lower() in {"перейти", "ссылка", "отчет", "обзор",
                        "содержание", "оглавление"}:
        return True
    if len(text) < 3:                           # «Y», «-», одиночные значки
        return True
    # Инфографика-«цифры»: строки, начинающиеся с не-бук/не-цифры («>5,7%», «~800»)
    if not re.match(r"^[А-ЯЁа-яёA-Za-z«\"0-9]", text):
        return True
    if text.count("%") and len(text) < 12:      # ">5,7%", ">22%"
        return True
    # Милстоуны: ведущая цифра БЕЗ нумерационного хвоста («93 АС», «116,5 ПБ»,
    # «3 PLATFORM V»); «1. Управление рисками» — не попадает (нумерация!).
    # Исключение — части 2018 без точки: «1 Портрет Группы» (Title Case).
    numbered_part = bool(re.match(r"^\d{1,2}\s+[A-ZА-ЯЁ][a-zа-яё]+", text))
    if len(text) < 25 and re.match(r"^\d[\d.,+~ ]", text) \
            and not _NUMBERED.match(text) and not numbered_part:
        return True
    # Одно-буквенное слово + число («В 15 раз»)
    if re.match(r"^[А-ЯЁA-Za-z]\s+\d", text):
        return True
    # Сокращённые стандарты-метки («GRI 3-3», «GRI 2-1, 2-3») — капс + цифра
    if re.match(r"^GRI[\s-]*\d", text, re.IGNORECASE):
        return True
    if len(text) <= 16 and any(ch.isdigit() for ch in text) \
            and _is_mostly_uppercase(text):
        return True
    # Служебная нумерация обложки/колонтитула: «2) БПС-СБЕРБАНК»
    if len(text) < 60 and re.match(r"^\d{1,2}\)\s+\S", text):
        return True
    # Строка-предложение, ошибочно распознанная как Title
    if re.match(r"^(Кроме того|При этом|Также|Таким образом|Помимо этого)\b",
                text, re.IGNORECASE):
        return True
    # Короткие OCR-метки инфографики: «ZN (9)»
    if re.fullmatch(r"[A-ZА-ЯЁ]{1,3}\s*\(\d{1,3}\)", text):
        return True
    # ФИО-строка после номера: «N. Фамилия Имя Отчество» (регистр любой — OCR
    # делает его несогласованным, поэтому punctuation-only фильтр)
    if re.match(r"^\d{1,2}[.)]\s+\S+(?:\s+\S+){1,2}\s*$", text):
        words = re.findall(r"[А-ЯЁа-яё]{3,}", text)
        if len(words) >= 3 and not re.search(r"[•|#;:/\"']", text):
            return True
    return False


def _best_match_level(stripped: str,
                      pairs: Sequence[Tuple[str, int]]) -> Optional[int]:
    """Уровень заголовка по предвычисленным парам «нормализованное имя → уровень».

    Точный матч делает вызывающий код по словарю; здесь — подстрочный (fuzzy),
    нужный для межколоночных склеек («Кредитные рейтинги Обзор рынка»).
    Гварды: совпадающая часть короче 8 символов — не матч, иначе «ROE» и прочие
    короткие имена тянут хвост на всё.

    Args:
        stripped: Текст Title-заголовка.
        pairs: Список ``(нормализованное имя, уровень)``.

    Returns:
        Уровень из источника или None (нет совпадения).
    """
    norm = normalize_section_name(stripped)
    if len(norm) < 8:
        return None
    for entry_norm, level in pairs:
        # Пары могут прийти как нормализованными, так и сырыми — нормализация
        # идемпотентна, поэтому безопасно применить её ещё раз.
        entry_norm = normalize_section_name(entry_norm)
        if norm in entry_norm and len(norm) >= 8:
            return level
        if len(entry_norm) >= 8 and entry_norm in norm:
            return level
    return None


def _best_toc_level(stripped: str,
                    toc_norm: Sequence[Tuple[str, int]]) -> Optional[int]:
    """Уровень заголовка по TOC: точный матч уже сделан, здесь — fuzzy.

    Обёртка над ``_best_match_level`` (сохранено имя для тестов и читаемости).

    Args:
        stripped: Текст Title-заголовка.
        toc_norm: Предвычисленный список ``(нормализованное имя TOC, уровень)``.

    Returns:
        Уровень из TOC или None (нет совпадения).
    """
    return _best_match_level(stripped, toc_norm)


def _outline_level_pairs(outline: Sequence[Any]) -> List[Tuple[str, int]]:
    """Пары ``(нормализованное имя закладки, уровень)`` из outline.

    Принимает и ``OutlineEntry`` (``pdf_outline``), и словари артефакта — так
    ``structure_builder`` не импортирует ``pdf_outline`` (нет цикла), а тесты
    могут передавать простые структуры.

    Args:
        outline: Закладки PDF в порядке документа.

    Returns:
        Список пар для точного/fuzzy-матча с Title-элементами.
    """
    pairs: List[Tuple[str, int]] = []
    for entry in outline:
        if isinstance(entry, Mapping):
            title, level = entry.get("title"), entry.get("level")
        else:
            title, level = getattr(entry, "title", None), getattr(entry, "level", None)
        if not title:
            continue
        norm = normalize_section_name(str(title))
        if norm:
            pairs.append((norm, int(level or 0)))
    return pairs


def read_outline_safely(pdf_path: Optional[Path]) -> List[Any]:
    """Прочитать закладки PDF, не роняя сборку структуры.

    Импорт ``pdf_outline`` локальный: он сам импортирует этот модуль, поэтому
    обратный импорт на уровне модуля дал бы цикл.

    Args:
        pdf_path: Путь к PDF (None → пустой список).

    Returns:
        Закладки ``OutlineEntry`` или пустой список (нет PDF/битый outline).
    """
    if pdf_path is None:
        return []
    try:
        from src.ingestion.pdf_outline import read_outline

        return read_outline(Path(pdf_path))
    except Exception:  # noqa: BLE001 — структура строится и без закладок
        return []


def build_from_elements_file(elements_json: Path,
                             pdf_path: Optional[Path] = None,
                             outline: Optional[Sequence[Any]] = None,
                             ) -> DocumentStructure:
    """Построить DocumentStructure из артефакта парсера (``*_elements.json``).

    Используется CLI-режимом ``--structure`` (шаг 2 плана, задача 8.6):
    structure_builder читает ТОЛЬКО элементы артефакта — пере парсинг не нужен.
    Закладки PDF (приоритетный источник уровней) читаются из ``pdf_path`` или
    принимаются готовыми через ``outline``.

    Args:
        elements_json: Путь к ``{doc_id}_elements.json``.
        pdf_path: Исходный PDF (из него читается outline, если не передан).
        outline: Готовые закладки (``OutlineEntry``); None → читаются из PDF.

    Returns:
        ``DocumentStructure`` с уровнями по приоритету outline → TOC → эвристика.

    Raises:
        FileNotFoundError: если файла элементов нет.
    """
    import json

    from src.core.models import RawElement as _RawElement

    path = Path(elements_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт парсера не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    elements = [_RawElement.from_dict(item) for item in raw]
    toc = parse_toc_entries(elements)
    if outline is None:
        outline = read_outline_safely(pdf_path)
    return build_document_structure(elements, toc=toc, outline=outline)


def save_sections(structure: DocumentStructure, structure_dir: Path) -> Path:
    """Сохранить артефакт ``{doc_id}_sections.json`` (аналогично parser-уровню).

    Args:
        structure: Готовая структура документа.
        structure_dir: Каталог ``data/structure`` (создаётся при отсутствии).

    Returns:
        Путь к записанному файлу.
    """
    structure_dir = Path(structure_dir)
    target = structure_dir / f"{structure.doc_id}_sections.json"
    atomic_write_json(target, structure.to_dict(), indent=1)
    return target


def load_sections(sections_json: Path) -> DocumentStructure:
    """Прочитать артефакт ``{doc_id}_sections.json``.

    Args:
        sections_json: Путь к файлу разделов.

    Returns:
        ``DocumentStructure`` (дерево восстанавливается по parent_id).

    Raises:
        FileNotFoundError: если файла нет.
    """
    import json

    path = Path(sections_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт разделов не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        return DocumentStructure.from_dict(json.load(fh))


def _resolve_level(stripped: str, normalized: str,
                   outline_map: Mapping[str, int],
                   outline_pairs: Sequence[Tuple[str, int]],
                   toc_map: Mapping[str, int],
                   toc_norm: Sequence[Tuple[str, int]]) -> Optional[int]:
    """Выбрать уровень по приоритету: outline → TOC → None (эвристика).

    Args:
        stripped: Текст Title-заголовка.
        normalized: Нормализованный текст Title.
        outline_map: Точная карта «имя закладки → уровень».
        outline_pairs: Пары для fuzzy-матча по закладкам.
        toc_map: Точная карта «имя TOC → уровень».
        toc_norm: Пары для fuzzy-матча по TOC.

    Returns:
        Уровень или None, если ни один источник не дал совпадения.
    """
    if normalized in outline_map:
        return outline_map[normalized]
    level = _best_match_level(stripped, outline_pairs)
    if level is not None:
        return level
    if normalized in toc_map:
        return toc_map[normalized]
    return _best_match_level(stripped, toc_norm)


def _cluster_1d(values: Sequence[float], k: int) -> Dict[float, int]:
    """1D k-means для высот заголовков; возвращает «значение → уровень».

    Уровень 0 — самый высокий блок (крупный кегль/часть отчёта), далее по
    убыванию. Кластеризация устойчивее фиксированных порогов: в разных годах
    страницы и кегли отличаются, а соотношение высот заголовков сохраняется.
    """
    unique = sorted({round(float(value), 4) for value in values})
    if len(unique) <= 1:
        return {}
    k = max(2, min(k, len(unique)))
    centers = [unique[int((index + 0.5) * len(unique) / k)] for index in range(k)]
    for _ in range(30):
        groups: List[List[float]] = [[] for _ in range(k)]
        for value in values:
            nearest = min(range(k), key=lambda j: abs(value - centers[j]))
            groups[nearest].append(value)
        updated = [sum(group) / len(group) if group else centers[j]
                   for j, group in enumerate(groups)]
        if all(abs(updated[j] - centers[j]) < 1e-7 for j in range(k)):
            centers = updated
            break
        centers = updated
    # Большая высота = более верхний уровень (0).
    order = sorted(range(k), key=lambda j: centers[j], reverse=True)
    rank = {cluster: level for level, cluster in enumerate(order)}
    assignment: Dict[float, int] = {}
    for value in values:
        nearest = min(range(k), key=lambda j: abs(value - centers[j]))
        assignment[round(float(value), 4)] = rank[nearest]
    return assignment


def _toc_page_numbers(elements: Sequence[RawElement]) -> set[int]:
    """Страницы оглавления — только те, где реально есть TOC-строки.

    Раньше брались ``header .. header+4``, из-за чего тело (2016: стр. 7–9)
    ошибочно исключалось из дерева. Теперь окно растёт, пока на странице есть
    точечные лидеры/хвостовые номера, и обрывается на первой странице тела.
    """
    pages: set[int] = set()
    order: List[int] = []
    for element in elements:
        if element.page not in order:
            order.append(element.page)
    signals: Dict[int, int] = {}
    totals: Dict[int, int] = {}
    for element in elements:
        totals[element.page] = totals.get(element.page, 0) + 1
        text = element.text or ""
        if _DOTTED_LEADERS_RE.search(text) or (
                element.html and _DOTTED_LEADERS_RE.search(element.html)):
            signals[element.page] = signals.get(element.page, 0) + 1
        elif re.search(r"\s\d{1,3}$", (text or "").strip()):
            signals[element.page] = signals.get(element.page, 0) + 1

    for element in elements:
        if not _is_toc_header(element.text):
            continue
        pages.add(element.page)
        try:
            position = order.index(element.page)
        except ValueError:
            continue
        for page in order[position + 1:]:
            if page > element.page + 4:
                break
            count = totals.get(page, 0)
            if count and signals.get(page, 0) / count >= 0.4:
                pages.add(page)
            else:
                break
    return pages



def _height_level_map(elements: Sequence[RawElement]) -> Dict[str, int]:
    """Уровни заголовков по высоте bbox (сигнал, не зависящий от TOC/outline).

    Нужен для 5 из 9 отчётов без закладок и для случаев, когда TOC не
    распознался. Требует схему v2 (``bbox`` + ``page_height``); на старых
    артефактах просто возвращает пустую карту.
    """
    items: List[Tuple[str, float]] = []
    for element in elements:
        if element.type != "Title" or is_junk_title(element.text):
            continue
        if not element.bbox or not element.page_height:
            continue
        height = float(element.bbox[3]) - float(element.bbox[1])
        if height <= 0:
            continue
        items.append((element.element_id, height / float(element.page_height)))
    if len(items) < 3:
        return {}
    ratios = [ratio for _, ratio in items]
    assignment = _cluster_1d(ratios, k=3)
    if len(set(assignment.values())) < 2:
        return {}
    levels: Dict[str, int] = {}
    for element_id, ratio in items:
        level = assignment.get(round(ratio, 4))
        if level is not None:
            levels[element_id] = level
    return levels


def _outline_parts(item: Any) -> Tuple[str, int, Optional[int]]:
    """Достать ``(title, level, page)`` из закладки-объекта или словаря."""
    if isinstance(item, Mapping):
        return (str(item.get("title", "")), int(item.get("level", 0)),
                item.get("page"))
    return (str(getattr(item, "title", "")), int(getattr(item, "level", 0)),
            getattr(item, "page", None))


def _resolve_title(title_index: Optional[Mapping[str, Tuple[str, int]]],
                   key: str) -> Optional[Tuple[str, int]]:
    """Найти Title в теле по нормализованному имени (точное, затем fuzzy)."""
    if not title_index:
        return None
    exact = title_index.get(key)
    if exact is not None:
        return exact
    for title_key, value in title_index.items():
        if _similar_names(key, title_key):
            return value
    return None


def _calibrate_toc_offset(toc: Sequence["TocEntry"],
                          title_index: Optional[Mapping[str, Tuple[str, int]]]
                          ) -> int:
    """Сдвиг печатных страниц TOC относительно PDF (2016: +3).

    В части отчётов оглавление ссылается на печатную нумерацию, а элементы
    парсера — на PDF-страницы. Сдвиг оценивается по точным совпадениям имён
    TOC и Title тела; применяется только при устойчивом большинстве.
    """
    if not toc or not title_index:
        return 0
    offsets: List[int] = []
    for entry in toc:
        if entry.page is None or not entry.name:
            continue
        resolved = title_index.get(normalize_section_name(entry.name))
        if resolved is None:
            continue
        offsets.append(int(resolved[1]) - int(entry.page))
    if len(offsets) < 3:
        return 0
    offset, support = Counter(offsets).most_common(1)[0]
    if support < max(3, len(offsets) // 3):
        return 0
    if not -20 <= offset <= 20:
        return 0
    return int(offset)


def _skeleton_entries(toc: Sequence["TocEntry"], outline: Sequence[Any],
                      last_page: Optional[int] = None,
                      title_index: Optional[Mapping[str, Tuple[str, int]]] = None
                      ) -> List[Tuple[str, int, int]]:
    """Собрать «скелет» разделов из закладок PDF и/или оглавления.

    Сырые ``Title`` от Unstructured содержат десятки рекламных подзаголовков и
    наград, поэтому деревом они быть не могут. Оглавление/закладки — это
    собственный «контракт структуры» отчёта: имена и страницы уже выверены.

    Для 2016–2018 добавлено:
    - калибровка печатных страниц TOC по Title тела (иначе 2016 смещён на 3);
    - части оглавления (``TocEntry.group``) как level 0, их строки — level 1;
    - плоские закладки (2018: 162 из 172 одного уровня) переуровниваются по
      нумерации заголовков (``1 Портрет Группы`` → 0, ``3.1 ...`` → 1).
    """
    toc_offset = _calibrate_toc_offset(toc, title_index)

    outline_all: List[Tuple[str, int, int, int]] = []
    for index, item in enumerate(outline or []):
        title, level, page = _outline_parts(item)
        if not title or page is None or is_junk_title(title):
            continue
        if not _looks_like_toc_name(title):
            continue
        outline_all.append((_display_title(title), max(0, min(level, 3)),
                            int(page), index))
    outline_total = len(outline_all)

    # Плоские закладки (2018) несут уровень 0 у почти всех пунктов. Иерархию
    # восстанавливаем по нумерации: явный номер задаёт уровень, следующие без
    # номера — дети последнего явного пункта.
    if outline_total >= 10:
        level_counts = Counter(item[1] for item in outline_all)
        dominant_level, dominant_count = level_counts.most_common(1)[0]
        if dominant_count / outline_total >= 0.85:
            relabeled: List[Tuple[str, int, int, int]] = []
            last_explicit: Optional[int] = None
            for title, _level, page, index in outline_all:
                explicit = infer_title_level(title)
                if explicit is None and re.match(
                        r"^\d{1,2}\s+[A-ZА-ЯЁ][a-zа-яё]+", title):
                    # 2018: части набраны «1 Портрет Группы», «2 Стратегический
                    # отчет» — без точки, но это те же level 0.
                    explicit = 0
                if explicit is not None:
                    level = explicit
                    last_explicit = explicit
                elif last_explicit is not None:
                    level = min(last_explicit + 1, 3)
                else:
                    level = dominant_level
                relabeled.append((title, level, page, index))
            outline_all = relabeled

    outline_raw = [item for item in outline_all
                   if last_page is None or item[2] <= last_page]

    toc_raw: List[Tuple[str, int, int, float]] = []
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for index, entry in enumerate(toc or []):
        if not entry.name or is_junk_title(entry.name):
            continue
        clean_name = _clean_toc_name(entry.name)
        if not clean_name:
            continue
        key = normalize_section_name(clean_name)
        if not key:
            continue
        page = entry.page
        if page is not None:
            page = int(page) + toc_offset
        display_name = clean_name
        resolved = _resolve_title(title_index, key)
        if resolved is not None:
            body_name, body_page = resolved
            # Заменяем обрывок TOC («Сбербанка и», «в 2017 году») полным
            # заголовком из тела; страницу берём из тела, если TOC её не дала.
            if page is None or abs(int(page) - int(body_page)) <= 2:
                display_name = body_name
                page = body_page
        if page is None:
            continue
        if last_page is not None and int(page) > last_page:
            continue
        # Обрывок имени остаётся обрывком, даже если в теле нашёлся такой же
        # битый Title («в 2017 году», «Сбербанка и»).
        if _is_dangling_fragment(display_name):
            continue
        # Короткий обрывок на служебном слове («Сбербанка», «Цели в области»)
        # — не раздел. Полные заголовки из тела (3+ слова) сохраняем.
        if entry.group and _is_fragment_tail(display_name):
            continue
        if entry.group:
            group_name = _display_title(entry.group)
            groups.setdefault(group_name, []).append((int(page), index))
        if entry.level is not None:
            level = max(0, min(entry.level, 3))
        elif entry.group:
            level = 1
        else:
            level = 0
        toc_raw.append((display_name, level, int(page), float(index)))

    # Части TOC → level 0. Начало части — первая по порядку чтения строка.
    # Если строка помечена частью, но её страница РАНЬШЕ начала части (2016:
    # таблица со стр. 6 прочитана после «Приложений», но относится к
    # «Стратегическому отчету»), перекладываем её по диапазону страниц.
    group_starts: List[Tuple[int, float, str]] = []
    if groups:
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: min(index for _page, index in item[1]))
        last_start = -1
        for group_name, members in ordered_groups:
            page, first_index = min(members, key=lambda item: item[1])
            page = max(int(page), last_start)
            group_starts.append((page, float(first_index), group_name))
            last_start = page
        starts_by_name = {name: start for start, _first, name in group_starts}
        leveled: List[Tuple[str, int, int, float]] = []
        for title, level, page, order in toc_raw:
            if level == 0 and _NUMBERED.match(title):
                leveled.append((title, level, page, order))
                continue
            original = next((name for name, members in groups.items()
                             if (page, order) in members), None)
            parent: Optional[str] = original
            original_start = starts_by_name.get(original) if original else None
            if original is None or (original_start is not None
                                    and original_start > page):
                parent = None
                for start, _first, group_name in group_starts:
                    if start <= page:
                        parent = group_name
                    else:
                        break
            leveled.append((title, 1 if parent else level, page, order))
        toc_raw = leveled

    for start, first_index, group_name in group_starts:
        toc_raw.append((group_name, 0, int(start), float(first_index) - 0.5))

    # Один источник — основной. Закладки чище и полнее; TOC используем, когда
    # закладок нет, или чтобы закрыть ранние страницы, которые закладки 2018
    # начинают только со стр. 168. Смешивать оба «в лоб» — источник двойных
    # корней и OCR-мусора (2023: TOC-части + outline-подразделы).
    if outline_total >= 20:
        primary = list(outline_raw)
        first_outline_page = min((item[2] for item in outline_all),
                                 default=10 ** 9)
        if first_outline_page > 10:
            early_toc = [item for item in toc_raw
                         if item[2] < first_outline_page]
            primary = early_toc + primary
    elif toc_raw:
        primary = toc_raw
    else:
        return []

    primary.sort(key=lambda item: (item[2], item[3]))
    result: List[Tuple[str, int, int]] = []
    for title, level, page, _ in primary:
        # Нумерованные части («4 Корпоративное управление») повторяют имя
        # раннего подраздела, но это другой раздел — дедуп к ним не применяем.
        numbered_part = bool(_NUMBERED.match(title)) or bool(
            re.match(r"^\d{1,2}\s+[A-ZА-ЯЁ][a-zа-яё]+", title))
        if not numbered_part and any(
                (_very_similar_names(title, existing[0])
                 or (_similar_names(title, existing[0])
                     and abs(page - existing[2]) <= 2))
                for existing in result):
            continue
        result.append((title, level, page))
    return result


def _build_from_skeleton(entries: Sequence[Tuple[str, int, int]],
                         doc_id: str, last_page: int) -> "DocumentStructure":
    """Построить дерево из готового скелета ``(title, level, page)``."""
    sections: List[SectionNode] = []
    stack: List[SectionNode] = []
    for title, level, page in entries:
        stack = [node for node in stack if node.level < level]
        breadcrumbs = [node.title for node in stack] + [title]
        node = SectionNode(
            id=section_path(breadcrumbs), title=title, level=level,
            page_start=page, page_end=page, breadcrumbs=breadcrumbs,
            parent_id=stack[-1].id if stack else None)
        if stack:
            stack[-1].children_ids.append(node.id)
        stack.append(node)
        sections.append(node)
    for current, following in zip(sections, sections[1:]):
        current.page_end = max(current.page_start, following.page_start)
    if sections:
        sections[-1].page_end = max(sections[-1].page_start, last_page)
    return DocumentStructure(doc_id=doc_id, sections=sections)


def build_document_structure(elements: Sequence[RawElement],
                             doc_id: Optional[str] = None,
                             toc: Optional[Sequence[TocEntry]] = None,
                             outline: Optional[Sequence[Any]] = None,
                             ) -> DocumentStructure:
    """Строит дерево разделов (spec_project.md §5.1, адаптировано под depth=None).

    Приоритет источников уровня: PDF outline (закладки) → TOC (точный, затем
    fuzzy) → эвристика по тексту Title.

    Args:
        elements: Плоский список RawElement в порядке документа; Title = разделы.
        doc_id: Идентификатор документа; None → берётся из первого элемента.
        toc: Записи оглавления (``parse_toc_entries``); имя Title, совпавшее с
            записью TOC по нормализованному имени, получает уровень из TOC —
            это главный корректор локальных эвристик (нумерация/капс).
        outline: Закладки PDF (``OutlineEntry`` из ``pdf_outline`` или словари
            ``{"title", "level"}``) — самый точный источник уровней.

    Returns:
        ``DocumentStructure`` с проставленными page_start/page_end. Если
        ``_skeleton_entries`` дал ≥5 записей, дерево строится только по
        outline/TOC; иначе уровни Title уточняются по TOC и эвристикам.
    """
    if doc_id is None and elements:
        doc_id = elements[0].doc_id
    height_levels = _height_level_map(elements)
    last_page = max((element.page for element in elements), default=0)
    toc_pages = _toc_page_numbers(elements)
    title_index: Dict[str, Tuple[str, int]] = {}
    for element in elements:
        if element.type != "Title" or element.page in toc_pages:
            continue
        text = _display_title(element.text)
        if not text or is_junk_title(text):
            continue
        title_index.setdefault(normalize_section_name(text),
                               (text, element.page))
    skeleton = _skeleton_entries(toc or [], outline or [],
                                 last_page=last_page or None,
                                 title_index=title_index)
    if len(skeleton) >= 5:
        return _build_from_skeleton(skeleton, doc_id or "", last_page)
    toc_map = toc_level_map(toc) if toc else {}
    # Предвычисленный (name, level) для fuzzy-матча (см. _best_match_level)
    toc_norm: List[Tuple[str, int]] = [
        (normalize_section_name(e.name), e.level or 0) for e in (toc or [])]
    outline_pairs = _outline_level_pairs(outline or [])
    outline_map: Dict[str, int] = {}
    for norm, level in outline_pairs:
        outline_map.setdefault(norm, level)
    sections: List[SectionNode] = []
    stack: List[SectionNode] = []   # stack[level] = последний открытый раздел
    last_inferred = 0
    last_added: Optional[str] = None   # нормализованное имя прошлого раздела
    last_page = -1
    for element in elements:
        if element.type != "Title" or not (element.text or "").strip():
            continue
        if element.page in toc_pages:
            continue
        stripped = element.text.strip()
        if is_junk_title(stripped):
            continue
        # Дедуп: одинаковые заголовки подряд на одной странице (двойная считка)
        normalized = normalize_section_name(stripped)
        if normalized != "" and normalized == last_added and element.page <= last_page + 1:
            continue
        level = _resolve_level(stripped, normalized,
                               outline_map, outline_pairs, toc_map, toc_norm)
        if level is None:
            explicit = infer_title_level(stripped)
            if explicit is not None:
                level = explicit
            elif element.element_id in height_levels:
                level = height_levels[element.element_id]
            else:
                level = last_inferred
        last_inferred = level
        # Уровни глубже нового закрываются: «сёстры» → «дети» одного родителя.
        # Фильтр по level, а не срез [:level]: на срезе без корневых заголовков
        # стек разрежен (например, только level 1), и соседний заголовок того же
        # уровня не должен становиться родителем.
        stack = [node for node in stack if node.level < level]
        breadcrumbs = [s.title for s in stack] + [element.text.strip()]
        node = SectionNode(
            id=section_path(breadcrumbs),
            title=element.text.strip(),
            level=level,
            page_start=element.page,
            page_end=element.page,          # допишется по следующему разделу
            breadcrumbs=breadcrumbs,
            parent_id=stack[-1].id if stack else None,
        )
        if stack:
            stack[-1].children_ids.append(node.id)
        stack.append(node)
        sections.append(node)
        last_added = normalized
        last_page = element.page

    # page_end: раздел живёт до следующего любого уровня (граница по разделам)
    for current, following in zip(sections, sections[1:]):
        current.page_end = max(current.page_start, following.page_start)

    return DocumentStructure(doc_id=doc_id or "", sections=sections)
