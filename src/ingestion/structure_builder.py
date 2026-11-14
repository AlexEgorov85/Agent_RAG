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
уровень живёт только в ``SectionNode``. Курсовой контракт чанка не меняется:
``section_path``/``breadcrumbs`` — опциональные поля поверх Занятие 2-контракта
(см. `spec_project.md` §5.2)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class DocumentStructure:
    """Плоский список разделов документа; дерево восстанавливается по parent_id.

    Attributes:
        doc_id: Идентификатор документа (как в курсовом реестре, ``sber_2015``).
        sections: Разделы в порядке появления в документе.
    """

    doc_id: str
    sections: List[SectionNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация (для артефакта ``{doc_id}_sections.json``)."""
        return {"doc_id": self.doc_id,
                "sections": [s.to_dict() for s in self.sections]}

    def render_tree(self) -> str:
        """Многострочное представление дерева для глаз-проверки (отступ × уровень)."""
        lines = [
            "  " * s.level + f"[{s.level}] {s.title} (стр. {s.page_start})"
            for s in self.sections
        ]
        return "\n".join(lines)


def _section_path(breadcrumbs: Sequence[str]) -> str:
    """Из breadcrumbs делает строковый ключ: ``раздел/подраздел``."""
    return "/".join(breadcrumbs)


# ===================== Оглавление (второй источник уровней) ==============
#: Заголовок оглавления (варианты набора в разных годах)
_TOC_HEADER_RE = re.compile(r"^\s*СОДЕРЖАНИ[ЕИЮ]\s*$", re.IGNORECASE)
#: Строка TOC: «< название > < номер страницы >(+ числа суб-столбцов)»
_TOC_ENTRY_RE = re.compile(
    r"^(?P<name>.{2,80}?)\s+(?P<pages>\d{1,3}(?:\s+\d{1,3})*)$",
    re.DOTALL,
)
#: Максимальное «значение страницы» TOC (годы вида 2023 не считаются)
_TOC_MAX_PAGE = 2000


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


@dataclass
class TocEntry:
    """Запись оглавления: имя части и стартовая страница.

    Attributes:
        name: Название части (нормализуется по краям, без хвостовых точек).
        page: Начальная страница от TOC (первое число строки); None у заголовка.
        level: Уровень, вычисленный по нумерации имени («1.», «1.1.») или None.
    """

    name: str
    page: Optional[int] = None
    level: Optional[int] = None

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
    in_toc = False
    consecutive_misses = 0
    for element in elements:
        text = (element.text or "").strip()
        if not text:
            continue
        if _TOC_HEADER_RE.match(text):
            in_toc = True
            consecutive_misses = 0
            continue
        if not in_toc:
            continue
        match = _TOC_ENTRY_RE.match(text)
        if match is not None:
            name = match.group("name").strip(" .-—–;")
            if name:
                entries.append(TocEntry(name=name,
                                        page=int(match.group("pages").split()[0]),
                                        level=infer_title_level(name)))
            consecutive_misses = 0
            continue
        # Блоки «вслитую» TOC (Table/Image): жадный скан «имя -> номер»
        scanned = _scan_toc_text(text)
        if scanned:
            entries.extend(scanned)
            consecutive_misses = 0
            continue
        # Строки без хвостового номера: продолжение имени (перенос) или конец
        consecutive_misses += 1
        if consecutive_misses >= 2 and entries:
            # Прекращаем сбор построчно: два подряд «не-TOC» элемента
            in_toc = False
    return entries


def toc_level_map(entries: Sequence[TocEntry]) -> Dict[str, int]:
    """Нормализованное имя → уровень (по нумерации имени внутри TOC).

    Записи TOC без нумерации считаются верхними частями (level 0): в годовых
    отчётах крупные части обоходятся без цифр, а числовые подразделы
    перечисляются с номерами — это и есть различающий сигнал.
    """
    out: Dict[str, int] = {}
    for entry in entries:
        out[normalize_section_name(entry.name)] = entry.level or 0
    return out


def normalize_section_name(text: str) -> str:
    """Нормализация имени для сопоставления Title ↔ записи TOC.

    Схлопывает пробельные переносы, склеивает дефис-переносы («отчет- ность»),
    выкидывает пунктуацию, приводит к нижнему регистру.
    """
    collapsed = re.sub(r"[\s\u00ad]+", " ", text or "")
    collapsed = re.sub(r"([А-Яа-яЁёA-Za-z])- ", r"\1", collapsed)
    collapsed = re.sub(r"[^0-9A-Za-zА-Яа-яЁё ]", " ", collapsed)
    return re.sub(r"\s+", " ", collapsed).strip().lower()


def _infer_stack_level(text: str, last_inferred: int) -> int:
    """Определить уровень заголовка: явная эвристика или фолбэк «наследует».

    Явный сигнал (нумерация/капс) доверяем как есть. Без сигнала заголовок
    продолжает размётку последнего уровня, но не «ныряет» глубже одного шага
    за раз и не уходит глубже уровня 2 — иначе случайные строки глубоко рушат
    дерево.
    """
    inferred = infer_title_level(text)
    if inferred is not None:
        return inferred
    return min(last_inferred + 1, 2)


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
    if text.lower() in {"перейти", "ссылка", "отчет", "обзор"}:
        return True
    if len(text) < 3:                           # «Y», «-», одиночные значки
        return True
    # Инфографика-«цифры»: строки, начинающиеся с не-бук/не-цифры («>5,7%», «~800»)
    if not re.match(r"^[А-ЯЁа-яёA-Za-z«\"0-9]", text):
        return True
    if text.count("%") and len(text) < 12:      # ">5,7%", ">22%"
        return True
    # Милстоуны: ведущая цифра БЕЗ нумерационного хвоста («93 АС», «116,5 ПБ»,
    # «3 PLATFORM V»); «1. Управление рисками» — не попадает (нумерация!)
    if len(text) < 25 and re.match(r"^\d[\d.,+~ ]", text) \
            and not _NUMBERED.match(text):
        return True
    # Одно-буквенное слово + число («В 15 раз»)
    if re.match(r"^[А-ЯЁA-Za-z]\s+\d", text):
        return True
    # Сокращённые стандарты-метки («GRI 3-3») — капс + цифра, короткие
    if len(text) < 12 and any(ch.isdigit() for ch in text) \
            and _is_mostly_uppercase(text):
        return True
    # ФИО-строка после номера: «N. Фамилия Имя Отчество» (регистр любой — OCR
    # делает его несогласованным, поэтому punctuation-only фильтр)
    if re.match(r"^\d{1,2}[.)]\s+\S+(?:\s+\S+){1,2}\s*$", text):
        words = re.findall(r"[А-ЯЁа-яё]{3,}", text)
        if len(words) >= 3 and not re.search(r"[•|#;:/\"']", text):
            return True
    return False


def _best_toc_level(stripped: str,
                    toc_norm: Sequence[Tuple[str, int]]) -> Optional[int]:
    """Уровень заголовка по TOC: точный матч или подстрочный (fuzzy), guard длины.

    Года-склейки меж колоночных TOC-строк («Кредитные рейтинги Обзор рынка»)
    ломают точный матч; подстрочный (имя целиком внутри TOC-имени или наоборот)
    вытаскивает их. Гварды: совпадающая часть короче 8 символов — не матч,
    иначе «ROE» и прочие коротачки тянут хвост на всё.

    Args:
        stripped: Текст Title-заголовка.
        toc_norm: Предвычисленный список ``(нормализованное имя TOC, уровень)``.

    Returns:
        Уровень из TOC или None (нет совпадения).
    """
    norm = normalize_section_name(stripped)
    if len(norm) < 8:
        return None
    for entry_norm, level in toc_norm:
        if norm in entry_norm and len(norm) >= 8:
            return level
        if len(entry_norm) >= 8 and entry_norm in norm:
            return level
    return None


def build_from_elements_file(elements_json: Path) -> DocumentStructure:
    """Построить DocumentStructure из артефакта парсера (``*_elements.json``).

    Используется CLI-режимом ``--structure`` (шаг 2 плана, задача 8.6):
    structure_builder читает ТОЛЬКО элементы артефакта — пере парсинг не нужен.

    Args:
        elements_json: Путь к ``{doc_id}_elements.json``.

    Returns:
        ``DocumentStructure`` (TOC-калибровка не применяется: артефакт
        содержит весь документ, TOC-строки найдутся в нём сами).

    Raises:
        FileNotFoundError: если файла нет.
    """
    import json

    from src.core.models import RawElement as _RawElement

    path = Path(elements_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт парсера не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    elements = [_RawElement.from_dict(item) for item in raw]
    return build_document_structure(elements)


def save_sections(structure: DocumentStructure, structure_dir: Path) -> Path:
    """Сохранить артефакт ``{doc_id}_sections.json`` (аналогично parser-уровню).

    Args:
        structure: Готовая структура документа.
        structure_dir: Каталог ``data/structure`` (создаётся при отсутствии).

    Returns:
        Путь к записанному файлу.
    """
    import json
    import os
    import tempfile
    from pathlib import Path

    structure_dir = Path(structure_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)
    target = structure_dir / f"{structure.doc_id}_sections.json"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json",
        dir=structure_dir, delete=False)
    try:
        json.dump(structure.to_dict(), handle, ensure_ascii=False, indent=1)
        handle.flush()
    finally:
        handle.close()
    os.replace(handle.name, target)
    return target


def build_document_structure(elements: Sequence[RawElement],
                             doc_id: Optional[str] = None,
                             toc: Optional[Sequence[TocEntry]] = None,
                             ) -> DocumentStructure:
    """Строит дерево разделов (spec_project.md §5.1, адаптировано под depth=None).

    Args:
        elements: Плоский список RawElement в порядке документа; Title = разделы.
        doc_id: Идентификатор документа; None → берётся из первого элемента.
        toc: Записи оглавления (``parse_toc_entries``); имя Title, совпавшее с
            записью TOC по нормализованному имени, получает уровень из TOC —
            это главный корректор локальных эвристик (нумерация/капс).

    Returns:
        ``DocumentStructure`` с проставленными page_start/page_end.
    """
    if doc_id is None and elements:
        doc_id = elements[0].doc_id
    toc_map = toc_level_map(toc) if toc else {}
    # Предвычисленный (name, level) для fuzzy-матча (см. _best_toc_level)
    toc_norm: List[Tuple[str, int]] = [
        (normalize_section_name(e.name), e.level or 0) for e in (toc or [])]
    sections: List[SectionNode] = []
    stack: List[SectionNode] = []   # stack[level] = последний открытый раздел
    last_inferred = 0
    last_added: Optional[str] = None   # нормализованное имя прошлого раздела
    last_page = -1
    for element in elements:
        if element.type != "Title" or not (element.text or "").strip():
            continue
        stripped = element.text.strip()
        if is_junk_title(stripped):
            continue
        # Дедуп: одинаковые заголовки подряд на одной странице (двойная считка)
        normalized = normalize_section_name(stripped)
        if normalized != "" and normalized == last_added and element.page <= last_page + 1:
            continue
        norm_for_toc = normalize_section_name(stripped)
        toc_level = toc_map.get(norm_for_toc)
        if toc_level is None:
            # Точного совпадения нет — пробуем подстрочный матч (TOC-склейки
            # межколоночных строк вида «Кредитные рейтинги Обзор рынка»)
            toc_level = _best_toc_level(stripped, toc_norm)
        if toc_level is not None:
            level = toc_level
        else:
            level = _infer_stack_level(stripped, last_inferred)
        last_inferred = level
        # Уровни глубже нового закрываются: «сёстры» → «дети» одного родителя
        stack = stack[:level]
        breadcrumbs = [s.title for s in stack] + [element.text.strip()]
        node = SectionNode(
            id=_section_path(breadcrumbs),
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
