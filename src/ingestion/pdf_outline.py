"""Чтение встроенной структуры PDF: закладки (outline) и внутренние ссылки.

Закладки — самый сильный источник уровней: в отчётах 2018/2019/2023/2024 они
содержат настоящую иерархию с точными PDF-страницами. Ссылки (``/Link``)
сохраняются отдельным артефактом: они дают пары «страница-источник →
страница-цель» и позволяют позже выравнивать печатные номера TOC.

Модуль только читает метаданные PDF (без запуска OCR/Unstructured) и умеет
атомарно сохранять/загружать артефакты ``{doc_id}_outline.json`` и
``{doc_id}_links.json`` рядом с ``{doc_id}_sections.json``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.ingestion.structure_builder import (
    is_junk_title,
    normalize_section_name,
)

#: Служебные закладки Word («_GoBack») и внутренние метки («Par763»).
_JUNK_OUTLINE_RE = re.compile(r"^(?:_.*|Par\d+)$")


@dataclass
class OutlineEntry:
    """Закладка PDF (строка outline).

    Attributes:
        title: Название закладки (как в PDF).
        level: Глубина вложенности (0 = верхний уровень).
        page: PDF-страница цели (1-based).
    """

    title: str
    level: int
    page: int

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь для артефакта."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutlineEntry":
        """Восстановить закладку из словаря."""
        return cls(title=str(data.get("title", "")),
                   level=int(data.get("level", 0)),
                   page=int(data.get("page", 1)))


def _is_junk_outline_title(title: str) -> bool:
    """Отфильтровать служебные и пустые закладки (``_GoBack``, ``Par763``)."""
    stripped = (title or "").strip()
    if not stripped or len(stripped) < 2:
        return True
    if _JUNK_OUTLINE_RE.match(stripped):
        return True
    return is_junk_title(stripped)


def read_outline(pdf_path: Path) -> List[OutlineEntry]:
    """Прочитать иерархические закладки PDF с разрешёнными страницами.

    Args:
        pdf_path: Путь к PDF.

    Returns:
        Закладки в порядке документа (пустой список, если outline нет или
        остались только служебные записи).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if not reader.outline:
        return []

    entries: List[OutlineEntry] = []

    def walk(items: Sequence[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            title = (getattr(item, "title", "") or "").strip()
            if _is_junk_outline_title(title):
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:  # noqa: BLE001 — битая цель не должна ронять чтение
                continue
            entries.append(OutlineEntry(title=title, level=min(level, 3), page=page))

    walk(reader.outline, 0)
    return entries


def read_links(pdf_path: Path) -> List[Dict[str, int]]:
    """Прочитать внутренние link-аннотации PDF (пары страниц).

    Внешние (``/URI``) и неразрешимые цели пропускаются.

    Args:
        pdf_path: Путь к PDF.

    Returns:
        Список словарей ``{"source_page": int, "target_page": int}`` (1-based).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    named = reader.named_destinations
    links: List[Dict[str, int]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        annots = page.get("/Annots")
        if not annots:
            continue
        for annot in annots.get_object():
            obj = annot.get_object()
            if obj.get("/Subtype") != "/Link":
                continue
            dest = obj.get("/Dest")
            if dest is None and obj.get("/A") is not None:
                action = obj["/A"].get_object()
                if action.get("/S") == "/URI":
                    continue
                dest = action.get("/D")
            if dest is None:
                continue
            if isinstance(dest, str):
                dest = named.get(dest)
            if dest is None:
                continue
            try:
                target = reader.get_destination_page_number(dest) + 1
            except Exception:  # noqa: BLE001 — неразрешимая цель пропускается
                continue
            links.append({"source_page": page_index, "target_page": target})
    return links


def outline_level_map(entries: Sequence[OutlineEntry]) -> Dict[str, int]:
    """Карта «нормализованное имя закладки → уровень» (первое вхождение важнее).

    Args:
        entries: Закладки из ``read_outline``.

    Returns:
        Словарь для точного сопоставления с Title-элементами.
    """
    mapping: Dict[str, int] = {}
    for entry in entries:
        key = normalize_section_name(entry.title)
        if key and key not in mapping:
            mapping[key] = entry.level
    return mapping


def _atomic_write_json(target: Path, payload: Any) -> None:
    """Атомарно записать JSON (как в парсере: tempfile + ``os.replace``)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json",
        dir=target.parent, delete=False)
    try:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.flush()
    finally:
        handle.close()
    os.replace(handle.name, target)


def save_outline(entries: Sequence[OutlineEntry], doc_id: str,
                 structure_dir: Path) -> Path:
    """Сохранить ``{doc_id}_outline.json`` в каталог структуры."""
    target = Path(structure_dir) / f"{doc_id}_outline.json"
    _atomic_write_json(target, [entry.to_dict() for entry in entries])
    return target


def save_links(links: Sequence[Dict[str, int]], doc_id: str,
               structure_dir: Path) -> Path:
    """Сохранить ``{doc_id}_links.json`` в каталог структуры."""
    target = Path(structure_dir) / f"{doc_id}_links.json"
    _atomic_write_json(target, list(links))
    return target


def load_outline(outline_json: Path) -> List[OutlineEntry]:
    """Прочитать артефакт ``{doc_id}_outline.json``."""
    path = Path(outline_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт закладок не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        return [OutlineEntry.from_dict(item) for item in json.load(fh)]


def load_links(links_json: Path) -> List[Dict[str, int]]:
    """Прочитать артефакт ``{doc_id}_links.json``."""
    path = Path(links_json)
    if not path.is_file():
        raise FileNotFoundError(f"Артефакт ссылок не найден: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
