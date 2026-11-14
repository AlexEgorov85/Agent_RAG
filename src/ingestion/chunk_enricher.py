"""Обогащение чанков структурой документа (опциональные поля контракта К2).

Слой ``С`` из spec_project.md §5.2: базовые курсовые поля чанка не меняются,
добавляются только ``section_path``/``breadcrumbs``/``section_level``/
``parent_section``. Если для страницы чанка раздел не найден (обложка, TOC до
первого заголовка), поля не добавляются — фильтры трактуют их как «нет данных».
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.ingestion.structure_builder import DocumentStructure, SectionNode


def _section_path(breadcrumbs: Sequence[str]) -> str:
    """Склеить breadcrumbs в путь раздела через ``/``."""
    return "/".join(breadcrumbs)


def find_section_for_page(structure: DocumentStructure,
                          page: int) -> Optional[SectionNode]:
    """Найти раздел, действующий на странице ``page``.

    Раздел действует с собственной ``page_start`` до следующего заголовка,
    поэтому достаточно взять последний раздел с ``page_start <= page``.

    Args:
        structure: Дерево разделов документа.
        page: Номер PDF-страницы (1-based).

    Returns:
        Раздел или None, если страница раньше первого заголовка.
    """
    if not structure.sections:
        return None
    ordered = sorted(structure.sections, key=lambda s: (s.page_start, s.level))
    index = bisect_right([s.page_start for s in ordered], page) - 1
    if index < 0:
        return None
    return ordered[index]


def enrich_chunk(chunk: Mapping[str, Any],
                 structure: DocumentStructure) -> Dict[str, Any]:
    """Добавить структурные поля одному чанку по его ``page_pdf``.

    Args:
        chunk: Чанк контракта К2.
        structure: Дерево разделов документа.

    Returns:
        Новый словарь чанка с опциональными полями (если раздел найден).
    """
    enriched = dict(chunk)
    section = find_section_for_page(structure, int(chunk["page_pdf"]))
    if section is None:
        return enriched
    enriched["section_path"] = _section_path(section.breadcrumbs)
    enriched["breadcrumbs"] = list(section.breadcrumbs)
    enriched["section_level"] = section.level
    enriched["parent_section"] = (
        _section_path(section.breadcrumbs[:-1])
        if len(section.breadcrumbs) > 1 else None)
    return enriched


def enrich_chunks(chunks: Sequence[Mapping[str, Any]],
                  structure: DocumentStructure) -> List[Dict[str, Any]]:
    """Обогатить все чанки документа структурой (шаг 2.5 плана).

    Args:
        chunks: Чанки контракта К2 в порядке документа.
        structure: Дерево разделов того же документа.

    Returns:
        Список обогащённых чанков (исходные не мутируются).
    """
    return [enrich_chunk(chunk, structure) for chunk in chunks]
