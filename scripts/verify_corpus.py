# -*- coding: utf-8 -*-
"""Полная валидация корпуса годовых отчётов (P0, ``spec_target.md`` §3).

Скрипт проверяет реестр источников и все производные артефакты:

* реестр: одна запись на ``doc_id``, уникальные пути, год из ``doc_id``,
  существование PDF, совпадение SHA-256, отсутствие 2021 года, полнота
  целевых годов, маркировка ``sber_2015`` (БПС-Сбербанк);
* артефакты: наличие ``*_elements.json``/``*_stats.json``/
  ``*_sections.json``/``*_chunks.json`` по каждому документу — отсутствующие
  перечисляются явно и не скрываются;
* чанки: дубликаты ``chunk_id``, страницы вне PDF, пустой текст, HTML таблиц,
  полнота и совпадение ``source_sha256`` с реестром, доля чанков без структуры;
* согласованность структуры и чанков: ``section_path`` чанка обязан
  существовать в дереве разделов, ``id``/``breadcrumbs`` разделов согласованы.

Отчёт пишется в JSON и Markdown и печатается в консоль. Код возврата: 0 —
ошибок нет, 1 — есть ошибки, 2 — ошибка запуска.

Запуск::

    python scripts/verify_corpus.py
    python scripts/verify_corpus.py --json data/corpus_report.json \\
        --markdown data/corpus_report.md
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.source_registry import (  # noqa: E402
    SourceRecord,
    load_registry,
    registry_fingerprint,
    resolve_source_path,
    validate_registry,
)
from src.ingestion.corpus_validator import validate_corpus  # noqa: E402

#: Сколько однотипных находок сохранять до сворачивания в «ещё N».
MAX_FINDINGS_PER_CODE = 25

#: Обязательные поля структуры чанка.
STRUCTURE_FIELDS = ("section_path", "breadcrumbs", "section_level",
                    "parent_section")


@dataclass(frozen=True)
class Finding:
    """Одна ошибка или предупреждение корпусной проверки."""

    code: str
    message: str
    doc_id: str | None = None
    chunk_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать находку без пустых полей."""
        payload = {"code": self.code, "message": self.message}
        if self.doc_id is not None:
            payload["doc_id"] = self.doc_id
        if self.chunk_id is not None:
            payload["chunk_id"] = self.chunk_id
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass
class DocReport:
    """Сводка по одному документу корпуса."""

    doc_id: str
    year: int | None = None
    issuer: str | None = None
    country: str | None = None
    source_status: str | None = None
    source_sha256: str | None = None
    n_pages: int | None = None
    artifacts: dict[str, str | None] = field(default_factory=dict)
    n_chunks: int = 0
    n_table_chunks: int = 0
    n_tables_with_html: int = 0
    n_chunks_with_structure: int = 0
    n_empty_chunks: int = 0
    n_duplicate_chunk_ids: int = 0
    n_sections: int | None = None
    n_section_paths_missing_in_structure: int = 0
    pages_covered: int = 0
    page_range: list[int] | None = None
    missing_page_ranges: list[str] = field(default_factory=list)

    @property
    def structure_coverage(self) -> float:
        """Доля чанков с полной структурой."""
        if not self.n_chunks:
            return 0.0
        return self.n_chunks_with_structure / self.n_chunks

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать сводку документа."""
        return {
            "doc_id": self.doc_id,
            "year": self.year,
            "issuer": self.issuer,
            "country": self.country,
            "source_status": self.source_status,
            "source_sha256": self.source_sha256,
            "n_pages": self.n_pages,
            "artifacts": self.artifacts,
            "n_chunks": self.n_chunks,
            "n_table_chunks": self.n_table_chunks,
            "n_tables_with_html": self.n_tables_with_html,
            "n_chunks_with_structure": self.n_chunks_with_structure,
            "n_empty_chunks": self.n_empty_chunks,
            "n_duplicate_chunk_ids": self.n_duplicate_chunk_ids,
            "structure_coverage": round(self.structure_coverage, 4),
            "n_sections": self.n_sections,
            "n_section_paths_missing_in_structure": (
                self.n_section_paths_missing_in_structure),
            "pages_covered": self.pages_covered,
            "page_range": self.page_range,
            "missing_page_ranges": self.missing_page_ranges,
        }


@dataclass
class CorpusReport:
    """Полный отчёт ``verify_corpus``."""

    generated_at: str
    project_root: str
    registry_path: str
    registry_doc_ids: list[str]
    registry_fingerprint: str | None
    registry_valid: bool
    documents: list[DocReport]
    errors: list[Finding]
    warnings: list[Finding]
    chunk_count: int = 0
    table_chunk_count: int = 0
    empty_chunk_count: int = 0
    structure_coverage: float = 0.0
    min_structure_coverage: float | None = None
    missing_artifacts: list[Finding] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Нет ли ошибок в корпусе."""
        return not self.errors

    @property
    def documents_expected(self) -> int:
        """Сколько документов ожидается по реестру."""
        return len(self.documents)

    @property
    def documents_with_chunks(self) -> int:
        """Сколько документов имеют непустой артефакт чанков."""
        return sum(1 for doc in self.documents if doc.n_chunks > 0)

    @property
    def documents_missing_chunks(self) -> list[str]:
        """doc_id без собранных чанков."""
        return [doc.doc_id for doc in self.documents if doc.n_chunks == 0]

    def summary(self) -> dict[str, Any]:
        """Компактная сводка для JSON и консоли."""
        return {
            "is_valid": self.is_valid,
            "documents_expected": self.documents_expected,
            "documents_with_chunks": self.documents_with_chunks,
            "documents_missing_chunks": self.documents_missing_chunks,
            "chunk_count": self.chunk_count,
            "table_chunk_count": self.table_chunk_count,
            "empty_chunk_count": self.empty_chunk_count,
            "structure_coverage": round(self.structure_coverage, 4),
            "min_structure_coverage": self.min_structure_coverage,
            "missing_artifact_count": len(self.missing_artifacts),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать отчёт целиком."""
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "registry": {
                "path": self.registry_path,
                "doc_ids": self.registry_doc_ids,
                "fingerprint": self.registry_fingerprint,
                "is_valid": self.registry_valid,
            },
            "summary": self.summary(),
            "documents": [doc.to_dict() for doc in self.documents],
            "missing_artifacts": [item.to_dict()
                                  for item in self.missing_artifacts],
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


# ---------------------------------------------------------------------------
# Загрузка артефактов
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    """Прочитать JSON-артефакт (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))


def _count_pdf_pages(path: Path) -> int | None:
    """Число страниц PDF через pypdf (None при ошибке)."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - окружение проекта содержит pypdf
        return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 - битый PDF не должен ронять отчёт
        return None


def _page_ranges(pages: Iterable[int]) -> list[str]:
    """Свернуть набор страниц в компактные диапазоны ``1-3, 7``."""
    ordered = sorted(set(int(page) for page in pages))
    if not ordered:
        return []
    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def _cap_findings(findings: list[Finding],
                  limit: int = MAX_FINDINGS_PER_CODE) -> list[Finding]:
    """Свернуть однотипные находки, сохранив первые ``limit`` и счётчик."""
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.code, []).append(finding)
    capped: list[Finding] = []
    for code, items in grouped.items():
        capped.extend(items[:limit])
        if len(items) > limit:
            capped.append(Finding(
                code=f"{code}_truncated",
                message=(f"Ещё {len(items) - limit} находок кода {code!r} "
                         "не перечислены поштучно."),
            ))
    return capped


# ---------------------------------------------------------------------------
# Основная проверка
# ---------------------------------------------------------------------------

def _load_structures(
    doc_ids: Sequence[str],
    structure_dir: Path,
    errors: list[Finding],
) -> dict[str, dict[str, Any]]:
    """Прочитать ``*_sections.json``; вернуть только валидные документы."""
    structures: dict[str, dict[str, Any]] = {}
    for doc_id in doc_ids:
        path = structure_dir / f"{doc_id}_sections.json"
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(Finding(
                "invalid_structure_json", f"Не читается {path.name}: {exc}",
                doc_id, path=str(path)))
            continue
        if not isinstance(payload, Mapping) or not isinstance(
                payload.get("sections"), list):
            errors.append(Finding(
                "invalid_structure_format",
                f"{path.name}: ожидался объект с полем 'sections'.",
                doc_id, path=str(path)))
            continue
        if payload.get("doc_id") not in (None, doc_id):
            errors.append(Finding(
                "structure_doc_id_mismatch",
                f"{path.name}: doc_id={payload.get('doc_id')!r}, "
                f"ожидался {doc_id!r}.", doc_id, path=str(path)))
        structures[doc_id] = dict(payload)
    return structures


def _structure_paths(structure: Mapping[str, Any],
                     doc_id: str,
                     errors: list[Finding]) -> set[str]:
    """Множество путей разделов; попутно проверяет id/breadcrumbs."""
    paths: set[str] = set()
    for section in structure.get("sections") or []:
        if not isinstance(section, Mapping):
            errors.append(Finding(
                "invalid_section", "Раздел структуры не является объектом.",
                doc_id))
            continue
        breadcrumbs = section.get("breadcrumbs")
        section_id = section.get("id")
        if isinstance(breadcrumbs, list) and breadcrumbs:
            joined = "/".join(str(item) for item in breadcrumbs)
            paths.add(joined)
            if isinstance(section_id, str) and section_id != joined:
                errors.append(Finding(
                    "structure_section_id_mismatch",
                    f"id={section_id!r} не совпадает с breadcrumbs "
                    f"{joined!r}.", doc_id))
        elif isinstance(section_id, str) and section_id:
            paths.add(section_id)
    return paths


def verify_corpus(
    *,
    project_root: Path | str = PROJECT,
    registry_path: Path | str | None = None,
    chunks_dir: Path | str | None = None,
    elements_dir: Path | str | None = None,
    structure_dir: Path | str | None = None,
    verify_sha256: bool = True,
    count_pdf_pages: bool = True,
    min_structure_coverage: float | None = None,
    require_target_years: bool = True,
) -> CorpusReport:
    """Проверить реестр и все артефакты корпуса.

    Args:
        project_root: Корень проекта; от него разрешаются ``file_path``
            реестра и каталоги по умолчанию.
        registry_path: Путь к ``data/source_registry.json``.
        chunks_dir: Каталог ``*_chunks.json`` (по умолчанию ``data/chunks``).
        elements_dir: Каталог ``*_elements.json``/``*_stats.json``.
        structure_dir: Каталог ``*_sections.json``.
        verify_sha256: Пересчитывать ли SHA-256 исходных PDF.
        count_pdf_pages: Читать ли число страниц PDF для проверки границ.
        min_structure_coverage: Порог доли чанков со структурой (предупреждение).
        require_target_years: Требовать ли источники за все целевые годы
            (частичный корпус можно проверить с ``False``).

    Returns:
        :class:`CorpusReport` со всеми найденными проблемами.

    Raises:
        ValueError: если передан некорректный порог структуры.
    """
    if min_structure_coverage is not None and not (
            0.0 <= float(min_structure_coverage) <= 1.0):
        raise ValueError("min_structure_coverage должен быть в диапазоне 0..1.")

    root = Path(project_root).resolve()
    if registry_path is not None:
        registry_file = Path(registry_path)
    else:
        registry_file = root / "data" / "source_registry.json"

    chunks_path = Path(chunks_dir) if chunks_dir else root / "data" / "chunks"
    elements_path = Path(elements_dir) if elements_dir else (
        root / "data" / "elements")
    structures_path = Path(structure_dir) if structure_dir else (
        root / "data" / "structure")

    errors: list[Finding] = []
    warnings: list[Finding] = []
    missing_artifacts: list[Finding] = []

    # --- реестр ---------------------------------------------------------
    records: list[SourceRecord] = []
    registry_loaded = True
    try:
        records = load_registry(registry_file)
    except (OSError, ValueError) as exc:
        registry_loaded = False
        errors.append(Finding(
            "registry_unreadable", f"Реестр источников не читается: {exc}",
            path=str(registry_file)))

    registry_valid = False
    if registry_loaded:
        validation = validate_registry(
            records, project_root=root, verify_sha256=verify_sha256,
            require_target_years=require_target_years)
        registry_valid = validation.is_valid
        for issue in validation.errors:
            errors.append(Finding(issue.code, issue.message, issue.doc_id))
        for issue in validation.warnings:
            warnings.append(Finding(issue.code, issue.message, issue.doc_id))

    doc_ids = [record.doc_id for record in records]
    structures = _load_structures(doc_ids, structures_path, errors)

    # --- чанки ----------------------------------------------------------
    all_chunks: list[Mapping[str, Any]] = []
    chunks_by_doc: dict[str, list[Mapping[str, Any]]] = {}
    page_counts: dict[str, int] = {}
    docs: list[DocReport] = []

    for record in records:
        doc = DocReport(
            doc_id=record.doc_id,
            year=record.year,
            issuer=record.issuer,
            country=record.country,
            source_status=record.source_status,
            source_sha256=record.source_sha256,
        )

        pdf_path = resolve_source_path(record, root)
        if count_pdf_pages and pdf_path.is_file():
            n_pages = _count_pdf_pages(pdf_path)
            doc.n_pages = n_pages
            if n_pages is not None:
                page_counts[doc.doc_id] = n_pages
            else:
                warnings.append(Finding(
                    "pdf_pages_unreadable",
                    f"Не удалось прочитать число страниц {pdf_path.name}.",
                    doc.doc_id, path=str(pdf_path)))

        artifacts = {
            "elements": elements_path / f"{doc.doc_id}_elements.json",
            "stats": elements_path / f"{doc.doc_id}_stats.json",
            "sections": structures_path / f"{doc.doc_id}_sections.json",
            "chunks": chunks_path / f"{doc.doc_id}_chunks.json",
        }
        doc.artifacts = {
            kind: (str(path) if path.is_file() else None)
            for kind, path in artifacts.items()
        }
        for kind, path in artifacts.items():
            if path.is_file():
                continue
            finding = Finding(
                "missing_artifact",
                f"Нет артефакта {kind}: {path.name}",
                doc.doc_id, path=str(path))
            missing_artifacts.append(finding)
            if kind == "stats":
                warnings.append(finding)
            else:
                errors.append(finding)

        # stats: сверяем lineage (doc_id / sha / число страниц)
        stats_path = artifacts["stats"]
        if stats_path.is_file():
            try:
                stats = _read_json(stats_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(Finding(
                    "invalid_stats_json", f"Не читается {stats_path.name}: {exc}",
                    doc.doc_id, path=str(stats_path)))
                stats = None
            if isinstance(stats, Mapping):
                if stats.get("doc_id") not in (None, doc.doc_id):
                    errors.append(Finding(
                        "stats_doc_id_mismatch",
                        f"{stats_path.name}: doc_id={stats.get('doc_id')!r}.",
                        doc.doc_id, path=str(stats_path)))
                stats_sha = stats.get("source_sha256")
                if stats_sha and str(stats_sha).lower() != (
                        record.source_sha256.lower()):
                    errors.append(Finding(
                        "stats_source_sha256_mismatch",
                        f"{stats_path.name}: SHA-256 не совпадает с реестром.",
                        doc.doc_id, path=str(stats_path)))
                stats_pages = stats.get("n_pages")
                if (doc.n_pages is not None and isinstance(stats_pages, int)
                        and stats_pages != doc.n_pages):
                    errors.append(Finding(
                        "stats_page_count_mismatch",
                        f"{stats_path.name}: n_pages={stats_pages}, "
                        f"в PDF {doc.n_pages}.",
                        doc.doc_id, path=str(stats_path)))

        # chunks
        chunks_file = artifacts["chunks"]
        doc_chunks: list[Mapping[str, Any]] = []
        if chunks_file.is_file():
            try:
                payload = _read_json(chunks_file)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(Finding(
                    "invalid_chunks_json", f"Не читается {chunks_file.name}: {exc}",
                    doc.doc_id, path=str(chunks_file)))
                payload = None
            if payload is not None:
                if not isinstance(payload, list):
                    errors.append(Finding(
                        "invalid_chunks_format",
                        f"{chunks_file.name}: ожидался JSON-массив чанков.",
                        doc.doc_id, path=str(chunks_file)))
                else:
                    doc_chunks = [item for item in payload
                                  if isinstance(item, Mapping)]
                    if len(doc_chunks) != len(payload):
                        errors.append(Finding(
                            "invalid_chunk_record",
                            f"{chunks_file.name}: {len(payload) - len(doc_chunks)} "
                            "записей не являются объектами.", doc.doc_id))
        doc.n_chunks = len(doc_chunks)
        chunks_by_doc[doc.doc_id] = doc_chunks
        all_chunks.extend(doc_chunks)
        if doc.doc_id not in page_counts and doc_chunks:
            # Число страниц PDF недоступно (файл отсутствует/битый): не выдаём
            # ложные page_out_of_range, границы проверить нечем.
            observed = [int(chunk["page_pdf"]) for chunk in doc_chunks
                        if isinstance(chunk.get("page_pdf"), int)
                        and not isinstance(chunk.get("page_pdf"), bool)]
            page_counts[doc.doc_id] = max(observed) if observed else 1

        # метаданные чанков против реестра
        pages = set()
        mismatched_sha: list[str] = []
        empty_text = 0
        for index, chunk in enumerate(doc_chunks):
            chunk_doc_id = chunk.get("doc_id")
            chunk_id = chunk.get("chunk_id")
            if chunk_doc_id != doc.doc_id:
                errors.append(Finding(
                    "chunk_doc_id_mismatch",
                    f"чанк #{index}: doc_id={chunk_doc_id!r}, "
                    f"ожидался {doc.doc_id!r}.",
                    doc.doc_id,
                    chunk_id=chunk_id if isinstance(chunk_id, str) else None))
            if isinstance(chunk_id, str) and not chunk_id.startswith(
                    f"{doc.doc_id}:"):
                errors.append(Finding(
                    "chunk_id_prefix_mismatch",
                    f"chunk_id {chunk_id!r} не начинается с {doc.doc_id!r}.",
                    doc.doc_id, chunk_id=chunk_id))
            chunk_year = chunk.get("year")
            if chunk_year != record.year:
                errors.append(Finding(
                    "chunk_year_mismatch",
                    f"год чанка {chunk_year!r} не совпадает с реестром "
                    f"({record.year}).", doc.doc_id,
                    chunk_id=chunk_id if isinstance(chunk_id, str) else None))
            chunk_sha = chunk.get("source_sha256")
            if isinstance(chunk_sha, str) and chunk_sha.strip():
                if chunk_sha.lower() != record.source_sha256.lower():
                    mismatched_sha.append(str(chunk_id or index))
            page = chunk.get("page_pdf")
            if isinstance(page, int) and not isinstance(page, bool) and page > 0:
                pages.add(page)
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                empty_text += 1
            kind = chunk.get("kind")
            html = chunk.get("html")
            if kind == "table":
                doc.n_table_chunks += 1
                if isinstance(html, str) and "<table" in html.lower():
                    doc.n_tables_with_html += 1
            has_structure = all(
                field_name in chunk for field_name in STRUCTURE_FIELDS)
            if has_structure:
                doc.n_chunks_with_structure += 1

        if mismatched_sha:
            errors.append(Finding(
                "chunk_source_sha256_mismatch",
                f"{len(mismatched_sha)} чанков имеют SHA-256, отличный от "
                f"реестра (например, {mismatched_sha[0]}).", doc.doc_id))
        chunk_id_counts = Counter(
            chunk.get("chunk_id") for chunk in doc_chunks
            if isinstance(chunk.get("chunk_id"), str))
        doc.n_duplicate_chunk_ids = sum(
            count - 1 for count in chunk_id_counts.values() if count > 1)
        doc.n_empty_chunks = empty_text
        doc.pages_covered = len(pages)
        if pages:
            doc.page_range = [min(pages), max(pages)]
            if doc.n_pages is not None:
                missing_pages = sorted(set(range(1, doc.n_pages + 1)) - pages)
                doc.missing_page_ranges = _page_ranges(missing_pages)

        # --- согласованность структуры и чанков -------------------------
        structure = structures.get(doc.doc_id)
        if structure is not None:
            section_paths = _structure_paths(structure, doc.doc_id, errors)
            doc.n_sections = len(section_paths)
            missing_paths: dict[str, int] = {}
            for chunk in doc_chunks:
                section_path = chunk.get("section_path")
                if not isinstance(section_path, str) or not section_path.strip():
                    continue
                if section_path not in section_paths:
                    missing_paths[section_path] = (
                        missing_paths.get(section_path, 0) + 1)
            doc.n_section_paths_missing_in_structure = sum(
                missing_paths.values())
            for section_path, count in list(missing_paths.items())[:10]:
                errors.append(Finding(
                    "chunk_section_not_in_structure",
                    f"{count} чанков ссылаются на отсутствующий раздел "
                    f"{section_path!r}.", doc.doc_id))
            if doc.n_pages is not None:
                for section in structure.get("sections") or []:
                    if not isinstance(section, Mapping):
                        continue
                    page_start = section.get("page_start")
                    if isinstance(page_start, int) and not isinstance(
                            page_start, bool) and page_start > doc.n_pages:
                        errors.append(Finding(
                            "section_page_out_of_range",
                            f"раздел {section.get('id')!r} начинается на "
                            f"стр. {page_start} при {doc.n_pages} страницах.",
                            doc.doc_id))
        elif doc_chunks and doc.n_chunks_with_structure:
            errors.append(Finding(
                "missing_structure_artifact",
                "Чанки содержат structural-поля, но *_sections.json отсутствует.",
                doc.doc_id))

        if doc.n_pages is not None and doc.missing_page_ranges:
            warnings.append(Finding(
                "pages_without_chunks",
                f"{len(doc.missing_page_ranges)} диапазонов страниц без чанков: "
                + ", ".join(doc.missing_page_ranges[:10])
                + (" …" if len(doc.missing_page_ranges) > 10 else ""),
                doc.doc_id))

        docs.append(doc)

    # --- чанки в каталоге, которых нет в реестре ------------------------
    if chunks_path.is_dir():
        known_doc_ids = set(doc_ids)
        for extra in sorted(chunks_path.glob("*_chunks.json")):
            extra_doc_id = extra.stem.replace("_chunks", "")
            if extra_doc_id not in known_doc_ids:
                errors.append(Finding(
                    "unknown_document_chunks",
                    f"Артефакт {extra.name} относится к doc_id, которого нет "
                    "в реестре.", extra_doc_id, path=str(extra)))

    # --- общий контракт чанков через corpus_validator -------------------
    structure_coverage = 0.0
    if all_chunks:
        contract = validate_corpus(
            all_chunks,
            page_counts,
            min_structure_coverage=min_structure_coverage,
        )
        structure_coverage = contract.structure_coverage
        for finding in contract.errors:
            doc_id = None
            if finding.chunk_index is not None and (
                    0 <= finding.chunk_index < len(all_chunks)):
                doc_id = all_chunks[finding.chunk_index].get("doc_id")
            errors.append(Finding(
                finding.code, finding.message,
                doc_id if isinstance(doc_id, str) else None,
                finding.chunk_id))
        for finding in contract.warnings:
            warnings.append(Finding(finding.code, finding.message))

    table_chunks = sum(doc.n_table_chunks for doc in docs)
    empty_chunks = sum(doc.n_empty_chunks for doc in docs)

    return CorpusReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        project_root=str(root),
        registry_path=str(registry_file),
        registry_doc_ids=doc_ids,
        registry_fingerprint=(registry_fingerprint(records)
                              if records else None),
        registry_valid=registry_valid,
        documents=docs,
        errors=_cap_findings(errors),
        warnings=_cap_findings(warnings),
        chunk_count=len(all_chunks),
        table_chunk_count=table_chunks,
        empty_chunk_count=empty_chunks,
        structure_coverage=structure_coverage,
        min_structure_coverage=min_structure_coverage,
        missing_artifacts=missing_artifacts,
    )


# ---------------------------------------------------------------------------
# Отчёты
# ---------------------------------------------------------------------------

def format_console(report: CorpusReport) -> str:
    """Короткий читаемый отчёт для консоли."""
    summary = report.summary()
    lines = [
        "=== verify_corpus ===",
        f"Реестр: {report.registry_path}",
        f"doc_id: {len(report.registry_doc_ids)} "
        f"({', '.join(report.registry_doc_ids) or '—'})",
        f"Репозиторий: {report.project_root}",
        f"Чанков: {report.chunk_count} | таблиц: {report.table_chunk_count} | "
        f"пустых: {report.empty_chunk_count}",
        f"Структура: {report.structure_coverage:.1%} чанков",
        f"Документов с чанками: {summary['documents_with_chunks']}/"
        f"{summary['documents_expected']}",
    ]
    if summary["documents_missing_chunks"]:
        lines.append("Без чанков: "
                     + ", ".join(summary["documents_missing_chunks"]))
    if report.missing_artifacts:
        lines.append(f"Отсутствующих артефактов: {len(report.missing_artifacts)}")
        for item in report.missing_artifacts[:20]:
            lines.append(f"  - {item.doc_id}: {item.message}")
    for doc in report.documents:
        lines.append(
            f"  {doc.doc_id}: стр. {doc.n_pages if doc.n_pages is not None else '?'}, "
            f"чанков {doc.n_chunks}, структурных "
            f"{doc.n_chunks_with_structure}, разделов "
            f"{doc.n_sections if doc.n_sections is not None else '—'}")
    lines.append(f"Ошибок: {len(report.errors)} | "
                 f"предупреждений: {len(report.warnings)}")
    for item in report.errors[:30]:
        location = f" [{item.doc_id}]" if item.doc_id else ""
        lines.append(f"  ERROR {item.code}{location}: {item.message}")
    if len(report.errors) > 30:
        lines.append(f"  ... ещё {len(report.errors) - 30} ошибок в JSON/Markdown")
    lines.append("СТАТУС: " + ("OK" if report.is_valid else "ЕСТЬ ОШИБКИ"))
    return "\n".join(lines)


def _md_escape(value: Any) -> str:
    """Экранировать ``|`` для Markdown-таблицы."""
    return str(value if value is not None else "—").replace("|", "\\|")


def format_markdown(report: CorpusReport) -> str:
    """Читаемый Markdown-отчёт."""
    summary = report.summary()
    status = "OK" if report.is_valid else "ЕСТЬ ОШИБКИ"
    lines = [
        "# Отчёт о валидации корпуса",
        "",
        f"- Сгенерировано: {report.generated_at}",
        f"- Корень проекта: `{report.project_root}`",
        f"- Реестр: `{report.registry_path}` "
        f"(валиден: {'да' if report.registry_valid else 'нет'})",
        f"- Fingerprint корпуса: `{report.registry_fingerprint or '—'}`",
        f"- **Статус: {status}**",
        "",
        "## Сводка",
        "",
        f"- Документов ожидается: {summary['documents_expected']}",
        f"- Документов с чанками: {summary['documents_with_chunks']}",
        f"- Чанков: {report.chunk_count} (таблиц: {report.table_chunk_count}, "
        f"пустых: {report.empty_chunk_count})",
        f"- Покрытие структурой: {report.structure_coverage:.1%}"
        + (f" (порог {report.min_structure_coverage:.0%})"
           if report.min_structure_coverage is not None else ""),
        f"- Отсутствующих артефактов: {len(report.missing_artifacts)}",
        f"- Ошибок: {len(report.errors)}, предупреждений: "
        f"{len(report.warnings)}",
        "",
        "## Документы",
        "",
        "| doc_id | год | issuer | стр. | чанков | таблиц (html) | "
        "со структурой | разделов | страниц без чанков |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for doc in report.documents:
        lines.append(
            "| " + " | ".join([
                _md_escape(doc.doc_id),
                _md_escape(doc.year),
                _md_escape(doc.issuer),
                _md_escape(doc.n_pages),
                _md_escape(doc.n_chunks),
                f"{doc.n_table_chunks} ({doc.n_tables_with_html})",
                f"{doc.n_chunks_with_structure} "
                f"({doc.structure_coverage:.0%})",
                _md_escape(doc.n_sections),
                _md_escape(", ".join(doc.missing_page_ranges) or "нет"),
            ]) + " |")

    lines += ["", "## Отсутствующие артефакты", ""]
    if report.missing_artifacts:
        for item in report.missing_artifacts:
            lines.append(f"- `{item.doc_id}`: {item.message}")
    else:
        lines.append("Отсутствующих артефактов нет.")

    lines += ["", f"## Ошибки ({len(report.errors)})", ""]
    if report.errors:
        for item in report.errors:
            prefix = f"[{item.doc_id}] " if item.doc_id else ""
            suffix = f" (chunk `{item.chunk_id}`)" if item.chunk_id else ""
            lines.append(f"- `{item.code}` {prefix}{item.message}{suffix}")
    else:
        lines.append("Ошибок нет.")

    lines += ["", f"## Предупреждения ({len(report.warnings)})", ""]
    if report.warnings:
        for item in report.warnings:
            prefix = f"[{item.doc_id}] " if item.doc_id else ""
            lines.append(f"- `{item.code}` {prefix}{item.message}")
    else:
        lines.append("Предупреждений нет.")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: CorpusReport,
    *,
    json_path: Path | str | None = None,
    markdown_path: Path | str | None = None,
) -> tuple[Path | None, Path | None]:
    """Записать JSON и Markdown отчёты атомарно.

    Returns:
        Пути записанных файлов (``None``, если путь не задан).
    """
    written: list[Path | None] = []
    if json_path is not None:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=1) + "\n"
        _atomic_write(target, text)
        written.append(target)
    else:
        written.append(None)
    if markdown_path is not None:
        target = Path(markdown_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, format_markdown(report))
        written.append(target)
    else:
        written.append(None)
    return written[0], written[1]


def _atomic_write(path: Path, text: str) -> None:
    """Атомарная запись UTF-8 (tmp + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    """Собрать CLI-парсер отчёта."""
    parser = argparse.ArgumentParser(
        description="Полная валидация корпуса годовых отчётов Сбербанка.")
    parser.add_argument("--project-root", type=Path, default=PROJECT,
                        help="Корень проекта (по умолчанию — каталог скрипта)")
    parser.add_argument("--registry", type=Path, default=None,
                        help="Путь к source_registry.json")
    parser.add_argument("--chunks-dir", type=Path, default=None,
                        help="Каталог *_chunks.json (по умолчанию data/chunks)")
    parser.add_argument("--elements-dir", type=Path, default=None,
                        help="Каталог *_elements.json (по умолчанию data/elements)")
    parser.add_argument("--structure-dir", type=Path, default=None,
                        help="Каталог *_sections.json (по умолчанию data/structure)")
    parser.add_argument("--json", dest="json_path", type=Path, default=None,
                        help="Куда писать JSON-отчёт")
    parser.add_argument("--markdown", dest="markdown_path", type=Path,
                        default=None, help="Куда писать Markdown-отчёт")
    parser.add_argument("--min-structure-coverage", type=float, default=None,
                        help="Порог доли чанков со структурой (0..1)")
    parser.add_argument("--no-hash", action="store_true",
                        help="Не пересчитывать SHA-256 PDF (быстрый режим)")
    parser.add_argument("--no-page-count", action="store_true",
                        help="Не читать PDF через pypdf (нет проверки границ)")
    parser.add_argument("--quiet", action="store_true",
                        help="Не печатать отчёт в консоль")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Не требовать источники за все целевые годы")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
    args = build_arg_parser().parse_args(argv)
    root = args.project_root.resolve()
    json_path = args.json_path or root / "data" / "corpus_report.json"
    markdown_path = args.markdown_path or root / "data" / "corpus_report.md"
    try:
        report = verify_corpus(
            project_root=root,
            registry_path=args.registry,
            chunks_dir=args.chunks_dir,
            elements_dir=args.elements_dir,
            structure_dir=args.structure_dir,
            verify_sha256=not args.no_hash,
            count_pdf_pages=not args.no_page_count,
            min_structure_coverage=args.min_structure_coverage,
            require_target_years=not args.allow_partial,
        )
    except ValueError as exc:
        print(f"[verify_corpus] ошибка запуска: {exc}", file=sys.stderr)
        return 2

    write_reports(report, json_path=json_path, markdown_path=markdown_path)
    if not args.quiet:
        print(format_console(report))
        print(f"JSON: {json_path}")
        print(f"Markdown: {markdown_path}")
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
