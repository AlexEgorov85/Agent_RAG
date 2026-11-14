"""Помощники проверки корпуса чанков.

Валидатор намеренно не зависит от парсинга PDF и хранения корпуса: вызывающий
передаёт словари чанков и число страниц каждого документа явно. Он сообщает обо
всех найденных проблемах, а не останавливается на первом невалидном чанке.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import re
from typing import Any, Mapping, Sequence

from src.ingestion.pdf_to_chunks import REQUIRED_CHUNK_FIELDS


_STRUCTURE_FIELDS = (
    "section_path",
    "breadcrumbs",
    "section_level",
    "parent_section",
)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True)
class ValidationFinding:
    """Одна ошибка или предупреждение валидации.

    ``chunk_index`` считается с нуля и равен ``None`` для проблем всего корпуса.
    ``chunk_id`` заполняется, когда его можно прочитать из связанного чанка.
    """

    code: str
    message: str
    chunk_index: int | None = None
    chunk_id: str | None = None


@dataclass
class ValidationReport:
    """Структурированный результат, возвращаемый :func:`validate_corpus`."""

    chunk_count: int
    structure_coverage: float
    errors: list[ValidationFinding] = field(default_factory=list)
    warnings: list[ValidationFinding] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Проверяет, что в корпусе нет ошибок валидации."""
        return not self.errors

    @property
    def error_count(self) -> int:
        """Число ошибок в отчёте."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Число предупреждений в отчёте."""
        return len(self.warnings)


def validate_corpus(
    chunks: Sequence[Mapping[str, Any]],
    page_counts: Mapping[str, int],
    *,
    min_structure_coverage: float | None = None,
) -> ValidationReport:
    """Проверяет записи чанков на соответствие контракту ingestion и корпуса.

    Проверяются обязательные поля, непустой поисковый текст, уникальность ID,
    границы PDF-страниц с нумерацией от 1, полные SHA-256 источников,
    согласованность таблицы и HTML, а также необязательные структурные
    метаданные. Чанк со структурой имеет все четыре поля, согласованные
    breadcrumbs и путь.

    Args:
        chunks: Чанки корпуса, обычно загруженные из одного или нескольких
            артефактов.
        page_counts: Явное отображение ``doc_id`` в положительное число страниц
            исходного PDF. Числа не выводятся из содержимого чанков.
        min_structure_coverage: Необязательный порог от 0.0 до 1.0. Значение
            ниже порога добавляет предупреждение, но не делает корпус без ошибок
            невалидным.

    Returns:
        :class:`ValidationReport` со всеми найденными проблемами и счётчиками
        корпуса. ``structure_coverage`` — доля в [0.0, 1.0]; для пустого
        корпуса она равна 0.0.

    Raises:
        ValueError: если необязательный порог не является конечной долей от нуля
            до единицы.
    """
    if min_structure_coverage is not None and (
        not isinstance(min_structure_coverage, (int, float))
        or isinstance(min_structure_coverage, bool)
        or not isfinite(min_structure_coverage)
        or not 0.0 <= min_structure_coverage <= 1.0
    ):
        raise ValueError("min_structure_coverage must be between 0 and 1.")

    errors: list[ValidationFinding] = []
    warnings: list[ValidationFinding] = []
    seen_ids: dict[str, int] = {}
    covered_chunks = 0

    def error(
        code: str,
        message: str,
        index: int | None = None,
        chunk_id: str | None = None,
    ) -> None:
        errors.append(ValidationFinding(code, message, index, chunk_id))

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            error("invalid_chunk", "Chunk must be a mapping.", index)
            continue

        raw_id = chunk.get("chunk_id")
        chunk_id = raw_id if isinstance(raw_id, str) and raw_id else None

        for field_name in sorted(REQUIRED_CHUNK_FIELDS):
            if field_name not in chunk:
                error(
                    "missing_required_field",
                    f"Required field {field_name!r} is missing.",
                    index,
                    chunk_id,
                )

        if chunk_id is None:
            error("invalid_chunk_id", "chunk_id must be a non-empty string.",
                  index)
        elif chunk_id in seen_ids:
            error(
                "duplicate_chunk_id",
                f"chunk_id {chunk_id!r} duplicates chunk index "
                f"{seen_ids[chunk_id]}.",
                index,
                chunk_id,
            )
        else:
            seen_ids[chunk_id] = index

        for field_name in (
            "evidence_id", "doc_id", "title", "report_period", "kind", "text",
        ):
            value = chunk.get(field_name)
            if not isinstance(value, str) or not value.strip():
                error(
                    "invalid_required_field",
                    f"{field_name} must be a non-empty string.",
                    index,
                    chunk_id,
                )

        if not isinstance(chunk.get("year"), int) or isinstance(
            chunk.get("year"), bool
        ):
            error("invalid_year", "year must be an integer.", index, chunk_id)

        element_ids = chunk.get("element_ids")
        if (
            not isinstance(element_ids, list)
            or any(not isinstance(element_id, str) for element_id in element_ids)
        ):
            error(
                "invalid_element_ids",
                "element_ids must be a list of strings.",
                index,
                chunk_id,
            )

        page = chunk.get("page_pdf")
        page_is_valid = (
            isinstance(page, int) and not isinstance(page, bool) and page > 0
        )
        if not page_is_valid:
            error(
                "invalid_page_pdf",
                "page_pdf must be a positive 1-based integer.",
                index,
                chunk_id,
            )

        doc_id = chunk.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            page_count = page_counts.get(doc_id)
            if (
                not isinstance(page_count, int)
                or isinstance(page_count, bool)
                or page_count <= 0
            ):
                error(
                    "invalid_page_count",
                    f"A positive page count is required for {doc_id!r}.",
                    index,
                    chunk_id,
                )
            elif page_is_valid and page > page_count:
                error(
                    "page_out_of_range",
                    f"page_pdf {page} exceeds {doc_id!r}'s "
                    f"{page_count} pages.",
                    index,
                    chunk_id,
                )

        source_sha256 = chunk.get("source_sha256")
        if not isinstance(source_sha256, str) or not source_sha256.strip():
            error(
                "missing_source_sha256",
                "source_sha256 must be present and non-empty.",
                index,
                chunk_id,
            )
        elif _SHA256_PATTERN.fullmatch(source_sha256) is None:
            error(
                "invalid_source_sha256",
                "source_sha256 must contain exactly 64 hexadecimal characters.",
                index,
                chunk_id,
            )

        kind = chunk.get("kind")
        html = chunk.get("html")
        if kind == "table":
            if not isinstance(html, str) or not html.strip():
                error(
                    "table_html_missing",
                    "Table chunks must contain non-empty HTML.",
                    index,
                    chunk_id,
                )
            elif "<table" not in html.lower():
                error(
                    "table_html_invalid",
                    "Table chunk HTML must contain a <table> element.",
                    index,
                    chunk_id,
                )
            if not isinstance(chunk.get("text"), str) or not chunk.get(
                "text", ""
            ).strip():
                error(
                    "table_text_missing",
                    "Table chunks must contain searchable text.",
                    index,
                    chunk_id,
                )
        elif html is not None:
            error(
                "unexpected_html",
                "Only table chunks may contain HTML.",
                index,
                chunk_id,
            )

        present_structure = [
            field_name for field_name in _STRUCTURE_FIELDS
            if field_name in chunk
        ]
        if present_structure:
            structure_valid = True
            if len(present_structure) != len(_STRUCTURE_FIELDS):
                missing = sorted(set(_STRUCTURE_FIELDS) - set(present_structure))
                error(
                    "incomplete_structure",
                    "Structure metadata is partial; missing "
                    + ", ".join(missing)
                    + ".",
                    index,
                    chunk_id,
                )
                structure_valid = False

            section_path = chunk.get("section_path")
            breadcrumbs = chunk.get("breadcrumbs")
            section_level = chunk.get("section_level")
            parent_section = chunk.get("parent_section")

            if not isinstance(section_path, str) or not section_path.strip():
                error(
                    "invalid_section_path",
                    "section_path must be a non-empty string.",
                    index,
                    chunk_id,
                )
                structure_valid = False
            if (
                not isinstance(breadcrumbs, list)
                or not breadcrumbs
                or any(
                    not isinstance(crumb, str) or not crumb.strip()
                    for crumb in breadcrumbs
                )
            ):
                error(
                    "invalid_breadcrumbs",
                    "breadcrumbs must be a non-empty list of non-empty strings.",
                    index,
                    chunk_id,
                )
                structure_valid = False
            elif isinstance(section_path, str) and section_path != "/".join(
                breadcrumbs
            ):
                error(
                    "inconsistent_section_path",
                    "section_path must equal breadcrumbs joined by '/'.",
                    index,
                    chunk_id,
                )
                structure_valid = False

            if not isinstance(section_level, int) or isinstance(
                section_level, bool
            ) or section_level < 0:
                error(
                    "invalid_section_level",
                    "section_level must be a non-negative integer.",
                    index,
                    chunk_id,
                )
                structure_valid = False

            expected_parent = (
                "/".join(breadcrumbs[:-1])
                if isinstance(breadcrumbs, list) and len(breadcrumbs) > 1
                else None
            )
            if parent_section != expected_parent:
                error(
                    "inconsistent_parent_section",
                    "parent_section must match the breadcrumbs' parent path.",
                    index,
                    chunk_id,
                )
                structure_valid = False

            if structure_valid:
                covered_chunks += 1

    coverage = covered_chunks / len(chunks) if chunks else 0.0
    if (
        min_structure_coverage is not None
        and coverage < min_structure_coverage
    ):
        warnings.append(
            ValidationFinding(
                "structure_coverage_below_threshold",
                f"Structure coverage {coverage:.1%} is below the requested "
                f"{min_structure_coverage:.1%} threshold.",
            )
        )

    return ValidationReport(
        chunk_count=len(chunks),
        structure_coverage=coverage,
        errors=errors,
        warnings=warnings,
    )
