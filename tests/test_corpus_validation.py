# -*- coding: utf-8 -*-
"""Offline-тесты ``scripts/verify_corpus.py`` на tmp_path-корпусе."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import verify_corpus as corpus_check  # noqa: E402

from src.core.source_registry import (  # noqa: E402
    BPS_COUNTRY,
    BPS_ISSUER,
    SOURCE_STATUS_COMPARABLE,
    SOURCE_STATUS_INCOMPARABLE,
    SourceRecord,
    compute_sha256,
    save_registry,
)

DEFAULT_DOCS = ("sber_2022", "sber_2023")


# ---------------------------------------------------------------------------
# Построение фиктивного корпуса
# ---------------------------------------------------------------------------

def make_pdf(root: Path, doc_id: str, pages: int = 2) -> Path:
    """Создать настоящий (пустой) PDF на ``pages`` страниц."""
    from pypdf import PdfWriter

    path = root / "data" / "raw" / f"Копия Сбер {doc_id.rsplit('_', 1)[-1]}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    with open(path, "wb") as handle:
        writer.write(handle)
    return path


def make_record(root: Path, doc_id: str, pages: int = 2) -> SourceRecord:
    """Собрать запись реестра поверх реального PDF."""
    pdf = make_pdf(root, doc_id, pages)
    year = int(doc_id.rsplit("_", 1)[-1])
    if doc_id == "sber_2015":
        return SourceRecord(
            doc_id=doc_id,
            file_path=pdf.relative_to(root).as_posix(),
            source_sha256=compute_sha256(pdf),
            year=year,
            report_period=str(year),
            issuer=BPS_ISSUER,
            country=BPS_COUNTRY,
            source_status=SOURCE_STATUS_INCOMPARABLE,
            inclusion_note="Документ несопоставим с ПАО Сбербанк.",
        )
    return SourceRecord(
        doc_id=doc_id,
        file_path=pdf.relative_to(root).as_posix(),
        source_sha256=compute_sha256(pdf),
        year=year,
        report_period=str(year),
        issuer="ПАО Сбербанк",
        country="Россия",
        source_status=SOURCE_STATUS_COMPARABLE,
        inclusion_note="Включён в сопоставимый корпус.",
    )


def make_chunk(
    record: SourceRecord,
    *,
    page: int = 1,
    kind: str = "text",
    text: str = "Содержательный текст отчёта.",
    html: str | None = None,
    section: str | None = "Раздел",
    suffix: str = "0",
    source_sha256: str | None = None,
) -> dict:
    """Собрать чанк, проходящий контракт ingestion-слоя."""
    chunk = {
        "chunk_id": f"{record.doc_id}:p{page:03d}:c0:{suffix}",
        "evidence_id": f"{record.doc_id}:p{page:03d}",
        "doc_id": record.doc_id,
        "title": record.doc_id,
        "page_pdf": page,
        "page_label": str(page),
        "year": record.year,
        "report_period": record.report_period,
        "kind": kind,
        "element_ids": [f"{record.doc_id}:p{page:03d}"],
        "text": text,
        "html": html,
        "source_url": None,
        "source_sha256": (
            source_sha256 if source_sha256 is not None
            else record.source_sha256),
    }
    if section is not None:
        chunk.update(
            section_path=section,
            breadcrumbs=section.split("/"),
            section_level=section.count("/"),
            parent_section=(
                "/".join(section.split("/")[:-1])
                if "/" in section else None),
        )
    return chunk


def write_json(path: Path, payload) -> None:
    """Записать JSON-артефакт в UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def write_artifacts(
    root: Path,
    record: SourceRecord,
    *,
    chunks: list[dict] | None = None,
    sections: bool = True,
    stats: bool = True,
    elements: bool = True,
    pages: int = 2,
) -> list[dict]:
    """Записать полный набор артефактов для документа."""
    if chunks is None:
        chunks = [
            make_chunk(record, page=1, suffix="a"),
            make_chunk(
                record, page=2, kind="table", text="Выручка 100",
                html="<table><tr><td>Выручка</td></tr></table>",
                suffix="b"),
        ]
    write_json(root / "data" / "chunks" / f"{record.doc_id}_chunks.json",
               chunks)
    if sections:
        write_json(
            root / "data" / "structure" / f"{record.doc_id}_sections.json",
            {
                "doc_id": record.doc_id,
                "sections": [{
                    "id": "Раздел",
                    "title": "Раздел",
                    "level": 0,
                    "page_start": 1,
                    "page_end": pages,
                    "breadcrumbs": ["Раздел"],
                    "parent_id": None,
                    "children_ids": [],
                }],
            })
    if stats:
        write_json(root / "data" / "elements" / f"{record.doc_id}_stats.json",
                   {
                       "doc_id": record.doc_id,
                       "source_sha256": record.source_sha256,
                       "n_pages": pages,
                       "n_elements": 10,
                   })
    if elements:
        write_json(
            root / "data" / "elements" / f"{record.doc_id}_elements.json", [])
    return chunks


def build_project(
    root: Path,
    doc_ids=DEFAULT_DOCS,
    *,
    pages: int = 2,
) -> list[SourceRecord]:
    """Создать реестр и валидные артефакты для указанных документов."""
    records = [make_record(root, doc_id, pages) for doc_id in doc_ids]
    save_registry(records, root / "data" / "source_registry.json",
                  notes="тестовый корпус")
    for record in records:
        write_artifacts(root, record, pages=pages)
    return records


def chunk_file(root: Path, doc_id: str) -> Path:
    """Путь к артефакту чанков документа."""
    return root / "data" / "chunks" / f"{doc_id}_chunks.json"


def load_chunks(path: Path) -> list[dict]:
    """Прочитать чанки из файла."""
    return json.loads(path.read_text(encoding="utf-8"))


def error_codes(report) -> set[str]:
    """Коды ошибок отчёта."""
    return {item.code for item in report.errors}


def warning_codes(report) -> set[str]:
    """Коды предупреждений отчёта."""
    return {item.code for item in report.warnings}


# ---------------------------------------------------------------------------
# Положительный сценарий
# ---------------------------------------------------------------------------

def test_clean_corpus_is_valid(tmp_path: Path) -> None:
    build_project(tmp_path)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert report.is_valid
    assert report.errors == []
    assert report.warnings == []
    assert report.chunk_count == 4
    assert report.table_chunk_count == 2
    assert report.empty_chunk_count == 0
    assert report.documents_expected == 2
    assert report.documents_with_chunks == 2
    assert report.documents_missing_chunks == []
    assert report.structure_coverage == 1.0
    assert report.registry_fingerprint is not None
    assert len(report.registry_fingerprint) == 64
    for doc in report.documents:
        assert doc.n_sections == 1
        assert doc.pages_covered == 2
        assert doc.missing_page_ranges == []
        assert doc.n_tables_with_html == 1


def test_missing_artifacts_are_reported_explicitly(tmp_path: Path) -> None:
    build_project(tmp_path)
    chunk_file(tmp_path, "sber_2023").unlink()

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert not report.is_valid
    assert "sber_2023" in report.documents_missing_chunks
    missing = {(item.doc_id, item.message) for item in report.missing_artifacts}
    assert any(doc_id == "sber_2023" and "chunks" in message
               for doc_id, message in missing)
    assert "missing_artifact" in error_codes(report)
    markdown = corpus_check.format_markdown(report)
    assert "Отсутствующие артефакты" in markdown
    assert "sber_2023_chunks.json" in markdown


def test_missing_sections_and_elements_are_errors(tmp_path: Path) -> None:
    build_project(tmp_path)
    (tmp_path / "data" / "structure" / "sber_2023_sections.json").unlink()
    (tmp_path / "data" / "elements" / "sber_2023_elements.json").unlink()

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    codes = error_codes(report)

    assert "missing_structure_artifact" in codes
    assert "missing_artifact" in codes
    kinds = {item.message for item in report.missing_artifacts}
    assert any("sections" in message for message in kinds)
    assert any("elements" in message for message in kinds)


def test_missing_stats_is_a_warning(tmp_path: Path) -> None:
    build_project(tmp_path)
    (tmp_path / "data" / "elements" / "sber_2023_stats.json").unlink()

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert report.is_valid
    assert "missing_artifact" in warning_codes(report)


# ---------------------------------------------------------------------------
# Проверки чанков
# ---------------------------------------------------------------------------

def test_duplicate_chunk_id_is_an_error(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks.append(dict(chunks[0]))
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "duplicate_chunk_id" in error_codes(report)
    assert report.documents[0].n_duplicate_chunk_ids == 1


def test_page_outside_pdf_is_an_error(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[0]["page_pdf"] = 7
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "page_out_of_range" in error_codes(report)


def test_empty_chunk_text_is_counted_and_rejected(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[0]["text"] = "   "
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert report.empty_chunk_count == 1
    assert "invalid_required_field" in error_codes(report)


def test_table_without_html_is_rejected(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[1]["html"] = None
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "table_html_missing" in error_codes(report)
    assert report.documents[0].n_tables_with_html == 0


def test_non_table_html_is_rejected(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[0]["html"] = "<table></table>"
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "unexpected_html" in error_codes(report)


def test_source_sha256_mismatch_and_missing(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[0]["source_sha256"] = "b" * 64
    chunks[1]["source_sha256"] = ""
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    codes = error_codes(report)

    assert "chunk_source_sha256_mismatch" in codes
    assert "missing_source_sha256" in codes


def test_wrong_year_and_foreign_doc_id_are_rejected(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    chunks[0]["year"] = 2022
    chunks[1]["doc_id"] = "sber_2022"
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    codes = error_codes(report)

    assert "chunk_year_mismatch" in codes
    assert "chunk_doc_id_mismatch" in codes


def test_chunks_without_structure_are_measured(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    for chunk in chunks:
        for field_name in corpus_check.STRUCTURE_FIELDS:
            chunk.pop(field_name, None)
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False,
        min_structure_coverage=0.5)

    assert report.is_valid
    assert report.structure_coverage == 0.0
    assert report.documents[0].n_chunks_with_structure == 0
    assert "structure_coverage_below_threshold" in warning_codes(report)


def test_unknown_doc_chunks_are_reported(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    write_json(tmp_path / "data" / "chunks" / "sber_2099_chunks.json", [])

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "unknown_document_chunks" in error_codes(report)


def test_invalid_chunks_json_does_not_crash(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    (tmp_path / "data" / "chunks" / "sber_2023_chunks.json").write_text(
        "{}", encoding="utf-8")

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "invalid_chunks_format" in error_codes(report)
    assert report.chunk_count == 0


# ---------------------------------------------------------------------------
# Структура и страницы
# ---------------------------------------------------------------------------

def test_chunk_section_must_exist_in_structure(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    path = chunk_file(tmp_path, "sber_2023")
    chunks = load_chunks(path)
    for index, chunk in enumerate(chunks):
        chunk.update(
            section_path="Несуществующий раздел",
            breadcrumbs=["Несуществующий раздел"],
            section_level=0,
            parent_section=None,
        )
    write_json(path, chunks)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "chunk_section_not_in_structure" in error_codes(report)
    assert report.documents[0].n_section_paths_missing_in_structure == 2


def test_structure_section_id_and_pages_are_validated(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",), pages=2)
    sections_path = (tmp_path / "data" / "structure"
                     / "sber_2023_sections.json")
    payload = json.loads(sections_path.read_text(encoding="utf-8"))
    payload["sections"][0]["id"] = "Другой id"
    payload["sections"].append({
        "id": "Поздний раздел",
        "title": "Поздний раздел",
        "level": 0,
        "page_start": 9,
        "page_end": 9,
        "breadcrumbs": ["Поздний раздел"],
        "parent_id": None,
        "children_ids": [],
    })
    write_json(sections_path, payload)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    codes = error_codes(report)

    assert "structure_section_id_mismatch" in codes
    assert "section_page_out_of_range" in codes


def test_uncovered_pages_are_a_warning(tmp_path: Path) -> None:
    record = build_project(tmp_path, ("sber_2023",), pages=3)[0]
    write_artifacts(
        tmp_path, record,
        chunks=[make_chunk(record, page=1)],
        pages=3)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "pages_without_chunks" in warning_codes(report)
    assert report.documents[0].missing_page_ranges == ["2-3"]
    assert "2-3" in corpus_check.format_markdown(report)


def test_stats_lineage_mismatch_is_an_error(tmp_path: Path) -> None:
    build_project(tmp_path, ("sber_2023",))
    stats_path = (tmp_path / "data" / "elements"
                  / "sber_2023_stats.json")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["source_sha256"] = "c" * 64
    payload["n_pages"] = 5
    write_json(stats_path, payload)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    codes = error_codes(report)

    assert "stats_source_sha256_mismatch" in codes
    assert "stats_page_count_mismatch" in codes


# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------

def test_registry_with_2021_is_rejected(tmp_path: Path) -> None:
    records = build_project(tmp_path, ("sber_2022", "sber_2023"))
    extra = make_record(tmp_path, "sber_2021")
    save_registry(records + [extra], tmp_path / "data" / "source_registry.json")
    write_artifacts(tmp_path, extra)

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert not report.is_valid
    assert "missing_year_present" in error_codes(report)


def test_unreadable_registry_is_reported(tmp_path: Path) -> None:
    build_project(tmp_path)
    (tmp_path / "data" / "source_registry.json").unlink()

    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "registry_unreadable" in error_codes(report)
    assert report.documents == []
    assert not report.is_valid


def test_target_years_required_by_default(tmp_path: Path) -> None:
    build_project(tmp_path)

    strict = corpus_check.verify_corpus(project_root=tmp_path)
    relaxed = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    assert "missing_corpus_year" in error_codes(strict)
    assert relaxed.is_valid


# ---------------------------------------------------------------------------
# Отчёты и CLI
# ---------------------------------------------------------------------------

def test_reports_are_written_as_json_and_markdown(tmp_path: Path) -> None:
    build_project(tmp_path)
    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)
    json_path = tmp_path / "out" / "report.json"
    md_path = tmp_path / "out" / "report.md"

    corpus_check.write_reports(
        report, json_path=json_path, markdown_path=md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["is_valid"] is True
    assert payload["summary"]["chunk_count"] == 4
    assert len(payload["documents"]) == 2
    assert payload["documents"][0]["doc_id"] == "sber_2022"
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Отчёт о валидации корпуса" in markdown
    assert "sber_2022" in markdown
    assert "OK" in markdown


def test_cli_returns_zero_for_clean_corpus_and_one_for_broken(
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    build_project(tmp_path)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    argv = [
        "--project-root", str(tmp_path),
        "--json", str(json_path),
        "--markdown", str(md_path),
        "--allow-partial",
    ]

    exit_code = corpus_check.main(argv)
    assert exit_code == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"][
        "is_valid"] is True
    assert "СТАТУС: OK" in capsys.readouterr().out

    chunk_file(tmp_path, "sber_2023").unlink()
    exit_code = corpus_check.main(argv)
    assert exit_code == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"][
        "is_valid"] is False


def test_console_report_mentions_missing_documents(tmp_path: Path) -> None:
    build_project(tmp_path)
    chunk_file(tmp_path, "sber_2023").unlink()
    report = corpus_check.verify_corpus(
        project_root=tmp_path, require_target_years=False)

    console = corpus_check.format_console(report)

    assert "Без чанков: sber_2023" in console
    assert "ЕСТЬ ОШИБКИ" in console
