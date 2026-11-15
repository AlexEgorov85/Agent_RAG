"""Тесты реестра источников корпуса (``src/core/source_registry.py``)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.core.source_registry import (
    BPS_COUNTRY,
    BPS_ISSUER,
    RegistryError,
    SOURCE_STATUS_COMPARABLE,
    SOURCE_STATUS_INCOMPARABLE,
    SourceRecord,
    compute_sha256,
    load_registry,
    record_by_doc,
    registry_fingerprint,
    save_registry,
    validate_registry,
)


def make_pdf(root: Path, doc_id: str, content: bytes = b"%PDF-1.4 test") -> Path:
    """Создать фиктивный PDF и вернуть путь к нему."""
    path = root / "data" / "raw" / f"Сбер {doc_id.rsplit('_', 1)[-1]}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def make_record(
    root: Path,
    doc_id: str = "sber_2023",
    *,
    year: int | None = None,
    issuer: str = "ПАО Сбербанк",
    country: str = "Россия",
    status: str = SOURCE_STATUS_COMPARABLE,
    note: str = "Включён в сопоставимый корпус.",
    file_path: str | None = None,
    sha256: str | None = None,
) -> SourceRecord:
    """Собрать валидную запись поверх реально созданного файла."""
    pdf = make_pdf(root, doc_id)
    report_year = year if year is not None else int(doc_id.rsplit("_", 1)[-1])
    return SourceRecord(
        doc_id=doc_id,
        file_path=file_path or pdf.relative_to(root).as_posix(),
        source_sha256=sha256 or compute_sha256(pdf),
        year=report_year,
        report_period=str(report_year),
        issuer=issuer,
        country=country,
        source_status=status,
        inclusion_note=note,
    )


def codes(validation) -> set[str]:
    """Множество кодов ошибок валидации."""
    return {issue.code for issue in validation.errors}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Корень фиктивного проекта."""
    return tmp_path


def test_roundtrip_preserves_all_fields(project: Path) -> None:
    record = make_record(project)
    path = save_registry([record], project / "data" / "source_registry.json",
                         notes="тестовый корпус")

    loaded = load_registry(path)

    assert loaded == [record]
    assert loaded[0].to_dict() == record.to_dict()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["notes"] == "тестовый корпус"
    assert payload["sources"][0]["doc_id"] == "sber_2023"
    # Атомарная запись не оставляет временных файлов.
    assert not list(path.parent.glob("*.tmp"))


def test_save_overwrites_and_accepts_plain_list(project: Path) -> None:
    first = make_record(project, "sber_2022", file_path="data/raw/a.pdf")
    path = project / "registry.json"
    path.write_text(json.dumps([first.to_dict()], ensure_ascii=False),
                    encoding="utf-8")
    assert load_registry(path) == [first]

    second = replace(first, doc_id="sber_2023", year=2023, report_period="2023")
    save_registry([second], path)
    assert load_registry(path) == [second]


def test_load_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(broken)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text('{"items": []}', encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(wrong_shape)

    missing_field = tmp_path / "missing.json"
    missing_field.write_text('[{"doc_id": "sber_2023"}]', encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(missing_field)


def test_duplicate_doc_id_is_an_error(project: Path) -> None:
    first = make_record(project)
    second = replace(first, file_path="data/raw/other.pdf")

    validation = validate_registry(
        [first, second], project_root=project, require_target_years=False)

    assert not validation.is_valid
    assert "duplicate_doc_id" in codes(validation)


def test_duplicate_file_path_is_detected_across_doc_ids(project: Path) -> None:
    first = make_record(project, "sber_2022")
    second = replace(make_record(project, "sber_2023"),
                     file_path=first.file_path)

    validation = validate_registry(
        [first, second], project_root=project, require_target_years=False)

    assert "duplicate_file_path" in codes(validation)


def test_year_must_match_doc_id(project: Path) -> None:
    record = make_record(project, "sber_2023", year=2022)

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "doc_id_year_mismatch" in codes(validation)


def test_invalid_doc_id_is_rejected(project: Path) -> None:
    record = make_record(project, "report_2023")
    record = replace(record, doc_id="report")

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "invalid_doc_id" in codes(validation)


def test_missing_file_is_reported(project: Path) -> None:
    record = make_record(project)
    record = replace(record, file_path="data/raw/absent.pdf")

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "missing_file" in codes(validation)


def test_sha256_mismatch_is_reported(project: Path) -> None:
    record = make_record(project)
    record = replace(record, source_sha256=hashlib.sha256(b"other").hexdigest())

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "source_sha256_mismatch" in codes(validation)


def test_malformed_sha256_is_reported_without_reading_file(project: Path) -> None:
    record = replace(make_record(project), source_sha256="abc")

    validation = validate_registry(
        [record], project_root=project, verify_sha256=False,
        require_target_years=False)

    assert "invalid_source_sha256" in codes(validation)


def test_2021_must_not_be_in_registry(project: Path) -> None:
    record = make_record(project, "sber_2021")

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "missing_year_present" in codes(validation)


def test_2015_must_be_marked_as_bps(project: Path) -> None:
    wrong = make_record(project, "sber_2015")

    validation = validate_registry(
        [wrong], project_root=project, require_target_years=False)

    assert {
        "issuer_mismatch",
        "country_mismatch",
        "comparable_foreign_issuer",
    } <= codes(validation)

    correct = replace(
        wrong,
        issuer=BPS_ISSUER,
        country=BPS_COUNTRY,
        source_status=SOURCE_STATUS_INCOMPARABLE,
        inclusion_note="Документ несопоставим с ПАО Сбербанк.",
    )
    clean = validate_registry(
        [correct], project_root=project, require_target_years=False)
    assert clean.is_valid
    assert not correct.is_comparable


def test_bps_issuer_is_rejected_for_other_years(project: Path) -> None:
    record = replace(make_record(project), issuer=BPS_ISSUER)

    validation = validate_registry(
        [record], project_root=project, require_target_years=False)

    assert "mislabeled_issuer" in codes(validation)


def test_target_years_completeness_is_checked(project: Path) -> None:
    records = []
    for year in (2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024):
        doc_id = f"sber_{year}"
        record = make_record(project, doc_id)
        if year == 2015:
            record = replace(
                record,
                issuer=BPS_ISSUER,
                country=BPS_COUNTRY,
                source_status=SOURCE_STATUS_INCOMPARABLE,
                inclusion_note="Несопоставим с ПАО Сбербанк.",
            )
        records.append(record)

    complete = validate_registry(
        records, project_root=project, verify_sha256=False)
    assert complete.is_valid
    assert complete.doc_ids[0] == "sber_2015"

    incomplete = validate_registry(
        records[:-1], project_root=project, verify_sha256=False)
    assert "missing_corpus_year" in codes(incomplete)
    assert any(issue.doc_id == "sber_2024" for issue in incomplete.errors)

    relaxed = validate_registry(
        records[:-1], project_root=project, verify_sha256=False,
        require_target_years=False)
    assert relaxed.is_valid


def test_empty_registry_is_invalid(project: Path) -> None:
    validation = validate_registry(
        [], project_root=project, require_target_years=False)

    assert "empty_registry" in codes(validation)


def test_record_index_and_fingerprint(project: Path) -> None:
    first = make_record(project, "sber_2022")
    second = make_record(project, "sber_2023")

    index = record_by_doc([first, second])
    assert index["sber_2022"] is first

    fingerprint = registry_fingerprint([first, second])
    assert fingerprint == registry_fingerprint([second, first])
    assert len(fingerprint) == 64
    assert registry_fingerprint([first]) != fingerprint

    with pytest.raises(RegistryError):
        record_by_doc([first, first])
