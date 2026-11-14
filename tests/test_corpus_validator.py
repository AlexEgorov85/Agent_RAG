"""Focused deterministic tests for corpus validation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.ingestion.corpus_validator import validate_corpus


@pytest.fixture
def valid_chunk() -> dict[str, object]:
    """A minimal chunk matching the ingestion contract."""
    return {
        "chunk_id": "sber_2023:00001:c0:abc123",
        "evidence_id": "sber_2023:p001",
        "doc_id": "sber_2023",
        "title": "Annual report",
        "page_pdf": 1,
        "page_label": "1",
        "year": 2023,
        "report_period": "2023",
        "kind": "text",
        "element_ids": ["sber_2023:00001"],
        "text": "Revenue increased.",
        "html": None,
        "source_url": None,
        "source_sha256": "a" * 64,
    }


@pytest.fixture
def page_counts() -> dict[str, int]:
    """Explicit PDF page counts for the fixture document."""
    return {"sber_2023": 3}


def test_valid_chunk_returns_clean_report(valid_chunk, page_counts) -> None:
    report = validate_corpus([valid_chunk], page_counts)

    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.chunk_count == 1
    assert report.structure_coverage == 0.0


def test_collects_missing_fields_and_invalid_values(valid_chunk, page_counts) -> None:
    broken = deepcopy(valid_chunk)
    del broken["title"]
    broken["text"] = " "
    broken["year"] = True
    broken["source_sha256"] = ""

    report = validate_corpus([broken], page_counts)
    codes = {finding.code for finding in report.errors}

    assert not report.is_valid
    assert {
        "missing_required_field",
        "invalid_required_field",
        "invalid_year",
        "missing_source_sha256",
    } <= codes


@pytest.mark.parametrize(
    "source_sha256",
    [
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "a" * 63 + "!",
    ],
)
def test_rejects_malformed_source_sha256(
    valid_chunk, page_counts, source_sha256
) -> None:
    chunk = deepcopy(valid_chunk)
    chunk["source_sha256"] = source_sha256

    report = validate_corpus([chunk], page_counts)

    assert "invalid_source_sha256" in {
        finding.code for finding in report.errors
    }


def test_accepts_uppercase_sha256(valid_chunk, page_counts) -> None:
    chunk = deepcopy(valid_chunk)
    chunk["source_sha256"] = "ABCDEF0123456789" * 4

    report = validate_corpus([chunk], page_counts)

    assert report.is_valid
    assert "invalid_source_sha256" not in {
        finding.code for finding in report.errors
    }


def test_detects_duplicate_ids_and_page_bounds(valid_chunk, page_counts) -> None:
    first = deepcopy(valid_chunk)
    second = deepcopy(valid_chunk)
    second["page_pdf"] = 4

    report = validate_corpus([first, second], page_counts)
    codes = [finding.code for finding in report.errors]

    assert "duplicate_chunk_id" in codes
    assert "page_out_of_range" in codes
    assert report.error_count == len(report.errors)


@pytest.mark.parametrize("page", [0, -1, 1.5, True, None])
def test_page_pdf_must_be_positive_integer(valid_chunk, page_counts, page) -> None:
    chunk = deepcopy(valid_chunk)
    chunk["page_pdf"] = page

    report = validate_corpus([chunk], page_counts)

    assert "invalid_page_pdf" in {finding.code for finding in report.errors}


def test_requires_explicit_valid_page_count(valid_chunk) -> None:
    report = validate_corpus([valid_chunk], {})

    assert "invalid_page_count" in {finding.code for finding in report.errors}


def test_checks_table_html_and_non_table_html(valid_chunk, page_counts) -> None:
    table = deepcopy(valid_chunk)
    table.update(kind="table", html="<p>not a table</p>")
    ordinary = deepcopy(valid_chunk)
    ordinary.update(chunk_id="different", html="<table></table>")

    report = validate_corpus([table, ordinary], page_counts)
    codes = {finding.code for finding in report.errors}

    assert "table_html_invalid" in codes
    assert "unexpected_html" in codes


def test_validates_structure_metadata_and_coverage(valid_chunk, page_counts) -> None:
    chunk = deepcopy(valid_chunk)
    chunk.update(
        section_path="Risk/Credit",
        breadcrumbs=["Risk", "Credit"],
        section_level=1,
        parent_section="Risk",
    )

    report = validate_corpus(
        [chunk], page_counts, min_structure_coverage=1.0
    )

    assert report.is_valid
    assert report.structure_coverage == 1.0
    assert report.warning_count == 0

    chunk["section_path"] = "Risk"
    invalid = validate_corpus([chunk], page_counts)
    assert "inconsistent_section_path" in {
        finding.code for finding in invalid.errors
    }


def test_structure_threshold_is_a_warning(valid_chunk, page_counts) -> None:
    report = validate_corpus(
        [valid_chunk], page_counts, min_structure_coverage=0.5
    )

    assert report.is_valid
    assert report.warning_count == 1
    assert report.warnings[0].code == "structure_coverage_below_threshold"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_rejects_invalid_structure_threshold(
    valid_chunk, page_counts, threshold
) -> None:
    with pytest.raises(ValueError):
        validate_corpus(
            [valid_chunk], page_counts, min_structure_coverage=threshold
        )
