"""Тесты задач 2.2/2.3: sha256, число страниц, preflight среды OCR."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ingestion.pdf_parser import (
    ParserEnvironmentError,
    compute_source_sha256,
    count_pdf_pages,
    preflight_check_ocr,
)


def test_sha256_matches_reference(blank_pdf_3_pages: Path):
    expected = hashlib.sha256(blank_pdf_3_pages.read_bytes()).hexdigest()
    assert compute_source_sha256(blank_pdf_3_pages) == expected


def test_count_pages_threes(blank_pdf_3_pages: Path):
    assert count_pdf_pages(blank_pdf_3_pages) == 3


def test_preflight_ok(tmp_path: Path):
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "rus.traineddata").write_bytes(b"")
    (tessdata / "eng.traineddata").write_bytes(b"")
    # Фейковый "tesseract.exe" в явном каталоге
    t_dir = tmp_path / "t"
    t_dir.mkdir()
    (t_dir / "tesseract.exe").write_bytes(b"MZ fake")
    result = preflight_check_ocr(["rus", "eng"], tesseract_dir=str(t_dir),
                                 tessdata_prefix=tessdata)
    assert result is not None


def test_preflight_missing_tesseract():
    with pytest.raises(ParserEnvironmentError, match="winget"):
        preflight_check_ocr(["rus"], tesseract_dir=r"Z:\definitely\not\here")


def test_preflight_missing_rus_traineddata(tmp_path: Path):
    t_dir = tmp_path / "t"
    t_dir.mkdir()
    (t_dir / "tesseract.exe").write_bytes(b"MZ fake")
    tessdata = tmp_path / "td"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(b"")
    with pytest.raises(ParserEnvironmentError, match="rus.traineddata"):
        preflight_check_ocr(["rus", "eng"], tesseract_dir=str(t_dir),
                            tessdata_prefix=tessdata)
