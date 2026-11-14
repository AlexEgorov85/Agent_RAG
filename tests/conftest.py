"""Общие фикстуры тестов ingestion-слоя."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter


def make_blank_pdf(path: Path, n_pages: int = 3, width: int = 612, height: int = 792) -> Path:
    """Создать простой многополосный PDF (синие пустые страницы) для тестов."""
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=width, height=height)
    with open(path, "wb") as fh:
        writer.write(fh)
    return path


@pytest.fixture
def blank_pdf_3_pages(tmp_path: Path) -> Path:
    """PDF на 3 страницы для тестов метаданных."""
    return make_blank_pdf(tmp_path / "blank3.pdf", n_pages=3)
