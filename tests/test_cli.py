"""Тесты задачи 5.1: CLI-режимы `__main__` (python -m src.ingestion) на tmp-каталогах."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from src.core.config import ParserSettings
from src.ingestion.cli import main, build_arg_parser


@pytest.fixture
def parser_env(tmp_path: Path) -> ParserSettings:
    return ParserSettings(
        parse_strategy="fast",
        parse_page_batch=2,
        data_raw_dir=tmp_path / "raw",
        data_elements_dir=tmp_path / "elements",
        ocr_languages=("eng",),
    )


@pytest.fixture
def pdf_file(parser_env: ParserSettings) -> Path:
    from tests.conftest import make_blank_pdf

    parser_env.data_raw_dir.mkdir(parents=True, exist_ok=True)
    make_blank_pdf(parser_env.data_raw_dir / "Пустышка 2099.pdf", n_pages=3)
    return parser_env.data_raw_dir / "Пустышка 2099.pdf"


def fake_partition_factory(n_fail_pages: int = 0):
    """partition_pdf-подделка (протокол §3.1 v1.1): фиксирует strategy и вырезки."""
    slices_seen: List[Tuple[int, int]] = []
    strategies: List[str] = []

    def fake_partition_pdf(*, file, strategy, infer_table_structure,
                           extract_image_block_types, extract_image_block_to_payload,
                           starting_page_number, languages, **_ignored):
        from pypdf import PdfReader

        slices_seen.append((starting_page_number, len(PdfReader(file).pages)))
        strategies.append(strategy)
        return []

    fake_partition_pdf.slices_seen = slices_seen  # type: ignore[attr-defined]
    fake_partition_pdf.strategies = strategies  # type: ignore[attr-defined]
    return fake_partition_pdf


@pytest.fixture
def patch_partition(monkeypatch):
    """Подменить partition_pdf на тишину и вернуть заглушку."""
    fake = fake_partition_factory()
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    return fake


def test_cli_pdf_mode_creates_artifacts(parser_env, pdf_file, patch_partition):
    code = main(["--pdf", str(pdf_file)], settings=parser_env)
    assert code == 0
    assert (parser_env.data_elements_dir / "sber_2099_elements.json").is_file()
    assert (parser_env.data_elements_dir / "sber_2099_stats.json").is_file()
    assert patch_partition.strategies == ["fast", "fast"]  # батчи 1-2, 3-3


def test_cli_pdf_resume_skips(parser_env, pdf_file, patch_partition, capsys):
    assert main(["--pdf", str(pdf_file)], settings=parser_env) == 0
    calls_before = len(patch_partition.slices_seen)
    assert main(["--pdf", str(pdf_file)], settings=parser_env) == 0
    assert len(patch_partition.slices_seen) == calls_before
    out = capsys.readouterr().out
    assert "уже спарсен" in out


def test_cli_all_mode(parser_env, pdf_file, patch_partition):
    assert main(["--all"], settings=parser_env) == 0
    assert (parser_env.data_elements_dir / "sber_2099_stats.json").is_file()
    # второй прогон не пересоздаёт
    calls_before = len(patch_partition.slices_seen)
    assert main(["--all"], settings=parser_env) == 0
    assert len(patch_partition.slices_seen) == calls_before


def test_cli_stats_mode(parser_env, pdf_file, patch_partition, capsys):
    main(["--all"], settings=parser_env)
    code = main(["--stats"], settings=parser_env)
    assert code == 0
    out = capsys.readouterr().out
    # Режим --stats: doc_id, страницы, элементы, время
    assert "sber_2099" in out and "3" in out


def test_cli_fast_preview_no_artifacts(parser_env, pdf_file, patch_partition, capsys):
    code = main(["--pdf", str(pdf_file), "--fast-preview"], settings=parser_env)
    assert code == 0
    out = capsys.readouterr().out
    assert "fast-preview" in out
    assert list(parser_env.data_elements_dir.glob("*")) == []
    # fast обязательно активирован вне зависимости от env-стратегии
    assert set(patch_partition.strategies) == {"fast"}


def test_cli_fast_preview_overrides_hi_res_env(parser_env, pdf_file,
                                               monkeypatch, capsys):
    """Приоритет CLI над env: hi_res из настроек перекрыт --fast-preview."""
    hi_res_env = parser_env.model_copy(update={"parse_strategy": "hi_res"})
    fake = fake_partition_factory()
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    # hi_res (без fast-preview) потребовал бы tesseract; с fast-preview — нет OCR
    code = main(["--pdf", str(pdf_file), "--fast-preview"],
                settings=hi_res_env)
    assert code == 0
    assert set(fake.strategies) == {"fast"}


def test_cli_smoke_mode(parser_env, pdf_file, patch_partition, capsys):
    code = main(["--smoke", "--pages", "2"], settings=parser_env)
    assert code == 0
    out = capsys.readouterr().out
    assert "[smoke]" in out and "2" in out
    assert list(parser_env.data_elements_dir.glob("*")) == []


def test_cli_smoke_no_pdf_fails(parser_env):
    with pytest.raises(FileNotFoundError):
        main(["--smoke"], settings=parser_env)


def test_cli_no_mode_prints_help(capsys):
    code = main([], settings=None)
    assert code == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_modes_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--all", "--stats"])
