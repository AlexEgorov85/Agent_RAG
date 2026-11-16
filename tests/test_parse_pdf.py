"""Тесты задач 3.1–3.4, 4.1: parse_pdf (моки partition_pdf), resume, backup, parse_all, сериализация."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import pytest

from src.core.config import ParserSettings
from src.core.models import ParseResult, RawElement
from src.ingestion.pdf_parser import (
    format_stats_table,
    is_already_parsed,
    load_stats,
    pages_to_ranges,
    parse_all,
    parse_pdf,
    parse_sample,
    save_artifacts,
    select_sample_pages,
)


def fake_element(cat: str, text: str, page: int, depth=None, html=None, b64=None):
    """Поддельный Unstructured-элемент (duck-typing: атрибуты, которые читает parser)."""
    return SimpleNamespace(
        category=cat,
        text=text,
        metadata=SimpleNamespace(
            page_number=page,          # протокол парсера: abs. номер в metadata
            category_depth=depth,
            text_as_html=html,
            image_base64=b64,
            coordinates=SimpleNamespace(page_number=page,
                                        width=None, height=None),
        ),
    )


def fake_partition_factory(pages: int, fail_pages: set[int] | None = None,
                           expected_payload: bool = True):
    """partition_pdf-подделка под протокол §3.1 spec_pdf_parser.md v1.1.

    Новый протокол: ``file=BytesIO`` (вырезка страниц pypdf) + ``starting_page_number``.
    Подделка читает реальный буфер (число страниц в нём через pypdf) и возвращает
    по 2 элемента на каждую страницу вырезки с абсолютными номерами страниц.
    Опциональные сбои — по первой странице вырезки.
    """
    calls: List[Tuple[int, int]] = []

    def fake_partition_pdf(*, file, strategy, infer_table_structure,
                           extract_image_block_types, extract_image_block_to_payload,
                           starting_page_number, languages, **_ignored):
        from pypdf import PdfReader

        assert infer_table_structure == (strategy == "hi_res")
        # base64 кладётся в payload только если store_image_payload=True
        assert extract_image_block_to_payload is expected_payload
        buffer_pages = len(PdfReader(file).pages)
        start = starting_page_number
        calls.append((start, buffer_pages))
        if fail_pages and start in fail_pages:
            raise RuntimeError(f"mock OCR fail {start}")
        out = []
        for offset in range(buffer_pages):
            page = start + offset
            out.append(fake_element("Title", f"Раздел {page}", page, depth=page % 3))
            out.append(fake_element("NarrativeText", f"Текст стр {page}", page))
        return out

    fake_partition_pdf.calls = calls  # type: ignore[attr-defined]
    return fake_partition_pdf


@pytest.fixture
def parser_env(tmp_path: Path) -> ParserSettings:
    """Настройки с tmp-каталогами, без hi_res-проверок."""
    return ParserSettings(
        parse_strategy="fast",
        parse_page_batch=2,
        data_raw_dir=tmp_path / "raw",
        data_elements_dir=tmp_path / "elements",
        data_samples_dir=tmp_path / "samples",
        ocr_languages=("eng",),
    )


@pytest.fixture
def three_page_pdf(parser_env: ParserSettings) -> Path:
    raw = parser_env.data_raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    from tests.conftest import make_blank_pdf

    return make_blank_pdf(raw / "Пустышка 2099.pdf", n_pages=3)


def test_parse_pdf_batched(monkeypatch, parser_env, three_page_pdf):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    result = parse_pdf(three_page_pdf, settings=parser_env)
    assert result is not None
    # 3 страницы по 2 элемента = 6; батчи по 2 стр -> 2 вызова (протокол grow: вырезки)
    assert len(fake.calls) == 2
    # (стартовая страница вырезки, число страниц в вырезке)
    assert fake.calls == [(1, 2), (3, 1)]
    assert result.stats["n_elements"] == 6
    assert result.stats["n_pages"] == 3
    assert result.stats["doc_id"] == "sber_2099"
    assert result.stats["skipped_page_ranges"] == []
    by_type = result.stats["by_type"]
    assert by_type == {"Title": 3, "NarrativeText": 3}
    # Стабильные element_id по порядку
    assert [el.element_id for el in result.elements] == [
        f"sber_2099:{i:05d}" for i in range(6)
    ]
    # Артефакты записаны
    assert (parser_env.data_elements_dir / "sber_2099_elements.json").is_file()
    assert (parser_env.data_elements_dir / "sber_2099_stats.json").is_file()


def test_parse_pdf_retry_and_skipped_ranges(monkeypatch, parser_env, three_page_pdf):
    fake = fake_partition_factory(3, fail_pages={1})  # батч 1-2 падает дважды
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    result = parse_pdf(three_page_pdf, settings=parser_env)
    assert result is not None
    assert result.stats["skipped_page_ranges"] == ["1-2"]
    # Только страницы 3 попали в артефакт
    assert result.stats["n_elements"] == 2
    assert {el.page for el in result.elements} == {3}
    # Попыток было 2 на сбойный батч + 1 успех = 3 вызовов
    assert len(fake.calls) == 3


def test_build_page_slice_cuts_requested_range(parser_env, three_page_pdf):
    """Fix §3.1 v1.1: вырезка страниц pypdf именно запрошенного диапазона."""
    import io

    from pypdf import PdfReader

    from src.ingestion.pdf_parser import build_page_slice

    buffer = build_page_slice(three_page_pdf, start=2, end=3)
    assert isinstance(buffer, io.BytesIO)
    assert len(PdfReader(buffer).pages) == 2
    with pytest.raises(ValueError, match="вне документа"):
        build_page_slice(three_page_pdf, start=2, end=9)
    with pytest.raises(ValueError, match="вне документа"):
        build_page_slice(three_page_pdf, start=3, end=2)


def test_page_number_read_from_metadata_page_number(monkeypatch, parser_env,
                                                    three_page_pdf):
    """Real unstructured держит abs. номер в metadata.page_number (coords может не быть)."""
    def partition_mixed_sources(*, file, strategy, infer_table_structure,
                                extract_image_block_types,
                                extract_image_block_to_payload,
                                starting_page_number, languages, **_ignored):
        start = starting_page_number
        # Кейс 1: только metadata.page_number (реальное поведение unstructured);
        # Кейс 2: только coordinates.page_number (fallback на координаты).
        return [
            SimpleNamespace(
                category="NarrativeText", text="только meta",
                metadata=SimpleNamespace(page_number=start, category_depth=None,
                                         text_as_html=None, image_base64=None,
                                         coordinates=None)),
            SimpleNamespace(
                category="NarrativeText", text="только coords",
                metadata=SimpleNamespace(page_number=None, category_depth=None,
                                         text_as_html=None, image_base64=None,
                                         coordinates=SimpleNamespace(
                                             page_number=start, width=None,
                                             height=None))),
        ]
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf",
                        partition_mixed_sources)
    result = parse_pdf(three_page_pdf, settings=parser_env)
    assert result is not None
    # Батчи [1..2] и [3..3]: оба механизма чтения номера страницы работают
    pages = {el.page for el in result.elements}
    assert pages == {1, 3}
    assert result.stats["n_elements"] == 4


def test_page_label_filled_from_pdf(monkeypatch, parser_env, three_page_pdf):
    """Переиспользование контракта: у элементов заполнено поле page_label."""
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    result = parse_pdf(three_page_pdf, settings=parser_env)
    assert result is not None
    pages = {el.page for el in result.elements}
    assert pages == {1, 2, 3}
    for el in result.elements:
        assert el.page_label == str(el.page)


def test_element_stolen_page_number_fails_like_reference(monkeypatch, parser_env, three_page_pdf):
    """Валидация контракта: элемент с чужим номером страницы роняет парсинг."""
    def broken_partition(*, file, strategy, infer_table_structure,
                         extract_image_block_types, extract_image_block_to_payload,
                         starting_page_number, languages, **_ignored):
        return [fake_element("NarrativeText", "сдвинутый", page=99)]
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", broken_partition)
    with pytest.raises(ValueError, match="потерял номер исходной страницы"):
        parse_pdf(three_page_pdf, settings=parser_env)


def test_progress_log_format(monkeypatch, parser_env, three_page_pdf, caplog):
    """Задача 4.2: формат строки прогресса `[doc_id] batch i/n | pages a-b | X.Xs | N elements`."""
    import logging
    import re

    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    with caplog.at_level(logging.INFO, logger="src.ingestion.pdf_parser"):
        parse_pdf(three_page_pdf, settings=parser_env)
    progress_lines = [r.getMessage() for r in caplog.records if "batch " in r.getMessage()]
    assert progress_lines, "нет строк прогресса в логе"
    pattern = (r"\[sber_2099\] batch \d+/\d+ \| pages \d+-\d+ "
               r"\| \d+\.\d+s \| \d+ elements")
    assert re.fullmatch(pattern, progress_lines[0]), progress_lines[0]


def test_failed_batch_logged(monkeypatch, parser_env, three_page_pdf, caplog):
    """Задача 4.3: сбойный батч попадает в skipped_page_ranges и в лог (caplog)."""
    import logging

    fake = fake_partition_factory(3, fail_pages={1})  # батч 1-2 падает дважды
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    with caplog.at_level(logging.ERROR, logger="src.ingestion.pdf_parser"):
        result = parse_pdf(three_page_pdf, settings=parser_env)
    assert result is not None
    assert result.stats["skipped_page_ranges"] == ["1-2"]
    error_lines = [r.getMessage() for r in caplog.records if "не распарсен" in r.getMessage()]
    assert any("1-2" in line for line in error_lines), error_lines


def test_parse_pdf_resume_skips(monkeypatch, parser_env, three_page_pdf):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    assert parse_pdf(three_page_pdf, settings=parser_env) is not None
    first_calls = len(fake.calls)
    assert is_already_parsed(three_page_pdf, parser_env.data_elements_dir, "sber_2099")
    assert parse_pdf(three_page_pdf, settings=parser_env) is None
    assert len(fake.calls) == first_calls  # partition не вызывался второй раз


def test_parse_pdf_source_changed_moves_backup(monkeypatch, parser_env, three_page_pdf):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    parse_pdf(three_page_pdf, settings=parser_env)
    sha_before = load_stats(parser_env.data_elements_dir, "sber_2099")["source_sha256"]
    # «Подмена» PDF: пересоздаём файл с иным содержимым
    from tests.conftest import make_blank_pdf

    make_blank_pdf(three_page_pdf, n_pages=3, width=100, height=100)
    assert parse_pdf(three_page_pdf, settings=parser_env) is not None
    # Старые артефакты ушли в backup, новая статистика — с новым sha256
    backup = parser_env.data_elements_dir / "backup"
    assert backup.is_dir() and list(backup.glob("sber_2099_*"))
    sha_after = load_stats(parser_env.data_elements_dir, "sber_2099")["source_sha256"]
    assert sha_before != sha_after


def test_parse_all_skips_parsed(monkeypatch, parser_env, three_page_pdf):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    results = parse_all(parser_env.data_raw_dir, settings=parser_env)
    assert len(results) == 1
    assert parse_all(parser_env.data_raw_dir, settings=parser_env) == []


def test_save_artifacts_utf8_and_consistency(parser_env, three_page_pdf):
    el = RawElement(element_id="sber_2099:00000", doc_id="sber_2099",
                    type="Title", text="ГОДОВОЙ ОТЧЕТ ОАО «БПС-СБЕРБАНК»", page=1)
    stats = {"doc_id": "sber_2099", "n_elements": 1, "by_type": {"Title": 1}}
    save_artifacts("sber_2099", ParseResult([el], stats), parser_env.data_elements_dir)
    data = json.loads((parser_env.data_elements_dir / "sber_2099_elements.json")
                      .read_text(encoding="utf-8"))
    assert data[0]["text"].startswith("ГОДОВОЙ")
    st = load_stats(parser_env.data_elements_dir, "sber_2099")
    assert sum(st["by_type"].values()) == st["n_elements"] == 1


def test_format_stats_table(parser_env, three_page_pdf, monkeypatch):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    parse_pdf(three_page_pdf, settings=parser_env)
    table = format_stats_table(settings=parser_env)
    assert "sber_2099" in table and "6" in table


def test_no_image_payload_setting(monkeypatch, parser_env, three_page_pdf):
    """store_image_payload=False: payload выключен, подпись это фиксирует."""
    fake = fake_partition_factory(3, expected_payload=False)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    settings = parser_env.model_copy(update={"store_image_payload": False})
    result = parse_pdf(three_page_pdf, settings=settings)
    assert result is not None
    signature = result.stats["parse_signature"]
    assert signature["store_image_payload"] is False
    assert signature["schema_version"] == 2


def test_signature_change_reparses_with_backup(monkeypatch, parser_env,
                                               three_page_pdf):
    """Смена настроек парсинга делает артефакт неактуальным (не только sha256)."""
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    parse_pdf(three_page_pdf, settings=parser_env)
    assert is_already_parsed(three_page_pdf, parser_env.data_elements_dir,
                             "sber_2099", settings=parser_env)

    changed = parser_env.model_copy(update={"store_image_payload": False})
    assert not is_already_parsed(three_page_pdf, parser_env.data_elements_dir,
                                 "sber_2099", settings=changed)
    result = parse_pdf(three_page_pdf, settings=changed)
    assert result is not None
    backup = parser_env.data_elements_dir / "backup"
    assert backup.is_dir() and list(backup.glob("sber_2099_*"))
    assert result.stats["parse_signature"]["store_image_payload"] is False


def test_force_reparses_actual_artifact(monkeypatch, parser_env, three_page_pdf):
    """--force перепарсивает даже актуальный артефакт и сохраняет старый в backup."""
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    parse_pdf(three_page_pdf, settings=parser_env)
    result = parse_pdf(three_page_pdf, settings=parser_env, force=True)
    assert result is not None
    backup = parser_env.data_elements_dir / "backup"
    assert backup.is_dir() and list(backup.glob("sber_2099_*"))


def test_backup_keeps_elements_and_stats_files(monkeypatch, parser_env,
                                               three_page_pdf):
    """Багфикс: _elements.json и _stats.json имеют один суффикс — не затирать."""
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    parse_pdf(three_page_pdf, settings=parser_env)
    parse_pdf(three_page_pdf, settings=parser_env, force=True)
    names = [path.name for path in
             (parser_env.data_elements_dir / "backup").glob("sber_2099_*")]
    assert any("elements" in name for name in names), names
    assert any("stats" in name for name in names), names


def test_select_sample_pages_spans_toc_and_body():
    pages = select_sample_pages(100, total_pages=10, toc_pages=6)
    assert pages[:6] == [1, 2, 3, 4, 5, 6]
    assert len(pages) == 10
    assert pages == sorted(set(pages))
    assert all(7 <= page <= 100 for page in pages[6:])
    # Маленький документ: только TOC-страницы, без выхода за границы
    assert select_sample_pages(4, total_pages=10, toc_pages=6) == [1, 2, 3, 4]


def test_pages_to_ranges_groups_consecutive_pages():
    assert pages_to_ranges([1, 2, 4, 5, 6, 9]) == [(1, 2), (4, 6), (9, 9)]
    assert pages_to_ranges([]) == []


def test_parse_sample_writes_only_to_samples_dir(monkeypatch, parser_env,
                                                 three_page_pdf):
    fake = fake_partition_factory(3)
    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf", fake)
    result = parse_sample(three_page_pdf, settings=parser_env, pages=[1, 3])
    assert result.stats["sample_pages"] == [1, 3]
    assert result.stats["n_pages_sampled"] == 2
    assert {el.page for el in result.elements} == {1, 3}
    samples = parser_env.data_samples_dir
    assert (samples / "sber_2099_elements.json").is_file()
    assert (samples / "sber_2099_stats.json").is_file()
    assert (samples / "figures" / "sber_2099").is_dir()
    # Канонический data/elements не тронут
    assert not (parser_env.data_elements_dir / "sber_2099_elements.json").exists()


def test_image_element_keeps_bbox_path_and_prob(monkeypatch, parser_env,
                                                three_page_pdf):
    """Схема v2: bbox/page_size/prob/image_path доступны downstream-этапам."""
    def partition_one_image(*, file, strategy, infer_table_structure,
                            extract_image_block_types,
                            extract_image_block_to_payload,
                            starting_page_number, languages, **_ignored):
        return [SimpleNamespace(
            category="Image", text="график",
            metadata=SimpleNamespace(
                page_number=starting_page_number,
                category_depth=None, text_as_html=None, image_base64=None,
                detection_class_prob=0.875,
                image_path="data/figures/sber_2099/figure-1-1.jpg",
                coordinates=SimpleNamespace(
                    page_number=starting_page_number,
                    points=((10.0, 20.0), (10.0, 120.0),
                            (210.0, 120.0), (210.0, 20.0)),
                    layout_width=1000, layout_height=500,
                    width=None, height=None),
            ),
        )]

    monkeypatch.setattr("unstructured.partition.pdf.partition_pdf",
                        partition_one_image)
    settings = parser_env.model_copy(update={"store_image_payload": False})
    result = parse_pdf(three_page_pdf, settings=settings)
    assert result is not None
    element = next(el for el in result.elements if el.type == "Image")
    assert element.bbox == [10.0, 20.0, 210.0, 120.0]
    assert element.page_width == 1000 and element.page_height == 500
    assert element.detection_class_prob == 0.875
    assert element.image_path == "data/figures/sber_2099/figure-1-1.jpg"
    assert element.area_px == 20000
    assert element.image_b64 is None
