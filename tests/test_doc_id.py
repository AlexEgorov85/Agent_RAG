"""Тест задачи 2.1: make_doc_id на реальных именах файлов корпуса."""

from __future__ import annotations

import pytest

from src.ingestion.pdf_parser import element_id, make_doc_id


REAL_NAMES = [
    "Копия Сбер 2015.pdf", "Копия Сбер 2016.pdf", "Копия Сбер 2017.pdf",
    "Копия Сбер 2018.pdf", "Копия Сбер 2019.pdf", "Копия Сбер 2020.pdf",
    "Копия Сбер 2022.pdf", "Копия Сбер 2023.pdf", "Копия Сбер 2024.pdf",
    "Копия Сбер 2025.pdf",
]


def test_all_real_files_have_unique_doc_ids():
    doc_ids = [make_doc_id(n) for n in REAL_NAMES]
    assert len(set(doc_ids)) == 10
    assert doc_ids == [f"sber_{y}" for y in (2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025)]
    assert "sber_2021" not in doc_ids  # 2021 в корпусе отсутствует физически


def test_year_is_last_digit_group():
    # Перрон на пути: две 4-цифровые группы — берём последнюю
    assert make_doc_id("report 2000 final 2015.pdf") == "sber_2015"


def test_element_id_is_zero_padded():
    assert element_id("sber_2015", 42) == "sber_2015:00042"
    assert element_id("sber_2024", 99999) == "sber_2024:99999"


def test_missing_year_raises():
    with pytest.raises(ValueError):
        make_doc_id("без_года.pdf")
