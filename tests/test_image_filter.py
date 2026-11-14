"""Тест задачи 2.4: фильтр мелких изображений."""

from __future__ import annotations

from src.ingestion.pdf_parser import apply_min_image_area

MIN_AREA = 200 * 200


def test_small_icon_filtered():
    assert apply_min_image_area("data:image/png;base64,AAAA", 40 * 40, MIN_AREA) is None


def test_large_chart_kept():
    b64 = "data:image/png;base64,BBBB"
    assert apply_min_image_area(b64, 800 * 600, MIN_AREA) == b64


def test_borderline_exact_area_kept():
    b64 = "x"
    assert apply_min_image_area(b64, MIN_AREA, MIN_AREA) == b64


def test_unknown_area_kept():
    # Если площадь неизвестна, не отфильтровываем — консервативно store
    assert apply_min_image_area("abc", None, MIN_AREA) == "abc"


def test_absent_base64_stays_none():
    assert apply_min_image_area(None, 800 * 600, MIN_AREA) is None
