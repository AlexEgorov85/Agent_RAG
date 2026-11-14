"""Тест задачи 1.1: RawElement / ParseResult сериализация туда-обратно."""

from __future__ import annotations

import json

from src.core.models import ParseResult, RawElement


def test_raw_element_roundtrip():
    el = RawElement(
        element_id="sber_2015:00042",
        doc_id="sber_2015",
        type="Table",
        text="Ключевые показатели",
        page=17,
        category_depth=-1,
        html="<table><tr><td>ROE</td></tr></table>",
        image_b64=None,
        area_px=None,
        bbox=[248.03, 304.22, 1048.6, 452.09],
        page_width=4686,
        page_height=2619,
        detection_class_prob=0.7729,
        image_path=None,
    )
    data = json.loads(json.dumps(el.to_dict(), ensure_ascii=False))
    assert RawElement.from_dict(data) == el


def test_raw_element_legacy_json_without_bbox():
    """Старые артефакты (до схемы v2) читаются с дефолтными None-полями."""
    data = {"element_id": "sber_2015:00000", "doc_id": "sber_2015",
            "type": "Title", "text": "Раздел", "page": 1}
    el = RawElement.from_dict(data)
    assert el.bbox is None and el.page_width is None
    assert el.detection_class_prob is None and el.image_path is None


def test_raw_element_ignores_unknown_keys():
    data = {"element_id": "x", "doc_id": "d", "type": "Text", "extra": 1}
    el = RawElement.from_dict(data)
    assert el.element_id == "x" and not hasattr(el, "extra")


def test_parse_result_defaults():
    el = RawElement(element_id="sber_2015:00000", doc_id="sber_2015", type="Title", text="Т", page=1)
    res = ParseResult(elements=[el])
    assert res.stats == {}
    assert res.elements[0].category_depth == -1
