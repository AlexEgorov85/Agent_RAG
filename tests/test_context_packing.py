"""Offline tests for whole-chunk context packing."""

from __future__ import annotations

import json

import pytest

from src.engine.context import CONTEXT_FIELDS, pack_context
from src.retrieval.hybrid_retriever import RetrievalHit


def make_chunk(chunk_id: str, text: str = "Net income was 100.", **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": "sber_2023",
        "year": 2023,
        "issuer": "ПАО Сбербанк",
        "page_pdf": 42,
        "page_label": "41",
        "section_path": "Управление рисками/Кредитный риск",
        "kind": "text",
        "text": text,
        "html": None,
    }
    chunk.update(overrides)
    return chunk


def test_packed_chunk_exposes_all_contract_fields_and_manifest():
    html = "<table><tr><td>Прибыль</td><td>100</td></tr></table>"
    packed = pack_context(
        [make_chunk("c1", text="Table: profit 100", kind="table", html=html)]
    )

    assert packed.manifest.included == ["c1"]
    assert packed.manifest.skipped == []
    assert packed.manifest.duplicates == []
    assert packed.manifest.max_chars == 16_000
    assert packed.manifest.used_chars == len(packed.text)
    assert packed.chunk_ids == ["c1"]

    record = packed.records[0]
    assert set(record) == set(CONTEXT_FIELDS)
    assert record["html"] == html
    assert record["text"] == "Table: profit 100"
    for name in CONTEXT_FIELDS:
        assert f'"{name}"' in packed.text
    # A table's HTML remains in the same rendered record as its text.
    assert json.loads(packed.text)["html"] == html
    assert json.loads(packed.text)["text"] in packed.text


def test_budget_skips_oversized_chunk_whole_and_keeps_later_small_chunks():
    first = make_chunk("a", text="A" * 60)
    large = make_chunk("b", text="B" * 2_000)
    last = make_chunk("c", text="C" * 60)

    small_budget = len(pack_context([first, last], max_chars=1_000_000).text)
    unbounded = pack_context([first, large, last], max_chars=1_000_000).text
    assert len(unbounded) > small_budget

    packed = pack_context([first, large, last], max_chars=small_budget)

    assert packed.manifest.included == ["a", "c"]
    assert packed.manifest.skipped == ["b"]
    assert packed.manifest.used_chars == len(packed.text) == small_budget
    assert packed.text.count('"chunk_id"') == 2
    assert ("A" * 60) in packed.text
    assert ("C" * 60) in packed.text
    # The large chunk is absent entirely; no prefix of it was truncated in.
    assert ("B" * 2_000) not in packed.text
    assert '"chunk_id": "b"' not in packed.text


def test_single_chunk_that_does_not_fit_is_never_truncated():
    chunk = make_chunk("big", text="WHOLE-CHUNK-CONTENT-" * 5)
    full_size = len(pack_context([chunk], max_chars=1_000_000).text)

    packed = pack_context([chunk], max_chars=full_size - 1)

    assert packed.records == []
    assert packed.text == ""
    assert packed.manifest.included == []
    assert packed.manifest.skipped == ["big"]
    assert packed.manifest.used_chars == 0


def test_duplicate_chunk_id_is_packed_once():
    packed = pack_context(
        [
            make_chunk("c1", text="first version"),
            make_chunk("c1", text="second version"),
        ]
    )

    assert packed.manifest.included == ["c1"]
    assert packed.manifest.duplicates == ["c1"]
    assert packed.manifest.skipped == []
    assert len(packed.records) == 1
    assert packed.records[0]["text"] == "first version"
    assert "second version" not in packed.text


def test_accepts_retrieval_hits_nested_mappings_and_flat_chunks():
    object_hit = RetrievalHit(
        chunk_id="object-hit", chunk=make_chunk("object-hit", text="from object")
    )
    nested_hit = {
        "chunk_id": "nested-hit",
        "chunk": make_chunk("nested-hit", text="from nested mapping"),
    }
    flat_hit = make_chunk("flat-hit", text="from flat mapping")

    packed = pack_context([object_hit, nested_hit, flat_hit])

    assert packed.chunk_ids == ["object-hit", "nested-hit", "flat-hit"]
    assert packed.manifest.included == ["object-hit", "nested-hit", "flat-hit"]
    assert packed.records[0]["text"] == "from object"
    assert packed.records[1]["text"] == "from nested mapping"


def test_metadata_fallback_supplies_missing_chunk_fields():
    hit = {
        "chunk_id": "meta",
        "metadata": {
            "doc_id": "sber_2020",
            "year": 2020,
            "text": "Metadata text",
            "section_path": "Риски",
        },
    }

    packed = pack_context([hit])

    record = packed.records[0]
    assert record["doc_id"] == "sber_2020"
    assert record["year"] == 2020
    assert record["text"] == "Metadata text"
    assert record["section_path"] == "Риски"


def test_zero_budget_includes_nothing_and_invalid_budget_is_rejected():
    packed = pack_context([make_chunk("c1")], max_chars=0)
    assert packed.text == ""
    assert packed.manifest.included == []
    assert packed.manifest.skipped == ["c1"]

    with pytest.raises(ValueError):
        pack_context([], max_chars=-1)
    with pytest.raises(ValueError):
        pack_context([], max_chars=True)
