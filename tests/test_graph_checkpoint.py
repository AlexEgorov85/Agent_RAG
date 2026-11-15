"""Offline-тесты checkpoint/resume для сборки graph-индекса."""

from __future__ import annotations

from pathlib import Path

from scripts.build_full_graph import (
    _append_checkpoint,
    _batch_key,
    _load_checkpoint,
)


def _batch(*chunk_ids: str) -> list[dict[str, str]]:
    return [{"chunk_id": chunk_id, "text": "text"} for chunk_id in chunk_ids]


def test_checkpoint_roundtrip_and_resume_only_successful_batches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.jsonl"
    batch = _batch("a", "b")
    key = _batch_key(batch)

    _append_checkpoint(
        path,
        {
            "batch": key,
            "model": "model-1",
            "entities": [
                {
                    "name": "Сбербанк",
                    "entity_type": "Organization",
                    "chunk_id": "a",
                    "aliases": ["Сбер"],
                }
            ],
            "relations": [],
            "skipped": [],
            "calls": 1,
        },
    )
    _append_checkpoint(
        path,
        {
            "batch": "failed-batch",
            "model": "model-1",
            "failed": ["c"],
            "entities": [],
            "relations": [],
            "skipped": [],
            "calls": 1,
        },
    )

    loaded = _load_checkpoint(path, model="model-1")

    assert set(loaded) == {key}
    assert loaded[key]["entities"][0]["name"] == "Сбербанк"


def test_checkpoint_ignores_other_models_and_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    _append_checkpoint(
        path,
        {
            "batch": "a",
            "model": "other-model",
            "entities": [],
            "relations": [],
            "skipped": [],
            "calls": 1,
        },
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "{not json}\n",
        encoding="utf-8",
    )

    assert _load_checkpoint(path, model="model-1") == {}


def test_missing_checkpoint_is_empty_and_batch_key_is_stable() -> None:
    assert _load_checkpoint(None, model="model-1") == {}
    assert _batch_key(_batch("a", "b")) == _batch_key(_batch("a", "b"))
    assert _batch_key(_batch("a", "b")) != _batch_key(_batch("b", "a"))
