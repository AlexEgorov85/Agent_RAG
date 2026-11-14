"""Загрузка чанков из каталога артефактов (dev/канонический корпус)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """Прочитать все ``*_chunks.json`` из каталога и отсортировать."""
    chunks: List[Dict[str, Any]] = []
    for path in sorted(Path(chunks_dir).glob("*_chunks.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            chunks.extend(data)
    chunks.sort(key=lambda chunk: (chunk.get("doc_id", ""),
                                   int(chunk.get("page_pdf") or 0),
                                   chunk.get("chunk_id", "")))
    return chunks
