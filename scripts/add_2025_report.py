# -*- coding: utf-8 -*-
"""Добавить официальный годовой отчёт 2025 в реестр источников."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT / "data" / "source_registry.json"
PDF = PROJECT / "data" / "raw" / "Копия Сбер 2025.pdf"


def main() -> int:
    if not PDF.is_file():
        raise SystemExit(f"Нет файла: {PDF}")
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = [
        item for item in payload.get("sources", [])
        if item.get("doc_id") != "sber_2025"
    ]
    sources.append(
        {
            "doc_id": "sber_2025",
            "file_path": "data/raw/Копия Сбер 2025.pdf",
            "source_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
            "year": 2025,
            "report_period": "2025",
            "issuer": "ПАО Сбербанк",
            "country": "Россия",
            "source_status": "comparable",
            "inclusion_note": (
                "Годовой отчёт ПАО Сбербанк за 2025 год; официальная "
                "публикация 29.05.2026."
            ),
        }
    )
    payload["sources"] = sources
    notes = str(payload.get("notes", ""))
    notes = notes.replace("9 отчётов", "10 отчётов")
    notes = notes.replace("2015–2020, 2022–2024", "2015–2020, 2022–2025")
    payload["notes"] = notes
    REGISTRY.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"added sber_2025, sources={len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
