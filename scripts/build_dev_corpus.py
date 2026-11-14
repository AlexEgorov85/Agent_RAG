# -*- coding: utf-8 -*-
"""Собрать dev-корпус из частичных артефактов (без полного парсинга).

Берёт `*_elements.json` из `data/samples_contig*`, строит структуру/чанки и
пишет в `data/dev/chunks/`. Это тестовый корпус для end-to-end проверки
retrieval/engine/UI; канонический `data/elements` не трогается.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.ingestion import chunk_enricher, pdf_outline, pdf_to_chunks, structure_builder

RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "dev" / "chunks"
INPUT_DIRS = [
    PROJECT / "data" / "samples_contig",
    PROJECT / "data" / "samples_contig_1617",
]


def pdf_for(doc_id: str) -> Path | None:
    """Найти исходный PDF по году из ``doc_id``."""
    year = doc_id.rsplit("_", 1)[-1]
    for pdf in RAW.glob("*.pdf"):
        if year in pdf.name:
            return pdf
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen: set[str] = set()
    for directory in INPUT_DIRS:
        if not directory.is_dir():
            print(f"[dev] пропуск {directory} (нет каталога)")
            continue
        for elements_json in sorted(directory.glob("*_elements.json")):
            doc_id = elements_json.stem.replace("_elements", "")
            if doc_id in seen:
                continue
            pdf_path = pdf_for(doc_id)
            if pdf_path is None:
                print(f"[dev] {doc_id}: PDF не найден — пропуск")
                continue
            elements = pdf_to_chunks.load_elements(elements_json)
            source = pdf_to_chunks.source_meta_from_pdf(pdf_path)
            outline = pdf_outline.read_outline(pdf_path)
            structure = structure_builder.build_from_elements_file(
                elements_json, pdf_path=pdf_path, outline=outline)
            chunks = pdf_to_chunks.build_chunks(elements, source)
            chunks = chunk_enricher.enrich_chunks(chunks, structure)
            target = pdf_to_chunks.save_chunks(chunks, doc_id, OUT)
            with_section = sum(1 for chunk in chunks if "section_path" in chunk)
            pages = sorted({int(chunk["page_pdf"]) for chunk in chunks})
            manifest.append({
                "doc_id": doc_id,
                "source_pages": [min(pages), max(pages)] if pages else [],
                "n_pages_present": len(pages),
                "n_elements": len(elements),
                "n_chunks": len(chunks),
                "n_with_section": with_section,
                "file": str(target.relative_to(PROJECT)),
            })
            seen.add(doc_id)
            print(f"[dev] {doc_id}: элементов {len(elements)}, чанков {len(chunks)}, "
                  f"с разделом {with_section}, стр. {min(pages)}-{max(pages)} -> {target}")

    manifest_path = OUT.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"[dev] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
