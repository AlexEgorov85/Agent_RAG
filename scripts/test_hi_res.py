# -*- coding: utf-8 -*-
"""Проверочный запуск partition_pdf(strategy='hi_res') на первых страницах отчёта."""
from collections import Counter
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

PDF = Path("data/raw") / "Копия Сбер 2015.pdf"
FIGURES = Path("data/figures") / "sber_2015_smoke"  # вырезы не будут захватывать cwd
FIGURES.mkdir(parents=True, exist_ok=True)

elements = partition_pdf(
    filename=str(PDF),
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image"],
    pages=[1, 2, 3],
    languages=["rus", "eng"],
    extract_image_block_output_dir=str(FIGURES),
)

print(f"elementов извлечено: {len(elements)}")
for el_type, cnt in Counter(type(e).__name__ for e in elements).most_common():
    print(f"  {el_type}: {cnt}")

print("\nПримеры:")
for el in elements[:5]:
    text = (el.text or "")[:80].replace("\n", " ")
    print(f"  [{type(el).__name__}] {text}")
