# -*- coding: utf-8 -*-
"""Проверка реального парсинга на 3 страницах отчёта 2015 (без записи в data/elements)."""
import json
import shutil
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from src.core.config import ParserSettings
from src.ingestion.pdf_parser import parse_pdf

RAW = Path("data/raw/Копия Сбер 2015.pdf")
TMP = Path("tmp_verify_artifacts")
shutil.rmtree(TMP, ignore_errors=True)  # иначе resume верно пропустит спарсенный
TMP.mkdir(exist_ok=True)
sample = TMP / "Копия Сбер 2015.pdf"
writer = PdfWriter()
reader = PdfReader(str(RAW))
for p in reader.pages[:3]:
    writer.add_page(p)
with open(sample, "wb") as fh:
    writer.write(fh)

settings = ParserSettings(parse_strategy="hi_res",
                          data_elements_dir=TMP / "elements",
                          data_figures_dir=TMP / "figures")
result = parse_pdf(sample, settings=settings)
assert result is not None, "resume ложно сработал"

els_path = TMP / "elements" / "sber_2015_elements.json"
st_path = TMP / "elements" / "sber_2015_stats.json"
stats = json.loads(st_path.read_text(encoding="utf-8"))
elements = json.loads(els_path.read_text(encoding="utf-8"), )

print("--- stats ---")
for k, v in stats.items():
    if k != "by_type":
        print(f"{k}: {v}")
print("by_type:", stats["by_type"])
assert sum(stats["by_type"].values()) == stats["n_elements"] == len(elements)
assert stats["skipped_page_ranges"] == []
assert all(1 <= e["page"] <= 3 for e in elements)
assert all(str(e["page"]) == (e["page_label"] or str(e["page"])) for e in elements)
titles = [e["text"] for e in elements if e["type"] == "Title"]
tables = [e for e in elements if e["type"] == "Table"]
print("\nПервые заголовки:", titles[:3])
print("Таблиц с HTML:", sum(1 for t in tables if t["html"]), "из", len(tables))
imgs_with_b64 = sum(1 for e in elements if e["type"] == "Image" and e["image_b64"])
imgs_small = sum(1 for e in elements if e["type"] == "Image" and not e["image_b64"])
print(f"Картинки: {imgs_with_b64} с base64, {imgs_small} отфильтровано (<200x200)")
print("\nOK: артефакт корректен, контракт by_type/page/page_label сходится")
