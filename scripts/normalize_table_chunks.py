# -*- coding: utf-8 -*-
"""Пересобрать текстовое представление табличных чанков из их HTML.

Парсер сохраняет таблицу как HTML, а ``text`` иногда теряет часть ячеек:
числа остаются только в разметке. Из-за этого цитаты берутся из HTML
(``</td></tr>``), ломается поиск по числам и проверка цитат.

Скрипт перезаписывает ``text`` табличных чанков: сохраняет подпись перед
таблицей и добавляет ячейки построчно в том порядке, в котором они стоят в
HTML. Идентификаторы, годы и разметка не меняются.

Запуск::

    python scripts/normalize_table_chunks.py --dry-run
    python scripts/normalize_table_chunks.py
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

TABLE_MARKER = "<table"


class _TextCollector(HTMLParser):
    """Плоский текст таблицы: строки через перевод строки, ячейки через « | »."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag == "br" and self._row is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:  # noqa: ANN001
        if tag in {"td", "th"} and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = []
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._row is not None:
            self._cell.append(data)


def flatten_table(html: str) -> str:
    collector = _TextCollector()
    collector.feed(html)
    lines = [
        " | ".join(cell for cell in row if cell)
        for row in collector.rows
    ]
    return "\n".join(line for line in lines if line)


def rebuild_text(chunk: dict) -> str | None:
    """Новый текст чанка или ``None``, если пересобирать нечего."""
    if chunk.get("kind") != "table":
        return None
    html = chunk.get("html") or ""
    if TABLE_MARKER not in html:
        return None
    table_text = flatten_table(html)
    if not table_text.strip():
        return None
    current = chunk.get("text") or ""
    preamble = current.split(TABLE_MARKER, 1)[0].strip()
    return f"{preamble}\n{table_text}".strip() if preamble else table_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать text таблиц из HTML")
    parser.add_argument("--chunks-dir", type=Path, default=PROJECT / "data" / "chunks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    changed = 0
    total = 0
    grown = 0
    for path in sorted(args.chunks_dir.glob("*_chunks.json")):
        chunks = json.loads(path.read_text(encoding="utf-8"))
        file_changed = False
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            total += 1
            new_text = rebuild_text(chunk)
            if new_text is None or new_text == chunk.get("text"):
                continue
            if len(new_text) > len(chunk.get("text") or ""):
                grown += 1
            chunk["text"] = new_text
            changed += 1
            file_changed = True
        if file_changed and not args.dry_run:
            path.write_text(
                json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
            )
        print(f"{path.name}: {'изменён' if file_changed else 'без изменений'}")

    print(f"чанков: {total} | таблиц пересобрано: {changed} | text вырос: {grown}")
    if args.dry_run:
        print("dry-run: файлы не перезаписаны")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
