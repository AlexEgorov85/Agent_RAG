# -*- coding: utf-8 -*-
"""Собрать структурный индекс табличных значений (год → числа).

Источник — HTML таблиц из чанков; шапка с годами используется как есть.
Артефакт нужен, чтобы агент и проверка ответа опирались на значения,
привязанные к году, а не на строку чисел без подписей.

Запуск::

    python scripts/build_metrics.py
    python scripts/build_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.corpus import load_chunks  # noqa: E402
from src.core.metrics_table import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    build_index,
    save_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать индекс табличных значений")
    parser.add_argument("--chunks-dir", type=Path, default=PROJECT / "data" / "chunks")
    parser.add_argument("--out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks_dir)
    index = build_index(chunks)
    years = Counter()
    for year, values in index.values_by_year.items():
        years[year] = len(values)
    print(f"чанков: {len(chunks)} | табличных значений: {index.cell_count}")
    print(f"документов: {len(index.documents)} | годов: {len(years)}")
    print("значений по годам:", ", ".join(f"{year}: {count}" for year, count in sorted(years.items())))
    for metric, per_year in sorted(index.by_metric.items()):
        print(f"  {metric}: {len(per_year)} лет → " + ", ".join(
            f"{year}={cell['value']:g}" for year, cell in sorted(per_year.items())
        ))
    if args.dry_run:
        return 0
    path = save_index(index, args.out)
    size = path.stat().st_size / 1024
    print(f"Артефакт: {path.relative_to(PROJECT)} ({size:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
