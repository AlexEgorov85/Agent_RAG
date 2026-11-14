"""CLI парсера PDF (задача 5.1 change'а `add-pdf-parser`).

Запуск:
    python -m src.ingestion --pdf "data/raw/Копия Сбер 2023.pdf"
    python -m src.ingestion.pdf_parser --all
    python -m src.ingestion --stats
    python -m src.ingestion --pdf "<файл>" --fast-preview
    python -m src.ingestion --smoke

Приоритет: CLI-флаги выше env-конфигурации (см. спеку Requirement
«CLI и режимы запуска»), например ``--fast-preview`` принудительно включает
стратегию ``fast`` и выключает OCR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.core.config import ParserSettings, load_parser_settings
from src.ingestion import pdf_parser

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Собрать argparse-парсер с взаимно исключающими режимами."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.pdf_parser",
        description="PDF-парсер годовых отчётов Сбербанка (structure-aware Graph RAG).",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--pdf", metavar="PATH", type=Path,
                       help="Спарсить один PDF-файл")
    modes.add_argument("--all", action="store_true",
                       help="Спарсить все PDF из data/raw")
    modes.add_argument("--stats", action="store_true",
                       help="Печатать сводную таблицу по спарсенным артефактам")
    modes.add_argument("--smoke", action="store_true",
                       help="Smoke-проверка: первые N страниц первого PDF, без артефактов")
    parser.add_argument("--fast-preview", action="store_true",
                        help="Режим превью: стратегия fast, без OCR и без записи артефактов")
    parser.add_argument("--pages", type=int, default=3, metavar="N",
                        help="Число страниц для --smoke (по умолчанию 3)")
    parser.add_argument("--doc", type=str, default=None, metavar="NAME",
                        help="Имя файла (подстрока) для --smoke; по умолчанию первый PDF")
    parser.add_argument("--batch", type=int, default=None, metavar="N",
                        help="Переопределить размер батча страниц (CLI выше env)")
    return parser


def _pick_pdf(settings: ParserSettings, name: Optional[str]) -> Path:
    """Выбрать PDF в data/raw: по подстроке имени или первый по алфавиту."""
    pdfs = sorted(settings.data_raw_dir.glob("*.pdf"))
    if name is not None:
        pdfs = [p for p in pdfs if name in p.name]
    if not pdfs:
        raise FileNotFoundError(
            f"PDF не найден в {settings.data_raw_dir}"
            + (f" (фильтр: {name!r})" if name else ""))
    return pdfs[0]


def main(argv: Optional[Sequence[str]] = None,
         settings: Optional[ParserSettings] = None) -> int:
    """Точка входа CLI.

    Args:
        argv: Аргументы командной строки (None — ``sys.argv[1:]``).
        settings: Готовая конфигурация (None — из env/.env); параметр для тестов.

    Returns:
        Код возврата процесса (0 — успех).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = build_arg_parser().parse_args(argv)
    settings = settings or load_parser_settings()

    # --- CLI-флаги имеют приоритет над env -------------------------------
    updates: dict = {}
    if args.fast_preview:
        updates["parse_strategy"] = "fast"
    if args.batch is not None:
        updates["parse_page_batch"] = args.batch
    if updates:
        settings = settings.model_copy(update=updates)

    if args.stats:
        print(pdf_parser.format_stats_table(settings=settings))
        return 0

    if args.smoke:
        pdf_path = _pick_pdf(settings, args.doc)
        n_pages = min(args.pages, pdf_parser.count_pdf_pages(pdf_path))
        by_type = pdf_parser.preview_pdf(
            pdf_path, settings=settings, n_pages=n_pages)
        print(f"[smoke] {pdf_path.name}: первые {n_pages} стр., "
              f"стратегия {settings.parse_strategy}")
        for cat in sorted(by_type, key=by_type.get, reverse=True):
            print(f"  {cat}: {by_type[cat]}")
        print(f"[smoke] OK: {sum(by_type.values())} элементов (артефакты не созданы)")
        return 0

    if args.pdf is not None:
        if args.fast_preview:
            pdf_path = args.pdf
            by_type = pdf_parser.preview_pdf(
                pdf_path, settings=settings,
                n_pages=pdf_parser.count_pdf_pages(pdf_path))
            print(f"[fast-preview] {pdf_path.name}: "
                  f"стратегия fast, OCR не задействован")
            for cat in sorted(by_type, key=by_type.get, reverse=True):
                print(f"  {cat}: {by_type[cat]}")
            return 0
        result = pdf_parser.parse_pdf(args.pdf, settings=settings)
        if result is None:
            print(f"[resume] {args.pdf.name} уже спарсен — пропуск")
            return 0
        st = result.stats
        print(f"[done] {st['doc_id']}: страниц {st['n_pages']}, "
              f"элементов {st['n_elements']}, "
              f"время {st['parse_seconds']:.1f}s")
        print(f"       артефакты: {settings.data_elements_dir}")
        return 0

    if args.all:
        results = pdf_parser.parse_all(settings.data_raw_dir, settings=settings)
        print(pdf_parser.format_stats_table(settings=settings))
        print(f"[all] спарсено файлов: {len(results)}")
        return 0

    build_arg_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
