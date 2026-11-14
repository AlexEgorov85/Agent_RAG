"""CLI парсера PDF (задача 5.1 change'а `add-pdf-parser`).

Запуск:
    python -m src.ingestion --pdf "data/raw/Копия Сбер 2023.pdf"
    python -m src.ingestion.pdf_parser --all
    python -m src.ingestion --stats
    python -m src.ingestion --pdf "<файл>" --fast-preview
    python -m src.ingestion --smoke
    python -m src.ingestion --sample [--sample-pages 10] [--doc 2023]

Приоритет: CLI-флаги выше env-конфигурации (см. спеку Requirement
«CLI и режимы запуска»), например ``--fast-preview`` принудительно включает
стратегию ``fast`` и выключает OCR.
"""

from __future__ import annotations

import argparse
import json
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
    modes.add_argument("--structure", action="store_true",
                       help="Построить дерево разделов из data/elements/*_elements.json "
                            "(шаг 2 плана, артефакт *_sections.json)")
    modes.add_argument("--chunks", action="store_true",
                       help="Собрать чанки К2 с breadcrumbs из артефактов "
                            "(шаг 2.5, артефакт *_chunks.json)")
    modes.add_argument("--sample", action="store_true",
                       help="Быстрая межгодовая выборка: TOC + страницы по телу "
                            "документа, артефакты в data/samples (без полного парсинга)")
    parser.add_argument("--fast-preview", action="store_true",
                        help="Режим превью: стратегия fast, без OCR и без записи артефактов")
    parser.add_argument("--pages", type=int, default=3, metavar="N",
                        help="Число страниц для --smoke (по умолчанию 3)")
    parser.add_argument("--sample-pages", type=int, default=14, metavar="N",
                        help="Сколько страниц брать на документ в --sample "
                             "(по умолчанию 14: 10 первых + 4 по телу)")
    parser.add_argument("--doc", type=str, default=None, metavar="NAME",
                        help="Подстрока имени для --smoke/--sample (PDF) и "
                             "--structure/--chunks (doc_id)")
    parser.add_argument("--batch", type=int, default=None, metavar="N",
                        help="Переопределить размер батча страниц (CLI выше env)")
    parser.add_argument("--force", action="store_true",
                        help="Перепарсить файл(ы) даже при актуальном артефакте "
                             "(старые артефакты уходят в data/elements/backup/)")
    parser.add_argument("--no-image-payload", action="store_true",
                        help="Не класть base64 картинок в *_elements.json "
                             "(файлы data/figures/{doc_id} и image_path сохраняются)")
    parser.add_argument("--chunk-size", type=int, default=800, metavar="N",
                        help="Размер текстового чанка для --chunks (по умолчанию 800)")
    parser.add_argument("--chunk-overlap", type=int, default=100, metavar="N",
                        help="Перекрытие чанков для --chunks (по умолчанию 100)")
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


def _pdf_by_doc(settings: ParserSettings) -> dict:
    """Карта ``doc_id → путь PDF`` по файлам ``data/raw`` (битые имена пропускаются)."""
    from src.ingestion.pdf_parser import make_doc_id

    mapping = {}
    for pdf in settings.data_raw_dir.glob("*.pdf"):
        try:
            mapping[make_doc_id(pdf.name)] = pdf
        except ValueError:
            continue
    return mapping


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
    if args.no_image_payload:
        updates["store_image_payload"] = False
    if args.sample:
        # Выборка нужна для быстрого ревью: base64 не пишем, image_path хватает.
        updates["store_image_payload"] = False
    if updates:
        settings = settings.model_copy(update=updates)

    if args.sample:
        from src.ingestion import pdf_outline, structure_builder

        pdfs = sorted(settings.data_raw_dir.glob("*.pdf"))
        if args.doc:
            pdfs = [p for p in pdfs if args.doc in p.name]
        if not pdfs:
            print(f"[sample] нет PDF в {settings.data_raw_dir}")
            return 1
        summary = []
        for pdf_path in pdfs:
            result = pdf_parser.parse_sample(
                pdf_path, settings=settings, total_pages=args.sample_pages)
            outline = pdf_outline.read_outline(pdf_path)
            toc = structure_builder.parse_toc_entries(result.elements)
            structure = structure_builder.build_document_structure(
                result.elements, toc=toc, outline=outline)
            structure_builder.save_sections(structure, settings.data_samples_dir)
            elements = result.elements
            tables = [e for e in elements if e.type == "Table"]
            images = [e for e in elements if e.type == "Image"]
            row = {
                "doc_id": result.stats["doc_id"],
                "page_ranges": result.stats["sample_page_ranges"],
                "n_elements": len(elements),
                "by_type": result.stats["by_type"],
                "tables": len(tables),
                "tables_with_html": sum(1 for e in tables if e.html),
                "images": len(images),
                "images_with_path": sum(1 for e in images if e.image_path),
                "images_with_bbox": sum(1 for e in images if e.bbox),
                "toc_entries": len(toc),
                "outline_entries": len(outline),
                "sections": len(structure.sections),
            }
            summary.append(row)
            levels = "outline" if outline else ("TOC" if toc else "эвристика")
            print(f"[sample] {row['doc_id']}: стр. {', '.join(row['page_ranges'])} | "
                  f"элементов {row['n_elements']} | Title {row['by_type'].get('Title', 0)} | "
                  f"Table {row['tables']} (html {row['tables_with_html']}) | "
                  f"Image {row['images']} (path {row['images_with_path']}, "
                  f"bbox {row['images_with_bbox']}) | TOC {row['toc_entries']} | "
                  f"закладок {row['outline_entries']} | разделов {row['sections']} ({levels})")
            for section in structure.sections[:5]:
                print("    " + " > ".join(section.breadcrumbs))
        target = settings.data_samples_dir / "sample_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"[sample] сводка -> {target}")
        return 0

    if args.stats:
        print(pdf_parser.format_stats_table(settings=settings))
        return 0

    if args.structure:
        from src.ingestion import pdf_outline, structure_builder

        files = sorted(settings.data_elements_dir.glob("*_elements.json"))
        if args.doc:
            files = [p for p in files if args.doc in p.stem]
        if not files:
            print(f"[structure] нет артефактов в {settings.data_elements_dir}")
            return 1
        pdf_by_doc = _pdf_by_doc(settings)
        for path in files:
            doc_id = path.stem.replace("_elements", "")
            pdf_path = pdf_by_doc.get(doc_id)
            outline, links = [], []
            if pdf_path is not None:
                outline = pdf_outline.read_outline(pdf_path)
                links = pdf_outline.read_links(pdf_path)
                if outline:
                    pdf_outline.save_outline(
                        outline, doc_id, settings.data_structure_dir)
                if links:
                    pdf_outline.save_links(
                        links, doc_id, settings.data_structure_dir)
            structure = structure_builder.build_from_elements_file(
                path, pdf_path=pdf_path, outline=outline)
            target = structure_builder.save_sections(
                structure, settings.data_structure_dir)
            source = "outline" if outline else "toc/эвристика"
            print(f"[structure] {doc_id}: разделов {len(structure.sections)}, "
                  f"уровни из {source}, закладок {len(outline)}, "
                  f"ссылок {len(links)} -> {target}")
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

    if args.chunks:
        from src.ingestion import chunk_enricher, pdf_to_chunks, structure_builder

        files = sorted(settings.data_elements_dir.glob("*_elements.json"))
        if args.doc:
            files = [p for p in files if args.doc in p.stem]
        if not files:
            print(f"[chunks] нет артефактов в {settings.data_elements_dir}")
            return 1
        pdf_by_doc = _pdf_by_doc(settings)
        for path in files:
            elements = pdf_to_chunks.load_elements(path)
            doc_id = elements[0].doc_id
            pdf_path = pdf_by_doc.get(doc_id)
            if pdf_path is not None:
                source = pdf_to_chunks.source_meta_from_pdf(pdf_path)
            else:
                year = int(doc_id.rsplit("_", 1)[-1])
                source = {"doc_id": doc_id, "title": doc_id, "year": year,
                          "report_period": str(year), "url": None,
                          "sha256": None}
            structure_path = (settings.data_structure_dir
                              / f"{doc_id}_sections.json")
            if structure_path.is_file():
                structure = structure_builder.load_sections(structure_path)
            else:
                structure = structure_builder.build_from_elements_file(
                    path, pdf_path=pdf_path)
                structure_builder.save_sections(
                    structure, settings.data_structure_dir)
            chunks = pdf_to_chunks.build_chunks(
                elements, source,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap)
            chunks = chunk_enricher.enrich_chunks(chunks, structure)
            target = pdf_to_chunks.save_chunks(
                chunks, doc_id, settings.data_chunks_dir)
            with_section = sum(1 for c in chunks if "section_path" in c)
            print(f"[chunks] {doc_id}: чанков {len(chunks)}, "
                  f"с разделом {with_section} -> {target}")
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
        result = pdf_parser.parse_pdf(args.pdf, settings=settings,
                                      force=args.force)
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
        results = pdf_parser.parse_all(settings.data_raw_dir, settings=settings,
                                       force=args.force)
        print(pdf_parser.format_stats_table(settings=settings))
        print(f"[all] спарсено файлов: {len(results)}")
        return 0

    build_arg_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
