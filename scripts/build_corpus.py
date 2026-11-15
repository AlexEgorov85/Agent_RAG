# -*- coding: utf-8 -*-
"""Полная сборка корпуса: PDF → элементы → структура → чанки → статус.

Скрипт повторяет шаги ingestion-CLI, но добавляет обязательную проверку
реестра источников до сборки, выбор документов и отчёт
``data/corpus_build_status.md`` с фактическими счётчиками по каждому
``doc_id``. Это точка воспроизводимой сборки корпуса (``spec_target.md`` §3.2).

Запуск::

    python scripts/build_corpus.py                 # всё: парсинг + структура + чанки
    python scripts/build_corpus.py --skip-parse    # только structure/chunks
    python scripts/build_corpus.py --doc sber_2023 # один документ
    python scripts/build_corpus.py --status-only   # пересобрать статус-отчёт

Парсинг ``hi_res`` долгий (~5.9 ч на 2581 стр.), поэтому запускайте его в
фоновом режиме. Скрипт можно безопасно прерывать: resume парсера пропускает
уже готовые артефакты с совпадающими SHA-256 и подписью парсинга.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.core.config import ParserSettings, load_parser_settings  # noqa: E402
from src.core.source_registry import (  # noqa: E402
    SourceRecord,
    load_registry,
    registry_fingerprint,
    validate_registry,
)
from src.ingestion import (  # noqa: E402
    chunk_enricher,
    pdf_outline,
    pdf_parser,
    pdf_to_chunks,
    structure_builder,
)


# ---------------------------------------------------------------------------
# Статус сборки
# ---------------------------------------------------------------------------

@dataclass
class DocBuildStatus:
    """Фактические счётчики по одному документу."""

    doc_id: str
    year: int | None = None
    issuer: str | None = None
    source_sha256: str | None = None
    n_pages: int | None = None
    n_elements: int = 0
    n_titles: int = 0
    n_tables: int = 0
    n_images: int = 0
    n_sections: int = 0
    n_chunks: int = 0
    n_chunks_with_section: int = 0
    artifacts: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    """Прочитать JSON-артефакт (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))


def collect_status(
    root: Path,
    settings: ParserSettings,
    records: Sequence[SourceRecord] | None = None,
) -> list[DocBuildStatus]:
    """Собрать фактический статус артефактов по реестру источников.

    Args:
        root: Корень проекта.
        settings: Настройки парсера (каталоги артефактов).
        records: Записи реестра; ``None`` — прочитать ``data/source_registry``.

    Returns:
        Статусы по каждому ``doc_id`` в порядке реестра.
    """
    if records is None:
        records = load_registry(root / "data" / "source_registry.json")
    registry_by_doc = {record.doc_id: record for record in records}
    statuses: list[DocBuildStatus] = []

    elements_dir = root / settings.data_elements_dir
    structure_dir = root / settings.data_structure_dir
    chunks_dir = root / settings.data_chunks_dir

    doc_ids = list(registry_by_doc)
    for doc_id in doc_ids:
        record = registry_by_doc[doc_id]
        status = DocBuildStatus(
            doc_id=doc_id,
            year=record.year,
            issuer=record.issuer,
            source_sha256=record.source_sha256,
        )
        artifacts = {
            "elements": elements_dir / f"{doc_id}_elements.json",
            "stats": elements_dir / f"{doc_id}_stats.json",
            "sections": structure_dir / f"{doc_id}_sections.json",
            "chunks": chunks_dir / f"{doc_id}_chunks.json",
        }
        status.artifacts = {kind: path.is_file()
                            for kind, path in artifacts.items()}

        stats_path = artifacts["stats"]
        if stats_path.is_file():
            try:
                stats = _read_json(stats_path)
                status.n_pages = stats.get("n_pages")
                status.n_elements = int(stats.get("n_elements") or 0)
                by_type = stats.get("by_type") or {}
                status.n_titles = int(by_type.get("Title", 0))
                status.n_tables = int(by_type.get("Table", 0))
                status.n_images = int(by_type.get("Image", 0))
                if stats.get("source_sha256") and (
                        str(stats["source_sha256"]).lower()
                        != record.source_sha256.lower()):
                    status.notes.append("stats.sha256 ≠ реестр")
            except (OSError, ValueError, TypeError) as exc:  # noqa: BLE001
                status.notes.append(f"stats не читается: {exc}")

        sections_path = artifacts["sections"]
        if sections_path.is_file():
            try:
                payload = _read_json(sections_path)
                status.n_sections = len(payload.get("sections") or [])
            except (OSError, ValueError) as exc:  # noqa: BLE001
                status.notes.append(f"sections не читаются: {exc}")

        chunks_path = artifacts["chunks"]
        if chunks_path.is_file():
            try:
                chunks = _read_json(chunks_path)
                status.n_chunks = len(chunks)
                status.n_chunks_with_section = sum(
                    1 for chunk in chunks if chunk.get("section_path"))
            except (OSError, ValueError) as exc:  # noqa: BLE001
                status.notes.append(f"chunks не читаются: {exc}")

        statuses.append(status)
    return statuses


def format_status_markdown(
    statuses: Sequence[DocBuildStatus],
    *,
    root: Path,
    commands: Sequence[str],
    registry_path: Path,
    fingerprint: str | None,
) -> str:
    """Сформировать читаемый статус-отчёт сборки корпуса."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    complete = [status for status in statuses
                if status.artifacts.get("chunks") and status.n_chunks > 0]
    lines = [
        "# Статус сборки полного корпуса",
        "",
        f"- Обновлено: {now}",
        f"- Корень проекта: `{root}`",
        f"- Реестр: `{registry_path}`",
        f"- Fingerprint корпуса: `{fingerprint or '—'}`",
        f"- Документов в реестре: {len(statuses)}",
        f"- Документов с чанками: {len(complete)}",
        "",
        "## Команды сборки",
        "",
        "```bash",
        *commands,
        "```",
        "",
        "## Артефакты по документам",
        "",
        "| doc_id | год | issuer | стр. | элементов | Title | Table | Image |"
        " разделов | чанков | с разделом | elements | stats | sections |"
        " chunks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for status in statuses:
        artifacts = status.artifacts
        lines.append("| " + " | ".join([
            status.doc_id,
            str(status.year or "—"),
            status.issuer or "—",
            str(status.n_pages if status.n_pages is not None else "—"),
            str(status.n_elements),
            str(status.n_titles),
            str(status.n_tables),
            str(status.n_images),
            str(status.n_sections),
            str(status.n_chunks),
            str(status.n_chunks_with_section),
            "да" if artifacts.get("elements") else "НЕТ",
            "да" if artifacts.get("stats") else "НЕТ",
            "да" if artifacts.get("sections") else "НЕТ",
            "да" if artifacts.get("chunks") else "НЕТ",
        ]) + " |")

    missing = [status.doc_id for status in statuses
               if not status.artifacts.get("chunks") or status.n_chunks == 0]
    lines += ["", "## Пропуски", ""]
    if missing:
        lines.append("Без чанков: " + ", ".join(missing))
        lines += [
            "",
            "Полный graph index требует готовых чанков по всем документам. На "
            "текущем наборе его сборка будет частичной: артефакт получит "
            "`status=incomplete` и не будет активирован фабрикой. После сборки "
            "чанков запустите `scripts/build_full_graph.py`.",
        ]
    else:
        lines.append("Пропусков нет: чанки собраны по всем документам реестра.")

    lines += [
        "",
        "## Ограничения данных",
        "",
        "- Год 2021 физически отсутствует: записи `sber_2021` нет, "
        "интерполяция недопустима.",
        "- `sber_2015` — отчёт БПС-Сбербанка (Беларусь), помечен "
        "`incomparable_issuer`; по умолчанию не смешивается с ПАО Сбербанк.",
        "- Канонический парсинг 2015–2018 выполнен с `--no-image-payload`: "
        "base64 картинок не хранится, `image_path` и вырезы `data/figures` "
        "сохраняются. Документы 2019–2024 ещё не парсились.",
    ]

    noted = [status for status in statuses if status.notes]
    if noted:
        lines += ["", "## Замечания", ""]
        for status in noted:
            lines.append(f"- {status.doc_id}: " + "; ".join(status.notes))

    lines += [
        "",
        "## Как продолжить сборку",
        "",
        "```bash",
        ".venv\\Scripts\\python.exe scripts/build_corpus.py --skip-parse  "
        "# структура и чанки",
        ".venv\\Scripts\\python.exe scripts/verify_corpus.py  "
        "# полная валидация",
        ".venv\\Scripts\\python.exe scripts/build_full_graph.py  "
        "# граф по готовым чанкам",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_status(
    root: Path,
    settings: ParserSettings,
    *,
    commands: Sequence[str],
    records: Sequence[SourceRecord] | None = None,
    status_path: Path | None = None,
) -> Path:
    """Собрать и атомарно записать ``data/corpus_build_status.md``."""
    if records is None:
        records = load_registry(root / "data" / "source_registry.json")
    statuses = collect_status(root, settings, records)
    registry_path = root / "data" / "source_registry.json"
    text = format_status_markdown(
        statuses,
        root=root,
        commands=commands,
        registry_path=registry_path,
        fingerprint=registry_fingerprint(records) if records else None,
    )
    target = status_path or root / "data" / "corpus_build_status.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".md.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(target)
    return target


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------

def _pdf_by_doc(raw_dir: Path) -> dict[str, Path]:
    """Карта ``doc_id → PDF`` по именам файлов."""
    mapping: dict[str, Path] = {}
    for pdf in sorted(raw_dir.glob("*.pdf")):
        try:
            mapping[pdf_parser.make_doc_id(pdf.name)] = pdf
        except ValueError:
            continue
    return mapping


def _prepare_settings(args: argparse.Namespace,
                      settings: ParserSettings) -> ParserSettings:
    """Применить CLI-флаги поверх env-настроек (CLI приоритетнее)."""
    updates: dict[str, Any] = {}
    if args.strategy:
        updates["parse_strategy"] = args.strategy
    if args.batch is not None:
        updates["parse_page_batch"] = args.batch
    if not args.image_payload:
        updates["store_image_payload"] = False
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def _parse_documents(pdfs: Sequence[Path], settings: ParserSettings,
                     force: bool) -> list[str]:
    """Спарить документы; вернуть doc_id успешно обработанных файлов."""
    done: list[str] = []
    for pdf in pdfs:
        started = time.time()
        try:
            result = pdf_parser.parse_pdf(pdf, settings=settings, force=force)
        except Exception as exc:  # noqa: BLE001 - один файл не должен рушить сборку
            print(f"[parse] ОШИБКА {pdf.name}: {exc}", file=sys.stderr)
            continue
        doc_id = pdf_parser.make_doc_id(pdf.name)
        if result is None:
            print(f"[parse] {doc_id}: resume, артефакт актуален "
                  f"({time.time() - started:.1f}s)")
        else:
            stats = result.stats
            print(f"[parse] {doc_id}: страниц {stats['n_pages']}, "
                  f"элементов {stats['n_elements']}, "
                  f"{stats['parse_seconds']:.1f}s")
        done.append(doc_id)
    return done


def _build_structures(doc_ids: Sequence[str], pdf_by_doc: Mapping[str, Path],
                      settings: ParserSettings) -> None:
    """Построить ``*_sections.json`` из артефактов парсера."""
    for doc_id in doc_ids:
        elements_path = (settings.data_elements_dir
                         / f"{doc_id}_elements.json")
        if not elements_path.is_file():
            print(f"[structure] {doc_id}: нет {elements_path.name} — пропуск")
            continue
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
            elements_path, pdf_path=pdf_path, outline=outline)
        target = structure_builder.save_sections(
            structure, settings.data_structure_dir)
        source = "outline" if outline else "toc/эвристика"
        print(f"[structure] {doc_id}: разделов {len(structure.sections)}, "
              f"уровни из {source}, закладок {len(outline)} -> {target}")


def _build_chunks(doc_ids: Sequence[str], pdf_by_doc: Mapping[str, Path],
                  settings: ParserSettings, *, chunk_size: int,
                  chunk_overlap: int) -> None:
    """Построить ``*_chunks.json`` с breadcrumbs."""
    for doc_id in doc_ids:
        elements_path = (settings.data_elements_dir
                         / f"{doc_id}_elements.json")
        if not elements_path.is_file():
            print(f"[chunks] {doc_id}: нет {elements_path.name} — пропуск")
            continue
        elements = pdf_to_chunks.load_elements(elements_path)
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
                elements_path, pdf_path=pdf_path)
            structure_builder.save_sections(
                structure, settings.data_structure_dir)
        chunks = pdf_to_chunks.build_chunks(
            elements, source, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        chunks = chunk_enricher.enrich_chunks(chunks, structure)
        target = pdf_to_chunks.save_chunks(
            chunks, doc_id, settings.data_chunks_dir)
        with_section = sum(1 for chunk in chunks if "section_path" in chunk)
        print(f"[chunks] {doc_id}: чанков {len(chunks)}, "
              f"с разделом {with_section} -> {target}")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI полной сборки корпуса."""
    parser = argparse.ArgumentParser(
        description="Собрать полный корпус: элементы, структура, чанки.")
    parser.add_argument("--doc", action="append", default=None,
                        help="doc_id или год (можно несколько раз)")
    parser.add_argument("--skip-parse", action="store_true",
                        help="Не парсить PDF, использовать готовые elements")
    parser.add_argument("--skip-structure", action="store_true",
                        help="Не перестраивать дерево разделов")
    parser.add_argument("--skip-chunks", action="store_true",
                        help="Не пересобирать чанки")
    parser.add_argument("--status-only", action="store_true",
                        help="Только записать data/corpus_build_status.md")
    parser.add_argument("--force", action="store_true",
                        help="Перепарсить документы даже при актуальном артефакте")
    parser.add_argument("--strategy", choices=("hi_res", "fast"), default=None,
                        help="Стратегия парсинга (по умолчанию из env/hi_res)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Размер батча страниц")
    parser.add_argument("--image-payload", action="store_true",
                        help="Класть base64 картинок в elements (по умолчанию нет)")
    parser.add_argument("--chunk-size", type=int, default=pdf_to_chunks.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int,
                        default=pdf_to_chunks.CHUNK_OVERLAP)
    parser.add_argument("--status-path", type=Path, default=None,
                        help="Куда писать статус (data/corpus_build_status.md)")
    return parser


def _matches(doc_id: str, wanted: Sequence[str]) -> bool:
    """Проверить, выбран ли документ фильтром ``--doc``."""
    return any(item == doc_id or item == doc_id.rsplit("_", 1)[-1]
               or item in doc_id for item in wanted)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа сборки корпуса."""
    args = build_arg_parser().parse_args(argv)
    root = PROJECT
    settings = _prepare_settings(args, load_parser_settings())
    # Каталоги в настройках заданы относительно CWD; работаем от корня проекта.
    import os
    os.chdir(root)

    registry_path = root / "data" / "source_registry.json"
    try:
        records = load_registry(registry_path)
    except (OSError, ValueError) as exc:
        print(f"[registry] не читается: {exc}", file=sys.stderr)
        return 2
    validation = validate_registry(records, project_root=root)
    if not validation.is_valid:
        print("[registry] реестр невалиден — сборка остановлена:", file=sys.stderr)
        for issue in validation.errors:
            print(f"  ERROR {issue.code} [{issue.doc_id}]: {issue.message}",
                  file=sys.stderr)
        return 2

    commands = [
        ".venv\\Scripts\\python.exe scripts/build_corpus.py",
        ".venv\\Scripts\\python.exe -m src.ingestion --all --no-image-payload",
        ".venv\\Scripts\\python.exe -m src.ingestion --structure",
        ".venv\\Scripts\\python.exe -m src.ingestion --chunks",
        ".venv\\Scripts\\python.exe scripts/verify_corpus.py",
    ]
    if args.status_only:
        target = write_status(root, settings, commands=commands,
                              records=records, status_path=args.status_path)
        print(f"[status] {target}")
        return 0

    pdf_by_doc = _pdf_by_doc(root / "data" / "raw")
    selected = [record.doc_id for record in records
                if not args.doc or _matches(record.doc_id, args.doc)]
    unknown = [item for item in (args.doc or [])
               if not any(_matches(doc_id, [item]) for doc_id in selected)]
    if unknown:
        print(f"[build] неизвестные --doc: {', '.join(unknown)}",
              file=sys.stderr)
        return 2
    if not selected:
        print("[build] не выбрано ни одного документа", file=sys.stderr)
        return 2

    print(f"[build] документов {len(selected)}: {', '.join(selected)}")
    if not args.skip_parse:
        pdfs = [pdf_by_doc[doc_id] for doc_id in selected
                if doc_id in pdf_by_doc]
        missing_pdf = [doc_id for doc_id in selected
                       if doc_id not in pdf_by_doc]
        if missing_pdf:
            print("[build] нет PDF для: " + ", ".join(missing_pdf),
                  file=sys.stderr)
        _parse_documents(pdfs, settings, args.force)
    if not args.skip_structure:
        _build_structures(selected, pdf_by_doc, settings)
    if not args.skip_chunks:
        _build_chunks(selected, pdf_by_doc, settings,
                      chunk_size=args.chunk_size,
                      chunk_overlap=args.chunk_overlap)

    target = write_status(root, settings, commands=commands,
                          records=records, status_path=args.status_path)
    print(f"[status] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
