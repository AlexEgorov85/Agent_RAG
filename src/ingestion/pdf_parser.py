"""PDF-парсер годовых отчётов Сбербанка (OpenSpec change `add-pdf-parser`).

Обёртка над ``unstructured.partition_pdf``: batчевое извлечение элементов стратегии
``hi_res`` с OCR (rus+eng), resume по ``source_sha256``, фильтр мелких изображений.
Изолирует Unstructured от остального кода проекта — потребители читают
``data/elements/{doc_id}_elements.json``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.models import ParseResult, RawElement
from src.core.config import ParserSettings, load_parser_settings

logger = logging.getLogger(__name__)

#: Регулярка для извлечения года из имени файла («Копия Сбер 2015.pdf» -> 2015)
_YEAR_RE = re.compile(r"\d{4}")

#: Пять знаков на индекс в element_id: sber_2015:00000
_ELEMENT_INDEX_WIDTH = 5

#: Понятная пользователю ошибка установки tesseract
_TESSERACT_INSTALL_HINT = (
    "tesseract не найден. Установите UB-Mannheim Tesseract OCR: "
    "winget install --id UB-Mannheim.TesseractOCR -e "
    "или choco: choco install tesseract; затем добавьте в PATH."
)


class ParserEnvironmentError(RuntimeError):
    """Среда OCR не соответствует требованиям (нет бинарника/языковой модели)."""


def make_doc_id(filename: str) -> str:
    """Вычислить ``doc_id`` из имени файла.

    Год — последняя 4-цифровая группа в имени файла (заданная спекой: не из
    метаданных PDF, а из имени файла). ``sber_{year}``.

    Args:
        filename: Имя файла или путь к нему.

    Returns:
        Идентификатор документа, например ``sber_2015``.

    Raises:
        ValueError: Если в имени файла нет 4-цифрового года.
    """
    stem = Path(filename).name
    years = _YEAR_RE.findall(stem)
    if not years:
        raise ValueError(f"В имени файла {filename!r} нет 4-цифрового года")
    return f"sber_{years[-1]}"


def element_id(doc_id: str, index: int) -> str:
    """Сформировать стабильный ``element_id``: ``sber_2015:00000``."""
    return f"{doc_id}:{index:0{_ELEMENT_INDEX_WIDTH}d}"


def compute_source_sha256(pdf_path: Path, buffer_size: int = 1 << 20) -> str:
    """Посчитать SHA-256 файла PDF в потоковом режиме (без полной загрузки в память)."""
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        while True:
            chunk = fh.read(buffer_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_pdf_pages(pdf_path: Path) -> int:
    """Число страниц PDF через pypdf (используется в статистике артефакта)."""
    from pypdf import PdfReader  # локальный импорт — унифицирует накладные расходы

    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def page_labels_of(pdf_path: Path) -> List[str]:
    """Метка страницы с печати, как на заданных им стр. в PDF (переиспользовано из курса).

    Курсовой код (Занятие 2, ``page_element_records``) хранит у чанка ``page_label``
    из ``PdfReader.page_labels`` — она нужна фильтрам поиска по «напечатанной» странице.
    Если метки нет или их число не совпадает с числом страниц — пустой список.
    """
    from pypdf import PdfReader

    try:
        labels = list(PdfReader(str(pdf_path)).page_labels)
    except Exception as exc:  # noqa: BLE001 — битые метки не должны ронять парс
        logger.warning("page_labels недоступны для %s: %s", pdf_path, exc)
        return []
    return labels if len(labels) == count_pdf_pages(pdf_path) else []


def preflight_check_ocr(
    languages: Sequence[str],
    *,
    tesseract_dir: Optional[str] = None,
    tessdata_prefix: Optional[Path] = None,
) -> Optional[str]:
    """Проверить среду выполнения OCR до вызова Unstructured.

    Ожидается: tesseract в PATH (или ``tesseract_dir``, добавлен в PATH потребителем)
    и файлы ``{lang}.traineddata`` в ``TESSDATA_PREFIX``.

    Args:
        languages: Запрошенные OCR-языки (например, ("rus", "eng")).
        tesseract_dir: Опциональный явный каталог tesseract вместо PATH.
        tessdata_prefix: Явный каталог traineddata (иначе из env или по умолчанию
            распознаётся tesseract'ом сам).

    Returns:
        Эквивалентный ``TESSDATA_PREFIX`` (путь к tessdata), если задан явно
        и найдены все языковые файлы; иначе None.

    Raises:
        ParserEnvironmentError: tesseract не найден или отсутствует traineddata.
    """
    tesseract_exe = Path(tesseract_dir) / "tesseract.exe" if tesseract_dir else None
    if tesseract_exe is not None:
        if not tesseract_exe.is_file():
            raise ParserEnvironmentError(_TESSERACT_INSTALL_HINT)
    else:
        if shutil.which("tesseract") is None:
            raise ParserEnvironmentError(_TESSERACT_INSTALL_HINT)

    tessdata = tessdata_prefix or Path(os.environ.get("TESSDATA_PREFIX", ""))
    missing: List[str] = [
        f"{lang}.traineddata"
        for lang in languages
        if lang and not (tessdata / f"{lang}.traineddata").is_file()
    ]
    if missing:
        raise ParserEnvironmentError(
            f"Отсутствуют языковые модели tesseract: {', '.join(missing)}. "
            f"Проверьте TESSDATA_PREFIX (сейчас: {tessdata or '<не задан>'})"
        )
    return str(tessdata) if tessdata_prefix or tessdata != Path("") else None


def apply_min_image_area(
    image_b64: Optional[str],
    area_px: Optional[int],
    min_image_area: int,
) -> Optional[str]:
    """Фильтр мелких изображений: base64 сохраняется только для крупных.

    Фильтр из спеки change'а (2.1): мелкие значки (< 200×200 px) не закладываются
    в JSON — VLM их не увидит.

    Args:
        image_b64: Raw base64 (или None, если уже нет).
        area_px: Площадь изображения w*h или None (когда неизвестна).
        min_image_area: Минимальная площадь для сохранения base64.

    Returns:
        image_b64 — если картинка отсутствует или достаточно крупная;
        None — если картинка мелкая или base64 отсутствует.
    """  # noqa: E501 — docstring ru
    if image_b64 is None:
        return None
    if area_px is not None and area_px < min_image_area:
        return None
    return image_b64


def make_raw_element(
    element,  # unstructured элемент (Type[T]) — в сигнатуре без аннотации типа
    doc_id: str,
    index: int,
    *,
    min_image_area: int,
    page_label: Optional[str] = None,
) -> RawElement:
    """Сконвертировать Unstructured-элемент в нормализованный ``RawElement``.

    Args:
        element: Элемент из ``partition_pdf`` (unstructured.documents.elements).
        doc_id: Идентификатор документа.
        index: Порядковый индекс по документу (для стабильного element_id).
        min_image_area: Минимальная площадь картинки для сохранения base64.
        page_label: Метка страницы с печати (контракт чанка курса, Занятие 2).

    Returns:
        Нормализованный элемент.
    """
    import base64

    meta = getattr(element, "metadata", None)
    coords = getattr(meta, "coordinates", None)
    # Номер исходной страницы: course-протокол (Занятие 2) хранит его в
    # metadata.page_number; coordinates (bounding box) заполняется только при
    # layout-анализе и может отсутствовать вовсе (strategy="fast").
    page_number = getattr(meta, "page_number", None)
    if page_number is None and coords is not None:
        page_number = getattr(coords, "page_number", None)

    category_depth = -1
    if getattr(element, "category", "") == "Title":
        category_depth = getattr(meta, "category_depth", None)
        if category_depth is None:
            category_depth = 0

    html = getattr(meta, "text_as_html", None) if getattr(element, "category", "") == "Table" else None

    image_b64: Optional[str] = None
    area_px: Optional[int] = None
    if getattr(element, "category", "") == "Image":
        b64 = getattr(meta, "image_base64", None)
        coords_w = getattr(coords, "width", None) if coords is not None else None
        coords_h = getattr(coords, "height", None) if coords is not None else None
        if b64:
            # Проверка успешного декодирования добавлена ради fail-fast
            try:
                raw = base64.b64decode(b64.split(",")[-1], validate=False)
                image_b64 = b64
                if coords_w and coords_h:
                    area_px = int(coords_w * coords_h)
                else:
                    # Fallback: реальные размеры из самого изображения (PIL)
                    from io import BytesIO

                    from PIL import Image

                    with Image.open(BytesIO(raw)) as img:
                        area_px = int(img.width * img.height)
            except Exception as exc:  # noqa: BLE001 — кривой data-url не должен ронять парс
                logger.warning("Не удалось обработать base64 изображения на стр. %s: %s",
                               page_number, exc)
                image_b64 = None

    return RawElement(
        element_id=element_id(doc_id, index),
        doc_id=doc_id,
        type=getattr(element, "category", "UncategorizedText"),
        text=element.text or "",
        page=int(page_number),
        page_label=page_label,
        category_depth=int(category_depth or -1),
        html=html,
        image_b64=apply_min_image_area(image_b64, area_px, min_image_area),
        area_px=area_px,
    )


def _silence_pdfminer_noise() -> None:
    """Глушить известный шум pdfminer (векторная графика отчётов)."""
    warnings.filterwarnings("ignore", message="Cannot set non-stroke color")
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Сериализация артефактов (задача 4.1)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
    """Атомарная запись JSON: сначала во tmp-файл, затем rename на целевой путь."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp.replace(path)


def save_artifacts(doc_id: str, result: ParseResult,
                   elements_dir: Path) -> Tuple[Path, Path]:
    """Сохранить артефакты ``{doc_id}_elements.json`` и ``{doc_id}_stats.json``.

    Args:
        doc_id: Идентификатор документа.
        result: Результат парсинга.
        elements_dir: Каталог ``data/elements``.

    Returns:
        Кортеж путей (elements_json, stats_json).
    """
    elements_json = elements_dir / f"{doc_id}_elements.json"
    stats_json = elements_dir / f"{doc_id}_stats.json"
    _atomic_write_json(elements_json, [el.to_dict() for el in result.elements])
    _atomic_write_json(stats_json, result.stats)
    return elements_json, stats_json


def load_stats(elements_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    """Прочитать статистику артефакта, если он уже существует."""
    stats_path = elements_dir / f"{doc_id}_stats.json"
    if not stats_path.is_file():
        return None
    with open(stats_path, encoding="utf-8") as fh:
        return json.load(fh)


def is_already_parsed(pdf_path: Path, elements_dir: Path, doc_id: str) -> bool:
    """True, если артефакт существует и ``source_sha256`` источника совпал."""
    stats = load_stats(elements_dir, doc_id)
    if not stats:
        return False
    return stats.get("source_sha256") == compute_source_sha256(pdf_path)


def _move_to_backup(elements_dir: Path, doc_id: str) -> Optional[Tuple[Path, Path]]:
    """Перенести старые артефакты в ``data/elements/backup/`` (задача 3.3)."""
    moved: List[Path] = []
    for suffix in ("_elements.json", "_stats.json"):
        src = elements_dir / f"{doc_id}{suffix}"
        if src.is_file():
            moved.append(src)
    if not moved:
        return None
    backup_dir = elements_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for src in moved:
        src.replace(backup_dir / f"{doc_id}_{stamp}{src.suffix}")
    logger.info("[%s] старые артефакты перемещены в backup (%d шт.)", doc_id, len(moved))
    return backup_dir, elements_dir / f"{doc_id}_elements.json"


# ---------------------------------------------------------------------------
# Парсинг (задачи 3.1–3.3)
# ---------------------------------------------------------------------------

def build_page_slice(pdf_path: Path, start: int, end: int) -> "io.BytesIO":
    """Вырезать интервал страниц [start..end] (1-based) в свежий буфер BytesIO.

    Курсовой протокол Занятия 2 вместо ``partition_pdf(pages=...)``:
    в unstructured 0.27.36 параметр ``pages`` отсутствует в сигнатуре и молча
    игнорируется (уходит в ``**kwargs`` — парсится весь документ). Вырезка через
    pypdf ``PdfWriter`` + ``starting_page_number`` в ``_parse_batch`` гарантирует
    и абсолютную нумерацию страниц, и реальное ограничение объёма работы.

    Args:
        pdf_path: Путь к исходному PDF.
        start: Первая страница интервала, 1-based (включительно).
        end: Последняя страница интервала, 1-based (включительно).

    Returns:
        ``io.BytesIO`` с позиционированием на начало, пригодный для
        ``partition_pdf(file=...)``.

    Raises:
        ValueError: если интервал пустой или вышел за пределы документа.
    """
    import io

    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    if start < 1 or end > n_pages or start > end:
        raise ValueError(
            f"Интервал страниц {start}-{end} вне документа ({n_pages} стр.): {pdf_path}")
    writer = PdfWriter()
    for index in range(start - 1, end):
        writer.add_page(reader.pages[index])
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def _parse_batch(pdf_path: Path, start: int, end: int, settings: ParserSettings,
                 figures_dir: Optional[Path] = None):
    """Один батч страниц [start..end] (1-based включительно) через partition_pdf.

    Протокол spec_pdf_parser.md §3.1 (fix v1.1): страницы вырезаются в буфер
    (``build_page_slice``), номер исходной страницы сохраняется через
    ``starting_page_number``, base64 изображений кладётся в payload.

    Args:
        figures_dir: Куда Unstructured вырезает изображения (figure-*.jpg).
            Только для hi_res — иначе побочный вывод расползается по cwd;
            None — параметр в partition_pdf не передаётся вовсе.
    """
    from unstructured.partition.pdf import partition_pdf

    _silence_pdfminer_noise()
    buffer = build_page_slice(pdf_path, start, end)
    extra: Dict[str, Any] = {}
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        extra["extract_image_block_output_dir"] = str(figures_dir)
    return partition_pdf(
        file=buffer,
        strategy=settings.parse_strategy,
        infer_table_structure=(settings.parse_strategy == "hi_res"),
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,  # иначе metadata.image_base64 всегда None
        starting_page_number=start,
        languages=list(settings.ocr_languages),
        **extra,
    )


def _convert_batch(
    raw_batch: Sequence,
    doc_id: str,
    index_offset: int,
    *,
    page_range: Tuple[int, int],
    page_labels: List[str],
    min_image_area: int,
) -> List[RawElement]:
    """Сконвертировать батч элементов с валидацией страниц (переиспользовано из курса).

    Курсовая ``page_element_records`` (Занятие 2) роняет парс с сообщением
    «Элемент потерял номер исходной страницы», если метаданные элемента
    разошлись с ожидаемой страницей. Здесь та же проверка на интервал батча.
    """
    start, end = page_range
    converted: List[RawElement] = []
    for i, element in enumerate(raw_batch):
        meta = getattr(element, "metadata", None)
        item_page = getattr(meta, "page_number", None)
        if item_page is None:
            coords = getattr(meta, "coordinates", None)
            item_page = getattr(coords, "page_number", None)
        item_page = int(item_page or 1)
        label = page_labels[item_page - 1] if item_page - 1 < len(page_labels) else None
        item = make_raw_element(element, doc_id, index_offset + len(converted),
                                min_image_area=min_image_area, page_label=label)
        if not start <= item.page <= end:
            raise ValueError(
                f"Элемент потерял номер исходной страницы: {item.element_id} "
                f"(стр. {item.page}, батч {start}-{end})")
        converted.append(item)
    return converted


def parse_pdf(
    pdf_path: Path,
    *,
    settings: Optional[ParserSettings] = None,
    resume: bool = True,
) -> Optional[ParseResult]:
    """Парсить один PDF батчами по страницам.

    Args:
        pdf_path: Путь к PDF (имя файла содержит 4-цифровую группу-год).
        settings: Конфигурация; None -> ``load_parser_settings()``.
        resume: Пропустить уже спарсенный файл (совпадающий sha256).

    Returns:
        ``ParseResult``; None — файл уже спарсен (resume сработал).

    Raises:
        ParserEnvironmentError: предпосылки OCR не выполнены.
        ValueError: год не найден в имени файла.
    """
    started = datetime.now()
    settings = settings or load_parser_settings()
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    doc_id = make_doc_id(pdf_path.name)
    if resume and is_already_parsed(pdf_path, settings.data_elements_dir, doc_id):
        logger.info("[%s] уже спарсен (sha256 совпал) — пропускаю", doc_id)
        return None

    source_sha256 = compute_source_sha256(pdf_path)
    stats_prev = load_stats(settings.data_elements_dir, doc_id)
    if stats_prev and stats_prev.get("source_sha256") != source_sha256:
        logger.warning("[%s] source_sha256 изменился — пере-парс с бэкапом", doc_id)
        _move_to_backup(settings.data_elements_dir, doc_id)

    if settings.parse_strategy == "hi_res":
        preflight_check_ocr(settings.ocr_languages, tessdata_prefix=settings.tessdata_prefix)
        figures_dir: Optional[Path] = settings.data_figures_dir / doc_id
    else:
        figures_dir = None

    n_pages = count_pdf_pages(pdf_path)
    batch_size = max(1, settings.parse_page_batch)
    page_labels = page_labels_of(pdf_path)

    from unstructured.documents.elements import Element

    converted: List[RawElement] = []
    skipped_ranges: List[str] = []
    for start in range(1, n_pages + 1, batch_size):
        end = min(start + batch_size - 1, n_pages)
        batch_start_time = datetime.now()
        raw_batch: Optional[List[Element]] = None
        last_error: Optional[BaseException] = None
        for attempt in (1, 2):  # одна повторная попытка (спека: сбои батча)
            try:
                raw_batch = _parse_batch(pdf_path, start, end, settings,
                                         figures_dir=figures_dir)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 — парсеры дают широкий спектр
                last_error = exc
                logger.warning("[%s] батч %d-%d: попытка %d провалилась: %s",
                               doc_id, start, end, attempt, exc)
        elapsed = (datetime.now() - batch_start_time).total_seconds()
        if raw_batch is None:
            skipped_ranges.append(f"{start}-{end}")
            logger.error("[%s] батч %d-%d не распарсен: %s", doc_id, start, end, last_error)
            continue
        batch_elements = _convert_batch(raw_batch, doc_id, len(converted),
                                        page_range=(start, end), page_labels=page_labels,
                                        min_image_area=settings.parse_min_image_area)
        converted.extend(batch_elements)
        logger.info("[%s] batch %d/%d | pages %d-%d | %.1fs | %d elements",
                    doc_id,
                    (start - 1) // batch_size + 1,
                    (n_pages + batch_size - 1) // batch_size,
                    start, end, elapsed, len(batch_elements))

    by_type: Dict[str, int] = {}
    for el in converted:
        by_type[el.type] = by_type.get(el.type, 0) + 1

    stats: Dict[str, Any] = {
        "doc_id": doc_id,
        "source_url": str(pdf_path),
        "source_sha256": source_sha256,
        "year": int(doc_id.removeprefix("sber_")),
        "report_period": doc_id.removeprefix("sber_"),
        "n_pages": n_pages,
        "n_elements": len(converted),
        "by_type": by_type,
        "skipped_page_ranges": skipped_ranges,
        "parse_seconds": (datetime.now() - started).total_seconds(),
        "strategy": settings.parse_strategy,
        "ocr_languages": list(settings.ocr_languages),
        "parser_version": importlib.metadata.version("unstructured"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result = ParseResult(elements=converted, stats=stats)
    save_artifacts(doc_id, result, settings.data_elements_dir)
    logger.info("[%s] готово: %d элементов -> %s", doc_id, len(converted),
                settings.data_elements_dir)
    return result


def parse_all(raw_dir: Path, *, settings: Optional[ParserSettings] = None,
              resume: bool = True) -> List[ParseResult]:
    """Парсить все PDF каталога ``data/raw`` (задача 3.4).

    Args:
        raw_dir: Каталог с исходными PDF.
        settings: Конфигурация; None -> ``load_parser_settings()``.
        resume: Пропускает уже успешно спарсенные файлы.

    Returns:
        Список результатов по реально спарсенным файлам (пропущенные — None).
    """
    settings = settings or load_parser_settings()
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    if not pdfs:
        logger.warning("В каталоге %s нет PDF", raw_dir)
        return []
    results: List[ParseResult] = []
    for pdf_path in pdfs:
        try:
            result = parse_pdf(pdf_path, settings=settings, resume=resume)
        except ParserEnvironmentError:
            raise
        except Exception as exc:  # noqa: BLE001 — изолируем сбой одного файла
            logger.error("Файл %s не спарсен: %s", pdf_path, exc)
            continue
        if result is not None:
            results.append(result)
    logger.info("Итого спарсено %d из %d файлов", len(results), len(pdfs))
    return results


def preview_pdf(
    pdf_path: Path,
    *,
    settings: Optional[ParserSettings] = None,
    n_pages: int = 3,
) -> Dict[str, int]:
    """Быстрое превью первых страниц PDF без записи артефактов (задача 5.1).

    Используется режимами CLI ``--smoke`` и ``--pdf ... --fast-preview``.
    При стратегии ``hi_res`` выполняется preflight OCR-среды.

    Args:
        pdf_path: Путь к PDF.
        settings: Конфигурация; None -> ``load_parser_settings()``.
        n_pages: Сколько первых страниц распарсить.

    Returns:
        Словарь ``{тип элемента: количество}``.

    Raises:
        ParserEnvironmentError: предпосылки OCR не выполнены (для hi_res).
    """
    settings = settings or load_parser_settings()
    pdf_path = Path(pdf_path)
    total_pages = count_pdf_pages(pdf_path)
    n_preview = max(1, min(n_pages, total_pages))
    figures_dir: Optional[Path] = None
    if settings.parse_strategy == "hi_res":
        preflight_check_ocr(settings.ocr_languages, tessdata_prefix=settings.tessdata_prefix)
        figures_dir = settings.data_figures_dir / f"{make_doc_id(pdf_path.name)}_preview"
    raw_batch = _parse_batch(pdf_path, 1, n_preview, settings, figures_dir=figures_dir)
    by_type: Dict[str, int] = {}
    for element in raw_batch:
        category = getattr(element, "category", "UncategorizedText")
        by_type[category] = by_type.get(category, 0) + 1
    return by_type


def format_stats_table(pdfs: Optional[List[Path]] = None,
                       settings: Optional[ParserSettings] = None) -> str:
    """Свободная сводная таблица по спарсенным артефактам (режим CLI --stats).

    Args:
        pdfs: Отфильтровать до этих файлов; None — все артефакты каталога.
        settings: Конфигурация; None -> ``load_parser_settings()``.

    Returns:
        Многострочная строка (doc_id | pages | elements | seconds).
    """
    settings = settings or load_parser_settings()
    stats_files = sorted(settings.data_elements_dir.glob("*_stats.json"))
    rows = []
    for stats_file in stats_files:
        with open(stats_file, encoding="utf-8") as fh:
            st = json.load(fh)
        rows.append((st["doc_id"], st.get("n_pages"), st.get("n_elements"),
                     round(st.get("parse_seconds", 0))))
    header = f"{'doc_id':12} {'pages':>6} {'elements':>9} {'seconds':>8}"
    lines = [header, "-" * len(header)]
    lines += [f"{doc:12} {pages:>6} {elements:>9} {seconds:>8}"
              for doc, pages, elements, seconds in rows]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover — ручной запуск
    from src.ingestion.cli import main

    sys.exit(main())
