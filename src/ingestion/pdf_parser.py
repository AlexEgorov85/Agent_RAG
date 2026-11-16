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
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.io import atomic_write_json
from src.core.models import ParseResult, RawElement
from src.core.config import ParserSettings, load_parser_settings

logger = logging.getLogger(__name__)

#: Регулярка для извлечения года из имени файла («Копия Сбер 2015.pdf» -> 2015)
_YEAR_RE = re.compile(r"\d{4}")

#: Пять знаков на индекс в element_id: sber_2015:00000
_ELEMENT_INDEX_WIDTH = 5

#: Версия схемы артефакта. Меняется при изменении набора полей RawElement или
#: семантики парсинга; старые артефакты считаются устаревшими и перепарсиваются.
PARSE_SCHEMA_VERSION = 2

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
    """Метка страницы с печати, как на страницах PDF.

    Чанк хранит ``page_label`` из ``PdfReader.page_labels`` — она нужна
    фильтрам поиска по «напечатанной» странице.
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
        Эквивалентный ``TESSDATA_PREFIX``, если каталог задан явно или через
        env и найдены все языковые файлы; иначе None.

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


def _bbox_and_page_size(coords) -> Tuple[Optional[List[float]], Optional[int], Optional[int]]:
    """Извлечь bbox ``[x0, y0, x1, y1]`` и размеры layout-страницы.

    Unstructured хранит координаты в ``metadata.coordinates.points`` (четыре
    угла в PixelSpace). Для ``fast``-стратегии координат может не быть вовсе.

    Args:
        coords: ``metadata.coordinates`` элемента или None.

    Returns:
        Кортеж ``(bbox, page_width, page_height)``; отсутствующие значения — None.
    """
    if coords is None:
        return None, None, None
    # CoordinatesMetadata отдаёт наружу только points/system; layout_width/
    # layout_height живут в system (PixelSpace) и в to_dict()-представлении.
    page_width = getattr(coords, "layout_width", None)
    page_height = getattr(coords, "layout_height", None)
    if page_width is None or page_height is None:
        system = getattr(coords, "system", None)
        page_width = page_width or getattr(system, "layout_width", None)
        page_height = page_height or getattr(system, "layout_height", None)
    if page_width is None or page_height is None:
        try:
            flat = coords.to_dict()
            page_width = page_width or flat.get("layout_width")
            page_height = page_height or flat.get("layout_height")
        except (AttributeError, TypeError):
            pass

    bbox: Optional[List[float]] = None
    points = getattr(coords, "points", None)
    if points:
        try:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        except (TypeError, ValueError, IndexError):
            bbox = None
    if bbox is None:
        x = getattr(coords, "x", None)
        y = getattr(coords, "y", None)
        width = getattr(coords, "width", None)
        height = getattr(coords, "height", None)
        if None not in (x, y, width, height):
            bbox = [float(x), float(y), float(x + width), float(y + height)]

    return (bbox,
            int(page_width) if page_width is not None else None,
            int(page_height) if page_height is not None else None)


def _portable_image_path(value: Any) -> Optional[str]:
    """Привести ``metadata.image_path`` к пути относительно корня проекта."""
    if not value:
        return None
    path = Path(str(value))
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def parse_signature(settings: ParserSettings) -> Dict[str, Any]:
    """Подпись настроек парсинга (для корректного resume и версии схемы).

    Resume по одному ``source_sha256`` опасен: смена стратегии, OCR-языков или
    схемы артефакта не меняет PDF, и старые файлы молча остались бы в корпусе.
    """
    return {
        "schema_version": PARSE_SCHEMA_VERSION,
        "strategy": settings.parse_strategy,
        "ocr_languages": list(settings.ocr_languages),
        "min_image_area": int(settings.parse_min_image_area),
        "store_image_payload": bool(settings.store_image_payload),
        "infer_table_structure": settings.parse_strategy == "hi_res",
    }


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
        page_label: Метка страницы с печати (поле контракта чанка).

    Returns:
        Нормализованный элемент.
    """
    import base64

    meta = getattr(element, "metadata", None)
    coords = getattr(meta, "coordinates", None)
    # Номер исходной страницы: контракт чанка хранит его в
    # metadata.page_number; coordinates (bounding box) заполняется только при
    # layout-анализе и может отсутствовать вовсе (strategy="fast").
    page_number = getattr(meta, "page_number", None)
    if page_number is None and coords is not None:
        page_number = getattr(coords, "page_number", None)

    category = getattr(element, "category", "UncategorizedText")
    category_depth = -1
    if category == "Title":
        category_depth = getattr(meta, "category_depth", None)
        if category_depth is None:
            category_depth = 0

    html = getattr(meta, "text_as_html", None) if category == "Table" else None

    bbox, page_width, page_height = _bbox_and_page_size(coords)

    image_b64: Optional[str] = None
    area_px: Optional[int] = None
    image_path: Optional[str] = None
    if category == "Image":
        image_path = _portable_image_path(getattr(meta, "image_path", None))
        coords_w = getattr(coords, "width", None) if coords is not None else None
        coords_h = getattr(coords, "height", None) if coords is not None else None
        if (not coords_w or not coords_h) and bbox is not None:
            coords_w = bbox[2] - bbox[0]
            coords_h = bbox[3] - bbox[1]
        if coords_w and coords_h:
            # Площадь в единицах layout-пространства: тот же масштаб, что и
            # фильтр MIN_IMAGE_AREA (200×200), независимо от base64 payload.
            area_px = int(coords_w * coords_h)
        b64 = getattr(meta, "image_base64", None)
        if b64:
            # Проверка успешного декодирования добавлена ради fail-fast
            try:
                raw = base64.b64decode(b64.split(",")[-1], validate=False)
                image_b64 = b64
                if area_px is None:
                    # Fallback: реальные размеры из самого изображения (PIL)
                    from io import BytesIO

                    from PIL import Image

                    with Image.open(BytesIO(raw)) as img:
                        area_px = int(img.width * img.height)
            except Exception as exc:  # noqa: BLE001 — кривой data-url не должен ронять парс
                logger.warning("Не удалось обработать base64 изображения на стр. %s: %s",
                               page_number, exc)
                image_b64 = None
        image_b64 = apply_min_image_area(image_b64, area_px, min_image_area)

    detection_prob = getattr(meta, "detection_class_prob", None)
    try:
        detection_prob = float(detection_prob) if detection_prob is not None else None
    except (TypeError, ValueError):
        detection_prob = None

    return RawElement(
        element_id=element_id(doc_id, index),
        doc_id=doc_id,
        type=category,
        text=element.text or "",
        page=int(page_number),
        page_label=page_label,
        category_depth=int(category_depth or -1),
        html=html,
        image_b64=image_b64,
        area_px=area_px,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        detection_class_prob=detection_prob,
        image_path=image_path,
    )


def _silence_pdfminer_noise() -> None:
    """Глушить известный шум pdfminer (векторная графика отчётов)."""
    warnings.filterwarnings("ignore", message="Cannot set non-stroke color")
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Сериализация артефактов (задача 4.1)
# ---------------------------------------------------------------------------

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
    atomic_write_json(elements_json, [el.to_dict() for el in result.elements])
    atomic_write_json(stats_json, result.stats)
    return elements_json, stats_json


def load_stats(elements_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    """Прочитать статистику артефакта, если он уже существует."""
    stats_path = elements_dir / f"{doc_id}_stats.json"
    if not stats_path.is_file():
        return None
    with open(stats_path, encoding="utf-8") as fh:
        return json.load(fh)


def is_already_parsed(pdf_path: Path, elements_dir: Path, doc_id: str,
                      settings: Optional[ParserSettings] = None) -> bool:
    """True, если артефакт актуален: совпали sha256 источника и подпись парсинга.

    Args:
        pdf_path: Исходный PDF.
        elements_dir: Каталог ``data/elements``.
        doc_id: Идентификатор документа.
        settings: Текущие настройки; None — проверять только sha256
            (совместимость со старыми вызовами и тестами).
    """
    stats = load_stats(elements_dir, doc_id)
    if not stats:
        return False
    if stats.get("source_sha256") != compute_source_sha256(pdf_path):
        return False
    if settings is None:
        return True
    return stats.get("parse_signature") == parse_signature(settings)


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
        # В имени сохраняем исходный stem: у _elements.json и _stats.json
        # одинаковый суффикс .json, иначе второй файл затрёт первый.
        src.replace(backup_dir / f"{src.stem}_{stamp}{src.suffix}")
    logger.info("[%s] старые артефакты перемещены в backup (%d шт.)", doc_id, len(moved))
    return backup_dir, elements_dir / f"{doc_id}_elements.json"


# ---------------------------------------------------------------------------
# Парсинг (задачи 3.1–3.3)
# ---------------------------------------------------------------------------

def build_page_slice(pdf_path: Path, start: int, end: int) -> "io.BytesIO":
    """Вырезать интервал страниц [start..end] (1-based) в свежий буфер BytesIO.

    Протокол батчей вместо ``partition_pdf(pages=...)``:
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
        # base64 кладётся в payload только по настройке; файлы figure-*.jpg
        # сохраняются через extract_image_block_output_dir в любом случае.
        extract_image_block_to_payload=settings.store_image_payload,
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
    """Сконвертировать батч элементов с валидацией страниц.

    Парс падает с сообщением «Элемент потерял номер исходной страницы», если
    метаданные элемента разошлись с ожидаемой страницей. Здесь та же проверка
    на интервал батча.
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
    force: bool = False,
) -> Optional[ParseResult]:
    """Парсить один PDF батчами по страницам.

    Args:
        pdf_path: Путь к PDF (имя файла содержит 4-цифровую группу-год).
        settings: Конфигурация; None -> ``load_parser_settings()``.
        resume: Пропустить актуальный артефакт (совпали sha256 и подпись парсинга).
        force: Перепарсить даже при актуальном артефакте (старый уходит в backup).

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
    signature = parse_signature(settings)
    if resume and not force and is_already_parsed(
            pdf_path, settings.data_elements_dir, doc_id, settings=settings):
        logger.info("[%s] уже спарсен (sha256 и подпись совпали) — пропускаю", doc_id)
        return None

    source_sha256 = compute_source_sha256(pdf_path)
    stats_prev = load_stats(settings.data_elements_dir, doc_id)
    if stats_prev:
        sha_changed = stats_prev.get("source_sha256") != source_sha256
        signature_changed = stats_prev.get("parse_signature") != signature
        if force or sha_changed or signature_changed:
            reason = ("--force" if force else
                      "source_sha256 изменился" if sha_changed else
                      "изменилась подпись парсинга (схема/настройки)")
            logger.warning("[%s] %s — пере-парс с бэкапом", doc_id, reason)
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
        "parse_signature": signature,
        "parser_version": importlib.metadata.version("unstructured"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result = ParseResult(elements=converted, stats=stats)
    save_artifacts(doc_id, result, settings.data_elements_dir)
    logger.info("[%s] готово: %d элементов -> %s", doc_id, len(converted),
                settings.data_elements_dir)
    return result


def parse_all(raw_dir: Path, *, settings: Optional[ParserSettings] = None,
              resume: bool = True, force: bool = False) -> List[ParseResult]:
    """Парсить все PDF каталога ``data/raw`` (задача 3.4).

    Args:
        raw_dir: Каталог с исходными PDF.
        settings: Конфигурация; None -> ``load_parser_settings()``.
        resume: Пропускает уже успешно спарсенные файлы.
        force: Перепарсить все файлы, даже актуальные (старые — в backup).

    Returns:
        Список результатов по реально спарсенным файлам; пропущенные и упавшие
        файлы в список не входят.
    """
    settings = settings or load_parser_settings()
    pdfs = sorted(Path(raw_dir).glob("*.pdf"))
    if not pdfs:
        logger.warning("В каталоге %s нет PDF", raw_dir)
        return []
    results: List[ParseResult] = []
    for pdf_path in pdfs:
        try:
            result = parse_pdf(pdf_path, settings=settings, resume=resume, force=force)
        except ParserEnvironmentError:
            raise
        except Exception as exc:  # noqa: BLE001 — изолируем сбой одного файла
            logger.error("Файл %s не спарсен: %s", pdf_path, exc)
            continue
        if result is not None:
            results.append(result)
    logger.info("Итого спарсено %d из %d файлов", len(results), len(pdfs))
    return results


def select_sample_pages(n_pages: int, *, total_pages: int = 14,
                        toc_pages: int = 10) -> List[int]:
    """Выбрать страницы для быстрой межгодовой выборки (``--sample``).

    Первые ``toc_pages`` страниц дают оглавление/дизайн, остальные — равномерно
    по телу документа (квартили). Это позволяет проверить парсер, TOC, таблицы
    и картинки на всех 9 отчётах за минуты, не парся файлы целиком.

    Args:
        n_pages: Число страниц в PDF.
        total_pages: Желаемое число страниц выборки на документ.
        toc_pages: Сколько первых страниц брать под TOC/обложку.

    Returns:
        Отсортированный список уникальных 1-based страниц.
    """
    if n_pages <= 0:
        return []
    toc = list(range(1, min(toc_pages, n_pages) + 1))
    body_count = max(0, total_pages - len(toc))
    body: List[int] = []
    if body_count and n_pages > len(toc):
        for index in range(1, body_count + 1):
            # Квартили тела: 12.5%, 37.5%, 62.5%, 87.5% при body_count=4.
            fraction = (index - 0.5) / body_count
            page = int(round(n_pages * fraction))
            body.append(min(n_pages, max(len(toc) + 1, page)))
    return sorted(set(toc + body))


def pages_to_ranges(pages: Sequence[int]) -> List[Tuple[int, int]]:
    """Сгруппировать страницы в непрерывные интервалы ``[(start, end), ...]``."""
    ranges: List[Tuple[int, int]] = []
    for page in sorted(set(int(p) for p in pages)):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def parse_sample(pdf_path: Path, *, settings: Optional[ParserSettings] = None,
                 pages: Optional[Sequence[int]] = None,
                 total_pages: int = 14,
                 samples_dir: Optional[Path] = None) -> ParseResult:
    """Распарсить стратифицированную выборку страниц одного PDF.

    В отличие от ``parse_pdf`` ничего не пишет в ``data/elements`` и не трогает
    resume: результат идёт в ``data/samples`` (elements/stats/sections), чтобы
    быстро проверить качество на всех отчётах без полного прогона.

    Args:
        pdf_path: Исходный PDF.
        settings: Конфигурация парсера.
        pages: Явный список 1-based страниц; None — ``select_sample_pages``.
        total_pages: Размер выборки, если ``pages`` не задан.
        samples_dir: Каталог выборок; None — ``settings.data_samples_dir``.

    Returns:
        ``ParseResult`` с элементами только выбранных страниц.
    """
    settings = settings or load_parser_settings()
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")
    doc_id = make_doc_id(pdf_path.name)
    n_pages = count_pdf_pages(pdf_path)
    chosen = list(pages) if pages is not None else select_sample_pages(
        n_pages, total_pages=total_pages)
    chosen = [int(p) for p in chosen if 1 <= int(p) <= n_pages]
    if not chosen:
        raise ValueError(f"Пустая выборка страниц для {pdf_path.name}")

    if settings.parse_strategy == "hi_res":
        preflight_check_ocr(settings.ocr_languages,
                            tessdata_prefix=settings.tessdata_prefix)

    target_dir = Path(samples_dir or settings.data_samples_dir)
    figures_dir = target_dir / "figures" / doc_id
    page_labels = page_labels_of(pdf_path)
    converted: List[RawElement] = []
    skipped: List[str] = []
    for start, end in pages_to_ranges(chosen):
        try:
            raw_batch = _parse_batch(pdf_path, start, end, settings,
                                     figures_dir=figures_dir)
        except Exception as exc:  # noqa: BLE001 — выборка не должна падать целиком
            logger.warning("[%s] sample %d-%d не распарсен: %s",
                           doc_id, start, end, exc)
            skipped.append(f"{start}-{end}")
            continue
        converted.extend(_convert_batch(
            raw_batch, doc_id, len(converted), page_range=(start, end),
            page_labels=page_labels, min_image_area=settings.parse_min_image_area))

    by_type: Dict[str, int] = {}
    for element in converted:
        by_type[element.type] = by_type.get(element.type, 0) + 1
    stats: Dict[str, Any] = {
        "doc_id": doc_id,
        "source_url": str(pdf_path),
        "source_sha256": compute_source_sha256(pdf_path),
        "year": int(doc_id.removeprefix("sber_")),
        "report_period": doc_id.removeprefix("sber_"),
        "n_pages": n_pages,
        "sample_pages": chosen,
        "sample_page_ranges": [f"{start}-{end}"
                               for start, end in pages_to_ranges(chosen)],
        "n_pages_sampled": len(chosen),
        "n_elements": len(converted),
        "by_type": by_type,
        "skipped_page_ranges": skipped,
        "strategy": settings.parse_strategy,
        "ocr_languages": list(settings.ocr_languages),
        "parse_signature": parse_signature(settings),
        "parser_version": importlib.metadata.version("unstructured"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result = ParseResult(elements=converted, stats=stats)
    save_artifacts(doc_id, result, target_dir)
    logger.info("[%s] sample: %d стр., %d элементов -> %s",
                doc_id, len(chosen), len(converted), target_dir)
    return result


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


def format_stats_table(settings: Optional[ParserSettings] = None) -> str:
    """Свободная сводная таблица по спарсенным артефактам (режим CLI --stats).

    Args:
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
