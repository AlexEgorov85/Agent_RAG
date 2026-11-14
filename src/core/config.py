"""Конфигурация ingestion-слоя через pydantic-settings.

Конвенции env-переменных повторяют курсовые (см. course/, Занятие 1: ключ Cloud.ru),
значения читаются из ``.env`` в корне проекта (см. ``.env.example``) и переопределяются
CLI-флагами argparse (приоритет CLI > env описан в спеке change'а `add-pdf-parser`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class ParserSettings(BaseSettings):
    """Настройки PDF-парсера (change `add-pdf-parser`).

    Attributes:
        parse_strategy: Стратегия unstructured ("hi_res" | "fast").
        parse_page_batch: Число страниц за один вызов partition_pdf.
        parse_min_image_area: Минимальная площадь изображения (w*h) для сохранения base64.
        ocr_languages: OCR-языки (tesseract traineddata).
        data_raw_dir: Каталог исходных PDF.
        data_elements_dir: Каталог артефактов парсера.
        data_structure_dir: Каталог артефактов structure_builder
            (``{doc_id}_sections.json``, шаг 2 плана).
        data_figures_dir: Корневой каталог вырезов изображений Unstructured
            (figure-*.jpg); парсер пишет в подпапку ``data/figures/{doc_id}``.
        tessdata_prefix: Каталог с языковыми моделями tesseract
            (по умолчанию — env TESSDATA_PREFIX или <project>/tessdata).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    parse_strategy: str = "hi_res"
    parse_page_batch: int = 20
    parse_min_image_area: int = 200 * 200
    ocr_languages: Tuple[str, ...] = ("rus", "eng")
    data_raw_dir: Path = Path("data/raw")
    data_elements_dir: Path = Path("data/elements")
    #: Каталог артефактов structure_builder ({doc_id}_sections.json, шаг 2 плана).
    data_structure_dir: Path = Path("data/structure")
    #: Куда Unstructured пишет вырезы изображений figure-*.jpg (побочный вывод) —
    #: всегда внутри data/, подпапка на документ добавляется парсером.
    data_figures_dir: Path = Path("data/figures")
    tessdata_prefix: Path = Path("tessdata")


def load_parser_settings() -> ParserSettings:
    """Загрузить настройки парсера (env + .env; CLI-флаги применяются поверх выше)."""
    return ParserSettings()
