"""Общие нормализованные модели данных ingestion-слоя.

Идейная основа — контракт чанка из курса (Занятие 2–3), здесь — его «сырьевой»
уровень: нормализованные элементы, получаемые парсером PDF, до построения
чанков и графа.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawElement:
    """Нормализованный элемент Unstructured (см. OpenSpec capability `pdf-parsing`).

    Attributes:
        element_id: Стабильный идентификатор вида ``{doc_id}:{индекс:05d}``.
        doc_id: Идентификатор документа, например ``sber_2015``.
        type: Тип элемента Unstructured (Title, NarrativeText, Table, Image, ...).
        text: Текст элемента (может быть пустым для чисто графических).
        page: Номер страницы, 1-based (из ``metadata.coordinates``).
        page_label: Метка страницы с печати PDF (``PdfReader.page_labels``),
            как в курсовом контракте чанка (Занятие 2, полe ``page_label``).
        category_depth: Уровень заголовка для ``Title`` (0..3), ``-1`` для прочих.
        html: HTML-представление таблицы (``metadata.text_as_html``) либо None.
        image_b64: Base64 изображения (data-url) либо None, если картинка мелкая.
        area_px: Площадь изображения в пикселях (w*h) для картинок, иначе None.
    """

    element_id: str
    doc_id: str
    type: str
    text: str = ""
    page: int = 1
    page_label: Optional[str] = None
    category_depth: int = -1
    html: Optional[str] = None
    image_b64: Optional[str] = None
    area_px: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь (готова для ``json.dump(..., ensure_ascii=False)``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawElement":
        """Обратное восстановление из словаря (валидация ключей не выполняется)."""
        known = {"element_id", "doc_id", "type", "text", "page", "page_label",
                 "category_depth", "html", "image_b64", "area_px"}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ParseResult:
    """Результат парсинга одного PDF: элементы + статистика.

    Attributes:
        elements: Нормализованные элементы в порядке следования по документу.
        stats: Словарь статистики (см. контрак ``{doc_id}_stats.json`` в спеке).
    """

    elements: List[RawElement]
    stats: Dict[str, Any] = field(default_factory=dict)
