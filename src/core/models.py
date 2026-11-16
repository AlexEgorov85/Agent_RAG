"""Общие нормализованные модели данных ingestion-слоя.

Идейная основа — контракт чанка К2, здесь — его «сырьевой»
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
            как в контракте чанка (поле ``page_label``).
        category_depth: Уровень заголовка для ``Title`` (0..3), ``-1`` для прочих.
        html: HTML-представление таблицы (``metadata.text_as_html``) либо None.
        image_b64: Base64 изображения (data-url) либо None, если картинка мелкая
            или payload отключён настройкой ``store_image_payload``.
        area_px: Площадь изображения в пикселях (w*h) для картинок, иначе None.
        bbox: Bounding box элемента ``[x0, y0, x1, y1]`` в системе координат
            Unstructured (обычно PixelSpace); None, если координат нет.
        page_width: Ширина layout-страницы в единицах координат (``layout_width``).
        page_height: Высота layout-страницы в единицах координат (``layout_height``).
        detection_class_prob: Уверенность layout-модели в классе элемента (0..1).
        image_path: Путь к вырезанному изображению (``figure-*.jpg``), относительно
            корня проекта, если возможно; None для не-Image элементов.
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
    bbox: Optional[List[float]] = None
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    detection_class_prob: Optional[float] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь (готова для ``json.dump(..., ensure_ascii=False)``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawElement":
        """Обратное восстановление из словаря (валидация ключей не выполняется)."""
        known = {"element_id", "doc_id", "type", "text", "page", "page_label",
                 "category_depth", "html", "image_b64", "area_px", "bbox",
                 "page_width", "page_height", "detection_class_prob", "image_path"}
        payload = {k: v for k, v in data.items() if k in known}
        # JSON не различает tuple/list: bbox всегда восстанавливаем как list[float],
        # чтобы roundtrip сравнение dataclass'ов оставалось стабильным.
        if payload.get("bbox") is not None:
            payload["bbox"] = [float(value) for value in payload["bbox"]]
        return cls(**payload)


@dataclass
class ParseResult:
    """Результат парсинга одного PDF: элементы + статистика.

    Attributes:
        elements: Нормализованные элементы в порядке следования по документу.
        stats: Словарь статистики (см. контрак ``{doc_id}_stats.json`` в спеке).
    """

    elements: List[RawElement]
    stats: Dict[str, Any] = field(default_factory=dict)
