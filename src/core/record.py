"""Единый доступ к полям чанков и записей-словарей."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field_value(value: Any, name: str, default: Any = None) -> Any:
    """Прочитать поле из ``Mapping`` или атрибута объекта."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def nested_field_value(value: Any, name: str, default: Any = None) -> Any:
    """Прочитать поле верхнего уровня, затем ``metadata``."""
    if name in value:
        return value[name]
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return default


def chunk_text(chunk: Any, *, strip: bool = False) -> str:
    """Текст чанка вместе с табличным ``html``."""
    text = field_value(chunk, "text") or ""
    html = field_value(chunk, "html") or ""
    result = f"{text} {html}"
    return result.strip() if strip else result


def to_year(value: Any) -> int | None:
    """Привести значение к году; ``bool`` и мусор дают ``None``."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def chunk_year(chunk: Any) -> int | None:
    """Год чанка или ``None``, если поле отсутствует/некорректно."""
    return to_year(field_value(chunk, "year"))
