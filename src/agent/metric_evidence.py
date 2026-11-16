"""Подключение таблиц-источников для запрошенных метрик."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agent.state import QueryPlan
from src.agent.tools import chunk_from_hit


def attach_metric_evidence(
    tools: Any,
    plan: QueryPlan,
    evidence: dict[str, Any],
    manifest: set[str],
    search_manifest: set[str] | None = None,
) -> list[str]:
    """Добавить в evidence чанки таблиц, откуда взяты значения метрик.

    Значения берутся из структурного индекса, где год стоит в шапке таблицы.
    Чанк-источник становится частью манифеста, поэтому ответ может на него
    ссылаться. Уже присутствующие чанки не дублируются.
    """
    from src.core.metrics_table import reference_rows

    rows = reference_rows(plan)
    if not rows:
        return []
    added: list[str] = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in evidence:
            continue
        try:
            result = tools.invoke("get_chunk", {"chunk_id": chunk_id})
        except Exception:  # noqa: BLE001 - отсутствие чанка не должно ронять ответ
            continue
        resolved_id, chunk = chunk_from_hit(result)
        target_id = resolved_id or chunk_id
        if not isinstance(chunk, Mapping):
            continue
        evidence[target_id] = dict(chunk)
        manifest.add(target_id)
        if search_manifest is not None:
            search_manifest.add(target_id)
        added.append(target_id)
    return added
