"""Отдача исходных PDF по ``doc_id`` из ``data/source_registry.json``.

Endpoint ``GET /api/documents/{doc_id}/file`` (``web-ui-modernization.md``
§7.2, §5.2) — опциональная возможность для перехода из evidence к странице
PDF (``#page=N``). Источник истины о составе корпуса — реестр
:mod:`src.core.source_registry`; произвольные каталоги не сканируются.

Безопасность:

* ``doc_id`` — только ключ реестра, путь пользователем не передаётся;
* разрешённый путь обязан лежать внутри raw-каталога корпуса
  (``<registry_dir>/raw`` или ``data/raw`` проекта), после ``resolve()``;
* записи с ``..``, не-PDF и отсутствующие файлы не отдаются (404);
* тексты ошибок не содержат путей и деталей файловой системы.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import threading

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import FileResponse

from src.api.config import ApiConfig
from src.api.errors import not_found
from src.core.source_registry import (
    PROJECT_ROOT,
    RegistryError,
    SourceRecord,
    load_registry,
    record_by_doc,
    validate_registry,
)

from .deps import Owner, get_config

router = APIRouter()

#: Имя каталога исходников рядом с реестром и в ``data/`` проекта.
RAW_DIR_NAME = "raw"

#: PDF неизменяем между релизами корпуса; кэш приватный (owner-cookie).
CACHE_CONTROL = "private, max-age=3600"

#: Ответ не должен кэшироваться общими прокси.
PDF_MEDIA_TYPE = "application/pdf"

#: doc_id из реестра: ``sber_2023`` и подобные; отсекает ``..`` и пути.
_DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class _RegistrySnapshot:
    """Снимок реестра: записи по ``doc_id`` и признак валидности."""

    records: dict[str, SourceRecord]
    valid: bool


_EMPTY_SNAPSHOT = _RegistrySnapshot(records={}, valid=False)
_snapshot_lock = threading.Lock()
_snapshot_cache: dict[str, tuple[tuple[int, int], _RegistrySnapshot]] = {}


def _load_snapshot(path: Path) -> _RegistrySnapshot:
    """Прочитать и проверить реестр; ошибки не пробрасываются наружу."""
    try:
        records = load_registry(path)
        by_doc = record_by_doc(records)
        validation = validate_registry(
            records,
            # SHA-256 9 PDF на каждый запрос capabilities не пересчитываем:
            # для доступности endpoint достаточно структуры и наличия файлов.
            verify_sha256=False,
            require_target_years=False,
        )
    except (OSError, RegistryError):
        return _EMPTY_SNAPSHOT
    return _RegistrySnapshot(records=by_doc, valid=validation.is_valid)


def registry_snapshot(config: ApiConfig) -> _RegistrySnapshot:
    """Вернуть снимок реестра с кэшем по ``mtime``/размеру файла."""
    path = Path(config.source_registry_path)
    try:
        resolved = str(path.resolve())
        stat = path.stat()
    except OSError:
        return _EMPTY_SNAPSHOT
    signature = (stat.st_mtime_ns, stat.st_size)
    with _snapshot_lock:
        cached = _snapshot_cache.get(resolved)
        if cached is not None and cached[0] == signature:
            return cached[1]
    snapshot = _load_snapshot(path)
    with _snapshot_lock:
        _snapshot_cache[resolved] = (signature, snapshot)
    return snapshot


def _allowed_raw_roots(config: ApiConfig) -> tuple[Path, ...]:
    """Каталоги, внутри которых разрешено отдавать PDF."""
    candidates = (
        Path(config.source_registry_path).resolve().parent / RAW_DIR_NAME,
        PROJECT_ROOT / "data" / RAW_DIR_NAME,
    )
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(root))
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return tuple(roots)


def _within(path: Path, root: Path) -> bool:
    """Лежит ли ``path`` внутри ``root`` (с учётом регистра Windows)."""
    normalized_path = os.path.normcase(str(path))
    normalized_root = os.path.normcase(str(root))
    try:
        common = os.path.commonpath([normalized_path, normalized_root])
    except ValueError:
        return False
    return common == normalized_root


def document_path(record: SourceRecord, config: ApiConfig) -> Path | None:
    """Разрешить безопасный путь к PDF записи реестра.

    Returns:
        Абсолютный путь внутри raw-каталога или ``None``, если путь
        недопустим, выходит за разрешённые каталоги или файла нет.
    """
    raw = Path(record.file_path)
    if not record.file_path or "\x00" in record.file_path:
        return None
    if ".." in raw.parts:
        return None
    if raw.suffix.casefold() != ".pdf":
        return None
    candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not any(_within(resolved, root) for root in _allowed_raw_roots(config)):
        return None
    if not resolved.is_file():
        return None
    return resolved


def pdf_page_links_available(config: ApiConfig) -> bool:
    """Доступен ли endpoint перехода на страницу PDF.

    ``True`` только если реестр валиден (структура + существование файлов) и
    хотя бы один документ действительно отдаётся внутри raw-каталога.
    """
    snapshot = registry_snapshot(config)
    if not snapshot.valid:
        return False
    return any(
        document_path(record, config) is not None
        for record in snapshot.records.values()
    )


def _etag(stat_result: os.stat_result) -> str:
    """ETag в формате Starlette ``FileResponse`` (mtime + размер)."""
    base = f"{stat_result.st_mtime}-{stat_result.st_size}"
    digest = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()
    return f'"{digest}"'


def _if_none_match(request: Request, etag: str) -> bool:
    header = request.headers.get("if-none-match")
    if not header:
        return False
    for token in header.split(","):
        token = token.strip()
        if token == "*":
            return True
        if token.startswith("W/"):
            token = token[2:]
        if token == etag:
            return True
    return False


@router.get(
    "/documents/{doc_id}/file",
    tags=["documents"],
    response_class=FileResponse,
)
def get_document_file(
    doc_id: str,
    owner: Owner,
    request: Request,
    config: ApiConfig = Depends(get_config),
) -> Response:
    """Отдать исходный PDF документа для перехода на страницу ``#page=N``."""
    if _DOC_ID_PATTERN.fullmatch(doc_id) is None:
        raise not_found("Документ не найден.")
    record = registry_snapshot(config).records.get(doc_id)
    if record is None:
        raise not_found("Документ не найден.")
    path = document_path(record, config)
    if path is None:
        raise not_found("Документ не найден.")

    try:
        stat_result = path.stat()
    except OSError:
        raise not_found("Документ не найден.") from None

    etag = _etag(stat_result)
    if _if_none_match(request, etag):
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": CACHE_CONTROL},
        )

    return FileResponse(
        path,
        media_type=PDF_MEDIA_TYPE,
        filename=f"{record.doc_id}.pdf",
        content_disposition_type="inline",
        headers={"Cache-Control": CACHE_CONTROL, "ETag": etag},
        stat_result=stat_result,
    )
