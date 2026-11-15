"""Реестр источников корпуса годовых отчётов Сбербанка.

Реестр — единственный источник истины о том, какие PDF входят в корпус:
``doc_id``, путь к файлу, SHA-256 исходника, год/период, эмитент, страна,
статус включения и пояснение. Год нельзя выводить из содержимого PDF
(``spec_target.md`` §3.1); он задаётся здесь и в имени файла.

Ключевые правила корпуса:

* 2021 год отсутствует физически — записи ``sber_2021`` быть не должно;
* ``sber_2015`` — отчёт БПС-Сбербанка (Беларусь), он не сопоставим с ПАО
  Сбербанк и не должен по умолчанию попадать во временные сравнения.

Модуль не парсит PDF: файлы проверяются по существованию и SHA-256. Все
функции работают с UTF-8; запись атомарная (tmp + ``os.replace``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

#: Корень проекта (``src/core/source_registry.py`` → parents[2]).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Путь реестра по умолчанию (относительно корня проекта).
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "source_registry.json"

#: Целевой состав корпуса (``spec_target.md`` §3.1); 2021 отсутствует.
TARGET_YEARS: tuple[int, ...] = (
    2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024,
)

#: Годы, которых в корпусе быть не должно.
MISSING_YEARS: tuple[int, ...] = (2021,)

#: Документы, которые включены, но не сопоставимы с ПАО Сбербанк.
NONCOMPARABLE_DOC_IDS: frozenset[str] = frozenset({"sber_2015"})

#: Статус «включён в основной сопоставимый корпус».
SOURCE_STATUS_COMPARABLE = "comparable"

#: Статус «включён, но эмитент/страна другие — сравнение требует явного запроса».
SOURCE_STATUS_INCOMPARABLE = "incomparable_issuer"

ALLOWED_SOURCE_STATUSES: frozenset[str] = frozenset(
    {SOURCE_STATUS_COMPARABLE, SOURCE_STATUS_INCOMPARABLE}
)

#: Обязательные поля записи реестра (``spec_target.md`` §3.1).
REQUIRED_SOURCE_FIELDS: tuple[str, ...] = (
    "doc_id",
    "file_path",
    "source_sha256",
    "year",
    "report_period",
    "issuer",
    "country",
    "source_status",
    "inclusion_note",
)

#: Поля, которые обязаны быть непустыми строками.
_TEXT_FIELDS: tuple[str, ...] = (
    "doc_id",
    "file_path",
    "source_sha256",
    "report_period",
    "issuer",
    "country",
    "source_status",
    "inclusion_note",
)

_DOC_ID_PATTERN = re.compile(r"^(?P<name>[a-z0-9][a-z0-9_]*)_(?P<year>\d{4})$")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")

#: Ожидаемый эмитент/страна для несопоставимого документа 2015.
BPS_ISSUER = "БПС-Сбербанк"
BPS_COUNTRY = "Беларусь"

#: Схема JSON-файла реестра.
REGISTRY_SCHEMA_VERSION = 1


class RegistryError(ValueError):
    """Ошибка чтения/сериализации реестра источников."""


@dataclass(frozen=True)
class SourceRecord:
    """Одна запись реестра источников.

    Attributes:
        doc_id: Идентификатор документа, например ``sber_2023``.
        file_path: Путь к PDF относительно корня проекта (или абсолютный
            путь для локального overlay, который не коммитится).
        source_sha256: SHA-256 исходного PDF (64 hex-символа).
        year: Год отчёта; обязан совпадать с годом в ``doc_id``.
        report_period: Отчётный период строкой (например, ``"2023"``).
        issuer: Эмитент отчёта (ПАО Сбербанк / БПС-Сбербанк).
        country: Страна эмитента.
        source_status: ``comparable`` или ``incomparable_issuer``.
        inclusion_note: Пояснение, почему источник включён и как с ним
            обращаться при сравнении годов.
    """

    doc_id: str
    file_path: str
    source_sha256: str
    year: int
    report_period: str
    issuer: str
    country: str
    source_status: str
    inclusion_note: str

    def to_dict(self) -> dict[str, Any]:
        """Вернуть запись как JSON-совместимый словарь (порядок полей стабилен)."""
        return {field_name: getattr(self, field_name)
                for field_name in REQUIRED_SOURCE_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceRecord":
        """Собрать запись из словаря.

        Args:
            payload: Словарь записи реестра.

        Returns:
            :class:`SourceRecord`.

        Raises:
            RegistryError: если отсутствует обязательное поле или у поля
                неверный тип.
        """
        if not isinstance(payload, Mapping):
            raise RegistryError("Запись реестра должна быть JSON-объектом.")
        missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in payload]
        if missing:
            raise RegistryError(
                "В записи реестра нет полей: " + ", ".join(sorted(missing)) + ".")
        values: dict[str, Any] = {}
        for name in REQUIRED_SOURCE_FIELDS:
            value = payload[name]
            if name == "year":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RegistryError(
                        f"{payload.get('doc_id')!r}: year должен быть целым числом.")
                values[name] = value
            else:
                if not isinstance(value, str):
                    raise RegistryError(
                        f"{payload.get('doc_id')!r}: {name} должен быть строкой.")
                values[name] = value
        return cls(**values)

    @property
    def is_comparable(self) -> bool:
        """Сопоставим ли источник с остальным корпусом ПАО Сбербанк."""
        return self.source_status == SOURCE_STATUS_COMPARABLE

    @property
    def relative_file_path(self) -> Path:
        """Путь к файлу относительно корня проекта (абсолютный — как есть)."""
        return Path(self.file_path)


@dataclass(frozen=True)
class RegistryIssue:
    """Проблема реестра.

    Attributes:
        code: Машиночитаемый код (``duplicate_doc_id``, ``missing_file``…).
        message: Человекочитаемое описание.
        doc_id: Документ, к которому относится проблема (если применимо).
        severity: ``error`` или ``warning``.
    """

    code: str
    message: str
    doc_id: str | None = None
    severity: str = "error"


@dataclass
class RegistryValidation:
    """Результат :func:`validate_registry`.

    Attributes:
        records: Проверенные записи в исходном порядке.
        issues: Найденные проблемы (ошибки и предупреждения).
    """

    records: list[SourceRecord]
    issues: list[RegistryIssue]

    @property
    def errors(self) -> list[RegistryIssue]:
        """Только ошибки (``severity == "error"``)."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[RegistryIssue]:
        """Только предупреждения."""
        return [issue for issue in self.issues if issue.severity != "error"]

    @property
    def is_valid(self) -> bool:
        """Нет ли в реестре ошибок."""
        return not self.errors

    @property
    def doc_ids(self) -> list[str]:
        """doc_id в порядке реестра."""
        return [record.doc_id for record in self.records]


def resolve_source_path(
    record: SourceRecord,
    project_root: Path | str = PROJECT_ROOT,
) -> Path:
    """Разрешить путь к PDF относительно корня проекта.

    Args:
        record: Запись реестра.
        project_root: Корень проекта для относительных путей.

    Returns:
        Абсолютный путь к PDF.
    """
    path = Path(record.file_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def compute_sha256(path: Path) -> str:
    """Посчитать SHA-256 файла.

    Args:
        path: Путь к файлу.

    Returns:
        SHA-256 в нижнем hex-регистре.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> list[SourceRecord]:
    """Прочитать реестр источников из JSON (UTF-8).

    Поддерживаются два формата: JSON-массив записей и объект
    ``{"schema_version": 1, "sources": [...]}``.

    Args:
        path: Путь к ``data/source_registry.json``.

    Returns:
        Записи реестра в порядке файла.

    Raises:
        FileNotFoundError: если файла нет.
        RegistryError: если JSON некорректен или формат не поддерживается.
    """
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Реестр источников не найден: {registry_path}")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Реестр источников не читается: {exc}") from exc

    if isinstance(payload, list):
        raw_sources: Any = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("sources"), list):
        raw_sources = payload["sources"]
    else:
        raise RegistryError(
            "Ожидался список записей или объект с полем 'sources'.")

    records: list[SourceRecord] = []
    for index, item in enumerate(raw_sources):
        try:
            records.append(SourceRecord.from_dict(item))
        except RegistryError as exc:
            raise RegistryError(f"Запись #{index}: {exc}") from exc
    return records


def save_registry(
    records: Iterable[SourceRecord],
    path: Path | str = DEFAULT_REGISTRY_PATH,
    *,
    notes: str | None = None,
) -> Path:
    """Атомарно записать реестр источников в JSON (UTF-8).

    Args:
        records: Записи реестра.
        path: Куда писать (каталог создаётся при необходимости).
        notes: Необязательное пояснение к корпусу (сохраняется в файле).

    Returns:
        Путь записанного файла.
    """
    registry_path = Path(path)
    payload: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "sources": [record.to_dict() for record in records],
    }
    if notes is not None:
        payload["notes"] = notes
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(registry_path, text)
    return registry_path


def _atomic_write_text(path: Path, text: str) -> None:
    """Записать текст через временный файл рядом с целевым и ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def validate_registry(
    records: Sequence[SourceRecord],
    *,
    project_root: Path | str = PROJECT_ROOT,
    verify_sha256: bool = True,
    require_target_years: bool = True,
) -> RegistryValidation:
    """Проверить реестр источников по правилам корпуса.

    Проверяются: обязательные поля, ровно одна запись на ``doc_id``,
    уникальность ``file_path``, соответствие года в ``doc_id`` и ``year``,
    существование файлов, совпадение SHA-256, отсутствие 2021 года,
    полнота целевых годов и маркировка ``sber_2015`` как БПС-Сбербанка.

    Args:
        records: Записи реестра.
        project_root: Корень проекта для относительных ``file_path``.
        verify_sha256: Пересчитывать ли SHA-256 файлов.
        require_target_years: Требовать ли наличие всех целевых годов.

    Returns:
        :class:`RegistryValidation` со всеми найденными проблемами.
    """
    issues: list[RegistryIssue] = []
    root = Path(project_root)
    seen_doc_ids: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    years_seen: set[int] = set()

    if not records:
        issues.append(RegistryIssue(
            "empty_registry", "Реестр источников пуст."))

    for index, record in enumerate(records):
        prefix = f"#{index} {record.doc_id}"

        for field_name in REQUIRED_SOURCE_FIELDS:
            value = getattr(record, field_name)
            if field_name in _TEXT_FIELDS and not value.strip():
                issues.append(RegistryIssue(
                    "missing_field",
                    f"{prefix}: поле {field_name} пустое.",
                    record.doc_id))
        if record.year in (0,) or record.year < 1900 or record.year > 2100:
            issues.append(RegistryIssue(
                "invalid_year", f"{prefix}: неправдоподобный year={record.year}.",
                record.doc_id))

        match = _DOC_ID_PATTERN.match(record.doc_id)
        if match is None:
            issues.append(RegistryIssue(
                "invalid_doc_id",
                f"{prefix}: doc_id должен иметь вид '<name>_<YYYY>'.",
                record.doc_id))
        elif int(match.group("year")) != record.year:
            issues.append(RegistryIssue(
                "doc_id_year_mismatch",
                f"{prefix}: год в doc_id ({match.group('year')}) не совпадает "
                f"с year={record.year}.",
                record.doc_id))

        if record.doc_id in seen_doc_ids:
            issues.append(RegistryIssue(
                "duplicate_doc_id",
                f"{prefix}: doc_id уже встречался под индексом "
                f"{seen_doc_ids[record.doc_id]}.",
                record.doc_id))
        else:
            seen_doc_ids[record.doc_id] = index
        years_seen.add(record.year)

        if record.year in MISSING_YEARS:
            issues.append(RegistryIssue(
                "missing_year_present",
                f"{prefix}: год {record.year} отсутствует в корпусе и не должен "
                "иметь записи.",
                record.doc_id))

        if record.source_status not in ALLOWED_SOURCE_STATUSES:
            issues.append(RegistryIssue(
                "invalid_source_status",
                f"{prefix}: неизвестный source_status={record.source_status!r}; "
                f"допустимо: {sorted(ALLOWED_SOURCE_STATUSES)}.",
                record.doc_id))
        if _SHA256_PATTERN.fullmatch(record.source_sha256) is None:
            issues.append(RegistryIssue(
                "invalid_source_sha256",
                f"{prefix}: source_sha256 должен быть 64 hex-символа.",
                record.doc_id))

        if record.doc_id in NONCOMPARABLE_DOC_IDS:
            if record.issuer.strip() != BPS_ISSUER:
                issues.append(RegistryIssue(
                    "issuer_mismatch",
                    f"{prefix}: ожидался issuer={BPS_ISSUER!r}, "
                    f"получено {record.issuer!r}.",
                    record.doc_id))
            if record.country.strip() != BPS_COUNTRY:
                issues.append(RegistryIssue(
                    "country_mismatch",
                    f"{prefix}: ожидалась country={BPS_COUNTRY!r}, "
                    f"получено {record.country!r}.",
                    record.doc_id))
            if record.source_status == SOURCE_STATUS_COMPARABLE:
                issues.append(RegistryIssue(
                    "comparable_foreign_issuer",
                    f"{prefix}: документ БПС-Сбербанка не может иметь статус "
                    "comparable.",
                    record.doc_id))
            if "несопоставим" not in record.inclusion_note.lower():
                issues.append(RegistryIssue(
                    "inclusion_note_missing_incomparability",
                    f"{prefix}: inclusion_note должен явно говорить о "
                    "несопоставимости с ПАО Сбербанк.",
                    record.doc_id))
        else:
            if BPS_ISSUER.lower() in record.issuer.lower():
                issues.append(RegistryIssue(
                    "mislabeled_issuer",
                    f"{prefix}: эмитент БПС-Сбербанк допустим только для "
                    f"{sorted(NONCOMPARABLE_DOC_IDS)}.",
                    record.doc_id))
            if not record.is_comparable:
                issues.append(RegistryIssue(
                    "unexpected_incomparable_status",
                    f"{prefix}: статус {record.source_status!r} допустим только "
                    f"для {sorted(NONCOMPARABLE_DOC_IDS)}.",
                    record.doc_id))

        source_path = resolve_source_path(record, root)
        path_key = str(source_path.resolve()).lower()
        if path_key in seen_paths:
            issues.append(RegistryIssue(
                "duplicate_file_path",
                f"{prefix}: file_path уже используется {seen_paths[path_key]!r}.",
                record.doc_id))
        else:
            seen_paths[path_key] = record.doc_id

        if not source_path.is_file():
            issues.append(RegistryIssue(
                "missing_file",
                f"{prefix}: файл не найден: {source_path}.",
                record.doc_id))
            continue
        if verify_sha256:
            actual = compute_sha256(source_path)
            if actual.lower() != record.source_sha256.lower():
                issues.append(RegistryIssue(
                    "source_sha256_mismatch",
                    f"{prefix}: SHA-256 не совпадает (файл {actual[:12]}…, "
                    f"реестр {record.source_sha256[:12]}…).",
                    record.doc_id))

    if require_target_years:
        for year in TARGET_YEARS:
            if year not in years_seen:
                issues.append(RegistryIssue(
                    "missing_corpus_year",
                    f"В реестре нет источника за {year} год.",
                    f"sber_{year}"))
    for year in MISSING_YEARS:
        if year in years_seen:
            issues.append(RegistryIssue(
                "missing_year_present",
                f"Год {year} не должен присутствовать в корпусе.",
                f"sber_{year}"))

    return RegistryValidation(records=list(records), issues=issues)


def record_by_doc(records: Iterable[SourceRecord]) -> dict[str, SourceRecord]:
    """Индексировать записи по ``doc_id``.

    Raises:
        RegistryError: при повторяющемся ``doc_id``.
    """
    index: dict[str, SourceRecord] = {}
    for record in records:
        if record.doc_id in index:
            raise RegistryError(f"Повторяющийся doc_id: {record.doc_id}")
        index[record.doc_id] = record
    return index


def registry_fingerprint(records: Iterable[SourceRecord]) -> str:
    """Посчитать fingerprint корпуса по парам ``doc_id:source_sha256``.

    Fingerprint используется для lineage чанков/индексов: он меняется при
    любом изменении состава источников или их содержимого.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.doc_id):
        digest.update(f"{record.doc_id}:{record.source_sha256.lower()}\n".encode())
    return digest.hexdigest()


def comparable_records(records: Iterable[SourceRecord]) -> list[SourceRecord]:
    """Вернуть только сопоставимые источники (без БПС-Сбербанка)."""
    return [record for record in records if record.is_comparable]
