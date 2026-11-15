"""Загрузка, проверка и хеширование эталонных наборов benchmark.

Пакет ``eval`` — отдельный контур оценки Agent_RAG. Он не импортирует
``AgentService`` на этапе валидации датасета и не обращается к сети: схема,
chunk store и метрики проверяются офлайн.

Каноническая структура описана в ``spec_benchmark.md`` §3. Suite-файлы
``datasets/easy.json``, ``medium.json``, ``hard.json`` и ``multi_turn.json``
содержат эталон; манифест фиксирует версию набора и content hash корпуса.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.retrieval.hybrid_retriever import SearchFilters

SCHEMA_VERSION = "1.1"
SUITE_NAMES: tuple[str, ...] = ("easy", "medium", "hard", "multi_turn")
SPLITS: tuple[str, ...] = ("dev", "holdout")
CATEGORIES: tuple[str, ...] = (
    "fact_lookup",
    "temporal",
    "structural",
    "multi_hop",
    "abstain",
    "dialogue",
)
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
EXPECTED_STATUSES: tuple[str, ...] = ("answer", "abstain")
TOOL_ALLOWLIST: frozenset[str] = frozenset(
    {"search_reports", "get_structure_tree", "get_entities", "get_chunk"}
)
BEHAVIOR_CODES: frozenset[str] = frozenset(
    {
        "clarify_period",
        "continue_previous_request",
        "use_previous_evidence_and_calculate",
        "apply_correction",
        "recheck_primary_source",
        "change_metric",
        "compare_with_previous_year",
    }
)
SCENARIO_CHECKS: frozenset[str] = frozenset(
    {
        "context_preserved",
        "correct_entity",
        "correct_metric",
        "correct_period",
        "no_unnecessary_restart",
        "calculation_correct",
        "sources_preserved",
        "correction_applied",
        "contradiction_rechecked",
        "clarification_asked",
        "default_period_used",
        "no_reask",
        "evidence_not_lost",
        "gap_disclosed",
    }
)
REQUIRED_CHECKS: frozenset[str] = frozenset(
    {
        "context_preserved",
        "correct_metric",
        "correct_period",
        "no_unnecessary_restart",
        "sources_preserved",
    }
)
FILTER_FIELDS: frozenset[str] = frozenset(
    {
        "years",
        "year_from",
        "year_to",
        "report_periods",
        "issuers",
        "kinds",
        "section_path_prefix",
        "breadcrumb_contains",
        "max_section_level",
        "allowed_doc_ids",
    }
)
_NUMBER_PLACEHOLDERS = frozenset({"", "tbd", "todo", "n/a", "na", "none", "null"})
_ABSOLUTE_PATH = re.compile(
    r"(?:(?<=^)|(?<=[\s\"'(\[]))(?:[A-Za-z]:[\\/]|/home/|/Users/|/tmp/|\\\\[^\\\s]+)"
)
_SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=]|access[_-]?token\s*[:=]"
    r"|password\s*[:=]|secret\s*[:=]|bearer\s+[A-Za-z0-9._~+/=-]{8,})"
)


class DatasetError(ValueError):
    """Ошибка эталонного набора с машинным кодом и списком проблем."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: Sequence["DatasetIssue"] | None = None,
    ) -> None:
        self.code = code
        self.issues = list(issues or [])
        details = "; ".join(issue.message for issue in self.issues)
        super().__init__(f"{message}: {details}" if details else message)

    def as_dict(self) -> dict[str, Any]:
        """Представление ошибки для API/лога без traceback."""
        return {
            "code": self.code,
            "message": str(self),
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DatasetIssue:
    """Одна проблема датасета, привязанная к файлу и/или тесту."""

    code: str
    message: str
    test_id: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "test_id": self.test_id,
            "path": self.path,
        }


@dataclass
class DatasetBundle:
    """Проверенный набор: suite-файлы, тесты, манифест и хеши."""

    datasets_dir: Path
    suites: dict[str, dict[str, Any]]
    tests: list[dict[str, Any]]
    manifest: dict[str, Any] | None
    suite_hashes: dict[str, str]
    dataset_sha256: str
    expected_corpus_sha256: str | None
    chunk_count: int | None
    documents: list[str]
    issues: list[DatasetIssue] = field(default_factory=list)
    #: Некритичные замечания (например, пересечение holdout-gold с dev-gold).
    #: В отличие от ``issues`` они не отменяют запуск, но должны быть видны в
    #: CLI и ``summary.json``: иначе holdout выглядит изолированным, хотя это
    #: не так.
    warnings: list[DatasetIssue] = field(default_factory=list)

    def tests_for_suite(self, suite: str) -> list[dict[str, Any]]:
        return [test for test in self.tests if test.get("suite") == suite]

    def test_by_id(self, test_id: str) -> dict[str, Any]:
        for test in self.tests:
            if test.get("id") == test_id:
                return test
        raise KeyError(test_id)


def canonical_json_bytes(payload: Any) -> bytes:
    """Канонический JSON UTF-8: сортировка ключей и компактные разделители."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def suite_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 одного suite-файла по каноническому JSON."""
    return sha256_hex(canonical_json_bytes(payload))


def dataset_hash(suites: Mapping[str, Mapping[str, Any]]) -> str:
    """SHA-256 набора: имя файла + канонический JSON в фиксированном порядке."""
    digest = hashlib.sha256()
    for suite in SUITE_NAMES:
        payload = suites.get(suite)
        if payload is None:
            continue
        digest.update(f"{suite}.json\n".encode("utf-8"))
        digest.update(canonical_json_bytes(payload))
        digest.update(b"\n")
    return digest.hexdigest()


def corpus_content_sha256(chunks: Iterable[Mapping[str, Any]]) -> str:
    """Content hash корпуса по §3.1: ID, документ, текст и source hash.

    Модель эмбеддингов и конфигурация индекса сюда не входят: они относятся к
    ``corpus_fingerprint`` кэша, а не к версии эталона.
    """
    ordered = sorted(chunks, key=lambda chunk: str(chunk.get("chunk_id") or ""))
    payload = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "text": chunk.get("text", ""),
            "source_sha256": chunk.get("source_sha256"),
        }
        for chunk in ordered
    ]
    return sha256_hex(canonical_json_bytes(payload))


def load_chunks(chunks_dir: Path | str) -> list[dict[str, Any]]:
    """Прочитать ``*_chunks.json`` без индексов и без обращения к LLM."""
    directory = Path(chunks_dir)
    if not directory.exists():
        raise DatasetError(
            "chunk_store_missing",
            f"Каталог chunk store не найден: {directory}",
        )
    chunks: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_chunks.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(
                "chunk_store_invalid",
                f"Не удалось прочитать {path.name}: {exc}",
            ) from exc
        if not isinstance(payload, list):
            raise DatasetError(
                "chunk_store_invalid",
                f"{path.name}: ожидался JSON-массив чанков",
            )
        chunks.extend(item for item in payload if isinstance(item, dict))
    if not chunks:
        raise DatasetError(
            "chunk_store_empty",
            f"В каталоге {directory} нет чанков",
        )
    return chunks


def chunk_store(chunks: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Индекс ``chunk_id -> chunk``; дубликаты считаются ошибкой корпуса."""
    store: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        if chunk_id in store:
            raise DatasetError(
                "chunk_store_invalid",
                f"Дублирующийся chunk_id в корпусе: {chunk_id}",
            )
        store[chunk_id] = dict(chunk)
    return store


def corpus_years(chunks: Iterable[Mapping[str, Any]]) -> set[int]:
    return {
        int(chunk["year"])
        for chunk in chunks
        if isinstance(chunk.get("year"), int)
        and not isinstance(chunk.get("year"), bool)
    }


def corpus_documents(chunks: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(chunk["doc_id"])
        for chunk in chunks
        if isinstance(chunk.get("doc_id"), str) and chunk["doc_id"]
    }


def load_suite(path: Path) -> dict[str, Any]:
    """Прочитать один suite-файл и проверить, что это JSON-объект."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(
            "suite_unreadable", f"{path.name}: {exc}", issues=[]
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(
            "suite_invalid_json", f"{path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetError(
            "suite_invalid", f"{path.name}: корень файла должен быть объектом"
        )
    return payload


def _normalized_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk_strings(item)


def _check_text_safety(value: Any, *, where: str) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for text in _walk_strings(value):
        if _ABSOLUTE_PATH.search(text):
            issues.append(
                DatasetIssue(
                    "absolute_path",
                    f"{where}: абсолютный путь в эталоне запрещён",
                )
            )
            break
    for text in _walk_strings(value):
        if _SECRET.search(text):
            issues.append(
                DatasetIssue(
                    "secret_in_dataset",
                    f"{where}: в эталоне найден секрет или ключ",
                )
            )
            break
    return issues


def _check_numeric_values(value: Any, *, where: str) -> list[DatasetIssue]:
    """Запрет null/TBD/пустых unit и отрицательного tolerance."""
    issues: list[DatasetIssue] = []
    if isinstance(value, Mapping):
        if "value" in value:
            raw = value.get("value")
            text = str(raw).strip().casefold() if raw is not None else ""
            if raw is None or text in _NUMBER_PLACEHOLDERS:
                issues.append(
                    DatasetIssue(
                        "numeric_value_missing",
                        f"{where}: числовое value не заполнено",
                    )
                )
        if "unit" in value and not isinstance(value.get("unit"), str):
            issues.append(
                DatasetIssue("unit_invalid", f"{where}: unit должен быть строкой")
            )
        if "unit" in value and isinstance(value.get("unit"), str) and not value["unit"].strip():
            issues.append(
                DatasetIssue("unit_empty", f"{where}: unit пустой")
            )
        if "tolerance" in value:
            tolerance = value.get("tolerance")
            if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
                issues.append(
                    DatasetIssue(
                        "tolerance_invalid",
                        f"{where}: tolerance должен быть числом >= 0",
                    )
                )
        for key in ("from_year", "to_year", "year"):
            if key in value and value[key] is not None and not _is_int(value[key]):
                issues.append(
                    DatasetIssue(
                        "year_invalid", f"{where}: {key} должен быть целым годом"
                    )
                )
        for key, item in value.items():
            issues.extend(_check_numeric_values(item, where=f"{where}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            issues.extend(_check_numeric_values(item, where=f"{where}[{index}]"))
    return issues


def _validate_filters(raw: Any, *, where: str) -> list[DatasetIssue]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return [DatasetIssue("filters_invalid", f"{where}: filters должен быть объектом")]
    issues: list[DatasetIssue] = []
    unknown = sorted(set(raw) - FILTER_FIELDS)
    if unknown:
        issues.append(
            DatasetIssue(
                "filters_invalid",
                f"{where}: неизвестные поля filters: {', '.join(unknown)}",
            )
        )
    years = raw.get("years")
    if years is not None and (
        not isinstance(years, list)
        or any(not _is_int(year) for year in years)
    ):
        issues.append(
            DatasetIssue("filters_invalid", f"{where}: years должен быть списком целых лет")
        )
    for key in ("year_from", "year_to"):
        if raw.get(key) is not None and not _is_int(raw.get(key)):
            issues.append(
                DatasetIssue("filters_invalid", f"{where}: {key} должен быть целым годом")
            )
    if _is_int(raw.get("year_from")) and _is_int(raw.get("year_to")):
        if raw["year_from"] > raw["year_to"]:
            issues.append(
                DatasetIssue(
                    "filters_invalid",
                    f"{where}: year_from больше year_to",
                )
            )
    try:
        SearchFilters(**dict(raw))
    except (TypeError, ValueError) as exc:
        issues.append(
            DatasetIssue("filters_invalid", f"{where}: SearchFilters отклоняет фильтры: {exc}")
        )
    return issues


def _gold_groups(item: Mapping[str, Any], *, where: str) -> list[DatasetIssue]:
    gold = item.get("gold_evidence")
    if not isinstance(gold, Mapping):
        return [DatasetIssue("gold_evidence_missing", f"{where}: gold_evidence обязателен")]
    issues: list[DatasetIssue] = []
    groups = gold.get("groups")
    if groups is None:
        issues.append(
            DatasetIssue("gold_evidence_missing", f"{where}: gold_evidence.groups обязателен")
        )
        return issues
    if not isinstance(groups, list):
        return [
            DatasetIssue(
                "gold_evidence_invalid", f"{where}: gold_evidence.groups должен быть массивом"
            )
        ]
    min_coverage = gold.get("min_coverage", 1.0)
    if not isinstance(min_coverage, (int, float)) or isinstance(min_coverage, bool):
        issues.append(
            DatasetIssue(
                "gold_evidence_invalid", f"{where}: min_coverage должен быть числом"
            )
        )
    elif not 0 <= min_coverage <= 1:
        issues.append(
            DatasetIssue(
                "gold_evidence_invalid", f"{where}: min_coverage вне диапазона 0..1"
            )
        )
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            issues.append(
                DatasetIssue(
                    "gold_evidence_invalid", f"{where}: группа #{index} должна быть объектом"
                )
            )
            continue
        any_of = group.get("any_of")
        if not isinstance(any_of, list) or not any_of:
            issues.append(
                DatasetIssue(
                    "gold_evidence_invalid",
                    f"{where}: группа {group.get('id', index)} без непустого any_of",
                )
            )
            continue
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in any_of):
            issues.append(
                DatasetIssue(
                    "gold_evidence_invalid",
                    f"{where}: any_of должен содержать непустые chunk_id",
                )
            )
        if "section" in group and not isinstance(group.get("section"), str):
            issues.append(
                DatasetIssue(
                    "gold_evidence_invalid",
                    f"{where}: section группы должен быть строкой",
                )
            )
    return issues


def _check_gold_chunks(
    groups: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
    *,
    where: str,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for group in groups:
        for chunk_id in group.get("any_of", []) or []:
            if chunk_id not in chunks:
                issues.append(
                    DatasetIssue(
                        "unknown_chunk_id",
                        f"{where}: chunk_id {chunk_id} отсутствует в chunk store",
                    )
                )
    return issues


def _check_expected_metadata(
    item: Mapping[str, Any],
    chunks: Mapping[str, Mapping[str, Any]],
    *,
    where: str,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    expected = item.get("expected") or {}
    if not isinstance(expected, Mapping):
        return [DatasetIssue("expected_invalid", f"{where}: expected должен быть объектом")]

    gold_ids = [
        chunk_id
        for group in (item.get("gold_evidence") or {}).get("groups", [])
        if isinstance(group, Mapping)
        for chunk_id in group.get("any_of", []) or []
    ]
    gold_chunks = [chunks[cid] for cid in gold_ids if cid in chunks]
    gold_docs = {
        str(chunk.get("doc_id"))
        for chunk in gold_chunks
        if chunk.get("doc_id") is not None
    }
    gold_years = {
        int(chunk["year"])
        for chunk in gold_chunks
        if isinstance(chunk.get("year"), int) and not isinstance(chunk.get("year"), bool)
    }

    required_documents = expected.get("required_documents", []) or []
    if not isinstance(required_documents, list):
        issues.append(
            DatasetIssue(
                "expected_invalid",
                f"{where}: expected.required_documents должен быть массивом",
            )
        )
    else:
        for doc_id in required_documents:
            if not isinstance(doc_id, str) or not doc_id:
                issues.append(
                    DatasetIssue(
                        "expected_invalid",
                        f"{where}: required_documents содержит пустой doc_id",
                    )
                )
                continue
            if gold_docs and doc_id not in gold_docs:
                issues.append(
                    DatasetIssue(
                        "gold_document_mismatch",
                        f"{where}: required_document {doc_id} не подтверждён gold evidence",
                    )
                )

    required_years = expected.get("required_years", []) or []
    missing_years = expected.get("missing_years", []) or []
    if not isinstance(required_years, list) or any(not _is_int(year) for year in required_years):
        issues.append(
            DatasetIssue(
                "expected_invalid",
                f"{where}: expected.required_years должен быть массивом целых лет",
            )
        )
    if not isinstance(missing_years, list) or any(not _is_int(year) for year in missing_years):
        issues.append(
            DatasetIssue(
                "expected_invalid",
                f"{where}: expected.missing_years должен быть массивом целых лет",
            )
        )
    if gold_years and isinstance(required_years, list):
        for year in required_years:
            if not _is_int(year):
                continue
            if year in missing_years:
                # Год можно ожидать как gap, только если он явно назван в
                # missing_years; это допустимо и для answer-теста.
                continue
            if year not in gold_years:
                issues.append(
                    DatasetIssue(
                        "required_year_without_evidence",
                        f"{where}: required_year {year} не подтверждён gold evidence",
                    )
                )
    return issues


def _check_required_facts(expected: Mapping[str, Any], *, where: str) -> list[DatasetIssue]:
    """Для числового факта обязательны unit, tolerance >= 0 и целый year."""
    issues: list[DatasetIssue] = []
    facts = expected.get("required_facts", []) or []
    if not isinstance(facts, list):
        return [
            DatasetIssue(
                "expected_invalid",
                f"{where}: expected.required_facts должен быть массивом",
            )
        ]
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping):
            issues.append(
                DatasetIssue(
                    "expected_invalid",
                    f"{where}: required_facts[{index}] должен быть объектом",
                )
            )
            continue
        if "value" in fact:
            raw = fact.get("value")
            text = str(raw).strip().casefold() if raw is not None else ""
            if raw is None or text in _NUMBER_PLACEHOLDERS:
                issues.append(
                    DatasetIssue(
                        "numeric_value_missing",
                        f"{where}: required_facts[{index}].value не заполнено",
                    )
                )
            unit = fact.get("unit")
            if not isinstance(unit, str) or not unit.strip():
                issues.append(
                    DatasetIssue(
                        "unit_empty",
                        f"{where}: required_facts[{index}] без непустого unit",
                    )
                )
    return issues


def _check_behavior(item: Mapping[str, Any], *, where: str) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    behavior = item.get("expected_behavior", {}) or {}
    if not isinstance(behavior, Mapping):
        return [DatasetIssue("behavior_invalid", f"{where}: expected_behavior должен быть объектом")]
    for key in ("must_use_tools", "may_use_tools", "must_not_use_tools"):
        tools = behavior.get(key, []) or []
        if not isinstance(tools, list):
            issues.append(
                DatasetIssue("behavior_invalid", f"{where}: {key} должен быть массивом")
            )
            continue
        unknown = sorted(set(tools) - TOOL_ALLOWLIST)
        if unknown:
            issues.append(
                DatasetIssue(
                    "tool_not_allowed",
                    f"{where}: {key} содержит неизвестные инструменты: {', '.join(unknown)}",
                )
            )
    strict = behavior.get("must_use_tools_strict", False)
    if not isinstance(strict, bool):
        issues.append(
            DatasetIssue(
                "behavior_invalid",
                f"{where}: must_use_tools_strict должен быть булевым",
            )
        )
    elif strict and not behavior.get("must_use_tools"):
        issues.append(
            DatasetIssue(
                "behavior_invalid",
                f"{where}: must_use_tools_strict=true при пустом must_use_tools",
            )
        )
    for key in ("max_search_calls", "max_tool_calls"):
        value = behavior.get(key)
        if value is not None and (not _is_int(value) or value < 0):
            issues.append(
                DatasetIssue(
                    "behavior_invalid", f"{where}: {key} должен быть целым >= 0"
                )
            )
    for key in ("strict_search_budget", "must_cite_sources", "must_abstain_if_missing"):
        value = behavior.get(key)
        if value is not None and not isinstance(value, bool):
            issues.append(
                DatasetIssue("behavior_invalid", f"{where}: {key} должен быть булевым")
            )
    for key in ("retrieval_modes",):
        value = behavior.get(key, []) or []
        if not isinstance(value, list) or any(
            mode not in {"hybrid", "dense", "keyword"} for mode in value
        ):
            issues.append(
                DatasetIssue(
                    "behavior_invalid",
                    f"{where}: retrieval_modes вне hybrid/dense/keyword",
                )
            )
    return issues


def _check_checks(item: Mapping[str, Any], *, where: str) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    checks = item.get("checks", {}) or {}
    if not isinstance(checks, Mapping):
        return [DatasetIssue("checks_invalid", f"{where}: checks должен быть объектом")]
    for key in ("must_mention", "must_not_mention", "expected_warnings"):
        values = checks.get(key, []) or []
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            issues.append(
                DatasetIssue("checks_invalid", f"{where}: {key} должен быть массивом строк")
            )
    for value in checks.get("must_mention", []) or []:
        if not isinstance(value, str):
            continue
        if value.startswith("re:"):
            try:
                re.compile(value[3:])
            except re.error:
                issues.append(
                    DatasetIssue(
                        "checks_invalid",
                        f"{where}: некорректная регулярка must_mention: {value}",
                    )
                )
    for key in ("forbidden_values", "forbidden_years"):
        values = checks.get(key, []) or []
        if not isinstance(values, list):
            issues.append(
                DatasetIssue("checks_invalid", f"{where}: {key} должен быть массивом")
            )
    for key in ("must_not_invent_values", "strict_quotes"):
        if key in checks and not isinstance(checks[key], bool):
            issues.append(
                DatasetIssue("checks_invalid", f"{where}: {key} должен быть булевым")
            )
    return issues


def _turn_issues(
    turn: Mapping[str, Any],
    *,
    where: str,
    chunks: Mapping[str, Mapping[str, Any]],
    corpus_years: set[int],
    check_chunks: bool,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    question = turn.get("question")
    if not isinstance(question, str) or not question.strip():
        issues.append(
            DatasetIssue("question_missing", f"{where}: question обязателен")
        )
    expected_status = turn.get("expected_status")
    if expected_status not in EXPECTED_STATUSES:
        issues.append(
            DatasetIssue(
                "expected_status_invalid",
                f"{where}: expected_status должен быть answer или abstain",
            )
        )
    behavior = turn.get("behavior")
    if behavior is not None and behavior not in BEHAVIOR_CODES:
        issues.append(
            DatasetIssue("behavior_invalid", f"{where}: неизвестный behavior {behavior}")
        )
    if "filters" in turn:
        issues.extend(_validate_filters(turn.get("filters"), where=f"{where}.filters"))
    issues.extend(_gold_groups(turn, where=where))
    issues.extend(_check_behavior(turn, where=where))
    issues.extend(_check_checks(turn, where=where))
    expected = turn.get("expected")
    if not isinstance(expected, Mapping):
        issues.append(
            DatasetIssue("expected_invalid", f"{where}: expected обязателен и должен быть объектом")
        )
        expected = {}
    issues.extend(_check_numeric_values(expected, where=f"{where}.expected"))
    issues.extend(_check_required_facts(expected, where=f"{where}.expected"))

    gold = turn.get("gold_evidence") or {}
    groups = gold.get("groups", []) if isinstance(gold, Mapping) else []
    min_coverage = gold.get("min_coverage", 1.0) if isinstance(gold, Mapping) else 1.0
    reuse = bool(turn.get("reuse_previous_evidence"))
    if expected_status == "answer" and not reuse and (not groups or float(min_coverage or 0) <= 0):
        issues.append(
            DatasetIssue(
                "gold_evidence_missing",
                f"{where}: answer-тест требует непустой gold_evidence и min_coverage > 0",
            )
        )
    if expected_status == "abstain":
        reasons = expected.get("reason_codes", []) or []
        warnings = (turn.get("checks") or {}).get("expected_warnings", []) or []
        if not reasons and not warnings:
            issues.append(
                DatasetIssue(
                    "abstain_reason_missing",
                    f"{where}: abstain-тест требует reason_codes или expected_warnings",
                )
            )
    if check_chunks and groups:
        issues.extend(_check_gold_chunks(groups, chunks, where=where))
        issues.extend(_check_expected_metadata(turn, chunks, where=where))
    if corpus_years:
        required_years = expected.get("required_years", []) or []
        missing_years = expected.get("missing_years", []) or []
        for year in required_years:
            if not _is_int(year):
                continue
            if year not in corpus_years and year not in missing_years:
                issues.append(
                    DatasetIssue(
                        "required_year_not_in_corpus",
                        f"{where}: год {year} отсутствует в корпусе и не указан в missing_years",
                    )
                )
    return issues


def _test_issues(
    test: Mapping[str, Any],
    *,
    suite: str,
    path_name: str,
    chunks: Mapping[str, Mapping[str, Any]],
    corpus_years: set[int],
    check_chunks: bool,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    test_id = test.get("id")
    where = f"{path_name}:{test_id or '<без id>'}"
    if not isinstance(test_id, str) or not test_id:
        issues.append(DatasetIssue("test_id_missing", f"{where}: id обязателен"))
    if test.get("suite") != suite:
        issues.append(
            DatasetIssue(
                "suite_mismatch",
                f"{where}: suite={test.get('suite')!r} не совпадает с файлом {suite!r}",
            )
        )
    split = test.get("split")
    if split not in SPLITS:
        issues.append(
            DatasetIssue("split_invalid", f"{where}: split вне dev/holdout")
        )
    category = test.get("category")
    if category not in CATEGORIES:
        issues.append(
            DatasetIssue("category_invalid", f"{where}: неизвестная category {category!r}")
        )
    difficulty = test.get("difficulty")
    if difficulty not in DIFFICULTIES:
        issues.append(
            DatasetIssue("difficulty_invalid", f"{where}: difficulty вне easy/medium/hard")
        )
    if "tags" in test and (
        not isinstance(test.get("tags"), list)
        or any(not isinstance(tag, str) for tag in test.get("tags", []))
    ):
        issues.append(
            DatasetIssue("tags_invalid", f"{where}: tags должен быть массивом строк")
        )
    if "reference_answer" in test and not isinstance(test.get("reference_answer"), str):
        issues.append(
            DatasetIssue("reference_answer_invalid", f"{where}: reference_answer должен быть строкой")
        )
    if "notes" in test and not isinstance(test.get("notes"), str):
        issues.append(
            DatasetIssue("notes_invalid", f"{where}: notes должен быть строкой")
        )
    issues.extend(_check_text_safety(test, where=where))

    if suite == "multi_turn":
        issues.extend(_multi_turn_issues(test, where=where, chunks=chunks,
                                         corpus_years=corpus_years,
                                         check_chunks=check_chunks))
    else:
        question = test.get("question")
        if not isinstance(question, str) or not question.strip():
            issues.append(
                DatasetIssue("question_missing", f"{where}: question обязателен")
            )
        expected_status = test.get("expected_status")
        if expected_status not in EXPECTED_STATUSES:
            issues.append(
                DatasetIssue(
                    "expected_status_invalid",
                    f"{where}: expected_status должен быть answer или abstain",
                )
            )
        if "filters" in test:
            issues.extend(_validate_filters(test.get("filters"), where=f"{where}.filters"))
        issues.extend(_gold_groups(test, where=where))
        issues.extend(_check_behavior(test, where=where))
        issues.extend(_check_checks(test, where=where))
        expected = test.get("expected")
        if not isinstance(expected, Mapping):
            issues.append(
                DatasetIssue(
                    "expected_invalid",
                    f"{where}: expected обязателен и должен быть объектом",
                )
            )
            expected = {}
        issues.extend(_check_numeric_values(expected, where=f"{where}.expected"))
        issues.extend(_check_required_facts(expected, where=f"{where}.expected"))
        gold = test.get("gold_evidence") or {}
        groups = gold.get("groups", []) if isinstance(gold, Mapping) else []
        min_coverage = gold.get("min_coverage", 1.0) if isinstance(gold, Mapping) else 1.0
        if expected_status == "answer" and (not groups or float(min_coverage or 0) <= 0):
            issues.append(
                DatasetIssue(
                    "gold_evidence_missing",
                    f"{where}: answer-тест требует непустой gold_evidence и min_coverage > 0",
                )
            )
        if expected_status == "abstain":
            reasons = expected.get("reason_codes", []) or []
            warnings = (test.get("checks") or {}).get("expected_warnings", []) or []
            if not reasons and not warnings:
                issues.append(
                    DatasetIssue(
                        "abstain_reason_missing",
                        f"{where}: abstain-тест требует reason_codes или expected_warnings",
                    )
                )
        if check_chunks and groups:
            issues.extend(_check_gold_chunks(groups, chunks, where=where))
            issues.extend(_check_expected_metadata(test, chunks, where=where))
        if corpus_years:
            required_years = expected.get("required_years", []) or []
            missing_years = expected.get("missing_years", []) or []
            for year in required_years:
                if not _is_int(year):
                    continue
                if year not in corpus_years and year not in missing_years:
                    issues.append(
                        DatasetIssue(
                            "required_year_not_in_corpus",
                            f"{where}: год {year} отсутствует в корпусе и не указан в missing_years",
                        )
                    )
    return issues


def _all_gold_chunk_ids(test: Mapping[str, Any]) -> set[str]:
    """Все gold ``chunk_id`` теста, включая turn'ы multi-turn."""
    ids: set[str] = set()
    golds: list[Any] = [test.get("gold_evidence")]
    turns = test.get("turns")
    if isinstance(turns, list):
        golds.extend(
            turn.get("gold_evidence")
            for turn in turns
            if isinstance(turn, Mapping)
        )
    for gold in golds:
        if not isinstance(gold, Mapping):
            continue
        for group in gold.get("groups", []) or []:
            if not isinstance(group, Mapping):
                continue
            for chunk_id in group.get("any_of", []) or []:
                if isinstance(chunk_id, str) and chunk_id:
                    ids.add(chunk_id)
    return ids


def _holdout_overlap_warnings(tests: Sequence[Mapping[str, Any]]) -> list[DatasetIssue]:
    """Предупредить, если holdout-gold пересекается с dev-gold.

    Holdout нужен для проверки переноса на невиданные доказательства. Если
    holdout-тест опирается на тот же chunk, что и dev-тест, его pass rate
    частично измеряет уже настроенные dev-примеры. Это не ошибка схемы
    (поэтому warning, а не ``DatasetError``), но отчёт обязан её показать.
    """
    dev_tests_by_chunk: dict[str, list[str]] = {}
    holdout: list[tuple[str, set[str]]] = []
    for test in tests:
        split = test.get("split")
        if split not in SPLITS:
            continue
        test_id = str(test.get("id") or "<без id>")
        ids = _all_gold_chunk_ids(test)
        if split == "dev":
            for chunk_id in ids:
                dev_tests_by_chunk.setdefault(chunk_id, []).append(test_id)
        elif split == "holdout":
            holdout.append((test_id, ids))
    warnings: list[DatasetIssue] = []
    for test_id, ids in holdout:
        overlap = sorted(chunk_id for chunk_id in ids if chunk_id in dev_tests_by_chunk)
        if not overlap:
            continue
        dev_tests = sorted(
            {
                dev_id
                for chunk_id in overlap
                for dev_id in dev_tests_by_chunk[chunk_id]
            }
        )
        warnings.append(
            DatasetIssue(
                "holdout_gold_overlap",
                f"{test_id}: {len(overlap)} gold-чанк(ов) пересекаются с dev-тестами "
                f"({', '.join(dev_tests)}); holdout-метрики не изолированы",
                test_id=test_id,
            )
        )
    return warnings


def _multi_turn_issues(
    test: Mapping[str, Any],
    *,
    where: str,
    chunks: Mapping[str, Mapping[str, Any]],
    corpus_years: set[int],
    check_chunks: bool,
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    turns = test.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        return [DatasetIssue("turns_invalid", f"{where}: multi_turn требует минимум 2 turn'а")]
    policy = test.get("dialogue_policy")
    if policy is not None:
        if not isinstance(policy, Mapping):
            issues.append(
                DatasetIssue("dialogue_policy_invalid", f"{where}: dialogue_policy должен быть объектом")
            )
        elif policy.get("missing_period") not in (None, "clarify", "default_period"):
            issues.append(
                DatasetIssue(
                    "dialogue_policy_invalid",
                    f"{where}: missing_period вне clarify/default_period",
                )
            )
    scenario_checks = test.get("scenario_checks", []) or []
    required_checks = test.get("required_checks", []) or []
    for key, values in (("scenario_checks", scenario_checks), ("required_checks", required_checks)):
        if not isinstance(values, list):
            issues.append(
                DatasetIssue("scenario_check_invalid", f"{where}: {key} должен быть массивом")
            )
            continue
        unknown = sorted(set(values) - SCENARIO_CHECKS)
        if unknown:
            issues.append(
                DatasetIssue(
                    "scenario_check_invalid",
                    f"{where}: неизвестные {key}: {', '.join(unknown)}",
                )
            )
    if not isinstance(required_checks, list):
        required_checks = []
    not_declared = sorted(set(required_checks) - set(scenario_checks))
    if not_declared:
        issues.append(
            DatasetIssue(
                "scenario_check_invalid",
                f"{where}: required_checks не входят в scenario_checks: {', '.join(not_declared)}",
            )
        )
    questions: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            issues.append(
                DatasetIssue("turn_invalid", f"{where}: turn #{index} должен быть объектом")
            )
            continue
        issues.extend(
            _turn_issues(
                turn,
                where=f"{where}.turns[{index}]",
                chunks=chunks,
                corpus_years=corpus_years,
                check_chunks=check_chunks,
            )
        )
        questions.append(_normalized_question(turn.get("question")))
        if "reuse_previous_evidence" in turn and not isinstance(
            turn.get("reuse_previous_evidence"), bool
        ):
            issues.append(
                DatasetIssue(
                    "turn_invalid",
                    f"{where}.turns[{index}]: reuse_previous_evidence должен быть булевым",
                )
            )
    if len(set(questions)) != len(questions):
        issues.append(
            DatasetIssue("duplicate_question", f"{where}: вопросы turn'ов повторяются")
        )
    return issues


def _manifest_issues(
    manifest: Mapping[str, Any],
    *,
    suites: Mapping[str, Mapping[str, Any]],
    hashes: Mapping[str, str],
    expected_corpus: str | None,
    chunk_count: int | None,
    documents: list[str],
) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    version = manifest.get("dataset_version")
    if version != SCHEMA_VERSION:
        issues.append(
            DatasetIssue(
                "manifest_mismatch",
                f"manifest.json: dataset_version={version!r}, ожидался {SCHEMA_VERSION!r}",
                path="manifest.json",
            )
        )
    manifest_corpus = manifest.get("expected_corpus_sha256")
    if expected_corpus is not None and manifest_corpus != expected_corpus:
        issues.append(
            DatasetIssue(
                "manifest_mismatch",
                "manifest.json: expected_corpus_sha256 не совпадает с suite-файлами",
                path="manifest.json",
            )
        )
    manifest_hashes = manifest.get("suite_hashes")
    if not isinstance(manifest_hashes, Mapping):
        issues.append(
            DatasetIssue(
                "manifest_mismatch",
                "manifest.json: suite_hashes обязателен",
                path="manifest.json",
            )
        )
    else:
        for suite, digest in hashes.items():
            if manifest_hashes.get(suite) != digest:
                issues.append(
                    DatasetIssue(
                        "manifest_mismatch",
                        f"manifest.json: hash suite {suite} не совпадает с файлом",
                        path="manifest.json",
                    )
                )
    if chunk_count is not None:
        declared = manifest.get("chunk_count")
        if declared != chunk_count:
            issues.append(
                DatasetIssue(
                    "manifest_mismatch",
                    f"manifest.json: chunk_count={declared!r}, в chunk store {chunk_count}",
                    path="manifest.json",
                )
            )
    declared_docs = manifest.get("documents")
    if declared_docs is not None and chunk_count is not None:
        if not isinstance(declared_docs, list) or sorted(declared_docs) != sorted(documents):
            issues.append(
                DatasetIssue(
                    "manifest_mismatch",
                    "manifest.json: documents не совпадает с документами корпуса",
                    path="manifest.json",
                )
            )
    if "partial" in manifest and not isinstance(manifest.get("partial"), bool):
        issues.append(
            DatasetIssue(
                "manifest_mismatch",
                "manifest.json: partial должен быть булевым",
                path="manifest.json",
            )
        )
    return issues


def validate_dataset(
    datasets_dir: Path | str,
    *,
    chunks_dir: Path | str | None = None,
    chunks: Sequence[Mapping[str, Any]] | None = None,
    check_corpus: bool = True,
    require_manifest: bool = False,
) -> DatasetBundle:
    """Загрузить и проверить набор по §3.9.

    Args:
        datasets_dir: Каталог с suite-файлами и ``manifest.json``.
        chunks_dir: Каталог chunk store; игнорируется, если передан ``chunks``.
        chunks: Готовый список чанков (например, из тестовой фикстуры).
        check_corpus: Проверять существование ``chunk_id`` и content hash.
        require_manifest: Требовать ``manifest.json``; для полного набора — да.

    Raises:
        DatasetError: С кодом ``invalid_dataset`` или ``dataset_corpus_mismatch``.
    """
    directory = Path(datasets_dir)
    if not directory.is_dir():
        raise DatasetError(
            "datasets_dir_missing", f"Каталог датасета не найден: {directory}"
        )

    issues: list[DatasetIssue] = []
    suites: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for suite in SUITE_NAMES:
        path = directory / f"{suite}.json"
        if not path.exists():
            issues.append(
                DatasetIssue(
                    "suite_missing",
                    f"{suite}.json отсутствует",
                    path=path.name,
                )
            )
            continue
        payload = load_suite(path)
        suites[suite] = payload
        hashes[suite] = suite_hash(payload)
        if payload.get("schema_version") != SCHEMA_VERSION:
            issues.append(
                DatasetIssue(
                    "schema_version_invalid",
                    f"{path.name}: schema_version должен быть {SCHEMA_VERSION}",
                    path=path.name,
                )
            )
        if payload.get("suite") != suite:
            issues.append(
                DatasetIssue(
                    "suite_mismatch",
                    f"{path.name}: suite={payload.get('suite')!r} не совпадает с файлом",
                    path=path.name,
                )
            )
        corpus_hash = payload.get("expected_corpus_sha256")
        if not isinstance(corpus_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_hash):
            issues.append(
                DatasetIssue(
                    "corpus_hash_invalid",
                    f"{path.name}: expected_corpus_sha256 должен быть sha256 hex",
                    path=path.name,
                )
            )
        tests = payload.get("tests")
        if not isinstance(tests, list):
            issues.append(
                DatasetIssue(
                    "tests_invalid", f"{path.name}: tests должен быть массивом", path=path.name
                )
            )
            payload["tests"] = []

    if len(suites) != len(SUITE_NAMES):
        raise DatasetError(
            "invalid_dataset",
            "Неполный набор suite-файлов",
            issues=issues,
        )

    expected_hashes = {
        payload.get("expected_corpus_sha256")
        for payload in suites.values()
    }
    if len(expected_hashes) != 1:
        issues.append(
            DatasetIssue(
                "corpus_hash_mismatch",
                "suite-файлы содержат разные expected_corpus_sha256",
            )
        )
    expected_corpus = next(iter(expected_hashes)) if len(expected_hashes) == 1 else None

    store: dict[str, dict[str, Any]] = {}
    current_hash: str | None = None
    chunk_count: int | None = None
    documents: list[str] = []
    years: set[int] = set()
    if check_corpus:
        if chunks is None:
            if chunks_dir is None:
                raise DatasetError(
                    "chunk_store_missing",
                    "Для проверки chunk_id нужен chunks_dir или chunks",
                    issues=issues,
                )
            chunks = load_chunks(chunks_dir)
        store = chunk_store(chunks)
        current_hash = corpus_content_sha256(chunks)
        chunk_count = len(store)
        documents = sorted(corpus_documents(store.values()))
        years = corpus_years(store.values())

    if expected_corpus and current_hash and expected_corpus != current_hash:
        issues.append(
            DatasetIssue(
                "dataset_corpus_mismatch",
                "expected_corpus_sha256 не совпадает с текущим content hash корпуса",
            )
        )

    tests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for suite in SUITE_NAMES:
        payload = suites[suite]
        for test in payload.get("tests", []) or []:
            if not isinstance(test, dict):
                issues.append(
                    DatasetIssue(
                        "test_invalid",
                        f"{suite}.json: тест должен быть объектом",
                        path=f"{suite}.json",
                    )
                )
                continue
            test_id = test.get("id")
            if isinstance(test_id, str):
                if test_id in seen_ids:
                    issues.append(
                        DatasetIssue(
                            "duplicate_test_id",
                            f"Повторяющийся id: {test_id}",
                            test_id=test_id,
                        )
                    )
                seen_ids.add(test_id)
            tests.append(test)
            issues.extend(
                _test_issues(
                    test,
                    suite=suite,
                    path_name=f"{suite}.json",
                    chunks=store,
                    corpus_years=years,
                    check_chunks=check_corpus,
                )
            )
            question_values = (
                [test.get("question")]
                if suite != "multi_turn"
                else [turn.get("question") for turn in test.get("turns", []) or []
                      if isinstance(turn, Mapping)]
            )
            for question in question_values:
                normalized = _normalized_question(question)
                if not normalized:
                    continue
                if normalized in seen_questions:
                    issues.append(
                        DatasetIssue(
                            "duplicate_question",
                            f"Повторяющийся вопрос: {question!r}",
                            test_id=test_id if isinstance(test_id, str) else None,
                        )
                    )
                else:
                    seen_questions[normalized] = (
                        test_id if isinstance(test_id, str) else "<без id>"
                    )

    # Некритичные замечания: не отменяют запуск, но фиксируются в bundle.
    dataset_warnings = _holdout_overlap_warnings(tests)

    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(
                "manifest_invalid", f"manifest.json: {exc}", issues=issues
            ) from exc
        if not isinstance(manifest, dict):
            raise DatasetError(
                "manifest_invalid",
                "manifest.json: корень должен быть объектом",
                issues=issues,
            )
        issues.extend(
            _manifest_issues(
                manifest,
                suites=suites,
                hashes=hashes,
                expected_corpus=expected_corpus,
                chunk_count=chunk_count,
                documents=documents,
            )
        )
        partial = bool(manifest.get("partial"))
        if not partial:
            if len(tests) != 30:
                issues.append(
                    DatasetIssue(
                        "dataset_composition",
                        f"Полный набор должен содержать 30 тестов, найдено {len(tests)}",
                    )
                )
            split_counts = {
                split: sum(1 for test in tests if test.get("split") == split)
                for split in SPLITS
            }
            if split_counts["holdout"] < 6:
                issues.append(
                    DatasetIssue(
                        "dataset_composition",
                        "Полный набор требует не менее 6 holdout-тестов",
                    )
                )
            smoke_count = sum(
                1 for test in tests if "smoke" in (test.get("tags") or [])
            )
            if smoke_count != 12:
                issues.append(
                    DatasetIssue(
                        "dataset_composition",
                        f"Полный набор требует 12 smoke-тестов, найдено {smoke_count}",
                    )
                )
    elif require_manifest:
        issues.append(
            DatasetIssue(
                "manifest_missing",
                "manifest.json отсутствует",
                path="manifest.json",
            )
        )

    if issues:
        code = "dataset_corpus_mismatch" if any(
            issue.code == "dataset_corpus_mismatch" for issue in issues
        ) else "invalid_dataset"
        raise DatasetError(code, "Эталонный набор не прошёл валидацию", issues=issues)

    return DatasetBundle(
        datasets_dir=directory,
        suites=suites,
        tests=tests,
        manifest=manifest,
        suite_hashes=hashes,
        dataset_sha256=dataset_hash(suites),
        expected_corpus_sha256=expected_corpus,
        chunk_count=chunk_count,
        documents=documents,
        issues=[],
        warnings=dataset_warnings,
    )


def load_datasets(
    datasets_dir: Path | str | None = None,
    *,
    chunks_dir: Path | str | None = None,
    check_corpus: bool = True,
    require_manifest: bool = False,
) -> DatasetBundle:
    """Загрузить набор из каталога по умолчанию ``eval/datasets``."""
    root = Path(datasets_dir) if datasets_dir is not None else Path(__file__).parent / "datasets"
    return validate_dataset(
        root,
        chunks_dir=chunks_dir,
        check_corpus=check_corpus,
        require_manifest=require_manifest,
    )
