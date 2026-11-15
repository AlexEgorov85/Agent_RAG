"""Офлайн-тесты схемы, валидатора и хешей benchmark-датасета."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval.dataset import (
    DatasetError,
    SUITE_NAMES,
    corpus_content_sha256,
    dataset_hash,
    suite_hash,
    validate_dataset,
)

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT / "eval" / "datasets"
DEV_CHUNKS = PROJECT / "data" / "dev" / "chunks"


def make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "sber_2023:00001:c0:aaaaaaaaaaaa",
            "doc_id": "sber_2023",
            "year": 2023,
            "page_pdf": 8,
            "text": "Чистая прибыль за 2023 год составила 1 508,6 млрд рублей.",
            "source_sha256": "a" * 64,
        },
        {
            "chunk_id": "sber_2022:00002:c0:bbbbbbbbbbbb",
            "doc_id": "sber_2022",
            "year": 2022,
            "page_pdf": 30,
            "text": "В 2022 году Сбер продолжал развивать экосистему.",
            "source_sha256": "b" * 64,
        },
    ]


def valid_test(test_id: str = "easy_001", *, question: str | None = None) -> dict:
    return {
        "id": test_id,
        "suite": "easy",
        "split": "dev",
        "category": "fact_lookup",
        "difficulty": "easy",
        "tags": ["smoke"],
        "question": question or "Какая чистая прибыль за 2023 год?",
        "filters": {"years": [2023], "allowed_doc_ids": ["sber_2023"]},
        "expected_status": "answer",
        "expected": {
            "required_facts": [
                {
                    "metric": "net_profit",
                    "metric_aliases": ["чистая прибыль"],
                    "year": 2023,
                    "value": 1508.6,
                    "unit": "млрд руб.",
                    "tolerance": 0.01,
                }
            ],
            "required_years": [2023],
            "required_documents": ["sber_2023"],
            "required_answer_elements": ["значение"],
            "required_calculations": [],
            "required_structural_changes": [],
            "missing_years": [],
        },
        "gold_evidence": {
            "min_coverage": 1.0,
            "groups": [
                {
                    "id": "profit_2023",
                    "any_of": ["sber_2023:00001:c0:aaaaaaaaaaaa"],
                }
            ],
        },
        "expected_behavior": {
            "retrieval_modes": ["hybrid"],
            "max_search_calls": 2,
            "max_tool_calls": 4,
            "must_use_tools": ["search_reports"],
            "must_use_tools_strict": False,
            "may_use_tools": ["get_chunk"],
            "must_not_use_tools": [],
            "strict_search_budget": False,
            "must_cite_sources": True,
            "must_abstain_if_missing": True,
        },
        "checks": {
            "must_mention": ["чистая прибыль"],
            "must_not_mention": [],
            "forbidden_values": [],
            "forbidden_years": [],
            "expected_warnings": [],
            "must_not_invent_values": True,
            "strict_quotes": False,
        },
        "notes": "Синтетический валидный тест для офлайн-проверки схемы.",
    }


def write_dataset(
    directory: Path,
    *,
    tests: dict[str, list[dict]] | None = None,
    chunks: list[dict] | None = None,
    partial: bool = True,
    manifest: bool = True,
    corpus_hash: str | None = None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    chunks = chunks or make_chunks()
    expected = corpus_hash or corpus_content_sha256(chunks)
    by_suite = {suite: [] for suite in SUITE_NAMES}
    for suite, rows in (tests or {"easy": [valid_test()]}).items():
        by_suite[suite] = rows
    payloads = {
        suite: {
            "schema_version": "1.1",
            "suite": suite,
            "expected_corpus_sha256": expected,
            "tests": by_suite[suite],
        }
        for suite in SUITE_NAMES
    }
    for suite, payload in payloads.items():
        (directory / f"{suite}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    hashes = {suite: suite_hash(payload) for suite, payload in payloads.items()}
    if manifest:
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_version": "1.1",
                    "partial": partial,
                    "expected_corpus_sha256": expected,
                    "chunk_count": len(chunks),
                    "documents": sorted({chunk["doc_id"] for chunk in chunks}),
                    "suite_hashes": hashes,
                    "notes": "offline fixture",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {"payloads": payloads, "hashes": hashes, "chunks": chunks}


def rewrite_suite(directory: Path, suite: str, mutate) -> None:
    path = directory / f"{suite}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_shipped_full_dataset_is_valid_and_honest():
    if not DEV_CHUNKS.exists():
        pytest.skip("dev chunk store отсутствует в этом окружении")
    bundle = validate_dataset(DATASETS, chunks_dir=DEV_CHUNKS, require_manifest=True)
    assert len(bundle.tests) == 30
    counts = {
        suite: len(bundle.tests_for_suite(suite))
        for suite in SUITE_NAMES
    }
    assert counts == {"easy": 10, "medium": 10, "hard": 5, "multi_turn": 5}
    smoke = [test for test in bundle.tests if "smoke" in (test.get("tags") or [])]
    holdout = [test for test in bundle.tests if test.get("split") == "holdout"]
    assert len(smoke) == 12
    assert all(test.get("split") == "dev" for test in smoke)
    assert len(holdout) >= 6
    assert not any("smoke" in (test.get("tags") or []) for test in holdout)
    assert bundle.manifest["partial"] is False
    assert "Полный стартовый набор 30 тестов" in bundle.manifest["notes"]
    assert bundle.expected_corpus_sha256 == bundle.manifest["expected_corpus_sha256"]
    assert bundle.manifest["chunk_count"] == 1025
    assert bundle.manifest["documents"] == ["sber_2016", "sber_2017", "sber_2022", "sber_2023"]
    assert bundle.manifest["corpus_fingerprint"] == (
        "26f393fef970297dda2eb4af93a1df29233e7084af60983d2ff074710924a549"
    )
    assert bundle.dataset_sha256 == dataset_hash(bundle.suites)
    assert bundle.dataset_sha256 == bundle.manifest["dataset_sha256"]
    assert bundle.suite_hashes == bundle.manifest["suite_hashes"]
    # Holdout-gold пересекается с dev-gold: это warning, а не ошибка схемы,
    # но он обязан быть виден потребителю bundle.
    assert any(
        issue.code == "holdout_gold_overlap" for issue in bundle.warnings
    )


def test_valid_fixture_passes_and_hash_is_stable(tmp_path: Path):
    write_dataset(tmp_path)
    first = validate_dataset(
        tmp_path, chunks=make_chunks(), require_manifest=True
    )
    second = validate_dataset(
        tmp_path, chunks=make_chunks(), require_manifest=True
    )
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.dataset_sha256 == dataset_hash(first.suites)
    assert first.suite_hashes["easy"] == suite_hash(first.suites["easy"])


def test_unknown_chunk_id_is_rejected(tmp_path: Path):
    bad = valid_test()
    bad["gold_evidence"]["groups"][0]["any_of"] = ["sber_2023:fantom"]
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert error.value.code == "invalid_dataset"
    assert any(issue.code == "unknown_chunk_id" for issue in error.value.issues)


def test_duplicate_test_id_is_rejected(tmp_path: Path):
    write_dataset(
        tmp_path,
        tests={"easy": [valid_test(), valid_test(question="Другой вопрос?")]},
    )
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "duplicate_test_id" for issue in error.value.issues)


def test_duplicate_question_after_normalization_is_rejected(tmp_path: Path):
    first = valid_test()
    second = valid_test("easy_002", question="  КАКАЯ   чистая прибыль за 2023 год? ")
    write_dataset(tmp_path, tests={"easy": [first, second]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "duplicate_question" for issue in error.value.issues)


def test_answer_test_requires_gold_evidence(tmp_path: Path):
    bad = valid_test()
    bad["gold_evidence"] = {"min_coverage": 0.0, "groups": []}
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "gold_evidence_missing" for issue in error.value.issues)


def test_abstain_test_requires_reason(tmp_path: Path):
    bad = valid_test()
    bad["expected_status"] = "abstain"
    bad["expected"] = {
        "required_facts": [],
        "required_years": [],
        "required_documents": [],
        "required_answer_elements": [],
        "required_calculations": [],
        "required_structural_changes": [],
        "missing_years": [],
    }
    bad["gold_evidence"] = {"min_coverage": 0.0, "groups": []}
    bad["checks"]["expected_warnings"] = []
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "abstain_reason_missing" for issue in error.value.issues)


def test_required_year_not_in_corpus_is_rejected(tmp_path: Path):
    bad = valid_test()
    bad["expected"]["required_years"] = [2021]
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(
        issue.code == "required_year_not_in_corpus" for issue in error.value.issues
    )


def test_invalid_filters_and_tool_allowlist_are_rejected(tmp_path: Path):
    bad_filters = valid_test()
    bad_filters["filters"] = {"year_from": 2024, "year_to": 2020}
    bad_tool = valid_test("easy_002", question="Другой вопрос?")
    bad_tool["expected_behavior"]["must_not_use_tools"] = ["drop_database"]
    write_dataset(tmp_path, tests={"easy": [bad_filters, bad_tool]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    codes = {issue.code for issue in error.value.issues}
    assert "filters_invalid" in codes
    assert "tool_not_allowed" in codes


def test_numeric_fact_requires_unit(tmp_path: Path):
    bad = valid_test()
    del bad["expected"]["required_facts"][0]["unit"]
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "unit_empty" for issue in error.value.issues)


def test_strict_tools_require_nonempty_list(tmp_path: Path):
    bad = valid_test()
    bad["expected_behavior"]["must_use_tools"] = []
    bad["expected_behavior"]["must_use_tools_strict"] = True
    write_dataset(tmp_path, tests={"easy": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "behavior_invalid" for issue in error.value.issues)


def test_multi_turn_requires_two_turns(tmp_path: Path):
    turn = valid_test("multi_turn_001", question="Как менялась прибыль?")
    turn["suite"] = "multi_turn"
    turn["tags"] = ["smoke"]
    bad = {
        "id": "multi_turn_001",
        "suite": "multi_turn",
        "split": "dev",
        "category": "dialogue",
        "difficulty": "medium",
        "tags": ["smoke"],
        "turns": [turn],
        "scenario_checks": ["context_preserved"],
        "required_checks": ["context_preserved"],
    }
    write_dataset(tmp_path, tests={"multi_turn": [bad]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "turns_invalid" for issue in error.value.issues)


def test_corpus_hash_mismatch_is_reported_as_dataset_corpus_mismatch(tmp_path: Path):
    write_dataset(tmp_path, chunks=make_chunks(), corpus_hash="0" * 64)
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert error.value.code == "dataset_corpus_mismatch"
    assert any(
        issue.code == "dataset_corpus_mismatch" for issue in error.value.issues
    )


def test_manifest_suite_hash_mismatch_is_rejected(tmp_path: Path):
    write_dataset(tmp_path)
    rewrite_suite(tmp_path, "easy", lambda payload: payload["tests"].append(
        valid_test("easy_002", question="Ещё один вопрос?")
    ))
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "manifest_mismatch" for issue in error.value.issues)


def test_absolute_path_and_secret_are_rejected(tmp_path: Path):
    absolute = valid_test()
    absolute["notes"] = r"Смотри C:\secret\report.pdf"
    secret = valid_test("easy_002", question="Другой вопрос?")
    secret["notes"] = "api_key = sk-abcdefghijklmnop"
    write_dataset(tmp_path, tests={"easy": [absolute, secret]})
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    codes = {issue.code for issue in error.value.issues}
    assert "absolute_path" in codes
    assert "secret_in_dataset" in codes


def test_holdout_gold_overlap_is_a_warning_not_a_failure(tmp_path: Path):
    dev = valid_test()
    holdout = valid_test(
        "easy_002", question="Сколько составила чистая прибыль за 2023 год?"
    )
    holdout["split"] = "holdout"
    holdout["tags"] = []
    write_dataset(tmp_path, tests={"easy": [dev, holdout]})

    bundle = validate_dataset(
        tmp_path, chunks=make_chunks(), require_manifest=True
    )

    overlaps = [
        issue for issue in bundle.warnings
        if issue.code == "holdout_gold_overlap"
    ]
    assert [issue.test_id for issue in overlaps] == ["easy_002"]
    assert "easy_001" in overlaps[0].message
    assert bundle.issues == []


def test_disjoint_holdout_gold_has_no_overlap_warning(tmp_path: Path):
    dev = valid_test()
    holdout = valid_test(
        "easy_002", question="Что изменилось в экосистеме за 2022 год?"
    )
    holdout["split"] = "holdout"
    holdout["tags"] = []
    holdout["expected"] = {
        "required_facts": [],
        "required_years": [2022],
        "required_documents": ["sber_2022"],
        "required_answer_elements": [],
        "required_calculations": [],
        "required_structural_changes": [],
        "missing_years": [],
    }
    holdout["gold_evidence"] = {
        "min_coverage": 1.0,
        "groups": [
            {
                "id": "ecosystem_2022",
                "any_of": ["sber_2022:00002:c0:bbbbbbbbbbbb"],
            }
        ],
    }
    holdout["checks"]["must_mention"] = ["экосистем"]
    write_dataset(tmp_path, tests={"easy": [dev, holdout]})

    bundle = validate_dataset(
        tmp_path, chunks=make_chunks(), require_manifest=True
    )

    assert not [
        issue for issue in bundle.warnings
        if issue.code == "holdout_gold_overlap"
    ]


def test_holdout_gold_overlap_aggregates_multiturn_turns(tmp_path: Path):
    turn = valid_test("multi_turn_001", question="Прибыль за 2023?")
    turn["suite"] = "multi_turn"
    turn["split"] = "holdout"
    turn["tags"] = []
    second = copy.deepcopy(turn)
    second["question"] = "Уточни прибыль за 2023?"
    multi = {
        "id": "multi_turn_001",
        "suite": "multi_turn",
        "split": "holdout",
        "category": "dialogue",
        "difficulty": "medium",
        "tags": [],
        "turns": [turn, second],
        "scenario_checks": ["context_preserved"],
        "required_checks": ["context_preserved"],
    }
    write_dataset(
        tmp_path,
        tests={"easy": [valid_test()], "multi_turn": [multi]},
    )

    bundle = validate_dataset(
        tmp_path, chunks=make_chunks(), require_manifest=True
    )

    overlaps = [
        issue for issue in bundle.warnings
        if issue.code == "holdout_gold_overlap"
    ]
    assert [issue.test_id for issue in overlaps] == ["multi_turn_001"]
    # Один и тот же chunk в двух turn'ах не должен раздувать счётчик.
    assert "1 gold" in overlaps[0].message


def test_expected_corpus_hash_must_match_all_suites(tmp_path: Path):
    write_dataset(tmp_path)
    rewrite_suite(
        tmp_path,
        "medium",
        lambda payload: payload.__setitem__("expected_corpus_sha256", "f" * 64),
    )
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    assert any(issue.code == "corpus_hash_mismatch" for issue in error.value.issues)


def test_dataset_error_has_stable_code_and_dict(tmp_path: Path):
    with pytest.raises(DatasetError) as error:
        validate_dataset(tmp_path, chunks=make_chunks(), require_manifest=True)
    payload = error.value.as_dict()
    assert payload["code"] in {"invalid_dataset", "dataset_corpus_mismatch"}
    assert payload["issues"]
    assert copy.deepcopy(payload) == payload
