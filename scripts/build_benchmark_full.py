# -*- coding: utf-8 -*-
"""Собрать бенчмарк по полному корпусу (10 отчётов, 2015–2025).

Скрипт не просто пишет JSON: он проверяет каждый эталон по исходным чанкам —
gold-чанк существует, ожидаемое число буквально встречается в его тексте, а
год и документ совпадают с фильтром теста. Если проверка не прошла, файлы не
перезаписываются.

Результат: ``eval/datasets/full/{easy,medium,hard,multi_turn}.json`` и
``manifest.json``; запуск — ``python -m eval.runner --datasets eval/datasets/full``.

Пример::

    python scripts/build_benchmark_full.py
    python scripts/build_benchmark_full.py --check-only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from eval.dataset import (  # noqa: E402
    SCHEMA_VERSION,
    corpus_content_sha256,
    dataset_hash,
    load_chunks,
    suite_hash,
    validate_dataset,
)

OUT_DIR = PROJECT / "eval" / "datasets" / "full"
CHUNKS_DIR = PROJECT / "data" / "chunks"
INDEXES_DIR = PROJECT / "data" / "indexes"

#: Проверенные по тексту отчётов значения чистой прибыли, млрд руб.
NET_PROFIT: dict[int, tuple[float, str]] = {
    2015: (190.4, "sber_2015:00867:c0:76abcf6dad47"),
    2016: (541.9, "sber_2016:00480:c2:aadd91380839"),
    2017: (748.7, "sber_2017:00475:c2:6e59229e2a43"),
    2018: (831.7, "sber_2018:04972:c1:070d18939442"),
    2019: (845.0, "sber_2019:03567:c0:bd7785fd7bf8"),
    2020: (760.3, "sber_2020:03905:c0:3c7a954ad022"),
    2022: (270.5, "sber_2022:01988:c0:544166770eee"),
    2023: (1508.6, "sber_2023:02177:c0:1aacabe279c5"),
    2024: (1580.3, "sber_2024:02703:c1:5d346777e4b9"),
    2025: (1705.9, "sber_2025:00070:c0:00dabc46d736"),
}

#: Рентабельность капитала, %.
ROE: dict[int, tuple[float, str]] = {
    2016: (20.8, "sber_2016:00150:c0:7e5014204d8c"),
    2023: (25.3, "sber_2023:00078:c3:c5cc6e1f46e2"),
    2025: (22.7, "sber_2025:00085:c0:edf9540651f3"),
}

#: Дивиденд на акцию, руб.: год отчёта -> (значение, чанк, год, за который дивиденд).
DIVIDENDS: dict[int, tuple[float, str, int]] = {
    2017: (12.0, "sber_2017:00368:c2:df79bfbefd40", 2017),
    2024: (33.3, "sber_2024:04220:c0:ef833ca6f560", 2023),
    2025: (34.84, "sber_2025:06056:c0:4ab5ea1c02d7", 2024),
}

#: Активные розничные клиенты, млн.
CLIENTS: dict[int, tuple[float, str]] = {
    2019: (96.2, "sber_2019:00392:c0:275b7714ba8e"),
    2024: (109.9, "sber_2024:00137:c0:8e2cdd3d2dae"),
}

NET_PROFIT_UNIT = ("млрд руб.", ["млрд рублей", "млрд руб"])
ROE_UNIT = ("%", ["процент", "%"])
DIVIDEND_UNIT = ("руб.", ["рублей", "руб", "₽"])
CLIENTS_UNIT = ("млн", ["млн человек", "млн клиентов"])

#: Тесты, помеченные holdout: их gold-чанки не пересекаются с dev-набором.
HOLDOUT_IDS = {
    "medium_004", "medium_005", "medium_006", "medium_007", "medium_008",
    "hard_005", "hard_008",
}

_NON_DIGIT = re.compile(r"[^0-9,.]")


def _normalize_number(value: str | float | int) -> str:
    text = str(value)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace(",", ".")
    return re.sub(r"\s+", "", text)


def _text_numbers(text: str) -> set[str]:
    found: set[str] = set()
    for raw in re.findall(r"\d[\d\s\u00a0]{0,12}[.,]\d+", text):
        found.add(_normalize_number(raw))
    return found


def _fact(metric: str, aliases: list[str], year: int, value: float,
          unit: tuple[str, list[str]], tolerance: float = 0.01) -> dict:
    unit_name, unit_aliases = unit
    return {
        "metric": metric,
        "metric_aliases": aliases,
        "year": year,
        "value": value,
        "unit": unit_name,
        "unit_aliases": unit_aliases,
        "tolerance": tolerance,
    }


def _behavior(max_search_calls: int = 2, *, must_cite: bool = True) -> dict:
    return {
        "retrieval_modes": ["hybrid"],
        "max_search_calls": max_search_calls,
        "max_tool_calls": max_search_calls + 2,
        "must_use_tools": ["search_reports"],
        "must_use_tools_strict": False,
        "may_use_tools": ["get_chunk"],
        "must_not_use_tools": [],
        "strict_search_budget": False,
        "must_cite_sources": must_cite,
        "must_abstain_if_missing": True,
    }


def _checks(must_mention: list[str] | None = None,
            expected_warnings: list[str] | None = None) -> dict:
    return {
        "must_mention": must_mention or [],
        "must_not_mention": [],
        "forbidden_values": [],
        "forbidden_years": [],
        "expected_warnings": expected_warnings or [],
        "must_not_invent_values": True,
        "strict_quotes": False,
    }


#: Заполняются в main: чанки корпуса и эталонные значения по (год, метрика).
CHUNKS: dict[str, dict] = {}
VALUES: dict[tuple[int, str], float] = {}
_KEYWORDS = {
    "net_profit": "прибыл",
    "roe": "рентабельн",
    "dividends": "дивиденд",
    "clients": "клиент",
}


def _group(group_id: str, chunk_id: str) -> dict:
    """Группа evidence: якорный чанк плюс другие чанки с тем же фактом.

    Один показатель встречается в отчёте несколько раз (обзор, таблица,
    примечания), поэтому эталон не должен требовать конкретную страницу.
    """
    metric = next(
        (key for key in _KEYWORDS if group_id.startswith(f"{key}_")),
        group_id.split("_")[0],
    )
    chunk = CHUNKS.get(chunk_id) or {}
    year = chunk.get("year")
    value = VALUES.get((year, metric)) if isinstance(year, int) else None
    keyword = _KEYWORDS.get(metric)
    if value is None or not keyword:
        return {"id": group_id, "any_of": [chunk_id], "section": ""}
    return {
        "id": group_id,
        "any_of": _alternatives(anchor=chunk_id, value=value, keyword=keyword),
        "section": "",
    }


def _chunk_source(chunk: Mapping[str, Any]) -> str:
    """????? ????? ?????? ? html: ????????? ???????? ????? ???? ?????? ? html."""
    return f"{chunk.get('text') or ''} {chunk.get('html') or ''}"


def _alternatives(
    *,
    anchor: str,
    value: float,
    keyword: str,
    chunks: dict[str, dict] | None = None,
    limit: int = 12,
) -> list[str]:
    """Все чанки того же документа и года, где есть это число и ключевое слово.

    Один и тот же показатель в отчёте встречается несколько раз (обзор, таблица,
    примечания). Эталон не должен требовать конкретную страницу: достаточно,
    чтобы факт был найден в любом из этих чанков.
    """
    chunks = chunks if chunks is not None else CHUNKS
    anchor_chunk = chunks.get(anchor) or {}
    doc_id = anchor_chunk.get("doc_id")
    year = anchor_chunk.get("year")
    number = _normalize_number(value)
    found: list[str] = [anchor]
    for chunk_id, chunk in chunks.items():
        if chunk_id == anchor:
            continue
        if chunk.get("doc_id") != doc_id or chunk.get("year") != year:
            continue
        text = _chunk_source(chunk)
        if keyword.casefold() not in text.casefold():
            continue
        if number not in _text_numbers(text):
            continue
        found.append(chunk_id)
        if len(found) >= limit:
            break
    return found


def _test(
    *,
    test_id: str,
    suite: str,
    category: str,
    difficulty: str,
    question: str,
    reference: str,
    years: list[int],
    doc_ids: list[str],
    facts: list[dict],
    gold: list[dict],
    tags: list[str],
    must_mention: list[str] | None = None,
    max_search_calls: int = 3,
    notes: str = "",
    status: str = "answer",
    missing_years: list[int] | None = None,
    expected_warnings: list[str] | None = None,
) -> dict:
    filters: dict = {"years": years} if years else {}
    if doc_ids:
        filters["allowed_doc_ids"] = doc_ids
    return {
        "id": test_id,
        "suite": suite,
        "split": "holdout" if test_id in HOLDOUT_IDS else "dev",
        "category": category,
        "difficulty": difficulty,
        "tags": tags,
        "question": question,
        "filters": filters,
        "expected_status": status,
        "reference_answer": reference,
        "expected": {
            "required_facts": facts,
            "required_years": years,
            "required_documents": doc_ids,
            "required_issuers": [],
            "required_answer_elements": [],
            "required_calculations": [],
            "required_structural_changes": [],
            "missing_years": missing_years or [],
        },
        "gold_evidence": {
            "min_coverage": 1.0 if gold else 0.0,
            "groups": gold,
        },
        "expected_behavior": _behavior(
            max_search_calls=max_search_calls,
            must_cite=status == "answer",
        ),
        "checks": _checks(must_mention=must_mention, expected_warnings=expected_warnings),
        "notes": notes,
    }


def _profit_fact(year: int) -> dict:
    return _fact("net_profit", ["чистая прибыль"], year, NET_PROFIT[year][0], NET_PROFIT_UNIT)


def _roe_fact(year: int) -> dict:
    return _fact("roe", ["рентабельность капитала", "ROE"], year, ROE[year][0], ROE_UNIT)


def build_easy() -> list[dict]:
    tests: list[dict] = []
    # 10 фактов чистой прибыли по годам (2015 вынесен в hard как holdout).
    for year in (2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024, 2025):
        value, chunk_id = NET_PROFIT[year]
        tests.append(_test(
            test_id=f"easy_{len(tests) + 1:03d}",
            suite="easy",
            category="fact_lookup",
            difficulty="easy",
            question=f"Какая чистая прибыль Группы по МСФО за {year} год?",
            reference=f"{value:,.1f}".replace(",", " ").replace(".", ",") + " млрд рублей.",
            years=[year],
            doc_ids=[f"sber_{year}"],
            facts=[_profit_fact(year)],
            gold=[_group(f"net_profit_{year}", chunk_id)],
            tags=["smoke", "numeric", "net_profit"],
            must_mention=[str(year)],
            max_search_calls=2,
        ))
    # ROE по трём годам.
    for index, year in enumerate((2016, 2023, 2025)):
        value, chunk_id = ROE[year]
        tests.append(_test(
            test_id=f"easy_{len(tests) + 1:03d}",
            suite="easy",
            category="fact_lookup",
            difficulty="easy",
            question=f"Какая рентабельность капитала (ROE) была у Группы в {year} году?",
            reference=f"{value:.1f}".replace(".", ",") + "%.",
            years=[year],
            doc_ids=[f"sber_{year}"],
            facts=[_roe_fact(year)],
            gold=[_group(f"roe_{year}", chunk_id)],
            tags=["smoke", "numeric", "roe"],
            must_mention=[str(year)],
            max_search_calls=2,
        ))
    return tests


def build_medium() -> list[dict]:
    tests: list[dict] = []
    for first, second in ((2019, 2020), (2023, 2025)):
        doc_first, doc_second = f"sber_{first}", f"sber_{second}"
        tests.append(_test(
            test_id=f"medium_{len(tests) + 1:03d}",
            suite="medium",
            category="temporal",
            difficulty="medium",
            question=f"Как изменилась чистая прибыль Группы с {first} по {second} год?",
            reference=(
                f"С {NET_PROFIT[first][0]:,.1f} млрд рублей в {first} году до "
                f"{NET_PROFIT[second][0]:,.1f} млрд рублей в {second} году."
            ).replace(",", " ").replace(".", ","),
            years=[first, second],
            doc_ids=[doc_first, doc_second],
            facts=[_profit_fact(first), _profit_fact(second)],
            gold=[
                _group(f"net_profit_{first}", NET_PROFIT[first][1]),
                _group(f"net_profit_{second}", NET_PROFIT[second][1]),
            ],
            tags=["comparison", "net_profit"],
            must_mention=[str(first), str(second)],
        ))
    for year in (2023,):
        tests.append(_test(
            test_id=f"medium_{len(tests) + 1:03d}",
            suite="medium",
            category="temporal",
            difficulty="medium",
            question=f"Как изменилась рентабельность капитала и чистая прибыль с 2016 по {year} год?",
            reference=(
                f"ROE выросла с 20,8% в 2016 году до {ROE[year][0]:.1f}".replace(".", ",")
                + f"% в {year} году, чистая прибыль — с 541,9 до "
                + f"{NET_PROFIT[year][0]:,.1f}".replace(",", " ").replace(".", ",")
                + " млрд рублей."
            ),
            years=[2016, year],
            doc_ids=["sber_2016", f"sber_{year}"],
            facts=[_roe_fact(2016), _roe_fact(year)],
            gold=[
                _group(f"roe_2016", ROE[2016][1]),
                _group(f"roe_{year}", ROE[year][1]),
                _group(f"net_profit_{year}", NET_PROFIT[year][1]),
            ],
            tags=["comparison", "roe"],
            must_mention=["2016", str(year)],
        ))
    for report_year, (value, chunk_id, dividend_year) in DIVIDENDS.items():
        tests.append(_test(
            test_id=f"medium_{len(tests) + 1:03d}",
            suite="medium",
            category="fact_lookup",
            difficulty="medium",
            question=f"Какой размер дивиденда на акцию был объявлен по итогам {dividend_year} года?",
            reference=f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",") + " рубля на акцию.",
            years=[report_year],
            doc_ids=[f"sber_{report_year}"],
            facts=[_fact("dividends_per_share", ["дивиденд на акцию", "дивиденды на акцию"],
                         dividend_year, value, DIVIDEND_UNIT, tolerance=0.005)],
            gold=[_group(f"dividends_{dividend_year}", chunk_id)],
            tags=["dividends", "numeric"],
            must_mention=[str(dividend_year)],
            max_search_calls=2,
        ))
    for year, (value, chunk_id) in CLIENTS.items():
        tests.append(_test(
            test_id=f"medium_{len(tests) + 1:03d}",
            suite="medium",
            category="fact_lookup",
            difficulty="medium",
            question=f"Сколько активных розничных клиентов было у Сбера в {year} году?",
            reference=f"{value:.1f}".replace(".", ",") + " млн клиентов.",
            years=[year],
            doc_ids=[f"sber_{year}"],
            facts=[_fact("retail_clients", ["розничные клиенты", "активные клиенты"],
                         year, value, CLIENTS_UNIT, tolerance=0.05)],
            gold=[_group(f"clients_{year}", chunk_id)],
            tags=["clients", "numeric"],
            must_mention=[str(year)],
            max_search_calls=2,
        ))
    return tests


def build_hard() -> list[dict]:
    tests: list[dict] = [
        _test(
            test_id="hard_001",
            suite="hard",
            category="abstain",
            difficulty="hard",
            question="Какая чистая прибыль Сбербанка за 2021 год?",
            reference="Отчёта за 2021 год в корпусе нет, ответить нельзя.",
            years=[2021],
            doc_ids=[],
            facts=[],
            gold=[],
            tags=["abstain", "gap_year"],
            status="abstain",
            missing_years=[2021],
            expected_warnings=["missing_year"],
            max_search_calls=1,
            notes="2021 год физически отсутствует в корпусе.",
        ),
        _test(
            test_id="hard_002",
            suite="hard",
            category="abstain",
            difficulty="hard",
            question="Какие дивиденды на акцию были выплачены по итогам 2021 года?",
            reference="Отчёта за 2021 год в корпусе нет, ответить нельзя.",
            years=[2021],
            doc_ids=[],
            facts=[],
            gold=[],
            tags=["abstain", "gap_year", "dividends"],
            status="abstain",
            missing_years=[2021],
            expected_warnings=["missing_year"],
            max_search_calls=1,
        ),
        _test(
            test_id="hard_003",
            suite="hard",
            category="multi_hop",
            difficulty="hard",
            question="Во сколько раз выросла чистая прибыль Группы с 2020 по 2025 год?",
            reference="С 760,3 до 1 705,9 млрд рублей, примерно в 2,24 раза.",
            years=[2020, 2025],
            doc_ids=["sber_2020", "sber_2025"],
            facts=[_profit_fact(2020), _profit_fact(2025)],
            gold=[_group("net_profit_2020", NET_PROFIT[2020][1]),
                  _group("net_profit_2025", NET_PROFIT[2025][1])],
            tags=["comparison", "net_profit", "growth"],
            must_mention=["2020", "2025"],
        ),
        _test(
            test_id="hard_004",
            suite="hard",
            category="multi_hop",
            difficulty="hard",
            question="Сравни чистую прибыль за 2022 и 2023 годы: во сколько раз она изменилась?",
            reference="С 270,5 млрд рублей в 2022 году до 1 508,6 млрд рублей в 2023-м, в 5,2 раза.",
            years=[2022, 2023],
            doc_ids=["sber_2022", "sber_2023"],
            facts=[_profit_fact(2022), _profit_fact(2023)],
            gold=[_group("net_profit_2022", NET_PROFIT[2022][1]),
                  _group("net_profit_2023", NET_PROFIT[2023][1])],
            tags=["comparison", "net_profit"],
            must_mention=["2022", "2023"],
        ),
        _test(
            test_id="hard_005",
            suite="hard",
            category="fact_lookup",
            difficulty="hard",
            question="Какая чистая прибыль БПС-Сбербанка за 2015 год?",
            reference="190,4 млрд белорусских рублей.",
            years=[2015],
            doc_ids=["sber_2015"],
            facts=[_fact("net_profit", ["чистая прибыль"], 2015, 190.4,
                         ("млрд BYR", ["млрд белорусских рублей", "BYR"]))],
            gold=[_group("net_profit_2015", NET_PROFIT[2015][1])],
            tags=["numeric", "net_profit", "issuer"],
            must_mention=["2015"],
            notes="2015 год — отчёт БПС-Сбербанка (Беларусь), несопоставим по эмитенту.",
        ),
        _test(
            test_id="hard_006",
            suite="hard",
            category="multi_hop",
            difficulty="hard",
            question="На сколько процентов выросла чистая прибыль Группы с 2023 по 2024 год?",
            reference="С 1 508,6 до 1 580,3 млрд рублей, рост на 4,8%.",
            years=[2023, 2024],
            doc_ids=["sber_2023", "sber_2024"],
            facts=[_profit_fact(2023), _profit_fact(2024)],
            gold=[_group("net_profit_2023", NET_PROFIT[2023][1]),
                  _group("net_profit_2024", NET_PROFIT[2024][1])],
            tags=["comparison", "net_profit"],
            must_mention=["2023", "2024"],
        ),
        _test(
            test_id="hard_007",
            suite="hard",
            category="multi_hop",
            difficulty="hard",
            question="Как изменились чистая прибыль и ROE с 2023 по 2025 год?",
            reference="Прибыль выросла с 1 508,6 до 1 705,9 млрд рублей, ROE снизилась с 25,3% до 22,7%.",
            years=[2023, 2025],
            doc_ids=["sber_2023", "sber_2025"],
            facts=[_profit_fact(2023), _profit_fact(2025), _roe_fact(2023), _roe_fact(2025)],
            gold=[_group("net_profit_2023", NET_PROFIT[2023][1]),
                  _group("net_profit_2025", NET_PROFIT[2025][1]),
                  _group("roe_2023", ROE[2023][1]),
                  _group("roe_2025", ROE[2025][1])],
            tags=["comparison", "net_profit", "roe"],
            must_mention=["2023", "2025"],
            max_search_calls=4,
        ),
        _test(
            test_id="hard_008",
            suite="hard",
            category="multi_hop",
            difficulty="hard",
            question="Как изменилось число активных розничных клиентов с 2019 по 2024 год?",
            reference="С 96,2 млн в 2019 году до 109,9 млн в 2024 году.",
            years=[2019, 2024],
            doc_ids=["sber_2019", "sber_2024"],
            facts=[
                _fact("retail_clients", ["розничные клиенты", "активные клиенты"],
                      2019, CLIENTS[2019][0], CLIENTS_UNIT, tolerance=0.05),
                _fact("retail_clients", ["розничные клиенты", "активные клиенты"],
                      2024, CLIENTS[2024][0], CLIENTS_UNIT, tolerance=0.05),
            ],
            gold=[_group("clients_2019", CLIENTS[2019][1]),
                  _group("clients_2024", CLIENTS[2024][1])],
            tags=["comparison", "clients"],
            must_mention=["2019", "2024"],
        ),
    ]
    return tests


def build_multi_turn() -> list[dict]:
    """??????? ?? ????????? ?????????: ?????? ?????? ?????? ?????? ???."""
    scenarios = [
        ("multi_turn_001", "net_profit", "?????? ???????", "???????", 2025, 2024),
        ("multi_turn_002", "roe", "?????????????? ????????", "ROE", 2025, 2023),
    ]
    tests: list[dict] = []
    for test_id, metric, metric_label, short_label, first, second in scenarios:
        fact_first = (_profit_fact(first) if metric == "net_profit" else _roe_fact(first))
        fact_second = (_profit_fact(second) if metric == "net_profit" else _roe_fact(second))
        chunk_first = (NET_PROFIT if metric == "net_profit" else ROE)[first][1]
        chunk_second = (NET_PROFIT if metric == "net_profit" else ROE)[second][1]
        tests.append({
            "id": test_id,
            "suite": "multi_turn",
            "split": "dev",
            "category": "dialogue",
            "difficulty": "medium",
            "tags": ["context", "numeric", short_label],
            "dialogue_policy": {"follow_up_year": "reuse_topic"},
            "turns": [
                {
                    "question": f"????? {metric_label} ? ????????? ?? {first} ????",
                    "behavior": "continue_previous_request",
                    "expected_status": "answer",
                    "expected": {
                        "required_facts": [fact_first],
                        "required_years": [first],
                        "required_documents": [f"sber_{first}"],
                        "required_issuers": [],
                        "required_answer_elements": [],
                        "required_calculations": [],
                        "required_structural_changes": [],
                        "missing_years": [],
                    },
                    "gold_evidence": {
                        "min_coverage": 1.0,
                        "groups": [_group(f"{metric}_{first}", chunk_first)],
                    },
                    "expected_behavior": _behavior(max_search_calls=2),
                    "checks": _checks(must_mention=[str(first)]),
                },
                {
                    "question": f"? ?? {second} ????",
                    "behavior": "continue_previous_request",
                    "expected_status": "answer",
                    "expected": {
                        "required_facts": [fact_second],
                        "required_years": [second],
                        "required_documents": [f"sber_{second}"],
                        "required_issuers": [],
                        "required_answer_elements": [],
                        "required_calculations": [],
                        "required_structural_changes": [],
                        "missing_years": [],
                    },
                    "gold_evidence": {
                        "min_coverage": 1.0,
                        "groups": [_group(f"{metric}_{second}", chunk_second)],
                    },
                    "expected_behavior": _behavior(max_search_calls=2),
                    "checks": _checks(must_mention=[str(second)]),
                },
            ],
            "scenario_checks": [
                "context_preserved", "correct_metric", "correct_period",
                "no_unnecessary_restart", "sources_preserved",
            ],
            "required_checks": [
                "context_preserved", "correct_metric", "correct_period",
                "no_unnecessary_restart", "sources_preserved",
            ],
            "notes": "?????? ??? ?????? ?????? ???: ??????? ?????? ??????????? ?? ?????????.",
        })
    return tests

def verify_tests(tests: list[dict], chunks: dict[str, dict]) -> list[str]:
    """Проверить, что эталон опирается на реальные чанки и их текст."""
    problems: list[str] = []
    for test in tests:
        if test.get("suite") == "multi_turn":
            turns = [(f"{test['id']}#{index}", turn) for index, turn in enumerate(test["turns"])]
        else:
            turns = [(test["id"], test)]
        for where, turn in turns:
            groups = (turn.get("gold_evidence") or {}).get("groups") or []
            gold_chunks: list[tuple[str, dict]] = []
            for group in groups:
                if not isinstance(group.get("section"), str):
                    problems.append(f"{where}: section должен быть строкой")
                for chunk_id in group.get("any_of") or []:
                    chunk = chunks.get(chunk_id)
                    if chunk is None:
                        problems.append(f"{where}: нет чанка {chunk_id}")
                        continue
                    gold_chunks.append((chunk_id, chunk))
                    filters = turn.get("filters") or {}
                    years = filters.get("years")
                    if years and chunk.get("year") not in years:
                        problems.append(
                            f"{where}: год чанка {chunk.get('year')} вне фильтра {years}")
                    docs = filters.get("allowed_doc_ids")
                    if docs and chunk.get("doc_id") not in docs:
                        problems.append(
                            f"{where}: документ {chunk.get('doc_id')} вне фильтра {docs}")
            for fact in (turn.get("expected") or {}).get("required_facts") or []:
                expected = _normalize_number(fact["value"])
                candidates = [
                    (chunk_id, chunk) for chunk_id, chunk in gold_chunks
                    if chunk.get("year") == fact["year"]
                ] or gold_chunks
                if not candidates:
                    problems.append(f"{where}: у факта {fact['metric']} нет gold-чанка")
                    continue
                if not any(expected in _text_numbers(_chunk_source(chunk))
                           for _, chunk in candidates):
                    problems.append(
                        f"{where}: значение {fact['value']} отсутствует в тексте "
                        + ", ".join(chunk_id for chunk_id, _ in candidates)
                    )
            status = turn.get("expected_status")
            if status == "abstain" and groups:
                problems.append(f"{where}: у abstain-теста не должно быть gold-чанков")
            if status == "answer" and not groups:
                problems.append(f"{where}: у answer-теста нет gold-чанков")
    return problems


def build_manifest(*, suites: dict[str, dict], chunks: list[dict], corpus_hash: str) -> dict:
    latest_index = max(
        (path for path in INDEXES_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    manifest_path = latest_index / "manifest.json"
    fingerprint_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = sorted({str(chunk.get("doc_id")) for chunk in chunks if chunk.get("doc_id")})
    return {
        "dataset_version": SCHEMA_VERSION,
        "status": "full_corpus",
        "partial": False,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "expected_corpus_sha256": corpus_hash,
        "corpus_content_sha256": corpus_hash,
        "corpus_fingerprint": fingerprint_payload.get("corpus_fingerprint", latest_index.name),
        "corpus_fingerprint_source": str(manifest_path.relative_to(PROJECT)).replace("\\", "/"),
        "chunk_count": len(chunks),
        "documents": documents,
        "suite_hashes": {name: suite_hash(payload) for name, payload in suites.items()},
        "dataset_sha256": dataset_hash(suites),
        "notes": (
            "Бенчмарк по полному корпусу: 10 отчётов (2015–2020, 2022–2025), "
            f"{len(chunks)} чанков. Вопросы составлены по текстам отчётов; каждое "
            "ожидаемое значение проверено на дословное присутствие в gold-чанке "
            "скриптом scripts/build_benchmark_full.py."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать бенчмарк по полному корпусу")
    parser.add_argument("--check-only", action="store_true",
                        help="только проверить эталон, не писать файлы")
    args = parser.parse_args(argv)

    chunks_list = load_chunks(CHUNKS_DIR)
    chunks = {str(chunk["chunk_id"]): chunk for chunk in chunks_list}
    corpus_hash = corpus_content_sha256(chunks_list)

    CHUNKS.clear()
    CHUNKS.update(chunks)
    VALUES.clear()
    for year, (value, _chunk) in NET_PROFIT.items():
        VALUES[(year, "net_profit")] = value
    for year, (value, _chunk) in ROE.items():
        VALUES[(year, "roe")] = value
    for year, (value, _chunk, _dividend_year) in DIVIDENDS.items():
        VALUES[(year, "dividends")] = value
    for year, (value, _chunk) in CLIENTS.items():
        VALUES[(year, "clients")] = value

    suites = {
        "easy": {"schema_version": SCHEMA_VERSION, "suite": "easy",
                 "expected_corpus_sha256": corpus_hash, "tests": build_easy()},
        "medium": {"schema_version": SCHEMA_VERSION, "suite": "medium",
                   "expected_corpus_sha256": corpus_hash, "tests": build_medium()},
        "hard": {"schema_version": SCHEMA_VERSION, "suite": "hard",
                 "expected_corpus_sha256": corpus_hash, "tests": build_hard()},
        "multi_turn": {"schema_version": SCHEMA_VERSION, "suite": "multi_turn",
                       "expected_corpus_sha256": corpus_hash, "tests": build_multi_turn()},
    }

    problems = verify_tests(
        [test for payload in suites.values() for test in payload["tests"]], chunks)
    if problems:
        print("Эталон не сходится с корпусом:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    total = sum(len(payload["tests"]) for payload in suites.values())
    print(f"Проверка эталона: OK | тестов {total} | чанков {len(chunks_list)}")
    if args.check_only:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in suites.items():
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {path.relative_to(PROJECT)}: {len(payload['tests'])} тестов")
    manifest = build_manifest(suites=suites, chunks=chunks_list, corpus_hash=corpus_hash)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {(OUT_DIR / 'manifest.json').relative_to(PROJECT)}")

    bundle = validate_dataset(OUT_DIR, chunks_dir=CHUNKS_DIR, require_manifest=True)
    if bundle.issues:
        print("Валидация датасета не прошла:", file=sys.stderr)
        for issue in bundle.issues:
            print(f"  - {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    print(f"Валидация: OK | dataset_sha256={bundle.dataset_sha256[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
