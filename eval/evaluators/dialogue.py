"""Evaluator многошагового диалога.

Детерминированные проверки используют только trace и результаты answer
evaluator'а. Смысловые проверки (корректность сущности, причинность) остаются
судье/ревьюеру и здесь помечаются ``not_measured``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from eval.metrics import aggregate_mean


def _turns(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    turns = trace.get("turns")
    if isinstance(turns, list):
        return [turn for turn in turns if isinstance(turn, Mapping)]
    return []


def _plan(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = turn.get("query_plan")
    return plan if isinstance(plan, Mapping) else {}


def _filters(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    filters = turn.get("effective_filters")
    return filters if isinstance(filters, Mapping) else {}


def _years(turn: Mapping[str, Any]) -> set[int]:
    values: set[int] = set()
    for source in (_plan(turn), _filters(turn)):
        years = source.get("years")
        if isinstance(years, list):
            values.update(year for year in years if isinstance(year, int))
        for key in ("year_from", "year_to"):
            if isinstance(source.get(key), int):
                values.add(int(source[key]))
    return values


def _entities(turn: Mapping[str, Any]) -> set[str]:
    entities = _plan(turn).get("entities")
    return {str(item) for item in entities or []} if isinstance(entities, list) else set()


_WORD = re.compile(r"[0-9a-zа-я]+")


def _entity_words(values: set[str]) -> set[str]:
    return {
        word
        for value in values
        for word in _WORD.findall(value.casefold().replace("ё", "е"))
        if len(word) > 2
    }


def _same_word(left: str, right: str) -> bool:
    """Словоформы одного термина: ``риски`` / ``рисками``."""
    if left == right:
        return True
    prefix = 0
    for first, second in zip(left, right):
        if first != second:
            break
        prefix += 1
    return prefix >= 4 and prefix >= min(len(left), len(right)) - 3


def _entities_overlap(previous: set[str], current: set[str]) -> bool:
    """Пересечение сущностей с учётом словоформ и составных названий.

    Планировщик возвращает то английские ключи, то русские формулировки
    («управление рисками» → «климатические риски»), поэтому точного равенства
    множеств недостаточно: сохраняется общий термин диалога.
    """
    previous_words = _entity_words(previous)
    current_words = _entity_words(current)
    return any(
        _same_word(left, right)
        for left in previous_words
        for right in current_words
    )


def _cited(turn: Mapping[str, Any]) -> set[str]:
    answer = turn.get("answer")
    if not isinstance(answer, Mapping):
        return set()
    return {
        str(citation.get("chunk_id"))
        for citation in answer.get("citations", []) or []
        if isinstance(citation, Mapping) and citation.get("chunk_id")
    }


def _queries(turn: Mapping[str, Any]) -> list[str]:
    queries: list[str] = []
    for call in turn.get("retrieval_calls", []) or []:
        if not isinstance(call, Mapping):
            continue
        query = call.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip().casefold())
        for item in call.get("queries", []) or []:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip().casefold())
    return list(dict.fromkeys(queries))


def _search_count(turn: Mapping[str, Any]) -> int:
    return sum(
        1
        for event in turn.get("tool_events", []) or []
        if isinstance(event, Mapping) and event.get("tool_name") == "search_reports"
    )


def _turn_result(answer_evaluation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    turns = answer_evaluation.get("turns")
    if isinstance(turns, list) and index < len(turns):
        return turns[index] if isinstance(turns[index], Mapping) else {}
    return answer_evaluation if index == 0 else {}


def _check_value(name: str, value: bool | None, rows: dict[str, Any]) -> None:
    rows[name] = value


def evaluate_dialogue(
    test: Mapping[str, Any],
    trace: Mapping[str, Any],
    answer_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Проверить сценарий и вернуть сценарные checks, pass и cascade-флаг."""
    test_turns = [
        turn for turn in test.get("turns", []) or [] if isinstance(turn, Mapping)
    ]
    turns = _turns(trace)
    checks: dict[str, bool | None] = {}
    notes: list[str] = []
    not_measured: list[str] = []

    if not test_turns or len(turns) < len(test_turns):
        return {
            "pass": False,
            "checks": {},
            "required_checks": list(test.get("required_checks", []) or []),
            "critical_errors": [
                {"code": "dialogue_incomplete", "message": "Не все turn'ы сценария выполнены"}
            ],
            "warnings": [],
            "cascade_failure": False,
            "turn_pass": [],
            "not_measured": ["dialogue"],
        }

    # 1. Статус и факты по turn'ам.
    status_ok = all(
        _turn_result(answer_evaluation, index).get("metrics", {}).get(
            "status_accuracy"
        ) == 1.0
        for index in range(len(test_turns))
    )
    checks["context_preserved"] = None
    context_evidence: list[bool] = []
    #: Turn, который по замыслу меняет показатель/исправляет ответ: смена
    #: entity здесь ожидаема, контекст держится на сохранённом периоде.
    metric_change_behaviors = {"change_metric", "apply_correction"}
    for index in range(1, len(test_turns)):
        spec = test_turns[index]
        behavior = spec.get("behavior")
        if behavior == "clarify_period":
            continue
        previous = turns[index - 1]
        current = turns[index]
        previous_years = _years(previous)
        current_years = _years(current)
        previous_entities = _entities(previous)
        current_entities = _entities(current)
        if not (previous_years or current_years or previous_entities or current_entities):
            continue
        period_preserved = (
            True
            if not previous_years or not current_years
            else bool(previous_years & current_years)
        )
        if behavior in metric_change_behaviors:
            entity_preserved = True
        else:
            entity_preserved = (
                True
                if not previous_entities or not current_entities
                else _entities_overlap(previous_entities, current_entities)
            )
        context_evidence.append(period_preserved and entity_preserved)
    if context_evidence:
        checks["context_preserved"] = all(context_evidence)
    else:
        not_measured.append("context_preserved")

    # 2. Метрика и период по ожидаемым фактам/годам.
    fact_scores = [
        _turn_result(answer_evaluation, index).get("metrics", {}).get("fact_accuracy")
        for index, spec in enumerate(test_turns)
        if (spec.get("expected") or {}).get("required_facts")
    ]
    checks["correct_metric"] = (
        all(score == 1.0 for score in fact_scores) if fact_scores else None
    )
    if checks["correct_metric"] is None:
        not_measured.append("correct_metric")

    period_rows = []
    for index, spec in enumerate(test_turns):
        expected = spec.get("expected") or {}
        if not expected.get("required_years"):
            continue
        row = _turn_result(answer_evaluation, index)
        check = next(
            (item for item in row.get("checks", []) if item.get("name") == "required_years"),
            None,
        )
        period_rows.append(bool(check and check.get("pass")))
    checks["correct_period"] = all(period_rows) if period_rows else None
    if checks["correct_period"] is None:
        not_measured.append("correct_period")

    # 3. Повторный широкий поиск без основания.
    restart_rows: list[bool] = []
    for index in range(1, len(test_turns)):
        spec = test_turns[index]
        behavior = spec.get("behavior")
        if behavior not in {
            "continue_previous_request",
            "use_previous_evidence_and_calculate",
            "apply_correction",
            "change_metric",
            "compare_with_previous_year",
        }:
            continue
        previous = turns[index - 1]
        current = turns[index]
        same_query = bool(set(_queries(previous)) & set(_queries(current)))
        same_filters = _filters(previous) == _filters(current)
        restarted = (
            same_query
            and same_filters
            and _search_count(current) >= _search_count(previous)
            and _search_count(current) > 0
        )
        restart_rows.append(not restarted)
    if restart_rows:
        checks["no_unnecessary_restart"] = all(restart_rows)
    else:
        not_measured.append("no_unnecessary_restart")

    # 4. Источники остаются валидными.
    citation_rows = []
    for index in range(len(test_turns)):
        result = _turn_result(answer_evaluation, index)
        critical_codes = {
            error.get("code") for error in result.get("critical_errors", [])
        }
        if "invalid_citation" in critical_codes or "unsupported_claim" in critical_codes:
            citation_rows.append(False)
        elif _cited(turns[index]):
            citation_rows.append(True)
        elif (test_turns[index].get("expected_status") == "abstain"):
            citation_rows.append(True)
    checks["sources_preserved"] = all(citation_rows) if citation_rows else None
    if checks["sources_preserved"] is None:
        not_measured.append("sources_preserved")

    checks["correct_entity"] = None
    not_measured.append("correct_entity")

    calc_scores = [
        _turn_result(answer_evaluation, index).get("metrics", {}).get(
            "calculation_accuracy"
        )
        for index, spec in enumerate(test_turns)
        if (spec.get("expected") or {}).get("required_calculations")
    ]
    checks["calculation_correct"] = (
        all(score == 1.0 for score in calc_scores) if calc_scores else None
    )
    if checks["calculation_correct"] is None:
        not_measured.append("calculation_correct")

    policy = test.get("dialogue_policy") or {}
    if policy.get("missing_period") == "clarify":
        first = _turn_result(answer_evaluation, 0)
        first_spec = test_turns[0]
        required = (first_spec.get("checks") or {}).get("must_mention", [])
        first_answer = turns[0].get("answer", {})
        text = str(first_answer.get("text") or "")
        clarified = (
            first_spec.get("behavior") == "clarify_period"
            and first_answer.get("status") == "abstain"
            and all(str(item).casefold() in text.casefold() for item in required)
        )
        checks["clarification_asked"] = clarified or bool(
            first.get("metrics", {}).get("abstention_accuracy") == 1.0
        )
    elif policy.get("missing_period") == "default_period":
        checks["default_period_used"] = checks.get("correct_period")

    no_reask = True
    for index, spec in enumerate(test_turns):
        if spec.get("behavior") not in {
            "continue_previous_request",
            "use_previous_evidence_and_calculate",
            "apply_correction",
            "change_metric",
            "compare_with_previous_year",
        }:
            continue
        text = str(turns[index].get("answer", {}).get("text") or "").strip()
        if text.endswith("?"):
            no_reask = False
    checks["no_reask"] = no_reask

    evidence_rows = []
    for index in range(1, len(test_turns)):
        spec = test_turns[index]
        if not spec.get("reuse_previous_evidence"):
            continue
        current_cited = _cited(turns[index])
        previous_cited = _cited(turns[index - 1])
        evidence_rows.append(bool(current_cited) or bool(previous_cited))
    if evidence_rows:
        checks["evidence_not_lost"] = all(evidence_rows)

    gap_ok = True
    any_gap = False
    for index, spec in enumerate(test_turns):
        expected = spec.get("expected") or {}
        missing_years = expected.get("missing_years", []) or []
        if not missing_years:
            continue
        any_gap = True
        text = str(turns[index].get("answer", {}).get("text") or "")
        if not all(str(year) in text for year in missing_years):
            gap_ok = False
    if any_gap:
        checks["gap_disclosed"] = gap_ok

    checks["correction_applied"] = None
    checks["contradiction_rechecked"] = None
    if any(spec.get("behavior") == "apply_correction" for spec in test_turns):
        checks["correction_applied"] = bool(
            checks.get("correct_metric") and checks.get("correct_period")
        )
    if any(spec.get("behavior") == "recheck_primary_source" for spec in test_turns):
        used_chunk = any(
            event.get("tool_name") == "get_chunk"
            for turn in turns
            for event in turn.get("tool_events", []) or []
            if isinstance(event, Mapping)
        )
        # Поведение «проверь оба первоисточника» не обязано выражаться
        # конкретным инструментом: агент может получить те же чанки обычным
        # поиском. Наблюдаемый признак — цитаты из обоих документов и оба
        # конфликтующих значения в ответе.
        recheck_index = next(
            index
            for index, spec in enumerate(test_turns)
            if spec.get("behavior") == "recheck_primary_source"
        )
        cited_docs = {
            str(citation.get("doc_id") or str(citation.get("chunk_id", "")).split(":")[0])
            for citation in turns[recheck_index].get("answer", {}).get("citations", []) or []
            if isinstance(citation, Mapping) and citation.get("chunk_id")
        }
        required_docs = set(
            (test_turns[recheck_index].get("expected") or {}).get(
                "required_documents", []
            )
            or []
        )
        elements_check = next(
            (
                item
                for item in _turn_result(answer_evaluation, recheck_index).get(
                    "checks", []
                )
                if item.get("name") == "required_elements"
            ),
            None,
        )
        values_ok = bool(elements_check and elements_check.get("pass"))
        checks["contradiction_rechecked"] = bool(
            used_chunk
            or (
                required_docs
                and required_docs <= cited_docs
                and values_ok
            )
        )
    for key in ("correction_applied", "contradiction_rechecked"):
        if checks.get(key) is None and key in (test.get("scenario_checks") or []):
            not_measured.append(key)

    required_checks = list(test.get("required_checks", []) or [])
    required_values = {name: checks.get(name) for name in required_checks}
    # ``None`` — проверка не применима/не измерена (например, correct_metric у
    # сценария без required_facts). Проваливает сценарий только явный False;
    # при этом хотя бы одна проверка должна быть реально подтверждена.
    scenario_checks_pass = (
        bool(required_checks)
        and all(value is not False for value in required_values.values())
        and any(value is True for value in required_values.values())
    )

    turn_pass = []
    cascade_failure = False
    turn_critical: list[dict[str, Any]] = []
    for index in range(len(test_turns)):
        result = _turn_result(answer_evaluation, index)
        passed = bool(result.get("pass")) and not cascade_failure
        if not passed:
            cascade_failure = True
        turn_pass.append(passed)
        for error in result.get("critical_errors", []) or []:
            turn_critical.append({**error, "turn_index": index})

    scenario_pass = (
        status_ok
        and scenario_checks_pass
        and all(turn_pass)
        and not turn_critical
        and not cascade_failure
    )
    return {
        "pass": scenario_pass,
        "checks": checks,
        "required_checks": required_checks,
        "critical_errors": turn_critical,
        "warnings": [],
        "cascade_failure": cascade_failure,
        "turn_pass": turn_pass,
        "not_measured": sorted(set(not_measured)),
        "metrics": {
            "scenario_success": 1.0 if scenario_pass else 0.0,
            "turn_pass_rate": aggregate_mean(
                [1.0 if value else 0.0 for value in turn_pass]
            ),
        },
    }
