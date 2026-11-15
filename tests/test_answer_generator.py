"""Офлайн-тесты JSON-генератора ответов и его безопасных fallback-ов."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.engine.generator import AnswerGenerator
from src.engine.prompts import MAX_QUOTE_CHARS


def make_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def make_evidence():
    return {
        "c1": {
            "chunk_id": "c1",
            "doc_id": "sber_2023",
            "year": 2023,
            "issuer": "ПАО Сбербанк",
            "page_pdf": 42,
            "page_label": "41",
            "section_path": "Управление рисками/Кредитный риск",
            "kind": "text",
            "text": "Чистая прибыль составила 100 млрд руб.",
            "html": None,
        },
        "outside": {
            "chunk_id": "outside",
            "doc_id": "sber_2024",
            "year": 2024,
            "text": "OUTSIDE-CONTENT must never reach the prompt.",
        },
    }


def valid_payload(**overrides):
    payload = {
        "status": "answer",
        "answer": "Чистая прибыль составила 100 млрд руб.",
        "citations": [{"chunk_id": "c1", "quote": "Чистая прибыль составила"}],
        "facts": [
            {
                "claim": "Чистая прибыль составила 100 млрд руб.",
                "value": "100",
                "unit": "млрд руб.",
                "period": "2023",
                "chunk_id": "c1",
            }
        ],
        "warnings": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def generator(client, **kwargs):
    return AnswerGenerator(client, "test-model", **kwargs)


def test_valid_json_is_parsed_and_citation_metadata_comes_from_evidence():
    client = FakeClient([make_response(valid_payload())])
    instance = generator(client, temperature=0, max_tokens=512, max_retries=1)

    answer = instance.generate("Какая прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert answer.answer == "Чистая прибыль составила 100 млрд руб."
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.chunk_id == "c1"
    assert citation.quote == "Чистая прибыль составила"
    # Метаданные берутся из упакованного источника, а не из ответа модели.
    assert citation.doc_id == "sber_2023"
    assert citation.year == 2023
    assert citation.issuer == "ПАО Сбербанк"
    assert citation.page_pdf == 42
    assert citation.page_label == "41"
    assert citation.section_path == "Управление рисками/Кредитный риск"
    assert answer.facts[0]["chunk_id"] == "c1"

    call = client.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 512
    assert call["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in call["messages"]] == ["system", "user"]


def test_prompt_contains_only_manifest_chunk_ids():
    client = FakeClient([make_response(valid_payload())])
    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    call = client.completions.calls[0]
    system = call["messages"][0]["content"]
    user = call["messages"][1]["content"]
    prompt = system + "\n" + user
    assert "c1" in prompt
    assert "outside" not in prompt
    assert "OUTSIDE-CONTENT" not in prompt
    assert "sber_2024" not in prompt

    sources = json.loads(user)["sources"]
    assert [source["chunk_id"] for source in sources] == ["c1"]


def test_citation_outside_manifest_is_dropped_documented_choice():
    payload = valid_payload(
        citations=[
            {"chunk_id": "outside", "quote": "invented"},
            {"chunk_id": "c1", "quote": "Чистая прибыль составила"},
        ]
    )
    client = FakeClient([make_response(payload)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert "citation_outside_manifest" in answer.warnings
    assert len(client.completions.calls) == 1


def test_answer_with_only_outside_citations_retries_then_safe_abstain():
    payload = valid_payload(citations=[{"chunk_id": "outside", "quote": "invented"}])
    client = FakeClient([make_response(payload), make_response(payload), make_response(payload)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["generation_unavailable"]
    assert len(client.completions.calls) == 3
    assert "outside" not in answer.answer
    assert "invented" not in answer.answer


def test_abstain_must_have_empty_citations():
    invalid = valid_payload(
        status="abstain",
        answer="Недостаточно данных.",
        citations=[{"chunk_id": "c1", "quote": "Чистая прибыль"}],
        facts=[],
    )
    client = FakeClient([make_response(invalid), make_response(invalid), make_response(invalid)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["generation_unavailable"]
    assert len(client.completions.calls) == 3


def test_valid_abstain_keeps_safe_warning_code():
    payload = json.dumps(
        {
            "status": "abstain",
            "answer": "Отчёт за 2021 год в корпусе отсутствует.",
            "citations": [],
            "facts": [],
            "warnings": ["missing_year"],
        },
        ensure_ascii=False,
    )
    client = FakeClient([make_response(payload)])

    answer = generator(client).generate("Прибыль 2021?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.facts == []
    assert answer.answer == "Отчёт за 2021 год в корпусе отсутствует."
    assert answer.warnings == ["missing_year"]


def test_empty_manifest_abstains_without_calling_llm():
    client = FakeClient([])

    answer = generator(client).generate("Прибыль?", make_evidence(), set())

    assert answer.status == "abstain"
    assert answer.warnings == ["no_evidence"]
    assert client.completions.calls == []


def test_context_too_small_abstains_without_calling_llm():
    client = FakeClient([])
    instance = AnswerGenerator(client, "test-model", max_context_chars=10)

    answer = instance.generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.warnings == ["no_evidence"]
    assert client.completions.calls == []


def test_api_failure_is_retried_and_can_recover():
    client = FakeClient(
        [RuntimeError("provider exploded sk-secret"), make_response(valid_payload())]
    )

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert answer.citations[0].chunk_id == "c1"
    assert len(client.completions.calls) == 2


def test_retry_exhaustion_returns_safe_abstain_without_provider_text():
    secret = "sk-live-SUPERSECRET"
    provider_body = "PROMPT=system secret provider body"
    client = FakeClient(
        [
            RuntimeError(f"{secret} {provider_body}"),
            RuntimeError(f"{secret} {provider_body}"),
            RuntimeError(f"{secret} {provider_body}"),
        ]
    )

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["generation_unavailable"]
    visible = answer.answer + " ".join(answer.warnings)
    assert secret not in visible
    assert "PROMPT" not in visible
    assert "provider body" not in visible
    assert len(client.completions.calls) == 3


def test_invalid_json_is_retried_then_succeeds():
    client = FakeClient(
        [make_response("this is not json"), make_response(valid_payload())]
    )

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert len(client.completions.calls) == 2


def test_invalid_json_and_schema_failures_exhaust_retries():
    client = FakeClient(
        [
            make_response("not json"),
            make_response(json.dumps({"status": "maybe"})),
            make_response(json.dumps({"status": "maybe"})),
        ]
    )

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "abstain"
    assert answer.warnings == ["generation_unavailable"]
    assert len(client.completions.calls) == 3


def test_max_retries_zero_means_one_attempt():
    client = FakeClient([RuntimeError("failure")])

    answer = AnswerGenerator(client, "test-model", max_retries=0).generate(
        "Прибыль?", make_evidence(), {"c1"}
    )

    assert answer.status == "abstain"
    assert len(client.completions.calls) == 1


def test_model_warnings_are_allowlisted_and_cannot_leak_prompt_text():
    payload = valid_payload(warnings=["PROMPT BODY secret", "missing_year"])
    client = FakeClient([make_response(payload)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.warnings == ["answer_warning", "missing_year"]
    assert "secret" not in " ".join(answer.warnings)


def test_repair_prompt_passes_codes_and_ids_but_not_free_text():
    client = FakeClient([make_response(valid_payload())])
    diagnostics = {
        "reason": "answer_schema_or_citation_validation_failed",
        "invalid_citation_chunk_ids": ["ghost_chunk_123", "not a valid id"],
        "issues": [
            {
                "code": "fact_value_not_in_source",
                "severity": "fatal",
                "message": "LEAK-MESSAGE: содержимое чанка",
                "chunk_ids": ["c1"],
            },
            {
                "code": "fact_year_mismatch",
                "severity": "fatal",
                "message": "SECOND-LEAK",
            },
        ],
        "answer": "LEAK-ANSWER с текстом ответа модели",
        "chunk_text": "SECRET-CHUNK-BODY",
    }

    answer = generator(client).generate(
        "Прибыль?", make_evidence(), {"c1"}, repair=True, diagnostics=diagnostics
    )

    assert answer.status == "answer"
    user = client.completions.calls[0]["messages"][1]["content"]
    payload = json.loads(user)
    assert "repair_instruction" in payload
    forwarded = payload["repair_diagnostics"]
    assert forwarded["issue_codes"] == [
        "answer_schema_or_citation_validation_failed",
        "fact_value_not_in_source",
        "fact_year_mismatch",
    ]
    assert forwarded["invalid_chunk_ids"] == ["ghost_chunk_123"]
    # Текст чанков, сообщений и ответа модели в repair-промпт не попадает.
    assert "LEAK-MESSAGE" not in user
    assert "SECOND-LEAK" not in user
    assert "LEAK-ANSWER" not in user
    assert "SECRET-CHUNK-BODY" not in user
    assert "not a valid id" not in user


def test_long_quote_is_truncated_instead_of_rejecting_the_answer() -> None:
    payload = json.loads(valid_payload())
    payload["citations"][0]["quote"] = "я" * (MAX_QUOTE_CHARS + 50)
    client = FakeClient(
        [make_response(json.dumps(payload, ensure_ascii=False))]
    )

    answer = generator(client).generate(
        "прибыль?", make_evidence(), {"c1"}
    )

    assert answer.status == "answer"
    assert len(answer.citations[0].quote) <= MAX_QUOTE_CHARS
    assert "quote_truncated" in answer.warnings


def test_history_is_bounded_and_sent_as_dialogue():
    client = FakeClient([make_response(valid_payload())])
    history = [
        {"role": "user", "content": f"question-{index}-" + "x" * 5_000}
        for index in range(30)
    ]

    answer = generator(client).generate(
        "Прибыль?", make_evidence(), {"c1"}, history=history
    )

    assert answer.status == "answer"
    user = json.loads(client.completions.calls[0]["messages"][1]["content"])
    dialogue = user["recent_dialogue"]
    assert len(dialogue) == 20
    assert all(len(turn["content"]) <= 2_000 for turn in dialogue)
    assert all(set(turn) == {"role", "content"} for turn in dialogue)


def test_fact_referencing_dropped_citation_is_removed_with_warning():
    payload = valid_payload(
        facts=[
            {
                "claim": "Выдуманный факт",
                "value": "100",
                "unit": "млрд руб.",
                "period": "2023",
                "chunk_id": "outside",
            },
            {
                "claim": "Чистая прибыль составила 100 млрд руб.",
                "value": "100",
                "unit": "млрд руб.",
                "period": "2023",
                "chunk_id": "c1",
            },
        ]
    )
    client = FakeClient([make_response(payload)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]
    assert [fact["chunk_id"] for fact in answer.facts] == ["c1"]
    assert "fact_not_supported" in answer.warnings
    assert len(client.completions.calls) == 1


def test_retry_after_invalid_json_adds_repair_instruction():
    client = FakeClient(
        [make_response("not json"), make_response(valid_payload())]
    )

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    first = client.completions.calls[0]["messages"][1]["content"]
    second = client.completions.calls[1]["messages"][1]["content"]
    assert "repair_instruction" not in first
    assert "repair_instruction" in second


def test_fact_value_absent_from_cited_chunk_is_removed_with_warning():
    payload = valid_payload(
        facts=[
            {
                "claim": "Прибыль 999 млрд руб.",
                "value": "999",
                "unit": "млрд руб.",
                "period": "2023",
                "chunk_id": "c1",
            },
            {
                "claim": "Чистая прибыль составила 100 млрд руб.",
                "value": "100",
                "unit": "млрд руб.",
                "period": "2023",
                "chunk_id": "c1",
            },
        ]
    )
    client = FakeClient([make_response(payload)])

    answer = generator(client).generate("Прибыль?", make_evidence(), {"c1"})

    assert answer.status == "answer"
    assert [fact["value"] for fact in answer.facts] == ["100"]
    assert "fact_not_supported" in answer.warnings
    assert len(client.completions.calls) == 1
