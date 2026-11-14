"""Offline tests for RAGEngine orchestration with fake retriever/generator."""

from __future__ import annotations

import pytest

from src.agent.state import Answer, Citation
from src.engine.context import pack_context
from src.engine.generator import AnswerGenerator
from src.engine.rag_engine import RAGEngine, build_engine
from src.retrieval.hybrid_retriever import (
    RetrievalHit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)


def make_chunk(chunk_id: str, text: str = "Чистая прибыль 100 млрд руб.", **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": "sber_2023",
        "year": 2023,
        "issuer": "ПАО Сбербанк",
        "page_pdf": 42,
        "page_label": "41",
        "section_path": "Риски",
        "kind": "text",
        "text": text,
        "html": None,
    }
    chunk.update(overrides)
    return chunk


def make_hit(chunk_id: str, text: str = "Чистая прибыль 100 млрд руб."):
    return RetrievalHit(chunk_id=chunk_id, chunk=make_chunk(chunk_id, text=text))


def make_result(hits):
    return RetrievalResult(
        hits=list(hits),
        warnings=[],
        missing_years=[],
        effective_filters=SearchFilters(),
        trace=RetrievalTrace(),
    )


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query, *, filters=None):
        self.calls.append((query, filters))
        return self.result


class FakeGenerator:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, question, evidence, evidence_manifest, **kwargs):
        self.calls.append(
            {
                "question": question,
                "evidence": dict(evidence),
                "manifest": set(evidence_manifest),
                "kwargs": kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer
        chunk_id = sorted(evidence_manifest)[0]
        return Answer("answer", f"Ответ: {question}", [Citation(chunk_id)])


def test_ask_end_to_end_passes_only_packed_evidence():
    retriever = FakeRetriever(make_result([make_hit("c1"), make_hit("c2")]))
    generator = FakeGenerator(Answer("answer", "Прибыль 100", [Citation("c1")]))
    engine = RAGEngine(retriever, generator)
    filters = SearchFilters(years={2023})

    answer = engine.ask("Какая прибыль?", filters=filters, conversation_id="conv-1")

    assert retriever.calls == [("Какая прибыль?", filters)]
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["question"] == "Какая прибыль?"
    assert set(call["evidence"]) == {"c1", "c2"}
    assert call["manifest"] == {"c1", "c2"}
    assert call["evidence"]["c1"]["doc_id"] == "sber_2023"
    assert call["evidence"]["c1"]["page_pdf"] == 42
    assert answer.status == "answer"
    assert answer.answer == "Прибыль 100"
    assert answer.citations[0].chunk_id == "c1"


def test_retrieve_delegates_without_calling_generator():
    result = make_result([make_hit("c1")])
    retriever = FakeRetriever(result)
    generator = FakeGenerator()
    engine = RAGEngine(retriever, generator)
    filters = SearchFilters(years={2023})

    assert engine.retrieve("Вопрос", filters=filters) is result
    assert retriever.calls == [("Вопрос", filters)]
    assert generator.calls == []


def test_empty_retrieval_abstains_without_llm_call():
    retriever = FakeRetriever(make_result([]))
    generator = FakeGenerator()
    engine = RAGEngine(retriever, generator)

    answer = engine.ask("Вопрос вне корпуса", filters=None, conversation_id="c")

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["no_evidence"]
    assert generator.calls == []


def test_hits_that_do_not_fit_budget_abstain_without_llm_call():
    retriever = FakeRetriever(make_result([make_hit("c1", "x" * 500)]))
    generator = FakeGenerator()
    engine = RAGEngine(retriever, generator, max_context_chars=5)

    answer = engine.ask("Вопрос")

    assert answer.status == "abstain"
    assert answer.warnings == ["no_evidence"]
    assert generator.calls == []


def test_context_budget_passes_only_included_chunks_to_generator():
    small = make_hit("small", "S" * 40)
    huge = make_hit("huge", "H" * 2_000)
    budget = len(pack_context([small], max_chars=1_000_000).text)
    retriever = FakeRetriever(make_result([small, huge]))
    generator = FakeGenerator()
    engine = RAGEngine(retriever, generator, max_context_chars=budget)

    answer = engine.ask("Вопрос")

    assert answer.status == "answer"
    call = generator.calls[0]
    assert set(call["evidence"]) == {"small"}
    assert call["manifest"] == {"small"}
    assert "huge" not in call["evidence"]


def test_generator_exception_becomes_safe_abstain_without_secret():
    retriever = FakeRetriever(make_result([make_hit("c1")]))
    generator = FakeGenerator(error=RuntimeError("sk-secret PROMPT provider body"))
    engine = RAGEngine(retriever, generator)

    answer = engine.ask("Вопрос")

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["generation_unavailable"]
    visible = answer.answer + " ".join(answer.warnings)
    assert "sk-secret" not in visible
    assert "PROMPT" not in visible
    assert "provider body" not in visible


def test_outside_manifest_citation_from_injected_generator_is_rejected():
    retriever = FakeRetriever(make_result([make_hit("c1")]))
    generator = FakeGenerator(Answer("answer", "Claim", [Citation("outside")]))
    engine = RAGEngine(retriever, generator)

    answer = engine.ask("Вопрос")

    assert answer.status == "abstain"
    assert answer.citations == []
    assert answer.warnings == ["citation_validation_failed"]


def test_mapping_answer_is_coerced_to_answer_contract():
    retriever = FakeRetriever(make_result([make_hit("c1")]))
    generator = FakeGenerator(
        {
            "status": "answer",
            "answer": "Прибыль 100",
            "citations": [{"chunk_id": "c1", "quote": "Чистая прибыль"}],
            "facts": [{"claim": "Прибыль", "chunk_id": "c1"}],
            "warnings": ["low_confidence"],
        }
    )
    engine = RAGEngine(retriever, generator)

    answer = engine.ask("Вопрос")

    assert isinstance(answer, Answer)
    assert answer.status == "answer"
    assert [(citation.chunk_id, citation.quote) for citation in answer.citations] == [
        ("c1", "Чистая прибыль")
    ]
    assert answer.facts == [{"claim": "Прибыль", "chunk_id": "c1"}]
    assert answer.warnings == ["low_confidence"]


def test_build_engine_accepts_already_built_components():
    retriever = FakeRetriever(make_result([make_hit("c1")]))
    generator = FakeGenerator()

    built = build_engine(retriever, generator)

    assert isinstance(built, RAGEngine)
    assert built.retriever is retriever
    assert built.generator is generator


def test_build_engine_can_wrap_an_explicit_client_without_network_calls():
    class NeverCalledClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    raise AssertionError("factory must not call the provider")

            completions = Completions()

        chat = Chat()

    retriever = FakeRetriever(make_result([]))
    built = build_engine(retriever, client=NeverCalledClient(), model="test-model")

    assert isinstance(built, RAGEngine)
    assert isinstance(built.generator, AnswerGenerator)
    assert built.generator.model == "test-model"


def test_build_engine_requires_exactly_one_generator_source():
    retriever = FakeRetriever(make_result([]))

    with pytest.raises(ValueError):
        build_engine(retriever)
    with pytest.raises(ValueError):
        build_engine(retriever, FakeGenerator(), client=object(), model="m")


def test_invalid_max_context_chars_is_rejected():
    retriever = FakeRetriever(make_result([]))
    generator = FakeGenerator()

    with pytest.raises(ValueError):
        RAGEngine(retriever, generator, max_context_chars=-1)
    with pytest.raises(ValueError):
        RAGEngine(retriever, generator, max_context_chars=True)
