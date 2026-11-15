"""Тесты детектора служебных реплик вне корпуса (small talk)."""

from __future__ import annotations

import pytest

from src.agent.smalltalk import (
    BYE_REPLY,
    CAPABILITIES_REPLY,
    GREETING_REPLY,
    HOW_ARE_YOU_REPLY,
    THANKS_REPLY,
    smalltalk_reply,
)


@pytest.mark.parametrize(
    "query",
    [
        "Привет!",
        "привет",
        "Здравствуйте",
        "Добрый день!",
        "ДОБРОЕ УТРО",
        "приветик :)",
        "Hello!",
        "привет всем",
        "Привет, агент!",
        "  здравствуйте,  ",
    ],
)
def test_greeting_is_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) == GREETING_REPLY


@pytest.mark.parametrize(
    "query",
    ["Спасибо!", "спасибо большое", "Благодарю.", "thanks", "Thank you!"],
)
def test_thanks_is_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) == THANKS_REPLY


@pytest.mark.parametrize(
    "query", ["Пока!", "до свидания", "всего доброго", "bye", "Goodbye."]
)
def test_bye_is_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) == BYE_REPLY


@pytest.mark.parametrize(
    "query",
    ["Как дела?", "как ты", "How are you?"],
)
def test_how_are_you_is_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) == HOW_ARE_YOU_REPLY


@pytest.mark.parametrize(
    "query",
    [
        "Что ты умеешь?",
        "что умеешь",
        "Что ещё умеешь?",
        "что еще умеешь",
        "А что ещё ты умеешь?",
        "Что можешь?",
        "Что ещё можешь?",
        "Что ты можешь делать?",
        "Чем можешь помочь?",
        "Чем ты можешь помочь?",
        "Какие у тебя возможности?",
        "На что ты способен?",
        "Расскажи о себе",
        "Кто ты?",
        "как тебя зовут",
        "Помощь",
    ],
)
def test_capabilities_is_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) == CAPABILITIES_REPLY


@pytest.mark.parametrize(
    "query",
    [
        "Привет! Какая чистая прибыль Сбербанка за 2023 год?",
        "Здравствуйте, сравните 2022 и 2023",
        "Привет, что с рисками?",
        "спасибо, а какая прибыль за 2023?",
        "Какая чистая прибыль за 2023 год?",
        "покажи структуру отчёта",
        "Привет за 2023",
        "Что можешь сказать о рисках?",
        "Что можешь сказать о динамике прибыли?",
        "Что умеет GigaChat?",
        "Чем полезен этот отчёт?",
        "Что ты можешь сказать про кредитный риск?",
        "",
        "   ",
    ],
)
def test_regular_questions_are_not_smalltalk(query: str) -> None:
    assert smalltalk_reply(query) is None


def test_long_message_is_not_smalltalk() -> None:
    query = "привет " + "очень длинное сообщение " * 10
    assert smalltalk_reply(query) is None
