# -*- coding: utf-8 -*-
"""Проверка LLM: Cloud.ru Foundation Models (GigaChat, OpenAI-совместимый API)."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

response = client.chat.completions.create(
    model=os.environ.get("LLM_MODEL", "GigaChat/GigaChat-3-Pro"),
    messages=[{"role": "user", "content": "Привет! Ответь одним коротким предложением, что ты работаешь."}],
)
print("Модель:", response.model)
print("Ответ:", response.choices[0].message.content)
