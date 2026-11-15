# -*- coding: utf-8 -*-
"""Диагностика доступа к Cloud.ru Foundation Models без вывода секретов.

Проверяет отдельно:
* публичный список моделей;
* chat/completions;
* embeddings.

Ключ Cloud.ru — base64(UUID проекта + секрет). UUID печатается для сверки с
консолью Cloud.ru; секрет не выводится.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def project_id_from_key(key: str) -> str | None:
    """Извлечь UUID проекта из base64-ключа Cloud.ru, не раскрывая секрет."""
    try:
        raw = base64.b64decode(key + "=" * ((4 - len(key) % 4) % 4))
    except (ValueError, TypeError):
        return None
    if len(raw) < 36:
        return None
    candidate = raw[:36].decode("ascii", errors="ignore")
    parts = candidate.split("-")
    if [len(part) for part in parts] != [8, 4, 4, 4, 12]:
        return None
    return candidate


def _report(label: str, call) -> None:  # noqa: ANN001 - локальный diagnostic helper
    try:
        result = call()
        print(f"{label}: OK ({result})")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        body = (getattr(response, "text", "") or "").replace("\n", " ")[:300]
        print(f"{label}: FAIL {type(exc).__name__} {status} {body}".strip())


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    key = (os.getenv("LLM_API_KEY") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    chat_model = os.getenv("LLM_MODEL", "GigaChat/GigaChat-3-Pro")
    embed_model = os.getenv("EMBED_MODEL", "ai-sage/Giga-Embeddings-instruct-3B")
    print(f"base_url: {base_url or '<empty>'}")
    print(f"key present: {bool(key)} | length: {len(key)}")
    print(f"project UUID from key: {project_id_from_key(key) or '<unknown>'}")
    if not key or not base_url:
        print("LLM_API_KEY/LLM_BASE_URL не заданы.")
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url, timeout=30, max_retries=0)
    _report(
        "models.list",
        lambda: len(client.models.list().data),
    )
    _report(
        f"chat[{chat_model}]",
        lambda: (
            client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            ).choices[0].message.content
            or ""
        )[:40],
    )
    _report(
        f"embeddings[{embed_model}]",
        lambda: len(
            client.embeddings.create(model=embed_model, input=["ping"]).data[0].embedding
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
