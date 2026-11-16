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
    embed_base = (os.getenv("EMBED_BASE_URL")
                  or os.getenv("EMBEDDINGS_BASE_URL") or "").strip()
    embed_key = (os.getenv("EMBED_API_KEY")
                 or os.getenv("NEURALDEEP_API_KEY") or "").strip()
    print(f"base_url: {base_url or '<empty>'}")
    print(f"key present: {bool(key)} | length: {len(key)}")
    print(f"project UUID from key: {project_id_from_key(key) or '<unknown>'}")
    if embed_base or embed_key:
        print(f"embed_base_url: {embed_base or '<empty>'}")
        print(f"embed key present: {bool(embed_key)} | length: {len(embed_key)}")
        if not embed_base or not embed_key:
            print("EMBED_BASE_URL/EMBED_API_KEY заданы не полностью.")
            return 2
    if not (key and base_url) and not (embed_base and embed_key):
        print("LLM_API_KEY/LLM_BASE_URL (или EMBED_BASE_URL/EMBED_API_KEY) "
              "не заданы.")
        return 2

    from openai import OpenAI

    client = (
        OpenAI(api_key=key, base_url=base_url, timeout=30, max_retries=0)
        if key and base_url else None
    )
    embed_client = (
        OpenAI(api_key=embed_key, base_url=embed_base, timeout=30, max_retries=0)
        if embed_base and embed_key else client
    )
    if client is not None:
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
            embed_client.embeddings.create(
                model=embed_model, input=["ping"]).data[0].embedding
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
