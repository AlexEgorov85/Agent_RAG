# -*- coding: utf-8 -*-
"""Довести полный граф до status=complete, пережидая free-tier окна чата.

``build_full_graph.py`` умеет чекпоинт по пакетам, но при исчерпании лимита
чата (400 запросов / 3 часа на free) часть пакетов помечается failed. Этот
раннер перезапускает сборку после сброса окна: успешные пакеты берутся из
чекпоинта, failed/новые — считаются заново.

Запуск::

    python scripts/run_graph_full.py
    python scripts/run_graph_full.py --workers 3 --model qwen3.6-35b-a3b-noreason
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DEFAULT_MODEL = "qwen3.6-35b-a3b-noreason"
CHECKPOINT = PROJECT / "data" / "graph" / ".extraction_checkpoint.jsonl"


def _load_env() -> dict[str, str]:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    return {key: os.getenv(key, "") for key in (
        "EMBED_API_KEY", "NEURALDEEP_API_KEY", "EMBED_BASE_URL")}


def _chat_window() -> tuple[int, int, int]:
    """Вернуть (used, limit, reset_in_sec) для окна чата."""
    env = _load_env()
    key = (env.get("EMBED_API_KEY") or env.get("NEURALDEEP_API_KEY") or "").strip()
    base = (env.get("EMBED_BASE_URL") or "").strip().rstrip("/")
    if not key or not base:
        return 0, 1, 300
    try:
        request = urllib.request.Request(
            f"{base}/limits", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        session = payload["chat"]["session"]
        return (
            int(session.get("used") or 0),
            int(session.get("limit") or 1),
            max(0, int(session.get("reset_in_sec") or 0)),
        )
    except Exception:  # noqa: BLE001 - диагностика не должна ронять сборку
        return 0, 1, 300


def _latest_manifest(before: set[Path] | None = None) -> dict | None:
    files = sorted(
        (PROJECT / "data" / "graph").glob("*_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if before:
        fresh = [path for path in files if path.resolve() not in before]
        files = fresh or files
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("graph_fingerprint"):
            return payload
    return None


def _checkpoint_lines() -> int:
    if not CHECKPOINT.is_file():
        return 0
    try:
        return sum(1 for line in CHECKPOINT.read_text(
            encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Собрать полный граф с ожиданием free-tier окон.")
    parser.add_argument("--chunks-dir", type=Path,
                        default=PROJECT / "data" / "chunks")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT / "data" / "graph")
    parser.add_argument("--structure-dir", type=Path,
                        default=PROJECT / "data" / "structure")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunks-per-call", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--provider", choices=("cloud", "neuraldeep"),
                        default="cloud",
                        help="cloud — LLM_* из .env без квотных пауз; "
                             "neuraldeep — бесплатный тариф с ожиданием окон")
    args = parser.parse_args(argv)

    env = _load_env()
    cloud = args.provider == "cloud"
    if cloud:
        os.environ["LLM_MODEL"] = args.model
    else:
        api_key = (env.get("EMBED_API_KEY")
                   or env.get("NEURALDEEP_API_KEY") or "").strip()
        base_url = (env.get("EMBED_BASE_URL") or "").strip()
        if not api_key or not base_url:
            print("Не заданы EMBED_API_KEY/EMBED_BASE_URL.", file=sys.stderr)
            return 1
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_BASE_URL"] = base_url
        os.environ["LLM_MODEL"] = args.model
    os.environ["GRAPH_MAX_COMPLETION_TOKENS"] = str(args.max_completion_tokens)

    no_progress = 0
    for attempt in range(1, args.max_attempts + 1):
        if cloud:
            budget = None
            print(
                f"\n=== graph attempt {attempt}/{args.max_attempts} | "
                f"cloud {args.model} | checkpoint {_checkpoint_lines()} пакетов ===",
                flush=True,
            )
        else:
            used, limit, reset = _chat_window()
            remaining = max(0, limit - used)
            if remaining < 20:
                no_progress += 1
                wait = reset + 30 if no_progress < 3 else max(reset + 30, 1800)
                print(
                    f"\n=== graph: жду окно чата {wait} с "
                    f"(чат {used}/{limit}) ===",
                    flush=True,
                )
                time.sleep(wait)
                continue
            budget = min(remaining - 15, 800)
        if not cloud:
            print(
                f"\n=== graph attempt {attempt}/{args.max_attempts} | "
                f"checkpoint {_checkpoint_lines()} пакетов | "
                f"бюджет {budget} (чат {used}/{limit}) ===",
                flush=True,
            )
        before = {
            path.resolve()
            for path in args.out_dir.glob("*_manifest.json")
        }
        before_cp = _checkpoint_lines()
        sys.argv = [
            "build_full_graph.py",
            "--chunks-dir", str(args.chunks_dir),
            "--out-dir", str(args.out_dir),
            "--structure-dir", str(args.structure_dir),
            "--model", args.model,
            "--workers", str(args.workers),
            "--chunks-per-call", str(args.chunks_per_call),
            "--checkpoint", str(CHECKPOINT),
            "--strict",
        ]
        if budget is not None:
            sys.argv.extend(["--max-batches", str(budget)])
        try:
            runpy.run_path(
                str(PROJECT / "scripts" / "build_full_graph.py"),
                run_name="__main__",
            )
        except SystemExit:
            pass

        manifest = _latest_manifest(before)
        status = (manifest or {}).get("status")
        print(
            f"=== attempt {attempt}: status={status} | "
            f"entities={(manifest or {}).get('entity_count')} | "
            f"relations={(manifest or {}).get('relation_count')} | "
            f"failed={len((manifest or {}).get('failed_chunk_ids') or [])} | "
            f"checkpoint {_checkpoint_lines()} ===",
            flush=True,
        )
        if status == "complete":
            print("Граф собран полностью.", flush=True)
            return 0

        if _checkpoint_lines() == before_cp:
            no_progress += 1
            wait = 60 if no_progress < 3 else 1800
            print(
                f"Нет прироста чекпоинта — пауза {wait} с.",
                flush=True,
            )
        else:
            no_progress = 0
            wait = 20 if cloud else 30
        time.sleep(wait)

    print("Не удалось завершить граф за отведённые попытки.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
