# -*- coding: utf-8 -*-
"""Дождаться парсинга, проверить корпус и выключить ПК только при успехе.

Запускается отдельным процессом. Если``verify_corpus`` нашёл ошибки или не все
документы собраны, скрипт пишет отчёт и НЕ выключает машину.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import psutil  # noqa: E402

LOG_PATH = PROJECT / "logs" / "verify_and_shutdown.log"
REPORT_JSON = PROJECT / "data" / "corpus_report.json"
REPORT_MD = PROJECT / "data" / "corpus_report.md"


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _build_processes() -> list[psutil.Process]:
    found: list[psutil.Process] = []
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError):
            continue
        if "build_corpus.py" in cmdline and process.info.get("name") in {
            "python.exe",
            "python3.11.exe",
        }:
            found.append(process)
    return found


def wait_for_parsing(chain_pid: int | None) -> None:
    if chain_pid and psutil.pid_exists(chain_pid):
        log(f"Жду завершения цепочки PID {chain_pid}.")
        while psutil.pid_exists(chain_pid):
            time.sleep(30)
    log("Цепочка завершилась; проверяю, не осталось ли парсеров.")
    quiet = 0
    while quiet < 2:
        active = _build_processes()
        if active:
            quiet = 0
            log(
                "Ещё работают парсеры: "
                + ", ".join(f"{item.pid}" for item in active)
            )
            time.sleep(30)
        else:
            quiet += 1
            time.sleep(30)


def run_verification() -> tuple[bool, str]:
    command = [
        str(PROJECT / ".venv" / "Scripts" / "python.exe"),
        "-X",
        "utf8",
        str(PROJECT / "scripts" / "verify_corpus.py"),
        "--json",
        str(REPORT_JSON),
        "--markdown",
        str(REPORT_MD),
    ]
    log("Запуск verify_corpus.py")
    result = subprocess.run(
        command,
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    details = (result.stdout + "\n" + result.stderr).strip()
    try:
        payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        summary = payload.get("summary") if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        summary = {}
    documents = int((summary or {}).get("documents_with_chunks") or 0)
    errors = int((summary or {}).get("error_count") or 0)
    is_valid = bool((summary or {}).get("is_valid"))
    ok = is_valid and errors == 0 and documents >= 10
    log(
        f"verify_corpus: exit={result.returncode}, valid={is_valid}, "
        f"documents={documents}/9, errors={errors}, "
        f"missing_chunks={len((summary or {}).get('documents_missing_chunks') or [])}"
    )
    if not ok:
        log("Проверка НЕ пройдена; выключение отменено.")
        log(details[-4000:])
    return ok, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-pid", type=int, default=None)
    parser.add_argument("--delay", type=int, default=60,
                        help="Задержка перед shutdown, секунды")
    args = parser.parse_args()

    log("Watcher запущен.")
    wait_for_parsing(args.chain_pid)
    ok, _details = run_verification()
    if not ok:
        return 1

    log(f"Проверка успешна. Выключаю ПК через {args.delay} с.")
    result = subprocess.run(
        [
            "shutdown",
            "/s",
            "/t",
            str(max(0, args.delay)),
            "/c",
            "Parsing verified by verify_corpus.py",
        ],
        check=False,
    )
    log(f"shutdown exit={result.returncode}")
    if result.returncode != 0:
        log("shutdown.exe не сработал; пробую Stop-Computer -Force.")
        fallback = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Stop-Computer -Force",
            ],
            check=False,
        )
        log(f"Stop-Computer exit={fallback.returncode}")
        return 0 if fallback.returncode == 0 else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
