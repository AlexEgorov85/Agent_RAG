# -*- coding: utf-8 -*-
"""Короткий отчёт только по локальному парсингу PDF: прогресс и скорость."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import psutil  # noqa: E402

DOCS = [
    "sber_2015",
    "sber_2016",
    "sber_2017",
    "sber_2018",
    "sber_2019",
    "sber_2020",
    "sber_2022",
    "sber_2023",
    "sber_2024",
]
TARGETS = ["sber_2019", "sber_2020", "sber_2022", "sber_2023", "sber_2024"]
STATE_PATH = PROJECT / "logs" / "parse_status_state.json"


def _pages() -> dict[str, int]:
    try:
        payload = json.loads(
            (PROJECT / "data" / "corpus_report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(item.get("doc_id")): int(item.get("n_pages") or 0)
        for item in payload.get("documents", [])
        if isinstance(item, dict)
    }


def _is_chunks(doc_id: str) -> bool:
    return (
        (PROJECT / "data" / "elements" / f"{doc_id}_elements.json").is_file()
        and (PROJECT / "data" / "structure" / f"{doc_id}_sections.json").is_file()
        and (PROJECT / "data" / "chunks" / f"{doc_id}_chunks.json").is_file()
    )


def _current_parser() -> psutil.Process | None:
    candidates: list[psutil.Process] = []
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError):
            continue
        if "build_corpus.py" in cmdline and process.info.get("name") in {
            "python.exe",
            "python3.11.exe",
        }:
            candidates.append(process)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item.cpu_times().user + item.cpu_times().system,
    )


def _page(pid: int) -> int | None:
    spy = PROJECT / ".venv" / "Scripts" / "py-spy.exe"
    if not spy.is_file():
        return None
    try:
        result = subprocess.run(
            [str(spy), "dump", "--pid", str(pid), "--locals"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    starts = [int(value) for value in re.findall(r"starting_page_number:\s*(\d+)", result.stdout)]
    local_pages = [int(value) for value in re.findall(r"page_(\d+)\.png", result.stdout)]
    start = max(starts) if starts else None
    local_page = max(local_pages) if local_pages else None
    if start is not None and local_page is not None:
        return start + local_page - 1
    return start


def _current_doc(process: psutil.Process) -> str | None:
    cmdline = process.cmdline()
    for index, item in enumerate(cmdline):
        if item in {"--doc", "-d"} and index + 1 < len(cmdline):
            return cmdline[index + 1]
    return None


def _load_state() -> dict[str, object]:
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_state(doc_id: str | None, page: int | None) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "at": datetime.now().isoformat(timespec="seconds"),
                "doc_id": doc_id,
                "page": page,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fmt_minutes(value: float) -> str:
    minutes = max(0, int(round(value)))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} ч {minutes} мин"
    return f"{minutes} мин"


def main() -> int:
    now = datetime.now()
    pages = _pages()
    total_pages = sum(pages.get(doc_id, 0) for doc_id in DOCS)
    done = [doc_id for doc_id in DOCS if _is_chunks(doc_id)]
    done_pages = sum(pages.get(doc_id, 0) for doc_id in done)
    process = _current_parser()
    doc_id = _current_doc(process) if process is not None else None
    page = _page(process.pid) if process is not None else None

    print(f"# Парсинг — {now.strftime('%H:%M')}")
    if process is None:
        print("Активный парсер не найден.")
        print(f"Готово: {len(done)}/9 документов, {done_pages}/{total_pages} страниц.")
        _save_state(None, None)
        return 0

    elapsed_min = max(
        0.1,
        (now.timestamp() - process.create_time()) / 60.0,
    )
    doc_total = pages.get(doc_id or "", 0)
    doc_page = min(page or 0, doc_total or (page or 0))
    speed = doc_page / elapsed_min if doc_page else 0.0

    previous = _load_state()
    interval_speed = None
    if (
        previous.get("doc_id") == doc_id
        and isinstance(previous.get("page"), int)
        and doc_page
        and previous["page"] < doc_page
    ):
        try:
            prev_at = datetime.fromisoformat(str(previous.get("at")))
            minutes = max(0.1, (now - prev_at).total_seconds() / 60.0)
            interval_speed = (doc_page - int(previous["page"])) / minutes
        except (ValueError, TypeError):
            interval_speed = None
    effective_speed = interval_speed if interval_speed else speed

    if doc_id:
        percent = f" ({doc_page * 100 / doc_total:.0f}%)" if doc_total else ""
        print(f"Текущий: {doc_id} — {doc_page}/{doc_total or '?'} стр.{percent}")
        print(
            f"Скорость: {effective_speed:.1f} стр/мин "
            f"(~{effective_speed * 60:.0f} стр/ч)"
        )
        if doc_total and effective_speed > 0:
            print(
                f"До конца документа: {doc_total - doc_page} стр., "
                f"ETA {_fmt_minutes((doc_total - doc_page) / effective_speed)}"
            )
    else:
        print("Текущий документ: не определён")

    parsed_pages = done_pages + doc_page
    print(
        f"Всего: {parsed_pages}/{total_pages} стр. "
        f"({parsed_pages * 100 / total_pages:.0f}%) | "
        f"готово {len(done)}/9 документов"
    )
    remaining = [item for item in TARGETS if not _is_chunks(item)]
    remaining_pages = sum(
        pages.get(item, 0) - (doc_page if item == doc_id else 0)
        for item in remaining
    )
    if effective_speed > 0:
        print(
            f"Осталось: {remaining_pages} стр., "
            f"ETA всего {_fmt_minutes(remaining_pages / effective_speed)}"
        )
    print("Осталось документов: " + (", ".join(remaining) if remaining else "нет"))
    _save_state(doc_id, doc_page or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
