# -*- coding: utf-8 -*-
"""Обновить счётчики корпуса в документации после добавления 2025."""

from pathlib import Path

FILES = ["README.md", "TODO_AGENT_RAG.md", "spec_target.md", "openspec/project.md"]
REPLACEMENTS = {
    "12044": "13969",
    "8116": "10041",
    "9/9": "10/10",
    "9 PDF": "10 PDF",
    "9 документов": "10 документов",
    "2022–2024": "2022–2025",
    "2581": "2995",
    "0 ошибок, 3 warning": "0 ошибок, 4 warning",
    "3 warnings": "4 warnings",
}


def main() -> None:
    for name in FILES:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8", newline="\n")
    print("updated", ", ".join(FILES))


if __name__ == "__main__":
    main()
