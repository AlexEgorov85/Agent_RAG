"""Тонкий alias ``python -m eval.run_eval`` → ``eval.runner``.

Канонический запуск — ``python -m eval.runner``; alias сохраняется для
совместимости с ``spec_ui_eval.md`` и старыми инструкциями.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from eval.runner import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


def run(argv: Sequence[str] | None = None) -> int:
    return main(argv)
