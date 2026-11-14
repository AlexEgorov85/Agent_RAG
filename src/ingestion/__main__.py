"""Точка входа пакета ingestion: ``python -m src.ingestion <режим>``."""

from __future__ import annotations

import sys

from src.ingestion.cli import main

if __name__ == "__main__":
    sys.exit(main())
