# -*- coding: utf-8 -*-
"""Структурный индекс табличных значений: (год → числа из таблиц отчёта).

Проблема, которую решает модуль: парсер сохраняет таблицы как HTML с шапкой
вида ``<th>2019 года</th><th>2018 года</th>``, но текстовый чанк содержит
только строку чисел «845,0 831,7». По такому тексту нельзя понять, какое
число относится к какому году, поэтому модель приписывает 831,7 (2018 год) к
2020-му. Здесь шапка используется по назначению: для каждого чанка-таблицы
строится отображение «год → значения», по которому проверяются числа в ответе.

Индекс не вызывает LLM и не угадывает значения. Если шапки с годами нет, годы
могут быть выведены из текста перед таблицей; такие ячейки помечаются
``inferred`` и не участвуют в каноническом отображении «метрика → год».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "metrics" / "metrics.json"

_YEAR_IN_TEXT = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_NUMBER = re.compile(r"^\(?-?\d[\d\s\u00a0]*(?:[.,]\d+)?\)?$")
_METRIC_LABEL_EXCLUDE = re.compile(
    r"акци|дивиденд|сотрудник|клиент|офис|отделени|банкомат|"
    r"на одну|прибыль на|количество|численност|доля|процент\s+п\.?п",
    re.I,
)


@dataclass(frozen=True)
class MetricCell:
    """Одно число таблицы с привязкой к году, документу и чанку."""

    doc_id: str
    year: int
    value: float
    row_label: str
    unit: str | None
    chunk_id: str
    page_pdf: int | None = None
    section_path: str | None = None
    #: Канонический ключ метрики из ``coverage.normalize_metric``, если строка
    #: однозначно опознана как показатель (``net_profit``, ``roe``, ...).
    metric: str | None = None
    #: Год взят из текста перед таблицей, а не из шапки: такие ячейки слабее.
    inferred: bool = False
    #: Значение есть в текстовой версии чанка — её и цитирует ответ.
    in_text: bool = False


class _TableParser(HTMLParser):
    """Собрать строки таблицы: отдельно ``thead`` и ``tbody``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.head_rows: list[list[str]] = []
        self.body_rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] = []
        self._in_head = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        """Обработать открывающий тег таблицы."""
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:  # noqa: ANN001
        """Обработать закрывающий тег."""
        if tag == "thead":
            self._in_head = False
        elif tag in {"td", "th"} and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = []
        elif tag == "tr" and self._row is not None:
            target = self.head_rows if self._in_head else self.body_rows
            target.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        """Накопить текст ячейки."""
        if self._row is not None:
            self._cell.append(data)


def parse_table(
    html: str, *, text_hint: str = ""
) -> tuple[list[int | None], list[list[str]], bool]:
    """Вернуть (годы по колонкам, строки данных, годы выведены из текста).

    ``years[i]`` — год для i-й колонки или ``None``. Если ``thead`` не содержит
    годов, он считается первой строкой данных (так бывает, когда строка годов
    в PDF вынесена над таблицей), а годы берутся из ``text_hint`` — текста
    чанка, где они идут перед значениями («2023 2024 2025»).
    """
    parser = _TableParser()
    parser.feed(html)
    years: list[int | None] = []
    for row in parser.head_rows:
        for index, cell in enumerate(row):
            match = _YEAR_IN_TEXT.search(cell)
            if match is None:
                if index >= len(years):
                    years.append(None)
                continue
            while len(years) <= index:
                years.append(None)
            years[index] = int(match.group(1))
    rows = list(parser.body_rows)
    inferred = False
    if not any(year is not None for year in years):
        # Шапка без годов: это данные, а годы — в тексте перед таблицей.
        header_as_data = [row for row in parser.head_rows if any(row)]
        rows = header_as_data + rows
        hint_years: list[int] = []
        for match in _YEAR_IN_TEXT.finditer(text_hint[:400]):
            year = int(match.group(1))
            if year not in hint_years:
                hint_years.append(year)
        if len(hint_years) >= 2 and rows:
            value_columns = _value_columns(rows)
            if len(value_columns) >= len(hint_years):
                years = [None] * (max(value_columns) + 1)
                for year, column in zip(sorted(hint_years), value_columns):
                    years[column] = year
                inferred = True
            else:
                years = []
    return years, rows, inferred


def _value_columns(rows: list[list[str]]) -> list[int]:
    """Индексы колонок со значениями: числа, но не проценты изменения."""
    columns: list[int] = []
    for row in rows:
        for index, cell in enumerate(row):
            if index in columns:
                continue
            value = parse_number(cell)
            if value is None or "%" in cell:
                continue
            columns.append(index)
        if len(columns) >= 2:
            break
    return sorted(columns)


def parse_number(cell: str) -> float | None:
    """Разобрать число таблицы."""
    text = cell.replace("\u00a0", " ").strip()
    if not text or not _NUMBER.match(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(" ", "").replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _unit_from_header(header_row: Iterable[str]) -> str | None:
    for cell in header_row:
        folded = cell.casefold()
        if "млрд" in folded or "миллиард" in folded:
            return "млрд руб."
        if "млн" in folded or "миллион" in folded:
            return "млн руб."
        if "трлн" in folded:
            return "трлн руб."
    return None


def _metric_for_label(label: str) -> str | None:
    """Канонический ключ метрики, только если строка — сам показатель.

    Подпись таблицы бывает длинной («Прибыль на акцию», «Чистая прибыль
    Группы по МСФО»). Канонизируем лишь короткие подписи, у которых после
    удаления служебных слов остаётся точное название метрики.
    """
    if not label or len(label) > 60 or _METRIC_LABEL_EXCLUDE.search(label):
        return None
    from src.agent.coverage import METRIC_ALIASES

    cleaned = label.casefold()
    for noise in (
        "группы", "группе", "банка", "сбербанка", "сбера", "по мсфо",
        "за год", "за года", "года", "год", "итогам", "млрд", "млн",
        "трлн", "руб.", "рублей", "руб", "%", ":", "—", "-",
    ):
        cleaned = cleaned.replace(noise, " ")
    cleaned = " ".join(cleaned.split())
    for metric, aliases in METRIC_ALIASES.items():
        if cleaned in aliases:
            return metric
    return None


def extract_cells(chunk: Mapping[str, Any]) -> list[MetricCell]:
    """Все значения годовых колонок одной таблицы-чанка."""
    if str(chunk.get("kind") or "") != "table":
        return []
    html = chunk.get("html") or ""
    if not html:
        return []
    years, rows, inferred = parse_table(
        str(html), text_hint=str(chunk.get("text") or ""))
    if not any(year is not None for year in years):
        return []
    unit = _unit_from_header(
        cell for cell in (chunk.get("text") or "").split("\n")[:1]
    )
    chunk_id = str(chunk.get("chunk_id") or "")
    doc_id = str(chunk.get("doc_id") or "")
    page = chunk.get("page_pdf")
    cells: list[MetricCell] = []
    for row in rows:
        if not row:
            continue
        label = next((cell for cell in row if cell.strip()), "")
        for index, year in enumerate(years):
            if year is None or index >= len(row):
                continue
            value = parse_number(row[index])
            if value is None:
                continue
            cells.append(
                MetricCell(
                    doc_id=doc_id,
                    year=year,
                    value=value,
                    row_label=label,
                    unit=unit,
                    chunk_id=chunk_id,
                    page_pdf=int(page) if isinstance(page, int) else None,
                    section_path=chunk.get("section_path"),
                    metric=_metric_for_label(label),
                    inferred=inferred,
                )
            )
    return cells


@dataclass
class MetricsIndex:
    """Индекс: значения по годам и канонические показатели по (метрика, год)."""

    #: год → список значений (для проверки «есть ли такое число за этот год»).
    values_by_year: dict[int, list[float]] = field(default_factory=dict)
    #: metric → год → значение и его источник.
    by_metric: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    cell_count: int = 0
    documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-представление индекса."""
        return {
            "schema_version": "1.0",
            "cell_count": self.cell_count,
            "documents": list(self.documents),
            "values_by_year": {
                str(year): sorted(set(values))
                for year, values in sorted(self.values_by_year.items())
            },
            "by_metric": self.by_metric,
            "chunks": self.chunks,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricsIndex":
        """Восстановить индекс из JSON."""
        return cls(
            values_by_year={
                int(year): [float(value) for value in values]
                for year, values in (payload.get("values_by_year") or {}).items()
            },
            by_metric={
                str(metric): {int(year): cell for year, cell in per_year.items()}
                for metric, per_year in (payload.get("by_metric") or {}).items()
            },
            chunks=dict(payload.get("chunks") or {}),
            cell_count=int(payload.get("cell_count") or 0),
            documents=list(payload.get("documents") or []),
        )


def _doc_year(doc_id: str) -> int | None:
    match = re.search(r"(20\d{2})", doc_id or "")
    return int(match.group(1)) if match else None


def build_index(chunks: Iterable[Mapping[str, Any]]) -> MetricsIndex:
    """Построить индекс по чанкам."""
    index = MetricsIndex()
    documents: set[str] = set()
    #: (метрика, год) → значения-кандидаты из таблиц с явной шапкой годов.
    candidates: dict[tuple[str, int], list[tuple[tuple[int, int], MetricCell]]] = {}
    for chunk in chunks:
        cells = extract_cells(chunk)
        if not cells:
            continue
        documents.add(str(chunk.get("doc_id") or ""))
        for cell in cells:
            index.cell_count += 1
            index.values_by_year.setdefault(cell.year, []).append(cell.value)
            index.chunks.setdefault(cell.chunk_id, {
                "doc_id": cell.doc_id,
                "page_pdf": cell.page_pdf,
                "section_path": cell.section_path,
            })
            if cell.metric is None or cell.inferred:
                # Год из текста перед таблицей — подсказка, а не канон.
                continue
            score = (
                0 if cell.in_text else 1,
                0 if not cell.inferred else 1,
                0 if _doc_year(cell.doc_id) == cell.year else 1,
            )
            candidates.setdefault((cell.metric, cell.year), []).append(
                (score, cell)
            )
    for (metric, year), items in candidates.items():
        counts: dict[float, int] = {}
        for _score, cell in items:
            counts[cell.value] = counts.get(cell.value, 0) + 1
        # Побеждает значение, которое повторяется в отчётах чаще; при равенстве —
        # ячейка из отчёта того же года.
        best_value = max(
            counts,
            key=lambda value: (
                counts[value],
                -min(
                    score[0] for score, cell in items if cell.value == value
                ),
            ),
        )
        best_cell = min(
            (item for item in items if item[1].value == best_value),
            key=lambda item: item[0],
        )[1]
        index.by_metric.setdefault(metric, {})[year] = {
            "value": best_cell.value,
            "unit": best_cell.unit,
            "chunk_id": best_cell.chunk_id,
            "doc_id": best_cell.doc_id,
            "row_label": best_cell.row_label,
            "page_pdf": best_cell.page_pdf,
            "inferred_year": False,
        }
    index.documents = sorted(doc for doc in documents if doc)
    return index


def save_index(index: MetricsIndex, path: Path | str = DEFAULT_METRICS_PATH) -> Path:
    """Атомарно сохранить индекс."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return target


def reference_rows(plan: Any) -> list[dict[str, Any]]:
    """Значения запрошенных метрик за запрошенные годы из таблиц отчётов.

    Возвращает только ячейки с явной шапкой года, поэтому значение нельзя
    спутать с соседней колонкой. Используется и промптом (справочник), и
    сборкой evidence (чанк таблицы становится цитируемым источником).
    """
    index = load_index()
    if index is None or plan is None:
        return []
    metrics = [
        str(metric)
        for metric in (getattr(plan, "required_metrics", None) or [])
        if str(metric).strip()
    ]
    years = sorted({
        int(year)
        for year in (getattr(plan, "years", None) or [])
        if isinstance(year, int) or str(year).isdigit()
    })
    if not metrics or not years:
        return []
    excluded = {
        int(year)
        for year in (getattr(plan, "missing_years", None) or [])
        if isinstance(year, int) or str(year).isdigit()
    }
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for year in years:
            if year in excluded:
                continue
            cell = index.by_metric.get(metric, {}).get(year)
            if not cell:
                continue
            rows.append({
                "metric": metric,
                "year": year,
                "value": cell.get("value"),
                "unit": cell.get("unit"),
                "chunk_id": cell.get("chunk_id"),
                "page_pdf": cell.get("page_pdf"),
            })
    return rows


@lru_cache(maxsize=1)
def load_index(path: str | None = None) -> MetricsIndex | None:
    """Прочитать индекс; ``None``, если артефакт ещё не собран."""
    target = Path(path) if path else DEFAULT_METRICS_PATH
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return MetricsIndex.from_dict(payload)
