# Proposal: add-pdf-parser

## Why

Девять PDF-отчётов Сбербанка — единственный источник данных проекта, но сейчас их можно
читать только вручную, ad-hoc скриптом (`scripts/test_hi_res.py` hacked на 3 страницы).
Все последующие этапы (дерево разделов, чанки, facts из таблиц, описания графиков, Knowledge Graph)
наступают на артефакты парсера, поэтому нужен повторяемый, resumable, документированный
слой извлечения с фиксированным контрактом входа/выхода.

## What Changes

- Новый модуль `src/ingestion/pdf_parser.py`: обёртка над `unstructured.partition_pdf协定`
  (strategy `hi_res`, OCR `rus+eng`, `infer_table_structure`, батчи по страницам, resume).
- Новый модуль `src/core/models.py` — dataclass `RawElement` (нормализованный элемент)
  и `src/core/config.py` (расширенный pydantic-settings из курса).
- Новый промежуточный data-слой `data/elements/`: `{doc_id}_elements.json` + `{doc_id}_stats.json`
  (UTF-8, ensure_ascii=False), со `source_sha256` для будущего chunk-контракта.
- CLI `python -m src.ingestion.pdf_parser` с режимами `--pdf`, `--all`, `--stats`, `--fast-preview`, `--smoke`.
- Фильтр мелких изображений `< 200×200 px` (не кладём их в JSON, VLM их не увидит).
- Резилиентность: resume по `source_sha256`, backup при изменении источника, батчевая ошибка не роняет файл.

## Capabilities

### New Capabilities

- `pdf-parsing`: детерминированное извлечение типизированных элементов из PDF-отчётов
  в JSON-артефакты с полными метаданными, resume и контролем версий источника.

### Modified Capabilities

(пусто — capability новая)

## Impact

- **Код (новый):** `src/ingestion/pdf_parser.py`, `src/core/models.py`, `src/core/config.py`
- **Данные (новый):** `data/elements/*`; каталоги `data/raw/`, `tessdata/` — только для чтения
- **Зависимости:** добавляется `unstructured-inference>=1.6` в `requirements.txt` (явно — в ново
  unstructured он вне extras)
- **Потребители (будущие changes):** `structure_builder` (Title+category_depth), `chunk_enricher`
  (все элементы+sha256), `table_structurer` (Table+html), `visual_describer` (Image+b64)
- **Курсовой код `course/`:** не затрагивается
