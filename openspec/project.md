# Project Context

## Overview

**Sberbank Annual Reports Graph RAG (2015–2024)** — Structure-Aware Graph RAG для анализа
9 годовых отчётов Сбербанка с temporal-навигацией и пониманием структуры документа
(Части → Главы → Разделы → Подразделы). Отличие от baseline-курса: система знает,
**где в документе** находится информация, и использует это для точности retrieval.

- Спецификация проекта: `spec_project.md` (v3.0)
- Деталь фичи парсинга: `spec_pdf_parser.md` (v1.0, справочный документ — артефакты
  повествуют про change `add-pdf-parser` в OpenSpec)

## Stack

| Слой | Технология |
|---|---|
| Парсинг PDF | `unstructured[all-docs]` 0.27+ (+ явный `unstructured-inference>=1.6`), стратегия `hi_res`, OCR tesseract 5.4 (`rus`, `eng`), `TESSDATA_PREFIX` → `<project>/tessdata` |
| RAG-фреймворк | LlamaIndex (`PropertyGraphIndex`, graph + vector hybrid) |
| Граф | `networkx` (in-memory) |
| Векторы | `chromadb` |
| Эмбеддинги | `sentence-transformers/paraphrase-multilingual` (контекст: `EMBED_QUERY_PREFIX=query:`) |
| Reranker | `CrossEncoder cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM / VLM | GigaChat-3-Pro через OpenAI-совместимый API: `base_url=https://foundation-models.api.cloud.ru/v1`, ключ в `.env` (`LLM_API_KEY`). Отдельного пакета `llama-index-llms-yandex` на PyPI не существует — только `llama-index-llms-openai`. |
| UI | Streamlit + streamlit-agraph |
| Eval | RAGAS (известная проблема: ragas 0.4.3 несовместим с langchain-community 0.4.x — решается пиннингом community <0.4) |
| Python | venv `.venv` на 3.11.9 (llama-index-интеграции адаптеров требуют <3.12) |

## Code Map

```
course/       — код курса (контракт чанка, DenseIndex, filters, embeddings, prompts): НЕ редактируем
src/ingestion/  — pdf_parser.py, structure_builder.py, chunk_enricher.py, table_structurer.py, visual_describer.py
src/graph/      — kg_extractor.py, graph_builder.py (PropertyGraphIndex + DocumentNode), community.py
src/retrieval/  — query_router.py, structure_filter.py, temporal_filter.py, reranker.py, graph_retriever.py, hybrid_retriever.py
src/engine/     — graph_rag_engine.py (наследник курсовского RAGEngine)
src/ui/         — app.py, graph_view.py, structure_tree.py, timeline_component.py
data/           — raw/ (9 PDF, только чтение), elements/ (артефакты парсера), figures/ (вырезы Image hi_res по doc_id), chunks/, document_structure/, cache/, graph/
scripts/        — проверочные скрипты (test_hi_res.py, test_llm.py)
```

## Data

- `data/raw/Копия Сбер {2015–2020, 2022–2024}.pdf` (9 файлов, ~168 МБ, 2021 отсутствует физически)
- Тест-контроль качества парсинга: на 3 страницах отчёта 2015 — 1143 элемента
  (421 NarrativeText, 279 Text, 133 ListItem, 103 Image, 100 Title, 70 Header, 25 Table, 12 FigureCaption)

## Conventions

- Контракт чанка (из курса, `REQUIRED_CHUNK_FIELDS`) — не менять; structure-поля (`section_path`,
  `breadcrumbs`, `section_level`, `parent_section`) добавляются опционально.
- `doc_id` из имени файла: `sber_{year}` (год только из имени файла).
- Docstrings Google-style, type hints, pydantic-settings для конфигурации.
- Курсовой код в `course/` — только чтение; все расширения в `src/`.

## Commands

- `.venv\Scripts\python.exe -X utf8 scripts\test_hi_res.py` — smoke парсинга (3 страницы 2015)
- `.venv\Scripts\python.exe -X utf8 scripts\test_llm.py` — smoke LLM (GigaChat)
- CLI парсера (после реализации): `python -m src.ingestion.pdf_parser --pdf|--all|--stats|--fast-preview --smoke`
- `openspec validate --all --strict` — валидация OpenSpec-артефактов
