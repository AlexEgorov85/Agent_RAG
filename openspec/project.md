# Project Context

> Актуальная целевая спецификация: [`spec_target.md`](../spec_target.md).
> Профильные контракты: `spec_agent.md`, `spec_retrieval_engine.md`, `spec_graph.md`,
> `spec_ui_eval.md`; parser contract: `spec_pdf_parser.md`.
> Этот OpenSpec context — краткая справка, не источник статуса реализации.

## Overview

**Sberbank Annual Reports Graph RAG (2015–2024)** — Structure-Aware Graph RAG для анализа
9 годовых отчётов Сбербанка с temporal-навигацией и пониманием структуры документа
(Части → Главы → Разделы → Подразделы). Отличие от baseline-курса: система знает,
**где в документе** находится информация, и использует это для точности retrieval.

- Исторический дизайн проекта: `spec_project.md` (v3.1)
- Целевой контракт/приёмка: `spec_target.md`
- Деталь фичи парсинга: `spec_pdf_parser.md` (v1.1)

## Stack

| Слой | Технология |
|---|---|
| Парсинг PDF | `unstructured[all-docs]` 0.27+ (+ явный `unstructured-inference>=1.6`), стратегия `hi_res`, OCR tesseract 5.4 (`rus`, `eng`), `TESSDATA_PREFIX` → `<project>/tessdata` |
| Agent orchestration | Ограниченный LangGraph `ReportAgent` поверх `RAGEngine`; только allowlisted tools |
| Граф | `networkx.MultiDiGraph` (MVP, граф — дополнительный retriever) |
| Векторы | NumPy DenseIndex; persistent vector DB — только при измеренной необходимости |
| Эмбеддинги | `ai-sage/Giga-Embeddings-instruct-3B` через Cloud.ru (курс, Занятие 3; `EMBED_QUERY_PREFIX=query:`) |
| Reranker | LLM `ai-sage/GigaChat3-10B-A1.8B`, pointwise scores 0–3 (курс, Занятие 5; не CrossEncoder) |
| LLM / VLM | GigaChat-3-Pro через OpenAI-совместимый API: `base_url=https://foundation-models.api.cloud.ru/v1`, ключ в `.env` (`LLM_API_KEY`) |
| UI | Streamlit + streamlit-agraph |
| Eval | `ranking_metrics` + LLM-судья (курс, Занятие 8); RAGAS — только опциональная перекрёстная проверка |
| Python | целевая и проверенная среда `.venv` на Python 3.11.9 |

## Current Implementation (2026-11)

- Реализованы: PDF parser, outline reader, structure builder, chunk builder/enricher, CLI.
- Не реализованы: embeddings/indexes, BM25/hybrid retrieval, reranker, query router,
  answer engine, graph builder/retriever, UI, evaluation.
- Имеются raw PDF; `data/` игнорируется Git. Не считать полный корпус или production-ready
  RAG существующим до проверки артефактов и прохождения критериев `spec_target.md`.

## Target Code Map

```
course/       — учебные материалы (ноутбуки); production modules должны быть в src/
src/ingestion/  — реализуемые parser/outline/structure/chunk stages
src/graph/      — graph schema, builder, entity extraction
src/retrieval/  — dense/BM25, RRF, router, filters, reranker, graph retriever
src/engine/     — context packing, prompts, RAGEngine/GraphRAGEngine
src/agent/      — ограниченный LangGraph агент, state, tools, streaming service
src/ui/         — Streamlit app, citations, structure, graph, timeline
eval/           — dataset, metrics, judge, reproducible runner
data/           — raw/ (9 PDF, только чтение), elements/ (артефакты парсера), figures/ (вырезы Image hi_res по doc_id), chunks/, document_structure/, cache/, graph/
scripts/        — проверочные скрипты (test_hi_res.py, test_llm.py)
```

## Data

- `data/raw/Копия Сбер {2015–2020, 2022–2024}.pdf` (9 файлов, ~168 МБ, 2021 отсутствует физически)
- Тест-контроль качества парсинга: полный отчёт 2015 (74 страницы) — 1143 элемента
  (421 NarrativeText, 279 Text, 133 ListItem, 103 Image, 100 Title, 70 Header, 25 Table, 12 FigureCaption)

## Conventions

- Контракт чанка (из курса, `REQUIRED_CHUNK_FIELDS`) — не менять; structure-поля (`section_path`,
  `breadcrumbs`, `section_level`, `parent_section`) добавляются опционально.
- `doc_id` из имени файла: `sber_{year}` (год только из имени файла); issuer — отдельное
  поле метаданных, не выводится из `doc_id`.
- Docstrings Google-style, type hints, pydantic-settings для конфигурации.
- Курсовой код в `course/` — только чтение; все расширения в `src/`.

## Commands

- `.venv\Scripts\python.exe -X utf8 scripts\test_hi_res.py` — smoke парсинга (3 страницы 2015)
- `.venv\Scripts\python.exe -X utf8 scripts\test_llm.py` — smoke LLM (GigaChat)
- CLI парсера: `python -m src.ingestion --pdf|--all|--stats|--fast-preview|--smoke`
- `openspec validate --all --strict` — валидация OpenSpec-артефактов
