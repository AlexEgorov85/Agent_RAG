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
| Векторы | Постоянный FAISS `IndexFlatIP` (`data/indexes/{fingerprint}`) + in-memory `DenseIndex` как fallback |
| Эмбеддинги | `ai-sage/Giga-Embeddings-instruct-3B` через Cloud.ru (курс, Занятие 3; Instruct/Query `query_prefix`) |
| Reranker | LLM `ai-sage/GigaChat3-10B-A1.8B`, pointwise scores 0–3 (курс, Занятие 5; не CrossEncoder) |
| LLM / VLM | GigaChat-3-Pro через OpenAI-совместимый API: `base_url=https://foundation-models.api.cloud.ru/v1`, ключ в `.env` (`LLM_API_KEY`) |
| UI | Streamlit (legacy) + FastAPI adapter + React/TypeScript/Vite WebUI |
| История | SQLite (`var/api.sqlite3`) для диалогов; отдельная SQLite для evaluation |
| Eval | `ranking_metrics` + LLM-судья (курс, Занятие 8); RAGAS — только опциональная перекрёстная проверка |
| Python | целевая и проверенная среда `.venv` на Python 3.11.9; Node LTS ≥ 20 для frontend |

## Current Implementation (2026-11)

- Реализованы: PDF parser (hi_res/fast, OCR, батчи, resume по SHA-256 и подписи
  параметров), outline reader, structure builder (закладки → TOC → эвристика),
  chunk builder/enricher, CLI, реестр источников и скрипты сборки/валидации
  корпуса. Канонические артефакты готовы для 2015–2018: 3928 чанков, 98% со
  структурой. Документы 2019–2024 не парсились.
- Реализованы: embeddings/indexes, BM25/hybrid retrieval, RRF, reranker, query
  rewriting, diversity, graph builder/retriever, graph-augmented поиск.
  Dense-индекс — постоянный FAISS `IndexFlatIP` (`data/indexes/{fingerprint}/`),
  с fallback на in-memory `DenseIndex`; сборка — `scripts/build_indexes.py`.
  Полный graph index не построен (≈982 LLM-пакета на 3928 чанков), используется
  dev-граф.
- Реализованы: answer engine (context packing, citations, abstain, расчёты),
  LangGraph-агент с LLM/Simple router, selector, CoverageReport, verification и
  repair, temporal/structure analysis, retrieval fallback для structural-запросов,
  retrieval trace stages и diagnostics.
- Реализованы: Streamlit UI (чат, citations, coverage, verification, стадии
  retrieval, structure, graph-only evidence, ссылки на страницу PDF).
- Реализованы: HTTP API (`src/api/`: CRUD, messages, SSE с replay/resync,
  evidence, PDF `inline`, capabilities, SQLite-история, cookie-владелец) и React
  WebUI этапов 1–4: чат на assistant-ui, история, этапы, tool trace, evidence,
  кликабельные citations `[n]`, PDF-ссылки `#page=N`, таблицы/графики/CSV,
  структурное сравнение, отмена и восстановление. Этап 5 закрыт частично:
  Streamlit не удалён; пагинация истории и часть a11y закрыты, vendor-split
  убрал warning >500 kB.
- Реализован Evaluation API (`/api/evaluation/*`: suites, runs, tests, trace,
  report, cancel, compare) и React-раздел `/evaluation` (обзор, запуски, тесты,
  сравнение) с навигацией чат↔Evaluation. Раздел выключен по умолчанию
  (`EVAL_UI_ENABLED=0`), прогресс — polling, SSE-канала нет.
- Тесты: полный `pytest` — `854 passed` в `.venv` (Python 3.11.9, снимок
  15.11.2026), frontend — `160 passed` в 24 файлах Vitest, Playwright E2E —
  `7/7` (5 чат + 2 Evaluation); `tsc -b` проходит,
  `openspec validate --all --strict` проходит.
- Приёмка и benchmark: реальная dev-приёмка 4 сценария — `4/4` (factoid,
  temporal, structural, abstain; снимок 02:44, после P1/P2 не перепрогонялась).
  Датасет benchmark — 30 тестов (10/10/5/5), 12 smoke и 6 holdout. Реальный
  smoke — `12/12` (Recall@5 0.955). Последний полный реальный прогон 30 тестов —
  `23/30`, `pass_rate 0.767`, execution coverage 1.0, Recall@5 0.889,
  Recall@10 0.926, MRR 0.889 (easy 100%, medium 80%, hard 40%, multi_turn 60%;
  dev 19/24, holdout 4/6), артефакт
  `eval/results/20261115_183541_0a97989/`. Предыдущий прогон — `24/30`
  (`eval/results/20261115_174124_0a97989/`), per-test стабильность не доказана
  (`repeat=1`). В последнем прогоне есть `holdout_gold_overlap` в 4 тестах, а
  `context_chunk_ids` доступны не для всех кейсов.
- Не завершены: полный корпус из 9 PDF (2019–2024 не парсились), полный graph
  index (`b288dd3c…` — 1916/3928, `status=incomplete`), прогон benchmark на
  полном корпусе и `expected_corpus_sha256` для него, judge-калибровка и
  опциональный Evaluation SSE; issuer-фильтр и изоляция 2015 реализованы.
- Имеются raw PDF; `data/` игнорируется Git. Не считать полный корпус или
  production-ready RAG существующим до проверки артефактов и прохождения
  критериев `spec_target.md`; benchmark 23/30 на dev-корпусе не является
  приёмкой полного продукта.

## Target Code Map

```
course/       — учебные материалы (ноутбуки); production modules должны быть в src/
src/ingestion/  — реализуемые parser/outline/structure/chunk stages
src/graph/      — graph schema, builder, entity extraction
src/retrieval/  — dense/BM25, RRF, router, filters, reranker, graph retriever
src/engine/     — context packing, prompts, RAGEngine/GraphRAGEngine
src/agent/      — ограниченный LangGraph агент, state, tools, streaming service
src/api/        — FastAPI adapter: чат/SSE/evidence + Evaluation API
src/ui/         — Streamlit app, citations, structure, graph, timeline
frontend/       — React + TypeScript WebUI (этапы 1–4, часть 5 + Evaluation UI)
eval/           — dataset, metrics, judge, reproducible runner
data/           — raw/ (9 PDF, только чтение), elements/ (артефакты парсера), figures/ (вырезы Image hi_res по doc_id), chunks/, structure/, cache/, graph/
scripts/        — сборка корпуса/графа, проверки, dev-вопросы, smoke-тесты API
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
- Сборка корпуса: `python scripts/build_corpus.py [--status-only|--doc NAME]`
- Проверка корпуса: `python scripts/verify_corpus.py`
- FAISS-индекс: `python scripts/build_indexes.py [--dry-run|--reuse-cache-only]`
- Граф: `python scripts/build_full_graph.py [--dry-run|--workers N]`
- Benchmark: `python -m eval.runner --suite smoke --split dev [--fake-service]`;
  полный набор: `python -m eval.runner --suite full --no-judge`
- Приёмка dev: `python scripts/acceptance_dev.py`
- Streamlit: `streamlit run src/ui/app.py`
- HTTP API: `python -m uvicorn src.api.app:app --port 8000`
- React WebUI: `cd frontend && npm install && npm run dev`
- Тесты: `python -m pytest`; frontend — `cd frontend && npm run test:run && npm run e2e`
- `openspec validate --all --strict` — валидация OpenSpec-артефактов
