# Sberbank Annual Reports Graph RAG

Structure-aware RAG по годовым отчётам Сбербанка за 2015–2020 и 2022–2025
(10 PDF-файлов, 2021 года в корпусе нет).

Система извлекает из PDF текст, таблицы и изображения, строит дерево разделов,
чанки и граф сущностей, затем отвечает на вопросы по найденным фрагментам и
возвращает проверяемые цитаты с `chunk_id`, страницей PDF и путём раздела.

Проект состоит из Python-ядра (ingestion, retrieval, graph, agent, engine),
трёх интерфейсов (Streamlit, React WebUI, HTTP API) и отдельного контура
benchmark/evaluation.

> **Статус на 2026-11-15.**
>
> - Backend-контур, LangGraph-агент, Streamlit UI, HTTP API, React WebUI
>   (этапы 1–4 и часть 5) и Evaluation API/UI реализованы. Evaluation-раздел
>   выключен по умолчанию (`EVAL_UI_ENABLED=0`).
> - Офлайн-набор: `863 passed` (Python 3.11.9, `.venv`), 160 frontend-тестов в
>   24 файлах Vitest, 7/7 Playwright E2E, `tsc -b` и `vite build` проходят.
> - Реальная dev-приёмка (снимок 15.11.2026 02:44) — 4/4 сценария: factoid,
>   temporal, structural, abstain.
> - Benchmark: датасет 30 тестов (10/10/5/5), из них 12 smoke и 6 holdout.
>   Реальный smoke — **12/12** (Recall@5 0.955), артефакт
>   `eval/results/20261115_164758_0a97989/`.
> - Последний полный реальный прогон 30 тестов — **23/30**, `pass_rate 0.767`,
>   execution coverage 1.0, Recall@5 0.889, Recall@10 0.926, MRR 0.889,
>   evidence coverage 0.897, context coverage 0.981; артефакт
>   `eval/results/20261115_183541_0a97989/`. Предыдущий прогон дал 24/30
>   (`eval/results/20261115_174124_0a97989/`); per-test стабильность не
>   измерялась (`repeat=1`).
> - Полный корпус из 10 PDF **спарсен**: `elements`, `structure` и `chunks`
>   есть для 2015–2020 и 2022–2025, всего **13969 чанка**, 98.4% со структурой;
>   2021 отсутствует физически. FAISS для всех 13969 чанков собран
>   (`84cd7bd4…`, 2048 измерений, ≈114 МБ). Полный graph-индекс строится;
>   partial graph-артефакт `b288dd3c…` имеет `status=incomplete`, поэтому
>   фабрика пока использует dev-граф `a8ce50f6…` (536 чанков, 2022–2023,
>   163 раздела).
> - Чат и benchmark по умолчанию работают на dev-корпусе (2016, 2017, 2022,
>   2023; 1025 чанков, `data/dev/`). Цифры benchmark нельзя переносить на
>   полный корпус 10 PDF.
>
> Подробности — в разделах «Архитектура», «Данные и состояние корпуса»,
> «Ограничения», а также в `data/corpus_build_status.md` и `eval/results/`.

---

## Содержание

1. [Что умеет](#что-умеет)
2. [Быстрый старт](#быстрый-старт)
3. [Архитектура](#архитектура)
4. [Установка](#установка)
5. [Конфигурация](#конфигурация)
6. [Данные и состояние корпуса](#данные-и-состояние-корпуса)
7. [Запуск](#запуск)
8. [Benchmark и Evaluation](#benchmark-и-evaluation)
9. [Тесты](#тесты)
10. [Структура проекта](#структура-проекта)
11. [Ограничения](#ограничения)
12. [Документация](#документация)

---

## Что умеет

- **Парсинг PDF** (`src/ingestion`) — Unstructured `hi_res`/`fast`, OCR (rus+eng),
  батчи по страницам, resume по SHA-256 и подписи параметров, фильтр мелких
  изображений, вырезы `data/figures/`.
- **Структура документа** — дерево разделов с breadcrumbs; приоритет
  источников: закладки PDF → оглавление (TOC) → эвристика по заголовкам.
- **Чанки К2** — 800/100 символов, «таблица = один чанк» с HTML, структурные
  поля `section_path`/`breadcrumbs`.
- **Hybrid retrieval** (`src/retrieval`) — BM25 с лемматизацией + dense-поиск
  (постоянный FAISS-индекс), слияние RRF, LLM-реранкер, жёсткие фильтры по
  году/документу/эмитенту/разделу, query rewriting и diversity после rerank.
- **Graph RAG** (`src/graph`, `src/retrieval/graph_hybrid.py`) — граф
  документов, разделов, чанков и сущностей; graph-augmented поиск с provenance,
  диагностикой и fallback к обычному hybrid retrieval.
- **Ответы** (`src/engine`) — упаковка контекста целыми чанками, строгая
  проверка цитат, безопасный `abstain`, структурированные факты и расчёты,
  сравнение структуры отчётов.
- **Агент** (`src/agent`) — LangGraph с белым списком инструментов и лимитами
  вызовов; router/selector, CoverageReport (матрица «метрика × год»),
  verification с одним repair-шагом, temporal/structure analysis, trace stages;
  для structural-запросов — retrieval fallback с citations.
- **Streamlit UI** (`src/ui`) — чат с фильтрами, трассой инструментов, coverage,
  verification, структурными изменениями, graph-only evidence и ссылками на
  страницу PDF.
- **HTTP API** (`src/api`) — FastAPI-adapter над `AgentService`: health,
  capabilities, corpus, CRUD диалогов, сообщения, SSE с replay/resync,
  evidence, PDF-файл `inline` для перехода на страницу, локальная остановка;
  SQLite-история, cookie-владелец, честные флаги возможностей.
- **React WebUI** (`frontend/`) — чат на assistant-ui, история диалогов,
  фильтры, реальные SSE-этапы, tool trace, источники (полный текст chunk с
  подсветкой цитаты), кликабельные citations `[n]`, переход к странице PDF,
  таблицы, графики и экспорт CSV, структурное сравнение отчётов, отмена и
  восстановление после перезагрузки.
- **Evaluation API и UI** (`/api/evaluation/*`,
  `frontend/src/features/evaluation`) — запуск suite, история, результаты
  тестов, retrieval trace, отчёт, отмена и сравнение двух запусков; раздел
  включается `EVAL_UI_ENABLED=1`, прогресс обновляется polling'ом.
- **Benchmark** (`eval/`) — датасет с gold evidence, детерминированные
  проверки, retrieval/answer/dialogue-метрики, trace, отчёты, сравнение
  запусков, офлайн-режим `--fake-service`.

---

## Быстрый старт

### 0. Что установить

| Компонент | Версия / примечание |
|---|---|
| Python | 3.11.x (проверено на 3.11.9) |
| Node.js + npm | Node LTS ≥ 20 — нужен только для React WebUI и E2E |
| Tesseract OCR | 5.x с языками `rus` и `eng` |
| Poppler | нужен Unstructured/`pdf2image` для работы с PDF |
| Git | для клонирования и `git_sha` в benchmark |

В репозитории уже лежат `tessdata/rus.traineddata`, `eng.traineddata` и
`osd.traineddata`. Если системный Tesseract не знает, где их искать, задайте
`TESSDATA_PREFIX=<путь к проекту>/tessdata`.

### 1. Python-окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# либо воспроизводимый снимок проверенной среды:
pip install -r requirements.lock
```

### 2. Конфигурация

```powershell
Copy-Item .env.example .env
```

Минимум для реальных ответов, эмбеддингов и реранкера:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=https://foundation-models.api.cloud.ru/v1
LLM_MODEL=GigaChat/GigaChat-3-Pro
EMBED_MODEL=ai-sage/Giga-Embeddings-instruct-3B
RERANKER_MODEL=ai-sage/GigaChat3-10B-A1.8B
RERANK_ENABLED=1
```

Если LLM остаётся на Cloud.ru, а эмбеддинги нужно вынести на другой
OpenAI-совместимый провайдер (например, NeuralDeep `giga-embeddings`), добавьте:

```dotenv
EMBED_BASE_URL=https://api.neuraldeep.ru/v1
EMBED_API_KEY=sk-...
EMBED_MODEL=giga-embeddings
```

Без ключа работают офлайн-тесты и `eval.runner --fake-service`; реальный чат,
индексация эмбеддингов, реранкер и benchmark требуют доступа к Cloud.ru API.

### 3. Данные

Каталог `data/` и исходные PDF **не коммитятся** в git. Для запуска нужен один
из вариантов:

- локально уже есть `data/raw/*.pdf`, `data/source_registry.json` и готовые
  артефакты (`data/dev/` или `data/chunks/`);
- либо положите 10 PDF в `data/raw/` и соберите корпус командами из раздела
  «Запуск» (`scripts/build_corpus.py`, `scripts/build_dev_corpus.py`).

По умолчанию чат использует готовый dev-корпус из 4 документов
(`CHUNKS_DIR=data/dev/chunks`, 1025 чанков), поэтому полный парсинг для
первого запуска не нужен.

### 4. Запустить чат

```powershell
streamlit run src/ui/app.py
```

### 5. Запустить React WebUI и HTTP API

```powershell
# терминал 1 — API
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# терминал 2 — Vite dev-сервер
cd frontend
npm install
npm run dev            # http://localhost:5173, /api проксируется на 127.0.0.1:8000
```

Либо оба сервиса в отдельных окнах:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1
# с разделом Evaluation:
powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1 -EvalUi
```

### 6. Проверить установку

```powershell
python -m pytest
cd frontend; npm run test:run; npm run e2e; cd ..
python -m eval.runner --validate-only
python scripts/check_llm_access.py
```

---

## Архитектура

### 1. Общая схема

```text
┌──────────────────────────── ОФЛАЙН: сборка корпуса ────────────────────────────┐
│                                                                                 │
│  data/raw/*.pdf + data/source_registry.json                                    │
│        │                                                                        │
│        ├─► ingestion/pdf_parser.py       → data/elements/{doc_id}_elements.json │
│        │                                   data/elements/{doc_id}_stats.json    │
│        ├─► ingestion/pdf_outline.py      → закладки / TOC                       │
│        ├─► ingestion/structure_builder   → data/structure/{doc_id}_sections.json│
│        ├─► ingestion/pdf_to_chunks.py    → data/chunks/{doc_id}_chunks.json     │
│        ├─► ingestion/chunk_enricher.py   → breadcrumbs / section_path           │
│        ├─► graph/graph_builder.py        → data/graph/{fingerprint}.json        │
│        └─► core/embeddings.py            → data/cache/embeddings_*.json         │
│                                           + data/indexes/{fingerprint}/index.faiss│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── ОНЛАЙН: путь запроса ──────────────────────────────┐
│                                                                                 │
│  question + filters                                                             │
│        │                                                                        │
│        ▼                                                                        │
│  AgentService (LangGraph)                                                       │
│   normalize_request → analyze_query → clarify_period → choose_action            │
│        │                                                    │                   │
│        │                              ┌─────────────────────┘                   │
│        │                              ▼                                         │
│        │              tools: search_reports | get_structure_tree |              │
│        │                     get_entities | get_chunk                           │
│        │                              │                                         │
│        │                              ▼                                         │
│        │              HybridRetriever / PerYearSearch / GraphAugmentedRetriever │
│        │              dense(FAISS) + BM25 → RRF → LLM-reranker → filters        │
│        │                              │                                         │
│        │                              ▼                                         │
│        │              harvest_tool_result → assess_evidence                    │
│        │              (CoverageReport, structural changes, calculations)        │
│        │                              │                                         │
│        │                              ▼                                         │
│        │              generate_answer → validate_answer → repair (1 раз)        │
│        │                              │                                         │
│        ▼                              ▼                                         │
│  Answer: status/citations/facts/calculations/coverage/verification/warnings      │
│        │                                                                        │
│        ├──► Streamlit UI (src/ui)                                               │
│        ├──► HTTP API (src/api) ──► SSE ──► React WebUI (frontend)               │
│        └──► eval.runner ──► summary/cases/traces/report                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Слои и ответственность

| Слой | Пакет / каталог | Что внутри | Не должен делать |
|---|---|---|---|
| Конфигурация и данные | `src/core/` | `config.py`, `rag_config.py`, `models.py`, `corpus.py`, `source_registry.py`, `embeddings.py`, `cache.py`, `dense_index.py`, `faiss_index.py`, `llm_lock.py` | знать про UI и HTTP |
| Ingestion | `src/ingestion/` | PDF parser, outline, structure builder, chunks, enricher, corpus validator, CLI | вызывать LLM-генерацию ответа |
| Retrieval | `src/retrieval/` | BM25, dense, RRF, reranker, rewrite, temporal search, graph retriever, graph hybrid | формировать финальный ответ |
| Graph | `src/graph/` | модели графа, entity extractor, builder, загрузка артефактов | быть обязательным для каждого вопроса |
| Engine | `src/engine/` | context packing, prompts, generator, calculations, structure compare | хранить историю диалогов |
| Agent | `src/agent/` | LangGraph, tools, router, selector, coverage, verification, service, factory | дублировать retrieval/генерацию |
| HTTP API | `src/api/` | FastAPI adapter, SSE, SQLite-история, auth owner, Evaluation API | содержать вторую оркестрацию |
| Streamlit UI | `src/ui/` | чат, citations, coverage, verification, diagnostics | считать метрики или обходить `AgentService` |
| React WebUI | `frontend/` | assistant-ui чат, history, evidence, charts, Evaluation UI | вычислять CAGR/разницы и «достаточность» доказательств |
| Evaluation | `eval/` | датасет, метрики, evaluators, runner, report, trace | менять поведение агента под тест |
| Скрипты | `scripts/` | сборка корпуса/графа/индексов, проверки, приёмка, smoke API | быть единственным путём для продукта |

### 3. Offline: как собирается корпус

1. `data/source_registry.json` фиксирует `doc_id`, `file_path`,
   `source_sha256`, год, период, issuer, страну и статус сопоставимости.
2. `src/ingestion/pdf_parser.py` извлекает `RawElement` (Title, Text, Table,
   Image и т.д.) с геометрией, `page_label`, HTML таблиц и путями к вырезам.
3. `pdf_outline.py` и `structure_builder.py` строят дерево разделов. Источники
   приоритета: закладки PDF → TOC → эвристика по заголовкам.
4. `pdf_to_chunks.py` + `chunk_enricher.py` создают чанки 800/100, таблица
   остаётся одним чанком с HTML; добавляются `section_path`/`breadcrumbs`.
5. `src/core/embeddings.py` считает векторы и кэширует их по fingerprint
   корпуса; `scripts/build_indexes.py` строит постоянный FAISS `IndexFlatIP`.
6. `src/graph/graph_builder.py` строит граф документов/разделов/чанков/сущностей
   и пишет артефакт с manifest. Неполный артефакт (`status=incomplete`) не
   активируется.

### 4. Online: путь запроса

```text
вопрос
  → router (LLM или SimpleRouter) → QueryPlan
  → PlanAwareSelector: какой инструмент вызвать и с какими фильтрами
  → search_reports: dense + BM25 → RRF → rerank → diversity → top-k
    (для сравнения по годам — PerYearSearch, для graph-запросов —
     GraphAugmentedRetriever с общим rerank)
  → get_structure_tree / get_entities / get_chunk при необходимости
  → CoverageReport: какая пара «метрика × год» подтверждена
  → pack_context: целые чанки в бюджет токенов и символов
  → LLM-генерация Answer
  → validation цитат + verification + один repair
  → answer | abstain
```

Ключевые ограничения:

- не более 4 tool calls и 2 поисковых вызовов на turn (настраивается
  `MAX_TOOL_CALLS`, `MAX_SEARCH_CALLS`);
- citations разрешены только на чанки из фактически переданного контекста;
- при пустом retrieval различаются «нет отчёта за год» и «показатель не найден»;
- 2015 год (БПС-Сбербанк) по умолчанию исключается из межгодовых сравнений;
- graph/reranker/structure — необязательные каналы: их сбой не отменяет
  baseline retrieval.

### 5. Агент

LangGraph-граф (`src/agent/agent_graph.py`):

```text
START
  → normalize_request
  → analyze_query
  → clarify_period
       ├─ нужен уточняющий вопрос → generate_answer
       └─ choose_action
            ├─ tools (search_reports / get_structure_tree / get_entities / get_chunk)
            │     → harvest_tool_result → assess_evidence
            │          ├─ достаточно / бюджета нет → generate_answer
            │          └─ недостаточно и бюджет есть → choose_action
            └─ нет действия → generate_answer
  → validate_answer
       └─ один repair → END или abstain
```

Инструменты строго allowlisted. LLM не может расширить бюджеты или вызвать
произвольный URL/файл. Служебные реплики («Привет!», «Спасибо») обрабатываются
`src/agent/smalltalk.py` без retrieval и без citations.

### 6. HTTP API и UI

- `src/api/app.py` — ASGI-приложение; импорт не читает env и не создаёт БД,
  инициализация происходит в `lifespan`.
- `AgentService` создаётся лениво и один раз на процесс (`src/api/services.py`).
- `ExecutionGate` сериализует вызовы LLM внутри API (chat имеет приоритет над
  evaluation), `LLMExecutionLock` — между API и CLI benchmark.
- `src/api/store.py` — SQLite истории диалогов; `src/api/evaluation_store.py` —
  отдельная SQLite для benchmark.
- SSE: `GET /api/requests/{id}/events` с `Last-Event-ID`, replay, `resync`,
  heartbeat и терминальным событием.
- React WebUI ходит в `/api/*`; в production может раздаваться тем же FastAPI
  при `API_SERVE_SPA=1`.

### 7. Evaluation

```text
eval/datasets/*.json
  → eval.runner (валидация, выбор suite/split, запуск AgentService)
  → eval/trace + eval/evaluators + eval/metrics
  → eval/results/{run_id}/
       summary.json
       cases.jsonl
       retrieval.csv
       traces/*.json
       report.html / report.md
       manual_review.json
       judge.jsonl (если judge включён)
  → React Evaluation UI / API / eval.report --compare
```

Метрики разделены на четыре независимые группы: retrieval, answer, reasoning,
dialogue. `pass_rate` всегда показывается вместе с `execution_coverage`.
Сравнение запусков блокирует дельты при разных `dataset.sha256` или
`corpus.content_sha256` (`comparable=false`).

---

## Установка

### Windows

```powershell
# Пример через Chocolatey; можно установить вручную и добавить в PATH.
choco install python --version=3.11.9
choco install nodejs-lts
choco install tesseract
choco install poppler

python --version        # 3.11.x
node --version          # >= 20
tesseract --version
pdftoppm -v             # проверка poppler
```

Если Tesseract установлен, но `rus` не находится:

```powershell
$env:TESSDATA_PREFIX = "$PWD\tessdata"
```

В репозитории уже есть `tessdata/rus.traineddata`, `eng.traineddata`,
`osd.traineddata`. Для постоянного хранения добавьте `TESSDATA_PREFIX` в `.env`.

### Linux / macOS

```bash
# Debian/Ubuntu — пример
sudo apt-get install python3.11 python3.11-venv tesseract-ocr tesseract-ocr-rus poppler-utils
# macOS
brew install python@3.11 tesseract poppler node
```

Unstructured `hi_res` дополнительно скачивает layout/OCR-модели при первом
запуске; нужен доступ в сеть. Для `fast`-стратегии модели не требуются.

### Python-зависимости

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# воспроизводимый снимок:
pip install -r requirements.lock
```

Основные группы зависимостей: `unstructured[all-docs]` + `unstructured-inference`,
`langchain*`/`langgraph`, `numpy`/`faiss-cpu`/`rank-bm25`/`pymorphy3`,
`networkx`, `fastapi`/`uvicorn`/`httpx`, `streamlit`, `pydantic-settings`,
`pytest`/`pytest-timeout`.

### Frontend

```powershell
cd frontend
npm install
npm run build       # tsc -b + vite build
```

E2E дополнительно требует установленный Chromium:

```powershell
npx playwright install chromium
```

---

## Конфигурация

Все настройки читаются из `.env` и переменных окружения. Полный пример —
`.env.example`.

### LLM и модели

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_API_KEY` | — | ключ Cloud.ru; без него реальный чат не работает |
| `LLM_BASE_URL` | `https://foundation-models.api.cloud.ru/v1` | OpenAI-совместимый endpoint |
| `LLM_MODEL` | `GigaChat/GigaChat-3-Pro` | генерация ответа |
| `EMBED_MODEL` | `ai-sage/Giga-Embeddings-instruct-3B` | эмбеддинги |
| `EMBED_BASE_URL` | пусто → `LLM_BASE_URL` | отдельный endpoint эмбеддингов (например, NeuralDeep) |
| `EMBED_API_KEY` | пусто → `LLM_API_KEY` | ключ отдельного провайдера эмбеддингов (`NEURALDEEP_API_KEY` — алиас) |
| `EMBED_QUERY_PREFIX` | `query:` | префикс запроса эмбеддингов; документы встраиваются без префикса |
| `RERANKER_MODEL` | `ai-sage/GigaChat3-10B-A1.8B` | LLM-реранкер |
| `RERANK_ENABLED` | `1` | `0` отключает реранкер |
| `ROUTER_MODEL` | пусто → `LLM_MODEL` | модель router/rewrite/structure |

### Корпус и индексы

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `CHUNKS_DIR` | `data/dev/chunks` | каталог чанков для чата/API/Streamlit |
| `CACHE_DIR` | `data/dev/cache` | кэш эмбеддингов |
| `INDEXES_DIR` | `data/indexes` | постоянные FAISS-индексы |
| `SOURCE_REGISTRY_PATH` | `data/source_registry.json` | реестр источников |
| `FAISS_ENABLED` | `1` | `0` → in-memory `DenseIndex` |
| `FAISS_PERSIST` | `1` | сохранять FAISS-индекс на диск |
| `DATA_RAW` | `data/raw` | исходные PDF |
| `PARSE_STRATEGY` | `hi_res` | `hi_res` или `fast` |
| `PARSE_PAGE_BATCH` | `20` | страниц на батч |
| `STORE_IMAGE_PAYLOAD` | `1` | base64 картинок в elements JSON |
| `TESSDATA_PREFIX` | `tessdata` | языковые модели Tesseract |

Чтобы работать не на dev-корпусе, а на канонических 2015–2018:

```dotenv
CHUNKS_DIR=data/chunks
CACHE_DIR=data/cache
INDEXES_DIR=data/indexes
```

Для полного корпуса после парсинга 2019–2024 используйте те же `CHUNKS_DIR` и
`CACHE_DIR`, затем пересоберите индексы.

### Бюджеты агента и retrieval

`src/core/rag_config.py` задаёт безопасные значения по умолчанию:
`TOP_K=5`, `CANDIDATES=20`, `RERANK_POOL=10`, `MAX_TOOL_CALLS=4`,
`MAX_SEARCH_CALLS=2`, `PER_YEAR_K=2`, `MAX_YEARS=12`, `MAX_QUERIES=3`,
`DIVERSITY_ENABLED=1`, `COVERAGE_ENABLED=1`, `VERIFICATION_ENABLED=1`.
Все они переопределяются env (см. `.env.example`).

### HTTP API

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_DB_PATH` | `var/api.sqlite3` | история диалогов |
| `API_CORS_ORIGINS` | `http://localhost:5173` | dev-origin Vite |
| `API_SERVE_SPA` | `0` | раздавать `frontend/dist` из FastAPI |
| `API_SPA_DIST_DIR` | `frontend/dist` | каталог собранного SPA |
| `API_MAX_MESSAGE_CHARS` | `16000` | лимит сообщения |
| `API_RATE_LIMIT_PER_MINUTE` | `120` | rate limit POST-сообщений |

### Evaluation

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `EVAL_UI_ENABLED` | `0` | включить `/api/evaluation/*` и раздел UI |
| `EVAL_DB_PATH` | `data/evaluation/evaluation.sqlite3` | отдельная БД benchmark |
| `EVAL_RESULTS_DIR` | `eval/results` | артефакты запусков |
| `EVAL_DATASETS_DIR` | `eval/datasets` | датасет benchmark |
| `EVAL_QUEUE_SIZE` | `3` | очередь запусков |
| `EVAL_JUDGE_ENABLED` | `0` | включить LLM-судью |
| `EVAL_FAKE_SERVICE` | `0` | офлайн-запуск без LLM |
| `LLM_LOCK_PATH` | `data/evaluation/llm.lock` | общий межпроцессный lock |
| `LLM_LOCK_TIMEOUT_S` | `900` | ожидание lock |
| `LLM_LOCK_STALE_S` | `900` | возраст heartbeat для stale-lock |

---

## Данные и состояние корпуса

- `data/raw/` — 10 исходных PDF (2015–2020, 2022–2025); 2021 года нет.
- `data/source_registry.json` — реестр источников; проверяется
  `scripts/verify_corpus.py`.
- `data/corpus_build_status.md` — статус сборки по `doc_id`, обновляется
  `scripts/build_corpus.py`.
- Канонические артефакты на 16.11.2026: `elements`, `structure` и `chunks` —
  для всех 10 документов (2015–2020, 2022–2025), **13969 чанка**, 98.4% со
  структурой, 0 пустых, 1650 таблиц; 2021 отсутствует физически.
- Для чата, API и benchmark по умолчанию используется **полный корпус**
  (`CHUNKS_DIR=data/chunks`, `CACHE_DIR=data/cache` в `.env`). Dev-корпус из 4
  документов (2016, 2017, 2022, 2023; 1025 чанков, `data/dev/`) остаётся для
  быстрых прогонов: `scripts/ask_dev.py`, `scripts/acceptance_dev.py`,
  `scripts/build_graph_dev.py`; включается через
  `CHUNKS_DIR=data/dev/chunks`, `CACHE_DIR=data/dev/cache`.
- Постоянный FAISS-индекс: `data/indexes/{fingerprint}/index.faiss` +
  `manifest.json`. Fingerprint совпадает с fingerprint кэша эмбеддингов
  (тексты чанков, модель, `query_prefix`, SHA-256 источников). Индекс
  переиспользуется, если совпали корпус, модель и порядок `chunk_id`; иначе
  пересобирается.
- На 16.11.2026 собран FAISS-индекс для всех 10 документов / 13969 чанков:
  `84cd7bd4…`, 2048 измерений, ≈114 МБ; dev-индекс — `26f393fe…` (1025 чанков).
  Эмбеддинги считаются через Cloud.ru
  (`ai-sage/Giga-Embeddings-instruct-3B`); NeuralDeep больше не используется.
  Артефакты не коммитятся.
- Активный граф: `48d7165f…` — полный корпус, 13 969 чанков, **9 996
  сущностей**, **7 987 связей**, 1 491 раздел, `status=complete`, 0 failed.
  Чекпоинт извлечения — `data/graph/.extraction_checkpoint.jsonl`; при
  добавлении документов старые пакеты переиспользуются по `chunk_id`, LLM
  вызывается только для новых чанков. Модель извлечения — `Qwen/Qwen3.6-35B-A3B`
  (Cloud), 8 чанков за вызов, лимит вывода 16384 токена.
- Графовый поиск использует два канала: сущностный (`MENTIONS`/`RELATED_TO`) и
  структурный — соседи верхних базовых хитов по разделу (`CONTAINS`) и странице,
  `GRAPH_STRUCTURE_BUDGET=2` на сид (`0` выключает). Бюджет измерен офлайн:
  полнота пула 0.847 → 0.908 на dev-корпусе; после реранкера прирост пула
  сохраняется (+0.03…0.05), а на выданном top-5 эффект в пределах разброса
  реранкера (`spec_graph.md` §8.6).
- `data/metrics/metrics.json` — структурный индекс табличных значений:
  из HTML таблиц берётся шапка с годами, поэтому каждое число получает год.
  Если шапки нет, годы берутся из строки «2023 2024 2025» перед таблицей
  (признак `inferred`), а канонические показатели строятся только по таблицам
  с явной шапкой. Собирается `python scripts/build_metrics.py` (13 969 чанков →
  8 432 значения, 10 документов).
- Числа в ответе проверяются по этому индексу: `numeric_claim_year_mismatch`
  (число отнесено к году, за который его нет в таблицах и фактах) и
  `numeric_claim_contradicts_table` (для метрики и года в таблице другое
  значение). Проверка «темп роста» сверяет процент из ответа с расчётом по
  фактам двух лет (`growth_rate_mismatch`). Все три — `fatal`: ответ уходит
  на повторную генерацию, а при неудаче — в abstain, а не в уверенную ошибку.
  Справочник значений (`metrics_reference`) передаётся в промпт, а таблицы-чанки
  добавляются в evidence, чтобы на них можно было сослаться.
- 2015 год — отчёт БПС-Сбербанка (Беларусь); он помечен в реестре как
  несопоставимый и не должен попадать в сравнения по умолчанию.

Проверить текущее состояние:

```powershell
python scripts/build_corpus.py --status-only
python scripts/verify_corpus.py --json data/corpus_report.json --markdown data/corpus_report.md
```

`verify_corpus.py` на неполном корпусе завершается кодом 1 и перечисляет
отсутствующие артефакты — это ожидаемое состояние, а не «зелёная» приёмка
полного корпуса.

---

## Запуск

### Dev-корпус и быстрые вопросы

```powershell
python scripts/build_dev_corpus.py          # пересобрать dev-корпус, если нужно
python scripts/ask_dev.py "Какая чистая прибыль за 2023 год?" --top-k 5
python scripts/ask_dev.py "Сравни выручку за 2022–2023" --agent
```

`ask_dev.py` и `acceptance_dev.py` требуют `LLM_API_KEY`/`LLM_BASE_URL`.

### Реальная приёмка 4 сценариев

```powershell
python scripts/acceptance_dev.py
```

Результат — `data/dev/acceptance/acceptance_report.md` и `.json`. Снимок
15.11.2026: `passed=4` (factoid, temporal, structural, abstain). После правок
генератора приёмку стоит перепрогнать.

### Полный корпус: парсинг, структура, чанки

```powershell
# Проверить статус без запуска парсинга
python scripts/build_corpus.py --status-only

# Парсинг + structure + chunks с resume; можно ограничить документом
python scripts/build_corpus.py
python scripts/build_corpus.py --doc sber_2023
python scripts/build_corpus.py --skip-parse

# Отдельные шаги ingestion CLI
python -m src.ingestion --all
python -m src.ingestion --structure
python -m src.ingestion --chunks
python -m src.ingestion --stats

# Быстрая проверка парсера без полного прогона
python -m src.ingestion --sample --sample-pages 14
python -m src.ingestion --smoke --pages 3

# Валидация корпуса
python scripts/verify_corpus.py
```

### FAISS-индекс

```powershell
python scripts/build_indexes.py --dry-run
python scripts/build_indexes.py
python scripts/build_indexes.py --reuse-cache-only   # только из готового кэша

# Большой корпус: чекпоинты и pacing под RPM провайдера
python scripts/reindex_embeddings.py
python scripts/build_indexes.py --reuse-cache-only
```

`build_indexes.py` считает эмбеддинги одним проходом и пишет кэш только в конце,
поэтому на большом корпусе после 429/обрыва теряется вся работа.
`reindex_embeddings.py` режет запросы по лимиту payload, выдерживает паузу под
RPM и сохраняет `*.partial.json`: после сбоя команду можно просто перезапустить.

Нужны `EMBED_MODEL` и ключ эмбеддингов (`EMBED_API_KEY` или `LLM_API_KEY`; кроме
`--reuse-cache-only` при готовом кэше). Без `faiss-cpu` используется in-memory
`DenseIndex`.

### Граф

```powershell
# dry-run: сколько батчей нужно, без вызовов LLM
python scripts/build_full_graph.py --dry-run

# Полная сборка на Cloud: 8 чанков за вызов, 16k вывода, 8-16 воркеров
python scripts/run_graph_full.py --provider cloud --model "Qwen/Qwen3.6-35B-A3B" `
  --chunks-per-call 8 --max-completion-tokens 16384 --workers 16

# Точечные прогоны
python scripts/build_full_graph.py --doc sber_2023 --limit 200 --workers 8
python scripts/build_full_graph.py --doc sber_2023 --limit 200 `
  --chunks-per-call 8 --workers 8

# Бесплатный тариф NeuralDeep: раннер сам ждёт окна квоты и продолжает с чекпоинта
python scripts/run_graph_full.py --provider neuraldeep --model qwen3.6-35b-a3b-noreason `
  --chunks-per-call 12 --max-completion-tokens 8192 --workers 3

# Добор только failed-чанков после сбоя/лимита
python scripts/patch_graph_failed_chunks.py `
  --graph data/graph/<fingerprint>.json --chunks data/chunks --workers 3

# Dev-граф и сравнение поиска
python scripts/build_graph_dev.py --limit 200
python scripts/compare_graph_dev.py --questions questions.txt
```

Извлечение переиспользуется из чекпоинта **по `chunk_id`**, поэтому добавление
новых отчётов не требует повторных вызовов LLM для старых чанков. В чекпоинте
фиксируется модель: при смене `--model` старые записи игнорируются, и граф
считается заново. Крупные пакеты (8+ чанков) требуют лимита вывода ≥16384
токенов, иначе JSON обрезается и пакет уходит в failed
(`GRAPH_MAX_COMPLETION_TOKENS`).

Активным считается только артефакт с `status=complete`; среди полных
выбирается самый покрытый, затем самый свежий.

### Streamlit UI

```powershell
streamlit run src/ui/app.py
```

### HTTP API, React WebUI, production-сборка

```powershell
# API
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# React dev
cd frontend
npm install
npm run dev            # http://localhost:5173

# Production: собрать SPA и раздать тем же API
npm run build
$env:API_SERVE_SPA="1"
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000
```

Раздел Evaluation включается отдельно: `EVAL_UI_ENABLED=1` перед запуском API.
При `0` endpoints отвечают `404 evaluation_disabled`, а
`capabilities.evaluation_ui=false`.

Быстрый запуск обоих сервисов:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1
powershell -ExecutionPolicy Bypass -File scripts\dev_ui.ps1 -EvalUi
```

### Проверка внешнего API

```powershell
python scripts/check_llm_access.py
python -m scripts.smoke_api --live
python scripts/check_embedding_compat.py
```

`check_llm_access.py` раздельно проверяет `models.list`, chat и embeddings и
печатает UUID проекта из ключа (без секрета). Chat и `models.list` идут через
`LLM_BASE_URL`, embeddings — через `EMBED_BASE_URL`, если он задан. Это
позволяет отличить «ключ неверный» (`400 invalid api key secret`) от «проект
ключа недоступен» (`403 Project not found`). `smoke_api` — smoke-тест внешнего
LLM/embeddings API, а не HTTP API проекта. `check_embedding_compat.py` считает
векторы нового эмбеддера для сэмпла чанков и сравнивает косинус с готовым
кэшем: так решается, переиспользовать кэш или переиндексировать корпус.

---

## Benchmark и Evaluation

### CLI benchmark

```powershell
# Проверить датасет и хэши без запуска тестов
python -m eval.runner --validate-only

# Офлайновый smoke без LLM и сети
python -m eval.runner --suite smoke --split dev --fake-service

# Реальный smoke
python -m eval.runner --suite smoke --split dev --limit 12 --no-judge

# Полный прогон 30 тестов, включая holdout
python -m eval.runner --suite full --no-judge

# Бенчмарк по полному корпусу (30 тестов, 10 отчётов): пересобрать эталон и прогнать
python scripts/build_benchmark_full.py
python -m eval.runner --datasets eval/datasets/full --chunks data/chunks `
  --suite full --fresh-service --output-dir eval/results/full-$(Get-Date -Format yyyy-MM-dd)
```
# Отдельные режимы retrieval
python -m eval.runner --suite smoke --retrieval-mode dense --no-judge
python -m eval.runner --suite smoke --retrieval-mode all --no-judge

# Отчёт по последнему запуску и сравнение
python -m eval.report --latest
python -m eval.report --compare RUN_A RUN_B
```

`eval.runner` пишет артефакты атомарно, держит общий `LLMExecutionLock` с
API и возвращает exit code 5 при занятом lock (`llm_busy`). При
`--retrieval-mode all` каждый режим получает отдельный результат и trace.

### Датасет и результаты

Датасет: 30 тестов (10 easy / 10 medium / 5 hard / 5 multi_turn), из них
12 smoke (split `dev`) и 6 holdout. `expected_corpus_sha256` зафиксирован для
dev-корпуса из 4 документов; после сборки полного корпуса хэш нужно обновить.

Последние реальные прогоны (dev-корпус, `--no-judge`):

| Прогон | Артефакт | Результат |
|---|---|---|
| smoke, 12 тестов | `eval/results/20261115_164758_0a97989/` | 12/12, pass_rate 1.0, Recall@5 0.955, evidence/context coverage 1.0, dialogue 1.0 |
| full, 30 тестов (предыдущий) | `eval/results/20261115_174124_0a97989/` | 24/30, pass_rate 0.8, Recall@5 0.845, Recall@10 0.862, MRR 0.798 |
| full, 30 тестов (последний) | `eval/results/20261115_183541_0a97989/` | 23/30, pass_rate 0.767, Recall@5 0.889, Recall@10 0.926, MRR 0.889, evidence coverage 0.897, context coverage 0.981 |

Разбивка последнего full-прогона: easy 10/10, medium 8/10, hard 2/5,
multi_turn 3/5; dev 19/24, holdout 4/6. Основные причины падений:
`fact_missing` (9 срабатываний), `wrong_document` (4), `invalid_citation` (2),
`wrong_status` (2), `fact_wrong_value` (1). По категориям: fact_lookup 15/15,
abstain 1/1, structural 2/2, temporal 2/4, dialogue 3/5, multi_hop 0/3.

Дополнительные предупреждения прогона:

- `holdout_gold_overlap` для `easy_010`, `medium_009`, `hard_005`,
  `multi_turn_005` — часть gold-чанков holdout пересекается с dev; holdout не
  является полностью независимым;
- `context_chunk_ids` недоступны для части кейсов — context coverage
  считается частично (`not_measured`), это видно в `limitations` summary.

`pass_rate 0.767` — результат на dev-корпусе из 1025 чанков, а не на полном
корпусе 10 PDF. Per-test стабильность не измерялась (`repeat=1`).

### Бенчмарк по полному корпусу

`eval/datasets/full/` — отдельный набор из 30 тестов, привязанный к полному
корпусу (10 отчётов, 13 969 чанков). Вопросы составлены по текстам отчётов;
`scripts/build_benchmark_full.py` проверяет каждый эталон: gold-чанк существует,
ожидаемое число дословно есть в его тексте, год и документ совпадают с фильтрами.
Gold-группа принимает любой чанк того же документа и года с этим числом, потому
что показатель в отчёте повторяется в обзоре, таблице и примечаниях.

Состав: 12 easy (чистая прибыль 2016–2025, ROE, дивиденды), 8 medium
(сравнения по годам, прибыль/ROE 2016→2023, дивиденды, клиенты), 8 hard
(abstain по отсутствующему 2021 году, сравнения через пропущенные годы,
2015 — БПС-Сбербанк), 2 multi-turn (перенос метрики на следующий год).
12 smoke, 7 holdout.

Последний прогон (`eval/results/full-2026-11-16-v2/20261116_154358_ea7a45b/`,
Cloud, `--no-judge`):

| Метрика | Значение |
|---|---:|
| Pass rate / Pass@1 | **73,3%** (22/30) |
| Fact accuracy | 92,9% |
| Citation validity | 100% |
| Citation coverage | 92,9% |
| Status accuracy | 98,3% |
| Abstention accuracy | 100% |
| Retrieval Recall@5 | 0,702 |
| Retrieval MRR | 0,607 |
| Evidence / context coverage | 0,827 / 0,774 |

Разбивка: easy **12/12**, medium 5/8, hard 5/8, multi-turn 0/2; holdout **7/7**,
dev 15/23. По категориям: fact_lookup **18/18**, abstain **2/2**,
multi_hop 2/5, temporal 0/3, dialogue 0/2.

Известные дефекты, которые видны на этом прогоне:

- `medium_001` (прибыль 2019→2020): за 2020 год названо **831,7** млрд — это
  значение 2018 года;
- `medium_002` (прибыль 2023→2025): оба значения верные, но темп роста назван
  «7,9%» (это 2025 к 2024), фактически +13,1%;
- диалог без фильтров («Какая чистая прибыль у Сбербанка за 2025 год?») может
  уйти в смежную метрику: в multi-turn прогоне ответ был про чистые процентные
  доходы, а на вопрос про ROE — про кредиты B2B/B2C. При прямом вопросе с
  фильтрами года ответ корректный;
- `hard_003`, `hard_004`, `hard_006`: ответы по числам верные, но gold-чанки
  не попали в top-5 → срабатывает `evidence_coverage_below_min`
  (`citation_quote_mismatch` — цитаты в ответе перефразированы).

Воспроизвести:

```powershell
python scripts/build_benchmark_full.py          # пересобрать и проверить эталон
python -m eval.runner --datasets eval/datasets/full --chunks data/chunks `
  --suite full --output-dir eval/results/full-$(Get-Date -Format yyyy-MM-dd)
```

### Evaluation WebUI

```powershell
$env:EVAL_UI_ENABLED="1"
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
cd frontend; npm run dev
# http://localhost:5173/evaluation
```

Раздел показывает обзор, запуски, тесты, сравнение, trace и отчёт. Прогресс
обновляется polling'ом (SSE-канала нет); отмена работает для `queued` и
`running`; judge включается только при `EVAL_JUDGE_ENABLED=1`.

---

## Тесты

```powershell
python -m pytest
cd frontend; npm run test:run
cd frontend; npm run e2e
openspec validate --all --strict
```

Результаты на 15.11.2026 (Python 3.11.9, `.venv`):

- backend — `863 passed`;
- frontend unit — `160 passed` в 24 файлах Vitest;
- Playwright E2E — `7/7` (5 сценариев чата + 2 сценария Evaluation) против
  fixture-backend на изолированных портах (API 8010, UI 5273);
- `npx tsc -b` проходит, `vite build` успешен; vendor-split убрал warning
  о чанке >500 kB (остаётся notice про dynamic+static import `api-client`);
- `openspec validate --all --strict` проходит.

Тесты офлайновые: внешние API подменяются заглушками, ключи и сеть не нужны.
Без сети остаются непроверенными реальный LLM-путь и качество ответов — для
этого служат `scripts/acceptance_dev.py` и реальные прогоны `eval.runner`.

---

## Структура проекта

```text
src/
  core/        настройки, модели, эмбеддинги, кэш, FAISS-индекс,
               реестр источников, загрузка корпуса, LLM-lock
  ingestion/   PDF-парсер, outline, структура, чанки, CLI, валидатор корпуса
  retrieval/   BM25, dense, RRF, реранкер, rewrite, temporal search,
               graph-augmented поиск
  graph/       модели графа, извлечение сущностей, сборка/загрузка графа
  engine/      упаковка контекста, промпты, генерация, расчёты,
               structure compare
  agent/       LangGraph-оркестрация, инструменты, coverage, verification,
               smalltalk, сервис, фабрика
  api/         FastAPI adapter: чат/SSE/evidence/PDF + Evaluation API
  ui/          Streamlit-чат, citations, coverage/verification/diagnostics
eval/          dataset, metrics, evaluators, runner, report, traces, результаты
frontend/      React + TypeScript WebUI + Evaluation UI + Vitest/Playwright
scripts/       сборка корпуса/графа/FAISS-индекса, проверки, dev-вопросы,
               приёмка, smoke API
tests/         офлайновые unit/интеграционные тесты
data/          исходные PDF и производные артефакты (10 отчётов, 2015–2020, 2022–2025)
docs/specs/    спецификации React WebUI и Evaluation WebUI
openspec/      OpenSpec context и change add-pdf-parser
```

---

## Ограничения

- Для ответов, эмбеддингов и LLM-реранкера нужен доступ к Cloud.ru API.
- Полный корпус 10 PDF спарсен (`verify_corpus.py`: 10/10, 13969 чанка, 0 ошибок),
  но embeddings/FAISS и graph для новых 10041 чанков ещё не собраны: нужен
  Cloud.ru API. Чат и benchmark по-прежнему работают на dev-корпусе из 4
  документов (1025 чанков), поэтому 23–24/30 нельзя переносить на полный корпус.
- FAISS-индекс не входит в git (как и `data/`): его нужно собрать командой
  `python scripts/build_indexes.py`. Без артефакта фабрика считает эмбеддинги
  и строит индекс при первом запуске; при отсутствии `faiss-cpu` используется
  in-memory fallback.
- Полный graph index собран частично: 1916/3928 чанков старого canonical-
  корпуса, артефакт `b288dd3c…` имеет `status=incomplete` (2012 чанков
  2017–2018 упали на 403 Cloud.ru). Для новых 10041 чанков graph не строился.
  Используется dev-граф `a8ce50f6…` (536 чанков, 2022–2023, 163 раздела);
  активен только артефакт со `status=complete`.
- Benchmark: последний полный прогон 30 тестов — 23/30 (`pass_rate 0.767`) на
  dev-корпусе; предыдущий — 24/30. `expected_corpus_sha256` привязан к
  dev-корпусу, а не к полному из 10 PDF. Holdout пересекается с dev по
  gold-чанкам в 4 тестах; per-test стабильность не измерялась.
- Judge-калибровка на размеченных ответах не выполнена; judge выключен по
  умолчанию (`EVAL_JUDGE_ENABLED=0`).
- Evaluation UI не имеет SSE-прогресса (polling — основной путь,
  `evaluation_sse=false`); выбор `attempt` > 0 доступен в API, но не в UI;
  переход из chunk Evaluation в Evidence Panel чата не реализован.
- React WebUI закрывает этапы 1–4; этап 5 закрыт частично: пагинация истории,
  UUID-fallback и часть a11y сделаны, но нет scroll-anchoring, полного
  focus-trap/аудита доступности и удаления Streamlit. Bundle-split убрал
  warning >500 kB.
- Серверная отмена чата не реализована: UI останавливает локальный поток, а
  `AgentService`/`EventHub` возвращают `cancel_not_supported`. Endpoint и
  capability существуют, но backend продолжает выполнение запроса.
- 2021 год в корпусе отсутствует; 2015 — отчёт БПС-Сбербанка, не сопоставимый
  с ПАО Сбербанк. Чанки обогащаются issuer из реестра, а 2015 автоматически
  исключается из многотемных сравнений с warning `incomparable_year_excluded`.
- Требования спек, которые ещё не реализованы (полный корпус/граф, benchmark на
  полном корпусе, judge-калибровка, Evaluation SSE и др.), перечислены в
  `TODO_AGENT_RAG.md` §10–§11.
- Качество структуры и таблиц зависит от PDF и OCR; для 2017 структура
  приблизительная. Спорные случаи проверяются через `--sample`, `--stats` и
  артефакты в `data/samples`.

---

## Документация

| Документ | Что фиксирует |
|---|---|
| [`spec_target.md`](spec_target.md) | целевой контракт и фактический статус слоёв |
| [`spec_agent.md`](spec_agent.md) | LangGraph-агент, инструменты, события |
| [`spec_retrieval_engine.md`](spec_retrieval_engine.md) | retrieval, RRF, rerank, context packing, Answer |
| [`spec_graph.md`](spec_graph.md) | схема графа, extraction, graph retrieval, измерения |
| [`spec_pdf_parser.md`](spec_pdf_parser.md) | контракт PDF-парсера и артефактов |
| [`spec_benchmark.md`](spec_benchmark.md) | датасет, trace, метрики, runner, отчёты |
| [`spec_agent_enhancement.md`](spec_agent_enhancement.md) | усиление агента и roadmap |
| [`spec_ui_eval.md`](spec_ui_eval.md) | исторический контракт Streamlit/eval |
| [`docs/specs/web-ui-modernization.md`](docs/specs/web-ui-modernization.md) | React WebUI и HTTP adapter |
| [`docs/specs/evaluation-webui.md`](docs/specs/evaluation-webui.md) | Evaluation API и раздел Benchmark в UI |
| [`src/api/README.md`](src/api/README.md) | endpoint'ы, SSE, storage, безопасность API |
| [`frontend/README.md`](frontend/README.md) | запуск, архитектура и ограничения React UI |
| [`TODO_AGENT_RAG.md`](TODO_AGENT_RAG.md) | сделанное, долги и известные расхождения |
| [`data/corpus_build_status.md`](data/corpus_build_status.md) | фактический статус сборки корпуса |
