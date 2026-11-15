# Спецификация проекта: Sberbank Annual Reports Graph RAG (2015–2024)
## Версия 3.1 — Structure-Aware + переиспользование всех 8 занятий курса

> **Статус документа:** подробный исторический дизайн и учебный контекст. Текущий
> проверяемый статус, границы MVP и критерии реализации фиксируются в
> [`spec_target.md`](spec_target.md); отсутствующие компоненты детализированы в
> `spec_agent.md`, `spec_retrieval_engine.md`, `spec_graph.md`, `spec_ui_eval.md`. Если описания
> расходятся, приоритет у этих целевых спецификаций. Наличие раздела здесь не означает,
> что соответствующий код уже реализован.

> Изменение v3.0 → v3.1: проведён аудит **всех 8 занятий** курса. Матрица переиспользования
> (§3) переписана под реальный код: hybrid search + RRF + LLM-реранкер берём из Занятия 5,
> агентный цикл и query-analyzer — из Занятий 6–7, метрики и LLM-судья — из Занятия 8.
> Заявленный в v3.0 стек (CrossEncoder, RAGAS, sentence-transformers) заменён на курсовой.
>
> **Актуализация 15.11.2026.** Раздел §2 и схема §4 остаются историческим
> проектным замыслом. Фактическая реализация: Python-ядро + постоянный FAISS
> `IndexFlatIP` (fallback `DenseIndex`), FastAPI adapter и React WebUI
> (Streamlit сохранён как legacy), SQLite для истории и отдельная SQLite для
> evaluation, `LLMExecutionLock` для межпроцессной сериализации. Текущий статус,
> цифры benchmark и ограничения — в `README.md`, `spec_target.md`,
> `docs/specs/web-ui-modernization.md` и `docs/specs/evaluation-webui.md`.

---

## 0. Аудит занятий курса (что реально есть)

| Занятие | Тема | Главное, что можно забрать |
|---|---|---|
| 1 | GigaChat API через Cloud.ru, Gradio | `get_credentials()`, OpenAI-клиент (`timeout=90, max_retries=2`), паттерн JSON-экстракции с валидацией |
| 2 | PDF → чанки | `unstructured` per-page (`hi_res`, `starting_page_number`), контракт чанка (`chunk_id, evidence_id, page_pdf, page_label, year, kind, html, source_sha256`), чанкинг 800/100, «таблица = один чанк», assert-валидации корпуса |
| 3 | Векторный поиск | `Settings`, `ProjectConfig`, `CourseEmbeddings` (query_prefix), `DenseIndex` (косинус), `filter_chunks`, `corpus_fingerprint` + `cached_document_vectors`, формат dev-вопросов (answer/abstain, `relevance={eid:2}`) |
| 4 | Базовый RAG | `pack_context` (бюджет символов, таблицы не режутся), `SYSTEM_PROMPT` (JSON: status/answer/citations/facts), `parse_json`, `RAGEngine` + `build_engine`, `QueryCache` |
| 5 | Hybrid search + reranking | BM25 (`rank_bm25` + pymorphy3-лемматизация), `rrf(rankings, k=10, constant=60)`, **LLM-реранкер** (`scores 0–3`, модель `GigaChat3-10B-A1.8B`), мета-фильтры до top-k, query rewriting (2 перефраза) |
| 6 | LangChain | LCEL-цепочки, `ChatPromptTemplate`/few-shot/`JsonOutputParser`, `@tool` + `create_agent`, `format_docs` |
| 7 | RAG с LangGraph | `StateGraph`/`RAGState`, `query_analyzer` (structured output → фильтр метаданных), условные рёбра, агентный цикл tools⇄model, `InMemorySaver` + `thread_id`, streaming |
| 8 | Метрики + LLM-судья | `ranking_metrics` (P@k, Recall@k, RR@k, AP@k, nDCG@k — готовая реализация), адаптер MTEB, `judge_case` + `RUBRICS` (4 рубрики 0–2), `JUDGE_INSTRUCTION`, калибровка судьи |

**Уточнения по реальности курса (важно):**
- Реранкер курса — **не CrossEncoder**, а pointwise LLM-скоринг через chat API (JSON-массив scores 0–3, batch = все 10 кандидатов одним запросом).
- BM25-вариант в Занятии 5 оценивает чанки **только с score>0**, фильтр метаданных применяется **до** top-k, BM25-индекс при фильтрации не пересчитывается.
- Занятие 7 (файл «agents_langchain_langgraph»/«agentic_rag_deepagents») — это **LangGraph**; deepagents/subagents/MCP в курсе отсутствуют.
- RAGAS в курсе НЕ используется — есть собственный LLM-судья. Пакетное вычисление mean-метрик по набору вопросов и сериализация результатов eval на диск курсом НЕ реализованы — пишем сами (обёртка над `ranking_metrics`).

---

## 1. Обзор проекта

**Название:** Structure-Aware Graph RAG для анализа годовых отчётов Сбербанка (2015–2024) с temporal-навигацией.

**Данные:** 9 PDF-файлов годовых отчётов Сбербанка (2015–2020, 2022–2024).

**Ключевая идея:**
- **Структура отчёта (Части → Главы → Разделы → Подразделы)** — центральная фича системы, а не дополнение.
- **~70% кода берём из курса (Занятия 1–8)** — парсинг, чанки, векторный поиск, BM25+RRF, реранкер, промпт генерации, метрики, судья, агентный цикл.
- Своё строим только там, где курс не даёт материала: дерево разделов, breadcrumbs, граф, UI-дерево, расширение eval-датасета.

**Что отличает от baseline-курса:**
- Курсовой RAG отвечает на factoid-вопросы («Какая прибыль в 2025?») по 3 отчётам 2024–2025.
- Наш RAG **понимает, где в документе находится информация**, работает по 9 отчётам за 10 лет и использует структуру для точности.

---

## 2. Стек технологий

| Слой | Библиотека / модель | Назначение | Источник |
|---|---|---|---|
| **Парсинг PDF + структура** | `unstructured[all-docs]` | Layout-aware извлечение + `category_depth` для заголовков | Курс Занятие 2 + своё (структура) |
| **Чанкинг** | `langchain-text-splitters.RecursiveCharacterTextSplitter` | 800/100, таблица = 1 чанк | Курс Занятие 2, as-is |
| **Agent / RAG engine** | LangGraph `ReportAgent` + собственный `RAGEngine` | Агент оркестрирует allowlisted tools; engine отдельно тестируется | К7 + К+С |
| **Graph Store** | `networkx.MultiDiGraph` | Один graph store для структуры, chunks и сущностей | Своё |
| **Vector Store** | FAISS `IndexFlatIP` + in-memory NumPy `DenseIndex` как fallback | Эмбеддинги чанков, постоянный индекс в `data/indexes/{fingerprint}` | Курс Занятие 3 + своё |
| **Keyword Index** | `rank-bm25` + `pymorphy3` | BM25 с лемматизацией | Курс Занятие 5, as-is |
| **Embeddings** | `ai-sage/Giga-Embeddings-instruct-3B` через Cloud.ru | query_prefix-протокол | Курс Занятие 3, as-is |
| **Reranker** | LLM `ai-sage/GigaChat3-10B-A1.8B`, scores 0–3 | Pointwise-реранкинг пула после RRF | Курс Занятие 5, as-is |
| **LLM** | `GigaChat/GigaChat-3-Pro` через Cloud.ru (`base_url=.../v1`, `timeout=90, max_retries=2`) | Extraction, routing, generation | Курс Занятие 1, as-is |
| **VLM** | Vision-модель через тот же endpoint | Описание графиков | Своё |
| **UI** | Streamlit (legacy) + React/TypeScript/Vite + FastAPI (фактический) | Чат, evidence, таблицы/графики, структура; React — целевой WebUI | Своё (в курсе только Gradio, Занятие 1 — для прототипов) |
| **Eval retrieval** | `ranking_metrics` (курс) + mean-агрегация | Recall@k, MRR@k, nDCG@k | Курс Занятие 8 + своя агрегация |
| **Eval качества ответа** | LLM-judge (курсовая модель, 5 критериев 0–2 в фактическом контракте; 4 legacy-критерия сохранены для совместимости) | Factual accuracy, completeness, sources, analysis, constraints | Курс Занятие 8 + `spec_benchmark.md` §5.4 |
| **Eval абстрактное (опция)** | `ragas` | Только для сверки с публичным бенчмарком, не основа | Своё/опция |

> Смена стека относительно v3.0: sentence-transformers-эмбеддинги и CrossEncoder заменены
> на курсовые модели. Причина: кэш векторов с `corpus_fingerprint`, протокол `query_prefix`,
> реранкер и судья курса работают на моделях Cloud.ru — переиспользование «под ключ» вместо
> параллельной инфраструктуры.

---

## 3. Матрица источников кода (по компонентам)

Легенда: **К** = курс as-is, **К+** = курс с точечными изменениями, **С** = своё, **С+К** = своё поверх курсовых примитивов.

### 3.1. Ingestion

| Компонент | Источник | Что делаем |
|---|---|---|
| Реестр PDF (`SOURCES`: doc_id, sha256, url, report_period) | К, Занятие 2 | **К+**: формат as-is, заполняем 9 отчётами 2015–2024 (`spec_pdf_parser.md` v1.1 уже реализует: `make_doc_id`, `RawElement`) |
| `partition_pdf` per-page, `strategy="hi_res"`, `starting_page_number`, `infer_table_structure` | К, Занятие 2 (`read_page_elements`) | **К**: протокол as-is; наш `src/ingestion/pdf_parser.py` — обёртка с сохранением Image и `category_depth` |
| Контракт чанка: `chunk_id, evidence_id, doc_id, title, page_pdf, page_label, year, report_period, kind, element_ids, text, html, source_url, source_sha256` | К, Занятие 2 (`chunk_record`) | **К**: ID-схемы (`{doc_id}:p{page:03d}`, digest sha256 по `[text, html, element_ids]`) — as-is |
| Чанкинг: `RecursiveCharacterTextSplitter(800, 100)`, группа закрывается на смене страницы/перед таблицей, таблица = 1 чанк с html | К, Занятие 2 (`make_chunks`) | **К** |
| Assert-валидации корпуса (year int, ≤ chunk_size, уникальность chunk_id, html↔text таблиц) | К, Занятие 2 | **К**: блок проверок переносим в `tests/test_corpus.py` |
| `StructureBuilder` → `SectionNode`/`DocumentStructure` (дерево из `Title` + `category_depth`) | **С** | НОВОЕ: курс не делает структуру |
| `ChunkEnricher` → breadcrumbs, section_path, section_level, parent_section | **С** | НОВОЕ (расширение контрактных полей — опциональные) |
| `TableStructurer` → FinancialFact (LLM → atomic JSON) | **С** | НОВОЕ: курс оставляет таблицу-HTML как есть; атомарные факты — надстройка, HTML-чанк остаётся базой |
| `VisualDescriber` → ChartDescription (VLM) | **С** | НОВОЕ: курс отбрасывает `Image`; мы сохраняем крупные (≥200×200) через `spec_pdf_parser.md` |

### 3.2. Retrieval

| Компонент | Источник | Что делаем |
|---|---|---|
| `normalize`, `DenseIndex` (косинус, фильтры до top-k, сортировка `(-score, chunk_id)`) | К, Занятие 3 | **К** as-is |
| `filter_chunks(year, report_period, kind, allowed_doc_ids)` | К, Занятие 3 | **К** as-is |
| `CourseEmbeddings` + `make_embeddings` (query_prefix, check_embedding_ctx_length=False) | К, Занятие 3 | **К** as-is |
| `corpus_fingerprint` + `cached_document_vectors` (кэш `embeddings_<fp>.json`, атомарная запись, batch 8) | К, Занятие 3 | **К** as-is — критично для 9 отчётов |
| `QueryCache` (кэш векторов вопросов) | К, Занятие 4 | **К** as-is |
| `lemmatize` (pymorphy3, lru_cache) + `tokenize` | К, Занятие 5 | **К** as-is |
| BM25 (`BM25Okapi`, k1=1.5, b=0.75, score>0) | К, Занятие 5 | **К** as-is |
| `rrf(rankings, k=10, constant=60)` | К, Занятие 5 | **К** as-is |
| LLM-реранкер: `rerank(query, hits)`, scores 0–3, `response_format=json_object`, 1 повтор при невалидности, tie → порядок RRF | К, Занятие 5 | **К** as-is; модель `GigaChat3-10B-A1.8B` |
| Конвейер-константы: `CANDIDATES=20` (каждый поиск) → `RERANK_POOL=10` (после RRF) → `TOP_K=5` | К, Занятие 5 | **К** as-is (в `ProjectConfig`: top_k=5 уже есть) |
| Query rewriting: 2 перефраза, RRF по всем формулировкам, промпт «не добавляй факты/годы/суммы» | К, Занятие 5 | **К** as-is, включаем для temporal/multi_hop |
| Metadata-фильтр до top-k (через `eligible`-индексы, без пересчёта BM25) | К, Занятие 5 | **К+**: обобщаем наш структурный фильтр в тот же протокол `eligible` |
| `filter_chunks_by_structure` (section_path_prefix, breadcrumb_contains, max_level) | **С** | НОВОЕ: компонируется с курсовым фильтром КАК ещё один `eligible`-предикат |
| Query Router (LLM-классификатор type/section_filter/year_range/entities) | **С+К** | Свои Pydantic-схемы, НО механика — курсовой `query_analyzer` (Занятие 7): structured output `json_mode` + условное ребро LangGraph |
| Мульти-план retrieval (factoid/temporal/structural/multi_hop) | **С** | НОВОЕ: комбинации уже имеющихся кусков (rewriting + eligible-фильтры + годы) |

### 3.3. Engine / Generation

| Компонент | Источник | Что делаем |
|---|---|---|
| `pack_context(hits, max_chars=16000)`: карточка с фикс. ключами, чанк сверх бюджета пропускается ЦЕЛИКОМ | К, Занятие 4 | **К** as-is |
| `SYSTEM_PROMPT` (JSON: status/answer/citations/facts; abstain-протокол; anti-mixing правил) | К, Занятие 4 | **К** as-is для factoid; citing = chunk_id |
| `parse_json` (regex ```json-fence), `message_text` | К, Занятие 4 | **К** as-is |
| `generate_answer` (пустой контекст → abstain без вызова LLM) | К, Занятие 4 | **К** as-is |
| `RAGEngine` + `build_engine` | К, Занятие 4 | **К+**: расширяем до `GraphRAGEngine` (graph_retrieve, structure_filter, router) |
| `make_chat` (temperature=0.1, max_tokens=1400, `use_responses_api=False`, `disabled_params={"parallel_tool_calls": None}`) | К, Занятие 4 | **К** as-is |
| Промпты temporal/structural (breadcrumbs в цитатах, хронология) | **С** | НОВОЕ: наследуют формат вывода `SYSTEM_PROMPT` (status/citations/facts) |
| `format_docs`: строка `"[{chunk_id}] {title}; PDF-страница {page_pdf}\n{text}"` | К, Занятие 6 | **К+**: добавляем breadcrumbs после chunk_id |
| LangGraph-цикл агента (query_or_respond → ToolNode → tools_condition → возврат в модель) | К, Занятие 7 | **К+**: инструменты наши (`search_reports` + `get_structure_tree` + `get_entities`), граф as-is |
| Память диалога `InMemorySaver` + `thread_id`, `recursion_limit` | К, Занятие 7 | **К** as-is |
| Стриминг (`stream_mode="updates"/"messages"`) | К, Занятие 7 | **К** as-is → Streamlit |

### 3.4. Graph

| Компонент | Источник | Что делаем |
|---|---|---|
| NetworkX-граф: Document/Section/Chunk/Entity, extractor, graph retrieval | **С** | НОВОЕ: в курсе графа знаний нет; без дублирующего LlamaIndex store |
| Связи Entity/DocumentNode/Chunk | **С** | НОВОЕ |

Статус и измеренный результат: `spec_graph.md` §8. На dev-корпусе 2022–2023
граф поднял recall свидетельств на multi-hop наборе с 0.62 до 0.75 (+13 п.п.)
при небольшом снижении MRR. `GraphAugmentedRetriever` подключён в фабрике и
участвует в общем rerank; Streamlit показывает graph-only evidence, отдельная
панель визуализации графа в React не включена (`graph_panel=false`).

### 3.5. Eval

| Компонент | Источник | Что делаем |
|---|---|---|
| `ranking_metrics` (P@k, Recall@k, RR@k, AP@k, nDCG@k — бинарная релевантность из `relevance`) | К, Занятие 8 | **К** as-is; следим за соглашением нормировки AP@k |
| Mean-агрегация по набору вопросов + сериализация в JSON/CSV | **С** | НОВОЕ: курс не пишет результаты на диск |
| LLM-judge: `RUBRICS` (5 критериев 0–2 в `eval/evaluators/judge.py`; 4 legacy-критерия — для совместимости), `Judgment`, `judge_case`, `JUDGE_INSTRUCTION`, `CRITERION_FIELDS`, `response_format=json_object`, t=0 | К, Занятие 8 | **К+**: интерфейс `Judge`/`NullJudge`, реальная калибровка не выполнена; модель задаётся конфигурацией |
| Калибровка судьи на 3 контрольных кейса (12 вызовов) | К, Занятие 8 | **К+**: как в занятии, кейсы наши |
| Dev-вопросы: формат `{id, split, question, report_period, category, expected_status, expected_facts, reference_facts, relevance={eid:2}}` | К, Занятие 3–4 (`development_questions`) | **К+**: формат as-is; вопросные пары переадресуем на наш корпус (курсовые ссылки `sber_ifrs_2025:p006` и т.п. указывают на отчёты 2024–2025, которых у нас нет); abstain-кейс (dev08-паттерн) сохраняем |
| Урезанные MTEB-задачи (RuBQ 5 вопросов) | К, Занятие 8 | **К** опционально — как процедура выбора эмбеддинг-модели |
| RAGAS (faithfulness, context precision) | **С/опция** | v3.1: заменён курсовым судьёй; ragas оставлен только как перекрёстная проверка |
| Temporal accuracy, Structure accuracy (цитаты из правильного раздела) | **С** | НОВОЕ: кастомные метрики поверх формата `citations` |

### 3.6. Инфраструктура / UI

| Компонент | Источник | Что делаем |
|---|---|---|
| `get_credentials` (env-версия, не Colab) + OpenAI-клиент | К, Занятие 1 | **К+**: переносим в `src/core/credentials.py` (конфиг — `config.py`; `.env` вместо getpass) |
| JSON-экстракция с валидацией набора полей | К, Занятие 1 | **К** как паттерн для TableStructurer |
| Streamlit UI (чат, граф agraph, дерево структуры, timeline) | **С** | НОВОЕ: в курсе только Gradio-форма (Занятие 1); для agraph-визуализации графа и дерева — Streamlit |
| `stream` отображение в UI | К, Занятие 7 | **К** механика streaming → Streamlit |

---

## 4. Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│  PRESENTATION (Streamlit legacy / React WebUI) ← ПИШЕМ С НУЛЯ  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Chat UI  │  │Graph View│  │Structure │  │Timeline View  │  │
│  │          │  │(agraph)  │  │   Tree   │  │               │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│  стриминг ответа → курсовой streaming (Занятие 7)               │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER (LangGraph)               ← ГИБРИД           │
│  граф: analyze_query → route → retrieve_tools ⇄ answer          │
│  ┌──────────────┐  ┌─────────────────────────────────────────┐ │
│  │ Query Router │  │ Агент (курсовой цикл Занятия 7 as-is):   │ │
│  │ (query_      │  │ query_or_respond → ToolNode([search_…])  │ │
│  │  analyzer-   │  │ → tools_condition → (цикл в модель)      │ │
│  │  паттерн)    │  │ память: InMemorySaver + thread_id        │ │
│  └──────────────┘  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  CORE LAYER                                  ← КУРС ~80%         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GraphRAGEngine (расширение RAGEngine, К4)               │  │
│  │  ├─ dense_search()   ← К3 DenseIndex as-is               │  │
│  │  ├─ keyword_search() ← К5 BM25+pymorphy                │  │
│  │  ├─ rrf()            ← К5 as-is (constant=60)            │  │
│  │  ├─ structure_filter() → С (eligible-предикат в протоколе К5)│ │
│  │  ├─ rerank()         ← К5 LLM-реранкер as-is             │  │
│  │  ├─ pack_context()   ← К4 as-is                          │  │
│  │  ├─ graph_retrieve() → С (networkx.MultiDiGraph)        │  │
│  │  └─ answer()         ← К4 SYSTEM_PROMPT + С промпты      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                  ← ГИБРИД            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  chunks.json             ← К2 контракт + С breadcrumbs    │  │
│  │  ├─ chunk_id, evidence_id, doc_id, year, kind, html       │  │
│  │  ├─ source_url, source_sha256, element_ids                │  │
│  │  └─ breadcrumbs, section_path (НОВОЕ — ключевое!)         │  │
│  │  document_structure.json ← НОВОЕ (дерево разделов)        │  │
│  │  cache/embeddings_<fingerprint>.json ← К3 as-is           │  │
│  │  graph.json              ← НОВОЕ                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Поток поиска (константы Занятия 5):** вопрос → rewriting (0–2 перефраза) → параллельно BM25 + Dense (по `CANDIDATES=20` с учётом `eligible`-фильтров) → `rrf(constant=60)` → пул `RERANK_POOL=10` → LLM-реранкер (scores 0–3) → `TOP_K=5` → `pack_context(16000)` → генерация.

---

## 5. Структура отчёта — центральная фича (ЧТО НЕ ПЕРЕИСПОЛЬЗОВАТЬ ИЗ КУРСА И ПОЧЕМУ)

| Потребность | Есть в курсе? | Решение |
|---|---|---|
| Дерево разделов | Нет — курс живёт на 3 файлах/39 страницах без секций | Своё: `structure_builder.py` на `Title` + `category_depth` |
| Запись «где в документе» (breadcrumbs) | Нет (в чанке только страница/год/тип) | Своё: `chunk_enricher.py`, опциональные поля контракта |
| Разбор страниц на элементы | Да — Занятие 2 (per-page `hi_res` + `starting_page_number` + `page_label`) | Переиспользуем протокол, расширен в `spec_pdf_parser.md` v1.1 |
| Таблица как неделимый HTML-чанк | Да — Занятие 2 | As-is; LLM-структурирование фактов — надстройка С (`table_structurer.py`) |
| Визуельный контент | Нет — Занятие 2 отбрасывает `Image` | Своё: `visual_describer.py` (VLM для ≥200×200) |

### 5.1. Извлечение структуры (Unstructured → DocumentStructure)

Unstructured возвращает `Title` элементы с `metadata.category_depth`. Этого достаточно для построения дерева:

```python
# src/ingestion/structure_builder.py
@dataclass
class SectionNode:
    id: str                          # "part_1/chapter_3/section_2"
    title: str                       # "3.2. Операционные риски"
    level: int                       # 0=Part, 1=Chapter, 2=Section, 3=Subsection
    page_start: int
    page_end: int
    breadcrumbs: List[str]           # ["Управление рисками", "Операционные риски", "Киберриски"]
    parent_id: Optional[str]
    children_ids: List[str]

def build_document_structure(elements) -> DocumentStructure:
    """
    Строит дерево разделов из Title элементов Unstructured.
    Использует metadata.category_depth для определения уровня.
    """
    sections = []
    stack = []  # Стек текущих разделов по уровням

    for el in elements:
        if isinstance(el, Title):
            level = el.metadata.category_depth or 0
            breadcrumbs = [s.title for s in stack[:level]] + [el.text]
            section = SectionNode(
                id=build_path(breadcrumbs),
                title=el.text,
                level=level,
                page_start=el.metadata.coordinates.page_number,
                breadcrumbs=breadcrumbs,
                parent_id=stack[level-1].id if level > 0 and stack else None,
            )
            stack = stack[:level] + [section]
            sections.append(section)

    return DocumentStructure(sections=sections)
```

**Результат для отчёта Сбера:**
```
├── Стратегический отчет (level 0)
│   ├── Сообщение Председателя Правления (level 1)
│   └── Обзор ключевых показателей (level 1)
├── Корпоративное управление (level 0)
├── Управление рисками (level 0)
│   ├── Кредитный риск (level 1)
│   ├── Рыночный риск (level 1)
│   └── Операционные риски (level 1)
│       └── Киберриски (level 2)
└── Финансовая отчетность по МСФО (level 0)
```

### 5.2. Расширение контракта чанка (обратно совместимо)

```python
# Базовые поля — из курса Занятия 2, НЕ МЕНЯЕМ
REQUIRED_CHUNK_FIELDS = {
    "chunk_id", "evidence_id", "doc_id", "title",
    "page_pdf", "page_label", "year", "report_period",
    "kind", "element_ids", "text", "html",
    "source_url", "source_sha256",
}

# НОВОЕ — опциональные поля для structure-aware
OPTIONAL_STRUCTURE_FIELDS = {
    "section_path",      # "Управление рисками/Операционные риски/Киберриски"
    "breadcrumbs",       # ["Управление рисками", "Операционные риски", "Киберриски"]
    "section_level",     # 2 (глубина вложенности)
    "parent_section",    # "Управление рисками/Операционные риски"
}

# kind: курс "text"|"table" → наш супerset: + "table_fact"|"chart"
```

**Ключевой принцип:** курсовые функции `filter_chunks`, `DenseIndex`, `pack_context` работают как раньше — новые поля игнорируются, если не используются. Базовые assert-валидации Занятия 2 не ослабляем.

### 5.3. Structure-aware фильтры

Фильтр пишем в протокол курса Занятия 5 («eligible-индексы до top-k, без пересчёта BM25»):

```python
# src/retrieval/structure_filter.py
def structure_eligible(chunk, section_path_prefix=None,
                       breadcrumb_contains=None, max_level=None) -> bool:
    """Предикат, композируемый с filter_chunks (К3) в один eligible-список (К5)."""
    if section_path_prefix and not chunk.get("section_path", "").startswith(section_path_prefix):
        return False
    if breadcrumb_contains and not any(breadcrumb_contains in b for b in chunk.get("breadcrumbs", [])):
        return False
    if max_level is not None and chunk.get("section_level", 99) > max_level:
        return False
    return True

def apply_all_filters(chunks, **kwargs) -> list[dict]:
    """Курсовой фильтр метаданных (К3) + структурный предикат (С) → eligible."""
    base = filter_chunks(chunks, **{k: v for k, v in kwargs.items()
                                    if k in {"year", "report_period", "kind", "allowed_doc_ids"}})
    return [c for c in base if structure_eligible(c, **{k: v for k, v in kwargs.items()
            if k in {"section_path_prefix", "breadcrumb_contains", "max_level"}})]
```

---

## 6. Ingestion Pipeline

```
PDF файлы (9 отчётов)
    │
    ▼
[1. Unstructured partition_pdf]  ← К2: per-page hi_res, starting_page_number,
    │                              infer_table_structure; + С: Image/category_depth
    ▼
[2. Structure Builder]  ← С ──→ DocumentStructure (дерево разделов с breadcrumbs)
    │
    ├─→ [3. Table Structurer]  ← С ──→ List[FinancialFact] (LLM; курсовый паттерн
    │                                    JSON-экстракции из Занятия 1 + json-fence из К4)
    ├─→ [4. Visual Describer]  ← С ──→ List[ImageDescription] (VLM; только ≥200×200)
    └─→ [5. Chunk Enricher]    ← С ──→ List[HierarchicalChunk]
                                       К2-контракт + breadcrumbs + section_path
        │
        ▼
[6. Валидация корпуса]  ← К2 assert-блок as-is → tests/test_corpus.py
        │
        ▼
[7. Graph Builder (networkx)]  ← С ──→ Knowledge Graph
        ├─→ KG Extractor (LLM) ──→ Entities + Relations
        ├─→ DocumentNode ──→ узлы структуры (разделы)
        └─→ Vector Index ──→ Эмбеддинги (Giga-Embeddings, К3)
```

### Обработка разнородных данных

| Тип данных | Инструмент | Результат | Источник |
|---|---|---|---|
| **Текст** | Unstructured `NarrativeText`, сплиттер 800/100 | `TextChunk` с breadcrumbs | К2 + С(enricher) |
| **Таблицы** | Unstructured → `text_as_html`; HTML = 1 чанк (К2); LLM-факты поверх | `TableChunk` + `table_fact` | К2 + С(LLM) |
| **Картинки/графики** | Unstructured `Image` → фильтр ≥200×200 → VLM | `ChartDescription` | С |
| **Заголовки** | Unstructured `Title` + `category_depth` | `SectionNode` (дерево) | С |

---

## 7. Retrieval Pipeline

```
User Query
    │
    ▼
[1. Query Router]  ← механика К7 (query_analyzer: model.with_structured_output,
    │                 method="json_mode"), Pydantic-схема — С
    │               → RoutingDecision:
    │                 ├─ type: factoid/temporal/structural/multi_hop
    │                 ├─ section_filter: "Управление рисками" | None
    │                 ├─ year_range: (2020, 2024)
    │                 └─ needs_rewriting / needs_graph
    ▼
[2. Query Rewriting]  ← К5 as-is: 2 перефраза, RRF по всем формулировкам
    │                    (включаем для temporal/multi_hop)
    ▼
[3. Hybrid Retrieval]  ← К5 as-is:
    │   BM25 (k1=1.5, b=0.75, pymorphy3)  +  DenseIndex (К3, query_prefix)
    │   каждый поиск k=CANDIDATES=20, мета/структурные фильтры через eligible ДО top-k
    ▼
[4. RRF fusion]  ← К5 as-is: rrf(rankings, constant=60) → пул RERANK_POOL=10
    │
    ▼
[5. Rerank]  ← К5 as-is: LLM-реранкер, scores 0–3, tie → порядок RRF
    │            (CrossEncoder — только если API-реранкер не уложится в SLA)
    ▼
[6. Structure Filter]  ← С: в eligible-протоколе К5 (при routing-решении)
    │                    (применяется ДО top-k, поэтому фактически шаг 3)
    ▼
[7. Temporal Filter]  ← К3 filter_chunks(year=…) + С: явное «2021 нет в данных»
    ▼
[8. Graph Retrieve (опция)]  ← С: ограниченный обход networkx по сущностям
    ▼
[9. pack_context(max_chars)]  ← К4 as-is (таблицы не режутся)
    ▼
[10. LLM Generation]
    ├─ Factoid → SYSTEM_PROMPT К4 as-is
    ├─ Temporal → промпт С (хронология фактов по годам, формат facts К4)
    └─ Structural → промпт С (breadcrumbs в цитатах: [chunk_id | Раздел > Подраздел], стр. N)
    ▼
[11. Response] c citations: chunk_id + [Год] [Раздел > Подраздел] (стр. N)
```

---

## 8. Граф знаний с DocumentNode (полностью своё)

В курсе графа знаний нет: retrieval ограничен BM25/dense/LLM-реранком. Поэтому уровень графа — чистое наше строительство; от курса берём только эмбеддинг-модель (К3) и JSON-контракт чанков (К2).

В графе **три типа узлов**:
- `Entity` — сущности (Risk, Strategy, FinancialMetric, TechPlatform)
- `DocumentNode` — **узлы структуры** (разделы отчёта)
- `Chunk` — текстовые фрагменты

**Типы связей:**
- `Chunk --[mentions]--> Entity`
- `Entity --[relates_to]--> Entity`
- `DocumentNode --[contains]--> DocumentNode` (parent-child)
- `DocumentNode --[contains]--> Chunk`
- `DocumentNode --[has_theme]--> Entity`

**Преимущество:** можно спрашивать «что связано с разделом "Киберриски"?» и получать не только текст, но и связанные сущности.

**Минимальный объём (MVP):** строим граф на 3 отчётах (2022–2024), in-memory `networkx`; эмбеддинги сущностей — модель К3. Развитие до остальных лет — по остаточному принципу плана.

---

## 9. Структура проекта

Курсовой код разносим **по слоям нашего проекта** (не держим отдельным пакетом `course/` — курсовой код привязан к Colab-путям `/content/rag_work`), сохраняя имена функций/классов курса. Пометки: **КN** — занятие N курса (as-is или К+ с точечными изменениями), **С** — своё, **С↑КN** — своё поверх механизма занятия N.

```
sberbank_graph_rag/
│
├── src/core/
│   ├── config.py            # С  env/.env, ProjectConfig ← К3/К4 Settings+ProjectConfig
│   ├── models.py            # С  RawElement (spec_pdf_parser.md) + контракт чанка К2 (+С-поля)
│   ├── embeddings.py        # К3 CourseEmbeddings, make_embeddings, normalize; К4 QueryCache
│   ├── cache.py             # К3 corpus_fingerprint, cached_document_vectors
│   ├── dense_index.py       # К3 DenseIndex + filter_chunks
│   └── credentials.py       # К1 env-версия get_credentials + OpenAI-клиент params
│
├── src/ingestion/
│   ├── pdf_parser.py        # С↑К2 partition_pdf harness (spec_pdf_parser.md v1.1; fix §3.1 внедрён: build_page_slice + payload=True)
│   ├── pdf_to_chunks.py     # К2  read_page_elements/page_element_records/chunk_record/make_chunks
│   ├── structure_builder.py # С   дерево разделов из Title + category_depth
│   ├── chunk_enricher.py    # С   breadcrumbs, section_path в чанки К2-контракта
│   ├── table_structurer.py  # С   LLM → FinancialFact (паттерн JSON-экстракции К1)
│   └── visual_describer.py  # С   VLM → chart_description (курс Image отбрасывает)
│
├── src/retrieval/
│   ├── bm25.py              # К5  lemmatize, tokenize, BM25Okapi, keyword_search
│   ├── rrf.py               # К5  rrf(rankings, constant=60) as-is
│   ├── reranker.py          # К5  rerank() — LLM scores 0–3, модель GigaChat3-10B-A1.8B
│   ├── rewrite.py           # К5  query rewriting (2 перефраза + RRF по формулировкам)
│   ├── structure_filter.py  # С   structure_eligible — предикат в eligible-протоколе К5
│   ├── query_router.py      # С↑К7 Pydantic RoutingDecision + structured output json_mode
│   ├── graph_retriever.py   # С   ограниченный обход networkx (см. целевую спеку)
│   └── hybrid_retriever.py  # С↑К5 конвейер eligible → BM25+dense(20) → RRF → rerank(10) → top(5)
│
├── src/engine/
│   ├── context.py           # К4  pack_context as-is
│   ├── prompts.py           # К4  SYSTEM_PROMPT factoid as-is; С temporal/structural промпты
│   ├── rag_engine.py        # К4  RAGEngine/build_engine/generate_answer/parse_json
│   ├── graph_rag_engine.py  # С   GraphRAGEngine — наследник К4 RAGEngine
│   └── agent_graph.py       # К7  StateGraph-цикл + InMemorySaver; С: инструменты search_reports/
│                            #     get_structure_tree/get_entities
│
├── src/graph/               # С   kg_extractor.py, graph_builder.py, community.py
│
├── src/ui/                  # С   streamlit: structure_tree/graph_view/timeline;
│                            #     механика стрима из К7
├── eval/
│   ├── metrics.py           # К8  ranking_metrics as-is; С mean-агрегация + сериализация
│   ├── judge.py             # К8  RUBRICS, Judgment, judge_case, JUDGE_INSTRUCTION as-is
│   ├── calibration.py       # К8+ калибровка на 3 контрольных кейсах (кейсы наши)
│   ├── dataset.json         # К3+ формат dev-вопросов; вопросы переадресованы на 2015–2024 + abstain
│   └── run_eval.py          # С   прогон dev+наших вопросов, отчёт JSON/CSV
│
├── data/
│   ├── raw/                 # 9 PDF (SOURCES-формат К2)
│   ├── elements/            # С (spec_pdf_parser.md)
│   ├── chunks/              # chunks.json — К2 контракт + С breadcrumbs
│   ├── document_structure/  # С {year}_structure.json
│   ├── cache/               # embeddings_<fingerprint>.json (К3 as-is)
│   └── graph/               # С graph.json
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 10. Сценарии для скринкаста

### Сценарий 1: Structure-Aware Retrieval
**Запрос:** «Что говорится о киберрисках?»
- **Без структуры:** находит упоминания во всех разделах — каша
- **Со структурой:** находит **только в разделе «Управление рисками > Операционные риски > Киберриски»**
- **Демонстрация:** в UI видно дерево, подсвечен раздел, цитата с breadcrumbs; citations — курсовой chunk_id + наш раздел

### Сценарий 2: Temporal Evolution
**Запрос:** «Как менялась стратегия цифровизации с 2015 по 2024?»
- **Демонстрация:**
  - 2015–2018: «цифровая трансформация»
  - 2019–2021: «экосистема» (2021 отсутствует — router/prompt явно указывает на пробел, abstain-механика курса)
  - 2022–2024: «AI-трансформация», «GigaChat»
- **Визуализация:** Timeline + граф с эволюцией сущностей

### Сценарий 3: Structure + Temporal
**Запрос:** «Какие меры по снижению операционных рисков описаны в разделе "Управление рисками" за последние 3 года?»
- **Демонстрация:**
  - Query Router (structured output К7-механика) определяет MULTI_HOP + STRUCTURAL
  - structure_filter компонируется с filter_chunks → eligible до top-k (протокол К5)
  - Temporal Filter ограничивает 2022–2024
  - Ответ с точными ссылками на подразделы

### Сценарий 4: Table-Aware
**Запрос:** «Какой был ROE и чистая прибыль в 2020 и 2023?»
- **Демонстрация:** находит `table`-чанки К2 (html неделим) и `table_fact` чанки С; цифры с цитированием таблиц; pack_context не режет таблицы (К4 as-is)

### Сценарий 5: Навигация по структуре
- Кликнуть на «Управление рисками» в дереве → увидеть связанные сущности
- Кликнуть на «Киберриски» → drill-down в подраздел
- **Демонстрация:** интерактивное исследование отчёта как карты

### Сценарий 6: Factoid (курсовой baseline) + механика поиска
**Запрос:** «Какова чистая прибыль Группы за 2024 год?» (переадресованный курсовой dev01-паттерн)
- **Демонстрация:** работает как в курсе; дополнительно показываем rerank-шаг (scores 0–3) и abstain при отсутствии данных (dev08-паттерн)

---

## 11. План реализации (8 шагов)

| Шаг | Задача | Источник кода | Результат |
|---|---|---|---|
| **1** | Адаптируем курсовой клиент (К1: env-креды, OpenAI-клиент) в `src/core/`. Переносим парсинг per-page (К2) — `pdf_parser.py` доведён до v1.1: **fix §3.1 внедрён (задачи 7.1–7.4 done)** — батчи через pypdf-slice (`build_page_slice`) + `starting_page_number` + `extract_image_block_to_payload=True`; бенчмарк снят: hi_res steady 8.3 с/стр. → ≈ 5.9 ч на 2581 стр. | К1, К2 | `src/core/credentials.py`, `pdf_parser.py` v1.1 |
| **2** | Строим `DocumentStructure`. `structure_builder.py` готов: уровни по эвристике + TOC-калибровка (точный + fuzzy-матч) + junk-фильтры. Артефакт `data/structure/{doc_id}_sections.json`, CLI `--structure`. Проверено малыми срезами 2023 и на 2015. | Свой + К2 | Дерево разделов |
| **2.5** | Чанкинг К2 + breadcrumbs: `pdf_to_chunks.py` (800/100, таблица = 1 чанк, chunk_id/evidence_id), `chunk_enricher.py` (section_path/breadcrumbs/section_level/parent_section), CLI `--chunks`, `data/chunks/{doc_id}_chunks.json` | К2 + С | HierarchicalChunk |
| **3** | Таблицы → FinancialFact (LLM). Картинки → chart_description (VLM). | Свой (+ промпт-паттерн К1) | Структурированные данные |
| **4** | Масштабируем на все 9 отчётов. Кэш векторов с `corpus_fingerprint` (К3) на Giga-Embeddings | К3 | Полный набор чанков 2015–2024, кэш |
| **5** | Строим граф `networkx.MultiDiGraph`: Document/Section/Chunk/Entity + provenance; scope фиксируется manifest | Своё | Проверяемый graph index |
| **6** | Собираем hybrid retrieval: BM25 + dense → structure/metadata eligibility → RRF → rerank → citations/abstain; graph — дополнительный retriever | К5/К7-паттерны + своё | Работающий retrieval |
| **7** | Streamlit UI: чат (К7-агентный цикл + streaming) + граф + дерево структуры + timeline | К7 поверх + свой | Интерактивный прототип |
| **8** | Eval: dev-вопросы (формат К3, переадресованные) + свои; метрики К8 `ranking_metrics` + mean-агрегация (своё); LLM-judge К8 + калибровка; скринкаст | К8 + свой | Готовый проект |

> Отличие от v3.0: раньше шаг 6 был сборкой retrieval «с нуля», теперь почти половина retrieval уже лежит в К5 (as-is), и шаг 6 — в основном склейка, а не написание. Ориентировочно сокращает работы шага на ~25%.

**Фактический статус шагов на 15.11.2026:** шаги 1, 2, 2.5, 5 (dev), 6 и 8
(инфраструктура) реализованы; шаг 4 выполнен частично — канонически спарсены
2015–2018, dev-корпус из 4 документов, полный набор 9 PDF не собран; шаг 3
(таблицы → FinancialFact, картинки → VLM) не выделен в отдельный production-
контракт; шаг 7 реализован в двух клиентах — Streamlit (legacy) и React WebUI
с FastAPI adapter. Полный graph index, benchmark на полном корпусе и
judge-калибровка остаются незакрытыми.

---

## 12. Критерии приёмки

Статус на 15.11.2026: критерии ниже — целевые. Фактически benchmark идёт на
30 тестах (не на 50), полный корпус 9 PDF не собран, judge не калиброван,
temporal/structure accuracy отдельно не измерялись.

### Функциональные
- [x] Factoid-вопросы отвечают с citations; отдельного dense-only baseline на 50 вопросах нет
- [~] Система отвечает на все 4 типа, но последний full-прогон даёт multi_hop 0/3 и temporal 2/4
- [x] Каждый answerable-ответ имеет citations с `chunk_id`, годом, разделом и страницей
- [~] UI показывает дерево/динамику; интерактивный граф есть только в Streamlit, в React — обычные evidence-карточки
- [x] Structure-aware запросы ограничивают поиск разделом (eligible до top-k)
- [x] Abstain работает для отсутствующего года и отсутствующего показателя

### Качественные (eval на 50 вопросах)
- [~] Retrieval: последний full (30 тестов) — Recall@5 0.889, MRR 0.889; nDCG@5 0.852. Цель по Recall@5 достигнута, по 50-вопросному набору — не проверено
- [ ] Судья К8 (4 рубрики 0–2): не калиброван, реальные прогоны идут `--no-judge`
- [ ] Судья откалиброван на 3 контрольных кейсах (К8) до запуска на датасете; согласованность ≥ 90%
- [ ] Temporal accuracy ≥ 0.80 (кастомная, поверх facts-формата К4) — отдельно не измерена
- [ ] Structure accuracy ≥ 0.85 (кастомная: цитаты из правильного раздела) — отдельно не измерена
- [ ] (Опционально) перекрёстная проверка ragas — faithfulness ≥ 0.85, context precision ≥ 0.75

### Инженерные
- [x] Курсовые модули (DenseIndex, BM25-лемматизация, rrf, rerank, pack_context, SYSTEM_PROMPT, ranking_metrics, judge) работают без изменения логики
- [x] Курсовой контракт чанка К2 не изменён (только опциональные breadcrumbs-поля)
- [~] Кэш векторов с `corpus_fingerprint` переиспользован для 4 dev-документов и 2015–2018; полный набор 9 PDF не собран
- [x] `pack_context` не режет таблицы (К4 as-is)
- [x] Прогон eval сериализуется в JSON/CSV/HTML
- [x] Код покрыт docstrings (Google-style), типизация (Python type hints)
- [x] README с инструкцией установки, запуска и eval

---

## 13. Управление рисками

| Риск | Митигация |
|---|---|
| Unstructured `hi_res` медленный | Курсовой per-page протокол (`PdfWriter`-вырезка + `starting_page_number`) + `fast` для превью; полный прогон ~5.9 ч hi_res (бенчмарк 7.4 снят: 8.3 с/стр., резерв ×1.5 → план 6–9 ч) — запускать фоном на ночь |
| `category_depth` не всегда корректен | Валидация через LLM, ручная корректировка для сложных отчётов |
| LLM-реранкер (К5): latency/дороговизна, невалидный JSON | 1 повтор (К5 as-is), затем фолбэк на порядок RRF; проверка SLA на 10-чанковом пуле; опция CrossEncoder — ТОЛЬКО если LLM-вариант не проходит SLA |
| Экстракций LLM (таблицы/KG) дорого | Курсовой паттерн JSON-экстракции (К1) + response_format json_object; KG только на 3 отчётах |
| VLM дорого для всех картинок | Фильтр мелких (<200×200), только крупные графики |
| Граф слишком большой | In-memory NetworkX, MVP 3 отчёта |
| Отсутствие 2021 года | Router/промпт явно указывает на пробел (паттерн abstain курса) |
| Курсовые dev-вопросы не переносимы (целятся на отчёты 2024–2025) | Формат К3 оставляем, вопросные пары пишем по нашему корпусу (см. §3.5); сохраняем abstain-кейс |
| Судья К8 нестабилен | К8-калибровка; temperature=0; фиксация промпта; выбор модели (`DeepSeek-V4-Flash`) из К8 |
| Курсовой код привязан к Colab (getpass/userdata, `/content/rag_work`) | При переносе заменяется только слой путей/кредов (К1-собственная env-версия); логика не меняется |

---

## 14. Что даёт структура (vs курсовой baseline)

| Сценарий | Курсовой RAG (К2–К8) | Наш Structure-Aware RAG |
|---|---|---|
| «Что о киберрисках?» | Все упоминания — каша | **Только из раздела «Киберриски»** |
| «Какая прибыль?» | Может перепутать контексты | Понимает контекст через раздел |
| «Сравни риски 2020 и 2024» | Все упоминания рисков | **Только из «Управление рисками»** |
| «Покажи структуру отчёта» | Не может | **Интерактивное дерево** |
| Разброс тем по годам | Формат facts К4 работает на 1 отчёт | Хронология по 9 отчётам + timeline |
| Factoid-вопросы | ✅ Работает | ✅ Работает (переиспользуем курс) |
| Оценка качества | Ручная разметка (К5) / судья на WikiEval (К8) | Судья К8 на наших данных + IR-метрики по своему датасету |

---

## 15. Следующий шаг

Кодовый задел включает parser v2, outline, structure и chunking. Полный корпус не
подтверждён артефактами; retrieval, graph, engine, UI и eval ещё предстоит реализовать.
Актуальные этапы и критерии — в `spec_target.md`. Ближайшая последовательность:

1. Проверить реестр источников и включить issuer/country в metadata (особенно `sber_2015`).
2. Прогнать corpus pipeline, вручную проверить артефакты и добавить корпусные валидаторы.
3. Реализовать dense/BM25 baseline и engine с citations/abstain, измерить на dev set.
4. Добавить structure-aware retrieval, reranking, graph retriever, UI и evaluation по
   профильным спецификациям; не заявлять метрики до фактического прогона.
