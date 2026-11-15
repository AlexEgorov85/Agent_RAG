# Целевая спецификация проекта

**Проект:** Structure-Aware Graph RAG по годовым отчётам Сбербанка
**Версия:** 1.0
**Статус:** целевой контракт для реализации
**Связанные документы:** [spec_agent.md](spec_agent.md), [spec_retrieval_engine.md](spec_retrieval_engine.md), [spec_graph.md](spec_graph.md), [spec_ui_eval.md](spec_ui_eval.md), [spec_pdf_parser.md](spec_pdf_parser.md)

> Этот документ фиксирует целевое поведение и критерии готовности. Он не утверждает,
> что все описанные компоненты уже существуют. Backend-контур (ingestion, retrieval,
> engine, агент), Streamlit UI, HTTP API, React WebUI этапов 1–4 и инфраструктура
> benchmark реализованы; полный корпус 9 PDF, benchmark на полном корпусе и хвост
> этапа 5 React WebUI ещё не готовы. См. §2.

## 1. Цель и границы

Система отвечает на вопросы по годовым отчётам сбербанковской группы, приводит проверяемые
цитаты на исходные PDF-страницы и позволяет учитывать раздел документа и год отчёта.
Поддерживаемые сценарии:

1. **Фактический:** найти конкретное значение/утверждение в одном или нескольких отчётах.
2. **Структурный:** искать в заданной части дерева документа или показать структуру.
3. **Временной:** сравнить значения/утверждения по нескольким доступным годам.
4. **Составной:** ответить на вопрос, требующий нескольких фактов, разделов или отчётов.
5. **Отказ:** сообщить, что данных нет, год отсутствует или извлечённые свидетельства
   недостаточны.

Graph RAG — расширение retrieval, а не обязательный посредник для каждого вопроса.
Для фактического вопроса система должна сохранять корректный ответ и citations и при
недоступном графе.

**Не входит в MVP:** автоматическая финансовая рекомендация, прогнозирование будущих
показателей, гарантированное извлечение каждого графика, публичный многопользовательский
сервис и построение полного графа знаний вручную проверенного качества.

## 2. Фактический статус репозитория

Статус сверяется с кодом; наличие требования в спецификации не означает, что функция
реализована.

| Слой | Сейчас | Целевое состояние |
|---|---|---|
| PDF ingestion | `pdf_parser.py`, модели/конфиг, CLI; батчевый `hi_res`/`fast`, OCR и resume; `source_registry`, `scripts/build_corpus.py`, `verify_corpus.py`. В `data/raw` 9 PDF; `elements` готовы для 2015–2018, документы 2019–2024 не парсились | Проверенные артефакты всех входных PDF, отчёт о пропусках/качестве |
| Структура | `pdf_outline.py`, `structure_builder.py` (закладки → TOC → эвристика); sections для 2015–2018, для 2017 структура приблизительная | Воспроизводимое дерево для каждого обработанного отчёта |
| Чанки | `pdf_to_chunks.py`, `chunk_enricher.py`, CLI; 3928 чанков для 2015–2018 (98% со структурой) и отдельный dev-корпус из 4 документов (1025 чанков) | Корпус чанков с метаданными, структурой и валидацией |
| Dense/keyword retrieval | BM25 + dense, RRF, metadata/structure eligibility, filters, rewriting, diversity, `RetrievalTrace`; dense — постоянный FAISS `IndexFlatIP` с fallback на in-memory `DenseIndex` | Dense + BM25 с metadata/structure eligibility |
| Router/reranker/generation | LLM/Simple router, selector, CoverageReport, verification и repair, calculations, answer с citations/abstain, retrieval fallback для structural-запросов — реализованы | Query planning, hybrid retrieval, reranking, ответ с citations/abstain |
| Graph | Graph builder/retriever, `GraphAugmentedRetriever`, `use_graph`, provenance и fallback; в фабрике активен полный dev-граф `66c0dbe3…` (536 чанков, 2022–2023); полный `b288dd3c…` собран на 1916/3928 и имеет `status=incomplete` | Индекс разделов/чанков/сущностей и ограниченный graph retrieval |
| UI | Streamlit: чат, citations, coverage metric×year, verification, retrieval stages, structure, graph-only evidence, ссылки на PDF; React WebUI этапов 1–4 (чат, история, evidence, таблицы/графики/CSV, структурное сравнение, cancel/resync, Evaluation UI) | Streamlit — legacy, React — целевой UI |
| Eval | `eval/`: dataset validation, metrics, runner, report, judge-интерфейс; датасет 30 тестов (10/10/5/5), 12 smoke + 6 holdout; реальный smoke 12/12, последний полный прогон 23/30 (`pass_rate 0.767`), предыдущий 24/30 | Версионируемый датасет, retrieval и answer metrics, отчёты |
| API / React WebUI | `src/api/` — полный adapter (CRUD, SSE, evidence, PDF `inline`, capabilities); `frontend/` — React-чат этапов 1–4 (citations `[n]`, PDF-ссылки, таблицы/графики/CSV, структурное сравнение, cancel/resync) и Evaluation UI; E2E 7/7, 160 unit-тестов в 24 файлах | HTTP API и React WebUI по `docs/specs/*` |
| Evaluation API | `/api/evaluation/*`: suites, runs, tests, trace, report, cancel, compare; 20 тестов; выключен по умолчанию (`EVAL_UI_ENABLED=0`), прогресс — polling, SSE нет | — |

В текущем окружении проектные тесты проходят в `.venv` (Python 3.11.9):
`854 passed` (снимок 15.11.2026); frontend — `160 passed` в 24 файлах,
Playwright E2E — `7/7`, `tsc -b` и `vite build` без ошибок;
`openspec validate --all --strict` проходит. Реальная приёмка dev-корпуса —
4/4 (factoid, temporal, structural, abstain; снимок 02:44, после P1/P2 не
перепрогонялась). Benchmark: реальный smoke `12/12` (Recall@5 0.955);
последний полный реальный прогон 30 тестов — `23/30` (`pass_rate 0.767`,
execution coverage 1.0, Recall@5 0.889, Recall@10 0.926, MRR 0.889, evidence
coverage 0.897; easy 100%, medium 80%, hard 40%, multi_turn 60%; dev 19/24,
holdout 4/6), артефакт `eval/results/20261115_183541_0a97989/`. Предыдущий
прогон дал `24/30` (`174124`), поэтому per-test стабильность не доказана
(`repeat=1`). Проверка только ingestion-тестов не является приёмкой целого
продукта.

Не готово:

- полный корпус 9 PDF: документы 2019–2024 не парсились, производные артефакты
  и `verify_corpus.py` относятся только к 2015–2018;
- полный graph index: `b288dd3c…` собран на 1916/3928 чанков и имеет
  `status=incomplete`; активен только dev-граф `66c0dbe3…` (536 чанков,
  2022–2023);
- benchmark на полном корпусе и `expected_corpus_sha256` для него: текущие
  цифры (23/30) получены на dev-корпусе из 1025 чанков; holdout пересекается
  с dev по gold-чанкам в 4 тестах (`holdout_gold_overlap`);
- `context_chunk_ids` доступны не для всех кейсов, поэтому context coverage
  частично `not_measured`;
- judge-калибровка на размеченных ответах; реальные прогоны идут `--no-judge`;
- Evaluation SSE-прогресс (polling реализован; SSE остаётся опциональным);
- хвост этапа 5 React WebUI: удаление Streamlit; пагинация истории и часть
  a11y закрыты, bundle-split убрал warning >500 kB, но нет scroll-anchoring,
  полного focus-trap/аудита доступности и SSE Evaluation.

Issuer-фильтр в чанках и изоляция 2015 (БПС-Сбербанк) реализованы
(`enrich_chunks_with_registry`, `PerYearSearch._excluded_years`).

## 3. Корпус и интерпретация данных

### 3.1. Состав корпуса

Целевой список: `sber_2015`, `sber_2016`, `sber_2017`, `sber_2018`, `sber_2019`,
`sber_2020`, `sber_2022`, `sber_2023`, `sber_2024`. Отчёта за 2021 год нет.
Год нельзя выводить из содержимого PDF; он задаётся реестром источников и именем файла.

**Важное ограничение:** файл, названный «Копия Сбер 2015.pdf», по проверке содержит
отчёт БПС-Сбербанка (белорусская дочерняя компания), а не ПАО Сбербанк. Не считать его
сопоставимым с остальными годами без явной маркировки.

Каждый источник обязан иметь в реестре:

```text
doc_id, file_path, source_sha256, year, report_period, issuer,
country, source_status, inclusion_note
```

Для 2015 `issuer`/`country` должны явно указывать БПС-Сбербанк/Беларусь. UI и генератор
ответов должны показывать issuer в citations. По умолчанию временное сравнение
российского Сбербанка не смешивает 2015 с другими годами. Пользователь может запросить
этот документ явно и получить предупреждение о несопоставимости.

2021 считается **отсутствующим годом**, а не нулевым значением. Запрос «за 2021» должен
явно сообщать об отсутствии источника; временной ответ не должен интерполировать этот
год или выдавать значение соседнего периода за 2021.

### 3.2. Артефакты и lineage

Реестр источников хранится в `data/source_registry.json` и валидируется до первого
полного прогона. Для каждого `doc_id` ровно одна запись; `file_path` разрешается
относительно корня проекта или задаётся абсолютным путём в локальном overlay, который
не коммитится.

Все производные данные воспроизводимы из источников и настроек. Для каждого набора
чанков и индексов фиксируются:

- `source_sha256` исходных PDF;
- версии схемы элементов, чанков и индексов;
- параметры парсинга/чанкинга/эмбеддингов;
- timestamp сборки и версия приложения;
- список источников и пропущенных страниц.

Артефакты строятся атомарно. Resume допустим только при совпадении fingerprint источника,
версии схемы и параметров обработки. Сырые PDF остаются неизменяемым входом.

Целевые расположения:

```text
data/source_registry.json
data/elements/{doc_id}_elements.json
data/elements/{doc_id}_stats.json
data/structure/{doc_id}_sections.json
data/chunks/{doc_id}_chunks.json
data/cache/embeddings_{corpus_fingerprint}.json
data/indexes/{index_fingerprint}/manifest.json
data/graph/{graph_fingerprint}.json
```

Файлы корпуса могут быть исключены из Git из-за размера/лицензирования, но их hashes,
реестр без локальных секретов и инструкция повторного получения должны быть описаны.

## 4. Целевая архитектура

```text
PDF / реестр источников
  → parser (hi_res)
  → outline + дерево разделов
  → chunks + breadcrumbs
  → корпусная валидация
  → dense vectors + BM25 index + graph index
  → router / query plan
  → filtered hybrid retrieval
  → RRF → reranker
  → context packing
  → answer JSON (citations/facts/abstain)
  → Streamlit
  → evaluation artifacts
```

Компоненты должны быть вызываемы отдельно от UI. UI не содержит реализации retrieval,
индексации или генерации. UI вызывает `AgentService`; ограниченный агент оркестрирует
инструменты, а `RAGEngine` остаётся отдельным тестируемым backend-компонентом.

### 4.1. Модули целевого состояния

```text
src/core/
  config.py, models.py, credentials.py, rag_config.py
  embeddings.py, cache.py, dense_index.py, faiss_index.py
  corpus.py, source_registry.py, llm_lock.py
src/ingestion/
  pdf_parser.py, pdf_outline.py, structure_builder.py
  pdf_to_chunks.py, chunk_enricher.py, corpus_validator.py, cli.py
src/retrieval/
  bm25.py, rrf.py, reranker.py, rewrite.py
  hybrid_retriever.py, temporal_search.py
  graph_retriever.py, graph_hybrid.py
src/graph/
  graph_models.py, graph_builder.py, entity_extractor.py
src/engine/
  context.py, prompts.py, rag_engine.py, generator.py
  calculations.py, structure_compare.py, llm.py
src/agent/
  agent_graph.py, state.py, tools.py, service.py, factory.py
  selector.py, simple_router.py, llm_router.py
  coverage.py, verification.py, smalltalk.py
src/api/
  app.py, config.py, schemas.py, services.py, serialize.py, store.py
  auth.py, bootstrap.py, streaming.py, execution.py, evaluation_store.py
  routes/
src/ui/
  app.py, citations.py
eval/
  dataset.py, metrics.py, runner.py, report.py, trace.py
  run_eval.py            # alias eval.runner
scripts/
  build_corpus.py, build_dev_corpus.py, build_full_graph.py
  verify_corpus.py, acceptance_dev.py, eval_retrieval.py
```

Реализованные файлы сохраняют текущие публичные функции либо предоставляют явный
адаптер. Новая логика помещается в соответствующий слой, а не в CLI/UI.

## 5. Основные контракты данных

### 5.1. Chunk

Текущий контракт обязательных полей чанка из [spec_project.md](spec_project.md) §5.2
сохраняется. Дополнительно, если значение известно, используются:

```text
section_path: str
breadcrumbs: list[str]
section_level: int
parent_section: str | null
issuer: str
country: str
```

Минимум для retrieval: `chunk_id`, `evidence_id`, `doc_id`, `year`, `kind`, `text`,
`page_pdf`, `source_sha256`. `html` таблицы хранится без потери; таблица не может
разрезаться обычным текстовым splitter.

`chunk_id` уникален в корпусе. `evidence_id` идентифицирует исходную страницу/свидетельство,
но не заменяет `chunk_id` в citations.

### 5.2. Answer

Внутренний ответ движка:

```json
{
  "status": "answer",
  "answer": "Текст ответа",
  "citations": [
    {
      "chunk_id": "sber_2023:00042:c0:...",
      "doc_id": "sber_2023",
      "year": 2023,
      "issuer": "ПАО Сбербанк",
      "page_pdf": 42,
      "page_label": "41",
      "section_path": "Управление рисками/Кредитный риск",
      "quote": "Короткий дословный фрагмент источника"
    }
  ],
  "facts": [],
  "warnings": []
}
```

`status` принимает `answer` или `abstain`. Цитата обязана ссылаться только на переданный
в контекст чанк. При `abstain` поле `citations` пустое, а `warnings` объясняет причину
(`missing_year`, `no_evidence`, `low_confidence` или `scope_mismatch`). Полная схема
описана в `spec_retrieval_engine.md`.

## 6. Нефункциональные требования

- Python 3.11.x — целевая версия для воспроизводимой установки; зависимости должны быть
  разрешаемы из зафиксированного lock-файла или проверенной версии `requirements.txt`.
- Секреты только из env/`.env`; `.env` не коммитится и не выводится в логи.
- CLI и UI должны сообщать о незаданных ключах и недоступности API понятной ошибкой.
- При сбое LLM-reranker система использует документированный fallback на порядок RRF;
  при сбое генерации не выдаёт неподтверждённый ответ.
- Логи содержат этап, duration, model/provider, число кандидатов и статус. Ключи,
  base64-изображения и полный текст документов в логах запрещены.
- Retrieval должен работать на предварительно построенных локальных индексах;
  недоступность внешнего API не должна блокировать открытие корпуса/дерева структуры.
- Все JSON-артефакты UTF-8. Пути берутся из конфигурации и не зашиваются относительно
  конкретной машины.

## 7. План реализации и exit criteria

Переход между этапами — только после выполнения exit criteria.

### Этап A — достоверный корпус

1. Проверить реестр источников, включая issuer 2015 и отсутствие 2021.
2. Создать элементы, структуры и chunks для всех 9 документов либо зафиксировать
   обоснованное исключение по каждому недоступному документу.
3. Выполнить `verify_corpus`: дубликаты ID, страницы вне PDF, пустые чанки,
   несогласованные HTML-таблицы, неполные source hashes, доля чанков без структуры.
4. Ручное ревью минимум по одному разделу/таблице из каждого отчёта.

**Выход:** отчёт корпуса и версионированные артефакты; пропуски не скрыты.

### Этап B — baseline RAG

1. Dense retrieval и BM25 на одном контракте Chunk.
2. Метаданные, фильтры годов/issuer, контекстный budget и citations.
3. Запустить baseline dense-only и hybrid без графа на контрольном датасете.

**Выход:** end-to-end запрос из Python API возвращает валидный ответ или abstain с
проверяемыми citations.

### Этап C — structure-aware hybrid

Добавить router, structural eligibility, query rewriting только для нужных типов,
RRF и reranker. Структурный фильтр применяется до top-k и не выполняется только после
получения короткого пула кандидатов.

**Выход:** фактические и структурные сценарии проходят функциональные тесты; качество
сопоставлено с baseline.

### Этап D — graph, UI, evaluation

Построить graph index, подключить его как дополнительный retriever, добавить Streamlit
визуализации и evaluation pipeline. Провести ручной smoke, все автоматические тесты,
evaluation и повторную сборку артефактов из чистой среды.

**Выход:** демонстрация 4 сценариев из §1, воспроизводимый отчёт и инструкция README.

## 8. Критерии готовности проекта

Проект считается готовым к демонстрации, если:

- обработаны и проверены источники, issuer 2015 явно маркирован;
- для всех ответов на фактические вопросы есть citations на реальные chunks и страницы;
- реализованы фактологический, структурный, временной, multi-hop и abstain сценарии;
- отсутствующий 2021 год не подменяется и не интерполируется;
- структура и граф отображаются и объяснимы по исходным документам;
- evaluation запускается одной документированной командой и сохраняет JSON/CSV;
- результат сравнивается с dense-only baseline;
- тесты проходят в заявленной Python-версии; README описывает установку, обработку данных,
  сборку индексов, запуск UI и eval;
- все ограничения, ошибки парсинга и проваленные метрики перечислены, а не замалчиваются.

## 9. Источники правды и устранение расхождений

- Целевой контракт проекта: этот файл и связанные спецификации.
- Детали parser/artifact API: `spec_pdf_parser.md`.
- `spec_project.md` содержит подробный дизайн и учебный контекст; при расхождении
  приоритет имеют целевые контракты `spec_target.md` и профильные спеки.
- OpenSpec change `add-pdf-parser` описывает историческую работу парсера, не весь проект.
- Не заявлять в документации о наличии retrieval, graph, UI или eval до появления
  соответствующего кода и тестов.
