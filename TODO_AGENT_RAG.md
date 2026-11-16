# TODO: усиление Agent_RAG

**Статус:** backend-волна RAG (retrieval/agent) завершена (2026-11-14), HTTP API
и React WebUI этапов 1–4 работают, Evaluation API и React-раздел Evaluation
реализованы (2026-11-15). Архитектура не меняется: один LangGraph-агент,
`AgentService`, существующие инструменты и retrieval. Полный `pytest` —
`863 passed` (Python 3.11.9, снимок 15.11.2026), frontend - `160 passed` в 24
файле, Playwright E2E — `7/7`, реальная приёмка dev — `4/4`. Benchmark: датасет
30 тестов (10/10/5/5) + 6 holdout, реальный smoke `12/12`; последний полный
real run — `23/30` (pass_rate 0.767, Recall@5 0.889), предыдущий — `24/30`
(pass_rate 0.8). Чекбоксы ниже отражают наличие кода и тестов, а не
готовность полного корпуса или целевое качество benchmark.

**Источники требований:**
[spec_agent_enhancement.md](spec_agent_enhancement.md),
[spec_benchmark.md](spec_benchmark.md),
[docs/specs/web-ui-modernization.md](docs/specs/web-ui-modernization.md),
[docs/specs/evaluation-webui.md](docs/specs/evaluation-webui.md).

Обозначения: `[x]` сделано, `[~]` в работе, `[ ]` осталось.
Приоритеты: P0 — без этого нет демонстрации, P1 — ближайший этап,
P2 — после базового контура, P3 — полировка.

---

## 0. Уже реализовано

- [x] **A1. LLM Router + deterministic fallback** — `src/agent/llm_router.py`,
  `src/agent/simple_router.py`; годы берутся из `available_years`, без хардкода.
- [x] **A2. Plan-aware ActionSelector** — `src/agent/selector.py`: широкий поиск →
  недостающие годы → graph/structure.
- [x] **A3. Сужение фильтров действия** — `_merge_action_filters` в
  `src/agent/agent_graph.py`; действие не может расширить фильтры пользователя.
- [x] **A4. Полнота периода** — `expected_years`, `covered_years`, `missing_years`,
  `search_attempts`, `incomplete_period`.
- [x] **A5. Структурированные факты и расчёты** — `src/engine/calculations.py`,
  `Answer.calculations`, блок «Расчёт по подтверждённым фактам».
- [x] **A6. Temporal Structure Analysis** — `src/engine/structure_compare.py`,
  `get_structure_tree`, added/removed/renamed/restructured/split/merge,
  expanded/reduced, content `evolved`, опциональная LLM-проверка спорных пар.
  Отклонение от первоначального плана: отдельного инструмента
  `compare_structures` нет; сравнение запускается внутри `assess_evidence`
  для structural-запросов и складывает результат в `structural_changes`.
- [x] **A7. Query rewriting** — `src/retrieval/rewrite.py`, безопасные варианты,
  multi-query RRF.
- [x] **A8. Diversity после rerank** — `HybridRetriever(diversity=True)`.
- [x] **A9. Диагностика retrieval** — `RetrievalTrace`, `ToolEvent.diagnostics`,
  `AgentRunResult.diagnostics`, UI expander.
- [x] **A10. Контракт `search_reports`** — `queries`, `top_k`, `retrieval_mode`;
  `PerYearSearch` также принимает `top_k`/`retrieval_mode`.
- [x] **A11. UI: динамические годы/документы и блоки анализа** — `src/ui/app.py`.

---

## 1. CoverageReport и матрица «метрика × год» (P1) — сделано

Цель: отличать **«года нет в корпусе»** от **«год есть, но показатель не найден»**.

- [x] **C1.** `src/agent/coverage.py`:
  - `CoverageCell(metric, year, status, chunk_ids)`,
    `status = covered | not_found | year_absent`;
  - `CoverageReport(status, expected_years, covered_years, missing_years, cells)`;
  - `to_dict()`, `missing_cells()`, `is_covered()`, `is_year_absent()`.
- [x] **C2.** `QueryPlan` расширен с defaults:
  `aspects`, `required_metrics`, `requires_comparison`.
- [x] **C3.** Валидация новых полей в `_as_plan`: списки строк, лимиты,
  неизвестный metric id отбрасывается.
- [x] **C4.** Router:
  - `SimpleRouter` извлекает metric id по словарю;
  - `LLMRouter` принимает/возвращает `aspects`, `required_metrics`,
    `requires_comparison`.
- [x] **C5.** Selector использует coverage:
  - широкий поиск → `CoverageReport`;
  - адресные поиски по недостающим годам;
  - в запрос добавляются labels required metrics;
  - `year_absent` не ищется как обычный missing year.
- [x] **C6.** Generator получает `coverage` в `analysis_context`; промпт различает
  `not_found`, `year_absent` и запрещает интерполяцию.
- [x] **C7.** `AgentRunResult.diagnostics["coverage"]` заполняется фактическим
  отчётом.
- [x] **C8.** Тесты `tests/test_coverage.py`:
  `year_absent` vs `not_found`, complete/partial, helpers, serialization.

**Ограничение:** повторный поиск по конкретной паре `(metric, year)` ограничен
годовым уровнем селектора; отдельный per-metric retry — следующий шаг.

---

## 2. Verification ответа (P1) — сделано

- [x] **V1.** `verification.py`: issues/result, коды проверок.
- [x] **V2.** Проверка чисел/единиц/года факта по цитируемому чанку.
- [x] **V3.** Проверка причинных утверждений.
- [x] **V4.** Интеграция в `AgentService`, `Answer.verification`,
  `AgentRunResult.diagnostics["verification"]`; один repair-шаг.
- [x] **V5.** Тесты verification (`tests/test_verification.py`, 27 тестов).

---

## 3. Agent-level eval / benchmark (P1–P2)

- [x] **E1.** Датасет `eval/datasets/`: manifest, easy/medium/hard/multi_turn.
  30 тестов (10/10/5/5), `partial=false`; 12 smoke-тестов в split `dev` и
  6 holdout-тестов. Валидатор (`--validate-only`) проходит.
  **Факт качества:** последний реальный full-прогон 30 тестов — `23/30`
  (`pass_rate 0.767`, execution coverage 1.0, Recall@5 0.889, Recall@10 0.926,
  MRR 0.889), артефакт `eval/results/20261115_183541_0a97989/`; smoke —
  `12/12`. Предыдущий full-прогон — `24/30` (`174124`), поэтому per-test
  стабильность не доказана (`repeat=1`). Итог ниже цели: 7 падений, holdout
  даёт 4/6. В прогоне есть warning `holdout_gold_overlap` (4 теста).
  **Оговорка:** прогон выполнен на dev-корпусе из 1025 чанков, не на 10 PDF.
- [x] **E2.** `eval/trace.py`: единый `AgentRunResult.diagnostics`.
- [x] **E3.** `eval/metrics.py`: retrieval + behavior + temporal/structure/calculation.
- [x] **E4.** `eval/runner.py` + CLI, `summary.capabilities`, JSON/CSV.
- [~] **E5.** `eval/evaluators/judge.py` + judge-интерфейс; judge выключен по
  умолчанию (`EVAL_JUDGE_ENABLED`), калибровка судьи на размеченных ответах
  не выполнена, реальные прогоны идут с `--no-judge`.
- [x] **E6.** `eval/report.py`: Markdown/JSON-отчёт, сравнение запусков.
- [x] **E7.** Тесты схемы, метрик, runner и manifest-хэшей.

---

## 4. Graph retrieval в стандартном контуре (P2) — сделано

- [x] **G1.** `GraphAugmentedRetriever` в `create_agent_service` как доп. канал.
- [x] **G2.** Provenance: graph-only chunk в citations только после проверки.
- [x] **G3.** Fallback без graph index.
- [x] **G4.** Trace graph stage.
- [x] **G5.** Тесты graph unavailable/duplicates/provenance.

---

## 5. Retrieval trace stages и context diagnostics (P2) — сделано

- [x] **T1.** `RetrievalTrace.stages` (dense/BM25/graph/RRF/rerank/context).
- [x] **T2.** `PerYearSearch`/`GraphAugmentedRetriever` дополняют stages.
- [x] **T3.** `pack_context` → `context_chunk_ids`, `context_source`.
- [x] **T4.** UI: стадии и отброшенные чанки.
- [x] **T5.** Тесты на отсутствие синтетических стадий.

---

## 6. Query planning: aspects и качество rewriting (P2)

- [x] **Q1.** Использовать `QueryPlan.aspects` для отдельных поисков.
- [ ] **Q2.** Оценить rewriting на eval (recall/latency): в реальных прогонах
  `capabilities.rewriting=false`, отдельного замера нет.
- [~] **Q3.** Кэш/лимиты перефразов: лимиты есть (`max_variants=2`,
  `max_tokens`), кэша перефразов нет.
- [x] **Q4.** Тесты на отсутствие новых годов/сумм/сущностей.

---

## 7. Полный корпус (P0 для демонстрации)

> Корпус спарсен: `data/elements`, `data/structure`, `data/chunks` есть для
> всех 10 документов (13969 чанка, 98.4% со структурой, 1650 таблиц, 0 пустых),
> `verify_corpus.py` — 0 ошибок. Не закрыто: embeddings/FAISS для новых 10041
> чанков, полный graph index и benchmark на полном корпусе (нужен Cloud.ru API).
>
> Dev-приёмка 4 сценариев (factoid, temporal, structural, abstain) проходит
> `4/4` на dev-корпусе, но это не заменяет D5 на полном корпусе.

- [x] **D1.** 10 PDF, реестр источников, issuer 2015, отсутствие 2021.
- [~] **D2.** elements/structure/chunks для 2019–2024 пересобраны и сведены;
  embeddings/FAISS для новых 10041 чанков ещё нет.
- [ ] **D3.** Пересобрать graph index и provenance.
- [x] **D4.** Corpus validation на полном наборе: 10/10, 0 ошибок, 4 warning.
- [ ] **D5.** 4 обязательных сценария на полном корпусе: factoid, temporal,
  structural, abstain.
- [ ] **D6.** `expected_corpus_sha256` для benchmark (сейчас зафиксирован хэш
  dev-корпуса из 4 документов).

---

## 8. UI (P2–P3)

- [x] **U1.** Матрица coverage: metric × year.
- [x] **U2.** Таблица структурных изменений.
- [x] **U3.** Verification issues/severity.
- [x] **U4.** Diagnostics: stages, filters, budgets.
- [x] **U5.** React/Evaluation WebUI по `docs/specs/*` — этапы 1–4 реализованы:
  чат на assistant-ui, история, SSE, tool trace, evidence, кликабельные citations
  `[n]`, PDF-ссылки `#page=N`, таблицы/графики/CSV, структурное сравнение,
  отмена и восстановление, Evaluation UI и навигация чат↔Evaluation.
  **Оговорка:** этап 5 закрыт частично (нет удаления Streamlit; a11y и
  пагинация частично закрыты, vendor-split убрал warning >500 kB).
  Evaluation UI использует polling вместо SSE.

---

## 9. Инженерные задачи (P2–P3)

- [x] **I1.** Лимиты/флаги в `RAGSettings`/env: retrieval-лимиты и бюджеты
  агента (`max_tool_calls`, `max_search_calls`, `per_year_k`, `max_years`,
  `max_structure_docs`) вынесены в `src/core/rag_config.py`. Значения по
  умолчанию — 4/2, как в `spec_agent.md` §3.1.
- [ ] **I2.** Логи без prompt/CoT/секретов.
- [ ] **I3.** Cost/latency: кэш rewrites, лимит LLM-проверок — не реализованы;
  есть только лимит вариантов перефраза и `max_structure_docs`.
- [x] **I4.** README приведён к фактическому состоянию.
- [x] **I5.** Полный прогон тестов в Python 3.11: `863 passed` в `.venv`
  (Python 3.11.9), `160 passed` в 24 файлах frontend, Playwright E2E `7/7`,
  `tsc -b` без ошибок; `vite build` успешен, vendor-split убрал warning >500 kB.

---

## Definition of Done для Coverage-этапа

- [x] Для временного вопроса агент различает «нет года» и «нет метрики».
- [x] CoverageReport попадает в `AgentRunResult.diagnostics`.
- [x] Ответ не интерполирует отсутствующие значения.
- [x] Есть тесты на оба типа пропусков.
- [ ] Отдельный адресный retry по каждой недостающей `(metric, year)`-ячейке.

---

## 10. Известные расхождения кода и спецификаций (аудит 15.11.2026)

Требования спек, которые не реализованы и ранее не были отмечены как долг.
Приоритеты: P0 — влияет на корректность ответов, P1 — неполный контракт,
P2 — конфигурируемость/качество, P3 — документация и структура.

- [x] **K1 (P0).** Чанки обогащаются `issuer`/`country`/`source_status`/
  `report_period` из `data/source_registry.json` при сборке сервиса
  (`src/core/corpus.enrich_chunks_with_registry`, `create_agent_service`).
  Фильтр `issuers` работает, citations получают эмитента; отсутствующий реестр
  не блокирует запуск. Тесты — `tests/test_agent_factory.py`.
- [x] **K2 (P0).** `sber_2015` исключается из многочисленных временных сравнений
  по умолчанию (`PerYearSearch._excluded_years` + реестр в фабрике); прямой
  вопрос про 2015 или явный `allowed_doc_ids` сохраняет доступ. Warning —
  `incomparable_year_excluded`. Тесты — `tests/test_temporal_search.py`.
- [x] **K3 (P0).** `_load_graph_artifact` выбирает только артефакты с
  `status="complete"`, среди них — с наибольшим покрытием (`chunk_count`) и
  свежести; неполный индекс игнорируется, артефакт без манифеста остаётся
  fallback'ом для тестов/локальных сборок.
- [x] **K4 (P1).** Фабрика передаёт общий реранкер в `GraphAugmentedRetriever`
  (`max_candidates=settings.candidates`, `base_candidate_k=settings.rerank_pool`):
  базовые и графовые кандидаты реранжируются одним пулом (спека §5/§8.5).
- [x] **K5 (P1).** `/api/capabilities` отдаёт
  `structure_content_comparison=true`: backend уже умеет
  `compare_content_changes`/`content_changes` (`src/engine/structure_compare.py`).
  Флаг синхронизирован с `src/api/README.md`, спекой WebUI и моками frontend.
- [x] **K6 (P1).** Бюджеты агента приведены к `spec_agent.md` §3.1: 4 tool /
  2 search в `src/agent/agent_graph.py`, `RAGSettings` и
  `PlanAwareSelector.DEFAULT_MAX_SEARCHES`. Тесты обновлены.
- [x] **K7 (P1).** Evaluation и chat проходят через process-wide
  `ExecutionGate` (`src/api/execution.py`, приоритет chat > evaluation) и
  межпроцессный `LLMExecutionLock`; evaluation worker получает gate-прокси.
  Benchmark пишет `report.html` и `manual_review.json`; SSE-прогресс остаётся
  опциональным по `evaluation-webui.md` (polling — основной путь, capability
  `evaluation_sse=false` — честный флаг).
- [x] **K8 (P1).** Benchmark: `report.html`/`comparison.html` создаются
  (`eval/report.py`), `llm_lock`/`wait_for_llm` и exit code 5 (`llm_busy`)
  реализованы (`src/core/llm_lock.py`, `eval/runner.py`), `python -m
  eval.run_eval` — alias `eval.runner`. `manual_review.json` создаётся
  шаблоном для hard/multi_turn.
- [x] **K9 (P2).** `RAGSettings` содержит флаги `max_queries`,
  `diversity_enabled`, `max_per_doc`, `max_per_section`, `coverage_enabled`,
  `max_coverage_searches`, `verification_enabled`; diversity-лимиты и
  `verification` больше не зашиты в фабрике/сервисе.
- [x] **K10 (P2).** `SimpleRouter` выставляет `needs_graph` по словам
  связи/причинности и `intent="multi_hop"` при ≥2 метриках
  (`spec_agent_enhancement.md` P2, AC-P2-2/3).
- [x] **K11 (P2).** При пустом retrieval ответ различает «нет отчёта за год»
  (`no_evidence_for_period`) и «показатель не найден» (`missing_metric`);
  общий `no_evidence` остаётся только для случая без периода/coverage
  (`spec_retrieval_engine.md` §4.2/§6).
- [x] **K12 (P3).** Карта модулей `spec_target.md` §4.1 приведена к
  фактическим файлам (`temporal_search`, `graph_hybrid`, `selector`,
  `llm_router`, `src/api`, `eval/report.py` и т.д.); вымышленные
  `query_router.py`/`structure_filter.py`/`graph_rag_engine.py` убраны.
  `eval/run_eval.py` существует как явный alias `eval.runner`.
- [ ] **K13 (P3).** Покрыто: `eval/report.py --compare` (md/html),
  `LLMExecutionLock`/`llm_busy`/exit 5, `ExecutionGate`, `main()` Streamlit
  (headless `st`-стаб). Остаются непокрытыми: реальный LLM-путь (нужна сеть) и
  worker Evaluation с настоящим `run_suite` (offline-тесты используют fake
  runner; программный `run_suite` покрыт отдельно).

## 11. Волна 15.11.2026: FAISS, P1/P2

- [x] **F1.** Постоянный FAISS-индекс (`src/core/faiss_index.py`):
  `IndexFlatIP`, нормализованные векторы, атомарная запись
  `data/indexes/{fingerprint}/index.faiss + manifest.json`, проверка
  fingerprint/модели/порядка `chunk_id`, fallback на `DenseIndex`.
  Сборка — `scripts/build_indexes.py`; собраны индексы канонических 3928
  чанков и dev-корпуса 1025 чанков. Тесты — `tests/test_faiss_index.py`.
- [x] **P1.** Нормализация метрик (operating income/CET1/ROE), безопасный
  `section_year_absent`, проверка чисел-«годов» с счётными словами,
  repair-диагностика без утечки текста, `rerank_per_year` и `top_k` до 10.
- [x] **P2-eval.** `rewriting` и `graph_fingerprint` из trace/артефакта,
  strict structural/quotes, предупреждение о пересечении holdout/dev,
  warnings/failed checks в отчёте.
- [x] **P2-UI.** Пагинация истории, UUID v4 fallback, вкладки/фокус/aria-live,
  опциональная раздача SPA (`API_SERVE_SPA`), vendor-split без warning >500 kB.
- [~] **F2.** Полный graph-индекс по каноническим 3928 чанкам: собрано
  1916/3928 чанков (2015, 2016, часть 2017); 2012 чанков (2017–2018) упали
  из-за `PermissionDeniedError: Project not found` Cloud.ru. Артефакт
  `b288dd3c…` имеет `status=incomplete` и фабрикой не активируется; после
  восстановления ключа — `patch_graph_failed_chunks.py --workers 3`.
  Добавлен параллельный режим `build_full_graph.py --workers` и прогресс.
- [~] **F3.** Full benchmark после P1/P2 перепрогнан: `183541` — `23/30`
  (`pass_rate 0.767`, Recall@5 0.889, Recall@10 0.926, MRR 0.889; dev 19/24,
  holdout 4/6). README обновлён этим результатом. Реальная dev-приёмка после
  P1/P2 ещё не перепрогонялась: последний снимок `4/4` от 02:44. Прогон на
  полном корпусе 10 PDF по-прежнему требует готового корпуса.
- [x] **F4.** Инженерные исправления без API: `import sys` в `pdf_parser.py`,
  autouse-изоляция `LLM_LOCK_PATH` в `tests/conftest.py`, `pytest-timeout`,
  JSONL-чекпоинт graph-извлечения (`--checkpoint`/resume), TTL-retry
  `AgentProvider`, диагностика `scripts/check_llm_access.py`.
- [~] **F5.** Локальный парсинг 2019–2024 запущен без Cloud.ru API
  (`build_corpus.py --doc ...`); после него будут structure/chunks и
  `verify_corpus.py`; embeddings/graph по-прежнему требуют API.
- [ ] **F6.** Убрать `holdout_gold_overlap`: в `183541` gold-чанки
  `easy_010`, `medium_009`, `hard_005`, `multi_turn_005` пересекаются с dev, а
  значит holdout не полностью независим. Нужно пересмотреть gold или split.
- [ ] **F7.** Довести `context_chunk_ids` до всех кейсов: в `183541`
  context coverage частично `not_measured`; summary честно помечает это в
  `limitations`, но для полного отчёта нужен context manifest по каждому turn.

## 12. Волна 16.11.2026: структурный слой графа

- [x] **S1.** Пересобран dev-артефакт графа без повторного извлечения LLM:
  `a8ce50f6…` вместо `66c0dbe3…`. Причина бага — сборка 14.11 шла до
  пересборки `data/structure/*_sections.json`, поэтому в артефакт попали
  3 `Section` и 184 `CONTAINS`. Теперь 163 раздела, 485/536 `CONTAINS`
  (90.5%), медиана 4 чанка на раздел. `MENTIONS`/`RELATED_TO` идентичны,
  2 сироты `Other` слились с одноимёнными типизированными (1134 → 1132).
  Проверено `verify_graph_provenance.py` (0 нарушений) и загружено фабрикой.
- [x] **S2.** Структурное расширение в ретривере:
  `structural_neighbors()` (`graph_retriever`) + `_structure_candidates()`
  (`graph_hybrid`), бюджет `GRAPH_STRUCTURE_BUDGET=2` на сид из top-5 базовых
  хитов, потолок 8 кандидатов, скор 0.25 (ниже сущностных). Измерено офлайн:
  полнота пула 0.847 → 0.908 (`spec_graph.md` §8.6); случайный контроль того
  же объёма — 0. Тесты: `tests/test_graph.py`, `tests/test_graph_hybrid.py`,
  `tests/test_agent_factory.py`.
- [~] **S3.** Реранк-проверка выполнена на 37 размеченных вопросах (пул 20 и
  выданный top-5, по два прогона на бюджет). Итог: полнота пула устойчиво
  +0.03…+0.05 (0.74 → 0.77–0.79), MRR пула на 0.02–0.07 ниже (неразмеченные
  соседние чанки получают балл gold), на выданном top-5 эффект меньше разброса
  одинаковых конфигураций (±0.05 MRR). Бюджет 2 оставлен ради полноты пула,
  качество ответа не заявляется (`spec_graph.md` §8.6). Не покрыто: abstain-
  регрессия и полный корпус 3928 чанков — полный артефакт `b288dd3c…` неполный.
