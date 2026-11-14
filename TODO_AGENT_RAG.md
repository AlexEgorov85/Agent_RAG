# TODO: усиление Agent_RAG

**Статус:** рабочий checklist. Архитектура не меняется: один LangGraph-агент,
`AgentService`, существующие инструменты и retrieval.

**Текущая зона этого агента:** CoverageReport, `required_metrics`, матрица
«метрика × год». Файлы соседнего Verification-агента
(`src/agent/verification.py`, `tests/test_verification.py`) не изменяются.

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

## 2. Verification ответа (P1) — зона параллельного агента

> Эти файлы не изменяются текущим Coverage-агентом:
> `src/agent/verification.py`, `tests/test_verification.py`.

- [~] **V1.** `verification.py`: issues/result, коды проверок.
- [~] **V2.** Проверка чисел/единиц/года факта по цитируемому чанку.
- [~] **V3.** Проверка причинных утверждений.
- [~] **V4.** Интеграция в `AgentService`, `Answer.verification`,
  `AgentRunResult.diagnostics["verification"]`.
- [~] **V5.** Тесты verification.

---

## 3. Agent-level eval / benchmark (P1–P2)

- [ ] **E1.** Датасет `eval/datasets/`: manifest, easy/medium/hard/multi_turn.
- [ ] **E2.** `eval/trace.py`: единый `AgentRunResult.diagnostics`.
- [ ] **E3.** `eval/metrics.py`: retrieval + behavior + temporal/structure/calculation.
- [ ] **E4.** `eval/runner.py` + CLI, `summary.capabilities`, JSON/CSV.
- [ ] **E5.** `eval/judge.py` + калибровка.
- [ ] **E6.** `eval/report.py`.
- [ ] **E7.** Тесты схемы, метрик, runner, manifest/lock.

---

## 4. Graph retrieval в стандартном контуре (P2)

- [ ] **G1.** `GraphAugmentedRetriever` в `create_agent_service` как доп. канал.
- [ ] **G2.** Provenance: graph-only chunk в citations только после проверки.
- [ ] **G3.** Fallback без graph index.
- [ ] **G4.** Trace graph stage.
- [ ] **G5.** Тесты graph unavailable/duplicates/provenance.

---

## 5. Retrieval trace stages и context diagnostics (P2)

- [ ] **T1.** `RetrievalTrace.stages` (dense/BM25/graph/RRF/rerank/context).
- [ ] **T2.** `PerYearSearch`/`GraphAugmentedRetriever` дополняют stages.
- [ ] **T3.** `pack_context` → `context_chunk_ids`, `context_source`.
- [ ] **T4.** UI: стадии и отброшенные чанки.
- [ ] **T5.** Тесты на отсутствие синтетических стадий.

---

## 6. Query planning: aspects и качество rewriting (P2)

- [ ] **Q1.** Использовать `QueryPlan.aspects` для отдельных поисков.
- [ ] **Q2.** Оценить rewriting на eval (recall/latency).
- [ ] **Q3.** Кэш/лимиты перефразов.
- [ ] **Q4.** Тесты на отсутствие новых годов/сумм/сущностей.

---

## 7. Полный корпус (P0 для демонстрации)

- [ ] **D1.** 9 PDF, реестр источников, issuer 2015, отсутствие 2021.
- [ ] **D2.** Пересобрать elements/structure/chunks/embeddings.
- [ ] **D3.** Пересобрать graph index и provenance.
- [ ] **D4.** Corpus validation.
- [ ] **D5.** 4 обязательных сценария: factoid, temporal, structural, abstain.
- [ ] **D6.** `expected_corpus_sha256` для benchmark.

---

## 8. UI (P2–P3)

- [ ] **U1.** Матрица coverage: metric × year.
- [ ] **U2.** Таблица структурных изменений.
- [ ] **U3.** Verification issues/severity.
- [ ] **U4.** Diagnostics: stages, filters, budgets.
- [ ] **U5.** React/Evaluation WebUI по `docs/specs/*`.

---

## 9. Инженерные задачи (P2–P3)

- [ ] **I1.** Лимиты/флаги в `RAGSettings`/env.
- [ ] **I2.** Логи без prompt/CoT/секретов.
- [ ] **I3.** Cost/latency: кэш rewrites, лимит LLM-проверок.
- [ ] **I4.** README.
- [ ] **I5.** Полный прогон тестов в Python 3.11.

---

## Definition of Done для Coverage-этапа

- [x] Для временного вопроса агент различает «нет года» и «нет метрики».
- [x] CoverageReport попадает в `AgentRunResult.diagnostics`.
- [x] Ответ не интерполирует отсутствующие значения.
- [x] Есть тесты на оба типа пропусков.
- [ ] Отдельный адресный retry по каждой недостающей `(metric, year)`-ячейке.
