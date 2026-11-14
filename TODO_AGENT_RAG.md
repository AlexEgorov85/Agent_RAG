# TODO: усиление Agent_RAG

**Статус:** рабочий checklist. Архитектура не меняется: один LangGraph-агент,
`AgentService`, существующие инструменты и retrieval.

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

## 1. CoverageReport и матрица «метрика × год» (P1)

Цель: отличать **«года нет в корпусе»** от **«год есть, но показатель не найден»**.

- [ ] **C1.** Добавить `src/agent/coverage.py`:
  - `CoverageCell(metric, year, status, chunk_ids)`,
    `status = covered | not_found | year_absent`;
  - `CoverageReport(status, expected_years, covered_years, missing_years, cells)`;
  - `to_dict()` для `AgentRunResult.diagnostics["coverage"]`.
- [ ] **C2.** Расширить `QueryPlan` (обратная совместимость, defaults):
  - `aspects: list[str] = []` (≤ 4 поисковых аспекта);
  - `required_metrics: list[str] = []` (≤ 5 канонических metric id);
  - `requires_comparison: bool = False`.
- [ ] **C3.** Валидация новых полей в `_as_plan`:
  - списки строк, обрезка до лимитов;
  - неизвестный metric id отбрасывается, а не ломает весь план.
- [ ] **C4.** Router:
  - `SimpleRouter` извлекает metric id из вопроса по словарю
    (`net_profit`, `revenue`, `roe`, `roa`, `assets`, `capital`, `margin`);
  - `LLMRouter` принимает/возвращает `aspects`, `required_metrics`,
    `requires_comparison` и не выдумывает метрики вне словаря.
- [ ] **C5.** Selector:
  - после широкого поиска строит `CoverageReport`;
  - адресные поиски выполняет по недостающим `(metric, year)`, а не только по годам;
  - не повторяет already attempted `(metric, year)`;
  - при отсутствии года в корпусе ставит `year_absent`, а не `not_found`.
- [ ] **C6.** Generator получает `coverage` в `analysis_context` и обязан:
  - прямо называть отсутствующие годы/показатели;
  - не подставлять значение соседнего года.
- [ ] **C7.** Diagnostics: `AgentRunResult.diagnostics["coverage"]` заполняется
  фактическим отчётом, `None` — только если coverage не применялся.
- [ ] **C8.** Тесты:
  - `year_absent` vs `not_found`;
  - несколько метрик в одном вопросе;
  - selector ищет только недостающие ячейки;
  - генератор не интерполирует.

**Файлы:** `src/agent/coverage.py`, `state.py`, `simple_router.py`,
`llm_router.py`, `selector.py`, `agent_graph.py`, `generator.py`.

---

## 2. Verification ответа (P1)

Цель: проверять не только membership `chunk_id`, но и смысл числовых/причинных
утверждений.

- [ ] **V1.** Добавить `src/agent/verification.py`:
  - `VerificationIssue(code, severity, message, chunk_ids)`;
  - `VerificationResult(status, issues)`;
  - коды: `citation_outside_manifest`, `fact_without_citation`,
    `fact_value_not_in_source`, `unit_mismatch`, `period_mismatch`,
    `period_incomplete`, `causal_claim_unsupported`, `structure_unverified`.
- [ ] **V2.** Проверка чисел:
  - нормализованный `value` факта должен встречаться в тексте/HTML
    цитируемого чанка;
  - `unit`/`scale` не должны противоречить источнику;
  - `year` факта не подменяется годом чанка.
- [ ] **V3.** Проверка причинности:
  - утверждения «потому что / вследствие / из-за» без citation на источник
    причины помечаются `causal_claim_unsupported`;
  - при severe-статусе ответ переводится в `abstain` или repair.
- [ ] **V4.** Интеграция:
  - `AgentService._generate_answer` вызывает verification после citations/facts;
  - `AgentRunResult.diagnostics["verification"]` заполняется `to_dict()`;
  - один repair-цикл, без нового агента.
- [ ] **V5.** Тесты:
  - число в ответе не найдено в чанке;
  - разные единицы;
  - причинный вывод без источника;
  - валидный ответ проходит.

**Файлы:** `src/agent/verification.py`, `service.py`, `state.py`,
`diagnostics`, `tests/test_verification.py`.

---

## 3. Agent-level eval / benchmark (P1–P2)

- [ ] **E1.** Датасет `eval/datasets/`:
  - `manifest.json` с `dataset_version`, `expected_corpus_sha256`, `suite_hashes`;
  - `easy.json`, `medium.json`, `hard.json`, `multi_turn.json`;
  - поля: `id`, `question`, `expected_intent`, `expected_years`,
    `required_metrics`, `gold_evidence`, `expected_status`, `tags`.
- [ ] **E2.** `eval/trace.py`: единый `AgentRunResult.diagnostics` без prompt/CoT.
- [ ] **E3.** `eval/metrics.py`:
  - retrieval: hit/MRR/recall/nDCG;
  - behavior: tool sequence, coverage completeness, citation validity,
    abstain correctness;
  - temporal accuracy, structure accuracy, calculation correctness.
- [ ] **E4.** `eval/runner.py` + CLI:
  - запуск suite, baseline retrieval vs agent;
  - сохранение JSON/CSV, `summary.capabilities`;
  - недоступная возможность = `skipped`/`not_measured`, не заглушка.
- [ ] **E5.** `eval/judge.py`: LLM-судья только после детерминированных метрик;
  калибровка на 3 контрольных кейсах.
- [ ] **E6.** `eval/report.py`: Markdown/JSON отчёт с проваленными тестами и
  ограничениями.
- [ ] **E7.** Тесты схемы, метрик, runner, lock/manifest.

**Файлы:** `eval/`, `data/eval/`, `scripts/eval_agent.py`, `README.md`.

---

## 4. Graph retrieval в стандартном контуре (P2)

- [ ] **G1.** Подключить `GraphAugmentedRetriever` в `create_agent_service` как
  дополнительный канал, если graph artifact существует.
- [ ] **G2.** Provenance: graph-only chunk попадает в citations только после
  `get_chunk`/поиска и проверки manifest.
- [ ] **G3.** Fallback: нет graph index → обычный hybrid + `graph_unavailable`,
  без падения.
- [ ] **G4.** Trace: отдельный graph stage (`graph_candidates`, `graph_added`,
  `graph_failed`).
- [ ] **G5.** Тесты: graph unavailable, duplicate evidence, provenance violation.

**Файлы:** `factory.py`, `graph_hybrid.py`, `tools.py`, `state.py`.

---

## 5. Retrieval trace stages и context diagnostics (P2)

- [ ] **T1.** `RetrievalTrace.stages` (dense/BM25/graph/RRF/rerank/returned/context)
  с опциональными списками; при отсутствии данных `available=false`.
- [ ] **T2.** `PerYearSearch` и `GraphAugmentedRetriever` копируют/дополняют stages.
- [ ] **T3.** `pack_context` возвращает `context_chunk_ids`; service пишет
  `diagnostics["context_chunk_ids"]` и `context_source`.
- [ ] **T4.** UI: отображать стадии, сколько отброшено, какие чанки попали в контекст.
- [ ] **T5.** Тесты на отсутствие синтетических стадий и на context source.

---

## 6. Query planning: aspects и качество rewriting (P2)

- [ ] **Q1.** Использовать `QueryPlan.aspects` в selector: отдельный поиск на
  аспект (например, «прибыль», «доходы», «расходы») в пределах бюджета.
- [ ] **Q2.** Оценить rewriting на eval: сколько даёт recall, сколько стоит latency.
- [ ] **Q3.** Ограничить/кэшировать перефразы; не дублировать запросы.
- [ ] **Q4.** Тесты на отсутствие новых годов/сумм/сущностей.

---

## 7. Полный корпус (P0 для демонстрации)

- [ ] **D1.** Собрать PDF 9 отчётов, проверить реестр источников,
  `issuer`/`country` для 2015 (БПС-Сбербанк), отсутствие 2021.
- [ ] **D2.** Пересобрать `elements`, `structure`, `chunks`, embeddings cache.
- [ ] **D3.** Пересобрать graph index и manifest/provenance.
- [ ] **D4.** Запустить corpus validation: дубликаты ID, страницы, пустые чанки,
  HTML-таблицы, source hashes.
- [ ] **D5.** Прогнать 4 обязательных сценария: factoid, temporal, structural,
  abstain (2021).
- [ ] **D6.** Зафиксировать `expected_corpus_sha256` для benchmark.

**Файлы/данные:** `data/raw/`, `data/source_registry.json`, `data/elements/`,
`data/structure/`, `data/chunks/`, `data/graph/`, `eval/datasets/`.

---

## 8. UI (P2–P3)

- [ ] **U1.** Матрица coverage: metric × year, `covered/not_found/year_absent`.
- [ ] **U2.** Таблица структурных изменений: год, раздел, тип, evidence.
- [ ] **U3.** Verification: issues/severity, без traceback и секретов.
- [ ] **U4.** Diagnostics: retrieval stages, filters, budgets.
- [ ] **U5.** React/Evaluation WebUI — по
  `docs/specs/web-ui-modernization.md` и `docs/specs/evaluation-webui.md`;
  Streamlit остаётся рабочим fallback.

---

## 9. Инженерные задачи (P2–P3)

- [ ] **I1.** Вынести бюджеты/лимиты в `RAGSettings`/env:
  `MAX_TOOL_CALLS`, `MAX_SEARCH_CALLS`, `per_year_k`, `max_years`,
  `max_structure_docs`, `LLM_ROUTER_ENABLED`, `LLM_REWRITE_ENABLED`,
  `LLM_STRUCTURE_VERIFY`.
- [ ] **I2.** Логи: request_id, intent, tool sequence, coverage, duration,
  без prompt/CoT/секретов.
- [ ] **I3.** Cost/latency: кэш rewrites, лимит LLM-проверок структуры,
  отключаемый reranker.
- [ ] **I4.** README: запуск, флаги, ограничения, benchmark.
- [ ] **I5.** Полный прогон тестов в Python 3.11.

---

## Рекомендуемый порядок

1. **C1–C8** — CoverageReport + `required_metrics`: закрывает главный разрыв
   «год есть / показателя нет».
2. **V1–V5** — verification чисел и причинности.
3. **E1–E7** — agent-level eval, чтобы измерять эффект, а не обсуждать.
4. **U1–U4** — показать coverage/verification/structure в UI.
5. **G1–G5**, **T1–T5**, **Q1–Q4** — graph, stages, aspects.
6. **D1–D6** — полный корпус для демонстрации.
7. **I1–I5** — эксплуатационная полировка.

## Definition of Done для следующего этапа

- [ ] Для временного вопроса агент различает «нет года» и «нет метрики».
- [ ] CoverageReport попадает в `AgentRunResult.diagnostics`.
- [ ] Селектор делает адресные поиски по недостающим ячейкам, не повторяя их.
- [ ] Ответ не интерполирует отсутствующие значения.
- [ ] Есть тесты на оба типа пропусков и на повторный поиск.
