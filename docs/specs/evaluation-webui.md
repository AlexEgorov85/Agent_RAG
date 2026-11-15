# Evaluation / Benchmark WebUI — Agent_RAG

**Статус:** In progress (обновление 15.11.2026): Evaluation API
(`/api/evaluation/*`) и storage реализованы и покрыты тестами (20 API-тестов);
React-раздел `/evaluation` реализован (обзор, запуски, тесты, сравнение, trace,
отмена), есть навигация чат↔Evaluation и 2 Playwright E2E-сценария. Прогресс
отдаётся polling'ом, SSE-канала нет. Chat и evaluation сериализуют LLM через
`ExecutionGate` и межпроцессный `LLMExecutionLock` (занятый lock → `llm_busy`).
Раздел выключен по умолчанию (`EVAL_UI_ENABLED=0`).
Полный прогон 30 тестов доступен из UI; последний зафиксированный результат —
23/30 (`pass_rate 0.767`, Recall@5 0.889) на dev-корпусе, см.
`eval/results/20261115_183541_0a97989/`; предыдущий — 24/30 (`174124`). В UI
пока нет SSE-прогресса, выбора `attempt > 0` и перехода из chunk теста в
Evidence Panel чата.

**Версия:** 1.1
**Дата:** 14.11.2026
**Тип изменения:** UI-раздел + Evaluation API
**Целевая архитектура:** React + TypeScript WebUI, FastAPI adapter, существующий Python Agent_RAG
**Размещение:** `docs/specs/evaluation-webui.md`

**Связанные документы:** [spec_benchmark.md](../../spec_benchmark.md),
[web-ui-modernization.md](web-ui-modernization.md), [spec_ui_eval.md](../../spec_ui_eval.md),
[spec_agent.md](../../spec_agent.md), [spec_agent_enhancement.md](../../spec_agent_enhancement.md)

> Это целевой контракт, а не описание уже реализованного. Benchmark-контур описан в
> `spec_benchmark.md`; React WebUI и общий API — в `web-ui-modernization.md`. Этот
> документ связывает их и не создаёт вторую реализацию агента.
>
> **Иерархия источников истины.** `spec_benchmark.md` — датасет, trace, метрики,
> артефакты, CLI-runner. `web-ui-modernization.md` — общий React UI и API adapter.
> Этот документ — только раздел Benchmark в React и Evaluation API.
> `spec_ui_eval.md` §1 (Streamlit) — исторический контракт и здесь не действует.

---

## 1. Назначение

Добавить в WebUI Agent_RAG отдельный раздел **Evaluation / Benchmark**, который
позволяет:

- запускать smoke/full/category benchmark;
- видеть историю запусков;
- просматривать сводный отчёт;
- открывать детализацию отдельного теста и retrieval trace;
- сравнивать два запуска и видеть улучшения и регрессии;
- контролировать качество после изменения retrieval, reranker, prompt или agent loop.

Раздел не встраивается в обычный чат и не смешивается с историей пользовательских
исследований.

### 1.1. Две ответственности

| Компонент | Отвечает за | Не должен |
|---|---|---|
| **Benchmark Runner** | Загрузка набора, запуск `AgentService`, оценка, запись артефактов | Показывать UI, хранить пользовательские диалоги |
| **Evaluation Dashboard** | История, статус, отчёты, тесты, сравнение | Запускать retrieval/LLM, пересчитывать метрики, менять эталоны |

Runner — это `eval.runner` из `spec_benchmark.md`. Dashboard — React-раздел и
тонкий Evaluation API. Dashboard читает результаты, но не выполняет benchmark
самостоятельно.

### 1.2. Не цели

- Перенос benchmark в React или браузер.
- Вторая копия `AgentService` или второй retrieval.
- Запуск benchmark внутри обычного HTTP-запроса.
- Смешивание benchmark-результатов с историей чата.
- Редактор эталонов в первой версии.
- Автоматическое создание новых тестов.
- Сложная аналитическая платформа и кастомные дашборды.
- Автоматическое изменение gold evidence по результатам прогона.

### 1.3. Критерий успеха

Разработчик открывает раздел Evaluation, запускает Smoke, видит прогресс, после
завершения открывает сводку, находит проваленный тест, понимает на каком этапе
потеряно доказательство, сравнивает запуск с предыдущим и видит, что именно
улучшилось или сломалось.

### 1.4. Источники данных

| Артефакт `spec_benchmark.md` | Где используется |
|---|---|
| `summary.json` | Обзор, карточки, история, сравнение |
| `cases.jsonl` | Список тестов, каталог, сравнение |
| `traces/{test_id}.{retrieval_mode}[.attempt_n].json` | Детализация теста, retrieval trace |
| `retrieval.csv` | Таблица этапов и loss attribution |
| `report.html` / `report.md` | Отчёт запуска (опционально) |
| `dataset/*.json` + `manifest.json` | Определения тестов, gold evidence, версия корпуса |

UI не создаёт эти артефакты и не пересчитывает их метрики.

Идентичность результата теста — `(run_id, test_id, retrieval_mode, attempt_index)`.
`attempt_index` относится к повтору конкретной пары `test_id + retrieval_mode`.
Поэтому `test_id` сам по себе не уникален внутри run'а с `retrieval_mode=all`.

---

## 2. Место в WebUI

### 2.1. Навигация

Верхний уровень приложения:

```text
Agent_RAG
├── Research / Chat
└── Evaluation / Benchmark
```

Правила:

- Evaluation — отдельный route, а не вкладка внутри исследования;
- переход не теряет выбранное исследование;
- в Evaluation нет composer и пользовательских сообщений;
- при `evaluation_ui=false` пункт скрыт, прямой route показывает честное
  `evaluation_disabled`, а не пустой экран;
- в local-режиме раздел доступен локальному владельцу; при включённой
  аутентификации — только роли `evaluator` (см. §12).

### 2.2. Четыре вкладки

```text
Evaluation
├── Обзор
├── Запуски
├── Тесты
└── Сравнение
```

| Вкладка | Назначение |
|---|---|
| Обзор | Последний запуск, сравнение с предыдущим, четыре направления, проблемы, динамика |
| Запуски | Запуск benchmark, очередь, статус, история, отчёт |
| Тесты | Каталог вопросов, фильтры, детализация теста и trace |
| Сравнение | Два запуска, метрики, дельты, регрессии |

Вкладки не дублируют данные: Обзор — агрегат, Запуски — операции, Тесты —
диагностика, Сравнение — анализ изменений.

---

## 3. Модель данных и статусы

### 3.1. Evaluation Run

```typescript
type EvaluationRunStatus =
  | "queued"
  | "running"
  | "cancel_requested"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

interface EvaluationRun {
  id: string;
  status: EvaluationRunStatus;
  suite: "smoke" | "full" | "easy" | "medium" | "hard" | "multi_turn";
  split: "dev" | "holdout" | "all";
  retrieval_mode: "hybrid" | "dense" | "keyword" | "all";
  judge: boolean;
  repeat: number;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  progress: EvaluationProgress;
  dataset: {
    version: string;
    sha256: string;
    tests_total: number;
  };
  versions: EvaluationVersions;
  summary?: EvaluationSummary;
  error?: EvaluationError;
  artifacts_url?: string;
}

// Полная схема — summary.json из spec_benchmark.md §6.2.
type EvaluationSummary = Record<string, unknown>;

interface EvaluationProgress {
  tests_total: number;
  tests_completed: number;
  current_test_id?: string;
  current_turn?: number;
  percent: number;
}

interface EvaluationVersions {
  git_sha: string;
  git_dirty: boolean;
  corpus_content_sha256: string;
  corpus_fingerprint: string;
  config_fingerprint: string;
  llm_model: string;
  reranker_model?: string;
  judge_model?: string;
  prompt_hashes: Record<string, string>;
}

interface EvaluationError {
  code: string;
  message: string;
  retryable: boolean;
  test_id?: string;
}
```

`summary` — объект `summary.json` из `spec_benchmark.md` §6.2. Dashboard не
пересчитывает его поля.

### 3.2. Результат теста

```typescript
interface EvaluationTestResult {
  test_id: string;
  retrieval_mode: "hybrid" | "dense" | "keyword";
  attempt_index: number;
  suite: string;
  category: string;
  difficulty: "easy" | "medium" | "hard";
  status: "completed" | "failed" | "timeout" | "error" | "skipped";
  pass: boolean;
  metrics: {
    recall_at_5?: number | null;
    recall_at_10?: number | null;
    mrr?: number | null;
    retrieval_coverage?: number | null;
    evidence_coverage?: number | null;
    context_coverage?: number | null;
    loss_stage?: string | null;
    not_measured?: string[];
  };
  answer: {
    automatic_pass?: boolean | null;
    rubric_total?: number | null;
    fact_accuracy?: number | null;
    citation_validity?: number | null;
    citation_coverage?: number | null;
    calculation_accuracy?: number | null;
  };
  critical_errors: Array<{ code: string; message?: string }>;
  warnings: string[];
  efficiency: {
    tool_calls?: number;
    search_calls?: number;
    llm_calls?: number | null;
    latency_ms?: number;
  };
  trace_path?: string | null;
  trace_available: boolean;
}

// Уникальный ключ: run_id + test_id + retrieval_mode + attempt_index.
// test_id сам по себе не уникален при retrieval_mode="all".
```

### 3.3. Переходы статусов

```text
queued → running → completed
              ├→ cancel_requested → cancelled
              ├→ failed
              └→ interrupted
queued → cancelled
queued → interrupted
```

Правила:

- `completed` не означает, что все тесты прошли; это статус выполнения.
- `failed` — ошибка runner/окружения, а не провал теста.
- `cancel_requested` — не терминальный статус: пользователь запросил отмену,
  worker завершает текущий turn и освобождает LLM-lock.
- `cancelled` — отменено пользователем; частичные результаты сохраняются.
- `interrupted` — API перезапущен во время выполнения; UI не показывает вечный
  spinner и предлагает повторить.
- Переходы назад запрещены.
- Если worker не смог остановиться за per-turn timeout, run переходит в `failed` с
  `error.code=cancel_timeout`; зависший `cancel_requested` запрещён.

### 3.4. Независимость статусов

Четыре уровня статусов не выводятся один из другого:

| Уровень | Где | Значения |
|---|---|---|
| Agent execution | `AgentRunResult.status` | `completed` / `abstained` / `failed` |
| Answer | `Answer.status` | `answer` / `abstain` |
| Test result | `cases.jsonl`, `EvaluationTestResult.status` | `completed` / `failed` / `timeout` / `error` / `skipped` |
| Evaluation run | `EvaluationRun.status` | `queued` / `running` / `cancel_requested` / `completed` / `failed` / `cancelled` / `interrupted` |

Правила:

- `abstained` — успешное выполнение агента; UI не показывает его как `failed` или
  ошибку benchmark.
- `Answer.status=abstain` — штатный ответ; тест проходит, если
  `expected_status=abstain`.
- Run `completed` может содержать проваленные тесты: это разные уровни.
- `unsupported_mode`/`invalid_dataset`/`judge_error` — причины и коды, а не статусы
  теста; mapping определён в `spec_benchmark.md` §5.9.
- Полные правила прохождения и denominator — `spec_benchmark.md` §5.9–§5.10; этот
  документ их не переопределяет.

---

## 4. Вкладка «Обзор»

### 4.1. Карточки

Показываются четыре независимые метрики последнего завершённого запуска:

| Карточка | Значение | Источник |
|---|---|---|
| Успешные тесты | `passed / eligible` и `pass_rate`; рядом `execution_coverage` | `summary.coverage`, `summary.pass_rate`, `summary.execution_coverage` |
| Recall@10 | macro mean | `summary.groups.retrieval.recall_at_10` |
| Качество ответов | `8.2 / 10` rubric mean, если есть judge/manual; иначе automatic pass rate с явной подписью | `summary.groups.answer.rubric_mean` или `automatic_pass_rate` |
| Multi-turn | scenario success rate | `summary.groups.dialogue.scenario_success_rate` |

Требования:

- не показывать `0`, если метрика не измерялась; показывать `—`;
- pass rate всегда сопровождается coverage: `18/20 eligible · 20/30 executed` — это
  разные числа, и оба обязаны быть видны;
- рядом с каждой карточкой — дельта к предыдущему сопоставимому запуску с тем же
  suite/split: `▲ +2.1 п.п.`, `▼ −1.4 п.п.` или `—`;
- дельта показывается только при сопоставимости по §7.4; иначе `—` и ссылка на
  side-by-side;
- если предыдущего сопоставимого запуска нет, дельта отсутствует, а не сравнивается
  с другим набором;
- не сворачивать четыре карточки в один `Agent Quality`;
- значения не вычисляются во frontend из сырых traces.

### 4.2. Последние запуски

Таблица:

| Дата | Набор | Режим | Результат | Coverage | Длительность | Статус |
|---|---|---|---|---|---|---|
| 14.11 | Smoke | hybrid | 90% (9/10) | 10/12 | 4 мин | completed |
| 12.11 | Full | hybrid | 87% (26/30) | 30/30 | 38 мин | completed |
| 09.11 | Full | dense | 83% (25/30) | 30/30 | 35 мин | completed |

Правила:

- сортировка по `created_at` убыванию;
- статус `running`/`cancel_requested` показывает progress;
- `failed`/`interrupted` не маскируются процентом;
- `pass_rate` не показывается без `execution_coverage`;
- клик открывает карточку запуска.

### 4.3. Качество по направлениям

Четыре независимые полосы:

```text
Retrieval       91%
Answer          84%
Analysis        78%
Dialogue        80%
```

- `Retrieval` — `recall_at_10` и `evidence_coverage`;
- `Answer` — automatic pass rate / rubric;
- `Analysis` — pass rate medium/hard reasoning;
- `Dialogue` — multi-turn success;
- пустое значение — `—`, не `0%`.

### 4.4. Динамика качества

Линейный график по завершённым запускам:

- ось X — дата/ID запуска;
- ось Y — pass rate или выбранная метрика;
- точки только для реальных запусков;
- пропуск запуска не соединяется прямой и не изображается нулём;
- переключатель метрики: pass rate / Recall@10 / Answer quality / Multi-turn;
- tooltip показывает suite, split, mode, git SHA;
- сравнение несопоставимых наборов визуально помечается.

### 4.5. Проблемы

Блок «Обнаруженные проблемы»:

| Проблема | Кол-во | Тесты |
|---|---|---|
| Пропущенные доказательства | 4 | hard_002, ... |
| Неверные числовые ответы | 2 | medium_004, ... |
| Потеря контекста диалога | 3 | multi_turn_002, ... |
| Невалидные citations | 1 | ... |

Счётчики берутся из `summary.critical_errors` и `problem_tests`; UI не
классифицирует ошибки самостоятельно.

### 4.6. Пустые состояния

- Нет запусков: объяснение и кнопка «Запустить Smoke».
- Последний запуск `running`: карточки показывают progress, а не старые значения.
- Последний запуск `failed`: показывается ошибка и ссылка на предыдущий успешный.
- Нет `summary`: значения `—`, без выдуманных чисел.

---

## 5. Вкладка «Запуски»

### 5.1. Форма запуска

Поля:

| Поле | Значение | Правила |
|---|---|---|
| Набор | `smoke`, `full`, `easy`, `medium`, `hard`, `multi_turn` | `smoke` всегда `dev`; `full` включает holdout с подтверждением |
| Split | `dev`, `holdout`, `all` | holdout требует `confirm_holdout=true` |
| Retrieval mode | `hybrid`, `dense`, `keyword`, `all` | проверяется capability; неподдержанный режим — 422 |
| Judge | on/off | по умолчанию off; недоступен без judge-модели |
| Repeat | 1–5 | по умолчанию 1; >1 помечается как проверка стабильности |
| Комментарий | текст | необязательный, до 500 символов |

Кнопки:

- «Запустить Smoke» — основная;
- «Запустить Full» — требует подтверждения;
- «Запустить holdout» — отдельное подтверждение и предупреждение, что набор
  не используется для настройки;
- «Отменить» — для активного запуска.

Правила:

- одновременно допускается один выполняющийся benchmark;
- Full и holdout показывают предупреждение о длительности (десятки минут) до
  подтверждения;
- при активном запуске кнопки запуска disabled с объяснением;
- очередь ограничена; при переполнении — понятная ошибка `queue_full`;
- повторный клик с тем же `client_request_id` не создаёт второй запуск;
- `retrieval_mode=all` создаёт отдельный результат и отдельный trace на каждый режим;
  UI не схлопывает `hybrid`/`dense`/`keyword` в одну строку;
- форма не позволяет выбрать несуществующий suite.

### 5.2. История запусков

Таблица с фильтрами:

- suite;
- status;
- retrieval mode;
- split;
- дата;
- только с регрессиями (для завершённых).

Колонки: дата, suite, split, mode, статус, pass rate, длительность, git SHA,
модель, действия.

### 5.3. Карточка запуска

Показывает:

- статус и progress;
- suite/split/mode/judge/repeat;
- версии: git SHA + dirty, dataset hash/version, corpus content hash, corpus fingerprint,
  model;
- сводку четырёх групп и coverage (eligible/selected/executed);
- предупреждение, если часть возможностей (`capabilities`) недоступна и метрики
  помечены `not_measured`;
- список problem tests;
- ссылки: «Открыть отчёт», «Открыть тесты», «Сравнить с предыдущим»;
- ошибку, если запуск failed;
- частичные результаты, если cancelled/interrupted.

### 5.4. Запуск в фоне

`POST /api/evaluation/runs` не выполняет benchmark в HTTP-запросе:

1. Валидирует запрос и capability.
2. Создаёт запись `queued`.
3. Ставит задачу в очередь worker'а.
4. Возвращает `202 Accepted` с `run_id` и `status_url`.

UI:

- сразу показывает запуск в списке;
- опрашивает `GET /runs/{id}` каждые 2 секунды, пока статус `queued`/`running`/
  `cancel_requested`;
- при ошибке polling — backoff и честный статус;
- после терминального статуса загружает summary и tests;
- SSE-обновления — опциональная capability, не обязательны в первой версии.

---

## 6. Вкладка «Тесты»

### 6.1. Каталог

Источник — `eval/datasets/*.json`; последний статус — из последнего завершённого
запуска. Если запуск использовал несколько режимов, каталог показывает статус по
выбранному режиму (по умолчанию `hybrid`) и не смешивает режимы в одной ячейке.

Фильтры:

- difficulty: easy / medium / hard / multi_turn;
- category;
- suite;
- статус последнего выполнения: passed / failed / not_run;
- наличие critical errors;
- текстовый поиск по ID/вопросу.

Колонки: ID, вопрос (кратко), difficulty, category, последний статус, coverage,
critical errors, последний запуск.

### 6.2. Детализация теста

Экран должен отвечать на вопрос «где именно сломалось», а не только «тест
провален».

Секции:

1. **Заголовок** — test_id, retrieval mode, attempt, статус, difficulty, category,
   suite, запуск.
2. **Вопрос и эталон** — вопрос, expected status, expected facts/years/documents,
   reference answer (если есть).
3. **Query Plan** — intent, years, metrics/aspects, needs_graph, answer_mode,
   effective filters.
4. **Retrieval** — таблица этапов:
   - Dense candidates;
   - BM25 candidates;
   - Graph;
   - RRF;
   - Reranked;
   - Returned;
   - Context.
   Для каждого этапа: count, top chunks, gold/missing, loss stage.
5. **Evidence** — покрытие по годам/показателям:
   - `найдено`;
   - `не найдено`;
   - `год отсутствует в корпусе`.
   Пропуск не показывается как ноль.
6. **Ответ агента** — текст, citations, facts, calculations, warnings, status.
7. **Оценка** — четыре группы, rubric (если есть), automatic checks.
8. **Critical errors** — коды и пояснения.
9. **Диагностика** — вероятная причина:
   - `never_retrieved`;
   - `lost_at_fusion`;
   - `lost_at_rerank`;
   - `lost_at_context`;
   - `not_cited`;
   - `unsupported_claim`;
   - `dialogue_context_lost`.
10. **Trace** — ссылка/раскрываемый просмотр полного trace, без prompt и
    chain-of-thought.

Пример диагностики:

```text
Тест: сравнение показателей 2020–2024
Failed · Hard · Temporal comparison

Query Plan: years=[2020..2024], intent=temporal
Retrieval: Dense 15 · BM25 20 · RRF 18 · Reranked 10
Evidence: 2020 ✓ · 2021 — · 2022 ✓ · 2023 ✓ · 2024 ✓
Ответ: полный пятилетний ряд без раскрытия пропуска 2021
Critical: missing_required_year_claimed_complete
Diagnostics: evidence за 2021 отсутствует; ответ не назвал gap
```

Требования:

- gold evidence и найденные chunks визуально различаются;
- клик по chunk открывает существующий Evidence Panel (по `chunk_id`);
- если trace недоступен, показывается `trace_unavailable`, а не пустая таблица;
- если `context_chunk_ids` недоступен, явно указывается источник coverage;
- UI не вычисляет retrieval-метрики и не интерпретирует scores как confidence.

### 6.3. История теста

Для выбранного `test_id` показывается история по завершённым запускам. Если запуск
использовал `retrieval_mode=all`, история группируется по режиму, а не смешивает
`hybrid`/`dense`/`keyword` в одной точке:

- дата/run_id;
- retrieval mode;
- pass/fail;
- Recall@k;
- Evidence Coverage;
- critical errors;
- ссылка на детали.

Это позволяет увидеть, что тест сломался именно после изменения reranker.

---

## 7. Вкладка «Сравнение»

### 7.1. Выбор запусков

Два селектора:

- базовый запуск;
- сравниваемый запуск.

По умолчанию — предыдущий и последний завершённые запуски с тем же suite/split.

### 7.2. Сводная таблица

| Метрика | До | После | Δ |
|---|---:|---:|---:|
| Pass rate | 83% | 87% | +4 п.п. |
| Recall@10 | 0.82 | 0.91 | +0.09 |
| Evidence Coverage | 0.88 | 0.93 | +0.05 |
| Answer Quality | 8.1 | 8.2 | +0.1 |
| Citation Accuracy | 0.89 | 0.94 | +0.05 |
| Multi-turn | 80% | 80% | 0 |
| Mean latency | 12 с | 15 с | +3 с |

Правила:

- улучшения и регрессии показываются одинаково явно;
- ухудшение простых тестов при улучшении сложных видно отдельно;
- нет «лучшего» запуска — есть две версии и дельты;
- если метрика `null`, дельта не вычисляется.

### 7.3. Регрессии

Блок «Регрессии»:

- тесты, которые были `pass`, стали `fail`;
- новые critical errors;
- рост latency без улучшения качества;
- потеря Evidence Coverage;
- потеря multi-turn контекста.

Блок «Улучшения» — обратные изменения. Оба блока фильтруются по suite/difficulty.

### 7.4. Сопоставимость

Правила совпадают с `spec_benchmark.md` §8.3; UI не вводит своих условий.

Полное сравнение с дельтами разрешено, когда совпадают:

- `dataset.sha256`;
- набор тестов и split;
- `corpus.content_sha256` (`dataset.expected_corpus_sha256`).

Разрешено, но показывается как различие версий:

- разный git SHA / `dirty`;
- другая LLM/reranker/judge-модель;
- другой retrieval config (`mode`, `top_k`, `rerank_pool`, `graph`);
- другой `corpus.index_fingerprint` при том же content hash (переиндексация).

Запрещает дельты (`not_comparable`, только side-by-side):

- разный `dataset.sha256`;
- разный `corpus.content_sha256`;
- разный набор тестов или split.

Иначе:

- показывается баннер `not_comparable`;
- причины перечислены;
- side-by-side просмотр разрешён, но дельты и «регрессии» не выдаются как
  надёжный вывод;
- UI не скрывает различие версий и не называет изменением агента то, что изменило
  эталонную базу.

---

## 8. Evaluation API

### 8.1. Общие правила

- база: `/api/evaluation`;
- JSON, snake_case, ISO 8601 UTC, UUID;
- единый формат ошибок;
- пагинация для списков;
- evaluation endpoints не доступны обычному чату и не смешиваются с `/api/research`;
- API не запускает LLM в request handler;
- API не пересчитывает метрики: читает `summary.json`/`cases.jsonl`/trace;
- результат теста адресуется `(test_id, retrieval_mode, attempt_index)`; endpoint
  одного результата не угадывает режим;
- trace не встраивается в список; только по отдельному запросу;
- секреты, prompt и полный текст корпуса не возвращаются.

### 8.2. Endpoints

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/capabilities` | доступность Evaluation, режимы, judge, cancel, SSE (расширение существующего endpoint) |
| GET | `/api/evaluation/suites` | список suite, количество тестов, holdout |
| POST | `/api/evaluation/runs` | создать запуск, `202` + `run_id` |
| GET | `/api/evaluation/runs` | история запусков |
| GET | `/api/evaluation/runs/{id}` | статус, progress, summary |
| GET | `/api/evaluation/runs/{id}/tests` | результаты тестов |
| GET | `/api/evaluation/runs/{id}/tests/{test_id}` | детализация теста |
| GET | `/api/evaluation/runs/{id}/tests/{test_id}/trace` | полный trace |
| GET | `/api/evaluation/runs/{id}/report` | HTML/Markdown-отчёт (реализовано) |
| POST | `/api/evaluation/runs/{id}/cancel` | отмена (реализовано для queued/running) |
| GET | `/api/evaluation/runs/{id}/events` | SSE-прогресс — **не реализовано**, UI использует polling |
| GET | `/api/evaluation/tests` | каталог тестов |
| GET | `/api/evaluation/tests/{test_id}` | определение + история |
| GET | `/api/evaluation/compare` | сравнение двух запусков |

`GET /api/capabilities` расширяется evaluation-полями; существующие флаги чата
сохраняются:

```json
{
  "evaluation_ui": true,
  "evaluation_runs": true,
  "evaluation_sse": false,
  "evaluation_cancel": true,
  "retrieval_modes": ["hybrid", "dense", "keyword"],
  "judge": false,
  "max_repeat": 5,
  "max_queued": 3
}
```

### 8.3. Создание запуска

`POST /api/evaluation/runs`

```json
{
  "suite": "smoke",
  "split": "dev",
  "retrieval_mode": "hybrid",
  "judge": false,
  "repeat": 1,
  "notes": "После изменения reranker",
  "client_request_id": "uuid",
  "confirm_holdout": false
}
```

Ответ `202 Accepted`:

```json
{
  "id": "run_20261114_120000_f77652a",
  "status": "queued",
  "status_url": "/api/evaluation/runs/run_...",
  "created_at": "2026-11-14T12:00:00Z"
}
```

Правила:

- `suite` — только из whitelist; путь/файл не принимается;
- `smoke` принудительно `split=dev`; holdout в smoke запрещён;
- если `split` не задан: `smoke` → `dev`, `full` → `all`, остальные → `dev`;
- `full` с `split=all` или `split=holdout` требует `confirm_holdout=true`;
- `repeat` 1–5;
- `retrieval_mode` проверяется по capability;
- `judge=true` при отсутствии judge-модели → `422 judge_unavailable`;
- одновременно один running; новый запуск ставится в очередь или получает
  `429 queue_full`;
- повтор с тем же `client_request_id` возвращает существующий `run_id`;
- создание не запускает LLM в HTTP-запросе.

### 8.4. История запусков

`GET /api/evaluation/runs?limit=20&offset=0&suite=smoke&status=completed`

```json
{
  "items": [
    {
      "id": "run_...",
      "status": "completed",
      "suite": "smoke",
      "split": "dev",
      "retrieval_mode": "hybrid",
      "created_at": "2026-11-14T12:00:00Z",
      "finished_at": "2026-11-14T12:04:00Z",
      "pass_rate": 0.9,
      "execution_coverage": 1.0,
      "tests_total": 12,
      "tests_eligible": 10,
      "tests_executed": 12,
      "tests_passed": 9,
      "tests_skipped": 2,
      "duration_ms": 240000,
      "git_sha": "f77652a",
      "git_dirty": false,
      "dataset_sha256": "...",
      "corpus_content_sha256": "..."
    }
  ],
  "total": 3,
  "next_offset": null
}
```

- `limit` по умолчанию 20, максимум 100;
- сортировка по `created_at` убыванию;
- `pass_rate` — из summary, не пересчитывается;
- для `running` вместо `pass_rate` — progress.

### 8.5. Карточка запуска

`GET /api/evaluation/runs/{id}`

Возвращает `EvaluationRun` из §3.1. Для `running`:

```json
{
  "id": "run_...",
  "status": "running",
  "progress": {
    "tests_total": 12,
    "tests_completed": 5,
    "current_test_id": "medium_002",
    "current_turn": 1,
    "percent": 41.7
  },
  "summary": null
}
```

Для `completed` — полный `summary` из `spec_benchmark.md` §6.2.
`artifacts_url` указывает на `/api/evaluation/runs/{id}/report`, если endpoint
включён; иначе UI строит отчёт из `summary` и `tests` и не показывает битую ссылку.

### 8.6. Результаты тестов запуска

`GET /api/evaluation/runs/{id}/tests?status=failed&critical=true&limit=50&offset=0`

```json
{
  "items": [
    {
      "test_id": "hard_002",
      "retrieval_mode": "dense",
      "attempt_index": 0,
      "suite": "hard",
      "category": "temporal",
      "difficulty": "hard",
      "status": "completed",
      "pass": false,
      "metrics": {
        "recall_at_10": 0.75,
        "evidence_coverage": 0.75,
        "loss_stage": "never_retrieved"
      },
      "critical_errors": [
        {"code": "missing_required_year_claimed_complete"}
      ],
      "trace_available": true
    }
  ],
  "total": 4,
  "next_offset": null
}
```

Список не содержит полного trace, answer text и chunks.

### 8.7. Детализация теста

`GET /api/evaluation/runs/{id}/tests/{test_id}?retrieval_mode=hybrid&attempt_index=0`

`test_id` не уникален, если run выполнялся с `retrieval_mode=all`. Поэтому:

- `retrieval_mode` и `attempt_index` — явные query-параметры;
- `attempt_index` по умолчанию 0;
- если `retrieval_mode` не задан и у теста есть результаты в нескольких режимах,
  API возвращает `409 ambiguous_test_result` со списком `available_modes`;
- `retrieval_mode=all` в этом endpoint недопустим → `422 invalid_retrieval_mode`.

```json
{
  "test": {
    "id": "hard_002",
    "question": "Сравни чистую прибыль за 2020–2024 годы.",
    "difficulty": "hard",
    "category": "temporal",
    "expected_status": "answer",
    "expected": {},
    "gold_evidence": {}
  },
  "retrieval_mode": "dense",
  "attempt_index": 0,
  "result": {},
  "query_plan": {},
  "retrieval_calls": [],
  "evidence": {
    "manifest": [],
    "context_chunk_ids": [],
    "coverage": [],
    "loss_stage": "never_retrieved"
  },
  "answer": {},
  "evaluation": {
    "pass": false,
    "critical_errors": [],
    "rubric": null
  },
  "diagnostics": {
    "likely_cause": "retrieval",
    "notes": []
  },
  "trace_url": "/api/evaluation/runs/.../tests/hard_002/trace?retrieval_mode=dense&attempt_index=0"
}
```

Правила:

- `gold_evidence` и expected facts доступны только в Evaluation API, не в чате;
- если trace отсутствует, `trace_url=null` и `trace_available=false`;
- `diagnostics.likely_cause` — из evaluator, не вычисляется frontend;
- поля `query_plan`, `retrieval_calls`, `evidence` — flatten
  `AgentRunResult.diagnostics`; API не создаёт альтернативную диагностику;
- `answer` может содержать текст и citations; полный контекст — только trace endpoint.

### 8.8. Trace

`GET /api/evaluation/runs/{id}/tests/{test_id}/trace?retrieval_mode=hybrid&attempt_index=0`

Параметры и правило `ambiguous_test_result` те же, что в §8.7. Возвращает trace из
`spec_benchmark.md` §4.1. Ограничения:

- без prompt, chain-of-thought, ключей и полного текста корпуса;
- размер ограничен; при обрезке `truncated=true`;
- `Cache-Control: private, no-store`;
- неизвестный trace → 404.

`GET /runs/{id}/report` отдаёт уже сгенерированный `report.html`/`report.md` без
пересчёта. HTML отдаётся с CSP `sandbox`, без внешних ресурсов и скриптов; если
endpoint отключён, UI строит отчёт из `summary`/`tests`.

### 8.9. Сравнение

`GET /api/evaluation/compare?base={run_id}&candidate={run_id}`

```json
{
  "comparable": true,
  "reasons": [],
  "base": {
    "id": "...",
    "git_sha": "...",
    "dataset_sha256": "...",
    "corpus_content_sha256": "..."
  },
  "candidate": {
    "id": "...",
    "git_sha": "...",
    "dataset_sha256": "...",
    "corpus_content_sha256": "..."
  },
  "metrics": [
    {"key": "pass_rate", "base": 0.83, "candidate": 0.87, "delta": 0.04},
    {"key": "recall_at_10", "base": 0.82, "candidate": 0.91, "delta": 0.09}
  ],
  "regressions": [
    {"test_id": "easy_004", "from": "pass", "to": "fail", "reasons": ["invalid_citation"]}
  ],
  "improvements": [],
  "test_changes": []
}
```

- разные `dataset.sha256` или `corpus.content_sha256` → `comparable=false` + reasons;
  HTTP 200;
- разный набор тестов/split → `comparable=false`;
- разный git SHA, модель или retrieval config → `comparable=true`, различия в
  `reasons`/`versions`, дельты разрешены;
- отсутствующий run → 404;
- незавершённый run → 409 `run_not_finished`;
- метрики `null` не превращаются в ноль;
- дельты считает API, не frontend.

### 8.10. Каталог тестов

`GET /api/evaluation/tests?difficulty=hard&status=failed&q=прибыль&retrieval_mode=hybrid&limit=50`

```json
{
  "items": [
    {
      "id": "hard_002",
      "suite": "hard",
      "category": "temporal",
      "difficulty": "hard",
      "question": "Сравни чистую прибыль за 2020–2024 годы.",
      "last_run": {
        "run_id": "...",
        "retrieval_mode": "hybrid",
        "pass": false,
        "status": "completed",
        "critical_errors": []
      }
    }
  ],
  "total": 5,
  "next_offset": null
}
```

`GET /api/evaluation/tests/{test_id}` возвращает определение теста и историю
запусков (кратко).

### 8.11. Ошибки

| Код | HTTP | Значение |
|---|---|---|
| `evaluation_disabled` | 404 | раздел отключён конфигурацией |
| `run_not_found` | 404 | неизвестный run_id |
| `test_not_found` | 404 | неизвестный test_id |
| `ambiguous_test_result` | 409 | у test_id несколько режимов, `retrieval_mode` не указан |
| `invalid_retrieval_mode` | 422 | `retrieval_mode=all` в endpoint одного результата |
| `run_not_finished` | 409 | сравнение/детали недоступны до завершения |
| `run_in_progress` | 409 | очередь отключена и уже есть активный запуск |
| `queue_full` | 429 | очередь benchmark заполнена |
| `invalid_suite` | 422 | suite вне whitelist |
| `invalid_split` | 422 | недопустимый split |
| `invalid_repeat` | 422 | repeat вне 1–5 |
| `mode_not_supported` | 422 | режим не поддержан backend |
| `judge_unavailable` | 422 | судья недоступен |
| `holdout_confirmation_required` | 422 | нет `confirm_holdout` |
| `not_comparable` | 200 | сравнение с `comparable=false` |
| `cancel_not_supported` | 409 | отмена недоступна |
| `llm_busy` | 503 | другой процесс удерживает `LLMExecutionLock` |
| `llm_lock_unavailable` | 503 | файловая система не поддерживает эксклюзивный lock |
| `internal_error` | 500 | ошибка без traceback |

---

## 9. Фоновая работа и очередь

### 9.1. Worker

- Run creation только создаёт запись и ставит задачу.
- Worker — один фоновый исполнитель (`ThreadPoolExecutor(max_workers=1)` или
  эквивалент), без Celery/Redis.
- Worker вызывает `eval.runner.run_suite(..., llm_lock=LLMExecutionLock(...))`, а не
  копию benchmark.
- Между тестами и turn'ами worker обновляет progress в store.
- По завершении импортирует `summary.json`/`cases.jsonl` в БД и фиксирует
  `artifacts_dir`.
- Если артефакты не созданы — run `failed`, а не `completed` с нулями.

### 9.2. Общая очередь LLM

Локальная конфигурация не должна выполнять benchmark и пользовательский чат
одновременно.

Требования:

- все вызовы `AgentService.ask/stream` проходят через process-wide
  `ExecutionGate` в API adapter (`src/api/services.py`), а не через изменение
  `AgentService`;
- `max_concurrency = 1`;
- приоритет: `chat` > `evaluation`; wrapper принимает
  `priority="chat"|"evaluation"` и не меняет публичный `AgentService.ask`;
- benchmark удерживает gate на один turn, а не на весь suite;
- между turn'ами/тестами benchmark освобождает gate и делает короткий yield;
- chat-запрос может подождать максимум один benchmark-turn; UI показывает
  «Выполняется benchmark»;
- benchmark не запускает параллельные тесты;
- при появлении нескольких chat-запросов они обслуживаются в порядке очереди;
- wrapper делегирует в единственный singleton `AgentService`; вторая копия
  запрещена;
- метрика `llm_concurrency` в summary равна 1.

Если backend/provider поддерживает изолированные квоты, политику можно
пересмотреть отдельным решением; в первой версии — строгая сериализация.

#### Межпроцессная сериализация

`ExecutionGate` внутри API-процесса не защищает от CLI-runner'а: это два разных
процесса. Поэтому запрет одновременной работы должен быть техническим, а не
инструкцией оператору.

Механизм — `LLMExecutionLock`: эксклюзивный межпроцессный файловый lock
(`data/evaluation/llm.lock`, путь настраивается `LLM_LOCK_PATH`). Его берут **оба**
пути:

| Путь | Что удерживает lock |
|---|---|
| API chat | один `AgentService.ask()` / `stream()` запрос |
| API evaluation worker | один turn benchmark-теста |
| CLI `eval.runner` | один turn benchmark-теста |
| `scripts/eval_*`, вызывающие LLM | один вызов или один тест; тот же lock |

Правила:

- lock берётся перед первым LLM-вызовом turn'а и освобождается после завершения
  turn'а; benchmark не удерживает его на весь suite;
- reranker, router-LLM и judge входят в тот же turn и покрыты тем же lock'ом;
- между turn'ами benchmark делает yield (`EVAL_YIELD_MS`) и проверяет маркер
  `llm.chat_waiting`; если чат ждёт, benchmark приостанавливается;
- chat имеет приоритет и ждёт максимум один benchmark-turn;
- benchmark не запускает тесты параллельно;
- lock-файл содержит `pid`, `kind` (`chat|evaluation|cli`), `run_id`, `test_id`,
  `acquired_at`, `heartbeat_at`; секретов и prompt в нём нет;
- heartbeat обновляется, пока turn выполняется. Lock с мёртвым PID или устаревшим
  heartbeat (`LLM_LOCK_STALE_S`) считается брошенным и снимается с warning
  `stale_llm_lock_recovered`;
- если lock не взят за `LLM_LOCK_TIMEOUT_S`, CLI завершается кодом 5 (`llm_busy`),
  evaluation API возвращает `503 llm_busy`, chat ждёт;
- если файловая система не поддерживает эксклюзивные lock'и, запуск завершается
  `llm_lock_unavailable`; молчаливый параллельный вызов запрещён;
- API, который просто запущен и простаивает, не блокирует CLI: важен факт удержания
  lock'а, а не существование процесса.

Это не очередь и не планировщик: тот же `ThreadPoolExecutor(max_workers=1)` остаётся
достаточным внутри API.

### 9.3. Отмена

- `POST /runs/{id}/cancel`:
  - `queued` → сразу `cancelled`;
  - `running` → `cancel_requested`, runner завершает текущий turn и
    останавливается, после чего run становится `cancelled`;
  - `completed`/`failed`/`cancelled` → `409 cancel_not_supported`.
- Если worker не остановился за per-turn timeout, run становится `failed` с
  `error.code=cancel_timeout`; `cancel_requested` не остаётся навсегда.
- `cancel_requested` — не терминальный статус и не показывается как завершённый.
- Частичные результаты сохраняются.
- Отмена не удаляет уже записанные traces.
- UI показывает «Останавливается…» до терминального статуса.

### 9.4. Перезапуск API

При старте:

- все `queued`/`running`/`cancel_requested` записи помечаются `interrupted`;
- `interrupted` не выдаётся за `completed`;
- UI показывает причину и кнопку «Повторить»;
- артефакты, если есть, остаются доступными для просмотра.

### 9.5. Таймауты

- per-turn timeout;
- per-run timeout;
- превышение → `failed`/`timeout` с `test_id`;
- worker не зависает навсегда;
- timeout не превращается в `0` в метриках.

---

## 10. Хранилище

### 10.1. Разделение

Evaluation storage отделён от истории пользовательских диалогов:

```text
data/evaluation/
├── evaluation.sqlite3
└── llm.lock                # межпроцессный LLMExecutionLock

eval/results/{run_id}/          # канонические артефакты runner'а
├── summary.json
├── cases.jsonl
├── traces/
├── report.html
└── report.md
```

Полные traces/отчёты — на диске в `eval/results/` (каталог может быть
перенастроен через `EVAL_RESULTS_DIR`); SQLite хранит только индекс, статусы и
summary. Это не даёт benchmark-артефактам раздувать чат-БД.

### 10.2. Схема

```sql
CREATE TABLE evaluation_runs (
  id TEXT PRIMARY KEY,
  client_request_id TEXT UNIQUE,
  status TEXT NOT NULL,
  suite TEXT NOT NULL,
  split TEXT NOT NULL,
  retrieval_mode TEXT NOT NULL,
  judge INTEGER NOT NULL DEFAULT 0,
  repeat INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  progress_json TEXT,
  dataset_sha256 TEXT,
  corpus_content_sha256 TEXT,
  git_sha TEXT,
  config_json TEXT,
  summary_json TEXT,
  error_json TEXT,
  artifacts_dir TEXT
);

CREATE TABLE evaluation_test_results (
  run_id TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  test_id TEXT NOT NULL,
  retrieval_mode TEXT NOT NULL,
  attempt_index INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  pass INTEGER NOT NULL,
  metrics_json TEXT,
  critical_json TEXT,
  trace_path TEXT,
  duration_ms REAL,
  PRIMARY KEY (run_id, test_id, retrieval_mode, attempt_index)
);

CREATE INDEX ix_eval_runs_created
  ON evaluation_runs(created_at DESC);

CREATE INDEX ix_eval_results_failed
  ON evaluation_test_results(run_id, retrieval_mode, pass);
```

Правила:

- `attempt_index` относится к повтору конкретной пары `test_id + retrieval_mode`;
- `retrieval_mode` обязателен в ключе: при `retrieval_mode=all` один `test_id` даёт
  три строки, и они не должны конфликтовать;
- `trace_path` — относительный путь
  `traces/{test_id}.{retrieval_mode}[.attempt_n].json`;
- SQLite WAL, `PRAGMA foreign_keys=ON`, миграции при старте, без ORM;
- полный trace/HTML не дублируются в БД;
- удаление run каскадно удаляет результаты и (опционально) artifacts по retention;
- retention: `EVAL_RETENTION_RUNS` (по умолчанию 50), не удалять активный run;
- `summary_json` — только уже вычисленный summary, не пересчитывается API;
- БД не содержит API-ключей и prompt.

### 10.3. Импорт результатов

- После завершения worker читает `summary.json`/`cases.jsonl`.
- `cases.jsonl` → `evaluation_test_results`, ключ
  `(run_id, test_id, retrieval_mode, attempt_index)`. Повторный импорт обновляет
  строку, а не создаёт дубль.
- `summary.json` → `evaluation_runs.summary_json`, `dataset_sha256` и
  `corpus_content_sha256`.
- Ошибка импорта → `failed(artifact_import_error)`, artifacts сохраняются.

---

## 11. Версии и воспроизводимость

В карточке запуска и сравнении показываются:

| Поле | Источник |
|---|---|
| Git SHA + dirty | runner summary |
| Dataset version/hash | runner summary |
| Corpus content hash | runner summary (`dataset.expected_corpus_sha256`) |
| Corpus index fingerprint | runner summary (`corpus.index_fingerprint`) |
| Retrieval mode/top_k | runner summary |
| LLM/reranker/judge model | runner summary |
| Prompt hashes | runner summary |
| Python/deps | runner summary |

Требования:

- полное сравнение с дельтами — только при совпадении `dataset.sha256`, набора
  тестов, split и `corpus.content_sha256`; иначе `not_comparable` и side-by-side;
- разный git SHA не блокирует сравнение, но показывается как версия;
- другая модель или retrieval config не блокируют сравнение — это как раз то, что мы
  измеряем; различия показываются рядом с дельтами;
- разный `corpus.index_fingerprint` при том же content hash — переиндексация, а не
  другой корпус; это видно, но не запрещает дельты;
- UI не скрывает `dirty=true`;
- нельзя подменять версию «последней» без отображения.

---

## 12. Безопасность и доступ

- Evaluation — developer-инструмент, а не пользовательская функция.
- `EVAL_UI_ENABLED=1` включает раздел; при `0` — `evaluation_disabled`.
- В local-режиме доступен локальному владельцу.
- При включённой аутентификации требуется роль `evaluator`; обычный `user`
  получает 403/404 без раскрытия данных.
- Все endpoints проверяют доступ; нельзя открыть чужой run по ID.
- `suite` — whitelist, не путь; path traversal запрещён.
- Отдаются только артефакты из `artifacts_dir` текущего run.
- Trace/HTML санитизируются; Markdown — существующим безопасным рендерером.
- В API/UI/logs нет ключей, prompt, полного текста корпуса и chain-of-thought.
- Ограничения размера: список ≤ 100, trace ≤ 5 МБ, report ≤ разумного лимита.
- CSRF/CORS — как в `web-ui-modernization.md`.

### 12.1. Конфигурация

| Переменная | Default | Смысл |
|---|---|---|
| `EVAL_UI_ENABLED` | `0` | Включает раздел и endpoints |
| `EVAL_RESULTS_DIR` | `eval/results` | Каталог артефактов runner'а |
| `EVAL_DATASETS_DIR` | `eval/datasets` | Каталог suite-файлов датасета |
| `EVAL_DB_PATH` | `data/evaluation/evaluation.sqlite3` | Отдельная БД evaluation |
| `EVAL_MAX_CONCURRENCY` | `1` | В первой версии только 1 |
| `EVAL_QUEUE_SIZE` | `3` | Максимум ожидающих запусков |
| `EVAL_YIELD_MS` | `250` | Пауза между turn'ами/тестами для chat |
| `EVAL_JUDGE_ENABLED` | `0` | Включает LLM-судью (`capabilities.judge`) |
| `EVAL_FAKE_SERVICE` | `0` | Офлайн-запуск без LLM и сети |
| `LLM_LOCK_PATH` | `data/evaluation/llm.lock` | Общий межпроцессный LLM-lock (API + CLI) |
| `LLM_LOCK_TIMEOUT_S` | `900` | Сколько ждать lock перед `llm_busy` |
| `LLM_LOCK_STALE_S` | `900` | Возраст heartbeat, после которого lock считается брошенным |
| `EVAL_RETENTION_RUNS` | `50` | Сколько завершённых run'ов хранить |
| `EVAL_TURN_TIMEOUT_S` | `300` | Таймаут одного turn'а |
| `EVAL_RUN_TIMEOUT_S` | `7200` | Таймаут всего запуска |

`EVAL_ALLOWED_ROLES` в текущем коде не читается: реализован только
`AUTH_MODE=local` с одним владельцем. Роли `evaluator`/`user` и
соответствующая авторизация остаются целью при появлении многопользовательского
режима; сейчас изоляция обеспечивается `owner_id` и cookie владельца.

Секреты и модель не задаются во frontend; они читаются backend-конфигурацией.

---

## 13. Первая версия

| Функция | Первая версия |
|---|---|
| Smoke / Full запуск | Да |
| Запуск отдельной категории | Да |
| История запусков | Да |
| Сводный отчёт | Да |
| Детализация теста и Retrieval Trace | Да |
| Сравнение двух запусков | Да |
| График динамики качества | Да |
| Отмена запуска | Да, если worker поддерживает |
| SSE-прогресс | Опционально; polling обязателен |
| Редактор benchmark-вопросов | Позже |
| Автоматическое построение тестов | Не нужно |
| Сложная аналитика и кастомные дашборды | Не нужно |
| Планировщик регулярных запусков | Позже |
| Многопользовательские роли evaluation | Позже, кроме базового `evaluator` |

---

## 14. Критерии приёмки

### Навигация и обзор

- [x] **AC-E01.** Evaluation — отдельный route, не вкладка чата; переход не теряет
  выбранное исследование.
- [x] **AC-E02.** Раздел содержит четыре вкладки: Обзор, Запуски, Тесты, Сравнение.
- [x] **AC-E03.** При `evaluation_ui=false` пункт скрыт; прямой route показывает
  `evaluation_disabled`, а не пустой экран.
- [x] **AC-E04.** Обзор показывает независимые метрики последнего завершённого
  запуска; неизмеренная метрика — `—`, не `0`.
- [x] **AC-E05.** Дельта к предыдущему запуску показывается только при наличии
  предыдущего; значения не вычисляются из сырых traces во frontend.
- [x] **AC-E06.** График динамики использует только реальные запуски; пропуски
  не превращаются в ноль.

### Запуски

- [x] **AC-E07.** `POST /runs` возвращает `202` и `run_id`; benchmark не
  выполняется в HTTP-запросе.
- [x] **AC-E08.** UI показывает `queued → running → cancel_requested →
  completed/failed/cancelled` и progress без вечного spinner; `interrupted` виден
  после рестарта API.
- [x] **AC-E09.** Smoke и Full запускаются; holdout требует явного подтверждения;
  smoke не включает holdout.
- [x] **AC-E10.** Повторный запуск при активном benchmark не создаёт второй
  параллельный run; очередь ограничена и сообщает `queue_full`.
- [x] **AC-E11.** После завершения доступны summary, tests, report; при ошибке
  показывается код и `run_id`.
- [x] **AC-E12.** Отмена (если включена) останавливает runner между turn'ами и
  сохраняет частичные результаты.

### Тесты

- [x] **AC-E13.** Каталог тестов фильтруется по difficulty/category/status/
  critical; показывает последний статус.
- [x] **AC-E14.** Детализация теста показывает query plan, retrieval-этапы,
  evidence coverage, ответ, critical errors и диагностику.
- [x] **AC-E15.** Пропущенный год/фрагмент отображается как пропуск, а не ноль;
  `not_found` и `year_absent` различаются.
- [x] **AC-E16.** Trace доступен отдельным запросом; в нём нет prompt,
  chain-of-thought, ключей и полного текста корпуса.
- [~] **AC-E17.** При отсутствии trace показывается `trace_unavailable`; клик по
  chunk в Evaluation **пока не** открывает Evidence Panel чата (chunk_id
  отображается, но перехода нет).

### Сравнение

- [x] **AC-E18.** Сравнение двух запусков показывает метрики, дельты, улучшения
  и регрессии; ухудшения не скрываются.
- [x] **AC-E19.** Разные `dataset.sha256` или `corpus.content_sha256` дают
  `comparable=false` с причинами; side-by-side разрешён, но дельты не выдаются как
  надёжные. Разный git SHA/модель/config дельты не блокирует.
- [x] **AC-E20.** Изменения статусов тестов (`pass→fail`, `fail→pass`) видны
  списком и фильтруются.

### Архитектура и безопасность

- [x] **AC-E21.** Runner вызывает тот же `AgentService`; второй копии агента нет.
- [x] **AC-E22.** Все вызовы LLM сериализованы через `ExecutionGate` внутри API и
  `LLMExecutionLock` между API и CLI; benchmark и чат не конкурируют; benchmark
  удерживает lock не дольше одного turn.
- [x] **AC-E23.** Evaluation storage отделён от истории чата; полные traces
  лежат на диске, БД хранит индекс/summary.
- [x] **AC-E24.** При перезапуске API `running` → `interrupted`; UI не показывает
  завершённый результат.
- [x] **AC-E25.** Версии (git SHA/dirty, dataset hash, corpus content hash, corpus
  fingerprint, model, prompt hashes) видны в карточке и сравнении.
- [~] **AC-E26.** Evaluation endpoints защищены владельцем (`AUTH_MODE=local`,
  cookie): чужой run недоступен. Роль `evaluator` и многопользовательская
  авторизация не реализованы.
- [x] **AC-E27.** В API/UI/логах нет секретов, prompt и полного текста корпуса.
- [x] **AC-E28.** Frontend не запускает retrieval/LLM, не пересчитывает метрики и
  не меняет gold evidence.
- [x] **AC-E29.** Empty/error/loading состояния работают для всех четырёх вкладок.
- [x] **AC-E30.** Offline contract/component/E2E-тесты проходят на fixtures без
  сети и LLM.

### Контракты и согласованность

- [x] **AC-E31.** Результат теста адресуется `(run_id, test_id, retrieval_mode,
  attempt_index)`; при `retrieval_mode=all` три режима дают три строки БД и три
  trace-файла без перезаписи.
- [x] **AC-E32.** `EvaluationRunStatus` включает `cancel_requested`; переходы
  `running → cancel_requested → cancelled` и `queued/running → interrupted`
  реализованы; `abstained` не отображается как ошибка.
- [x] **AC-E33.** CLI и API используют один `LLMExecutionLock`; параллельный вызов
  LLM из двух процессов технически невозможен, а не запрещён инструкцией.
- [x] **AC-E34.** Сопоставимость в UI и API совпадает с `spec_benchmark.md` §8.3;
  разный content hash корпуса даёт `not_comparable`.
- [x] **AC-E35.** Pass rate показывается вместе с execution coverage; eligible,
  selected, executed, skipped и errors не смешиваются.
- [x] **AC-E36.** Endpoint детализации и trace требуют `retrieval_mode` при
  неоднозначности и возвращают `ambiguous_test_result`, а не первый попавшийся режим.

---

## 15. Тестовая матрица

| Уровень | Что проверяется | Где | Блокирует merge |
|---|---|---|---|
| API contract | POST/GET схемы, коды, пагинация, idempotency | `tests/test_web_api_evaluation.py` (20 тестов) | да |
| Store | runs/results, WAL, cascade, interrupted, retention, составной ключ с режимом | `tests/test_web_api_evaluation.py`, `tests/test_web_api_store.py` | да |
| Worker | queue, один running, progress, cancel, artifact import | `tests/test_web_api_evaluation.py` | да |
| Gate | chat priority, один LLM-call, yield между turn'ами | `tests/test_api_execution.py` | да |
| Cross-process lock | CLI + API не вызывают LLM одновременно, stale lock | `tests/test_llm_lock.py`, `tests/test_api_execution.py` | да |
| Result identity | режимы, attempts, trace paths, `ambiguous_test_result` | `tests/test_web_api_evaluation.py` | да |
| Frontend unit | cards, statuses, filters, compare, empty states | `frontend/tests/unit/evaluation/*` | да |
| Frontend contract | типы ответов, `null`/`—`, not_comparable | `frontend/tests/unit/evaluation/contracts.test.ts` | да |
| E2E | запуск smoke → прогресс → summary → тест → trace → сравнение | `frontend/tests/e2e/evaluation.spec.ts` | да |
| Live smoke | реальный `eval.runner --suite smoke` через API | manual/nightly | нет |

Все unit/integration-тесты используют fake runner/store; сеть и LLM не нужны.
E2E — против fixture-backend с реальными API-контрактами.

---

## 16. Этапы реализации

### Этап 0. Зависимости

- Готов `eval.runner` и dataset из `spec_benchmark.md`.
- Готов базовый React WebUI и FastAPI adapter из `web-ui-modernization.md`
  (этап 1).
- Добавлен `run_suite(...)` как программный API runner'а; CLI — тонкая обёртка.
- Определены capabilities evaluation.

### Этап 1. Evaluation API и worker

- `src/api/routes/evaluation.py`;
- `src/api/evaluation_store.py`;
- `src/api/evaluation_service.py`;
- `src/api/evaluation_schemas.py`;
- отдельная SQLite;
- очередь/worker, progress, cancel, interrupted;
- `ExecutionGate`;
- contract/store/worker тесты.

Критерий: `POST /runs` запускает smoke в фоне, `GET /runs/{id}` показывает
progress и summary, artifacts импортируются.

### Этап 2. Обзор и Запуски

- верхняя навигация Evaluation;
- карточки, trend chart, problem list;
- форма запуска;
- история, статусы, карточка запуска;
- polling;
- empty/error states.

Критерий: пользователь запускает smoke/full и видит результат без перезагрузки.

### Этап 3. Тесты и trace

- каталог тестов;
- фильтры;
- детализация;
- retrieval stage table;
- evidence coverage;
- critical errors и diagnostics;
- trace endpoint и viewer;
- переход к Evidence Panel.

Критерий: по проваленному тесту понятно, на каком этапе потеряно доказательство.

### Этап 4. Сравнение

- выбор двух запусков;
- метрики/дельты;
- регрессии/улучшения;
- test changes;
- not_comparable;
- график динамики.

Критерий: после изменения reranker видно, что улучшилось и что сломалось.

### Этап 5. Полировка и безопасность

- отмена, SSE (опционально);
- лимиты и retention;
- роли/доступ;
- E2E;
- документация;
- проверка больших traces и списков.

---

## 17. Риски и меры

| Риск | Проявление | Мера |
|---|---|---|
| Full-прогон долгий | UI висит, пользователь думает, что сломалось | background worker, progress, polling, timeout |
| Benchmark конкурирует с чатом | ответы чата замедляются | `ExecutionGate`, `LLMExecutionLock`, приоритет chat, один turn |
| CLI и API одновременно | два процесса вызывают LLM | общий файловый lock, `llm_busy`, stale recovery |
| Режим `all` схлопывает результаты | потеря двух из трёх прогонов | `retrieval_mode` в ключе и имени trace |
| Dataset/корпус дрейфует | сравнение врёт | dataset hash + content hash, `not_comparable` |
| Очередь переполнена | новые запуски не создаются | `queue_full`, лимит, отмена |
| Trace слишком большой | медленный API/UI | отдельный endpoint, лимит, pagination |
| Артефакты потеряны | summary есть, traces нет | `trace_unavailable`, failed import |
| API перезапущен | run завис | `interrupted`, кнопка повтора |
| SQLite locked | ошибки записи | WAL, один worker, короткие транзакции |
| Чужой run | утечка данных | owner/role check, 404 |
| Секреты в trace | утечка | redaction, contract tests |
| Frontend считает метрики | расхождения | API отдаёт готовые значения |
| Автообновление эталонов | самоподтверждение | редактор только позже, отдельный review |

---

## 18. Definition of Done

- [x] Evaluation — отдельный route с четырьмя вкладками.
- [x] Smoke/Full запускаются через API в фоне; HTTP не блокируется.
- [x] История запусков и summary сохраняются после перезапуска API.
- [x] Детализация теста показывает query plan, retrieval, evidence, answer,
  critical errors и diagnostics.
- [x] Trace доступен отдельно и не содержит prompt/секретов/полного корпуса.
- [x] Сравнение показывает дельты, улучшения и регрессии; разные datasets —
  `not_comparable`.
- [x] График динамики строится по реальным запускам без нулей на пропусках.
- [x] Benchmark использует тот же `AgentService`; второй копии нет.
- [x] LLM-вызовы сериализованы; benchmark не конкурирует с чатом ни внутри API, ни с
  CLI-runner'ом.
- [x] Результаты различаются по `retrieval_mode` и `attempt_index`; trace-файлы не
  перезаписываются при `retrieval_mode=all`.
- [x] Статусы Evaluation run включают `cancel_requested`; `abstained` не выдаётся за
  ошибку.
- [x] Pass rate показывается вместе с execution coverage.
- [x] Evaluation storage отделён от чата.
- [x] Версии и dataset hash видны в UI.
- [x] Empty/error/loading состояния работают.
- [x] Offline contract/component/E2E-тесты проходят.
- [x] README и связанные specs обновлены ссылками.

Не входит в текущую приёмку: переход из chunk Evaluation в Evidence Panel чата,
SSE-прогресс Evaluation (`evaluation_sse=false`), UI-выбор `attempt > 0` и
многопользовательские роли (`AUTH_MODE=local`).

---

## 19. Что не входит в объём

- Запуск benchmark из браузера без backend runner.
- Вторая реализация `AgentService`/retrieval/LLM.
- Benchmark внутри обычного chat endpoint.
- Хранение benchmark-результатов в истории пользовательских диалогов.
- Редактор эталонов и gold evidence в первой версии.
- Автоматическая генерация тестов.
- Автоматическое изменение expected/gold после прогона.
- Сложные кастомные дашборды и BI.
- Планировщик регулярных запусков (cron) в первой версии.
- Многопользовательские роли сверх базового `evaluator`.
- Пересчёт метрик во frontend.
- Показ скрытых рассуждений модели, prompt или полного контекста.
