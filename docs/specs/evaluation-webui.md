# Evaluation / Benchmark WebUI — Agent_RAG

**Статус:** Proposed
**Версия:** 1.0
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
| `traces/{test_id}[.attempt_n].json` | Детализация теста, retrieval trace |
| `retrieval.csv` | Таблица этапов и loss attribution |
| `report.html` / `report.md` | Отчёт запуска (опционально) |
| `dataset/*.json` | Определения тестов и gold evidence |

UI не создаёт эти артефакты и не пересчитывает их метрики.

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
  attempt_index: number;
  suite: string;
  category: string;
  difficulty: "easy" | "medium" | "hard";
  status: "completed" | "failed" | "timeout" | "error" | "skipped";
  pass: boolean;
  retrieval_mode: string;
  metrics: {
    recall_at_5?: number | null;
    recall_at_10?: number | null;
    mrr?: number | null;
    retrieval_coverage?: number | null;
    evidence_coverage?: number | null;
    context_coverage?: number | null;
    loss_stage?: string | null;
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
  trace_available: boolean;
}
```

### 3.3. Переходы статусов

```text
queued → running → completed
              ├→ failed
              ├→ cancelled
              └→ interrupted
queued → cancelled
queued → interrupted
```

Правила:

- `completed` не означает, что все тесты прошли; это статус выполнения.
- `failed` — ошибка runner/окружения, а не провал теста.
- `cancelled` — отменено пользователем; частичные результаты сохраняются.
- `interrupted` — API перезапущен во время выполнения; UI не показывает вечный
  spinner и предлагает повторить.
- Переходы назад запрещены.

---

## 4. Вкладка «Обзор»

### 4.1. Карточки

Показываются четыре независимые метрики последнего завершённого запуска:

| Карточка | Значение | Источник |
|---|---|---|
| Успешные тесты | `passed / total` и pass rate | `summary.pass_at_1` / `summary.coverage` |
| Recall@10 | macro mean | `summary.groups.retrieval.recall_at_10` |
| Качество ответов | `8.2 / 10` rubric mean, если есть judge/manual; иначе automatic pass rate с явной подписью | `summary.groups.answer.rubric_mean` или `automatic_pass_rate` |
| Multi-turn | scenario success rate | `summary.groups.dialogue.scenario_success_rate` |

Требования:

- не показывать `0`, если метрика не измерялась; показывать `—`;
- рядом с каждой карточкой — дельта к предыдущему завершённому запуску с тем же
  suite/split: `▲ +2.1 п.п.`, `▼ −1.4 п.п.` или `—`;
- если предыдущего сопоставимого запуска нет, дельта отсутствует, а не сравнивается
  с другим набором;
- не сворачивать четыре карточки в один `Agent Quality`;
- значения не вычисляются во frontend из сырых traces.

### 4.2. Последние запуски

Таблица:

| Дата | Набор | Режим | Результат | Длительность | Статус |
|---|---|---|---|---|---|
| 14.11 | Smoke | hybrid | 90% | 4 мин | completed |
| 12.11 | Full | hybrid | 87% | 38 мин | completed |
| 09.11 | Full | dense | 83% | 35 мин | completed |

Правила:

- сортировка по `created_at` убыванию;
- статус `running` показывает progress;
- `failed`/`interrupted` не маскируются процентом;
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
- версии: git SHA + dirty, dataset hash/version, corpus fingerprint, model;
- сводку четырёх групп;
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
- опрашивает `GET /runs/{id}` каждые 2 секунды, пока статус `queued`/`running`;
- при ошибке polling — backoff и честный статус;
- после терминального статуса загружает summary и tests;
- SSE-обновления — опциональная capability, не обязательны в первой версии.

---

## 6. Вкладка «Тесты»

### 6.1. Каталог

Источник — `eval/datasets/*.json`; последний статус — из последнего завершённого
запуска.

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

1. **Заголовок** — test_id, статус, difficulty, category, suite, запуск, attempt.
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

Для выбранного test_id показывается история по завершённым запускам:

- дата/run_id;
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

Сравнение допустимо, если:

- совпадает `dataset.sha256`;
- совпадает набор тестов и split;
- известны версии/конфиги обоих запусков.

Иначе:

- показывается баннер `not_comparable`;
- причины перечислены;
- side-by-side просмотр разрешён, но дельты и «регрессии» не выдаются как
  надёжный вывод;
- UI не скрывает различие версий.

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
| GET | `/api/evaluation/runs/{id}/report` | HTML/Markdown-отчёт (опционально) |
| POST | `/api/evaluation/runs/{id}/cancel` | отмена (опционально) |
| GET | `/api/evaluation/runs/{id}/events` | SSE-прогресс (опционально) |
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
      "tests_total": 12,
      "tests_passed": 11,
      "duration_ms": 240000,
      "git_sha": "f77652a",
      "git_dirty": false,
      "dataset_sha256": "..."
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

`GET /api/evaluation/runs/{id}/tests/{test_id}`

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
  "trace_url": "/api/evaluation/runs/.../tests/hard_002/trace"
}
```

Правила:

- `gold_evidence` и expected facts доступны только в Evaluation API, не в чате;
- если trace отсутствует, `trace_url=null` и `trace_available=false`;
- `diagnostics.likely_cause` — из evaluator, не вычисляется frontend;
- `answer` может содержать текст и citations; полный контекст — только trace endpoint.

### 8.8. Trace

`GET /api/evaluation/runs/{id}/tests/{test_id}/trace`

Возвращает trace из `spec_benchmark.md` §4.1. Ограничения:

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
  "base": {"id": "...", "git_sha": "...", "dataset_sha256": "..."},
  "candidate": {"id": "...", "git_sha": "...", "dataset_sha256": "..."},
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

- разные `dataset.sha256` → `comparable=false` + reasons; HTTP 200;
- отсутствующий run → 404;
- незавершённый run → 409 `run_not_finished`;
- метрики `null` не превращаются в ноль;
- дельты считает API, не frontend.

### 8.10. Каталог тестов

`GET /api/evaluation/tests?difficulty=hard&status=failed&q=прибыль&limit=50`

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
| `internal_error` | 500 | ошибка без traceback |

---

## 9. Фоновая работа и очередь

### 9.1. Worker

- Run creation только создаёт запись и ставит задачу.
- Worker — один фоновый исполнитель (`ThreadPoolExecutor(max_workers=1)` или
  эквивалент), без Celery/Redis.
- Worker вызывает `eval.runner.run_suite(...)`, а не копию benchmark.
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

CLI-запуск `eval.runner` не проходит через API-очередь. В local-режиме нельзя
одновременно запускать CLI benchmark и активный чат через API; оператор обязан
использовать один путь. UI-запуск всегда идёт через общую очередь.

### 9.3. Отмена

- `POST /runs/{id}/cancel`:
  - `queued` → сразу `cancelled`;
  - `running` → `cancel_requested`, runner завершает текущий turn и
    останавливается;
  - `completed`/`failed` → `409 cancel_not_supported`.
- Частичные результаты сохраняются.
- Отмена не удаляет уже записанные traces.
- UI показывает «Останавливается…» до терминального статуса.

### 9.4. Перезапуск API

При старте:

- все `queued`/`running` записи помечаются `interrupted`;
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
└── evaluation.sqlite3

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
  git_sha TEXT,
  config_json TEXT,
  summary_json TEXT,
  error_json TEXT,
  artifacts_dir TEXT
);

CREATE TABLE evaluation_test_results (
  run_id TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
  test_id TEXT NOT NULL,
  attempt_index INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  pass INTEGER NOT NULL,
  metrics_json TEXT,
  critical_json TEXT,
  trace_path TEXT,
  duration_ms REAL,
  PRIMARY KEY (run_id, test_id, attempt_index)
);

CREATE INDEX ix_eval_runs_created
  ON evaluation_runs(created_at DESC);

CREATE INDEX ix_eval_results_failed
  ON evaluation_test_results(run_id, pass);
```

Правила:

- SQLite WAL, `PRAGMA foreign_keys=ON`, миграции при старте, без ORM;
- полный trace/HTML не дублируются в БД;
- удаление run каскадно удаляет результаты и (опционально) artifacts по retention;
- retention: `EVAL_RETENTION_RUNS` (по умолчанию 50), не удалять активный run;
- `summary_json` — только уже вычисленный summary, не пересчитывается API;
- БД не содержит API-ключей и prompt.

### 10.3. Импорт результатов

- После завершения worker читает `summary.json`/`cases.jsonl`.
- `cases.jsonl` → `evaluation_test_results`.
- `summary.json` → `evaluation_runs.summary_json`.
- Ошибка импорта → `failed(artifact_import_error)`, artifacts сохраняются.

---

## 11. Версии и воспроизводимость

В карточке запуска и сравнении показываются:

| Поле | Источник |
|---|---|
| Git SHA + dirty | runner summary |
| Dataset version/hash | runner summary |
| Corpus fingerprint | runner summary |
| Retrieval mode/top_k | runner summary |
| LLM/reranker/judge model | runner summary |
| Prompt hashes | runner summary |
| Python/deps | runner summary |

Требования:

- сравнение разных dataset hash → `not_comparable`;
- разный git SHA не блокирует сравнение, но показывается как версия;
- разный corpus fingerprint — отдельное предупреждение;
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
| `EVAL_DB_PATH` | `data/evaluation/evaluation.sqlite3` | Отдельная БД evaluation |
| `EVAL_MAX_CONCURRENCY` | `1` | В первой версии только 1 |
| `EVAL_QUEUE_SIZE` | `3` | Максимум ожидающих запусков |
| `EVAL_YIELD_MS` | `250` | Пауза между turn'ами/тестами для chat |
| `EVAL_RETENTION_RUNS` | `50` | Сколько завершённых run'ов хранить |
| `EVAL_TURN_TIMEOUT_S` | `300` | Таймаут одного turn'а |
| `EVAL_RUN_TIMEOUT_S` | `7200` | Таймаут всего запуска |
| `EVAL_ALLOWED_ROLES` | `evaluator` | Роль при включённой аутентификации |

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

- [ ] **AC-E01.** Evaluation — отдельный route, не вкладка чата; переход не теряет
  выбранное исследование.
- [ ] **AC-E02.** Раздел содержит четыре вкладки: Обзор, Запуски, Тесты, Сравнение.
- [ ] **AC-E03.** При `evaluation_ui=false` пункт скрыт; прямой route показывает
  `evaluation_disabled`, а не пустой экран.
- [ ] **AC-E04.** Обзор показывает четыре независимые метрики последнего
  завершённого запуска; неизмеренная метрика — `—`, не `0`.
- [ ] **AC-E05.** Дельта к предыдущему запуску показывается только при наличии
  предыдущего; значения не вычисляются из сырых traces во frontend.
- [ ] **AC-E06.** График динамики использует только реальные запуски; пропуски
  не превращаются в ноль.

### Запуски

- [ ] **AC-E07.** `POST /runs` возвращает `202` и `run_id`; benchmark не
  выполняется в HTTP-запросе.
- [ ] **AC-E08.** UI показывает `queued → running → completed/failed/cancelled`
  и progress без вечного spinner.
- [ ] **AC-E09.** Smoke и Full запускаются; holdout требует явного подтверждения;
  smoke не включает holdout.
- [ ] **AC-E10.** Повторный запуск при активном benchmark не создаёт второй
  параллельный run; очередь ограничена и сообщает `queue_full`.
- [ ] **AC-E11.** После завершения доступны summary, tests, report; при ошибке
  показывается код и `run_id`.
- [ ] **AC-E12.** Отмена (если включена) останавливает runner между turn'ами и
  сохраняет частичные результаты.

### Тесты

- [ ] **AC-E13.** Каталог тестов фильтруется по difficulty/category/status/
  critical; показывает последний статус.
- [ ] **AC-E14.** Детализация теста показывает query plan, retrieval-этапы,
  evidence coverage, ответ, critical errors и диагностику.
- [ ] **AC-E15.** Пропущенный год/фрагмент отображается как пропуск, а не ноль;
  `not_found` и `year_absent` различаются.
- [ ] **AC-E16.** Trace доступен отдельным запросом; в нём нет prompt,
  chain-of-thought, ключей и полного текста корпуса.
- [ ] **AC-E17.** Клик по chunk открывает существующий Evidence Panel по
  `chunk_id`; при отсутствии trace показывается `trace_unavailable`.

### Сравнение

- [ ] **AC-E18.** Сравнение двух запусков показывает метрики, дельты, улучшения
  и регрессии; ухудшения не скрываются.
- [ ] **AC-E19.** Разные `dataset.sha256` дают `comparable=false` с причинами;
  side-by-side разрешён, но дельты не выдаются как надёжные.
- [ ] **AC-E20.** Изменения статусов тестов (`pass→fail`, `fail→pass`) видны
  списком и фильтруются.

### Архитектура и безопасность

- [ ] **AC-E21.** Runner вызывает тот же `AgentService`; второй копии агента нет.
- [ ] **AC-E22.** Все вызовы LLM сериализованы через `ExecutionGate`; benchmark
  и чат не конкурируют; benchmark удерживает gate не дольше одного turn.
- [ ] **AC-E23.** Evaluation storage отделён от истории чата; полные traces
  лежат на диске, БД хранит индекс/summary.
- [ ] **AC-E24.** При перезапуске API `running` → `interrupted`; UI не показывает
  завершённый результат.
- [ ] **AC-E25.** Версии (git SHA/dirty, dataset hash, corpus fingerprint, model,
  prompt hashes) видны в карточке и сравнении.
- [ ] **AC-E26.** Evaluation endpoints защищены: local owner или роль `evaluator`;
  чужой run недоступен.
- [ ] **AC-E27.** В API/UI/логах нет секретов, prompt и полного текста корпуса.
- [ ] **AC-E28.** Frontend не запускает retrieval/LLM, не пересчитывает метрики и
  не меняет gold evidence.
- [ ] **AC-E29.** Empty/error/loading состояния работают для всех четырёх вкладок.
- [ ] **AC-E30.** Offline contract/component/E2E-тесты проходят на fixtures без
  сети и LLM.

---

## 15. Тестовая матрица

| Уровень | Что проверяется | Где | Блокирует merge |
|---|---|---|---|
| API contract | POST/GET схемы, коды, пагинация, idempotency | `tests/api/test_evaluation_api.py` | да |
| Store | runs/results, WAL, cascade, interrupted, retention | `tests/api/test_evaluation_store.py` | да |
| Worker | queue, один running, progress, cancel, artifact import | `tests/api/test_evaluation_worker.py` | да |
| Gate | chat priority, один LLM-call, yield между turn'ами | `tests/api/test_execution_gate.py` | да |
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
| Benchmark конкурирует с чатом | ответы чата замедляются | `ExecutionGate`, приоритет chat, один turn |
| Очередь переполнена | новые запуски не создаются | `queue_full`, лимит, отмена |
| Trace слишком большой | медленный API/UI | отдельный endpoint, лимит, pagination |
| Артефакты потеряны | summary есть, traces нет | `trace_unavailable`, failed import |
| Dataset дрейфует | сравнение врёт | dataset hash, `not_comparable` |
| API перезапущен | run завис | `interrupted`, кнопка повтора |
| SQLite locked | ошибки записи | WAL, один worker, короткие транзакции |
| Чужой run | утечка данных | owner/role check, 404 |
| Секреты в trace | утечка | redaction, contract tests |
| Frontend считает метрики | расхождения | API отдаёт готовые значения |
| Автообновление эталонов | самоподтверждение | редактор только позже, отдельный review |

---

## 18. Definition of Done

- [ ] Evaluation — отдельный route с четырьмя вкладками.
- [ ] Smoke/Full запускаются через API в фоне; HTTP не блокируется.
- [ ] История запусков и summary сохраняются после перезапуска API.
- [ ] Детализация теста показывает query plan, retrieval, evidence, answer,
  critical errors и diagnostics.
- [ ] Trace доступен отдельно и не содержит prompt/секретов/полного корпуса.
- [ ] Сравнение показывает дельты, улучшения и регрессии; разные datasets —
  `not_comparable`.
- [ ] График динамики строится по реальным запускам без нулей на пропусках.
- [ ] Benchmark использует тот же `AgentService`; второй копии нет.
- [ ] LLM-вызовы сериализованы; benchmark не конкурирует с чатом.
- [ ] Evaluation storage отделён от чата.
- [ ] Версии и dataset hash видны в UI.
- [ ] Empty/error/loading состояния работают.
- [ ] Offline contract/component/E2E-тесты проходят.
- [ ] README и связанные specs обновлены ссылками.

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
