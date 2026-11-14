# Спецификация Benchmark / Evaluation Framework Agent_RAG

**Версия:** 1.0
**Статус:** целевой контракт; benchmark-контур ещё не реализован.
**База анализа:** `main` (`f77652a`) + рабочее дерево: `AgentService`, `AgentTools`,
`HybridRetriever`, `RetrievalTrace`, `scripts/eval_retrieval.py`.
**Связанные документы:** [spec_target.md](spec_target.md), [spec_agent.md](spec_agent.md),
[spec_retrieval_engine.md](spec_retrieval_engine.md), [spec_graph.md](spec_graph.md),
[spec_ui_eval.md](spec_ui_eval.md), [spec_agent_enhancement.md](spec_agent_enhancement.md),
[docs/specs/evaluation-webui.md](docs/specs/evaluation-webui.md),
[docs/specs/web-ui-modernization.md](docs/specs/web-ui-modernization.md)

> Этот документ уточняет `spec_ui_eval.md` §2–§5 для agent-level benchmark. Где форматы
> расходятся, приоритет у этого документа; UI-контракт §1 и общие release-метрики §3
> сохраняются. Benchmark не заменяет `scripts/eval_retrieval.py`: скрипт остаётся
> компонентным retrieval-пробником, новый контур оценивает всю цепочку агента.

## 1. Назначение и границы

### 1.1. Цель

Добавить в проект небольшой воспроизводимый контур оценки, который запускает реальный
`AgentService` на эталонном наборе и позволяет для каждого ответа восстановить цепочку:

```text
Question → Query Plan → Search Queries → Retrieved Chunks → Evidence → Answer
```

Проверяется не только итоговый текст, но и вся работа агента:

1. правильно ли понят вопрос;
2. правильно ли выбрана стратегия поиска;
3. найдены ли нужные документы и фрагменты;
4. достаточно ли собрано доказательств;
5. правильно ли сформирован ответ;
6. не выдумана ли информация;
7. сохраняется ли контекст в уточняющем диалоге.

Главное требование: **у каждого теста есть проверяемый эталон** — ожидаемые факты,
годы, документы, evidence-фрагменты и допустимое поведение. Коллекция вопросов без
эталона benchmark не является.

### 1.2. Принципы

| № | Принцип | Следствие |
|---|---|---|
| 1 | Benchmark наблюдает, а не управляет агентом | Рабочая логика `AgentService` не меняется ради оценки |
| 2 | Эталон отделён от ответа | Gold evidence и ожидаемые факты не передаются агенту |
| 3 | Сначала детерминированные проверки, потом LLM-судья | Числа, годы, цитаты и расчёты проверяет Python |
| 4 | Ошибка не превращается в ноль | Сбой, таймаут или неподдержанный режим видны как `error`/`skipped` |
| 5 | Четыре независимые группы оценок | Нет одного «Agent Quality = 87%», скрывающего причину |
| 6 | Эталон не обновляется по ответам агента | Иначе оценка превращается в самоподтверждение |
| 7 | Один прогон — один воспроизводимый артефакт | Фиксируются датасет, корпус, конфиг, модель и git revision |

### 1.3. Не цели

- Отдельная платформа оценки, LangSmith или облачный сервис.
- Отдельный агент-критик или multi-agent evaluation.
- Замена unit/integration-тестов benchmark-прогонами.
- Автоматическое изменение эталонных ответов после прогона.
- Полностью LLM-судейская оценка всех ответов.
- Сложная система весов и единственный итоговый балл.
- Переписывание `AgentService`, retrieval или генератора ради удобства benchmark.

### 1.4. Термины

| Термин | Значение |
|---|---|
| Test case | Один вопрос или один многошаговый сценарий с эталоном |
| Turn | Один запрос пользователя внутри сценария |
| Run | Один запуск набора тестов с фиксированными датасетом и конфигом |
| Trace | Наблюдаемая трасса выполнения: план, инструменты, поиски, evidence, ответ |
| Gold evidence | Набор chunk_id, которые действительно подтверждают эталон |
| Evidence group | Группа альтернативных фрагментов: достаточно любого одного |
| Critical error | Ошибка, которая делает тест провальным независимо от стиля ответа |

### 1.5. Канонический путь

Пакет называется `eval/` — как в Target Code Map и `spec_ui_eval.md`. Структура из
предложения сохраняется; переименование в `evaluation/` не требуется, иначе в проекте
появятся два конкурирующих контура.

```text
eval/
├── datasets/
│   ├── easy.json
│   ├── medium.json
│   ├── hard.json
│   └── multi_turn.json
├── evaluators/
│   ├── retrieval.py
│   ├── answer.py
│   ├── dialogue.py
│   └── judge.py          # опционально
├── runner.py
├── report.py
├── dataset.py
├── trace.py
├── metrics.py
└── results/
    └── .gitkeep
```

`eval/results/` игнорируется Git, кроме `.gitkeep`. Датасеты коммитятся: они являются
частью спецификации качества, а не локальным артефактом.

Ранее предложенный в `spec_ui_eval.md` единый `eval/dataset.jsonl` заменяется
suite-файлами `eval/datasets/*.json`; JSONL остаётся только для результатов
(`cases.jsonl`). Это позволяет держать эталон в ревьюируемом виде и считать hash
набора.

`data/eval/retrieval_questions.json` и `scripts/eval_retrieval.py` остаются
компонентным retrieval-пробником. Их не нужно расширять до agent benchmark и
дублировать в них ответы/факты; канонический эталон — `eval/datasets/`.

## 2. Архитектура контура

```text
┌─────────────────────┐
│ Benchmark Dataset   │  вопросы · эталонные факты · gold evidence · сценарии
└──────────┬──────────┘
           │ load + validate
           ▼
┌─────────────────────┐
│ Evaluation Runner   │  реальный AgentService, один conversation_id на сценарий
└──────────┬──────────┘
           │ ask() / stream()
           ▼
┌─────────────────────┐
│ Agent_RAG           │  Router → ActionSelector → Tools → Retrieval → Evidence → Generator
└──────────┬──────────┘
           │ AgentRunResult + diagnostics
           ▼
┌─────────────────────┐
│ Execution Trace     │  query plan · search calls · dense/BM25/RRF/rerank · evidence
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Evaluators          │  Retrieval · Answer · Reasoning · Dialogue
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Benchmark Report    │  summary.json · cases.jsonl · traces · HTML/Markdown
└─────────────────────┘
```

### 2.1. Границы

- Runner вызывает только публичный `AgentService.ask()`; для одного сценария
  используется один `conversation_id` и последовательные turn'ы.
- Benchmark не вызывает retrieval напрямую для agent-level тестов. Прямой вызов
  `HybridRetriever` допустим только в компонентных тестах метрик.
- Benchmark не передаёт агенту `expected`, `gold_evidence`, `checks` и `reference_answer`.
- Benchmark не меняет prompt, бюджет инструментов, фильтры или ответ.
- Один экземпляр `AgentService` на прогон допустим; изоляция обеспечивается уникальным
  `conversation_id` на тест. Флаг `--fresh-service` создаёт сервис на каждый тест.
- Benchmark не является частью обычного `pytest`-прогона, потому что вызывает LLM и
  занимает заметное время. Offline-тесты проверяют только схему, метрики и отчёты.

### 2.2. Два уровня оценки

| Уровень | Что запускается | Назначение |
|---|---|---|
| Component | `HybridRetriever`, `GraphAugmentedRetriever`, `PerYearSearch` | Быстрая диагностика retrieval без LLM-генерации |
| Agent | `AgentService.ask()` | Полная цепочка: план, инструменты, evidence, ответ, диалог |

Основной benchmark — agent-level. Component-level используется для объяснения, почему
agent-level тест провален, и для проверки режимов `hybrid`/`dense`/`keyword`.

### 2.3. WebUI

Запуск и просмотр результатов в React WebUI описаны отдельно в
`docs/specs/evaluation-webui.md`. Этот документ остаётся источником истины для
датасета, trace, метрик и артефактов; UI-спецификация — для навигации, API,
очереди, хранения и отображения. Benchmark не запускается в браузере и не
дублирует `AgentService`.

## 3. Формат датасета

### 3.1. Файлы и общая оболочка

Каждый файл — UTF-8 JSON:

```json
{
  "schema_version": "1.0",
  "suite": "easy",
  "tests": []
}
```

`suite` обязан совпадать с именем файла: `easy`, `medium`, `hard`, `multi_turn`.
Допускается один файл на suite; дробить на десятки файлов не нужно.

### 3.2. Поля одиночного теста

| Поле | Обязательно | Назначение |
|---|---|---|
| `id` | да | Уникальный ID: `easy_001`, `hard_003`, `multi_turn_002` |
| `suite` | да | `easy` / `medium` / `hard` / `multi_turn` |
| `split` | да | `dev` или `holdout` |
| `category` | да | `fact_lookup`, `temporal`, `structural`, `multi_hop`, `abstain`, `dialogue` |
| `difficulty` | да | `easy` / `medium` / `hard` |
| `tags` | нет | Например, `smoke`, `numeric`, `citation` |
| `question` | да* | Вопрос; для `multi_turn` вместо него используется `turns` |
| `filters` | нет | Явные `SearchFilters`; по умолчанию пустые |
| `expected_status` | да | `answer` или `abstain` |
| `reference_answer` | нет | Эталонный текст только для судьи/ручной проверки, не для exact match |
| `expected` | да | Ожидаемые факты, годы, документы, элементы, расчёты, structural changes |
| `gold_evidence` | да | Группы фрагментов, подтверждающих ответ |
| `expected_behavior` | да | Ожидаемые инструменты, режимы, бюджеты, цитаты |
| `checks` | да | Запреты, обязательные упоминания, ожидаемые warnings |
| `notes` | нет | Источник эталона: страница, таблица, комментарий ревьюера |

\* Для `multi_turn` вопрос задаётся в каждом turn.

### 3.3. Пример одиночного теста

```json
{
  "id": "easy_001",
  "suite": "easy",
  "split": "dev",
  "category": "fact_lookup",
  "difficulty": "easy",
  "tags": ["smoke", "numeric"],
  "question": "Какова чистая прибыль Сбербанка за 2023 год?",
  "filters": {"years": [2023], "allowed_doc_ids": ["sber_2023"]},
  "expected_status": "answer",
  "reference_answer": "Чистая прибыль за 2023 год составила 1 508,6 млрд руб.",
  "expected": {
    "required_facts": [
      {
        "metric": "net_profit",
        "metric_aliases": ["чистая прибыль", "прибыль"],
        "year": 2023,
        "value": 1508.6,
        "unit": "млрд руб.",
        "unit_aliases": ["млрд рублей", "млрд RUB"],
        "tolerance": 0.01
      }
    ],
    "required_years": [2023],
    "required_documents": ["sber_2023"],
    "required_issuers": [],
    "required_answer_elements": ["значение", "единица измерения"],
    "required_calculations": [],
    "required_structural_changes": [],
    "missing_years": []
  },
  "gold_evidence": {
    "min_coverage": 1.0,
    "groups": [
      {
        "id": "net_profit_2023",
        "any_of": ["sber_2023:00078:c3:c5cc6e1f46e2"],
        "section": "Финансовые результаты"
      }
    ]
  },
  "expected_behavior": {
    "retrieval_modes": ["hybrid"],
    "max_search_calls": 2,
    "max_tool_calls": 4,
    "must_use_tools": ["search_reports"],
    "may_use_tools": ["get_chunk"],
    "must_not_use_tools": [],
    "strict_search_budget": false,
    "must_cite_sources": true,
    "must_abstain_if_missing": true
  },
  "checks": {
    "must_mention": ["чистая прибыль"],
    "must_not_mention": [],
    "forbidden_values": [],
    "forbidden_years": [2022, 2024],
    "expected_warnings": [],
    "must_not_invent_values": true,
    "strict_quotes": true
  },
  "notes": "Значение и chunk_id проверены по отчёту за 2023 год."
}
```

### 3.4. Gold evidence

`gold_evidence` — не «правильный ответ», а фрагменты, которые действительно подтверждают
ответ. Это позволяет отделить ошибку поиска от ошибки генерации.

```json
{
  "min_coverage": 1.0,
  "groups": [
    {
      "id": "profit_2022",
      "any_of": ["sber_2022:00123:c2:...", "sber_2022:00124:c0:..."],
      "section": "Финансовые результаты"
    },
    {
      "id": "profit_2024",
      "any_of": ["sber_2024:00089:c1:..."],
      "section": "Финансовые результаты"
    }
  ]
}
```

Правила:

- `any_of` — альтернативы; для покрытия группы достаточно одного chunk_id.
- Все группы обязательны, если не указано иное; `min_coverage` задаёт порог покрытия.
- `chunk_id` должны существовать в текущем chunk store. Фантазийные `chunk_123` запрещены.
- `section` — подсказка для диагностики, не обязательный фильтр.
- `min_coverage` по умолчанию `1.0`; для medium/hard допускается ниже, если это явно
  обосновано в `notes`.
- Если evidence-группа неполная, ответ обязан назвать gap; иначе это critical error
  `missing_required_year_claimed_complete` или `evidence_gap_not_disclosed`.

### 3.5. Ожидаемые факты и расчёты

Факт считается найденным, если одновременно совпали:

- `metric` (или один из `metric_aliases`);
- `year`;
- `value` с учётом `tolerance` (по умолчанию относительная, 0.01 = 1%);
- `unit` или один из `unit_aliases`;
- источник: structured `Answer.facts` с `chunk_id` из citations либо значение в тексте
  ответа рядом с метрикой/годом и подтверждающей citation.

Для `expected_status=abstain` обязательны `expected.reason_codes` (например,
`missing_year`, `no_evidence`, `scope_mismatch`) или эквивалентный
`checks.expected_warnings`.

Ожидаемый расчёт:

```json
{
  "metric": "net_profit",
  "from_year": 2020,
  "to_year": 2024,
  "kind": "percent_change",
  "value": 0.0,
  "tolerance": 0.02
}
```

Сравниваются поля `Answer.calculations` / `AgentRunResult.calculations`:
`metric`, `from_year`, `to_year`, `percent_change`, `cagr`, `absolute_change`.
`kind` определяет, какое поле проверяется. Если расчёт обязателен, а его нет —
`calculation_missing`; если значение не совпало — critical `calculation_mismatch`.
`tolerance` для `percent_change`/`cagr` задаётся в долях от ожидаемого значения,
для `absolute_change` — в единицах `unit`. Числа в примере — иллюстрация схемы; в
датасете они обязаны быть проверенными.

Ожидаемое структурное изменение:

```json
{
  "type": "renamed",
  "section": "Управление рисками",
  "from_doc": "sber_2022",
  "to_doc": "sber_2023"
}
```

`type` ∈ `renamed` / `added` / `removed` / `restructured`; проверяется по
`AgentRunResult.structural_changes`. Структурный diff — навигация, а не доказательство:
содержательное утверждение о разделе всё равно требует citation.

### 3.6. Ожидаемое поведение и проверки

`expected_behavior` описывает не стиль ответа, а наблюдаемые действия:

| Поле | Проверка |
|---|---|
| `retrieval_modes` | Какие режимы допустимы; обычно `["hybrid"]` |
| `max_search_calls` | Лимит `search_reports` на turn; превышение — warning `excess_search_calls` |
| `max_tool_calls` | Общий лимит инструментов на turn |
| `must_use_tools` | Инструменты, которые обязаны быть вызваны |
| `may_use_tools` | Разрешённые, но не обязательные |
| `must_not_use_tools` | Запрещённые для этого теста |
| `strict_search_budget` | Если `true`, превышение бюджета делает тест провальным |
| `must_cite_sources` | Требовать хотя бы одну валидную citation |
| `must_abstain_if_missing` | При отсутствии evidence ожидается abstain, а не догадка |

Если `max_search_calls`/`max_tool_calls` не заданы, действуют defaults:

| Difficulty | `max_search_calls` | `max_tool_calls` |
|---|---|---|
| easy | 2 | 4 |
| medium | 4 | 6 |
| hard | 5 | 8 |
| multi_turn (на turn) | 3 | 6 |

Defaults не могут превышать системные лимиты (`MAX_SEARCH_CALLS=5`,
`MAX_TOOL_CALLS=8` в текущем коде; при изменении констант таблица обновляется).
При `strict_search_budget=false` превышение даёт warning `excess_search_calls`;
при `true` — провал теста.

В multi-turn turn'е поле `behavior` — короткий машинный код ожидаемого поведения
(`clarify_period`, `continue_previous_request`, `use_previous_evidence_and_calculate`,
`apply_correction`, `recheck_primary_source`), а `expected_behavior` — объект проверок,
как в одиночном тесте. Это разделяет смысл turn'а и наблюдаемые ограничения.

`checks`:

| Поле | Проверка |
|---|---|
| `must_mention` | Подстроки/регулярки (с префиксом `re:`), регистронезависимо |
| `must_not_mention` | Запрещённые формулировки |
| `forbidden_values` | Числа, которые не должны появляться как ответ |
| `forbidden_years` | Годы, которые нельзя приписывать факту |
| `expected_warnings` | Ожидаемые коды, например `missing_year`, `incomplete_period`; проверяются по union `Answer.warnings` и `AgentRunResult.warnings` |
| `must_not_invent_values` | Включает проверку числовых утверждений без источника |
| `strict_quotes` | Если `true`, `quote` обязан быть подстрокой chunk после нормализации |

### 3.7. Многошаговый сценарий

```json
{
  "id": "multi_turn_001",
  "suite": "multi_turn",
  "split": "dev",
  "category": "dialogue",
  "difficulty": "medium",
  "tags": ["smoke", "period"],
  "dialogue_policy": {
    "missing_period": "clarify"
  },
  "turns": [
    {
      "question": "Как менялась чистая прибыль Сбербанка?",
      "behavior": "clarify_period",
      "expected_status": "abstain",
      "expected": {
        "required_facts": [],
        "required_years": [],
        "required_documents": [],
        "required_answer_elements": ["уточнение периода"],
        "required_calculations": [],
        "missing_years": [],
        "reason_codes": ["clarification_required"]
      },
      "gold_evidence": {"min_coverage": 0.0, "groups": []},
      "expected_behavior": {
        "max_search_calls": 0,
        "must_use_tools": [],
        "must_not_use_tools": ["search_reports"]
      },
      "checks": {
        "must_mention": ["период"],
        "forbidden_values": [],
        "forbidden_years": [],
        "expected_warnings": [],
        "must_not_invent_values": true
      }
    },
    {
      "question": "За последние пять лет.",
      "behavior": "continue_previous_request",
      "expected_status": "answer",
      "expected": {
        "required_facts": [],
        "required_years": [2020, 2021, 2022, 2023, 2024],
        "required_documents": ["sber_2020", "sber_2022", "sber_2023", "sber_2024"],
        "required_answer_elements": ["динамика", "пропуск 2021"],
        "required_calculations": [],
        "missing_years": [2021]
      },
      "gold_evidence": {
        "min_coverage": 0.75,
        "groups": [
          {"id": "profit_2020", "any_of": ["sber_2020:00067:c2:..."]},
          {"id": "profit_2022", "any_of": ["sber_2022:00123:c2:..."]},
          {"id": "profit_2023", "any_of": ["sber_2023:00078:c3:..."]},
          {"id": "profit_2024", "any_of": ["sber_2024:00089:c1:..."]}
        ]
      },
      "expected_behavior": {
        "max_search_calls": 4,
        "must_use_tools": ["search_reports"],
        "must_not_use_tools": []
      },
      "checks": {
        "must_mention": ["2020", "2024", "2021"],
        "forbidden_values": [],
        "forbidden_years": [],
        "expected_warnings": ["missing_year", "incomplete_period"],
        "must_not_invent_values": true
      }
    },
    {
      "question": "А на сколько процентов она выросла?",
      "behavior": "use_previous_evidence_and_calculate",
      "reuse_previous_evidence": true,
      "expected_status": "answer",
      "expected": {
        "required_facts": [],
        "required_years": [],
        "required_documents": [],
        "required_answer_elements": ["базовый год", "конечный год", "процент"],
        "required_calculations": [
          {
            "metric": "net_profit",
            "from_year": 2020,
            "to_year": 2024,
            "kind": "percent_change",
            "value": 0.0,
            "tolerance": 0.02
          }
        ],
        "missing_years": []
      },
      "gold_evidence": {"min_coverage": 0.0, "groups": []},
      "expected_behavior": {
        "max_search_calls": 1,
        "must_use_tools": [],
        "must_not_use_tools": []
      },
      "checks": {
        "must_mention": ["2020", "2024", "%"],
        "forbidden_values": [],
        "forbidden_years": [],
        "expected_warnings": [],
        "must_not_invent_values": true
      }
    }
  ],
  "scenario_checks": [
    "context_preserved",
    "correct_metric",
    "correct_period",
    "no_unnecessary_restart",
    "calculation_correct",
    "sources_preserved"
  ],
  "required_checks": [
    "context_preserved",
    "correct_metric",
    "correct_period",
    "no_unnecessary_restart",
    "sources_preserved"
  ],
  "notes": "Значения в required_calculations заполняются по эталону; 2021 отсутствует в корпусе."
}
```

Правила:

- `dialogue_policy.missing_period`: `clarify` или `default_period`. Политика фиксируется
  в тесте, чтобы не считать правильными одновременно два разных поведения.
- `turns` — минимум 2; каждый turn имеет свой вопрос и проверки.
- `reuse_previous_evidence=true` разрешает пустой `gold_evidence` для follow-up
  расчёта/уточнения; evaluator использует evidence предыдущего turn'а и проверяет
  `sources_preserved`/`evidence_not_lost`.
- `scenario_checks` — полный список проверок сценария; `required_checks` — те, без
  которых сценарий провален. Остальные проверки диагностические.
- `conversation_id` создаётся один на сценарий и не переиспользуется между тестами.
- При провале одного turn выполнение продолжается, но последующие проверки контекста
  помечаются `cascade_failure` и не считаются успешными по умолчанию.
- `reference_answer` для сценария может быть на уровне turn или всего сценария; он
  используется только судьёй/ревьюером.

### 3.8. Обязательные сценарии multi-turn

Минимум пять сценариев:

| ID | Сценарий | Что проверяется |
|---|---|---|
| `multi_turn_001` | Уточнение периода | Не начинает поиск без периода; использует уточнение или default policy |
| `multi_turn_002` | Углубление по разделу | Сохраняет раздел/документы; делает точечный поиск, а не всё заново |
| `multi_turn_003` | Противоречие в значении | Открывает первоисточник, объясняет расхождение, не выбирает значение наугад |
| `multi_turn_004` | Смена показателя | «Не чистую прибыль, а операционную»: меняет metric, сохраняет период/сущность |
| `multi_turn_005` | Сравнение с предыдущим годом | Понимает, что показатель тот же; добавляет год, не задаёт все вопросы заново |

### 3.9. Валидация датасета

Валидатор запускается до любого прогона и является обязательным:

1. `id` уникален во всех файлах; `suite` совпадает с файлом.
2. `split` ∈ `{dev, holdout}`; holdout — не менее 20% набора.
3. `expected_status=answer` требует непустой `gold_evidence.groups` и
   `min_coverage > 0`, кроме turn'а с `reuse_previous_evidence=true`, где evidence
   берётся из предыдущего turn'а и `groups` может быть пустым.
4. `expected_status=abstain` требует причину в `checks.expected_warnings` или
   `expected.reason_codes`.
5. Все `chunk_id` существуют в chunk store; `doc_id`/`year` в эталоне совпадают с
   метаданными чанка.
6. Все числовые `value` заполнены; `null`, `"TBD"` и пустые строки запрещены.
7. `tolerance >= 0`; `unit` непустой для числового факта.
8. `required_years` не содержит год, отсутствующий в корпусе, кроме случая, когда год
   явно указан в `missing_years` и ожидается abstain/gap.
9. `filters` проходят `SearchFilters`; `year_from <= year_to`; годы — целые.
10. Нет дубликатов вопросов после нормализации пробелов и регистра.
11. `multi_turn`: ≥2 turn'а, известные `scenario_checks`, уникальные вопросы.
12. Нет абсолютных путей, секретов и фантазийных `chunk_id`.
13. `reference_answer` не используется как exact match.
14. Инструменты в `must_use_tools`/`may_use_tools`/`must_not_use_tools` входят в
    allowlist `AgentTools`; `scenario_checks`/`behavior` — из известного словаря.
15. SHA-256 датасета считается по всем suite-файлам в фиксированном порядке
    (имя файла + канонический JSON с сортировкой ключей, UTF-8) и попадает в
    `summary.json`.

### 3.10. Состав и качество набора

Стартовый набор — **30 тестов**:

| Suite | Кол-во | Примеры |
|---|---|---|
| easy | 10 | конкретный факт, число, перечисление, раздел, риски |
| medium | 10 | временной ряд, сравнение 2022/2024, стратегия, цифровая трансформация, приоритеты |
| hard | 5 | структура отчётов, стратегия + финансовые показатели, риск-подход, приоритеты, противоречия |
| multi_turn | 5 | период, углубление, противоречие, смена метрики, сравнение с годом |

Тематический план первого набора:

| Suite | Темы |
|---|---|
| easy | чистая прибыль за год; активы; основные направления деятельности; раздел стратегии; основные риски |
| medium | динамика прибыли 2020–2024; сравнение активов 2022/2024; изменения стратегии за 3 года; цифровая трансформация; приоритетные направления бизнеса |
| hard | структура отчётов 2020–2024; стратегия и финансовые показатели; управление рисками и новые риски; приоритеты и их подтверждение; противоречия стратегии и результатов |
| multi_turn | уточнение периода; углубление по разделу рисков; проверка противоречия; смена метрики; сравнение с предыдущим годом |

Оставшиеся easy/medium-тесты — вариации по годам, документам и формулировкам, а не
копии одного и того же вопроса.

Требования:

- 10/10/5/5 — минимум; расширение до 50 — после стабилизации первых 30.
- Не менее 6 тестов (20%) — `holdout`; они не используются для настройки prompt,
  retrieval или порогов и открываются перед релизом.
- Smoke-набор — 12 dev-тестов: 4 easy, 4 medium, 2 hard, 2 multi_turn, с тегом
  `smoke`; holdout-тесты не помечаются `smoke`.
- Покрытие: разные годы, документы, issuer, типы вопросов; 2021 отсутствует физически,
  поэтому используется в abstain/gap-тестах, а не как answerable-факт.
- Каждый эталонный факт проверен по первоисточнику; для hard/holdout — второе ревью.
- Вопросы на русском, без двусмысленных дат и без подсказки ответа в формулировке.
- Вопрос не должен требовать внешних знаний, которых нет в корпусе.
- Эталонные ответы не обновляются автоматически после прогона.

## 4. Контракт trace

### 4.1. Что сохраняется

Один JSON trace на тест:

```json
{
  "trace_schema_version": "1.0",
  "run_id": "20261114_120000_f77652a",
  "test_id": "easy_001",
  "conversation_id": "bench-easy_001-...",
  "started_at": "2026-11-14T12:00:00Z",
  "duration_ms": 12345.6,
  "status": "completed",
  "git_sha": "f77652a",
  "git_dirty": false,
  "dataset_sha256": "...",
  "corpus_fingerprint": "...",
  "config_fingerprint": "...",
  "turns": [
    {
      "turn_index": 0,
      "question": "Какова чистая прибыль Сбербанка за 2023 год?",
      "request_id": "...",
      "status": "completed",
      "query_plan": {
        "intent": "factoid",
        "years": [2023],
        "year_from": null,
        "year_to": null,
        "section_filter": null,
        "entities": [],
        "needs_rewriting": false,
        "needs_graph": false,
        "answer_mode": "concise"
      },
      "effective_filters": {
        "years": [2023],
        "allowed_doc_ids": ["sber_2023"]
      },
      "tool_events": [
        {
          "tool_name": "search_reports",
          "arguments": {"retrieval_mode": "hybrid", "top_k": 5},
          "status": "completed",
          "result_count": 5,
          "duration_ms": 2400.0,
          "warning": null
        }
      ],
      "search_attempts": [
        {
          "kind": "broad",
          "year": null,
          "years": [2023],
          "section_path": null,
          "result_count": 5,
          "covered_years": [2023]
        }
      ],
      "retrieval_calls": [
        {
          "call_index": 0,
          "tool_name": "search_reports",
          "query": "чистая прибыль Сбербанка 2023",
          "queries": ["чистая прибыль Сбербанка 2023"],
          "retrieval_mode": "hybrid",
          "top_k": 5,
          "filters": {"years": [2023]},
          "status": "completed",
          "warnings": [],
          "counts": {
            "dense": 30,
            "bm25": 30,
            "rrf": 40,
            "rerank": 10,
            "returned": 5
          },
          "stages": {
            "dense": [
              {"chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2", "rank": 1, "score": 0.82}
            ],
            "bm25": [
              {"chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2", "rank": 2, "score": 18.4}
            ],
            "rrf": [
              {"chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2", "rank": 1, "score": 0.032}
            ],
            "rerank": [
              {"chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2", "rank": 1, "score": 3.0}
            ],
            "returned": [
              {"chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2", "rank": 1, "score": 3.0}
            ]
          },
          "fallback": {
            "reranker": false,
            "dense": false,
            "bm25": false
          }
        }
      ],
      "evidence_manifest": ["sber_2023:00078:c3:c5cc6e1f46e2"],
      "context_chunk_ids": ["sber_2023:00078:c3:c5cc6e1f46e2"],
      "context_source": "evidence_manifest",
      "covered_years": [2023],
      "missing_years": [],
      "answer": {
        "status": "answer",
        "text": "Чистая прибыль за 2023 год составила 1 508,6 млрд руб.",
        "citations": [
          {
            "chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2",
            "doc_id": "sber_2023",
            "year": 2023,
            "issuer": "ПАО Сбербанк",
            "page_pdf": 78,
            "section_path": "Финансовые результаты",
            "quote": "Чистая прибыль ..."
          }
        ],
        "facts": [],
        "warnings": [],
        "calculations": []
      },
      "timings": {
        "total_ms": 12345.6,
        "answer_ms": 4100.0
      }
    }
  ],
  "errors": []
}
```

### 4.2. Обязательные поля

| Поле | Назначение |
|---|---|
| `query_plan` | Проверка понимания вопроса: intent, годы, метрики, режим |
| `effective_filters` | Какие фильтры реально применены |
| `tool_events` | Наблюдаемые вызовы инструментов и их статусы |
| `search_attempts` | Вид поиска, годы, раздел, покрытие |
| `retrieval_calls` | Полные этапы каждого поиска |
| `evidence_manifest` | Чанки, переданные генератору |
| `context_chunk_ids` | Чанки, реально упакованные в контекст |
| `covered_years` / `missing_years` | Полнота периода |
| `answer` | Текст, citations, facts, calculations, warnings |
| `timings` | Общая длительность и этапы |
| `errors` | Ошибки запуска, не замаскированные нулём |

### 4.3. Этапы retrieval

Для каждого `search_reports` сохраняются:

| Этап | Содержимое |
|---|---|
| `dense` | `chunk_id`, rank, score по dense-каналу |
| `bm25` | `chunk_id`, rank, score по BM25 |
| `graph` | `chunk_id`, path/evidence, если использовался graph |
| `rrf` | `chunk_id`, rank, fused score |
| `rerank` | `chunk_id`, rank, rerank score |
| `returned` | Финальный top-k, переданный агенту |
| `context` | Фактически упакованный контекст, если генератор его раскрывает |

Правила:

- `call_index` соответствует порядку вызовов `search_reports` в `tool_events`.
- `score` — технический ранг/скор, не вероятность истинности.
- `rank` начинается с 1; дубликаты схлопываются по `chunk_id`.
- Если graph вызывается отдельным tool (`get_entities`), его результат сохраняется как
  отдельный `retrieval_call` с `tool_name=get_entities` и `stages.graph`; это не
  смешивается с dense/BM25.
- Если этап недоступен, поле равно `null`, а в `fallback`/`available` стоит `false`.
  Evaluator помечает метрику `not_measured`, а не `0`.
- При reranker fallback сохраняется исходный RRF-порядок и флаг
  `fallback.reranker=true`.
- Для `dense`/`keyword` соответствующие отсутствующие каналы не заполняются
  фиктивными данными.

### 4.4. Как получать trace без изменения логики

Достаточно наблюдаемости, а не нового workflow:

1. `HybridRetriever` дополняет `RetrievalTrace` опциональными списками этапов
   (`stages`) и счётчиками. Семантика поиска не меняется.
2. `PerYearSearch` и `GraphAugmentedRetriever` копируют базовый trace и добавляют
   свои этапы; при отсутствии данных ставят `available=false`.
3. `AgentService` уже хранит `state["retrieval_results"]`; в
   `AgentRunResult.diagnostics` добавляются `query_plan`, `effective_filters`,
   `search_attempts`, `evidence_manifest`, `context_chunk_ids`, `retrieval_calls`,
   `coverage`, `verification`.
4. `diagnostics` — новое поле с `field(default_factory=dict)`; старые вызовы и тесты
   не ломаются.
5. Если `context_chunk_ids` недоступны, используется `evidence_manifest` с
   `context_source="evidence_manifest"`; это явно помечается в отчёте.

Запрещено:

- сохранять chain-of-thought, полный prompt или полный текст корпуса;
- менять ответ, фильтры или бюджеты ради красивого trace;
- подменять отсутствующий этап синтетическими хитами.

### 4.5. Размер и редактирование

- В trace хранятся только ID, ранги, scores, метаданные и короткие цитаты.
- Полный текст чанка загружается evaluator'ом из chunk store по `chunk_id`.
- Поисковые запросы сохраняются: они нужны для диагностики и являются пользовательским
  текстом, а не секретом; API-ключи и заголовки не сохраняются.
- Секреты, заголовки запросов и ключи не сохраняются.
- Лимит одного trace — 5 МБ; при превышении списки этапов обрезаются с
  `truncated=true`, а не молча теряются.
- `cases.jsonl` не дублирует полные этапы; он ссылается на `traces/{test_id}.json`.

## 5. Оценка

### 5.1. Retrieval

Для теста с gold evidence:

```text
G = gold_evidence.groups
R = упорядоченный список уникальных chunk_id
covered(g, R) = any(chunk_id in R for chunk_id in g.any_of)
```

| Метрика | Формула | Смысл |
|---|---|---|
| `Recall@5` | `|{g ∈ G: covered(g, R[:5])}| / |G|` | Доля обязательных групп в первых 5 |
| `Recall@10` | `|{g ∈ G: covered(g, R[:10])}| / |G|` | Доля обязательных групп в первых 10 |
| `MRR` | `1 / rank` первого chunk из любой `g`; 0, если не найден | Как высоко первое доказательство |
| `Evidence Coverage` | `|{g: covered(g, E)}| / |G|`, где `E` — финальный `evidence_manifest` | Хватило ли доказательств генератору |
| `Context Coverage` | То же по `context_chunk_ids` | Дошли ли доказательства до контекста |
| `nDCG@5` | Вторичная метрика из `spec_ui_eval.md` | Совместимость с прежним отчётом |

Правила расчёта:

- Основной `Recall@k`/`MRR` считается по первому результативному `search_reports`;
  при нескольких вызовах дополнительно сохраняется `best_call` и `all_calls`.
- `Evidence Coverage` считается по `evidence_manifest`, то есть по чанкам, реально
  переданным генератору; дополнительно сохраняется `retrieval_coverage` по
  объединению всех `returned`, чтобы отличить потерю на retrieval от потери на
  упаковке контекста.
- Если `gold_evidence` пуст (abstain-тест), retrieval-метрики не считаются; тест
  учитывается в answer/abstention, а не в retrieval denominator.
- Агрегация — macro mean по тестам, затем breakdown по `category`, `difficulty`,
  `split` и `retrieval_mode`.
- `nDCG@5` — вторичная; при отсутствии graded relevance используется бинарная
  релевантность gold-группы.
- `P@k`/`AP@k` из `spec_ui_eval.md` остаются в компонентном
  `scripts/eval_retrieval.py`; agent-level отчёт не обязан их дублировать.

### 5.2. Диагностика потерь

Для каждого gold chunk определяется, на каком этапе он потерян:

| Код | Условие |
|---|---|
| `not_in_corpus` | chunk_id отсутствует в chunk store |
| `never_retrieved` | нет ни в dense, ни в BM25 |
| `lost_at_fusion` | был в канале, но не попал в RRF |
| `lost_at_rerank` | был в RRF, но не попал в rerank |
| `lost_at_context` | был в returned, но не попал в context |
| `not_cited` | был в контексте, но не процитирован |
| `ok` | найден и подтверждён |

Дополнительно сохраняются счётчики:

| Этап | Пример |
|---|---|
| Dense | 30 |
| BM25 | 30 |
| RRF | 40 |
| Reranker | 10 |
| Передано в контекст | 5 |

Если нужный chunk был в dense, но исчез после reranker, проблема не в генерации.
Если он был в returned, но не в context — проблема в упаковке контекста.

### 5.3. Ответ: детерминированные проверки

| Проверка | Способ | Результат |
|---|---|---|
| `status_accuracy` | `Answer.status` vs `expected_status` | critical при несовпадении |
| `fact_accuracy` | metric/year/value/unit с tolerance | critical при неверном значении/годе |
| `fact_completeness` | все `required_facts` найдены | fail при пропуске |
| `required_years` | `required_years - missing_years` в facts/тексте + citations | fail при пропуске; `missing_years` должны быть раскрыты |
| `forbidden_years` | годы не приписаны фактам | critical |
| `citation_validity` | все chunk_id существуют и входят в manifest; если `context_chunk_ids` доступны — также в контекст | critical |
| `citation_coverage` | у каждого факта есть подтверждающая citation | fail |
| `quote_validity` | `quote` — подстрока chunk (нормализованные пробелы) | warning; critical при `strict_quotes` |
| `citation_gold` | citations покрывают `gold_evidence` | fail при низком coverage |
| `calculation_accuracy` | сравнение `Answer.calculations` | critical при неверном расчёте |
| `abstention_accuracy` | abstain без citations и с причиной | fail |
| `required_elements` | `must_mention` / regex | fail |
| `forbidden_values` | числа в ответе | critical |
| `numeric_claim_unverified` | число с единицей без факта/citation | warning или critical по флагу |
| `wrong_document` | citation из неожидаемого документа | warning/critical для факта |
| `structural_changes` | `AgentRunResult.structural_changes` | fail для structural-тестов |
| `tool_behavior` | `must_use_tools` / `must_not_use_tools` по `tool_events` | fail при нарушении |

Если `context_chunk_ids` недоступны, `citation_validity` проверяется только по
`evidence_manifest`, а проверка контекста помечается `not_measured`; это не должно
молча считаться полной проверкой.

Важно:

- `expected.missing_years` — это годы, которых нет в корпусе или в найденном периоде.
  Они исключаются из обязательных фактов, но ответ обязан назвать пропуск. Если ответ
  заявляет полное сравнение без пропущенного года — critical
  `missing_required_year_claimed_complete`.
- Точное совпадение текста ответа не используется.
- `reference_answer` — только для судьи и ручной проверки.
- Если структурированных `facts` нет, но значение найдено в тексте рядом с метрикой,
  годом и валидной citation, факт считается найденным; это помечается
  `matched_in_text=true`.
- Если число в тексте не удаётся связать с фактом или citation, это
  `numeric_claim_unverified`, а не автоматический critical, кроме тестов с
  `must_not_invent_values=true` и явным `forbidden_values`.

### 5.4. Рубрика ответа

Каждый ответ оценивается по пяти критериям от 0 до 2:

| Критерий | 0 | 1 | 2 |
|---|---|---|---|
| Фактическая точность | Ошибка | Частично верно | Верно |
| Полнота | Существенно неполно | Частично | Полностью |
| Источники | Нет или неверные | Частично подтверждают | Подтверждают |
| Анализ | Неверный | Поверхностный | Корректный |
| Соблюдение запретов | Выдумывает | Есть сомнительные выводы | Не выходит за evidence |

Максимум — 10. Рубрика заполняется LLM-судьёй и/или человеком; детерминированные
проверки не маскируются под рубрику.

Для совместимости с `spec_ui_eval.md` судья может дополнительно выдавать
`faithfulness`, `answer_relevance`, `context_relevance`, `answer_correctness` (0–2).
Эти поля вторичны и не заменяют пять критериев.

### 5.5. Критические ошибки

Критическая ошибка делает тест провальным независимо от красоты ответа:

| Код | Значение | Детектор |
|---|---|---|
| `wrong_status` | Ответ вместо abstain или наоборот | deterministic |
| `invalid_citation` | chunk_id не существует, вне manifest или вне доступного контекста | deterministic |
| `citation_quote_mismatch` | `quote` не является фрагментом chunk | deterministic; critical при `strict_quotes=true`, иначе warning |
| `fact_wrong_value` | Неверное число | deterministic |
| `fact_wrong_year` | Число приписано неверному году | deterministic |
| `fact_missing` | Обязательный факт отсутствует | deterministic |
| `wrong_document` | Факт взят из не того документа | deterministic |
| `invented_numeric_value` | Число без источника | deterministic + judge |
| `calculation_mismatch` | Неверный процент/CAGR | deterministic |
| `missing_required_year_claimed_complete` | Заявлено полное сравнение при пропуске года | deterministic + judge |
| `evidence_gap_not_disclosed` | Известный пропуск evidence не назван | deterministic + judge |
| `false_causality` | Причинный вывод без подтверждения | judge/manual |
| `unsupported_claim` | Утверждение не подтверждено источниками | judge/manual |
| `contradiction_not_addressed` | Противоречие не исследовано | judge/manual |
| `dialogue_context_lost` | Потерян показатель/период/сущность | deterministic + judge |
| `dataset_error` | Ошибка эталона, не агента | validator |

Критические ошибки не компенсируются стилем, длиной или rubric score.

### 5.6. LLM-судья

Судья — вспомогательный, не единственный evaluator.

Вход:

- вопрос;
- эталонные факты и `reference_answer`;
- ответ агента;
- citations и найденные фрагменты;
- warnings и расчёт.

Запрещено:

- самостоятельно искать информацию в корпусе;
- видеть скрытые `expected_behavior`, которых нет в кейсе;
- переопределять deterministic critical errors;
- менять эталон.

Выход — строгий JSON с пятью оценками 0–2, списком issue codes и кратким
обоснованием. Невалидный JSON — `judge_error`, а не ноль.

Калибровка:

- 3 контрольных кейса: корректный ответ, неподтверждённое утверждение, неполный ответ.
- Для каждого вручную заполняются 5 критериев (15 пар manual/judge). Если используется
  legacy-набор `spec_ui_eval.md` из 4 критериев, допускается 12 пар; канонические
  5 критериев остаются обязательными для нового отчёта.
- В отчёте показываются все пары, exact matches и расхождения.
- Калибровочные кейсы не попадают в среднее dev score, если использовались для
  изменения judge prompt.
- Судья включается флагом `--judge`; `--no-judge` оставляет прогон полностью
  детерминированным.

### 5.7. Ручная проверка

Обязательна для:

- всех hard-тестов;
- всех multi_turn-сценариев;
- любых тестов с `false_causality`, `unsupported_claim`, `contradiction_not_addressed`;
- выборочно — для автоматических провалов, чтобы отличить ошибку агента от ошибки эталона.

Формат `manual_review.json`:

```json
{
  "reviewer": "initials",
  "tests": {
    "hard_001": {
      "scores": {
        "factual_accuracy": 2,
        "completeness": 1,
        "sources": 2,
        "analysis": 1,
        "constraints": 2
      },
      "notes": "Пропущен 2021, но gap раскрыт."
    }
  }
}
```

Ручная оценка не изменяет `expected` автоматически. Изменение эталона — отдельный
reviewed commit с обновлением hash.

### 5.8. Multi-turn evaluation

Проверки сценария:

| Код | Что проверяет |
|---|---|
| `context_preserved` | Новый вопрос относится к предыдущему; сущность/показатель/период сохранены |
| `correct_entity` | Нет подмены Сбербанка/Группы/issuer |
| `correct_metric` | Показатель верный; после correction старый не выдаётся за целевой |
| `correct_period` | Период сохранён или изменён согласно turn'у |
| `no_unnecessary_restart` | Нет повторного исследования с нуля без причины |
| `calculation_correct` | Процент/CAGR совпадает с эталоном |
| `sources_preserved` | Ранее найденные факты остаются подтверждёнными валидными citations |
| `correction_applied` | Исправление пользователя применено |
| `contradiction_rechecked` | Открыт первоисточник, расхождение объяснено |
| `clarification_asked` | Если политика `clarify`, период уточнён |
| `default_period_used` | Если политика `default_period`, диапазон назван явно |
| `no_reask` | Не задаёт вопрос, ответ на который уже есть в диалоге |
| `evidence_not_lost` | Предыдущие evidence не потеряны без необходимости |
| `gap_disclosed` | Пропуск года/данных назван |

`context_preserved`, `correct_metric`, `correct_period`, `sources_preserved` проверяются
по trace и citations; смысловые проверки — судьёй/человеком.

`no_unnecessary_restart` вычисляется по trace: если follow-up turn повторяет тот же
широкий запрос с теми же filters без нового основания (смена метрики/периода/раздела),
это false. `sources_preserved` не требует повторного использования тех же chunk_id:
доказательство может быть найдено заново, но факты предыдущего turn'а должны оставаться
подтверждёнными валидными citations.

Успех сценария:

```text
scenario_success =
  все turn'ы выполнены без execution error
  и все required_checks = true
  и все обязательные автоматические проверки turn'ов = true
  и нет critical errors
```

Частичный результат сохраняется для диагностики, но не считается успехом. При
`cascade_failure` последующие успехи не маскируют потерю контекста.

### 5.9. Ошибки выполнения

| Статус | Значение | Учёт |
|---|---|---|
| `completed` | Тест выполнен | Входит в denominator |
| `failed` | Агент вернул `failed` | Провал, входит в denominator |
| `timeout` | Превышен лимит turn'а | Провал, входит в denominator |
| `error` | Ошибка runner/evaluator | Провал, виден в coverage |
| `unsupported_mode` | Backend не поддерживает режим | `skipped`, не маскируется hybrid |
| `invalid_dataset` | Ошибка эталона | Прогон не стартует |
| `judge_error` | Судья вернул невалидный JSON | Виден отдельно, не ноль |

В `summary.json` всегда есть `coverage`: сколько тестов выполнено, провалено,
пропущено и почему. Ошибка отдельного теста не должна молча превращаться в `0.0`.
Автоматических retry нет: повтор — только явный `--repeat`, и все попытки сохраняются
в результатах.

### 5.10. Правило прохождения

```text
test_pass =
  execution status == completed/abstained
  и expected_status совпал
  и нет critical errors
  и все mandatory automatic checks прошли
  и Evidence Coverage >= gold_evidence.min_coverage
  и (если есть judge/manual) rubric total не ниже явно заданного порога теста
```

`test_pass` — для pass rate. Rubric score и retrieval-метрики публикуются отдельно.
Единый `Agent Quality` не вычисляется.

## 6. Отчёт и артефакты

### 6.1. Артефакты запуска

```text
eval/results/{run_id}/
├── summary.json
├── cases.jsonl
├── retrieval.csv
├── traces/
│   ├── easy_001.json
│   ├── easy_002.json
│   ├── medium_001.json
│   └── ...
├── judge.jsonl          # если судья включён
├── manual_review.json   # если есть ручная проверка
├── report.html
├── report.md
├── comparison.html      # при --compare
└── comparison.md
```

`run_id` — `YYYYMMDD_HHMMSS_{short_sha}`. Один JSON на тест в `traces/`; при
`--repeat` — `traces/{test_id}.attempt_{n}.json`. `cases.jsonl` содержит краткую
запись на attempt и ссылку на соответствующий trace.

`retrieval.csv` — плоская таблица для анализа:

```text
test_id, retrieval_mode, call_index, stage, rank, chunk_id, score, in_gold, loss_stage
```

`stage` ∈ `dense|bm25|graph|rrf|rerank|returned|context`; `in_gold` — бинарный флаг
попадания в любую gold-группу.

### 6.2. `summary.json`

Обязательные поля:

```json
{
  "run_id": "...",
  "started_at": "...",
  "finished_at": "...",
  "command": "...",
  "git": {"sha": "...", "dirty": false},
  "dataset": {"version": "1.0", "sha256": "...", "tests": 30},
  "corpus": {
    "chunk_count": 0,
    "documents": ["sber_2020", "sber_2022", "sber_2023", "sber_2024"],
    "index_fingerprint": "...",
    "graph_fingerprint": null
  },
  "config": {
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "candidates": 20,
    "rerank_pool": 10,
    "reranker": true,
    "graph": false
  },
  "config_fingerprint": "...",
  "models": {
    "llm": "...",
    "embeddings": "...",
    "reranker": "...",
    "judge": null,
    "generation": {"temperature": 0, "top_p": 1, "max_tokens": 2048}
  },
  "prompts": {
    "router": "sha256:...",
    "answer": "sha256:...",
    "judge": null
  },
  "coverage": {
    "tests_total": 30,
    "executed": 30,
    "failed": 0,
    "skipped": 0,
    "errors": 0
  },
  "pass_at_1": 0.0,
  "stability": null,
  "groups": {
    "retrieval": {
      "recall_at_5": 0.0,
      "recall_at_10": 0.0,
      "mrr": 0.0,
      "ndcg_at_5": null,
      "retrieval_coverage": 0.0,
      "evidence_coverage": 0.0,
      "context_coverage": 0.0,
      "by_mode": {}
    },
    "answer": {
      "automatic_pass_rate": 0.0,
      "fact_accuracy": 0.0,
      "citation_validity": 0.0,
      "citation_coverage": 0.0,
      "calculation_accuracy": 0.0,
      "abstention_accuracy": 0.0,
      "status_accuracy": 0.0,
      "rubric_mean": null
    },
    "reasoning": {
      "pass_rate": 0.0,
      "rubric_mean": null,
      "causality_issues": 0
    },
    "dialogue": {
      "scenario_success_rate": 0.0,
      "by_check": {}
    }
  },
  "by_difficulty": {
    "easy": {"pass_rate": 0.0},
    "medium": {"pass_rate": 0.0},
    "hard": {"pass_rate": 0.0},
    "multi_turn": {"pass_rate": 0.0}
  },
  "by_category": {},
  "by_split": {
    "dev": {"pass_rate": 0.0},
    "holdout": {"pass_rate": 0.0}
  },
  "critical_errors": [
    {"code": "invalid_citation", "count": 0, "test_ids": []}
  ],
  "problem_tests": [
    {"id": "hard_003", "reasons": ["lost_at_rerank", "unsupported_claim"]}
  ],
  "efficiency": {
    "mean_tool_calls": 0.0,
    "mean_search_calls": 0.0,
    "mean_latency_ms": 0.0,
    "llm_calls": null,
    "llm_concurrency": 1
  }
}
```

`0.0` в примере — заполнитель схемы; реальный отчёт не должен выдумывать значения.
Если метрика не измерялась, используется `null`, а не ноль.

`by_difficulty.multi_turn` — это bucket по suite `multi_turn`, а не значение поля
`difficulty`; так отчёт не смешивает многошаговые сценарии с medium-вопросами.

### 6.3. `cases.jsonl`

Одна строка на тест; при `--repeat N` — одна строка на attempt с `attempt_index`.
Итоговый `pass` теста считается по `pass_at_1`, а `stability` — по доле успешных
повторов.

```json
{
  "id": "easy_001",
  "attempt_index": 0,
  "suite": "easy",
  "split": "dev",
  "category": "fact_lookup",
  "difficulty": "easy",
  "retrieval_mode": "hybrid",
  "status": "completed",
  "pass": true,
  "expected_status": "answer",
  "actual_status": "answer",
  "retrieval": {
    "recall_at_5": 1.0,
    "recall_at_10": 1.0,
    "mrr": 1.0,
    "retrieval_coverage": 1.0,
    "evidence_coverage": 1.0,
    "context_coverage": 1.0,
    "loss_stage": "ok"
  },
  "answer": {
    "fact_accuracy": 1.0,
    "citation_validity": 1.0,
    "citation_coverage": 1.0,
    "calculation_accuracy": null
  },
  "critical_errors": [],
  "warnings": [],
  "efficiency": {
    "tool_calls": 1,
    "search_calls": 1,
    "llm_calls": null,
    "latency_ms": 12345.6
  },
  "trace_path": "traces/easy_001.json"
}
```

### 6.4. Главная страница отчёта

Отчёт — самодостаточный HTML без внешних запросов и без сети. Содержит:

- run id, дату, git SHA, модель, датасет, корпус, конфиг;
- четыре карточки:
  - Pass rate;
  - Retrieval Recall@10;
  - Answer quality (automatic pass rate);
  - Multi-turn success;
- результаты по сложности: easy / medium / hard / multi_turn;
- обнаруженные проблемы: пропущенные доказательства, неверные числа, потеря контекста,
  невалидные citations, превышение бюджета;
- таблицу тестов с фильтрами по suite/status/difficulty;
- список `problem_tests` с причинами.

Значения карточек не сворачиваются в один `Agent Quality = 87%`.

### 6.5. Детали теста

Для выбранного теста показываются:

- исходный вопрос и эталон;
- фактический ответ;
- оценки по пяти критериям (если есть judge/manual);
- найденные документы и chunks;
- gold evidence и покрытие;
- все поисковые запросы и фильтры;
- этапы dense/BM25/RRF/rerank/context с рангами;
- tool trace;
- причины провала и critical errors;
- расчёты;
- warnings.

Это позволяет понять не только «тест провален», но и **почему**.

### 6.6. Markdown-отчёт

`report.md` — краткая версия для PR/CI:

```text
Run: ... | Dataset: ... | Git: ...
Pass rate: ...
Retrieval Recall@10: ... | Evidence Coverage: ...
Answer automatic pass: ...
Multi-turn success: ...
Critical errors: ...
Problems: ...
```

### 6.7. Сравнение запусков

`python -m eval.report --compare run_a run_b` создаёт `comparison.html`/`comparison.md`
и показывает:

| Метрика | До | После | Δ |
|---|---|---|---|
| Easy pass rate | ... | ... | ... |
| Medium pass rate | ... | ... | ... |
| Hard pass rate | ... | ... | ... |
| Retrieval Recall@10 | ... | ... | ... |
| Multi-turn success | ... | ... | ... |
| Mean latency | ... | ... | ... |

Правила:

- Сравнение допустимо только при совпадении `dataset.sha256`; иначе — явный warning.
- Показываются не только средние, но и тесты, изменившие статус `pass`/`fail`.
- Ухудшение простых тестов при улучшении сложных видно отдельно.
- Не выбирается «лучший» прогон; сравниваются все зафиксированные метрики.

## 7. Runner и команды

### 7.1. Режимы

| Режим | Состав | Когда использовать |
|---|---|---|
| `smoke` | 12 тестов: 4 easy, 4 medium, 2 hard, 2 multi_turn | После изменения retrieval, reranking, prompt или agent loop |
| `full` | Все 30 тестов, включая holdout | Перед фиксацией изменений и сравнением версий |
| `easy`/`medium`/`hard`/`multi_turn` | Один suite | Локальная диагностика |
| `full --split holdout` | Только holdout | Перед релизом; не для настройки |

`full` по умолчанию включает holdout и поэтому не должен использоваться для ежедневной
настройки. Для итераций применяйте `smoke` или `full --split dev`; holdout открывается
перед релизом и не меняется по результатам итераций.

### 7.2. Команды

```powershell
# Проверить датасет без вызова агента
.venv\Scripts\python.exe -m eval.runner --validate-only

# Быстрый прогон
.venv\Scripts\python.exe -m eval.runner --suite smoke

# Полный прогон
.venv\Scripts\python.exe -m eval.runner --suite full

# Только holdout
.venv\Scripts\python.exe -m eval.runner --suite full --split holdout

# Отдельный retrieval-режим
.venv\Scripts\python.exe -m eval.runner --suite smoke --retrieval-mode dense

# Повтор нестабильных сценариев
.venv\Scripts\python.exe -m eval.runner --suite smoke --repeat 3

# С LLM-судьёй (опционально)
.venv\Scripts\python.exe -m eval.runner --suite full --judge

# Отчёт по последнему прогону
.venv\Scripts\python.exe -m eval.report --latest

# Сравнение двух прогонов
.venv\Scripts\python.exe -m eval.report --compare run_a run_b
```

`python -m eval.run_eval` сохраняется как тонкий alias для совместимости с
`spec_ui_eval.md`; канонический запуск — `eval.runner`.

`eval.runner` обязан предоставлять программный вход `run_suite(...)`, а не только
CLI. WebUI/API вызывает его в фоновом worker'е, передаёт callback прогресса и
сигнал отмены; CLI является тонкой обёрткой над тем же `run_suite(...)`. Парсить
stdout субпроцесса вместо вызова функции запрещено.

Минимальный контракт:

```python
def run_suite(
    *,
    suite: str,
    split: str = "all",
    ids: list[str] | None = None,
    retrieval_mode: str = "hybrid",
    judge: bool = False,
    repeat: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Any | None = None,
    service: Any | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Запускает suite, пишет артефакты и возвращает каталог run'а."""
```

`progress_callback` получает события `run_started`, `test_started`,
`test_completed`, `run_completed`, `run_failed`; для `test_completed` передаются
`test_id`, `status`, `pass` и текущий progress. `service` позволяет передать уже
созданный/сериализованный `AgentService` (например, `ExecutionGate`-обёртку).

### 7.3. Флаги

| Флаг | Назначение |
|---|---|
| `--suite` | `smoke`, `full`, `easy`, `medium`, `hard`, `multi_turn` |
| `--split` | `dev`, `holdout`, `all` (по умолчанию `all`) |
| `--ids` | Список ID через запятую |
| `--tag` | Фильтр по тегу, например `smoke` |
| `--limit` | Ограничить число тестов |
| `--retrieval-mode` | `hybrid`, `dense`, `keyword`, `all`; `all` гоняет каждый тест отдельно в каждом режиме и сохраняет `by_mode` |
| `--repeat` | Повторы; по умолчанию 1 |
| `--judge` / `--no-judge` | Включить/выключить судью; по умолчанию выключен (smoke должен быть дешёвым; `--no-judge` принимается для совместимости) |
| `--fresh-service` | Новый `AgentService` на каждый тест |
| `--validate-only` | Только валидация датасета и chunk store |
| `--out` | Каталог результатов |
| `--fail-under` | Опциональный порог pass rate для CI |
| `--timeout` | Лимит на turn, секунды |

### 7.4. Коды возврата

| Код | Значение |
|---|---|
| 0 | Прогон завершён; тесты могут быть провалены |
| 2 | Ошибка датасета/конфигурации |
| 3 | Ошибка runner/окружения |
| 4 | `--fail-under` не достигнут |

Провал теста — это результат оценки, а не сбой runner. Поэтому код 0 при наличии
failed-тестов допустим; CI-гейт включается явным `--fail-under`.

### 7.5. Требования к окружению

- Live-прогон требует `LLM_API_KEY`/`LLM_BASE_URL` и собранный chunk store.
- Offline-тесты benchmark используют fake `AgentService`/fake client и не ходят в сеть.
- Секреты не попадают в артефакты, консоль и HTML.
- При отсутствии ключа runner завершается понятной ошибкой, а не пустым отчётом.

## 8. Воспроизводимость

### 8.1. Что фиксируется

| Параметр | Источник |
|---|---|
| Git SHA + dirty | `git rev-parse HEAD`, `git status --porcelain` |
| Dataset hash | SHA-256 всех suite-файлов в фиксированном порядке, канонический JSON |
| Corpus fingerprint | Число чанков, doc_id, source hashes/index build hash |
| Retrieval config | mode, top_k, candidates, rerank_pool, graph on/off |
| LLM | model id, temperature, top_p, max_tokens |
| Reranker | model id, режим fallback |
| Judge | model id, prompt version/hash, temperature |
| Python/deps | Python version и версии ключевых пакетов |
| Время | start/finish, длительность |

### 8.2. Нестабильность

- Основной результат — `pass_at_1` на одном запуске.
- Для нестабильных сценариев допускается `--repeat N`; в отчёте показываются все
  повторы, `stability` и разброс метрик.
- Нельзя выбирать лучший повтор как итог.
- LLM-судья и reranker могут быть недетерминированы; это фиксируется, а не скрывается.
- Повторные запуски не должны автоматически менять эталон.

### 8.3. Сравнение версий

Для корректного сравнения:

- одинаковый `dataset.sha256`;
- одинаковый набор тестов и split;
- зафиксированные модели и параметры;
- одинаковый корпус;
- сохранённые конфиги обоих прогонов.

Если датасет или корпус изменились, сравнение помечается `not_comparable`, а не
выдаётся как регрессия.

Целевые значения agent quality не назначаются заранее: сначала фиксируется baseline
первого полного прогона, затем `--fail-under` и release gates обсуждаются отдельно.
Нельзя подгонять порог задним числом под полученный результат.

## 9. План внедрения

### Этап 1. Эталонный набор

- 30 тестов: 10 easy, 10 medium, 5 hard, 5 multi_turn.
- Для каждого — expected facts, years, documents, gold evidence, behavior, checks.
- 6 holdout-тестов; smoke-тег на 12 тестов.
- Валидатор датасета и проверка существования всех chunk_id.
- Exit criteria: `--validate-only` проходит; все значения заполнены; holdout
  зафиксирован.

### Этап 2. Runner и trace

- `eval/dataset.py`, `eval/runner.py`, `eval/trace.py`.
- Запуск через реальный `AgentService.ask()`.
- `AgentRunResult.diagnostics` и этапы retrieval без изменения логики.
- Артефакты `summary.json`, `cases.jsonl`, `traces/`.
- Exit criteria: smoke-прогон создаёт все файлы; trace содержит этапы и ранги.

### Этап 3. Evaluators

- Retrieval: Recall@5/10, MRR, Evidence/Context Coverage, loss attribution.
- Answer: факты, годы, citations, расчёты, abstention, critical errors.
- Dialogue: сценарии, контекст, correction, contradiction.
- Judge — опционально, после детерминированных проверок.
- Exit criteria: метрики проверены ручными примерами; multi-turn сценарий
  выполняется в одном `conversation_id`.

### Этап 4. Отчёт и сравнение

- `eval/report.py`: HTML и Markdown, детали теста, problem list.
- `--compare`: дельты и изменения статусов.
- `manual_review.json` и отображение ручных оценок.
- README и команды запуска.
- Exit criteria: отчёт открывается локально, содержит цепочку
  `Question → Query Plan → Search Queries → Retrieved Chunks → Evidence → Answer`;
  сравнение двух прогонов работает.

### 9.1. Карта изменений

| Файл | Тип | Работа |
|---|---|---|
| `eval/dataset.py` | новый | загрузка, схема, валидация, hash |
| `eval/runner.py` | новый | запуск suite, turn'ов, артефакты |
| `eval/trace.py` | новый | нормализация trace, redaction |
| `eval/metrics.py` | новый | формулы retrieval и числовые helpers |
| `eval/evaluators/retrieval.py` | новый | Recall/MRR/Coverage/loss attribution |
| `eval/evaluators/answer.py` | новый | факты, годы, citations, расчёты, critical errors |
| `eval/evaluators/dialogue.py` | новый | сценарные проверки |
| `eval/evaluators/judge.py` | новый, опционально | LLM-судья |
| `eval/report.py` | новый | HTML/Markdown/compare |
| `eval/datasets/*.json` | новый | 30 эталонных тестов |
| `eval/run_eval.py` | новый | alias для `spec_ui_eval.md` |
| `src/retrieval/hybrid_retriever.py` | изменение | опциональные stage lists в trace |
| `src/retrieval/temporal_search.py` | изменение | проброс stage lists |
| `src/retrieval/graph_hybrid.py` | изменение | проброс/добавление graph stage |
| `src/agent/state.py` | изменение | `AgentRunResult.diagnostics` |
| `src/agent/service.py` | изменение | заполнение diagnostics из state |
| `tests/test_eval_*.py` | новый | схема, метрики, trace, evaluators, runner, report |
| `README.md` | изменение | команды benchmark |
| `.gitignore` | изменение | `eval/results/*`, кроме `.gitkeep` |

### 9.2. Порядок и зависимости

1. Dataset и validator можно делать параллельно с trace-контрактом.
2. Runner зависит от `diagnostics`; без trace он всё равно запускает тесты, но
   retrieval-метрики помечаются `not_measured`.
3. Answer/dialogue evaluators зависят от dataset, но не от отчёта.
4. Report зависит от `summary.json` и `cases.jsonl`; его можно разрабатывать на
   фикстурах.
5. Judge подключается последним и не блокирует первые три этапа.

## 10. Acceptance criteria

### Dataset

- **AC-B01.** `--validate-only` проверяет все четыре файла, уникальность ID,
  `split`, `suite`, обязательные поля и отсутствие `null` в числовых значениях.
- **AC-B02.** Все `chunk_id` из `gold_evidence` существуют в chunk store;
  фантазийные ID и несовпадающие `doc_id`/`year` дают понятную ошибку.
- **AC-B03.** Набор содержит 30 тестов в пропорции 10/10/5/5, ≥6 holdout и 12 smoke.
- **AC-B04.** Каждый answerable-тест имеет непустой `gold_evidence` и `min_coverage`;
  каждый abstain-тест — причину отказа.
- **AC-B05.** Датасет не содержит `vague`-значений вроде `TBD`, `null`, пустых unit;
  `reference_answer` не участвует в exact match.

### Trace

- **AC-B06.** Для каждого `search_reports` в trace есть `retrieval_mode`, `top_k`,
  filters, counts и stage lists с `chunk_id`/rank для доступных этапов.
- **AC-B07.** Если этап недоступен, в trace стоит `available=false`/`null`, а
  evaluator выдаёт `not_measured`, не `0`.
- **AC-B08.** `evidence_manifest` и `context_chunk_ids` заполнены; при отсутствии
  второго используется первый с `context_source="evidence_manifest"`.
- **AC-B09.** Trace не содержит ключей, заголовков, полного prompt и полного текста
  корпуса; размер ограничен и обрезка помечена `truncated=true`.
- **AC-B10.** Добавление `AgentRunResult.diagnostics` не ломает существующие вызовы и
  тесты; значение по умолчанию — пустой dict.

### Metrics and evaluators

- **AC-B11.** `Recall@5`, `Recall@10`, `MRR`, `Evidence Coverage` проверены
  ручными примерами, включая alternative groups и пустой gold set.
- **AC-B12.** Loss attribution различает `never_retrieved`, `lost_at_fusion`,
  `lost_at_rerank`, `lost_at_context`, `not_cited`.
- **AC-B13.** Answer evaluator детерминированно проверяет status, факты, годы,
  citations, расчёты, abstention и forbidden values; числовое сравнение использует
  tolerance и unit aliases.
- **AC-B14.** Critical errors не компенсируются rubric score; тест с critical error
  всегда `pass=false`.
- **AC-B15.** `dense`, `keyword` и `hybrid` запускаются раздельно; неподдержанный
  режим даёт `unsupported_mode`/`skipped`, а не тихий hybrid. Это зависит от P1
  `spec_agent_enhancement.md`; до его реализации benchmark обязан честно показывать
  `unsupported_mode`, а не выдавать hybrid за dense/keyword.

### Dialogue

- **AC-B16.** Multi-turn сценарий использует один `conversation_id` и выполняет
  turn'ы последовательно.
- **AC-B17.** Проверяются `context_preserved`, `correct_metric`, `correct_period`,
  `no_unnecessary_restart`, `sources_preserved`; correction и contradiction
  сценарии имеют отдельные проверки.
- **AC-B18.** При провале turn'а последующие проверки не считаются успешными
  автоматически (`cascade_failure`).

### Report and reproducibility

- **AC-B19.** Один прогон создаёт `summary.json`, `cases.jsonl`, `retrieval.csv`,
  `traces/*.json`, `report.html`, `report.md`; при judge — `judge.jsonl`.
- **AC-B20.** `summary.json` содержит четыре независимые группы, breakdown по
  difficulty/category/split, coverage, critical errors и problem tests.
- **AC-B21.** HTML-отчёт самодостаточен, открывается локально без сети и показывает
  цепочку `Question → Query Plan → Search Queries → Retrieved Chunks → Evidence → Answer`.
- **AC-B22.** `--compare` показывает дельты и изменения статусов; разные dataset hash
  дают `not_comparable`, а не ложную регрессию.
- **AC-B23.** Ошибка/таймаут/неподдержанный режим видны в coverage и не превращаются
  в нулевую оценку.
- **AC-B24.** Повторный прогон с тем же датасетом/конфигом создаёт сопоставимые
  артефакты; нестабильность отражается через `--repeat`, а не скрывается.

### Safety and scope

- **AC-B25.** Benchmark не меняет ответы, фильтры, prompt и бюджеты агента; при
  отключённом benchmark `AgentService` работает как раньше.
- **AC-B26.** В артефактах нет API-ключей, полного prompt и полного текста корпуса.
- **AC-B27.** Offline-тесты benchmark не требуют сети и ключей; live-прогон требует
  явного окружения.
- **AC-B28.** README содержит единые команды smoke/full/validate/report/compare.

## 11. Тестовая матрица

| Уровень | Что проверяется | Где | Блокирует merge |
|---|---|---|---|
| Dataset schema | поля, уникальность, holdout, null, chunk_id | `tests/test_eval_dataset.py` | да |
| Metrics | Recall@k, MRR, Coverage, alternative groups | `tests/test_eval_metrics.py` | да |
| Trace | нормализация, stage lists, redaction, truncation | `tests/test_eval_trace.py` | да |
| Answer | факты, годы, citations, расчёты, critical errors | `tests/test_eval_answer.py` | да |
| Dialogue | контекст, correction, contradiction, cascade | `tests/test_eval_dialogue.py` | да |
| Runner | smoke на fake AgentService, артефакты, coverage | `tests/test_eval_runner.py` | да |
| Report | HTML/MD, summary, compare, пустые значения | `tests/test_eval_report.py` | да |
| Agent diagnostics | `AgentRunResult.diagnostics` не ломает старые тесты | `tests/test_agent_service.py` | да |
| Live smoke | реальный `AgentService`, 12 тестов | `eval.runner --suite smoke` | нет (manual/nightly) |
| Live full | 30 тестов + holdout | `eval.runner --suite full` | нет (перед релизом) |

Все unit/integration-тесты — offline: fake service, fake client, без сети и ключей.
Live-прогоны не входят в обычный `pytest`.

## 12. Риски и меры

| Риск | Проявление | Мера |
|---|---|---|
| Маленький набор | Высокая дисперсия pass rate | 30 тестов, breakdown, не заявлять статистическую значимость |
| Переобучение на dev | Завышенные метрики | 20% holdout, не менять его после результатов |
| Ошибка эталона | Агент «проваливает» верный ответ | Валидатор, второе ревью hard/holdout, `dataset_error` |
| LLM-судья смещён | Красивый ответ получает высокий балл | Судья вторичен, deterministic checks, калибровка, ручная выборка |
| Недетерминированный reranker | Разные результаты повторов | `--repeat`, stability, фиксация модели и параметров |
| Trace слишком большой | Медленный отчёт | Только ID/ранги/scores, лимит 5 МБ, `truncated` |
| Режим не поддержан | Метрики режима фиктивны | `unsupported_mode`, не подменять hybrid |
| Стоимость/latency | Долгий full-прогон | smoke 12, full 30, judge опционален |
| Секреты в артефактах | Утечка ключей/prompt | Redaction, offline contract-тесты, `.gitignore` |
| Benchmark меняет поведение | Агент оптимизируется под тест | Наблюдаемость вместо управления, эталон не передаётся агенту |

## 13. Definition of Done

- [ ] 30 эталонных тестов с проверенными facts/gold evidence; 6 holdout, 12 smoke.
- [ ] Валидатор датасета проходит; `null`/фантазийные chunk_id запрещены.
- [ ] Runner запускает реальный `AgentService` и сохраняет trace.
- [ ] Trace содержит query plan, search calls, этапы dense/BM25/RRF/rerank/context,
  evidence и ответ без секретов и chain-of-thought.
- [ ] Retrieval/Answer/Dialogue evaluators работают и покрыты offline-тестами.
- [ ] Critical errors отделены от rubric score.
- [ ] Multi-turn сценарии сохраняют контекст, correction и contradiction проверяются.
- [ ] `summary.json`, `cases.jsonl`, `traces/`, `report.html`, `report.md` создаются
  одним прогоном.
- [ ] `--compare` показывает дельты и не сравнивает разные датасеты как регрессию.
- [ ] Ошибки, таймауты и `unsupported_mode` видны в coverage, не равны нулю.
- [ ] Benchmark не меняет рабочую логику Agent_RAG; старые тесты проходят.
- [ ] README содержит команды validate/smoke/full/report/compare.
- [ ] В артефактах нет ключей, полного prompt и полного текста корпуса.
- [ ] WebUI-раздел Evaluation принимается отдельно по `docs/specs/evaluation-webui.md`;
  core benchmark не зависит от наличия UI.

## 14. Что сознательно не делается

- Не внедряется LangSmith или отдельная платформа оценки.
- Не создаётся отдельный LLM-агент-критик.
- Не оцениваются все ответы только другой LLM.
- Не вводится сложная система весов и единственный `Agent Quality`.
- Не смешиваются unit-тесты, интеграционные тесты и benchmark.
- Не обновляются эталонные ответы автоматически после прогона.
- Не сравнивается exact match текста ответа.
- Не добавляются синтетические этапы retrieval, если backend их не предоставляет.
- Не меняются prompt, фильтры или бюджеты агента ради прохождения benchmark.
- Не сохраняются секреты, полный prompt и полный текст корпуса в trace.

## 15. Главный критерий

Рекомендуемый старт — 30 эталонных тестов, четыре независимые группы оценок и один
Python-runner. Расширение до 50+ тестов и дополнительные evaluator'ы добавляются
только после фиксации первого baseline.

Benchmark считается полезным, если после изменения retrieval, reranker, ActionSelector
или prompt можно ответить на три вопроса:

1. Какие тесты улучшились и какие ухудшились?
2. На каком этапе потеряно доказательство: поиск, fusion, rerank, контекст, citation
   или генерация?
3. Это изменение агента или ошибка эталона/окружения?

Если отчёт показывает только итоговый процент и не позволяет восстановить цепочку
`Question → Query Plan → Search Queries → Retrieved Chunks → Evidence → Answer`,
он не выполняет цель этой спецификации.
