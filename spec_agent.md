# Спецификация агента и его WebUI

**Версия:** 1.0
**Статус:** целевой контракт; агент и WebUI ещё предстоит реализовать.
**Связанные документы:** [spec_target.md](spec_target.md), [spec_retrieval_engine.md](spec_retrieval_engine.md), [spec_graph.md](spec_graph.md), [spec_ui_eval.md](spec_ui_eval.md)

## 1. Назначение

Пользователь работает через WebUI с ограниченным агентом-исследователем отчётов.
Агент понимает тип вопроса, выбирает один или несколько разрешённых инструментов,
собирает подтверждения, а затем возвращает ответ в общем `Answer`-контракте.

Агент не является автономным исполнителем произвольных действий. Он не выполняет код,
не обращается к произвольным URL/файлам, не изменяет корпус и не делает записи в
системы вне проекта. Основные поисковые и генеративные операции делегируются
`RAGEngine`/retrieval-компонентам из профильных спецификаций.

## 2. Архитектура агента

Реализация — ограниченный граф состояний на LangGraph. Переходы и бюджет вызовов
детерминированы; LLM может выбирать только из валидированных действий и инструментов.

```text
START
  → normalize_request
  → analyze_query
  → choose_action
       ├─ search_reports ──────┐
       ├─ get_structure_tree ──┤
       ├─ get_entities ────────┤
       └─ get_chunk ───────────┘
  → assess_evidence
       ├─ enough / no evidence → generate_answer
       └─ insufficient and budget available → choose_action
  → validate_answer
       ├─ valid → END
       └─ one repair attempt → validate_answer → END/abstain
```

Simple factoid-вопрос может пройти через один `search_reports` вызов. Multi-hop запрос
может получить структуру/сущности, затем искать соответствующие chunks. Graph tool может
быть отключён; это не мешает ответам без графа.

### 2.1. AgentState

Минимальное состояние одного запуска:

```text
request_id: str
conversation_id: str
question: str
filters: SearchFilters
query_plan: QueryPlan | None
action: AgentAction | None
tool_events: list[ToolEvent]
retrieval_results: list[RetrievalResult]
evidence_manifest: set[str]          # только chunk_id из фактически полученных результатов
answer: Answer | None
status: queued | running | completed | abstained | failed
tool_calls_used: int
answer_repairs_used: int
warnings: list[str]
```

В state не сохраняются API keys. `conversation_id` изолирует историю одного пользователя.
Для MVP допускается `InMemorySaver`: перезапуск сервера может завершить сессии, но не
повреждает корпус. Персистентная память — отдельное решение и не должна неявно включать
retrieved document text без политики хранения.

## 3. Разрешённые инструменты

Все tools имеют типизированные входы/выходы и валидируют фильтры до исполнения.
Инструменты возвращают структурированные данные, а не произвольный исполняемый текст.

### `search_reports`

```text
input: query, filters, retrieval_mode (hybrid|dense|keyword), top_k
output: list[Hit], missing_years, warnings, retrieval_trace_id
```

- Делегирует retrieval engine; не вызывает генерацию ответа.
- `top_k` ограничен серверной конфигурацией.
- Применяет все явные фильтры и возвращает только hits с валидным `chunk_id`.

### `get_structure_tree`

```text
input: doc_id, section_path?, max_depth
output: document metadata, section nodes, matching chunks summary
```

- Читает только уже собранный локальный артефакт структуры.
- `max_depth` ограничивается конфигурацией.
- Не ищет по всей файловой системе и не запускает парсер.

### `get_entities`

```text
input: entity_names, filters, max_hops
output: nodes, typed paths, evidence_chunk_ids, warnings
```

- Использует готовый graph index.
- При отсутствующем graph index возвращает `tool_unavailable`, не ошибку всего чата.
- `max_hops` по умолчанию 2, верхний предел 2 в MVP.
- Каждый путь обязан указывать evidence chunks; голое ребро не является доказательством.

### `get_chunk`

```text
input: chunk_id
output: chunk content and metadata, or not_found
```

- Доступ только по ID из индекса/корпуса.
- Может использоваться агентом для уточнения evidence, но не для обхода каталога.

### 3.1. Бюджеты и защита цикла

- Не более 4 tool calls за один пользовательский turn.
- Не более 2 поисковых вызовов `search_reports` за turn.
- Не более 1 answer-repair попытки после schema/citation validation.
- Граф завершает run при исчерпании бюджета и формирует ответ только по собранным
  evidence либо `abstain`.
- Лимиты задаются сервером; LLM не может их увеличить.
- Ошибка одного необязательного инструмента (graph/structure) не должна обнулять
  успешные результаты поиска.

## 4. Выбор действий и безопасность ответа

`analyze_query` вызывает router из `spec_retrieval_engine.md`. `choose_action` возвращает
строго одну структуру `AgentAction`:

```text
tool_name: enum из разрешённых tools
arguments: schema для выбранного tool
reason_code: fact_lookup | clarify_structure | explore_entities | inspect_evidence
```

Невалидное имя инструмента или аргументы не исполняются. Агент возвращается к безопасному
`search_reports` с исходным запросом либо завершает run с ошибкой/abstain.

`assess_evidence` оценивает только наличие/релевантность полученных chunks и покрытия
вопроса. Он не добавляет внешние факты. Финальный генератор:

- получает `evidence_manifest` и только разрешённые chunks;
- создаёт `Answer` по схеме `spec_target.md`;
- может ссылаться только на chunk IDs из `evidence_manifest`;
- при отсутствии достаточного evidence отвечает `abstain`;
- не показывает внутренние рассуждения модели или скрытый chain-of-thought.

## 5. События для streaming

Backend публикует UI-safe события:

```text
run_started(request_id)
analysis_completed(intent, effective_filters)
tool_started(tool_name, short_arguments)
tool_completed(tool_name, result_count, duration_ms, status)
evidence_updated(citation_summaries)
answer_started()
answer_delta(text)                     # если провайдер поддерживает streaming
run_completed(answer_status, duration_ms)
run_failed(error_code, user_message)
```

События не содержат prompt целиком, секреты, полные внутренние рассуждения или полный
корпусный текст. `short_arguments` очищаются от потенциально длинных/чувствительных
значений. Для повторного подключения фронтенд может запросить итоговый state по
`request_id`; гарантированная доставка каждого промежуточного события не требуется.

## 6. WebUI агента

WebUI — Streamlit application и основной интерфейс работы с агентом. Чат показывает:

- сообщения пользователя и финальный ответ агента;
- этап обработки (анализ запроса, вызов поиска/структуры/графа, сбор evidence);
- список реально вызванных tools, их фильтры, число результатов, длительность и статус;
- раскрываемую трассу RAG-поиска: dense/BM25/graph candidate counts, применение RRF и
  reranker, порядок финальных кандидатов и fallback при сбое компонента;
- найденные evidence/citations с годом, issuer, разделом и страницей;
- явные warnings (missing year, issuer mismatch, graph unavailable, fallback);
- `answer`/`abstain`, а также краткую причину отказа;
- кнопку остановки/сброса текущего run, если backend поддерживает cancellation.

UI отображает **наблюдаемый план и tool trace**, но не скрытое рассуждение модели.
Трасса должна позволять пользователю понять, откуда взялись citations и какие ограничения
поиска были применены.
Сырые численные scores могут быть доступны в диагностическом режиме, но не должны
подаваться как вероятность истинности ответа.

### 6.1. Сессии

- Каждый browser session получает отдельный `conversation_id`.
- Кнопка «Новый диалог» создаёт новую изолированную сессию.
- История помогает разрешать контекстные follow-up вопросы, но каждый новый factual
  claim должен заново подтверждаться evidence; прошлый ответ не считается источником.
- Фильтры документов/годов видимы и могут быть изменены пользователем.
- Перед запуском показываются активные фильтры; router не может незаметно переопределить
  явные filters UI.

### 6.2. Ошибки и деградация

- Если Graph недоступен, UI показывает это в trace, а агент продолжает без graph.
- Если reranker недоступен, UI помечает fallback на RRF.
- Если LLM/API недоступен, UI показывает контролируемую ошибку; неподтверждённый
  partial answer не отображается как завершённый ответ.
- Если event streaming недоступен, UI может дождаться итогового `Answer`, сохраняя
  функциональность чата.
- Ошибки отображаются без traceback; request ID доступен для диагностики.

## 7. Интерфейс backend к UI

```python
class AgentService:
    def stream(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: SearchFilters | None = None,
    ) -> Iterator[AgentEvent]: ...

    def ask(
        self,
        query: str,
        *,
        conversation_id: str,
        filters: SearchFilters | None = None,
    ) -> AgentRunResult: ...
```

`ask()` возвращает `Answer`, `tool_events`, effective filters, warnings, timings и
request ID. `stream()` выдаёт те же данные прогрессивно; итоговый результат должен
совпадать со значением `ask()` при одинаковых зависимостях/config.

## 8. Тесты и приёмка

### Unit/integration

- Проверить граф состояний: simple factoid → один search → answer.
- Multi-hop: структура/graph tool → search → answer с evidence.
- Router/tool JSON невалиден — неизвестный tool не вызывается.
- Tool budget исчерпан — цикла нет, финальный answer/abstain корректен.
- Graph недоступен — успешный fallback в обычный retrieval.
- Generation не может сослаться на chunk вне `evidence_manifest`.
- История другого `conversation_id` недоступна текущей сессии.
- Сбой streaming event не теряет итоговый результат `ask()`.
- Тесты используют fake LLM/tools и работают без сетевых ключей.

### Ручной WebUI smoke

Проверить не менее 4 сценариев: factoid с citations; структурный запрос с видимым
structure tool; временной вопрос, где видно отсутствие 2021; вопрос без данных с
`abstain`. Для каждого проверить, что tool trace соответствует реальным вызовам,
а citations открывают chunks из корпуса.

Агент/WebUI считаются готовыми только вместе с retrieval, corpus artifacts и тестами
этих контрактов; один визуальный прототип без работающего `AgentService` не считается
выполнением спецификации.
