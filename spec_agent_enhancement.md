# Спецификация усиления Agent_RAG

**Версия:** 1.1
**Статус:** целевой контракт поверх реализованного Agent_RAG. Архитектура не переписывается.
**База анализа:** `main` (`f77652a`) + незакоммиченные файлы рабочего дерева.
**Связанные документы:** [spec_agent.md](spec_agent.md), [spec_retrieval_engine.md](spec_retrieval_engine.md), [spec_target.md](spec_target.md), [spec_graph.md](spec_graph.md), [spec_ui_eval.md](spec_ui_eval.md), [spec_benchmark.md](spec_benchmark.md)

> Спецификация составлена по статическому анализу исходников. Тесты в ходе анализа не запускались.
>
> **Ревизия 1.1.** Диагностика усиления сведена в единый
> `AgentRunResult.diagnostics` (контракт — `spec_benchmark.md` §4.4). Agent-level
> benchmark не дублируется: P8 расширяет общий `eval/`, а не создаёт второй набор и
> второй runner.

## 1. Назначение

Связать уже существующие компоненты в один исследовательский контур:

```text
Router → ActionSelector → Hybrid/Graph/Structure retrieval
       → Evidence Coverage → Generation → Verification → Answer
```

Цель — не добавить новые технологии, а устранить разрывы между компонентами и сделать качество результата проверяемым на уровне всего исследования.

### 1.1. Принципы

| № | Принцип | Следствие |
|---|---|---|
| 1 | Один агент, несколько стратегий поиска | Не вводим multi-agent и отдельного Critic Agent |
| 2 | Граф и структура — навигация, а не доказательство | Любое содержательное утверждение подтверждается chunk’ом |
| 3 | LLM извлекает и объясняет, Python считает | Арифметика только в `calculations.py` |
| 4 | Любое усиление деградирует в текущее поведение | Сбой нового компонента не ломает baseline retrieval |
| 5 | Контракт важнее компонента | `top_k`, режим поиска и coverage не могут молча игнорироваться |
| 6 | Качество измеряется на наборе вопросов | Eval входит в Definition of Done, а не в «потом» |

### 1.2. Не цели

- Новые БД, очереди, сервисы, векторные хранилища.
- Переписывание ingestion/PDF-парсера.
- Полноценный table-extraction engine.
- Автоматическое расширение графа новыми типами узлов.
- Изменение `Answer`-контракта несовместимым образом.
- Замена LangGraph или `AgentService` на другую оркестрацию.

## 2. Исходное состояние

### 2.1. Что уже работает

- `HybridRetriever`: dense + BM25 → RRF → LLM reranker, жёсткие фильтры до top-k.
- `PerYearSearch`: веерный поиск по годам (в рабочем дереве, подключён в `factory.py`).
- `SimpleRouter`: детерминированный период из `available_years` (в рабочем дереве, подключён).
- `GraphAugmentedRetriever`: графовые свидетельства поверх базового поиска (используется в eval, не в фабрике).
- `AgentTools` + `build_agent_graph`: белый список инструментов, бюджеты, `get_entities`, `get_chunk`.
- `AnswerGenerator`: JSON-схема, проверка `chunk_id` по manifest, retry, `abstain`.
- `calculations.py`: детерминированные абсолютные/процентные изменения и CAGR (не подключён).
- `structure_compare.py`: детерминированное сопоставление разделов (не подключён).
- `eval_retrieval.py`: hit, MRR, recall, labeled precision на dev-корпусе.

### 2.2. Подтверждённые разрывы

| ID | Факт | Где | Последствие |
|---|---|---|---|
| G1 | При наличии `backend.retrieve` теряются `top_k` и `retrieval_mode` | `src/agent/tools.py`, `AgentTools.invoke` | Контракт инструмента обещает больше, чем выполняется |
| G2 | `HybridRetriever.retrieve` не имеет `retrieval_mode` | `src/retrieval/hybrid_retriever.py` | Нельзя реально переключить dense/keyword/hybrid |
| G3 | `PerYearSearch.retrieve` не принимает `top_k`/`retrieval_mode`; ранние выходы игнорируют параметры | `src/retrieval/temporal_search.py` | Веерный поиск нельзя ограничить или переключить |
| G4 | `_DefaultSelector` всегда делает `hybrid`, `top_k=5`, не читает `QueryPlan` | `src/agent/agent_graph.py` | Router фактически не управляет стратегией |
| G5 | Фабрика не передаёт `action_selector` | `src/agent/factory.py` | Правила выбора действий отсутствуют |
| G6 | `assess_evidence` = `bool(evidence_manifest)`; `after_assess` не читает статус | `src/agent/agent_graph.py` | Один чанк считается достаточным evidence |
| G7 | Поля `expected_years`, `covered_years`, `missing_years`, `search_attempts`, `available_documents` объявлены, но не заполняются | `src/agent/state.py` | Полнота периода не контролируется |
| G8 | `RetrievalTrace` не доходит до агента и UI; нет счётчиков разделов/документов | `src/retrieval/hybrid_retriever.py` | Непонятно, почему поиск не нашёл нужное |
| G9 | `needs_rewriting` — только флаг; перефразирования нет | `src/agent/simple_router.py`, retriever | Сложные вопросы ищутся одной формулировкой |
| G10 | Нет выбора разнообразных результатов после RRF/rerank | `src/retrieval/hybrid_retriever.py` | Top-k может состоять из почти дублей одного раздела |
| G11 | `facts` без обязательных `metric/year/table`-координат; `calculations.py` не вызывается | `src/engine/generator.py`, `src/agent/service.py` | Числа не проверяются и не считаются детерминированно |
| G12 | `structure_compare.py` не подключён; `structural_changes` всегда пуст | `src/engine/structure_compare.py`, `src/agent/service.py` | Структурные изменения не используются |
| G13 | Граф передаётся только как `entity_backend`; `GraphAugmentedRetriever` не в фабрике | `src/agent/factory.py` | Graph RAG не участвует в стандартном поиске |
| G14 | Проверка ответа = membership цитат, без проверки утверждений, чисел, периодов и причинности | `src/agent/service.py`, `AgentService._valid_answer` | Валидный `chunk_id` не гарантирует поддержку утверждения |
| G15 | Eval только retrieval; нет agent-level метрик и nDCG | `scripts/eval_retrieval.py`, `data/eval/` | Изменения агента оцениваются субъективно |

### 2.3. Что уже готово, но не подключено

| Файл | Состояние | Действие в этой спецификации |
|---|---|---|
| `src/agent/simple_router.py` | В рабочем дереве, подключён в фабрике | P2: расширить аспектами/метриками, зафиксировать тестами |
| `src/retrieval/temporal_search.py` | В рабочем дереве, подключён | P1: привести контракт к `top_k`/`retrieval_mode`/`queries` |
| `src/engine/calculations.py` | В рабочем дереве, не подключён | P6: подключить к `AgentService`, ужесточить факты |
| `src/engine/structure_compare.py` | В рабочем дереве, не подключён | P5: обернуть инструментом `compare_structures` |
| Поля `AgentState` для coverage/structure | Объявлены, не используются | P3/P5: заполнять существующие поля, не плодить новые |

## 3. Границы ответственности

| Модуль | Отвечает за | Не должен |
|---|---|---|
| `simple_router.py` | Период, intent, аспекты, required metrics, `needs_graph` | Искать, генерировать факты, незаметно менять явные фильтры |
| `action_selector.py` (новый) | Выбор одного следующего действия по state и coverage | Выполнять поиск, писать ответ, обходить бюджеты |
| `AgentTools` | Валидация аргументов, маршрутизация в backend, серверные лимиты | Менять смысл запроса, добавлять чанки, скрывать потерю параметров |
| `HybridRetriever` | Каналы, RRF, rerank, diversity, trace | Знать про LangGraph, coverage и генерацию ответа |
| `PerYearSearch` | Веер по годам и слияние без дублей | Подменять годы, добавлять evidence |
| `coverage.py` (новый) | Матрица «показатель × год», gaps, статус полноты | Вызывать LLM, придумывать значения, считать причинность |
| `calculations.py` | Арифметика по `facts` | Извлекать факты из текста, доверять числам без `chunk_id` |
| `structure_compare.py` | Сопоставление деревьев разделов | Считать diff доказательством содержания раздела |
| `graph_hybrid.py` / `graph_retriever.py` | Кандидаты и evidence-пути графа | Добавлять graph-only чанк в цитаты без `get_chunk`/поиска |
| `generator.py` | Синтез строго по sources | Вычислять проценты/CAGR, добавлять внешние знания |
| `verification.py` (новый) | Проверка цитат, чисел, периодов, полноты, причинности | Быть отдельным агентом или ходить в сеть |
| `AgentService` | Сборка state, один repair-цикл, итоговый результат | Дублировать логику retriever/coverage/verification |
| `ui/app.py` | Отображение coverage, calculations, verification, trace | Показывать скрытые рассуждения и сырые секреты |

## 4. Общие контракты

### 4.1. SearchFilters

Контракт не меняется. Явные фильтры UI/API имеют приоритет над планом router:

- `years`, `year_from/year_to`, `issuers`, `kinds`, `allowed_doc_ids`;
- `section_path_prefix`, `breadcrumb_contains`, `max_section_level`;
- `normalized()` и `requested_years()` остаются единственной точкой нормализации.

### 4.2. QueryPlan

Расширение только с defaults, обратная совместимость сохраняется:

```python
@dataclass(frozen=True)
class QueryPlan:
    intent: Literal["factoid", "structural", "temporal", "multi_hop"] = "factoid"
    years: list[int] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    section_filter: str | None = None
    entities: list[str] = field(default_factory=list)
    needs_rewriting: bool = False
    needs_graph: bool = False
    answer_mode: Literal["concise", "comparison", "outline"] = "concise"
    # Новое:
    aspects: list[str] = field(default_factory=list)            # ≤ 4 поисковых аспекта
    required_metrics: list[str] = field(default_factory=list)   # ≤ 5 канонических metric id
    requires_comparison: bool = False
```

Правила валидации в `_as_plan`:

- `aspects` и `required_metrics` — списки непустых строк, обрезаются до лимитов;
- неизвестный metric id отбрасывается, а не превращает весь план в fallback;
- невалидный тип любого нового поля → безопасный `QueryPlan()`, как сейчас.

### 4.3. Вход `search_reports`

```text
query: str
queries: list[str] | None      # до 3 формулировок, original всегда первый
filters: dict | SearchFilters
retrieval_mode: hybrid | dense | keyword
use_graph: bool = False
top_k: int = 5
```

Правила:

- `queries` дедуплицируются casefold, original сохраняется первым;
- `top_k` ограничивается серверным `AgentTools.top_k`;
- при `use_graph=True` используется `graph_search_backend`, иначе — `search_backend`;
- если graph backend недоступен: обычный поиск + warning `graph_unavailable`, без падения tool call;
- backend, не поддерживающий запрошенный параметр, получает явную ошибку `backend_does_not_support_mode`, а не молчаливое игнорирование.

### 4.4. RetrievalTrace

Дополнительные поля (остальные сохраняются):

```text
retrieval_mode: str
query_count: int
dense_candidates: int
bm25_candidates: int
diversity_removed: int
distinct_docs: int
distinct_sections: int
distinct_years: int
```

`RetrievalResult` не меняет форму; trace передаётся в `ToolEvent.diagnostics` и в
`AgentRunResult.diagnostics["retrieval_calls"]` (единый контейнер, §4.9).

### 4.5. Facts

Целевой контракт числового факта:

```json
{
  "metric": "net_profit",
  "metric_label": "Чистая прибыль",
  "year": 2023,
  "value": "1 508,6",
  "unit": "млрд руб.",
  "scale": "billion",
  "chunk_id": "sber_2023:00078:c3:c5cc6e1f46e2",
  "table_id": "table_45",
  "row_label": "Чистая прибыль",
  "column_label": "2023"
}
```

Правила нормализации:

- legacy-поля `claim`, `period`, `value_num`, `amount` принимаются, но приводятся к контракту;
- числовой факт участвует в расчётах только при наличии `metric`, `year` и `chunk_id` из citations; иначе он сохраняется как справочный, но в `calculations` не попадает;
- `year` не подставляется из метаданных чанка: отсутствие года означает «не участвует в расчётах»;
- для чанка `kind="table"` обязательны `row_label` и `column_label`; они должны встречаться в `html`/`text` этого чанка;
- фактический `value` должен встречаться в цитируемом чанке после нормализации числа;
- качественные утверждения остаются в тексте ответа и не попадают в `facts`.

### 4.6. CoverageReport

```python
@dataclass
class CoverageCell:
    metric: str
    year: int | None
    status: Literal["covered", "not_found", "year_absent"]
    chunk_ids: list[str]

@dataclass
class CoverageReport:
    status: Literal["not_required", "complete", "partial", "insufficient"]
    expected_years: list[int]
    covered_years: list[int]
    missing_years: list[int]                 # года нет в корпусе
    metrics: dict[str, dict[str, str]]       # metric -> year|"*" -> status
    missing_metrics: list[str]
    missing_metric_years: list[dict]
    distinct_docs: int
    distinct_sections: int
    notes: list[str]
```

Правила статуса:

- `not_required` — нет требуемых лет и метрик (простой factoid);
- `insufficient` — manifest пуст или ни одна требуемая ячейка не покрыта;
- `complete` — все требуемые ячейки `covered` или `year_absent`;
- `partial` — остальное.

`year_absent` означает ограничение корпуса, а не ошибку поиска. `not_found` означает «год есть, показатель не найден».

### 4.7. VerificationResult

```python
@dataclass
class VerificationIssue:
    code: str
    severity: Literal["fatal", "warning"]
    message: str
    chunk_id: str | None = None
    fact_index: int | None = None

@dataclass
class VerificationResult:
    ok: bool
    issues: list[VerificationIssue]
    supported_citations: list[str]
    unsupported_facts: list[int]
```

Коды:

| Код | Severity | Значение |
|---|---|---|
| `citation_outside_manifest` | fatal | цитата вне manifest |
| `fact_without_citation` | fatal | факт не ссылается на возвращённую цитату |
| `fact_value_not_in_source` | fatal (для факта) | значение не найдено в цитируемом чанке |
| `fact_year_mismatch` | fatal (для факта) | год факта противоречит источнику |
| `table_anchor_missing` | warning | нет `row_label`/`column_label` для таблицы |
| `quote_not_found` | warning | `quote` не является фрагментом чанка |
| `gap_not_disclosed` | warning | в ответе не назван известный gap |
| `numeric_claim_unverified` | warning | число в тексте не связано с фактом/цитатой |
| `causal_claim_unverified` | warning | причинный вывод при наличии только совместного упоминания |

### 4.8. Конфигурация

Добавляется в `RAGSettings` с безопасными defaults:

| Параметр | Default | Смысл |
|---|---|---|
| `max_queries` | 3 | original + до 2 перефразирований |
| `diversity_enabled` | True | включить soft-diversity |
| `max_per_doc` | 2 | мягкий лимит чанков одного документа |
| `max_per_section` | 1 | мягкий лимит одного раздела |
| `coverage_enabled` | True | контроль полноты |
| `max_coverage_searches` | 1 | дополнительный поиск по gaps |
| `verification_enabled` | True | пост-проверка ответа |
| `structural_max_docs` | 5 | максимум документов для diff |

### 4.9. Контейнер диагностики

Канонический контракт `AgentRunResult.diagnostics` определён в
[spec_benchmark.md](spec_benchmark.md) §4.4. Эта спецификация не вводит параллельных
полей результата; она заполняет следующие ключи:

| Ключ | Заполняется в |
|---|---|
| `query_plan` | P2 |
| `search_attempts` | P3 |
| `retrieval_calls` | P1, P4 |
| `evidence_manifest` / `context_chunk_ids` | P1 |
| `coverage` | P3 |
| `verification` | P8 |
| `structural_changes` | P5 |

Правила:

- `AgentRunResult` получает одно поле `diagnostics: dict[str, Any]` с
  `default_factory=dict`; отдельные `coverage`, `verification`, `retrieval_calls`
  в результате не появляются;
- `AgentRunResult.structural_changes` и `AgentRunResult.calculations` остаются
  совместимыми зеркалами для UI и тестов; они заполняются из тех же объектов, а не
  пересчитываются;
- `retrieval_diagnostics` из рабочего дерева — deprecated alias на
  `diagnostics["retrieval_calls"]`; удаляется после миграции потребителей;
- если компонент ещё не реализован, ключ равен `null`, а не выдуманному значению.

## 5. Работы по приоритетам

### P1. Привести контракт поиска в соответствие реализации

**Цель:** `top_k`, `retrieval_mode` и `queries` реально доходят до retriever; каждый режим использует только свои каналы.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/retrieval/hybrid_retriever.py` | `retrieve(..., retrieval_mode="hybrid", queries=None)`; реальное переключение каналов; счётчики trace |
| `src/retrieval/temporal_search.py` | `retrieve(..., top_k=None, retrieval_mode="hybrid", queries=None)`; проброс во все ветки, включая early return |
| `src/agent/tools.py` | `SearchBackend` protocol; проверка поддержки kwargs; `graph_search_backend`; `queries`; `use_graph` |
| `src/agent/agent_graph.py` | `SearchInput`: `queries`, `use_graph`; defaults из tools, а не хардкод |
| `src/agent/factory.py` | `AgentTools(top_k=settings.top_k, document_catalog=...)` |
| `tests/test_hybrid_retriever.py`, `tests/test_temporal_search.py`, новый `tests/test_search_contract.py` | Контрактные тесты |

**Алгоритм:**

1. `hybrid`: оба канала, прежнее поведение и fallback.
2. `dense`: BM25 не вызывается; embedder вызывается; ранжирование по dense.
3. `keyword`: embedder не вызывается; только BM25.
4. Ошибка единственного выбранного канала → пустой результат + warning; тихий переход в другой канал запрещён.
5. `queries`: каждый канал выполняется по каждой формулировке; списки сливаются RRF; дубли по `chunk_id` схлопываются; `dense_rank`/`bm25_rank` — лучший ранг по формулировкам.
6. `PerYearSearch`: внутренние вызовы получают `retrieval_mode`/`queries`; итоговый merged список обрезается до `top_k`.

**Границы:** `HybridRetriever` не знает про `use_graph`; `AgentTools` не переписывает запрос, а только валидирует и передаёт.

**Приёмка:**

- AC-P1-1: `hybrid` вызывает оба канала; `dense` не вызывает BM25; `keyword` не вызывает embedder.
- AC-P1-2: spy-backend получает `top_k` и `retrieval_mode` без потерь.
- AC-P1-3: backend без поддержки `dense` при `retrieval_mode="dense"` даёт явную ошибку, а не hybrid.
- AC-P1-4: `PerYearSearch` при 5 годах и `top_k=3` возвращает ≤3 хитов.
- AC-P1-5: defaults не меняют текущее поведение (hybrid, top_k=5, reranker включён).

---

### P2. Завершить Router и ActionSelector, зафиксировать фильтры

**Цель:** Router определяет не только период, но и аспекты/показатели/потребность в графе; селектор выбирает стратегию, а не всегда `hybrid`.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/core/metrics.py` (новый) | Канонические метрики и лексикон словоформ |
| `src/agent/state.py` | Расширение `QueryPlan` (см. §4.2) |
| `src/agent/simple_router.py` | `required_metrics`, `aspects`, `requires_comparison`, `needs_graph`, `intent=multi_hop` |
| `src/agent/action_selector.py` (новый) | `RuleBasedActionSelector` |
| `src/agent/factory.py` | Передать `action_selector` |
| `src/agent/agent_graph.py` | `_as_plan` для новых полей; `_DefaultSelector` оставить fallback |
| `tests/test_simple_router.py`, новый `tests/test_action_selector.py` | Правила и fallback |

**Лексикон (минимум):**

```text
net_profit   → чистая прибыль, прибыль
revenue      → выручка, доходы
assets       → активы
equity       → капитал, собственный капитал
roe          → roe, рентабельность капитала
roa          → roa, рентабельность активов
nim          → nim, чистая процентная маржа
opex         → операционные расходы, расходы
cost_of_risk → стоимость риска
cet1         → cet1, достаточность капитала
```

**Правила `needs_graph`:** ключевые слова связи и причинности: «связ», «влия», «фактор», «почему», «причин», «зависит», «взаимосвяз». Для простого «какая прибыль» граф не включается.

**Правила `intent`:**

- `structural` — «раздел», «структура», «оглавление», «новые направления», «изменилась структура»; `answer_mode="outline"`;
- `multi_hop` — вопрос о связи/факторах и ≥2 метриках;
- `temporal` — период, динамика, сравнение лет; `answer_mode="comparison"` при >1 годе;
- иначе `factoid`, `answer_mode="concise"`.

**Правила селектора (базовые, расширяются в P3/P5/P7):**

1. Нет поиска → `search_reports`; `queries = [question] + plan.aspects` только при `needs_rewriting`, `requires_comparison` или ≥2 метриках, иначе `[question]`; `retrieval_mode="hybrid"`, `use_graph=plan.needs_graph`.
2. Есть evidence, coverage неполный, бюджет есть → точечный поиск по отсутствующим показателям/годам.
3. Coverage достаточен → `None` (переход к генерации).
4. Любое неизвестное действие → безопасный `search_reports`, как сейчас.

**Фильтры:** `_filters_with_plan` уже сужает только незаданные пользователем поля. Требуется зафиксировать тестами:

- explicit `years={2023}` + plan `2020..2024` → `{2023}`;
- explicit `year_from=2022` + plan `year_to=2023` → `2022..2023`;
- конфликт границ → пустой диапазон, а не расширение;
- explicit `section_path_prefix` побеждает `section_filter`.

**Приёмка:**

- AC-P2-1: «Сравни прибыль, активы и капитал за 2020–2024» → `required_metrics` содержит три метрики, `requires_comparison=True`.
- AC-P2-2: «Какая прибыль в 2024?» → `needs_graph=False`, один аспект.
- AC-P2-3: «Какие факторы повлияли на прибыль?» → `needs_graph=True`.
- AC-P2-4: явные фильтры не переопределяются router’ом (4 теста выше).
- AC-P2-5: невалидный plan → безопасный factoid, явные фильтры сохранены.

---

### P3. Evidence Coverage и контроль полноты

**Цель:** заменить `bool(evidence_manifest)` на проверку покрытия запроса.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/core/metrics.py` | `match_metrics(text) -> list[str]`, `metric_labels(ids)` |
| `src/agent/coverage.py` (новый) | `build_coverage(...)`, `CoverageReport`, `to_dict()` |
| `src/agent/state.py` | Заполнять существующие `expected_years`, `covered_years`, `missing_years`, `search_attempts`; добавить `coverage`, `coverage_status`, `coverage_searches_used` |
| `src/agent/agent_graph.py` | `assess_evidence` и `after_assess` используют coverage |
| `src/agent/service.py` | Инициализация state, передача gaps в генератор, `diagnostics["coverage"]` |
| `src/engine/generator.py`, `src/engine/prompts.py` | `coverage` в prompt как `evidence_gaps` |
| `src/ui/app.py` | Таблица покрытия и gaps |
| `tests/test_coverage.py` (новый), `tests/test_temporal_agent.py` | Матрица и маршрутизация |

**Алгоритм:**

1. `expected_years = filters.requested_years()`; если пусто — год не требуется.
2. `required_metrics = plan.required_metrics or match_metrics(question)`.
3. Если годы не заданы, матрица строится по годам, реально присутствующим в evidence; статус зависит только от покрытия метрик, а не от полноты лет.
4. Для каждого чанка: год берётся только из metadata; метрики ищутся в `text` + `html` по лексикону.
5. `covered` — есть чанк с этим годом и упоминанием метрики.
6. `year_absent` — год в `RetrievalResult.missing_years`.
7. `not_found` — год есть в корпусе, но чанка с метрикой нет.
8. `assess_evidence`: `complete`/`not_required` → генерация; `partial` → ещё один поиск, если бюджет; `insufficient` → `AgentService` не вызывает генератор и возвращает `abstain` с причиной («год отсутствует в корпусе» / «показатель не найден»).
9. `after_assess` больше не игнорирует статус: переход к генерации при `complete`/`not_required`/`insufficient`, к селектору — только при `partial` и наличии бюджета.

**Границы:** coverage не вызывает LLM и не проверяет истинность утверждений; это проверка полноты найденных свидетельств. Формулировка «показатель не найден» и «год отсутствует в корпусе» обязана различаться.

**Приёмка:**

- AC-P3-1: 3 метрики × 2020–2024 при отсутствии 2021 и капитала за 2021 даёт матрицу из отчёта, `status=partial`.
- AC-P3-2: один чанк по одному показателю не закрывает multi-metric вопрос.
- AC-P3-3: partial → ровно один дополнительный поиск, затем ответ с перечислением gaps.
- AC-P3-4: `missing_years` содержит 2021 и note «нет в корпусе», отдельно от `not_found`.
- AC-P3-5: budget exhausted → нет цикла, ответ/abstain корректен.
- AC-P3-6: простой factoid → `complete`/`not_required`, поведение не ухудшается.
- AC-P3-7: генератор получает `evidence_gaps` (spy на prompt).

---

### P4. Query rewriting, разнообразие и диагностика retrieval

**Цель:** 2–4 формулировки на сложный вопрос и отсутствие «пяти почти одинаковых чанков».

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/retrieval/hybrid_retriever.py` | `queries`, RRF по спискам «канал × формулировка» |
| `src/retrieval/diversity.py` (новый) | `select_diverse(hits, limit, ...)` |
| `src/retrieval/temporal_search.py` | Проброс `queries` |
| `src/agent/tools.py`, `src/agent/agent_graph.py` | Валидация и передача `queries` |
| `src/agent/state.py` | `ToolEvent.diagnostics`; `AgentRunResult.diagnostics["retrieval_calls"]`; использовать `search_attempts` |
| `src/agent/service.py` | Заполнение diagnostics в `ToolEvent` и итоговом результате |
| `src/ui/app.py` | Счётчики каналов, документов, разделов, лет |
| `tests/test_diversity.py` (новый), `tests/test_hybrid_retriever.py` | Разнообразие и multi-query |

**Diversity — мягкие ограничения:**

1. Первый хит всегда сохраняется.
2. Первый проход: по одному хиту на год, если вопрос сравнительный.
3. Второй проход: лимиты `max_per_doc` и `max_per_section`.
4. Третий проход: добор в исходном порядке, если лимиты не позволили заполнить `top_k`.

Жёсткая фильтрация запрещена: diversity не должна снижать recall на простых вопросах.

**Диагностика:** `dense_candidates`, `bm25_candidates`, `fused_candidates`, `rerank_candidates`, `returned`, `distinct_docs/sections/years`, `diversity_removed`.

**Бюджет формулировок:** 1 для простого factoid; до 3 для comparison/multi-metric; жёсткий cap `max_queries`. LLM-перефразирование не обязательно: контракт `queries` позволяет подключить его позже без изменения tool/retriever.

**Приёмка:**

- AC-P4-1: 6 хитов из одного документа/раздела → в top-k появляются другие документы, первый хит сохранён.
- AC-P4-2: при нехватке разнообразных хитов список добирается исходными, без потери recall.
- AC-P4-3: 3 формулировки дают один fused список без дублей; trace.query_count=3.
- AC-P4-4: отключение diversity флагом возвращает прежний порядок.
- AC-P4-5: reranker получает исходный вопрос, а не перефразирование.

---

### P5. Структурное сравнение документов

**Цель:** подключить существующий `structure_compare.py` как навигационный инструмент.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/agent/tools.py` | `StructureCompareBackend`, tool `compare_structures`, `document_catalog` |
| `src/agent/factory.py` | Backend поверх `*_sections.json`; year из payload/doc_id |
| `src/agent/agent_graph.py` | Схема `compare_structures`, allowlist, сбор `structural_changes` |
| `src/agent/state.py` | Заполнять `structure_documents`, `section_matches`, `structural_changes`, `structure_comparison_complete` |
| `src/agent/service.py` | `AgentRunResult.structural_changes` и `diagnostics["structural_changes"]` |
| `src/ui/app.py` | Блок «Изменения структуры» |
| `tests/test_structure_compare.py` (новый), `tests/test_agent_factory.py` | Diff и доступность |

**Контракт инструмента:**

```text
input:  doc_ids: list[str] (2..5), section_path: str | None, max_depth: int
output: documents, section_matches, structural_changes, warnings
```

**Правила:**

- <2 документов или нет артефактов → `ToolUnavailable("structure_unavailable")`.
- По умолчанию LLM-верификация выключена; сравнение детерминированное.
- `structural_changes` — навигация. Если ответ утверждает переименование/перенос, ему нужны evidence-чанки; иначе warning `structural_claim_without_citation`.
- Селектор вызывает `compare_structures` для `intent="structural"` при ≥2 доступных документах; `reason_code="clarify_structure"`, новые reason codes не вводятся.

**Приёмка:**

- AC-P5-1: переименованный раздел 2022→2023 определяется как `renamed`.
- AC-P5-2: перенесённый раздел — `restructured`; добавленный/удалённый — `added`/`removed`.
- AC-P5-3: результат детерминирован без LLM.
- AC-P5-4: `AgentRunResult.structural_changes` не пуст.
- AC-P5-5: <2 документов → tool_unavailable, чат продолжается.

---

### P6. Числовые факты и детерминированные расчёты

**Цель:** LLM не считает; Python считает по проверенным фактам.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/engine/prompts.py` | Новый facts-контракт, запрет арифметики, правила таблиц |
| `src/engine/generator.py` | Нормализация и проверка фактов; drop негрунтованных фактов с warning |
| `src/engine/calculations.py` | Конфликты фактов; группы по metric+unit; округление |
| `src/agent/service.py` | `calculations` в `Answer` и `AgentRunResult`; сохранить при `_answer_from` |
| `src/agent/state.py` | `calculations` уже объявлены — использовать |
| `src/ui/app.py` | Блок «Расчёт по подтверждённым фактам» |
| `tests/test_calculations.py` (новый), `tests/test_answer_generator.py` | Числа, единицы, таблицы |

**Правила генератора:**

1. Числовой факт без `metric`/`year` не участвует в расчётах.
2. `value` обязан парситься `parse_number`; иначе факт качественный и в `facts` не попадает.
3. Для `kind="table"` проверяются `row_label` и `column_label` в `html`/`text`; при провале факт отбрасывается, ответ не становится `abstain`.
4. `chunk_id` факта обязан быть среди возвращённых citations (существующее правило).
5. Максимум 50 фактов; лишние отбрасываются с warning.

**Расчёты:**

- группировка по `(metric.casefold(), unit.casefold())`; разные единицы не смешиваются;
- абсолютное изменение, процент, CAGR, min/max;
- <2 различных лет → расчёт не строится;
- одинаковый `metric+year` с разными значениями → `fact_conflict`, группа исключается из сводки;
- расчёт не является фактом из отчёта и помечается в UI.

**Приёмка:**

- AC-P6-1: «1 508,6», «(1 234)», «12,5%» парсятся корректно.
- AC-P6-2: CAGR считается только для положительных значений и span>0.
- AC-P6-3: разные единицы не объединяются.
- AC-P6-4: конфликт года даёт warning и не попадает в summary.
- AC-P6-5: таблица с неверной строкой → факт отброшен, `table_anchor_missing`.
- AC-P6-6: сервис возвращает непустые `calculations` для ответа с 3 годами.
- AC-P6-7: `_answer_from` не теряет `calculations`.

---

### P7. Маршрутизация Graph RAG

**Цель:** граф помогает находить связи, но не заменяет первичные свидетельства.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/agent/tools.py` | `graph_search_backend`, `use_graph`, fallback с warning |
| `src/retrieval/graph_hybrid.py` | `retrieve(..., queries=None, retrieval_mode="hybrid")` — проброс в base; graph остаётся hybrid-only |
| `src/agent/factory.py` | Один раз загрузить граф; собрать `GraphAugmentedRetriever` + `PerYearSearch`; передать оба backend |
| `src/agent/action_selector.py` | `use_graph=plan.needs_graph`; правило для `get_entities` |
| `src/agent/agent_graph.py` | Проброс `use_graph`; диагностика graph в tool event |
| `src/agent/state.py` | `graph_evidence_manifest` уже есть — использовать |
| `tests/test_graph_routing.py` (новый), `tests/test_agent_factory.py` | Маршрутизация и fallback |

**Таблица маршрутизации:**

| Вопрос | Основной механизм |
|---|---|
| Какая прибыль в 2024? | hybrid |
| Как менялась прибыль? | hybrid + PerYearSearch |
| Какие факторы повлияли на прибыль? | hybrid + `use_graph=True` |
| Как связаны инвестиции, технологии и расходы? | graph search + primary chunks |
| Какие направления появились/исчезли? | `compare_structures` + graph/hybrid |

**Правила evidence:**

- `get_entities` добавляет только `graph_evidence_manifest`; graph-only чанк не попадает в `evidence` и не может быть цитирован.
- Чтобы сослаться на связь, агент обязан получить чанк через поиск или `get_chunk` (authorization уже разрешает ID из graph manifest).
- `get_entities` вызывается селектором только при непустых `plan.entities` и вопросе о связях; после него обязателен `get_chunk` или поиск по evidence ids.
- `get_entities` — опциональный путь. Если router не выделил сущности, relation-вопрос обслуживается через `use_graph=True`, без отдельного NER в агенте.
- `graph_added=0` — валидный результат, не ошибка.
- Graph backend недоступен → hybrid + `graph_unavailable`; tool call считается completed.
- При `retrieval_mode="dense"`/`"keyword"` graph-обёртка отключается с warning: граф работает только поверх hybrid-базы.

**Приёмка:**

- AC-P7-1: `needs_graph=True` → используется graph backend; иначе hybrid.
- AC-P7-2: graph artifact отсутствует → hybrid + warning, ответ формируется.
- AC-P7-3: graph-only chunk нельзя цитировать до `get_chunk`.
- AC-P7-4: путь `get_entities → get_chunk → answer` проходит, цитата валидна.
- AC-P7-5: `graph_added` и matched entities видны в diagnostics.

---

### P8. Проверка ответа и полноценная оценка качества

**Цель:** проверять не только `chunk_id`, но и утверждение, число, период, полноту и причинность; измерять всё на наборе вопросов.

**Файлы:**

| Файл | Изменение |
|---|---|
| `src/engine/verification.py` (новый) | `verify_answer(...)`, issue codes |
| `src/agent/service.py` | Verification → один repair → sanitize facts → calculations |
| `src/agent/state.py` | `diagnostics["verification"]` в `AgentRunResult` |
| `src/agent/agent_graph.py` | `validate_answer` остаётся schema/citation-валидатором; verification в графе не дублируется |
| `src/ui/app.py` | Блок «Проверка ответа» |
| `scripts/eval_retrieval.py` | nDCG@K |
| `eval/runner.py` | использовать общий benchmark; второй agent-level eval не создаётся |
| `eval/datasets/*.json` | добавить типы задач P8 в общий набор |
| `tests/test_verification.py` | Проверки verification |

**Проверки `verify_answer`:**

1. Цитаты: manifest membership — fatal; `quote`-подстрока — warning.
2. Факты: `value` и `year` присутствуют в цитируемом чанке — fatal для факта (drop), не для всего ответа.
3. Таблицы: `row_label`/`column_label` присутствуют в источнике.
4. Полнота: известные gaps должны быть названы в ответе; иначе `gap_not_disclosed`.
5. Числа в тексте: `%`, «млрд», «млн» без соответствующего факта/цитаты → `numeric_claim_unverified`.
6. Причинность: маркеры «привело к», «в результате», «из-за», «влияет» при отсутствии подтверждённой причинной связи → `causal_claim_unverified` (warning, не блокирует).

**Политика:**

- Fatal по цитатам → один repair с diagnostics; повторный провал → `abstain`.
- Fatal по отдельному факту → факт удаляется, расчёт по нему не строится, ответ сохраняется.
- Warnings → ответ сохраняется, warnings показываются в UI.
- Отдельный Critic Agent не вводится.

**Eval-набор:**

Второй agent-level eval не создаётся. Набор, схема, runner и отчёт определены в
`spec_benchmark.md`; P8 добавляет в `eval/datasets/*.json` типы задач ниже и использует
общий `eval.runner`. Отдельный `agent_questions.json` или `scripts/eval_agent.py` не
вводится.

Типы задач (минимум 2 вопроса на тип):

| Тип | Что проверяется |
|---|---|
| `fact` | найден и корректен нужный факт |
| `temporal_comparison` | правильное сопоставление лет |
| `numeric_comparison` | извлечение значений и расчёт изменений |
| `structural_comparison` | переименованные/перемещённые разделы |
| `multihop` | подтверждённые связи |
| `completeness` | все обязательные элементы найдены |
| `citation` | цитаты подтверждают утверждения |
| `abstention` | корректный отказ при отсутствии данных |

**Метрики benchmark:**

| Уровень | Метрика |
|---|---|
| Retrieval | hit@K, MRR, recall@K, nDCG@K, покрытие документов/разделов |
| Agent answer | fact_accuracy, answer_completeness, citation_precision, citation_recall, calculation_accuracy, abstention_accuracy, status_accuracy |
| Operations | tool_calls, latency, warnings, coverage_status |

`citation_precision` считается детерминированным верификатором, а не LLM-судьёй.
Трасса сохраняется в `eval/results/{run_id}/traces/` по контракту
`spec_benchmark.md` §4: запросы, режимы, filters, tool events, coverage, verification,
calculations, warnings. Секреты и полный prompt не сохраняются.

**Приёмка:**

- AC-P8-1: цитата с несуществующим `quote` даёт warning, а не падение.
- AC-P8-2: значение 2022, приписанное 2023, даёт `fact_year_mismatch`, расчёт не строится.
- AC-P8-3: причинный вывод при совместном упоминании даёт warning.
- AC-P8-4: пропущенный обязательный показатель даёт `gap_not_disclosed` или `incomplete_period`.
- AC-P8-5: repair вызывается не более одного раза.
- AC-P8-6: `eval.runner --suite smoke` использует общий набор и не требует второго eval-скрипта.
- AC-P8-7: nDCG@K покрыт unit-тестом.
- AC-P8-8: отсутствие ключа API не ломает offline unit-тесты; live eval запускается только с env.

## 6. Порядок внедрения

| Шаг | Работа | Зависит от | Масштаб | Результат |
|---|---|---|---|---|
| 0 | P1: контракт поиска | — | небольшой | `top_k`/режимы реально работают |
| 1 | P2: Router + ActionSelector | P1 | небольшой/средний | стратегия выбирается по плану |
| 2 | P3: Evidence Coverage | P2 | средний | полнота проверяется, gaps видны |
| 3 | P4: rewriting + diversity + diagnostics | P1, P3 | небольшой/средний | сложные вопросы ищутся шире и разнообразнее |
| 4 | P5: structural comparison | P2, P3 | средний | diff разделов подключён |
| 5 | P6: numeric facts + calculations | P3 | средний | числа проверяются и считаются Python |
| 6 | P7: graph routing | P2, P4 | небольшой | граф используется по назначению |
| 7 | P8: verification + eval | P3–P7 | средний | качество измеряется и не регрессирует |

Порядок совпадает с приоритетами отчёта, кроме одного изменения: контракт поиска (P1) вынесен перед Router/ActionSelector, потому что без него новые режимы и `queries` не доходят до retriever.

Eval-набор начинать на P3: coverage-тесты сразу дают первые эталонные вопросы. P8 только автоматизирует и расширяет его.

## 7. Тестовая матрица

| Уровень | Что проверяется | Где | Блокирует merge |
|---|---|---|---|
| Contract | `top_k`, режимы, `queries`, `use_graph` доходят до backend | `tests/test_search_contract.py` | да |
| Unit | RRF, rerank fallback, diversity, coverage, calculations, verification | новые и расширенные тесты | да |
| Graph | маршрутизация, fallback, graph-only citation | `tests/test_graph_routing.py` | да |
| Agent integration | simple factoid, multi-hop, partial coverage, abstain, budget | `tests/test_agent_service.py`, `tests/test_temporal_agent.py` | да |
| Factory | сборка backend’ов, каталог документов, отсутствие graph | `tests/test_agent_factory.py` | да |
| Generator | facts-схема, таблицы, legacy-поля, drop факта | `tests/test_answer_generator.py` | да |
| UI helpers | безопасные warnings, trace, coverage-рендер | `tests/test_ui_helpers.py` | да |
| Eval schema | все gold `chunk_id` существуют в корпусе | `tests/test_eval_dataset.py` | да |
| Live eval | факты, цитаты, полнота, abstention | `eval.runner --suite smoke` | нет (nightly/manual) |

Все unit/integration-тесты офлайновые: fake client, fake embedder, без сети и ключей.

## 8. Definition of Done

- [ ] P1: ни один вызов `search_reports` не теряет `top_k`/режим.
- [ ] P2: Router и ActionSelector покрыты тестами; явные фильтры неприкосновенны.
- [ ] P3: coverage-матрица строится; `assess_evidence` не сводится к `bool(manifest)`.
- [ ] P4: multi-query и diversity включаются флагами и не ухудшают baseline.
- [ ] P5: `compare_structures` доступен, детерминирован, отображается в UI.
- [ ] P6: `calculations` заполняются сервисом; числа не считает LLM.
- [ ] P7: graph routing и fallback работают; graph-only чанк нельзя цитировать.
- [ ] P8: verification и agent-level eval работают; baseline зафиксирован.
- [ ] Все новые поля имеют defaults; старые вызовы не ломаются.
- [ ] `README.md` и связанные `spec_*.md` обновлены ссылками на эту спецификацию.
- [ ] Ни в логах, ни в событиях, ни в eval-артефактах нет ключей и полного prompt.

## 9. Риски и откаты

| Риск | Проявление | Мера |
|---|---|---|
| Coverage-эвристики дают ложные gaps | лишние предупреждения | статус `partial` не блокирует ответ; матрица видна пользователю; флаг отключения |
| Diversity снижает recall | потеря релевантного чанка | мягкие лимиты и добор; сравнение с baseline в eval |
| Graph-поиск дороже по latency | рост времени ответа | включать только при `needs_graph`; лимит лет и `graph_max_results`; fallback |
| Строгие facts ломают старые ответы | факт отброшен | legacy-поля нормализуются; drop факта, не ответа |
| Structure compare ошибочно склеивает разделы | неверный diff | порог + детерминированный режим; LLM-верификация выключена по умолчанию |
| Eval переобучен на dev | завышенные метрики | отдельный holdout; не тюнить на всём наборе |
| Новые поля ломают checkpoints | ошибка сериализации | хранить coverage/verification как dict, не dataclass |

## 10. Сводная таблица изменений по файлам

| Файл | Тип | Работа |
|---|---|---|
| `src/core/metrics.py` | новый | лексикон и канонические метрики |
| `src/agent/metrics.py` | не создавать | дублирование лексикона запрещено |
| `src/agent/simple_router.py` | изменение | аспекты, метрики, graph-флаги |
| `src/agent/action_selector.py` | новый | правила выбора действий |
| `src/agent/coverage.py` | новый | матрица полноты |
| `src/agent/tools.py` | изменение | контракт поиска, graph backend, compare_structures, каталог |
| `src/agent/agent_graph.py` | изменение | схемы, coverage-gate, новый tool, diagnostics |
| `src/agent/state.py` | изменение | QueryPlan, coverage/verification/diagnostics |
| `src/agent/factory.py` | изменение | selector, graph backend, settings, catalog |
| `src/agent/service.py` | изменение | coverage, calculations, verification, результат |
| `src/retrieval/hybrid_retriever.py` | изменение | режимы, multi-query, diversity, trace |
| `src/retrieval/diversity.py` | новый | soft-diversity |
| `src/retrieval/temporal_search.py` | изменение | проброс контракта |
| `src/retrieval/graph_hybrid.py` | изменение | проброс `queries`/`retrieval_mode` в base; graph только поверх hybrid |
| `src/engine/generator.py` | изменение | facts-контракт, drop негрунтованных фактов |
| `src/engine/prompts.py` | изменение | правила чисел/таблиц/расчётов/gaps |
| `src/engine/calculations.py` | изменение | конфликты, округление, подключение |
| `src/engine/structure_compare.py` | использование | без переписывания |
| `src/engine/verification.py` | новый | проверка утверждений и чисел |
| `src/ui/app.py` | изменение | coverage, calculations, verification, trace |
| `scripts/eval_retrieval.py` | изменение | nDCG@K |
| `eval/runner.py`, `eval/evaluators/answer.py` | использование | общий benchmark; второй agent-level eval не создаётся |
| `eval/datasets/*.json` | изменение | типы задач P8 в общем наборе |
| `tests/*` | новые/расширение | см. §7 |
| `README.md` | изменение | новые возможности и запуск eval |

## 11. Главный критерий

Усиление считается успешным, если на одном и том же наборе вопросов:

1. не выросло число неподтверждённых числовых утверждений;
2. выросла полнота ответов на multi-metric и temporal-вопросах;
3. citation precision не ниже baseline;
4. abstention остаётся корректным;
5. число tool calls и latency не выросли без измеримой пользы.

Если изменение не улучшает ни одну метрику и не устраняет подтверждённый разрыв G1–G15, оно не входит в релиз.
