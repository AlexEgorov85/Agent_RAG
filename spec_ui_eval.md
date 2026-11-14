# Спецификация UI и evaluation

**Версия:** 1.0
**Статус:** целевой контракт; UI и eval pipeline ещё не реализованы.
**Связанные документы:** [spec_target.md](spec_target.md), [spec_agent.md](spec_agent.md), [spec_retrieval_engine.md](spec_retrieval_engine.md), [spec_graph.md](spec_graph.md)

## 1. Streamlit UI

### 1.1. Экран чата

Основной экран — интерфейс к `AgentService` из `spec_agent.md` и содержит:

- историю текущей сессии;
- поле запроса и кнопку отправки;
- выбранные годы/issuer/документы как явные filters;
- streamed ответ и индикатор этапов агента;
- наблюдаемый tool trace (инструмент, применённые фильтры, количество результатов,
  длительность, статус), без chain-of-thought;
- раскрываемую трассу retrieval: dense/BM25/graph источники кандидатов, RRF, reranker и
  fallback; scores, если показаны, обозначаются как технические ранги, не confidence;
- expandable-блок «Источники» с годом, issuer, разделом, PDF-страницей и цитатой;
- предупреждения о пропущенном годе, несопоставимом issuer, пустом разделе или fallback;
- видимый статус `answer`/`abstain`.

Citation открывает карточку исходного chunk и позволяет увидеть весь текст/HTML таблицы.
Нельзя показывать citation, которого нет в `Answer.citations` или который не разрешается
в сохранённом chunk store.

### 1.2. Структура и граф

Структура отчёта выбирается по документу и отображается деревом из `DocumentStructure`.
Клик по узлу выставляет `section_path` filter и/или показывает chunks раздела.

Graph view показывает только ограниченный подграф текущего запроса/документа. Для узлов
видны тип, имя и год/issuer; для ребра — тип и evidence. Клик по evidence открывает
соответствующий citation/chunk. Отсутствие graph index должно скрывать/отключать этот
вид с понятным сообщением, но не ломать чат.

### 1.3. Временная визуализация

Timeline строится только по фактам, имеющим `period` и citation. Отсутствующий год
показывается как «нет отчёта», а не как нулевое значение. График не соединяет точки через
разрыв 2021 без явного визуального обозначения пропуска. 2015 маркируется отдельным
issuer и не включается в линию сравнения по умолчанию.

### 1.4. Сессия и ошибки

Session state содержит messages, выбранные filters и `conversation_id`; backend agent
держит изолированное состояние сессии согласно `spec_agent.md`. UI state не содержит
API secrets. Кнопка очистки стирает состояние текущей сессии. Ошибки API отображаются
кратко, техническая трассировка остаётся в логах. При недоступности streaming UI
получает итоговый `AgentRunResult`, не отключая чат.

## 2. Evaluation dataset

Файл `eval/dataset.jsonl`, одна запись на строку:

```json
{
  "id": "temporal_001",
  "split": "dev",
  "category": "temporal",
  "question": "Как менялся ...?",
  "filters": {},
  "expected_status": "answer",
  "expected_facts": [],
  "reference_facts": [],
  "relevance": {"sber_2023:00042:c0:...": 2},
  "expected_doc_ids": ["sber_2023"],
  "expected_sections": ["..."],
  "missing_years": []
}
```

`relevance` оценивает chunk relevance: 0 нерелевантен, 1 частично полезен, 2 прямо
подтверждает ответ. Для каждого вопроса с `expected_status=answer` обязателен хотя бы один
релевантный evidence chunk. Для abstain-вопросов relevance может быть пустым, но причина
отказа должна быть записана.

Минимальный dev set — 50 вопросов:

- 15 factoid;
- 10 structural;
- 10 temporal (включая период с отсутствующим годом);
- 10 multi-hop;
- 5 abstain/out-of-scope.

Вопросы покрывают разные годы и issuer. Dev set используется при разработке и не
представляется независимым тестом качества. Перед релизом нужно выделить небольшой
holdout-набор, не меняемый по результатам каждой итерации.

## 3. Метрики

### 3.1. Retrieval

Для каждого вопроса с evidence рассчитываются `P@k`, `Recall@k`, `RR@k`, `AP@k` и
`nDCG@k`; основной отчёт — `Recall@5`, `MRR@5` (mean RR), `nDCG@5`. Учитываются только
размеченные evidence chunks. В отчёте обязательны:

- macro mean по вопросам;
- breakdown по category и годам;
- число вопросов с пустой выдачей;
- сравнение dense-only, BM25-only, hybrid без rerank, hybrid с rerank, graph-off/on;
- параметры retrieval и fingerprints индексов.

Формулы должны быть закреплены тестами на ручных примерах; особенно convention AP@k и
нормировка nDCG при неполной выдаче.

### 3.2. Ответ и citations

Автоматические проверки:

- citation validity: доля citations, существующих в chunk store;
- citation coverage: доля ответов, где каждое проверяемое утверждение поддержано хотя бы
  одной citation;
- temporal accuracy: правильная привязка фактов к годам и отсутствие подмены пропуска;
- structure accuracy: доля citations из ожидаемого раздела для structural вопросов;
- abstain accuracy: корректность отказов и ответов на отсутствующие данные.

LLM-judge оценивает `faithfulness`, `answer_relevance`, `context_relevance`,
`answer_correctness` по шкале 0–2. Judge получает вопрос, эталон, answer, citations и
retrieved context; он не должен видеть ожидаемый system behavior, не связанный с кейсом.
Промпты/модель/версия фиксируются в отчёте. Судья дополняет, но не заменяет
детерминированные проверки.

### 3.3. Targets и трактовка

Целевые значения из проектного дизайна:

```text
Recall@5 >= 0.85
MRR@5    >= 0.70
nDCG@5   >= 0.80
faithfulness mean >= 1.7 / 2
answer_relevance mean >= 1.6 / 2
context_relevance mean >= 1.6 / 2
answer_correctness mean >= 1.6 / 2
temporal accuracy >= 0.80
structure accuracy >= 0.85
```

Значения — целевые release gates, а не заранее заявленный результат. Если датасет или
корпус не позволяет достичь цели, отчёт показывает провалы и сравнение с baseline;
порог нельзя менять задним числом ради успешного статуса.

## 4. Калибровка judge

Подготовить 3 контрольных кейса: корректный ответ, неподтверждённое утверждение и
частично релевантный/неполный ответ. Для каждого вручную задать 4 rubric scores (12
оценок). Проверить понятность rubric и объяснений judge. На маленькой выборке нельзя
заявлять статистическую «согласованность 90%» как надёжную оценку; в отчёте показываются
все 12 пар manual/judge, exact matches и расхождения.

Калибровочные кейсы не должны автоматически попадать в среднее dev score, если они
использовались для изменения judge prompt.

## 5. Артефакты evaluation

Одна команда запускает evaluation и сохраняет:

```text
eval/results/{run_id}/summary.json
eval/results/{run_id}/cases.jsonl
eval/results/{run_id}/retrieval.csv
eval/results/{run_id}/judge.jsonl    # если judge включён
```

`summary.json` включает: run id/time, git revision если доступна, Python/dependency
versions, dataset hash, chunk/index/graph fingerprints, config без секретов, метрики
overall/by-category, ошибки, пропущенные cases и model IDs.

Поддержать `--no-judge` для воспроизводимого локального retrieval eval без внешних LLM
вызовов. Ошибка отдельного кейса записывается и не должна молча превращаться в нулевую
оценку; итог указывает coverage и failures.

## 6. Команды и приёмка

Целевые команды:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m scripts.build_corpus --verify
.venv\Scripts\python.exe -m eval.run_eval --split dev --no-judge
.venv\Scripts\python.exe -m streamlit run src/ui/app.py
```

Конкретный CLI-маршрут можно изменить при реализации, но README и CI используют одну
согласованную команду на задачу.

Приёмка UI/eval:

- smoke UI запускается с mock LLM и без network credentials;
- citations и фильтры работают для реального chunk artifact;
- структура и graph view корректно обрабатывают отсутствующий индекс;
- один eval run создаёт все ожидаемые файлы, включая ошибки/coverage;
- метрики проверены тестовыми примерами;
- manual smoke проверяет factoid, structural, temporal с пропуском 2021 и abstain;
- evaluation сравнивает минимум dense-only и целевой hybrid режим.
