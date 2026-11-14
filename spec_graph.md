# Спецификация Graph RAG

**Версия:** 1.1
**Статус:** graph слой реализован на dev-корпусе (2022–2023); проведён
измеренный graph-on/off eval на multi-hop наборе; в WebUI пока не включён.
**Связанные документы:** [spec_target.md](spec_target.md), [spec_retrieval_engine.md](spec_retrieval_engine.md)

## 1. Цель и ограничения MVP

Граф связывает сущности и свидетельства, чтобы поддерживать multi-hop запросы и
исследование отчёта. Он не заменяет точный dense/BM25 поиск и не является источником
самостоятельно придуманных фактов.

MVP graph:

- граф на `networkx.MultiDiGraph` (один согласованный storage);
- строится из chunks и дерева структуры;
- покрывает все доступные документы, если размер и время сборки позволяют; иначе в
  артефакте точно перечисляется subset документов;
- retrieval возвращает исходные `chunk_id`, а не только имена узлов/сущностей;
- graph retrieval можно выключить без изменения baseline RAG.

LlamaIndex `PropertyGraphIndex` не является обязательной зависимостью MVP. Не хранить
одновременно два конкурирующих графа NetworkX/LlamaIndex без отдельной спецификации
синхронизации и выбора источника истины.

## 2. Схема графа

### 2.1. Типы узлов

```text
DocumentNode
  node_id = doc:{doc_id}
  doc_id, year, report_period, issuer, country, title, source_sha256

SectionNode
  node_id = section:{doc_id}:{stable_section_id}
  doc_id, title, level, section_path, page_start, page_end

ChunkNode
  node_id = chunk:{chunk_id}
  chunk_id, doc_id, year, issuer, kind, page_pdf, section_path

EntityNode
  node_id = entity:{canonical_type}:{normalized_name}
  canonical_name, aliases[], entity_type, extraction_version
```

Разрешённые `entity_type` MVP: `Organization`, `Person`, `FinancialMetric`,
`Risk`, `Strategy`, `Product`, `Technology`, `Other`. Типы ограничены перечислением,
не произвольными строками модели.

### 2.2. Рёбра

```text
DocumentNode -[:HAS_SECTION]-> SectionNode
SectionNode  -[:HAS_CHILD]-> SectionNode
SectionNode  -[:CONTAINS]-> ChunkNode
ChunkNode    -[:MENTIONS]-> EntityNode
EntityNode   -[:RELATED_TO]-> EntityNode
```

Рёбра содержат `source`, `confidence`, `extraction_version`, а для извлечённых из текста
связей — `evidence_chunk_ids`. `HAS_SECTION`, `HAS_CHILD` и `CONTAINS` выводятся из
детерминированных артефактов ingestion и не зависят от LLM.

## 3. Entity extraction и нормализация

Экстрактор принимает один chunk за запрос или пакет ограниченного размера. Возвращает
валидируемый JSON:

```json
{
  "entities": [
    {"name": "…", "type": "Risk", "aliases": []}
  ],
  "relations": [
    {"subject": "…", "predicate": "RELATED_TO", "object": "…"}
  ]
}
```

Требования:

- `name` нормализуется по пробелам/регистру, но исходное написание сохраняется в alias;
- relation допускается только при указании evidence в текущем chunk;
- свободные и неопределённые предикаты сводятся к `RELATED_TO` либо отбрасываются;
- модель не должна добавлять знание из внешнего мира;
- повторная сборка идемпотентна при тех же входах и версии extractor;
- сохраняются ошибки, пропущенные chunks и причины отказа.

В MVP entity extraction строится только по текстовым и табличным chunks. Описания графиков
подключаются, если отдельный visual pipeline уже создал их с provenance. Финансовое
число не считается доверенным фактом только потому, что оно появилось в графе.

### 3.1. Проверка provenance при извлечении

Модель может перепутать `chunk_id` внутри пакета, поэтому извлечённые записи
проверяются по тексту:

- имя сущности должно буквально встречаться в тексте заявленного чанка
  (нормализация без учёта регистра, пробелов и пунктуации);
- если имя найдено ровно в одном другом чанке пакета, `chunk_id` переприсваивается
  этому чанку; если совпадений несколько или нет — сущность отбрасывается;
- evidence связи обязан упоминать subject или object, иначе связь отбрасывается;
- дубликаты сущностей и связей объединяются детерминированно.

Рабочие параметры сборки (dev): 2 чанка на запрос, `max_tokens=4096`,
`max_retries=2`. Пакет из 4 чанков при `max_tokens=2048` давал около 9%
упавших пакетов из-за обрезанного JSON; после перехода на 2 чанка и 4096 токенов
упавшие чанки доливаются скриптом `scripts/patch_graph_failed_chunks.py`.

### 3.2. Дедупликация сущностей

На первом этапе допустима детерминированная нормализация `canonical_type + normalized
name` и таблица aliases. Нечёткое объединение разных имен через embedding/LLM запрещено
без ручной проверки: оно может объединить разные компании/метрики.

## 4. Построение и артефакты

Публичный интерфейс:

```python
build_graph(
    chunks: Sequence[Chunk],
    structures: Mapping[str, DocumentStructure],
    *,
    extractor: EntityExtractor | None,
    config: GraphBuildConfig,
) -> GraphBuildResult
```

Сохранение:

```text
data/graph/{graph_fingerprint}.json
data/graph/{graph_fingerprint}_manifest.json
```

JSON содержит версию схемы и список узлов/рёбер; сериализация не использует pickle.
Fingerprint включает hashes chunks, структур, extraction model/prompt version и параметры.
Частично построенный граф не становится активным артефактом. После ошибок сохраняется
manifest со статусом `incomplete`, не смешиваемый с полным graph.

Manifest:

```text
graph_schema_version, graph_fingerprint, created_at,
doc_ids, chunk_count, entity_count, relation_count,
extractor_name, extractor_version, prompt_version,
failed_chunk_ids, skipped_chunk_ids, status
```

## 5. Graph retrieval

Вход: разрешённые `doc_ids`/годы/issuer, entities, section filter, лимиты глубины и
количества результатов. Выход — ранжированный список chunks с объяснением пути:

```text
chunk_id, graph_score, matched_entities, path, evidence_chunk_ids
```

Правила:

- стартовые сущности должны быть извлечены из запроса или переданы пользователем;
- ограничение по документам/issuer/годам применяется к chunk-узлам до ранжирования;
- максимальная глубина обхода по умолчанию 2; циклы и повторные chunks дедуплицируются;
- relation без evidence не используется для передачи фактологического claim;
- **по умолчанию** кандидатами становятся только chunks, напрямую упоминающие
  запрошенные сущности (`MENTIONS`). Evidence соседних сущностей, найденных по
  `RELATED_TO`, подключается только явным флагом
  (`include_relation_evidence=True`), иначе граф притягивает нерелевантные
  чанки о связанных сущностях;
- graph result обязан содержать исходный chunk для последующей проверки/цитирования;
- при низкой уверенности graph возвращает пустой список, не расширяет ответ;
- graph candidates проходят общий RRF/reranker pipeline согласно retrieval spec;
- **измеренная политика для multi-hop (dev):** базовый пул 20 кандидатов,
  graph-кандидаты до 10, списки чередуются перед реранжированием, порог
  `rerank_score` не применяется (`threshold=0`), итоговый размер ограничивается
  токенным бюджетом 4000–8000, `top_k` остаётся аварийным потолком. Пороги 1–3
  снижали recall, поэтому по умолчанию выключены.

Graph не должен проводить сравнение числовых значений без табличного/text evidence для
каждого периода.

## 6. API для UI

```python
get_graph_subgraph(
    *,
    doc_ids: set[str] | None,
    section_path: str | None,
    entity_ids: set[str] | None,
    max_nodes: int = 200,
) -> GraphView
```

`GraphView` ограничивает payload узлами/рёбрами, которые нужны текущему экрану. UI получает
labels, types, year/issuer и список evidence chunks, но не получает скрытые внутренние
промпты или API credentials.

## 7. Приёмка

- Структурные узлы и рёбра совпадают с проверенными sections/chunks.
- Все `MENTIONS`/`RELATED_TO` рёбра имеют валидные endpoint IDs и provenance.
- Нет dangling `chunk_id` или section IDs.
- Повторная сборка с теми же входами создаёт тот же fingerprint и идентичное содержимое.
- Изменение source hash/prompt version создаёт новый fingerprint.
- Graph retrieval возвращает evidence chunks и подчиняется metadata filters.
- При выключенном/невалидном graph функциональные тесты baseline RAG продолжают проходить.
- Интеграционный eval graph-on vs graph-off сохраняется отдельно; graph не объявляется
  улучшением без измеренного сравнения. На dev-корпусе сравнение проведено — см. §8.

## 8. Измеренный результат на dev-корпусе (14.11.2026)

### 8.1. Артефакт графа

- 536 чанков (`sber_2022` + `sber_2023`), 1134 сущности, 851 связь.
- 1220 связей `MENTIONS`, 851 `RELATED_TO`.
- Проверка provenance: 0 нарушений (имя сущности буквально есть в чанке;
  evidence связи упоминает subject или object).
- Fingerprint: `66c0dbe3cbd3c00ff93b10f608b37f719f3f12fa68976c19f939dd402d14b001`.
- Команда проверки: `python scripts/verify_graph_provenance.py --graph <artifact>`.

### 8.2. Multi-hop eval (10 вопросов, граф только по прямым упоминаниям)

| Режим | hit | MRR | recall | labeled prec | чанков | токенов | latency |
|---|---|---|---|---|---|---|---|
| baseline: top-5 + reranker | 0.90 | **0.90** | 0.62 | 0.32 | 5.0 | — | 1.41 s |
| graph: pool 20, threshold 0, tok 4000 | 0.90 | 0.82 | **0.75** | 0.23 | 8.8 | 3 825 | 2.15 s |
| graph: pool 20, threshold 0, chars 16k | 0.90 | 0.82 | **0.75** | 0.17 | 12.1 | — | 2.15 s |
| graph: pool 30, threshold 0, tok 8000 | 0.90 | 0.83 | **0.75** | 0.17 | 12.4 | 5 656 | 2.36 s |

Interpretation:

- граф поднимает **recall свидетельств с 0.62 до 0.75 (+13 п.п.)** на
  многошаговых вопросах;
- MRR немного ниже (0.82 против 0.90): дополнительные свидетельства идут после
  первого правильного чанка, что приемлемо для multi-hop;
- пороги `1–3` снижали recall до 0.50–0.67, поэтому не используются;
- `include_relation_evidence=True` добавлял в среднем 6.5 чанка вместо 2.7, но
  recall не рос, MRR не улучшался (0.85–0.90), латентность увеличивалась — режим
  оставлен выключенным;
- на простом однофактном наборе (14 вопросов) production top-5 уже даёт
  hit=1.00 и MRR=0.96; граф там не нужен.

### 8.3. Ограничения измерения

- Наборы небольшие (14 + 10 вопросов) и размечены минимально необходимыми
  свидетельствами; `labeled precision` — нижняя оценка.
- Граф покрывает только 2022–2023; 2016–2017 в eval есть, но без graph-покрытия.
- Реранкер GigaChat нестабилен: иногда пропускает `chunk_id`; пропуск трактуется
  как 0, что видно в warnings.
- Граф ещё не включён в `factory`/UI; результаты получены через
  `scripts/eval_retrieval.py`, а не через WebUI.

### 8.4. Воспроизведение

```powershell
# проверка provenance
.venv\Scripts\python.exe -X utf8 scripts\verify_graph_provenance.py `
  --graph data\graph\66c0dbe3cbd3c00ff93b10f608b37f719f3f12fa68976c19f939dd402d14b001.json

# multi-hop eval
.venv\Scripts\python.exe -X utf8 scripts\eval_retrieval.py `
  --questions data\eval\retrieval_multihop.json `
  --graph data\graph\66c0dbe3cbd3c00ff93b10f608b37f719f3f12fa68976c19f939dd402d14b001.json `
  --pools 10,20,30 `
  --out data\eval\multihop_direct_v2.json
```

Датасеты и методика: `data/eval/README.md`.

### 8.5. Решение по использованию

- Не включать граф глобально для всех запросов: на однофактных вопросах
  production top-5 уже даёт hit=1.00 и MRR=0.96.
- Включать для multi-hop и сущностных запросов (router/agent выставляет
  `needs_graph`), с конфигурацией `pool=20`, `threshold=0`,
  `max_tokens=4000–8000`, только прямые упоминания.
- Baseline RAG остаётся fallback при пустом/сломанном графе.
- До E2E-проверки в WebUI граф не объявлять включённым по умолчанию; текущий
  измеренный результат получен через `scripts/eval_retrieval.py`.
