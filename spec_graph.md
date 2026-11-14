# Спецификация Graph RAG

**Версия:** 1.0
**Статус:** целевой контракт; graph слой ещё не реализован.
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

### 3.1. Дедупликация сущностей

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
- graph candidates проходят общий RRF/reranker pipeline согласно retrieval spec.

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
  улучшением без измеренного сравнения.
