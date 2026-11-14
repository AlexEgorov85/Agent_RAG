# Sberbank Annual Reports Graph RAG

Structure-aware RAG по годовым отчётам Сбербанка за 2015–2020 и 2022–2024
(9 PDF-файлов, 2021 года в корпусе нет).

Система извлекает из PDF текст, таблицы и изображения, строит дерево разделов и
граф сущностей, затем отвечает на вопросы только по найденным фрагментам и
обязательно возвращает цитаты с `chunk_id`.

## Что умеет

- **Парсинг PDF** (`src/ingestion`) — Unstructured `hi_res`/`fast`, OCR (rus+eng),
  батчи по страницам, resume по SHA-256, фильтр мелких изображений.
- **Структура документа** — дерево разделов с breadcrumbs; приоритет источников:
  закладки PDF → оглавление → эвристика по заголовкам.
- **Чанки К2** — 800/100 символов, «таблица = один чанк» с HTML, структурные
  поля `section_path`/`breadcrumbs`.
- **Поиск** (`src/retrieval`) — BM25 с лемматизацией + dense-индекс, слияние RRF,
  опциональный LLM-реранкер, жёсткие фильтры по году/документу/разделу.
- **Graph RAG** (`src/graph`) — граф документов, разделов, чанков и сущностей;
  обёртка `GraphAugmentedRetriever` добавляет графовые свидетельства к
  гибридному поиску.
- **Ответы** (`src/engine`) — упаковка контекста целыми чанками, строгая проверка
  цитат, безопасный `abstain` при недостатке источников.
- **Агент** (`src/agent`) — LangGraph с белым списком инструментов и лимитами
  вызовов; потоковые события для UI.
- **UI** (`src/ui`) — Streamlit-чат с фильтрами, трассой инструментов и цитатами.

## Схема

```
PDF → pdf_parser → elements.json
                  → structure_builder → sections.json
                  → pdf_to_chunks     → chunks.json
                                        → graph_builder → graph.json

вопрос → hybrid retrieval (BM25 + dense → RRF → reranker)
       → упаковка контекста → LLM → Answer с цитатами
```

## Установка

Нужны Python 3.11+, [poppler](https://poppler.freedesktop.org/) и
[tesseract](https://github.com/UB-Mannheim/tesseract/wiki) с языком `rus`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env           # заполнить LLM_API_KEY
```

Минимальные переменные `.env`:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=https://foundation-models.api.cloud.ru/v1
LLM_MODEL=GigaChat/GigaChat-3-Pro
EMBED_MODEL=ai-sage/Giga-Embeddings-instruct-3B
RERANKER_MODEL=ai-sage/GigaChat3-10B-A1.8B
RERANK_ENABLED=1
```

## Запуск

Положить PDF в `data/raw/`, затем:

```bash
# Парсинг одного файла или всех
python -m src.ingestion --pdf "data/raw/Сбер 2023.pdf"
python -m src.ingestion --all

# Дерево разделов и чанки
python -m src.ingestion --structure
python -m src.ingestion --chunks

# Быстрая проверка парсера без полного прогона
python -m src.ingestion --sample --sample-pages 14
python -m src.ingestion --smoke --pages 3
python -m src.ingestion --stats
```

Dev-корпус (если полный парсинг ещё не готов):

```bash
python scripts/build_dev_corpus.py
python scripts/ask_dev.py "Какая чистая прибыль за 2023 год?" --top-k 5
python scripts/ask_dev.py "..." --agent
```

Граф и сравнение поиска:

```bash
python scripts/build_graph_dev.py --limit 200
python scripts/compare_graph_dev.py --questions questions.txt
```

UI:

```bash
streamlit run src/ui/app.py
```

API smoke-тест (реальные запросы только с `--live`):

```bash
python -m scripts.smoke_api --live
```

## Тесты

```bash
pytest
```

Тесты офлайновые: внешние API подменяются заглушками, ключи и сеть не нужны.

## Структура проекта

```
src/
  core/        настройки, модели, эмбеддинги, кэш, загрузка корпуса
  ingestion/   PDF-парсер, структура, чанки, валидатор корпуса
  retrieval/   BM25, dense, RRF, реранкер, graph-augmented поиск
  graph/       модели графа, извлечение сущностей, сборка/загрузка графа
  engine/      упаковка контекста, промпты, генерация, RAGEngine
  agent/       LangGraph-оркестрация, инструменты, AgentService
  ui/          Streamlit-чат и отображение цитат
scripts/       dev-корпус, граф, сравнение поиска, smoke-тест API
tests/         офлайновые unit/интеграционные тесты
data/          исходные PDF и производные артефакты
```

## Ограничения

- Для ответов и эмбеддингов нужен доступ к Cloud.ru API.
- 2021 год в корпусе отсутствует; система должна сообщать об этом, а не
  подставлять соседний год.
- Качество структуры и таблиц зависит от PDF и OCR; спорные случаи проверяются
  через `--sample`, `--stats` и артефакты в `data/samples`.
