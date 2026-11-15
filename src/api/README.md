# src/api — HTTP/SSE adapter React WebUI

Тонкий FastAPI-слой над существующим `AgentService`
(`docs/specs/web-ui-modernization.md`, этапы 1–2) плюс Evaluation API
(`docs/specs/evaluation-webui.md`). Adapter не содержит второй
оркестрации, retrieval или генерации ответа: он принимает HTTP-запросы, хранит
историю исследований в SQLite и сериализует события Python-ядра.

Пакет реализует чат/исследования/SSE/evidence и **Evaluation API**
(`/api/evaluation/*` из `docs/specs/evaluation-webui.md`). Evaluation-раздел
выключен по умолчанию (`EVAL_UI_ENABLED=0`): endpoints отвечают
`404 evaluation_disabled`, а `capabilities.evaluation_ui=false`.

## Запуск

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.app:app --port 8000
```

Vite dev-сервер фронтенда проксирует `/api` на `http://127.0.0.1:8000`.
Production работает same-origin: `API_SERVE_SPA=1` раздаёт `frontend/dist`
(каталог `API_SPA_DIST_DIR`) с history-fallback на `index.html`; без флага SPA
отдаёт внешний reverse-proxy/Vite dev. Импорт `src.api.app` не читает env, не
создаёт БД и не строит индексы: всё происходит в lifespan.

## Endpoint'ы

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/health` | готовность БД и backend'а (503 при недоступном корпусе/LLM) |
| GET | `/api/capabilities` | честные флаги возможностей |
| GET | `/api/corpus` | годы, документы, issuers (пусто, если backend не умеет) |
| GET | `/api/research` | список, `q`, `limit`, `offset` |
| POST | `/api/research` | создать исследование |
| GET | `/api/research/{id}` | карточка |
| PATCH | `/api/research/{id}` | переименовать (`title` 1–200) |
| DELETE | `/api/research/{id}` | удалить каскадно (204) |
| GET | `/api/research/{id}/messages` | история, `limit`, `before` |
| POST | `/api/research/{id}/messages` | отправить вопрос (202) |
| GET | `/api/requests/{request_id}` | REST-статус и терминальное сообщение |
| GET | `/api/requests/{request_id}/events` | SSE-поток |
| POST | `/api/requests/{request_id}/cancel` | отмена |
| GET | `/api/evidence/{chunk_id}` | метаданные и текст chunk |
| GET | `/api/documents/{doc_id}/file` | исходный PDF для перехода на страницу (опционально) |
| GET | `/api/evaluation/suites` | доступные наборы и режимы (при `EVAL_UI_ENABLED=1`) |
| POST | `/api/evaluation/runs` | запустить benchmark (202) |
| GET | `/api/evaluation/runs` | история запусков |
| GET | `/api/evaluation/runs/{run_id}` | карточка запуска и progress |
| GET | `/api/evaluation/runs/{run_id}/tests` | результаты тестов |
| GET | `/api/evaluation/runs/{run_id}/tests/{test_id}` | детализация теста |
| GET | `/api/evaluation/runs/{run_id}/tests/{test_id}/trace` | retrieval trace |
| GET | `/api/evaluation/runs/{run_id}/report` | отчёт запуска |
| POST | `/api/evaluation/runs/{run_id}/cancel` | отмена запуска |
| GET | `/api/evaluation/tests` | каталог тестов датасета |
| GET | `/api/evaluation/tests/{test_id}` | определение теста и история запусков |
| GET | `/api/evaluation/compare` | сравнение двух запусков |

`research.id` — это `conversation_id` для `AgentService`. Чужой/несуществующий
`research_id` → 404; чужой `request_id` в `/api/requests` → 404, в `/events` →
410 (не раскрываем существование).

### Отправка сообщения

```http
POST /api/research/{id}/messages
{
  "content": "Сравни выручку за 2020–2024",
  "client_message_id": "uuid",
  "filters": {"years": [2020, 2021, 2022], "issuers": ["ПАО Сбербанк"]}
}
```

- `client_message_id` обязателен; повторный POST возвращает тот же `request_id`
  и не запускает второй прогон;
- один активный запрос на исследование: второй POST → `409 request_in_progress`
  с `details.active_request_id`;
- пустые фильтры не передаются, неизвестные поля игнорируются;
- пользовательское сообщение сохраняется до запуска агента, assistant-сообщение
  создаётся со статусом `streaming` и обновляется терминальным событием.

### Ошибки

Единый формат:

```json
{"error": {"code": "not_found", "message": "...", "retryable": false}}
```

Коды: `validation_error` (422), `not_found` (404), `request_in_progress` (409),
`request_not_found` (404), `request_not_replayable` (410), `payload_too_large`
(413), `rate_limited` (429), `backend_unavailable` (503), `internal_error` (500).
Traceback, prompt, ключи и тексты провайдера не возвращаются.

## SQLite

Файл по умолчанию `var/api.sqlite3` (WAL, `PRAGMA foreign_keys=ON`, миграции
при старте, без ORM). Таблицы соответствуют §7.9 спецификации:

- `researches(id, owner_id, title, created_at, updated_at)`;
- `messages(id, research_id → researches ON DELETE CASCADE, role, content,
  status, request_id, client_message_id, answer_json, filters_json, error_json,
  created_at)` + частичный unique index `(research_id, client_message_id)`;
- `requests(id, research_id → researches ON DELETE CASCADE,
  assistant_message_id, status, error_code, started_at, finished_at)`.

В `answer_json` — только уже полученные от backend evidence, `facts`,
`calculations`, `coverage`, `structural_changes`, `verification`, `warnings` и
`effective_filters`. Полный текст chunk и HTML таблиц в БД не дублируются.

При старте `queued`/`running` запросы помечаются `failed` с
`code=interrupted`, чтобы UI не висел после перезапуска API.

## SSE

`GET /api/requests/{request_id}/events` (`text/event-stream`):

```text
id: 17
event: agent
data: {"contract_version":"1","sequence":17,"type":"tool_started",...}
```

- `id` = монотонный `sequence` в пределах запроса; типы событий — как у
  `AgentEvent` (`run_started`, `analysis_completed`, `tool_started`,
  `tool_completed`, `answer_started`, `run_completed`, `run_failed`);
- adapter добавляет только `message_id` (в `run_started`), heartbeat,
  `run_cancelled` и `resync`;
- `answer_delta`/`evidence_updated` не публикуются, пока backend их не отдаёт
  (`token_streaming=false`, `incremental_evidence=false`);
- heartbeat — SSE-комментарий `: keep-alive` каждые 15 секунд (настраивается);
- терминальное событие всегда присутствует: `run_completed`/`run_failed`/
  `run_cancelled`. Если backend-поток оборвался без терминального события,
  adapter завершает запрос безопасным `run_failed` (`internal_error`);
- `Last-Event-ID` воспроизводит события из ограниченного буфера; если окно
  replay истекло — `event: resync` с `reason=replay_window_expired`;
- если запрос завершён, а буфер потерян (перезапуск API) — `resync` с
  `reason=request_finished`; UI перезагружает сообщения через REST;
- заголовки: `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`;
- синхронный `AgentService.stream` исполняется в отдельном потоке, SSE
  подписчик ждёт события через `asyncio.to_thread` — event loop не блокируется;
- если backend умеет только `ask()` (без `stream`), adapter публикует только
  `run_started` и `run_completed`, не имитируя промежуточные этапы;
- служебные реплики вне корпуса («Привет!», «Спасибо», «Что ты умеешь?») тоже
  завершаются только `run_started` + `run_completed`: `AgentService` отвечает на
  них без retrieval и без `no_evidence`, потому что это не ответ по документам
  (`src/agent/smalltalk.py`);
- в поток не попадают query, filters, prompt, chunk-текст и секреты. Аргументы
  инструментов публикуются только по allowlist (`top_k`, `doc_id`,
  `section_path`, `retrieval_mode` и т.п.).

## Evidence

`GET /api/evidence/{chunk_id}` ищет chunk только в существующем `ChunkBackend`
ядра (не в файловой системе), неизвестный ID → 404. Таблица возвращает `html`,
текст — `text`; `Cache-Control: private, max-age=300`. Issuer при отсутствии в
chunk берётся из `data/source_registry.json` (метаданные backend'а, не догадка).

Ответ сообщения содержит снимок `EvidenceItem` по §5.3: `evidence_id=chunk_id`,
`document_title`/`source_type` из chunk metadata, `section_path` из
`Citation.section_path`, `excerpt=quote`, `has_full_text` по доступности chunk.

## PDF-источники

`GET /api/documents/{doc_id}/file` отдаёт исходный PDF из
`data/source_registry.json` (не сканирует каталоги). Неизвестный `doc_id`,
отсутствующий файл и путь вне `data/raw` → `404 not_found`. Заголовки:
`Content-Type: application/pdf`, `Content-Disposition: inline;
filename="<doc_id>.pdf"`, `Cache-Control: private, max-age=3600`,
`ETag`/`Accept-Ranges` (Range и `If-None-Match` поддержаны).

Frontend строит ссылку на конкретную страницу так:

```text
/api/documents/{doc_id}/file#page={page_pdf}
```

Fragment `#page=N` обрабатывает браузерный PDF-вьюер; сервер его не видит и
отдаёт тот же файл. `page_pdf` берётся из `Citation.page_pdf` /
`EvidenceItem.page_start`. Ссылку показывать только при
`capabilities.pdf_page_links === true`; флаг включается, лишь когда реестр
валиден и хотя бы один файл реально отдаётся. При `false` страница PDF в UI
не предлагается, но evidence остаётся доступным.

## Evaluation / Benchmark

Тонкий adapter над `eval.runner` (`docs/specs/evaluation-webui.md`): suites,
запуск в очереди, история, результаты тестов, retrieval trace, отчёт, отмена и
сравнение двух запусков. Storage — отдельная SQLite-БД (`EVAL_DB_PATH`), не
смешивается с историей диалогов; артефакты — в `EVAL_RESULTS_DIR`.

- включение: `EVAL_UI_ENABLED=1`; при `0` все endpoints → `404
  evaluation_disabled`, `capabilities.evaluation_ui=false`;
- прогресс отдаётся polling'ом (`GET /runs/{id}`), `evaluation_sse=false`:
  SSE-канал в первой версии не реализован;
- `evaluation_cancel` поддерживается для queued/running;
- chat и evaluation проходят через `ExecutionGate` (`src/api/execution.py`,
  приоритет chat > evaluation) и общий межпроцессный `LLMExecutionLock`
  (`data/evaluation/llm.lock`, `LLM_LOCK_PATH`); занятый lock у CLI даёт exit 5
  `llm_busy`;
- judge выключен, пока `EVAL_JUDGE_ENABLED!=1`;
- ошибки — общий формат `{"error": {...}}`, коды те же, что у чата.

## Возможности (`/api/capabilities`)

```json
{
  "contract_version": "1",
  "token_streaming": false,
  "incremental_evidence": false,
  "coverage": true,
  "calculations": true,
  "structure_comparison": true,
  "structure_content_comparison": true,
  "graph_panel": false,
  "pdf_page_links": false,
  "evaluation_ui": false,
  "evaluation_runs": false,
  "evaluation_sse": false,
  "evaluation_cancel": false,
  "retrieval_modes": ["hybrid", "dense", "keyword"],
  "judge": false,
  "max_repeat": 5,
  "max_queued": 3
}
```

`evaluation_ui`/`evaluation_runs`/`evaluation_cancel` включаются вместе с
`EVAL_UI_ENABLED=1`; `evaluation_sse` остаётся `false` (UI использует polling),
`judge` — только при `EVAL_JUDGE_ENABLED=1`. `pdf_page_links` вычисляется
динамически: при валидном `data/source_registry.json` и доступном PDF endpoint
он равен `true` (в этом репозитории — `true`), иначе `false`.
`graph_panel=false`: graph retrieval работает, но отдельной панели визуализации
узлов/путей в React UI нет.

Флаги не заменяют данные конкретного ответа: UI включает блок только при
наличии флага **и** данных.

## Безопасность

- `AUTH_MODE=local`: один владелец, cookie `HttpOnly`, `SameSite=Lax`,
  `Secure` при HTTPS; EventSource использует cookie, токенов в URL нет.
- Каждая выборка/мутация фильтруется по `owner_id`; чужое → 404.
- POST/PATCH/DELETE из чужого `Origin` отклоняются.
- CORS dev: `http://localhost:5173`; production — same-origin.
- Лимиты: сообщение ≤ 16 000 символов, событие ≤ 256 КБ, evidence ≤ 1 МБ,
  rate limit на POST-сообщения (конфигурируется).
- `DELETE` исследования каскадно удаляет сообщения и запросы.

## Конфигурация (env, читается в lifespan)

`AUTH_MODE`, `API_DB_PATH`, `API_SOURCE_REGISTRY_PATH`, `API_CORPUS_NAME`,
`API_OWNER_COOKIE`, `API_COOKIE_SECURE`, `API_SESSION_MAX_AGE`,
`API_CORS_ORIGINS`, `API_MAX_MESSAGE_CHARS`, `API_MAX_EVENT_BYTES`,
`API_MAX_EVIDENCE_BYTES`, `API_REPLAY_BUFFER_EVENTS`, `API_REPLAY_TTL_SECONDS`,
`API_HEARTBEAT_SECONDS`, `API_PAGE_SIZE_DEFAULT`, `API_PAGE_SIZE_MAX`,
`API_HISTORY_RETENTION_DAYS`, `API_RATE_LIMIT_PER_MINUTE`, `API_SERVE_SPA`,
`API_SPA_DIST_DIR`, `EVAL_UI_ENABLED`, `EVAL_DB_PATH`, `EVAL_RESULTS_DIR`,
`EVAL_DATASETS_DIR`, `EVAL_MAX_CONCURRENCY`, `EVAL_QUEUE_SIZE`,
`EVAL_RETENTION_RUNS`, `EVAL_YIELD_MS`, `EVAL_TURN_TIMEOUT_S`,
`EVAL_RUN_TIMEOUT_S`, `EVAL_FAKE_SERVICE`, `EVAL_JUDGE_ENABLED`,
`LLM_LOCK_PATH`.

`AgentService` создаётся лениво при первом обращении (health чата) и один раз на
процесс. При недоступном корпусе/LLM `/api/health` возвращает 503, POST
сообщения — `backend_unavailable` без создания «висящего» сообщения.

## Отмена

`POST /api/requests/{request_id}/cancel`:

- если backend поддерживает `AgentService.cancel(request_id)` — запрос
  помечается `cancelled`, сообщение очищается (ответ не сохраняется), поток
  завершается `run_cancelled`;
- если метода нет — возвращается `200 {"status":"not_supported",
  "cancel_supported":false, "code":"cancel_not_supported"}`; adapter **не
  имитирует** отмену;
- повторная отмена идемпотентна, завершённый запрос → `already_finished`.

## Ограничения

- Токеновый streaming не имитируется: `answer_delta` отсутствует в backend.
- PDF-переходы включены при валидном реестре (`pdf_page_links=true`) и отдают
  файл с fragment `#page=N`; встроенного PDF-вьюера с подсветкой нет.
- Graph-панель (`graph_panel=false`) не реализована: нет опубликованного
  контракта узлов/путей для визуализации, хотя graph retrieval работает.
- `structure_content_comparison=true` означает поддержку backend; frontend не
  синтезирует split/merge или содержание, если их нет в `structural_changes`.
- Серверная отмена чата не реализована: endpoint возвращает
  `cancel_not_supported`, UI останавливает только локальную доставку.
- `AgentService._conversation_history` живёт в памяти процесса: перезагрузка
  страницы сохраняет видимую историю из SQLite, перезапуск API теряет контекст
  прошлых ходов (документированное ограничение §3.3).
- Retention-настройка предусмотрена конфигом, автоочистка в первом релизе не
  выполняется.
- Evaluation UI живёт в `frontend/`; в этом пакете только API. Раздел выключен
  конфигурацией и использует polling вместо SSE.

## Тесты

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_web_api_*.py
```

Тесты полностью офлайн: fake `AgentService`, fake chunk backend, временная
SQLite-БД. Сеть, LLM и реальный корпус не нужны.
