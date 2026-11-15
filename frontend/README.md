# Agent RAG — React WebUI

Фронтенд исследовательского чата по годовым отчётам. Реализует этапы 1–4 и
большую часть этапа 5 из
[`docs/specs/web-ui-modernization.md`](../docs/specs/web-ui-modernization.md):
историю диалогов, чат, реальные SSE-этапы, evidence panel, кликабельные ссылки на
первоисточники, аналитические таблицы/графики, структурное сравнение и Evaluation UI.

## Запуск

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, /api проксируется на 127.0.0.1:8000
npm run build
npm test -- --run    # Vitest (без сети и LLM)
npm run e2e          # Playwright против fixture-backend (нужен установленный chromium)
```

Backend: `uvicorn src.api.app:app --port 8000` (готовый adapter в `src/api/**`).

### Mock-режим

```bash
VITE_API_MODE=mock npm run dev
```

Включает встроенный fixture-backend (`src/mocks/`) для просмотра интерфейса без Python-ядра.
В шапке диалога постоянно висит баннер «Демонстрационные данные (mock)»; mock никогда не
выдаётся за результат RAG.

## Chat runtime (assistant-ui)

Чат построен на `@assistant-ui/react`:

- `useLocalRuntime` + собственный `ChatModelAdapter` (`src/features/assistant-runtime/chat-adapter.ts`).
- `AssistantRuntimeProvider` (`ResearchRuntimeProvider`, ключ — `researchId`).
- UI собран из примитивов `ThreadPrimitive` / `MessagePrimitive` / `ComposerPrimitive` /
  `ActionBarPrimitive`.

`useLocalRuntime` выбран потому, что backend — не Vercel AI SDK: запрос стартует
`POST /api/research/{id}/messages`, а прогресс приходит отдельным SSE-потоком
`GET /api/requests/{id}/events`. `ChatModelAdapter` — документированный custom-transport
для такого случая; `useChatRuntime`/`@assistant-ui/ai-sdk` потребовали бы подделывать
AI SDK data-stream протокол. Runtime хранит сообщения и streaming-состояние; параллельного
самодельного reducer чата нет.

### Маппинг SSE → message parts

| Backend event | assistant-ui part |
|---|---|
| `run_started`, `analysis_completed`, `tool_started`, `tool_completed`, `answer_started` | `data` part `execution` (этапы, инструменты, intent, effective filters) |
| `answer_delta` | накопительный `text` part |
| `run_completed.message.content` | финальный `text` part |
| `run_completed.message.evidence` | `data` part `evidence` (источники и Evidence Panel) |
| `analysis.facts` / `calculations` / `structural_changes` | `data` part `analysis` (таблицы, графики, структсравнение) |
| `analysis.coverage` | `data` part `coverage` |
| `verification` (если backend передал) | `data` part `verification` |
| `message.warnings` | `data` part `warnings` |
| `run_failed` | `data` part `run_error` + статус `incomplete(error)` |
| `run_cancelled` | `data` part `answer_state` + статус `incomplete(cancelled)` |

`sequence`, дубли, чужой `request_id` и события после терминального обрабатываются в
`run-projection.ts`. Переподключение использует `Last-Event-ID`; при `resync`/`410`
статус запрашивается через `GET /api/requests/{id}` и показывается честное состояние
`interrupted`, а не вечный spinner. Отмена из composer останавливает runtime, вызывает
`POST /api/requests/{id}/cancel` и не выдаёт частичный текст за готовый.

## Экраны и состояния

> В UI сущность называется «диалог»; в API и БД она по-прежнему `research`
> (`research.id` = `conversation_id`), пути `/api/research/*` не менялись.

- Левая панель: список диалогов, поиск (backend), создание, переименование, удаление
  с подтверждением, статус, дата; сворачивается до rail, на mobile — drawer.
- Header 48px: toggle панели, название диалога (редактируется), фильтры, источники,
  тема, disabled «Поделиться».
- Thread: ширина 44rem, welcome «Чем я могу помочь?» и chips
  («Прибыль за 2023», «Сравни 2020–2024», «Структура отчётов», «Риски», «Что с 2021?»).
- Composer: скруглённая рамка, drag/drop-зона (вложения честно не поддерживаются),
  disabled attachment, model picker «GigaChat», круглая ArrowUp / Square, авто-высота.
- Messages: user справа в компактном bubble, assistant на всю ширину, Markdown,
  copy/reload/export, scroll-to-bottom. Tool/stages — компактный сворачиваемый trace,
  без chain-of-thought.
- Evidence: карточки с document_title, year, issuer, section, pages, chunk_id, quote,
  source_type; отсутствующие поля — «не указано»; полный текст по `/api/evidence/{chunk_id}`
  с подсветкой цитаты и безопасным рендером таблиц.
- Ошибки — inline alert с `code`/`request_id`; abstained/cancelled/interrupted — честные
  текстовые состояния.
- Служебные реплики («Привет!», «Что ещё умеешь?») приходят как обычный готовый ответ
  без источников и без предупреждений: их формирует backend (`src/agent/smalltalk.py`),
  frontend их не имитирует.

## Evaluation / Benchmark

Отдельный route `/evaluation` (не вкладка чата), гейтится флагом
`capabilities.evaluation_ui`; при `false` показывает честный disabled-экран.
Внутри — вкладки «Обзор», «Запуски», «Тесты», «Сравнение»: история запусков,
polling каждые 2 c, отмена (по `evaluation_cancel`), отчёт, trace теста,
сравнение двух запусков. Judge включается только при `judge=true`.

Раздел требует `EVAL_UI_ENABLED=1` на backend; по умолчанию API отвечает
`404 evaluation_disabled`. Ссылка «Evaluation» есть в шапке чата, обратная
навигация сохраняет выбранный диалог, E2E покрывают оба режима. Пока нет:
SSE-прогресса (только polling), выбора `attempt` > 0 и перехода из chunk
Evaluation в Evidence Panel чата.

## Ограничения и что осталось

- PDF-вьюер: ссылка на страницу PDF открывается в браузере
  (`/api/documents/{doc_id}/file#page=N`), встроенного просмотрщика с подсветкой нет;
- панель Graph Retrieval не включена (нет опубликованного backend-контракта узлов/путей);
- шаринг диалога и вложения (нет backend-контракта);
- SSE-прогресс Evaluation не реализован, используется polling;
- возобновление прерванного run после перезагрузки: история восстанавливается,
  незавершённый ответ честно помечается `interrupted` с кнопкой повтора;
- начальные vendor-chunks ~1.0 MB из-за assistant-ui/markdown, Recharts вынесен
  в lazy-chunk (`AnalyticsCharts`); warning >500 kB снят vendor-split, остаётся
  notice о dynamic+static import `api-client`. Дальнейший code-splitting —
  задача этапа 5.

## Тесты

- `tests/unit/*` — 160 тестов в 24 файлах: projection SSE (sequence/дубли/чужой request_id/
  terminal/error/cancel/resync), adapter с mock-fetch, thread/evidence/composer/sidebar,
  санитизация Markdown и таблиц, аналитика/графики/CSV, citations и PDF-ссылки,
  структсравнение, Evaluation UI (обзор, запуски, тесты, сравнение).
- `tests/e2e/*` — 7 Playwright-тестов против `tests/e2e/fixture-server.mjs`:
  welcome → вопрос → этапы → ответ → источник → reload, ошибка backend, sidebar, mobile
  drawer, тема, переход чат ↔ Evaluation, disabled Evaluation.
- E2E использует изолированные порты fixture `8010` и UI `5273`, чтобы не зависеть от
  живого backend на `8000`.
- Скриншоты Base-компоновки сохраняются Playwright в `test-results/` (не в git).
