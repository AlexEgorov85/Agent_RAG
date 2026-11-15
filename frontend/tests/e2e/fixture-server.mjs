/**
 * Fixture backend for Playwright: speaks the same public API/SSE contract as
 * the FastAPI adapter (research CRUD, messages, events, evidence).
 * It is deliberately dumb: no retrieval, no generation, only scripted events.
 */
import http from 'node:http'

const PORT = Number(process.env.FIXTURE_PORT ?? 8000)

const iso = () => new Date().toISOString()

const evidenceCatalog = {
  'sber_2023:chunk-1': {
    evidence_id: 'sber_2023:chunk-1',
    chunk_id: 'sber_2023:chunk-1',
    document_id: 'sber_2023',
    document_title: 'Годовой отчёт 2023',
    issuer: 'ПАО Сбербанк',
    year: 2023,
    section_title: 'Ключевые показатели',
    section_path: ['Стратегия', 'Ключевые показатели'],
    page_start: 12,
    page_label: '12',
    excerpt: 'Чистая прибыль за 2023 год составила 1 508,6 млрд рублей.',
    source_type: 'text',
    has_full_text: true,
  },
  'sber_2023:chunk-2': {
    evidence_id: 'sber_2023:chunk-2',
    chunk_id: 'sber_2023:chunk-2',
    document_id: 'sber_2023',
    issuer: 'ПАО Сбербанк',
    source_type: 'table',
    excerpt: '',
    has_full_text: false,
  },
}

const evidenceDetails = {
  'sber_2023:chunk-1': {
    chunk_id: 'sber_2023:chunk-1',
    document_id: 'sber_2023',
    document_title: 'Годовой отчёт 2023',
    issuer: 'ПАО Сбербанк',
    year: 2023,
    section_path: ['Стратегия', 'Ключевые показатели'],
    page_start: 12,
    page_label: '12',
    source_type: 'text',
    text:
      'Стратегия\n\nКлючевые показатели\n\nЧистая прибыль за 2023 год составила 1 508,6 млрд рублей, ' +
      'увеличившись на 4,0% по сравнению с 2022 годом.',
    html: null,
  },
}

const state = {
  researches: [
    {
      id: 'research-e2e',
      title: 'E2E диалог',
      created_at: iso(),
      updated_at: iso(),
      last_status: 'idle',
      messages_count: 0,
    },
  ],
  messages: {},
  runs: new Map(),
  /**
   * Evaluation UI is opt-in per E2E test via POST /api/_fixture/evaluation-ui.
   * Default stays false, so the five chat specs keep exercising the
   * `evaluation_ui=false` contract. The fixture serves only EMPTY evaluation
   * collections: no pass rates, summaries or traces are invented here.
   */
  evaluationUi: process.env.FIXTURE_EVALUATION_UI === '1',
}

const EVALUATION_FIXTURE_NOTE = 'empty-evaluation-fixture'

const json = (res, data, status = 200) => {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' })
  res.end(JSON.stringify(data))
}

const fail = (res, status, code, message, extra = {}) =>
  json(res, { error: { code, message, retryable: status >= 500, ...extra } }, status)

const readBody = (req) =>
  new Promise((resolve) => {
    let raw = ''
    req.on('data', (chunk) => {
      raw += chunk
    })
    req.on('end', () => {
      try {
        resolve(raw ? JSON.parse(raw) : {})
      } catch {
        resolve({})
      }
    })
  })

const researchById = (id) => state.researches.find((item) => item.id === id)
let counter = 0
const id = (prefix) => `${prefix}-${++counter}`

function buildRun(research, content) {
  const requestId = id('req')
  const assistantId = id('msg')
  const abstain = /2021/.test(content)
  const failRun = /ошибк/i.test(content)
  const now = iso()
  const message = {
    id: assistantId,
    research_id: research.id,
    role: 'assistant',
    content: abstain
      ? 'За 2021 год в корпусе нет отчёта, поэтому ответ не сформирован.'
      : 'Чистая прибыль за 2023 год составила **1 508,6 млрд руб.**\n\n| Год | Значение |\n| --- | --- |\n| 2023 | 1 508,6 |',
    created_at: now,
    status: abstain ? 'abstained' : 'completed',
    request_id: requestId,
    effective_filters: { years: [2023] },
    evidence: abstain ? [] : [evidenceCatalog['sber_2023:chunk-1'], evidenceCatalog['sber_2023:chunk-2']],
    warnings: [],
  }

  const events = [
    { type: 'run_started', request_id: requestId, message_id: assistantId },
    {
      type: 'analysis_completed',
      request_id: requestId,
      intent: 'factoid',
      effective_filters: { years: [2023] },
    },
    {
      type: 'tool_started',
      request_id: requestId,
      tool_name: 'search_reports',
      short_arguments: { years: [2023] },
    },
    {
      type: 'tool_completed',
      request_id: requestId,
      tool_name: 'search_reports',
      status: 'success',
      result_count: 2,
      duration_ms: 640,
    },
    { type: 'answer_started', request_id: requestId },
    ...(abstain || failRun
      ? []
      : [
          { type: 'answer_delta', request_id: requestId, delta: 'Чистая прибыль ' },
          { type: 'answer_delta', request_id: requestId, delta: 'составила **1 508,6 млрд руб.**' },
        ]),
    failRun
      ? {
          type: 'run_failed',
          request_id: requestId,
          code: 'backend_unavailable',
          message: 'Fixture LLM недоступен',
          retryable: true,
        }
      : {
          type: 'run_completed',
          request_id: requestId,
          answer_status: abstain ? 'abstain' : 'answer',
          duration_ms: 1200,
          message: failRun ? { ...message, status: 'failed' } : message,
        },
  ].map((event, index) => ({ contract_version: '1', sequence: index + 1, ...event }))

  return {
    requestId,
    researchId: research.id,
    assistantId,
    events,
    message: failRun
      ? {
          ...message,
          status: 'failed',
          content: '',
          evidence: [],
          error: {
            code: 'backend_unavailable',
            message: 'Fixture LLM недоступен',
            request_id: requestId,
            retryable: true,
          },
        }
      : message,
    cancelled: false,
    finalized: false,
    status: 'running',
  }
}

const TERMINAL_EVENT_TYPES = new Set(['run_completed', 'run_failed', 'run_cancelled'])

/**
 * Persist the terminal assistant message BEFORE the terminal SSE frame is
 * written. Real API does the same (store.update_message + finish_request,
 * then channel.publish), so a reload right after the terminal event must see
 * the final row, not the streaming placeholder.
 */
function finalizeRun(run) {
  if (run.finalized) return
  run.finalized = true

  const research = researchById(run.researchId)
  const list = state.messages[run.researchId] ?? []
  const index = list.findIndex((item) => item.id === run.assistantId)
  if (index >= 0) list[index] = run.message
  else list.push(run.message)
  state.messages[run.researchId] = list

  run.status =
    run.message.status === 'abstained'
      ? 'abstained'
      : run.message.status === 'failed'
        ? 'failed'
        : 'completed'

  if (research) {
    research.last_status = run.status
    research.updated_at = iso()
    research.messages_count = list.length
  }
}

function sse(res, run, lastEventId) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'X-Accel-Buffering': 'no',
  })
  let closed = false
  res.on('close', () => {
    closed = true
  })

  const send = (event) => {
    if (closed) return
    if (TERMINAL_EVENT_TYPES.has(event.type)) finalizeRun(run)
    res.write(`id: ${event.sequence}\nevent: agent\ndata: ${JSON.stringify(event)}\n\n`)
  }

  const pending = run.events.filter((event) => event.sequence > lastEventId)
  let index = 0
  const tick = () => {
    if (closed || index >= pending.length) {
      if (!closed) res.end()
      return
    }
    send(pending[index++])
    setTimeout(tick, lastEventId ? 20 : 120)
  }
  tick()
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host}`)
  const path = url.pathname
  const method = req.method ?? 'GET'

  if (method === 'GET' && path === '/api/health') return json(res, { status: 'ok', corpus_ready: true })
  if (method === 'GET' && path === '/api/capabilities') {
    return json(res, {
      contract_version: '1',
      token_streaming: true,
      incremental_evidence: false,
      coverage: false,
      calculations: false,
      structure_comparison: false,
      structure_content_comparison: false,
      graph_panel: false,
      pdf_page_links: false,
      evaluation_ui: state.evaluationUi,
      evaluation_sse: false,
      evaluation_cancel: false,
      evaluation_runs: false,
      judge: false,
      max_repeat: 1,
      max_queued: 1,
      retrieval_modes: ['hybrid'],
    })
  }

  /**
   * Test-only switch for the Evaluation route. It is intentionally outside the
   * real API contract (`/api/_fixture/*`) and only exists in this fixture.
   */
  if (path === '/api/_fixture/evaluation-ui') {
    if (method === 'GET') return json(res, { evaluation_ui: state.evaluationUi })
    if (method === 'POST') {
      const body = await readBody(req)
      state.evaluationUi = Boolean(body.enabled)
      return json(res, { evaluation_ui: state.evaluationUi })
    }
  }
  if (method === 'GET' && path === '/api/corpus') {
    return json(res, {
      name: 'Fixture корпус',
      years: [2022, 2023, 2024],
      documents: [
        { doc_id: 'sber_2023', year: 2023, issuer: 'ПАО Сбербанк', title: 'Годовой отчёт 2023' },
      ],
      issuers: ['ПАО Сбербанк'],
      years_available: true,
    })
  }

  /**
   * Minimal Empty-Evaluation fixture. These endpoints exist so the Evaluation
   * route can be exercised end-to-end, but they never return fabricated
   * metrics: runs/tests collections are empty and explicitly marked.
   */
  if (path.startsWith('/api/evaluation')) {
    if (!state.evaluationUi) {
      return fail(res, 404, 'evaluation_disabled', 'Раздел Evaluation отключён (evaluation_ui=false).')
    }

    if (path === '/api/evaluation/suites' && method === 'GET') {
      return json(res, {
        items: ['smoke', 'full', 'easy', 'medium', 'hard', 'multi_turn'].map((suite) => ({
          suite,
          tests: null,
          dev: null,
          holdout: null,
          requires_holdout_confirmation: false,
        })),
        dataset_available: false,
        dataset_sha256: null,
        expected_corpus_sha256: null,
        retrieval_modes: ['hybrid'],
        max_repeat: 1,
        max_queued: 1,
        fixture: EVALUATION_FIXTURE_NOTE,
      })
    }

    if (path === '/api/evaluation/runs' && method === 'GET') {
      return json(res, {
        items: [],
        total: 0,
        next_offset: null,
        fixture: EVALUATION_FIXTURE_NOTE,
      })
    }

    if (path === '/api/evaluation/runs' && method === 'POST') {
      return fail(
        res,
        503,
        'evaluation_unavailable',
        'Fixture backend не выполняет benchmark: реальные метрики здесь не подменяются.',
      )
    }

    if (path === '/api/evaluation/tests' && method === 'GET') {
      return json(res, {
        items: [],
        total: 0,
        next_offset: null,
        dataset_available: false,
        run_id: null,
        retrieval_mode: url.searchParams.get('retrieval_mode') ?? 'hybrid',
        fixture: EVALUATION_FIXTURE_NOTE,
      })
    }

    if (path === '/api/evaluation/compare' && method === 'GET') {
      return fail(res, 404, 'run_not_found', 'Fixture не содержит запусков для сравнения.')
    }

    if (/^\/api\/evaluation\/runs\/[^/]+/.test(path)) {
      return fail(res, 404, 'run_not_found', 'Fixture не содержит запусков Evaluation.')
    }
    if (/^\/api\/evaluation\/tests\/[^/]+/.test(path)) {
      return fail(res, 404, 'test_not_found', 'Fixture не содержит тестов Evaluation.')
    }
  }

  if (path === '/api/research' && method === 'GET') {
    return json(res, { items: state.researches, total: state.researches.length, next_offset: null })
  }
  if (path === '/api/research' && method === 'POST') {
    const body = await readBody(req)
    const research = {
      id: id('research'),
      title: typeof body.title === 'string' && body.title.trim() ? body.title.trim() : 'Новый диалог',
      created_at: iso(),
      updated_at: iso(),
      last_status: 'idle',
      messages_count: 0,
    }
    state.researches.unshift(research)
    state.messages[research.id] = []
    return json(res, research, 201)
  }

  const researchMatch = /^\/api\/research\/([^/]+)(\/messages)?$/.exec(path)
  if (researchMatch) {
    const research = researchById(decodeURIComponent(researchMatch[1]))
    if (!research) return fail(res, 404, 'not_found', 'Диалог не найден')
    const isMessages = Boolean(researchMatch[2])

    if (isMessages && method === 'GET') {
      return json(res, { items: state.messages[research.id] ?? [], total: 0, next_before: null })
    }
    if (isMessages && method === 'POST') {
      const body = await readBody(req)
      const content = typeof body.content === 'string' ? body.content.trim() : ''
      if (!content) return fail(res, 422, 'validation_error', 'Пустой вопрос')
      const run = buildRun(research, content)
      const userMessage = {
        id: id('msg-user'),
        research_id: research.id,
        role: 'user',
        content,
        created_at: iso(),
        status: 'completed',
        evidence: [],
        warnings: [],
      }
      state.messages[research.id] = [
        ...(state.messages[research.id] ?? []),
        userMessage,
        { ...run.message, content: '', status: 'streaming', evidence: [] },
      ]
      state.runs.set(run.requestId, run)
      research.last_status = 'running'
      research.updated_at = iso()
      research.messages_count = state.messages[research.id].length
      return json(
        res,
        {
          request_id: run.requestId,
          user_message: userMessage,
          assistant_message_id: run.assistantId,
          events_url: `/api/requests/${run.requestId}/events`,
          research: { id: research.id, updated_at: research.updated_at },
        },
        202,
      )
    }
    if (method === 'GET') return json(res, research)
    if (method === 'PATCH') {
      const body = await readBody(req)
      const title = typeof body.title === 'string' ? body.title.trim() : ''
      if (!title || title.length > 200) return fail(res, 422, 'validation_error', 'Некорректное название')
      research.title = title
      research.updated_at = iso()
      return json(res, research)
    }
    if (method === 'DELETE') {
      state.researches = state.researches.filter((item) => item.id !== research.id)
      delete state.messages[research.id]
      res.writeHead(204)
      return res.end()
    }
  }

  const requestMatch = /^\/api\/requests\/([^/]+)(\/events|\/cancel)?$/.exec(path)
  if (requestMatch) {
    const run = state.runs.get(decodeURIComponent(requestMatch[1]))
    if (!run) return fail(res, 404, 'request_not_found', 'Запрос не найден')
    const suffix = requestMatch[2]
    if (suffix === '/events' && method === 'GET') {
      const lastEventId = Number(req.headers['last-event-id'] ?? 0)
      return sse(res, run, lastEventId)
    }
    if (suffix === '/cancel' && method === 'POST') {
      run.cancelled = true
      run.status = 'cancelled'
      return json(res, { request_id: run.requestId, status: 'cancelled' })
    }
    if (!suffix && method === 'GET') {
      return json(res, {
        request_id: run.requestId,
        status: run.status,
        message_id: run.assistantId,
        message: run.status === 'running' ? null : run.message,
        error: run.message.error ?? null,
      })
    }
  }

  if (method === 'GET' && path.startsWith('/api/evidence/')) {
    const chunkId = decodeURIComponent(path.replace('/api/evidence/', ''))
    const detail = evidenceDetails[chunkId]
    if (!detail) return fail(res, 404, 'not_found', 'Источник не найден')
    return json(res, detail)
  }

  return fail(res, 404, 'not_found', `Fixture не знает ${method} ${path}`)
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[fixture] listening on http://127.0.0.1:${PORT}`)
})
