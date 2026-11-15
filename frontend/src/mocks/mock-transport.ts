import {
  MOCK_BANNER,
  mockAbstainAnswer,
  mockAnswerMarkdown,
  mockCapabilities,
  mockCorpus,
  mockEvidence,
  mockEvidenceDetails,
  mockHealth,
  seedMessages,
  seedResearches,
} from './data'
import type {
  ChatMessage,
  FilterSet,
  Research,
  ResearchStatus,
  RequestStatus,
  SendMessageResponse,
} from '../types/api'
import type { EvidenceItem } from '../types/evidence'
import type { AgentEvent, AgentEventEnvelope } from '../types/events'
import { isTerminalEvent } from '../types/events'

/**
 * In-memory fixture backend for the explicit mock mode.
 *
 * It mimics HTTP/SSE contracts only; it contains no retrieval or generation
 * logic. Research data created here lives in memory and disappears on reload.
 */

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

interface MockRun {
  requestId: string
  researchId: string
  assistantMessageId: string
  events: AgentEventEnvelope[]
  status: RequestStatus
  cancelled: boolean
  terminalMessage: ChatMessage
  error: ChatMessage['error'] | null
}

interface MockState {
  researches: Research[]
  messages: Record<string, ChatMessage[]>
  runs: Map<string, MockRun>
}

const state: MockState = {
  researches: clone(seedResearches),
  messages: clone(seedMessages),
  runs: new Map(),
}

const nowIso = (): string => new Date().toISOString()

function newId(prefix: string): string {
  const uuid =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)
  return `${prefix}-${uuid}`
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function apiError(
  status: number,
  code: string,
  message: string,
  extra: Record<string, unknown> = {},
): Response {
  return json(
    {
      error: { code, message, retryable: status >= 500, ...extra },
      mock: MOCK_BANNER,
    },
    status,
  )
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

function parseBody(init: RequestInit): Record<string, unknown> {
  if (typeof init.body !== 'string' || !init.body) return {}
  try {
    return JSON.parse(init.body) as Record<string, unknown>
  } catch {
    return {}
  }
}

function findResearch(id: string): Research | undefined {
  return state.researches.find((research) => research.id === id)
}

function delayFor(type: string): number {
  switch (type) {
    case 'run_started':
      return 120
    case 'analysis_completed':
      return 260
    case 'tool_started':
      return 420
    case 'tool_completed':
      return 380
    case 'answer_started':
      return 220
    case 'run_completed':
      return 320
    default:
      return 180
  }
}

const EVENT_SEQUENCE: AgentEvent['type'][] = [
  'run_started',
  'analysis_completed',
  'tool_started',
  'tool_completed',
  'answer_started',
  'run_completed',
]

function buildRun(
  research: Research,
  content: string,
  filters: FilterSet,
): { run: MockRun; userMessage: ChatMessage; assistantId: string } {
  const requestId = newId('mock-request')
  const assistantMessageId = newId('msg-assistant')
  const createdAt = nowIso()
  const abstain = /2021/.test(content)
  const evidence: EvidenceItem[] = abstain ? [] : clone(mockEvidence)

  const userMessage: ChatMessage = {
    id: newId('msg-user'),
    research_id: research.id,
    role: 'user',
    content,
    created_at: createdAt,
    status: 'completed',
    evidence: [],
    warnings: [],
  }

  const terminalMessage: ChatMessage = {
    id: assistantMessageId,
    research_id: research.id,
    role: 'assistant',
    content: abstain ? mockAbstainAnswer : mockAnswerMarkdown,
    created_at: createdAt,
    status: abstain ? 'abstained' : 'completed',
    request_id: requestId,
    effective_filters: filters,
    evidence,
    warnings: abstain ? ['no_evidence_for_period'] : [],
  }

  const makeEvent = (type: string): AgentEvent => {
    switch (type) {
      case 'run_started':
        return { type: 'run_started', request_id: requestId, message_id: assistantMessageId }
      case 'analysis_completed':
        return {
          type: 'analysis_completed',
          request_id: requestId,
          intent: /сравн/i.test(content) ? 'temporal' : 'factoid',
          effective_filters: filters,
        }
      case 'tool_started':
        return {
          type: 'tool_started',
          request_id: requestId,
          tool_name: 'search_reports',
          short_arguments: { years: filters.years ?? [], query: 'демонстрационный запрос' },
        }
      case 'tool_completed':
        return {
          type: 'tool_completed',
          request_id: requestId,
          tool_name: 'search_reports',
          status: 'success',
          result_count: evidence.length,
          duration_ms: 840,
        }
      case 'answer_started':
        return { type: 'answer_started', request_id: requestId }
      case 'run_completed':
        return {
          type: 'run_completed',
          request_id: requestId,
          answer_status: abstain ? 'abstain' : 'answer',
          duration_ms: 1520,
          message: terminalMessage,
        }
      default:
        return { type: 'run_started', request_id: requestId, message_id: assistantMessageId }
    }
  }

  const events: AgentEventEnvelope[] = EVENT_SEQUENCE.map((type, index) => ({
    sequence: index + 1,
    ...makeEvent(type),
  })) as AgentEventEnvelope[]

  const run: MockRun = {
    requestId,
    researchId: research.id,
    assistantMessageId,
    events,
    status: 'running',
    cancelled: false,
    terminalMessage,
    error: null,
  }
  return { run, userMessage, assistantId: assistantMessageId }
}

function appendStreamingAssistant(run: MockRun, research: Research): void {
  const existing = state.messages[research.id] ?? []
  existing.push({
    ...run.terminalMessage,
    content: '',
    status: 'streaming',
    evidence: [],
  })
  state.messages[research.id] = existing
}

function finalizeRun(run: MockRun): void {
  const research = findResearch(run.researchId)
  const messages = state.messages[run.researchId] ?? []
  const index = messages.findIndex((message) => message.id === run.assistantMessageId)
  const cancelled = run.cancelled
  const finalMessage: ChatMessage = cancelled
    ? {
        ...run.terminalMessage,
        content: '',
        status: 'cancelled',
        evidence: [],
        warnings: [],
        error: undefined,
      }
    : run.terminalMessage

  if (index >= 0) messages[index] = finalMessage
  else messages.push(finalMessage)

  run.status = cancelled
    ? 'cancelled'
    : finalMessage.status === 'abstained'
      ? 'abstained'
      : finalMessage.status === 'failed'
        ? 'failed'
        : 'completed'
  run.error = finalMessage.error ?? null

  if (research) {
    research.updated_at = nowIso()
    research.messages_count = messages.length
    research.last_status = run.status as ResearchStatus
    research.last_message_preview = finalMessage.content || research.last_message_preview
  }
}

function sseResponse(run: MockRun, init: RequestInit): Response {
  const encoder = new TextEncoder()
  const lastEventIdHeader = new Headers(init.headers).get('Last-Event-ID')
  const lastEventId = lastEventIdHeader ? Number(lastEventIdHeader) : 0
  let aborted = false

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const close = () => {
        try {
          controller.close()
        } catch {
          // Already closed.
        }
      }
      init.signal?.addEventListener('abort', () => {
        aborted = true
        close()
      })

      void (async () => {
        for (const event of run.events) {
          if (event.sequence <= lastEventId) continue
          if (lastEventId === 0) await sleep(delayFor(event.type))
          if (aborted) return

          if (isTerminalEvent(event)) {
            finalizeRun(run)
            if (run.cancelled) {
              const cancelled: AgentEventEnvelope = {
                type: 'run_cancelled',
                request_id: run.requestId,
                sequence: event.sequence,
              }
              controller.enqueue(
                encoder.encode(
                  `id: ${cancelled.sequence}\nevent: agent\ndata: ${JSON.stringify(cancelled)}\n\n`,
                ),
              )
              close()
              return
            }
          }
          controller.enqueue(
            encoder.encode(`id: ${event.sequence}\nevent: agent\ndata: ${JSON.stringify(event)}\n\n`),
          )
        }
        close()
      })()
    },
  })

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    },
  })
}

function listResearches(url: URL): Response {
  const q = (url.searchParams.get('q') ?? '').trim().toLowerCase()
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 50) || 50, 200)
  const offset = Number(url.searchParams.get('offset') ?? 0) || 0

  const matches = state.researches
    .filter((research) => {
      if (!q) return true
      if (research.title.toLowerCase().includes(q)) return true
      if ((research.last_message_preview ?? '').toLowerCase().includes(q)) return true
      const messages = state.messages[research.id] ?? []
      return messages.some((message) => message.content.toLowerCase().includes(q))
    })
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))

  const page = matches.slice(offset, offset + limit)
  return json({
    items: page,
    total: matches.length,
    next_offset: offset + limit < matches.length ? offset + limit : null,
  })
}

function capabilitiesResponse(): Response {
  return json(mockCapabilities)
}

function corpusResponse(): Response {
  return json(mockCorpus)
}

function createResearch(body: Record<string, unknown>): Response {
  const rawTitle = typeof body.title === 'string' ? body.title.trim() : ''
  if (rawTitle.length > 200) {
    return apiError(422, 'validation_error', 'Название диалога длиннее 200 символов.')
  }
  const timestamp = nowIso()
  const research: Research = {
    id: newId('research'),
    title: rawTitle || 'Новый диалог',
    created_at: timestamp,
    updated_at: timestamp,
    last_status: 'idle',
    messages_count: 0,
  }
  state.researches.unshift(research)
  state.messages[research.id] = []
  return json(research, 201)
}

function sendMessage(research: Research, body: Record<string, unknown>): Response {
  const content = typeof body.content === 'string' ? body.content.trim() : ''
  if (!content) return apiError(422, 'validation_error', 'Текст сообщения пуст.')
  if (content.length > 16000) {
    return apiError(422, 'validation_error', 'Сообщение длиннее 16 000 символов.')
  }
  const filters = (body.filters ?? {}) as FilterSet

  const active = [...state.runs.values()].find(
    (run) => run.researchId === research.id && run.status === 'running',
  )
  if (active && active.status === 'running') {
    return apiError(409, 'request_in_progress', 'В этом диалоге уже выполняется запрос.', {
      request_id: active.requestId,
      details: { active_request_id: active.requestId },
    })
  }

  const { run, userMessage, assistantId } = buildRun(research, content, filters)
  const messages = state.messages[research.id] ?? []
  messages.push(userMessage)
  state.messages[research.id] = messages
  state.runs.set(run.requestId, run)

  appendStreamingAssistant(run, research)
  research.updated_at = nowIso()
  research.messages_count = messages.length + 1
  research.last_status = 'running'
  research.last_message_preview = content

  const response: SendMessageResponse = {
    request_id: run.requestId,
    user_message: userMessage,
    assistant_message_id: assistantId,
    events_url: `/api/requests/${run.requestId}/events`,
    research: { id: research.id, updated_at: research.updated_at },
  }
  return json(response, 202)
}

function requestStatus(run: MockRun | undefined): Response {
  if (!run) return apiError(404, 'request_not_found', 'Запрос не найден.')
  const research = findResearch(run.researchId)
  const messages = research ? (state.messages[research.id] ?? []) : []
  const message = messages.find((item) => item.id === run.assistantMessageId) ?? null
  return json({
    request_id: run.requestId,
    status: run.status,
    message_id: run.assistantMessageId,
    message: run.status === 'running' ? null : message,
    error: run.error,
  })
}

function cancelRequest(run: MockRun | undefined): Response {
  if (!run) return apiError(404, 'request_not_found', 'Запрос не найден.')
  if (run.status !== 'running') return json({ request_id: run.requestId, status: run.status })
  run.cancelled = true
  const research = findResearch(run.researchId)
  if (research) research.last_status = 'cancelled'
  return json({ request_id: run.requestId, status: 'cancelled' })
}

const RESEARCH_PATH = /^\/api\/research(?:\/([^/]+))?(?:\/(messages))?$/

export async function mockFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const method = (init.method ?? 'GET').toUpperCase()
  const parsed = new URL(url, 'http://mock.local')
  const path = parsed.pathname

  if (method === 'GET' && path === '/api/health') return json(mockHealth)
  if (method === 'GET' && path === '/api/capabilities') return capabilitiesResponse()
  if (method === 'GET' && path === '/api/corpus') return corpusResponse()

  const evidenceMatch = /^\/api\/evidence\/(.+)$/.exec(path)
  if (method === 'GET' && evidenceMatch?.[1]) {
    const chunkId = decodeURIComponent(evidenceMatch[1])
    const detail = mockEvidenceDetails[chunkId]
    if (!detail) return apiError(404, 'not_found', 'Источник не найден в демонстрационном наборе.')
    return json(detail)
  }

  if (path === '/api/research' && method === 'GET') return listResearches(parsed)
  if (path === '/api/research' && method === 'POST') return createResearch(parseBody(init))

  const researchMatch = RESEARCH_PATH.exec(path)
  if (researchMatch) {
    const researchId = researchMatch[1] ? decodeURIComponent(researchMatch[1]) : null
    const isMessages = Boolean(researchMatch[2])
    if (!researchId) return json({ items: [], total: 0, next_offset: null })
    const research = findResearch(researchId)
    if (!research) return apiError(404, 'not_found', 'Диалог не найден.')

    if (isMessages && method === 'GET') {
      return json({ items: state.messages[research.id] ?? [], total: 0, next_before: null })
    }
    if (isMessages && method === 'POST') return sendMessage(research, parseBody(init))
    if (method === 'GET') return json(research)
    if (method === 'PATCH') {
      const body = parseBody(init)
      const title = typeof body.title === 'string' ? body.title.trim() : ''
      if (!title || title.length > 200) {
        return apiError(422, 'validation_error', 'Название должно быть от 1 до 200 символов.')
      }
      research.title = title
      research.updated_at = nowIso()
      return json(research)
    }
    if (method === 'DELETE') {
      state.researches = state.researches.filter((item) => item.id !== researchId)
      delete state.messages[researchId]
      return new Response(null, { status: 204 })
    }
  }

  const requestMatch = /^\/api\/requests\/([^/]+)(\/events|\/cancel)?$/.exec(path)
  if (requestMatch?.[1]) {
    const requestId = decodeURIComponent(requestMatch[1])
    const run = state.runs.get(requestId)
    const suffix = requestMatch[2]
    if (!run) return apiError(404, 'request_not_found', 'Запрос не найден.')
    if (suffix === '/events' && method === 'GET') return sseResponse(run, init)
    if (suffix === '/cancel' && method === 'POST') return cancelRequest(run)
    if (!suffix && method === 'GET') return requestStatus(run)
  }

  return apiError(404, 'not_found', `Демонстрационный backend не знает путь ${method} ${path}.`)
}
