import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChatModelRunOptions } from '@assistant-ui/react'
import { createResearchChatAdapter } from '../../src/features/assistant-runtime/chat-adapter'
import type { ChatModelRunResult, ThreadMessage } from '@assistant-ui/react'
import { fullEvidence } from './fixtures/messages'

const REQUEST_ID = 'req-adapter'

function sseResponse(frames: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame))
      controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function frame(sequence: number, payload: Record<string, unknown>, requestId = REQUEST_ID): string {
  return `id: ${sequence}\nevent: agent\ndata: ${JSON.stringify({
    contract_version: '1',
    sequence,
    request_id: requestId,
    ...payload,
  })}\n\n`
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function postResponse() {
  return {
    request_id: REQUEST_ID,
    assistant_message_id: 'msg-a',
    events_url: `/api/requests/${REQUEST_ID}/events`,
    user_message: {
      id: 'msg-u',
      research_id: 'research-1',
      role: 'user',
      content: 'Какая чистая прибыль за 2023 год?',
      created_at: '2026-11-14T09:59:00Z',
      status: 'completed',
      evidence: [],
      warnings: [],
    },
    research: { id: 'research-1', updated_at: '2026-11-14T10:00:00Z' },
  }
}

function options(): ChatModelRunOptions {
  const userMessage = {
    id: 'runtime-user-1',
    role: 'user',
    createdAt: new Date('2026-11-14T09:59:00Z'),
    content: [{ type: 'text', text: 'Какая чистая прибыль за 2023 год?' }],
    attachments: [],
    metadata: {},
    status: { type: 'complete', reason: 'stop' },
  } as unknown as ThreadMessage

  return {
    messages: [userMessage],
    abortSignal: new AbortController().signal,
    runConfig: {},
    context: {},
  } as unknown as ChatModelRunOptions
}

async function collect(
  adapter: ReturnType<typeof createResearchChatAdapter>,
  runOptions: ChatModelRunOptions,
): Promise<ChatModelRunResult[]> {
  const generator = adapter.run(runOptions) as AsyncGenerator<ChatModelRunResult>
  const results: ChatModelRunResult[] = []
  for await (const result of generator) results.push(result)
  return results
}

function dataPart(result: ChatModelRunResult, name: string): Record<string, unknown> | undefined {
  const part = result.content?.find((item) => item.type === 'data' && item.name === name)
  return part && part.type === 'data' ? (part.data as Record<string, unknown>) : undefined
}

afterEach(() => vi.unstubAllGlobals())

describe('custom SSE transport → assistant-ui ChatModelAdapter', () => {
  it('публикует этапы, инструменты, текст и evidence; дубли sequence игнорируются', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/messages')) return jsonResponse(postResponse(), 202)
      if (url.includes('/events')) {
        return sseResponse([
          frame(1, { type: 'run_started', message_id: 'msg-a' }),
          frame(1, { type: 'run_started', message_id: 'msg-a' }),
          // Foreign request must not leak into this run.
          frame(2, { type: 'tool_started', tool_name: 'get_chunk' }, 'req-other'),
          frame(2, { type: 'tool_started', tool_name: 'search_reports', short_arguments: { years: [2023] } }),
          frame(3, { type: 'tool_completed', tool_name: 'search_reports', status: 'success', result_count: 7, duration_ms: 812 }),
          frame(4, { type: 'answer_delta', delta: 'Прибыль ' }),
          frame(5, { type: 'answer_delta', delta: 'выросла.' }),
          frame(6, {
            type: 'run_completed',
            answer_status: 'answer',
            duration_ms: 1500,
            message: {
              id: 'msg-a',
              research_id: 'research-1',
              role: 'assistant',
              content: 'Чистая прибыль составила 1 508,6 млрд руб.',
              created_at: '2026-11-14T10:00:00Z',
              status: 'completed',
              request_id: REQUEST_ID,
              evidence: [fullEvidence],
              warnings: [],
            },
          }),
        ])
      }
      throw new Error(`unexpected fetch ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createResearchChatAdapter({
      researchId: 'research-1',
      getFilters: () => ({ years: [2023] }),
    })
    const results = await collect(adapter, options())
    const last = results[results.length - 1]

    expect(last?.status).toEqual({ type: 'complete', reason: 'stop' })
    const execution = dataPart(last!, 'execution')
    expect(execution?.phase).toBe('completed')
    expect(execution?.sequence).toBe(6)
    const tools = execution?.tools as Array<Record<string, unknown>>
    expect(tools).toHaveLength(1)
    expect(tools[0]).toMatchObject({ tool_name: 'search_reports', result_count: 7 })

    const text = last?.content?.find((part) => part.type === 'text')
    expect(text).toMatchObject({ text: 'Чистая прибыль составила 1 508,6 млрд руб.' })
    expect((dataPart(last!, 'evidence')?.items as unknown[]).length).toBe(1)

    const postCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/messages'))
    const init = postCall?.[1] as RequestInit
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body.content).toBe('Какая чистая прибыль за 2023 год?')
    expect(body.filters).toEqual({ years: [2023] })
    expect(typeof body.client_message_id).toBe('string')
  })

  it('run_failed отдаёт ошибку с кодом, request_id и retryable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/messages')) return jsonResponse(postResponse(), 202)
        return sseResponse([
          frame(1, { type: 'run_started', message_id: 'msg-a' }),
          frame(2, {
            type: 'run_failed',
            code: 'backend_unavailable',
            message: 'LLM недоступен',
            retryable: true,
          }),
        ])
      }),
    )
    const adapter = createResearchChatAdapter({ researchId: 'research-1', getFilters: () => ({}) })
    const results = await collect(adapter, options())
    const last = results[results.length - 1]
    expect(last?.status).toEqual({ type: 'incomplete', reason: 'error' })
    expect(dataPart(last!, 'run_error')).toMatchObject({
      code: 'backend_unavailable',
      message: 'LLM недоступен',
      request_id: REQUEST_ID,
      retryable: true,
    })
    expect(dataPart(last!, 'answer_state')).toMatchObject({ state: 'failed' })
  })

  it('run_cancelled переводит сообщение в честное состояние cancelled', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/messages')) return jsonResponse(postResponse(), 202)
        return sseResponse([
          frame(1, { type: 'run_started', message_id: 'msg-a' }),
          frame(2, { type: 'run_cancelled' }),
        ])
      }),
    )
    const adapter = createResearchChatAdapter({ researchId: 'research-1', getFilters: () => ({}) })
    const results = await collect(adapter, options())
    const last = results[results.length - 1]
    expect(last?.status).toEqual({ type: 'incomplete', reason: 'cancelled' })
    expect(dataPart(last!, 'answer_state')).toMatchObject({ state: 'cancelled' })
  })

  it('ошибка POST (409) доходит до UI с кодом', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(
          {
            error: {
              code: 'request_in_progress',
              message: 'В этом диалоге уже выполняется запрос.',
              request_id: 'req-active',
              retryable: true,
            },
          },
          409,
        ),
      ),
    )
    const adapter = createResearchChatAdapter({ researchId: 'research-1', getFilters: () => ({}) })
    await expect(collect(adapter, options())).rejects.toThrow(/request_in_progress/)
  })
})
