import { afterEach, describe, expect, it, vi } from 'vitest'
import { consumeEventStream } from '../../src/services/event-stream'
import type { AgentEventEnvelope } from '../../src/types/events'

function sseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(stream, {
    status,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function eventFrame(sequence: number, payload: Record<string, unknown>): string {
  return `id: ${sequence}\nevent: agent\ndata: ${JSON.stringify({
    contract_version: '1',
    sequence,
    request_id: 'req-1',
    ...payload,
  })}\n\n`
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('consumeEventStream', () => {
  it('отправляет Last-Event-ID и возвращает события по порядку', async () => {
    const fetchMock = vi.fn(async () =>
      sseResponse([
        ': keep-alive\n\n',
        eventFrame(6, { type: 'answer_started' }),
        eventFrame(7, { type: 'run_completed', answer_status: 'answer', duration_ms: 10, message: {} }),
      ]),
    )
    vi.stubGlobal('fetch', fetchMock)

    const received: AgentEventEnvelope[] = []
    const outcome = await consumeEventStream({
      eventsUrl: '/api/requests/req-1/events',
      lastEventId: 5,
      onEvent: (event) => received.push(event),
    })

    expect(outcome).toEqual({ kind: 'ended' })
    expect(received.map((event) => event.sequence)).toEqual([6, 7])

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    const headers = new Headers(init.headers)
    expect(headers.get('Last-Event-ID')).toBe('5')
    expect(headers.get('Accept')).toBe('text/event-stream')
  })

  it('сообщает о resync с причиной и прекращает чтение', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => sseResponse(['event: resync\ndata: {"reason":"buffer_overflow"}\n\n'])),
    )
    const onResync = vi.fn()
    const onEvent = vi.fn()
    const outcome = await consumeEventStream({
      eventsUrl: '/api/requests/req-1/events',
      onEvent,
      onResync,
    })
    expect(outcome.kind).toBe('resync')
    expect(onResync).toHaveBeenCalledWith('buffer_overflow')
    expect(onEvent).not.toHaveBeenCalled()
  })

  it('410 превращается в gone без падения', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            error: {
              code: 'request_not_replayable',
              message: 'События недоступны',
              retryable: false,
              request_id: 'req-1',
            },
          }),
          { status: 410, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )
    const outcome = await consumeEventStream({
      eventsUrl: '/api/requests/req-1/events',
      onEvent: () => {},
    })
    expect(outcome.kind).toBe('gone')
    if (outcome.kind === 'gone') {
      expect(outcome.error.code).toBe('request_not_replayable')
      expect(outcome.error.requestId).toBe('req-1')
    }
  })

  it('игнорирует SSE-сообщения без валидного конверта', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([
          'event: agent\ndata: не-json\n\n',
          eventFrame(2, { type: 'run_started', message_id: 'm-1' }),
        ]),
      ),
    )
    const onEvent = vi.fn()
    const outcome = await consumeEventStream({
      eventsUrl: '/api/requests/req-1/events',
      onEvent,
    })
    expect(outcome.kind).toBe('ended')
    expect(onEvent).toHaveBeenCalledTimes(1)
  })

  it('сетевой сбой возвращается как network_error, а не исключение', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    const outcome = await consumeEventStream({
      eventsUrl: '/api/requests/req-1/events',
      onEvent: () => {},
    })
    expect(outcome.kind).toBe('network_error')
  })
})
