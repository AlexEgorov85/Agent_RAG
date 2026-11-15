import { ApiError, apiRequestRaw, isAbortError } from './api-client'
import { SseParser, parseSseJson } from './sse-parser'
import { isAgentEventEnvelope, type AgentEventEnvelope } from '../types/events'

export type StreamOutcome =
  /** Server closed the stream without a replay/resync signal. */
  | { kind: 'ended' }
  | { kind: 'resync'; reason: string }
  /** 410 `request_not_replayable`: replay buffer is gone. */
  | { kind: 'gone'; error: ApiError }
  | { kind: 'http_error'; error: ApiError }
  | { kind: 'network_error'; error: unknown }
  | { kind: 'aborted' }

export interface EventStreamOptions {
  eventsUrl: string
  lastEventId?: number
  signal?: AbortSignal
  onOpen?: () => void
}

export type EventStreamItem =
  | { kind: 'event'; event: AgentEventEnvelope }
  | { kind: 'resync'; reason: string }
  | { kind: 'outcome'; outcome: StreamOutcome }

export interface ConsumeEventStreamOptions extends EventStreamOptions {
  onEvent: (event: AgentEventEnvelope) => void
  onResync?: (reason: string) => void
}

function resyncReason(raw: string): string {
  const parsed = parseSseJson(raw)
  if (parsed && typeof parsed === 'object') {
    const reason = (parsed as { reason?: unknown }).reason
    if (typeof reason === 'string' && reason) return reason
  }
  return raw || 'resync'
}

/**
 * Pull-based SSE reader for the FastAPI adapter.
 *
 * Yields each public event as it arrives, so the assistant-ui adapter can
 * publish cumulative message parts without buffering the whole run.
 */
export async function* iterateEventStream(
  options: EventStreamOptions,
): AsyncGenerator<EventStreamItem, void> {
  const headers: Record<string, string> = { Accept: 'text/event-stream' }
  if (options.lastEventId !== undefined && options.lastEventId > 0) {
    headers['Last-Event-ID'] = String(options.lastEventId)
  }

  let response: Response
  try {
    response = await apiRequestRaw(options.eventsUrl, {
      method: 'GET',
      headers,
      signal: options.signal,
      cache: 'no-store',
    })
  } catch (error) {
    if (isAbortError(error)) {
      yield { kind: 'outcome', outcome: { kind: 'aborted' } }
      return
    }
    if (error instanceof ApiError && error.status === 410) {
      yield { kind: 'outcome', outcome: { kind: 'gone', error } }
      return
    }
    yield { kind: 'outcome', outcome: { kind: 'network_error', error } }
    return
  }

  if (!response.ok) {
    const error = await ApiError.fromResponse(response)
    if (response.status === 410) {
      yield { kind: 'outcome', outcome: { kind: 'gone', error } }
      return
    }
    yield { kind: 'outcome', outcome: { kind: 'http_error', error } }
    return
  }

  options.onOpen?.()

  if (!response.body) {
    yield {
      kind: 'outcome',
      outcome: {
        kind: 'network_error',
        error: new Error('Поток событий не содержит тела ответа.'),
      },
    }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = new SseParser()

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      for (const message of parser.push(text)) {
        if (message.event === 'resync') {
          yield { kind: 'resync', reason: resyncReason(message.data) }
          return
        }
        const payload = parseSseJson(message.data)
        if (!isAgentEventEnvelope(payload)) continue
        yield { kind: 'event', event: payload }
      }
    }
    yield { kind: 'outcome', outcome: { kind: 'ended' } }
  } catch (error) {
    if (isAbortError(error)) {
      yield { kind: 'outcome', outcome: { kind: 'aborted' } }
    } else {
      yield { kind: 'outcome', outcome: { kind: 'network_error', error } }
    }
  } finally {
    try {
      await reader.cancel()
    } catch {
      // The stream is already closed or cancelled.
    }
  }
}

/**
 * Convenience wrapper kept for callers that only need a callback.
 * Reconnection policy lives above this layer.
 */
export async function consumeEventStream(
  options: ConsumeEventStreamOptions,
): Promise<StreamOutcome> {
  for await (const item of iterateEventStream(options)) {
    if (item.kind === 'event') {
      options.onEvent(item.event)
      continue
    }
    if (item.kind === 'resync') {
      options.onResync?.(item.reason)
      return { kind: 'resync', reason: item.reason }
    }
    return item.outcome
  }
  return { kind: 'ended' }
}
