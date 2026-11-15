import type { ChatModelAdapter, ChatModelRunResult, ThreadMessage } from '@assistant-ui/react'
import { ApiError, isAbortError } from '../../services/api-client'
import { iterateEventStream, type StreamOutcome } from '../../services/event-stream'
import { messageApi } from '../../services/message-api'
import { compactFilters, isEmptyFilters } from '../../lib/filters'
import { newClientMessageId } from '../../lib/ids'
import { describeApiError } from '../../lib/errors'
import type { ChatMessage, FilterSet } from '../../types/api'
import { isTerminalEvent } from '../../types/events'
import {
  applyInterrupted,
  applyResync,
  applyRestoredMessage,
  applyRunEvent,
  applyTransportError,
  buildMessageParts,
  buildRunStatus,
  createRunProjection,
  isTerminalRunPhase,
  markReconnecting,
  type RunProjection,
} from './run-projection'
import type { RunErrorData } from './data-parts'

const MAX_RECONNECT_ATTEMPTS = 4
const MAX_STATUS_POLLS = 4

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

function reconnectDelay(attempt: number): number {
  return Math.min(500 * 2 ** Math.max(attempt - 1, 0), 4000)
}

export function lastUserText(messages: readonly ThreadMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (!message || message.role !== 'user') continue
    const text = message.content
      .flatMap((part) => (part.type === 'text' ? [part.text] : []))
      .join('\n')
      .trim()
    if (text) return text
  }
  return ''
}

function toRunResult(state: RunProjection): ChatModelRunResult {
  return {
    content: buildMessageParts(state),
    status: buildRunStatus(state),
  }
}

type Recovery =
  | { kind: 'message'; message: ChatMessage }
  | { kind: 'error'; error: RunErrorData }
  | { kind: 'none' }

async function recoverTerminal(requestId: string, signal: AbortSignal): Promise<Recovery> {
  for (let attempt = 0; attempt < MAX_STATUS_POLLS; attempt += 1) {
    if (signal.aborted) return { kind: 'none' }
    try {
      const status = await messageApi.requestStatus(requestId)
      if (status.message) return { kind: 'message', message: status.message }
      if (status.status === 'failed' && status.error) {
        return {
          kind: 'error',
          error: {
            code: status.error.code,
            message: status.error.message,
            request_id: status.error.request_id ?? requestId,
            retryable: status.error.retryable,
          },
        }
      }
      if (status.status === 'completed' || status.status === 'abstained' || status.status === 'cancelled') {
        return { kind: 'none' }
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return { kind: 'none' }
    }
    await sleep(600 * (attempt + 1))
  }
  return { kind: 'none' }
}

export interface ResearchChatAdapterOptions {
  researchId: string | null
  getFilters: () => FilterSet
  /** Called when a run settles (complete, failed, cancelled) to refresh backend history. */
  onSettled?: (researchId: string) => void
}

/**
 * Custom transport for the FastAPI SSE adapter.
 *
 * The backend is not Vercel AI SDK: `POST /api/research/{id}/messages` starts a
 * run, and `GET /api/requests/{id}/events` streams the public events. This
 * adapter maps them to assistant-ui message parts:
 *
 *   run_started / analysis_completed / tool_* / answer_* → `execution` data part
 *   answer_delta                                        → cumulative text part
 *   run_completed.message.content                       → final text part
 *   evidence                                            → `evidence` data part
 *   analysis.coverage / verification                    → data parts
 *   run_failed / run_cancelled                          → `run_error` / `answer_state`
 *
 * assistant-ui owns the thread and streaming state; no parallel chat reducer
 * is used by the UI.
 */
export function createResearchChatAdapter(
  options: ResearchChatAdapterOptions,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }): AsyncGenerator<ChatModelRunResult, void> {
      const researchId = options.researchId
      if (!researchId) throw new Error('Диалог не выбран.')
      const content = lastUserText(messages)
      if (!content) throw new Error('Пустой запрос: нет текста пользовательского сообщения.')

      const filters = compactFilters(options.getFilters())

      let response
      try {
        response = await messageApi.send(researchId, {
          content,
          client_message_id: newClientMessageId(),
          ...(isEmptyFilters(filters) ? {} : { filters }),
        })
      } catch (error) {
        if (isAbortError(error)) return
        throw new Error(describeApiError(error))
      }

      let state = createRunProjection(response, filters)
      yield toRunResult(state)

      const controller = new AbortController()
      const cancelOnAbort = () => {
        void messageApi.cancel(response.request_id).catch(() => {
          // Cancellation is best effort; the UI already reflects the abort.
        })
        controller.abort()
      }
      if (abortSignal.aborted) {
        controller.abort()
      } else {
        abortSignal.addEventListener('abort', cancelOnAbort)
      }

      try {
        let attempts = 0
        for (;;) {
          if (controller.signal.aborted) return

          let outcome: StreamOutcome | null = null
          for await (const item of iterateEventStream({
            eventsUrl: response.events_url,
            lastEventId: state.lastSequence,
            signal: controller.signal,
          })) {
            if (item.kind === 'event') {
              state = applyRunEvent(state, item.event)
              yield toRunResult(state)
              if (isTerminalEvent(item.event)) return
              continue
            }
            if (item.kind === 'resync') {
              state = applyResync(state, item.reason)
              yield toRunResult(state)
              break
            }
            outcome = item.outcome
            break
          }

          if (controller.signal.aborted) return
          if (isTerminalRunPhase(state.phase)) return

          if (
            !outcome ||
            outcome.kind === 'ended' ||
            outcome.kind === 'gone' ||
            outcome.kind === 'resync'
          ) {
            if (outcome?.kind === 'gone') {
              state = applyResync(state, 'request_not_replayable')
              yield toRunResult(state)
            }
            const recovery = await recoverTerminal(response.request_id, controller.signal)
            if (controller.signal.aborted) return
            if (recovery.kind === 'message') {
              state = applyRestoredMessage(state, recovery.message)
              yield toRunResult(state)
              return
            }
            if (recovery.kind === 'error') {
              state = applyTransportError(state, recovery.error)
              yield toRunResult(state)
              return
            }
            state = applyInterrupted(state)
            yield toRunResult(state)
            return
          }

          if (outcome.kind === 'aborted') return

          attempts += 1
          if (outcome.kind === 'http_error' && !outcome.error.retryable) {
            state = applyTransportError(state, {
              code: outcome.error.code,
              message: outcome.error.message,
              request_id: outcome.error.requestId ?? response.request_id,
              retryable: outcome.error.retryable,
            })
            yield toRunResult(state)
            return
          }
          if (attempts > MAX_RECONNECT_ATTEMPTS) {
            state = applyTransportError(state, {
              code: 'connection_lost',
              message:
                'Не удалось восстановить поток событий. Результат можно проверить после обновления страницы.',
              request_id: response.request_id,
              retryable: true,
            })
            yield toRunResult(state)
            return
          }

          state = markReconnecting(state)
          yield toRunResult(state)
          await sleep(reconnectDelay(attempts))
        }
      } catch (error) {
        if (isAbortError(error) || controller.signal.aborted) return
        throw error
      } finally {
        abortSignal.removeEventListener('abort', cancelOnAbort)
        options.onSettled?.(researchId)
      }
    },
  }
}
