import { describe, expect, it } from 'vitest'
import type { SendMessageResponse } from '../../src/types/api'
import type { AgentEventEnvelope } from '../../src/types/events'
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
} from '../../src/features/assistant-runtime/run-projection'
import {
  analysisCompleted,
  answerStarted,
  assistantMessage,
  fullEvidence,
  runCompleted,
  runStarted,
  toolCompleted,
  toolStarted,
} from './fixtures/messages'

const RESPONSE: SendMessageResponse = {
  request_id: 'req-1',
  assistant_message_id: 'msg-assistant',
  events_url: '/api/requests/req-1/events',
  user_message: {
    id: 'msg-user',
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

function base(requestedFilters = { years: [2023] }): RunProjection {
  return createRunProjection(RESPONSE, requestedFilters)
}

function apply(state: RunProjection, event: AgentEventEnvelope): RunProjection {
  return applyRunEvent(state, event)
}

function dataOf(parts: readonly { type: string; name?: string; data?: unknown }[], name: string) {
  return parts.find((part) => part.type === 'data' && part.name === name)?.data as
    | Record<string, unknown>
    | undefined
}

describe('run projection: SSE lifecycle → assistant-ui parts', () => {
  it('маппит события в execution/text/evidence/analysis части', () => {
    let state = base()
    state = apply(state, runStarted)
    state = apply(state, analysisCompleted)
    state = apply(state, toolStarted)
    state = apply(state, toolCompleted)
    state = apply(state, answerStarted)

    let parts = buildMessageParts(state)
    const execution = dataOf(parts, 'execution')
    expect(execution).toMatchObject({ request_id: 'req-1', phase: 'streaming', intent: 'Запрос факта' })
    expect((execution?.tools as Array<Record<string, unknown>>)[0]).toMatchObject({
      tool_name: 'search_reports',
      status: 'success',
      result_count: 7,
    })
    expect(parts.some((part) => part.type === 'text')).toBe(false)

    state = apply(state, {
      type: 'answer_delta',
      request_id: 'req-1',
      delta: 'Прибыль ',
      sequence: 6,
    })
    state = apply(state, {
      type: 'answer_delta',
      request_id: 'req-1',
      delta: 'выросла.',
      sequence: 7,
    })
    parts = buildMessageParts(state)
    expect(parts.find((part) => part.type === 'text')).toMatchObject({ text: 'Прибыль выросла.' })

    state = apply(state, {
      ...runCompleted({
        content: 'Прибыль выросла до 1 508,6 млрд руб.',
        evidence: [fullEvidence],
        analysis: {
          facts: [{ metric: 'profit', year: 2023, value: 1508.6, unit: 'млрд руб.', chunk_id: 'chunk-1' }],
          coverage: {
            status: 'partial',
            expected_years: [2022, 2023],
            covered_years: [2023],
            missing_years: [2022],
            metrics: {},
            missing_metrics: [],
            notes: [],
          },
        },
        warnings: ['table_row_unit_assumed'],
      }),
      sequence: 8,
    } as AgentEventEnvelope)

    expect(isTerminalRunPhase(state.phase)).toBe(true)
    expect(buildRunStatus(state)).toEqual({ type: 'complete', reason: 'stop' })

    parts = buildMessageParts(state)
    expect(parts.find((part) => part.type === 'text')).toMatchObject({
      text: 'Прибыль выросла до 1 508,6 млрд руб.',
    })
    expect((dataOf(parts, 'evidence')?.items as unknown[]).length).toBe(1)
    expect(dataOf(parts, 'analysis')?.facts).toHaveLength(1)
    expect(dataOf(parts, 'coverage')?.status).toBe('partial')
    expect(dataOf(parts, 'warnings')?.codes).toEqual(['table_row_unit_assumed'])
    expect(dataOf(parts, 'run_error')).toBeUndefined()
  })

  it('игнорирует дубли, чужие request_id и события после терминального', () => {
    let state = apply(base(), runStarted)
    const afterStart = state
    state = apply(state, runStarted)
    expect(state).toBe(afterStart)
    expect(state.lastSequence).toBe(1)

    state = apply(state, {
      type: 'tool_started',
      request_id: 'req-other',
      tool_name: 'search_reports',
      sequence: 99,
    })
    expect(state.lastSequence).toBe(1)
    expect(state.tools).toHaveLength(0)

    state = apply(state, runCompleted())
    const terminal = state
    state = apply(state, {
      type: 'answer_delta',
      request_id: 'req-1',
      delta: 'поздний текст',
      sequence: 7,
    })
    expect(state).toBe(terminal)
    expect(buildMessageParts(state).some((part) => part.type === 'text' && part.text.includes('поздний'))).toBe(false)
  })

  it('abstain и cancel отражаются честными частями и статусами', () => {
    let abstained = apply(base(), runStarted)
    abstained = apply(abstained, {
      ...runCompleted({ status: 'abstained', content: '', evidence: [] }),
      answer_status: 'abstain',
    } as AgentEventEnvelope)
    expect(buildRunStatus(abstained)).toEqual({ type: 'complete', reason: 'stop' })
    expect(dataOf(buildMessageParts(abstained), 'answer_state')).toMatchObject({
      state: 'abstained',
    })

    let cancelled = apply(base(), runStarted)
    cancelled = apply(cancelled, {
      type: 'run_cancelled',
      request_id: 'req-1',
      sequence: 2,
    })
    expect(buildRunStatus(cancelled)).toEqual({ type: 'incomplete', reason: 'cancelled' })
    expect(dataOf(buildMessageParts(cancelled), 'answer_state')).toMatchObject({
      state: 'cancelled',
    })
  })

  it('run_failed отдаёт код и request_id в красном inline-блоке', () => {
    let state = apply(base(), runStarted)
    state = apply(state, {
      type: 'run_failed',
      request_id: 'req-1',
      code: 'backend_unavailable',
      message: 'LLM недоступен',
      retryable: true,
      sequence: 2,
    })
    expect(buildRunStatus(state)).toEqual({ type: 'incomplete', reason: 'error' })
    const error = dataOf(buildMessageParts(state), 'run_error')
    expect(error).toMatchObject({
      code: 'backend_unavailable',
      message: 'LLM недоступен',
      request_id: 'req-1',
      retryable: true,
    })
    expect(dataOf(buildMessageParts(state), 'answer_state')).toMatchObject({ state: 'failed' })
  })

  it('показывает отличие применённых фильтров', () => {
    let state = base({ years: [2023] })
    state = apply(state, runStarted)
    state = apply(state, {
      type: 'analysis_completed',
      request_id: 'req-1',
      intent: 'temporal',
      effective_filters: { years: [2023, 2024] },
      sequence: 2,
    })
    const execution = dataOf(buildMessageParts(state), 'execution')
    expect(execution?.filter_notices).toEqual([
      'Годы: запрошено «2023», применено «2023, 2024».',
    ])
  })
})

describe('run projection: reconnects and REST recovery', () => {
  it('reconnecting сменяется streaming при следующем событии', () => {
    let state = apply(base(), runStarted)
    state = markReconnecting(state)
    expect(state.phase).toBe('reconnecting')
    state = apply(state, analysisCompleted)
    expect(state.phase).toBe('streaming')
  })

  it('resync показывает причину, restored подхватывает сообщение из REST', () => {
    let state = apply(base(), runStarted)
    state = applyResync(state, 'request_not_replayable')
    expect(state.phase).toBe('reconnecting')
    expect(state.resyncReason).toBe('request_not_replayable')

    state = applyRestoredMessage(state, assistantMessage({ content: 'Сохранённый ответ' }))
    expect(state.phase).toBe('completed')
    const parts = buildMessageParts(state)
    expect(parts.find((part) => part.type === 'text')).toMatchObject({ text: 'Сохранённый ответ' })
  })

  it('interrupted — ошибка с кодом, а не вечный spinner', () => {
    let state = apply(base(), runStarted)
    state = applyInterrupted(state)
    expect(state.phase).toBe('failed')
    expect(dataOf(buildMessageParts(state), 'run_error')).toMatchObject({ code: 'interrupted' })
  })

  it('transport error сохраняет код соединения', () => {
    let state = apply(base(), runStarted)
    state = applyTransportError(state, {
      code: 'connection_lost',
      message: 'поток потерян',
      request_id: 'req-1',
      retryable: true,
    })
    expect(state.phase).toBe('failed')
    expect(dataOf(buildMessageParts(state), 'run_error')?.code).toBe('connection_lost')
  })
})
