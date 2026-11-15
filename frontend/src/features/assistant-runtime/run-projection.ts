import type { MessageStatus, ThreadAssistantMessagePart } from '@assistant-ui/react'
import type { AnalysisResult } from '../../types/analytics'
import type { ChatMessage, FilterSet, SendMessageResponse } from '../../types/api'
import type { EvidenceItem } from '../../types/evidence'
import type { AgentEventEnvelope, AnalysisIntent } from '../../types/events'
import { isTerminalEvent } from '../../types/events'
import { diffFilters } from '../../lib/filters'
import {
  DATA_NAMES,
  type AnalysisData,
  type AnswerState,
  type ExecutionData,
  type RunErrorData,
  type StageSnapshot,
  type ToolSnapshot,
} from './data-parts'

/**
 * Pure projection of backend SSE events into assistant-ui message parts.
 *
 * assistant-ui owns messages and streaming; this module only enforces the
 * transport rules from the API contract: own request_id, strictly increasing
 * sequence, no events after a terminal one. Adapter yields `buildRunResult`
 * after each applied event, so the UI always renders backend state.
 */

export type RunPhase =
  | 'idle'
  | 'streaming'
  | 'reconnecting'
  | 'completed'
  | 'abstained'
  | 'failed'
  | 'cancelled'

export interface RunProjection {
  requestId: string
  assistantMessageId: string
  phase: RunPhase
  lastSequence: number
  intent?: AnalysisIntent
  effectiveFilters?: FilterSet
  requestedFilters?: FilterSet
  tools: ToolSnapshot[]
  stages: StageSnapshot[]
  streamedText: string
  evidence: EvidenceItem[]
  finalMessage?: ChatMessage
  error?: RunErrorData
  resyncReason?: string
  durationMs?: number
}

const INITIAL_STAGES: StageSnapshot[] = [
  { id: 'start', label: 'Запуск обработки', status: 'pending', source: 'run_started' },
  { id: 'analysis', label: 'Анализ запроса', status: 'pending', source: 'analysis_completed' },
  {
    id: 'search',
    label: 'Поиск и проверка данных',
    status: 'pending',
    source: 'tool_started / tool_completed',
  },
  { id: 'answer', label: 'Подготовка ответа', status: 'pending', source: 'answer_started' },
  { id: 'finish', label: 'Завершение', status: 'pending', source: 'run_completed' },
]

const TERMINAL_PHASES: ReadonlySet<RunPhase> = new Set(['completed', 'abstained', 'failed', 'cancelled'])

export function isTerminalRunPhase(phase: RunPhase): boolean {
  return TERMINAL_PHASES.has(phase)
}

export function createRunProjection(
  response: SendMessageResponse,
  requestedFilters?: FilterSet,
): RunProjection {
  return {
    requestId: response.request_id,
    assistantMessageId: response.assistant_message_id,
    phase: 'streaming',
    lastSequence: 0,
    tools: [],
    stages: INITIAL_STAGES.map((stage) =>
      stage.id === 'start' ? { ...stage, status: 'active' } : stage,
    ),
    streamedText: '',
    evidence: [],
    requestedFilters,
  }
}

function terminalStages(stages: StageSnapshot[], status: 'done' | 'failed' | 'skipped'): StageSnapshot[] {
  return stages.map((stage) => {
    if (stage.id === 'finish') return { ...stage, status }
    if (stage.status === 'active') return { ...stage, status }
    return stage
  })
}

function advanceTo(stages: StageSnapshot[], next: string, detail?: string): StageSnapshot[] {
  return stages.map((stage) => {
    if (stage.status === 'active') {
      return { ...stage, status: 'done' as const }
    }
    if (stage.id === next) {
      return { ...stage, status: 'active' as const, detail: detail ?? stage.detail }
    }
    return stage
  })
}

function findRunningTool(tools: ToolSnapshot[], toolName: string): number {
  for (let index = tools.length - 1; index >= 0; index -= 1) {
    const tool = tools[index]
    if (tool && tool.tool_name === toolName && tool.status === 'running') return index
  }
  return -1
}

export function intentLabel(intent: AnalysisIntent | undefined): string | undefined {
  if (!intent) return undefined
  switch (intent) {
    case 'factoid':
      return 'Запрос факта'
    case 'structural':
      return 'Структурный запрос'
    case 'temporal':
      return 'Сравнение периодов'
    case 'multi_hop':
      return 'Многошаговый запрос'
    default:
      return String(intent)
  }
}

export function applyRunEvent(state: RunProjection, event: AgentEventEnvelope): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  if (event.request_id !== state.requestId) return state
  if (!Number.isFinite(event.sequence)) return state
  if (event.sequence <= state.lastSequence) return state

  const base: RunProjection = {
    ...state,
    lastSequence: event.sequence,
    phase: state.phase === 'reconnecting' ? 'streaming' : state.phase,
  }

  switch (event.type) {
    case 'run_started':
      return {
        ...base,
        assistantMessageId: event.message_id || base.assistantMessageId,
        stages: advanceTo(base.stages, 'analysis'),
      }

    case 'analysis_completed':
      return {
        ...base,
        intent: event.intent,
        effectiveFilters: event.effective_filters,
        stages: base.stages.map((stage) =>
          stage.id === 'analysis'
            ? { ...stage, status: 'done' as const, detail: intentLabel(event.intent) }
            : stage,
        ),
      }

    case 'tool_started':
      return {
        ...base,
        tools: [
          ...base.tools,
          {
            id: `${event.tool_name}-${event.sequence}`,
            tool_name: event.tool_name,
            status: 'running',
            short_arguments: event.short_arguments,
          },
        ],
        stages: advanceTo(base.stages, 'search'),
      }

    case 'tool_completed': {
      const index = findRunningTool(base.tools, event.tool_name)
      const tools = base.tools.map((tool, toolIndex) =>
        toolIndex === index
          ? {
              ...tool,
              status: event.status === 'success' ? ('success' as const) : ('error' as const),
              result_count: event.result_count,
              duration_ms: event.duration_ms,
            }
          : tool,
      )
      return { ...base, tools }
    }

    case 'evidence_updated':
      return { ...base, evidence: event.evidence }

    case 'answer_started':
      return { ...base, stages: advanceTo(base.stages, 'answer') }

    case 'answer_delta':
      return { ...base, streamedText: base.streamedText + event.delta }

    case 'run_completed': {
      const message = event.message
      const phase: RunPhase = event.answer_status === 'abstain' ? 'abstained' : 'completed'
      return {
        ...base,
        phase,
        durationMs: event.duration_ms,
        finalMessage: message,
        evidence: message?.evidence ?? base.evidence,
        stages: terminalStages(
          base.stages.map((stage) =>
            stage.id === 'finish' ? { ...stage, status: 'done' as const } : stage,
          ),
          'done',
        ),
      }
    }

    case 'run_failed':
      return {
        ...base,
        phase: 'failed',
        error: {
          code: event.code,
          message: event.message,
          request_id: event.request_id,
          retryable: event.retryable,
        },
        stages: terminalStages(base.stages, 'failed'),
      }

    case 'run_cancelled':
      return {
        ...base,
        phase: 'cancelled',
        stages: terminalStages(base.stages, 'skipped'),
      }

    default:
      return base
  }
}

export function markReconnecting(state: RunProjection): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  return { ...state, phase: 'reconnecting' }
}

export function applyResync(state: RunProjection, reason: string): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  return { ...state, phase: 'reconnecting', resyncReason: reason }
}

export function applyInterrupted(state: RunProjection, message?: string): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  return {
    ...state,
    phase: 'failed',
    error: {
      code: 'interrupted',
      message:
        message ?? 'Выполнение прервано: события недоступны после перезапуска API.',
      request_id: state.requestId,
      retryable: true,
    },
    stages: terminalStages(state.stages, 'failed'),
  }
}

export function applyTransportError(state: RunProjection, error: RunErrorData): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  return { ...state, phase: 'failed', error, stages: terminalStages(state.stages, 'failed') }
}

export function applyRestoredMessage(state: RunProjection, message: ChatMessage): RunProjection {
  if (isTerminalRunPhase(state.phase)) return state
  const phase: RunPhase =
    message.status === 'completed'
      ? 'completed'
      : message.status === 'abstained'
        ? 'abstained'
        : message.status === 'cancelled'
          ? 'cancelled'
          : message.status === 'failed'
            ? 'failed'
            : 'streaming'
  return {
    ...state,
    phase,
    finalMessage: message,
    evidence: message.evidence ?? [],
    error: message.error,
    stages: terminalStages(state.stages, phase === 'failed' ? 'failed' : 'done'),
  }
}

function buildExecution(state: RunProjection): ExecutionData {
  return {
    request_id: state.requestId,
    phase: state.phase,
    stages: state.stages,
    tools: state.tools,
    intent: intentLabel(state.intent),
    effective_filters: state.effectiveFilters,
    requested_filters: state.requestedFilters,
    filter_notices: diffFilters(state.requestedFilters, state.effectiveFilters),
    sequence: state.lastSequence,
    duration_ms: state.durationMs,
    resync_reason: state.resyncReason,
  }
}

function analysisParts(analysis: AnalysisResult | undefined): ThreadAssistantMessagePart[] {
  if (!analysis) return []
  const parts: ThreadAssistantMessagePart[] = []

  const analysisPayload: AnalysisData = {
    facts: analysis.facts,
    calculations: analysis.calculations,
    structural_changes: analysis.structural_changes,
  }
  if (
    (analysisPayload.facts?.length ?? 0) > 0 ||
    (analysisPayload.calculations?.length ?? 0) > 0 ||
    (analysisPayload.structural_changes?.length ?? 0) > 0
  ) {
    parts.push({ type: 'data', name: DATA_NAMES.analysis, data: analysisPayload })
  }
  if (analysis.coverage) {
    parts.push({ type: 'data', name: DATA_NAMES.coverage, data: analysis.coverage })
  }
  if (analysis.verification) {
    parts.push({ type: 'data', name: DATA_NAMES.verification, data: analysis.verification })
  }
  return parts
}

function answerState(state: RunProjection): AnswerState | null {
  if (state.phase === 'abstained') return 'abstained'
  if (state.phase === 'cancelled') return 'cancelled'
  if (state.phase === 'failed') return state.error?.code === 'interrupted' ? 'interrupted' : 'failed'
  return null
}

/** Cumulative parts for one yield of the adapter. */
export function buildMessageParts(state: RunProjection): ThreadAssistantMessagePart[] {
  const final = state.finalMessage
  const parts: ThreadAssistantMessagePart[] = [
    { type: 'data', name: DATA_NAMES.execution, data: buildExecution(state) },
  ]

  const text = final?.content ?? state.streamedText
  if (text) parts.push({ type: 'text', text })

  const evidence = final?.evidence ?? state.evidence
  if (evidence.length > 0) {
    parts.push({ type: 'data', name: DATA_NAMES.evidence, data: { items: evidence } })
  }

  const analysis = final?.analysis
  if (analysis) parts.push(...analysisParts(analysis))
  const verification = (final as { verification?: unknown } | undefined)?.verification
  if (verification) {
    parts.push({ type: 'data', name: DATA_NAMES.verification, data: verification })
  }

  const warnings = final?.warnings ?? []
  if (warnings.length > 0) {
    parts.push({ type: 'data', name: DATA_NAMES.warnings, data: { codes: warnings } })
  }

  const stateName = answerState(state)
  if (stateName) {
    const detail =
      stateName === 'failed'
        ? state.error?.message
        : stateName === 'interrupted'
          ? 'События запроса недоступны после перезапуска API.'
          : stateName === 'cancelled'
            ? 'Выполнение остановлено пользователем.'
            : undefined
    parts.push({
      type: 'data',
      name: DATA_NAMES.answerState,
      data: { state: stateName, detail },
    })
  }

  if (state.error) {
    parts.push({ type: 'data', name: DATA_NAMES.runError, data: state.error })
  }

  return parts
}

export function buildRunStatus(state: RunProjection): MessageStatus {
  switch (state.phase) {
    case 'completed':
    case 'abstained':
      return { type: 'complete', reason: 'stop' }
    case 'failed':
      return { type: 'incomplete', reason: 'error' }
    case 'cancelled':
      return { type: 'incomplete', reason: 'cancelled' }
    default:
      return { type: 'running' }
  }
}

export function isTerminalEventForRun(event: AgentEventEnvelope): boolean {
  return isTerminalEvent(event)
}
