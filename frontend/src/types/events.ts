import type { ChatMessage, FilterSet } from './api'
import type { EvidenceItem } from './evidence'

/**
 * Public SSE contract, spec §4.2.
 *
 * `sequence` is the envelope id (also sent as the SSE `id:` line). The
 * transport layer drops duplicates and foreign request ids, the reducer does
 * the same so replays cannot corrupt the UI.
 */

export type AnalysisIntent = 'factoid' | 'structural' | 'temporal' | 'multi_hop'

export type ToolStatus = 'success' | 'error' | string

export interface RunStartedEvent {
  type: 'run_started'
  request_id: string
  message_id: string
}

export interface AnalysisCompletedEvent {
  type: 'analysis_completed'
  request_id: string
  intent: AnalysisIntent
  effective_filters: FilterSet
}

export interface ToolStartedEvent {
  type: 'tool_started'
  request_id: string
  tool_name: string
  short_arguments?: Record<string, unknown>
}

export interface ToolCompletedEvent {
  type: 'tool_completed'
  request_id: string
  tool_name: string
  status: ToolStatus
  result_count?: number
  duration_ms?: number
}

export interface EvidenceUpdatedEvent {
  type: 'evidence_updated'
  request_id: string
  evidence: EvidenceItem[]
}

export interface AnswerStartedEvent {
  type: 'answer_started'
  request_id: string
}

export interface AnswerDeltaEvent {
  type: 'answer_delta'
  request_id: string
  delta: string
}

export interface RunCompletedEvent {
  type: 'run_completed'
  request_id: string
  answer_status: 'answer' | 'abstain'
  duration_ms: number
  message: ChatMessage
}

export interface RunFailedEvent {
  type: 'run_failed'
  request_id: string
  code: string
  message: string
  retryable: boolean
}

export interface RunCancelledEvent {
  type: 'run_cancelled'
  request_id: string
}

export type AgentEvent =
  | RunStartedEvent
  | AnalysisCompletedEvent
  | ToolStartedEvent
  | ToolCompletedEvent
  | EvidenceUpdatedEvent
  | AnswerStartedEvent
  | AnswerDeltaEvent
  | RunCompletedEvent
  | RunFailedEvent
  | RunCancelledEvent

export type AgentEventType = AgentEvent['type']

export type TerminalEvent = RunCompletedEvent | RunFailedEvent | RunCancelledEvent

/** SSE payload: the event plus the replay envelope. */
export type AgentEventEnvelope = AgentEvent & {
  sequence: number
  contract_version?: string
}

export const TERMINAL_EVENT_TYPES: ReadonlySet<string> = new Set([
  'run_completed',
  'run_failed',
  'run_cancelled',
])

export const EVENT_TYPES: ReadonlySet<string> = new Set([
  'run_started',
  'analysis_completed',
  'tool_started',
  'tool_completed',
  'evidence_updated',
  'answer_started',
  'answer_delta',
  'run_completed',
  'run_failed',
  'run_cancelled',
])

export function isTerminalEvent(event: { type: string }): boolean {
  return TERMINAL_EVENT_TYPES.has(event.type)
}

/**
 * Lenient runtime guard: version skew must not crash the chat. Unknown event
 * types are accepted structurally and ignored by the reducer.
 */
export function isAgentEventEnvelope(value: unknown): value is AgentEventEnvelope {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.type === 'string' &&
    typeof candidate.request_id === 'string' &&
    typeof candidate.sequence === 'number' &&
    Number.isFinite(candidate.sequence)
  )
}
