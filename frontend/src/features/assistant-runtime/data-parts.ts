import type { ThreadMessage } from '@assistant-ui/react'
import type { CalculationSeries, CoverageReport, Fact, StructuralChange } from '../../types/analytics'
import type { FilterSet } from '../../types/api'
import type { EvidenceItem } from '../../types/evidence'

/**
 * Custom data parts of the assistant message.
 *
 * assistant-ui stores the streaming buffer; these parts only carry payloads
 * that the backend actually produced (evidence, calculations, errors, run
 * trace). No chain-of-thought or invented progress is encoded here.
 */
export const DATA_NAMES = {
  execution: 'execution',
  evidence: 'evidence',
  analysis: 'analysis',
  coverage: 'coverage',
  verification: 'verification',
  warnings: 'warnings',
  runError: 'run_error',
  answerState: 'answer_state',
} as const

export interface StageSnapshot {
  id: string
  label: string
  status: 'pending' | 'active' | 'done' | 'failed' | 'skipped'
  source: string
  detail?: string
}

export interface ToolSnapshot {
  id: string
  tool_name: string
  status: 'running' | 'success' | 'error'
  short_arguments?: Record<string, unknown>
  result_count?: number
  duration_ms?: number
}

export interface ExecutionData {
  request_id?: string
  phase: string
  stages: StageSnapshot[]
  tools: ToolSnapshot[]
  intent?: string
  effective_filters?: FilterSet
  requested_filters?: FilterSet
  filter_notices?: string[]
  sequence?: number
  duration_ms?: number
  resync_reason?: string
  cancel_error?: string
}

export interface EvidenceData {
  items: EvidenceItem[]
}

export interface AnalysisData {
  facts?: Fact[]
  calculations?: CalculationSeries[]
  structural_changes?: StructuralChange[]
}

export interface RunErrorData {
  code: string
  message: string
  request_id?: string
  retryable: boolean
}

export type AnswerState = 'abstained' | 'cancelled' | 'failed' | 'interrupted'

export interface AnswerStateData {
  state: AnswerState
  detail?: string
}

interface LoosePart {
  type: string
  text?: string
  name?: string
  data?: unknown
}

function looseContent(message: { content: readonly unknown[] }): readonly LoosePart[] {
  return message.content as unknown as readonly LoosePart[]
}

export function getDataPart<T>(message: { content: readonly unknown[] }, name: string): T | null {
  for (const part of looseContent(message)) {
    if (part.type === 'data' && part.name === name) return (part.data ?? null) as T
  }
  return null
}

export function textOfMessage(message: { content: readonly unknown[] }): string {
  return looseContent(message)
    .filter((part) => part.type === 'text')
    .map((part) => part.text ?? '')
    .join('\n')
    .trim()
}

export function evidenceOfMessage(message: { content: readonly unknown[] }): EvidenceItem[] {
  return getDataPart<EvidenceData>(message, DATA_NAMES.evidence)?.items ?? []
}

/** Latest assistant message that actually carries an evidence snapshot. */
export function latestEvidenceMessage(
  messages: readonly ThreadMessage[],
): ThreadMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (!message || message.role !== 'assistant') continue
    if (evidenceOfMessage(message).length > 0) return message
  }
  return null
}

export function coverageOfMessage(message: { content: readonly unknown[] }): CoverageReport | null {
  return getDataPart<CoverageReport>(message, DATA_NAMES.coverage)
}
