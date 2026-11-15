import type { MessageStatus, ThreadAssistantMessagePart, ThreadMessageLike } from '@assistant-ui/react'
import type { ChatMessage } from '../../types/api'
import {
  DATA_NAMES,
  type AnswerState,
  type ExecutionData,
  type RunErrorData,
} from './data-parts'

function safeDate(iso: string | undefined): Date {
  if (!iso) return new Date()
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? new Date() : date
}

function assistantParts(message: ChatMessage, interrupted: boolean): ThreadAssistantMessagePart[] {
  const execution: ExecutionData = {
    request_id: message.request_id,
    phase: interrupted ? 'interrupted' : message.status,
    stages: [],
    tools: [],
    effective_filters: message.effective_filters,
  }
  const parts: ThreadAssistantMessagePart[] = [
    { type: 'data', name: DATA_NAMES.execution, data: execution },
  ]

  const answerState: AnswerState | null =
    message.status === 'abstained'
      ? 'abstained'
      : message.status === 'cancelled'
        ? 'cancelled'
        : message.status === 'failed'
          ? 'failed'
          : interrupted
            ? 'interrupted'
            : null
  if (answerState) {
    parts.push({
      type: 'data',
      name: DATA_NAMES.answerState,
      data: {
        state: answerState,
        detail: interrupted
          ? 'История загружена после перезагрузки; поток этого ответа недоступен.'
          : message.error?.message,
      },
    })
  }

  if (message.content) parts.push({ type: 'text', text: message.content })

  const evidence = message.evidence ?? []
  if (evidence.length > 0) {
    parts.push({ type: 'data', name: DATA_NAMES.evidence, data: { items: evidence } })
  }

  if (message.analysis) {
    const analysis = message.analysis
    if (
      (analysis.facts?.length ?? 0) > 0 ||
      (analysis.calculations?.length ?? 0) > 0 ||
      (analysis.structural_changes?.length ?? 0) > 0
    ) {
      parts.push({
        type: 'data',
        name: DATA_NAMES.analysis,
        data: {
          facts: analysis.facts,
          calculations: analysis.calculations,
          structural_changes: analysis.structural_changes,
        },
      })
    }
    if (analysis.coverage) {
      parts.push({ type: 'data', name: DATA_NAMES.coverage, data: analysis.coverage })
    }
    if (analysis.verification) {
      parts.push({ type: 'data', name: DATA_NAMES.verification, data: analysis.verification })
    }
  }

  if (message.verification) {
    parts.push({ type: 'data', name: DATA_NAMES.verification, data: message.verification })
  }

  if (message.warnings && message.warnings.length > 0) {
    parts.push({ type: 'data', name: DATA_NAMES.warnings, data: { codes: message.warnings } })
  }

  if (message.error) {
    const error: RunErrorData = {
      code: message.error.code,
      message: message.error.message,
      request_id: message.error.request_id,
      retryable: message.error.retryable,
    }
    parts.push({ type: 'data', name: DATA_NAMES.runError, data: error })
  }

  return parts
}

function messageStatus(message: ChatMessage, interrupted: boolean): MessageStatus {
  if (interrupted) return { type: 'incomplete', reason: 'other' }
  switch (message.status) {
    case 'completed':
    case 'abstained':
      return { type: 'complete', reason: 'stop' }
    case 'failed':
      return { type: 'incomplete', reason: 'error' }
    case 'cancelled':
      return { type: 'incomplete', reason: 'cancelled' }
    default:
      return { type: 'incomplete', reason: 'other' }
  }
}

/**
 * History comes from the backend; the runtime gets it as initial messages.
 * `streaming` rows cannot be resumed from this UI yet, so they are marked as
 * interrupted instead of showing an eternal spinner.
 */
export function chatMessageToThreadMessageLike(message: ChatMessage): ThreadMessageLike {
  const interrupted = message.role === 'assistant' && (message.status === 'streaming' || message.status === 'pending')

  if (message.role === 'user') {
    return {
      id: message.id,
      role: 'user',
      createdAt: safeDate(message.created_at),
      content: [{ type: 'text', text: message.content }],
    }
  }

  return {
    id: message.id,
    role: 'assistant',
    createdAt: safeDate(message.created_at),
    content: assistantParts(message, interrupted),
    status: messageStatus(message, interrupted),
  }
}
