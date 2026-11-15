import type { ChatMessage } from '../../../src/types/api'
import type { EvidenceItem } from '../../../src/types/evidence'
import type { AgentEventEnvelope } from '../../../src/types/events'

export const fullEvidence: EvidenceItem = {
  evidence_id: 'chunk-1',
  chunk_id: 'chunk-1',
  document_id: 'sber_2023',
  document_title: 'Годовой отчёт 2023',
  issuer: 'ПАО Сбербанк',
  year: 2023,
  section_title: 'Ключевые показатели',
  section_path: ['Стратегия', 'Ключевые показатели'],
  page_start: 12,
  page_label: '12',
  excerpt: 'Чистая прибыль составила 1 508,6 млрд рублей.',
  source_type: 'text',
  has_full_text: true,
}

export const missingEvidence: EvidenceItem = {
  evidence_id: 'chunk-2',
  chunk_id: 'chunk-2',
  excerpt: '',
  has_full_text: false,
}

export const tableEvidence: EvidenceItem = {
  evidence_id: 'chunk-3',
  chunk_id: 'chunk-3',
  document_id: 'sber_2023',
  issuer: 'ПАО Сбербанк',
  section_path: ['Результаты', 'Таблица'],
  excerpt: '| Показатель | 2023 |',
  source_type: 'table',
  has_full_text: false,
}

export function assistantMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'msg-assistant',
    research_id: 'research-1',
    role: 'assistant',
    content: 'Ответ агента',
    created_at: '2026-11-14T10:00:00Z',
    status: 'completed',
    request_id: 'req-1',
    evidence: [fullEvidence],
    warnings: [],
    ...overrides,
  }
}

export function userMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'msg-user',
    research_id: 'research-1',
    role: 'user',
    content: 'Какая чистая прибыль за 2023 год?',
    created_at: '2026-11-14T09:59:00Z',
    status: 'completed',
    evidence: [],
    warnings: [],
    ...overrides,
  }
}

export const runStarted: AgentEventEnvelope = {
  type: 'run_started',
  request_id: 'req-1',
  message_id: 'msg-assistant',
  sequence: 1,
}

export const analysisCompleted: AgentEventEnvelope = {
  type: 'analysis_completed',
  request_id: 'req-1',
  intent: 'factoid',
  effective_filters: { years: [2023] },
  sequence: 2,
}

export const toolStarted: AgentEventEnvelope = {
  type: 'tool_started',
  request_id: 'req-1',
  tool_name: 'search_reports',
  short_arguments: { years: [2023] },
  sequence: 3,
}

export const toolCompleted: AgentEventEnvelope = {
  type: 'tool_completed',
  request_id: 'req-1',
  tool_name: 'search_reports',
  status: 'success',
  result_count: 7,
  duration_ms: 812,
  sequence: 4,
}

export const answerStarted: AgentEventEnvelope = {
  type: 'answer_started',
  request_id: 'req-1',
  sequence: 5,
}

export function runCompleted(overrides: Partial<ChatMessage> = {}): AgentEventEnvelope {
  return {
    type: 'run_completed',
    request_id: 'req-1',
    answer_status: 'answer',
    duration_ms: 1500,
    message: assistantMessage(overrides),
    sequence: 6,
  }
}
