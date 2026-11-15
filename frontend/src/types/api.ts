import type { AnalysisResult } from './analytics'
import type { EvidenceItem } from './evidence'

export type ResearchStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'abstained'
  | 'failed'
  | 'cancelled'

export interface Research {
  id: string
  title: string
  created_at: string
  updated_at: string
  last_message_preview?: string
  last_status: ResearchStatus
  messages_count: number
}

export interface ResearchListResponse {
  items: Research[]
  total: number
  next_offset: number | null
}

export type MessageStatus =
  | 'pending'
  | 'streaming'
  | 'completed'
  | 'abstained'
  | 'failed'
  | 'cancelled'

export interface MessageError {
  code: string
  message: string
  request_id?: string
  retryable: boolean
}

export interface FilterSet {
  years?: number[]
  year_from?: number
  year_to?: number
  issuers?: string[]
  allowed_doc_ids?: string[]
  kinds?: Array<'text' | 'table'>
}

export interface ChatMessage {
  id: string
  research_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  status: MessageStatus
  request_id?: string
  effective_filters?: FilterSet
  evidence: EvidenceItem[]
  analysis?: AnalysisResult
  warnings: string[]
  error?: MessageError
  /** Optional verification payload published by backend; rendered as a data part. */
  verification?: unknown
}

export interface SendMessageRequest {
  content: string
  client_message_id: string
  filters?: FilterSet
}

export interface SendMessageResponse {
  request_id: string
  user_message: ChatMessage
  assistant_message_id: string
  events_url: string
  research: { id: string; updated_at: string }
}

export type RequestStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'abstained'
  | 'failed'
  | 'cancelled'

export interface RequestStatusResponse {
  request_id: string
  status: RequestStatus
  message_id?: string
  message?: ChatMessage | null
  error?: MessageError | null
}

export interface Capabilities {
  contract_version: string
  token_streaming: boolean
  incremental_evidence: boolean
  coverage: boolean
  calculations: boolean
  structure_comparison: boolean
  structure_content_comparison: boolean
  graph_panel: boolean
  pdf_page_links: boolean
  evaluation_ui: boolean
  evaluation_sse?: boolean
  evaluation_cancel?: boolean
  retrieval_modes?: string[]
  judge?: boolean
}

export interface CorpusDocument {
  doc_id: string
  year?: number
  issuer?: string
  title?: string
}

export interface CorpusInfo {
  name?: string
  years: number[]
  documents: CorpusDocument[]
  issuers: string[]
  years_available: boolean
}

export interface HealthResponse {
  status?: string
  corpus_ready?: boolean
  [key: string]: unknown
}

/** Unified error envelope, spec §7.8. */
export interface ApiErrorBody {
  error: {
    code: string
    message: string
    request_id?: string
    retryable: boolean
    details?: unknown
  }
}
