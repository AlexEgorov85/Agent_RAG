/**
 * Evidence contract, spec web-ui-modernization §5.3 and §5.5.
 * The UI never derives missing fields: absent data is rendered as "не указано".
 */

export type SourceType = 'text' | 'table'

/** Snapshot stored inside a chat message (metadata + quote only). */
export interface EvidenceItem {
  /** Chunk identifier; also the navigation key of the panel. */
  evidence_id: string
  chunk_id: string
  document_id?: string | null
  document_title?: string
  issuer?: string
  year?: number
  section_title?: string
  section_path?: string[]
  page_start?: number
  page_end?: number
  page_label?: string
  /** `Citation.quote`; may be an empty string. */
  excerpt: string
  source_type?: SourceType
  /** Whether `/api/evidence/{chunk_id}` is available for this chunk. */
  has_full_text: boolean
}

/** Response of `GET /api/evidence/{chunk_id}`. */
export interface EvidenceDetail {
  chunk_id: string
  document_id?: string | null
  document_title?: string
  issuer?: string
  year?: number
  section_path?: string[]
  page_start?: number
  page_end?: number
  page_label?: string
  source_type?: SourceType
  text: string
  html?: string | null
}
