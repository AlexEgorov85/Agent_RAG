import { useEffect, useSyncExternalStore } from 'react'
import { apiRequest } from '../services/api-client'
import type { Capabilities } from '../types/api'
import type { EvidenceItem } from '../types/evidence'

/**
 * PDF links are optional (spec §5.2). The UI builds them only when backend
 * declares `pdf_page_links=true`; otherwise it says "PDF недоступен" and keeps
 * the evidence panel as the only navigation path. URLs are never guessed.
 */

export type PdfLinksState = 'unknown' | 'enabled' | 'disabled'

export interface PdfSource {
  document_id?: string | null
  page_start?: number
  page_end?: number
  page_label?: string
  /** Optional alias used by backend snapshots; preferred over `page_start`. */
  page_pdf?: number
}

export function documentFileUrl(documentId: string): string {
  return `/api/documents/${encodeURIComponent(documentId)}/file`
}

function positiveInt(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  const rounded = Math.trunc(value)
  return rounded > 0 ? rounded : undefined
}

/** Page actually returned by backend (`page_pdf` when present, else `page_start`). */
export function sourcePage(source: PdfSource): number | undefined {
  return positiveInt(source.page_pdf) ?? positiveInt(source.page_start)
}

/**
 * `null` means "do not render a link": capability is off, or document id is
 * missing. No `#page` fragment is added when the page is unknown.
 */
export function pdfFileHref(source: PdfSource, state: PdfLinksState): string | null {
  if (state !== 'enabled') return null
  const documentId = typeof source.document_id === 'string' ? source.document_id.trim() : ''
  if (!documentId) return null
  const page = sourcePage(source)
  const base = documentFileUrl(documentId)
  return page ? `${base}#page=${page}` : base
}

/** Human explanation for a disabled link; never invents a URL. */
export function pdfUnavailableReason(source: PdfSource, state: PdfLinksState): string {
  if (state === 'unknown') return 'проверка доступности PDF…'
  if (state === 'disabled') {
    return 'PDF недоступен: backend не сообщает pdf_page_links'
  }
  if (!source.document_id) return 'PDF недоступен: document_id не передан'
  return 'PDF недоступен'
}

/* -------------------------------------------------------------------------- */
/* Capabilities store                                                         */
/* -------------------------------------------------------------------------- */

let state: PdfLinksState = 'unknown'
let request: Promise<void> | null = null
const listeners = new Set<() => void>()

function publish(next: PdfLinksState): void {
  if (state === next) return
  state = next
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/**
 * One shared request per page load. A failure is treated as "unavailable":
 * we would rather hide the link than invent a working URL.
 */
function ensureCapabilities(): void {
  if (request) return
  request = apiRequest<Capabilities>('/api/capabilities')
    .then((capabilities) => publish(capabilities.pdf_page_links ? 'enabled' : 'disabled'))
    .catch(() => publish('disabled'))
}

export function usePdfLinksState(): PdfLinksState {
  useEffect(() => {
    ensureCapabilities()
  }, [])
  return useSyncExternalStore(
    subscribe,
    () => state,
    () => 'unknown' as PdfLinksState,
  )
}

export function usePdfPageLinks(): boolean {
  return usePdfLinksState() === 'enabled'
}

/** Test-only reset of the module cache. */
export function resetPdfLinksCache(): void {
  state = 'unknown'
  request = null
  listeners.clear()
}

/** Shared citation navigation: inline `[n]` markers resolve against it. */
export function isPdfEvidence(item: EvidenceItem): boolean {
  return Boolean(item.document_id)
}
