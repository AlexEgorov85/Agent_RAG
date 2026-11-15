import { createContext, useContext, type ReactNode } from 'react'
import { ExternalLink, FileText, Table2 } from 'lucide-react'
import type { EvidenceItem } from '../../types/evidence'
import { formatPages } from '../../lib/format'
import {
  pdfFileHref,
  pdfUnavailableReason,
  sourcePage,
  type PdfLinksState,
} from '../../lib/pdf-links'

/**
 * Citations under an assistant answer.
 *
 * Every citation keeps the assistant-ui data-part boundary: it is built from the
 * exact `evidence` snapshot received from backend. A PDF link is rendered only
 * when backend declares `pdf_page_links=true` and the document id is known; the
 * page fragment comes from the backend page field, never from the quote.
 */

export interface CitationNav {
  items: readonly EvidenceItem[]
  openEvidence: (evidenceId: string) => void
}

const CitationNavContext = createContext<CitationNav | null>(null)

export function CitationNavProvider({
  value,
  children,
}: {
  value: CitationNav
  children: ReactNode
}) {
  return <CitationNavContext.Provider value={value}>{children}</CitationNavContext.Provider>
}

export function useCitationNav(): CitationNav | null {
  return useContext(CitationNavContext)
}

export function citationLabel(item: EvidenceItem): string {
  return item.document_title?.trim() || item.document_id || 'документ не указан'
}

export function citationPageLabel(item: EvidenceItem): string {
  return formatPages(item)
}

function CitationPdfLink({
  item,
  index,
  state,
}: {
  item: EvidenceItem
  index: number
  state: PdfLinksState
}) {
  const href = pdfFileHref(item, state)
  if (!href) {
    return (
      <span className="text-[10px] text-muted" title={pdfUnavailableReason(item, state)}>
        PDF недоступен
      </span>
    )
  }

  const page = sourcePage(item)
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-0.5 text-[11px] text-ink transition-colors hover:border-accent/40 hover:bg-accent-soft/40"
      aria-label={`Открыть PDF источника ${index + 1}: ${citationLabel(item)}${
        page ? `, стр. ${page}` : ''
      }`}
      title={page ? `PDF, страница ${page}` : 'PDF документа'}
    >
      <ExternalLink className="size-3 text-muted" aria-hidden="true" />
      PDF{page ? `, стр. ${page}` : ''}
    </a>
  )
}

export function AssistantCitations({
  items,
  pdfState,
  onOpenEvidence,
  selectedEvidenceId,
  heading = 'Источники',
}: {
  items: readonly EvidenceItem[]
  pdfState: PdfLinksState
  onOpenEvidence: (evidenceId: string) => void
  selectedEvidenceId?: string | null
  heading?: string
}) {
  if (items.length === 0) return null

  return (
    <section className="mt-3" aria-label="Источники ответа">
      <p className="panel-title">
        {heading} ({items.length})
      </p>
      <ol className="mt-1 space-y-1.5">
        {items.map((item, index) => {
          const selected = selectedEvidenceId === item.evidence_id
          const sectionPath = (item.section_path ?? []).filter((part) => part.trim() !== '')
          return (
            <li
              key={item.evidence_id}
              id={`citation-${index + 1}`}
              data-citation-id={item.evidence_id}
              className={`rounded-lg border px-2.5 py-2 ${
                selected ? 'border-accent/40 bg-accent-soft/30' : 'border-line bg-raised/40'
              }`}
            >
              <div className="flex flex-wrap items-start gap-x-2 gap-y-1.5">
                <span className="mt-0.5 font-mono text-[11px] text-muted">[{index + 1}]</span>

                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-1 text-xs font-medium text-ink">
                    {item.source_type === 'table' ? (
                      <Table2 className="size-3 shrink-0 text-muted" aria-hidden="true" />
                    ) : (
                      <FileText className="size-3 shrink-0 text-muted" aria-hidden="true" />
                    )}
                    <span className="min-w-0 break-words">{citationLabel(item)}</span>
                  </p>
                  <p className="mt-0.5 text-[11px] text-muted">
                    {item.year !== undefined ? item.year : 'год не указан'} ·{' '}
                    {item.issuer?.trim() || 'issuer не указан'}
                  </p>
                  <p className="text-[11px] text-muted">
                    {item.document_id ? (
                      <span className="font-mono text-[10px]">{item.document_id}</span>
                    ) : (
                      'document_id не указан'
                    )}
                    {' · '}
                    {citationPageLabel(item)}
                    {sectionPath.length > 0 ? ` · ${sectionPath.join(' / ')}` : ' · раздел не указан'}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    className="rounded-md border border-line bg-surface px-2 py-0.5 text-[11px] text-ink transition-colors hover:border-accent/40 hover:bg-accent-soft/40"
                    onClick={() => onOpenEvidence(item.evidence_id)}
                    aria-label={`Открыть источник ${index + 1}: ${citationLabel(item)}`}
                  >
                    Цитата
                  </button>
                  <CitationPdfLink item={item} index={index} state={pdfState} />
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

/** Inline `[n]` marker from answer text; opens the same citation snapshot. */
export function CitationMarker({ n }: { n: number }) {
  const nav = useCitationNav()
  const item = nav?.items[n - 1]
  if (!nav || !item) {
    return <span className="font-mono text-[11px] text-muted">[{n}]</span>
  }

  return (
    <button
      type="button"
      onClick={() => {
        nav.openEvidence(item.evidence_id)
        if (typeof document !== 'undefined') {
          document.getElementById(`citation-${n}`)?.scrollIntoView?.({ block: 'nearest' })
        }
      }}
      className="mx-0.5 inline-flex items-center rounded px-0.5 align-super font-mono text-[11px] text-accent transition-colors hover:bg-accent-soft/60"
      aria-label={`Перейти к источнику ${n}: ${citationLabel(item)}`}
      title={citationLabel(item)}
    >
      [{n}]
    </button>
  )
}

/* -------------------------------------------------------------------------- */
/* Rehype plugin: numeric citation markers in answer text                     */
/* -------------------------------------------------------------------------- */

interface HastNode {
  type: string
  tagName?: string
  value?: string
  properties?: Record<string, unknown>
  children?: HastNode[]
}

const SKIPPED_TAGS = new Set(['a', 'code', 'pre', 'script', 'style', 'textarea'])

function citationNodes(n: number): HastNode[] {
  return [
    {
      type: 'element',
      tagName: 'a',
      properties: { href: `#citation-${n}`, 'data-citation': String(n) },
      children: [{ type: 'text', value: `[${n}]` }],
    },
  ]
}

function splitTextNode(node: HastNode): HastNode[] | null {
  const value = node.value ?? ''
  if (!/\[\d{1,3}\]/.test(value)) return null

  const parts: HastNode[] = []
  let lastIndex = 0
  const pattern = /\[(\d{1,3})\]/g
  for (let match = pattern.exec(value); match; match = pattern.exec(value)) {
    const index = match.index
    if (index > lastIndex) {
      parts.push({ type: 'text', value: value.slice(lastIndex, index) })
    }
    parts.push(...citationNodes(Number(match[1])))
    lastIndex = index + match[0].length
  }
  if (lastIndex < value.length) parts.push({ type: 'text', value: value.slice(lastIndex) })
  return parts
}

function walkCitationNode(node: HastNode): void {
  if (!node.children || node.children.length === 0) return
  if (node.tagName && SKIPPED_TAGS.has(node.tagName)) return

  const next: HastNode[] = []
  for (const child of node.children) {
    if (child.type === 'text') {
      const replacement = splitTextNode(child)
      if (replacement) next.push(...replacement)
      else next.push(child)
      continue
    }
    walkCitationNode(child)
    next.push(child)
  }
  node.children = next
}

/**
 * Converts `[1]`, `[2]` text markers into anchors after sanitization, so the
 * generated href cannot be influenced by model output. Skipped inside links and
 * code blocks; out-of-range markers are rendered as plain text by `CitationMarker`.
 */
export function rehypeCitationMarkers() {
  return (tree: unknown) => {
    walkCitationNode(tree as HastNode)
  }
}
