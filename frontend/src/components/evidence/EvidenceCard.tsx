import type { EvidenceItem } from '../../types/evidence'
import { formatPages, NOT_SPECIFIED, orNotSpecified } from '../../lib/format'
import { SOURCE_TYPE_LABEL } from '../../lib/labels'
import {
  pdfFileHref,
  pdfUnavailableReason,
  sourcePage,
  usePdfLinksState,
} from '../../lib/pdf-links'
import { ToneBadge } from '../common/ToneBadge'
import { EvidenceDetail } from './EvidenceDetail'

function MetaRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex gap-2 text-[11px]">
      <dt className="w-24 shrink-0 text-muted">{label}</dt>
      <dd className={`min-w-0 break-words text-ink ${mono ? 'font-mono text-[10px]' : ''}`}>
        {value}
      </dd>
    </div>
  )
}

export function EvidenceCard({
  evidence,
  index,
  expanded,
  onToggle,
}: {
  evidence: EvidenceItem
  index: number
  expanded: boolean
  onToggle: () => void
}) {
  const sectionPath = evidence.section_path ?? []
  const sectionTitle =
    evidence.section_title ?? (sectionPath.length > 0 ? sectionPath[sectionPath.length - 1] : undefined)
  const pages = formatPages(evidence)
  const pdfState = usePdfLinksState()
  const pdfHref = pdfFileHref(evidence, pdfState)
  const pdfPage = sourcePage(evidence)

  return (
    <li
      data-evidence-id={evidence.evidence_id}
      className={`rounded-lg border p-2.5 ${
        expanded ? 'border-accent/40 bg-accent-soft/30' : 'border-line bg-surface'
      }`}
      data-testid="evidence-card"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-ink">
            <span className="mr-1 font-mono text-[10px] text-muted">[{index + 1}]</span>
            {orNotSpecified(evidence.document_title)}
          </p>
          <p className="mt-0.5 text-[11px] text-muted">
            {evidence.year !== undefined ? `${evidence.year} · ` : 'год не указан · '}
            {orNotSpecified(evidence.issuer)}
          </p>
        </div>
        <ToneBadge tone={evidence.source_type === 'table' ? 'warn' : 'neutral'}>
          {evidence.source_type ? SOURCE_TYPE_LABEL[evidence.source_type] : 'тип не указан'}
        </ToneBadge>
      </div>

      <dl className="mt-2 space-y-1">
        <MetaRow label="Раздел" value={orNotSpecified(sectionTitle)} />
        <MetaRow
          label="Путь раздела"
          value={sectionPath.length > 0 ? sectionPath.join(' / ') : NOT_SPECIFIED}
        />
        <MetaRow label="Страницы" value={pages} />
        <MetaRow label="Метка страницы" value={orNotSpecified(evidence.page_label)} />
        <MetaRow label="document_id" value={orNotSpecified(evidence.document_id)} mono />
        <MetaRow label="chunk_id" value={evidence.chunk_id} mono />
      </dl>

      {evidence.excerpt ? (
        <blockquote className="mt-2 border-l-2 border-line pl-2 text-xs italic text-muted">
          «{evidence.excerpt}»
        </blockquote>
      ) : (
        <p className="mt-2 text-xs text-muted/80">Цитата не предоставлена в снимке evidence.</p>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn btn-ghost px-2 py-0.5 text-[11px]"
          onClick={onToggle}
          disabled={!evidence.has_full_text && !expanded}
          aria-expanded={expanded}
        >
          {expanded ? 'Свернуть' : 'Открыть источник'}
        </button>

        {pdfHref ? (
          <a
            href={pdfHref}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost px-2 py-0.5 text-[11px]"
            aria-label={`Открыть PDF: ${orNotSpecified(evidence.document_title)}${
              pdfPage ? `, стр. ${pdfPage}` : ''
            }`}
            title={pdfPage ? `PDF, страница ${pdfPage}` : 'PDF документа'}
          >
            Открыть PDF{pdfPage ? `, стр. ${pdfPage}` : ''}
          </a>
        ) : (
          <button
            type="button"
            disabled
            className="btn btn-ghost px-2 py-0.5 text-[11px]"
            title={pdfUnavailableReason(evidence, pdfState)}
            aria-label={`Открыть PDF: недоступно — ${pdfUnavailableReason(evidence, pdfState)}`}
          >
            Открыть PDF
          </button>
        )}

        {!evidence.has_full_text ? (
          <span className="text-[10px] text-muted">
            полный текст недоступен (has_full_text = false)
          </span>
        ) : null}
      </div>

      {expanded ? <EvidenceDetail chunkId={evidence.chunk_id} quote={evidence.excerpt} /> : null}
    </li>
  )
}
