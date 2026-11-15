import type { EvidenceDetail as EvidenceDetailData } from '../../types/evidence'
import { formatPages, NOT_SPECIFIED, orNotSpecified } from '../../lib/format'
import { SOURCE_TYPE_LABEL } from '../../lib/labels'
import {
  pdfFileHref,
  pdfUnavailableReason,
  sourcePage,
  usePdfLinksState,
} from '../../lib/pdf-links'
import { sanitizeTableHtml, splitByQuote } from '../../lib/table-html'
import { useEvidenceDetail } from '../../features/evidence/use-evidence'
import { ErrorState } from '../common/ErrorState'
import { Spinner } from '../common/Spinner'

function MetaLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <dt className="w-20 shrink-0">{label}</dt>
      <dd className="min-w-0 break-words text-ink">{value}</dd>
    </div>
  )
}

/** Full chunk text is fetched on demand; it is never persisted in message history. */
export function EvidenceDetail({ chunkId, quote }: { chunkId: string; quote: string }) {
  const query = useEvidenceDetail(chunkId, true)

  if (query.isLoading) return <Spinner label="Загрузка источника" />
  if (query.error) {
    return (
      <ErrorState
        error={query.error}
        title="Источник недоступен"
        onRetry={() => void query.refetch()}
      />
    )
  }
  if (!query.data) return null

  return <EvidenceBody detail={query.data} quote={quote} />
}

function EvidenceBody({ detail, quote }: { detail: EvidenceDetailData; quote: string }) {
  const pages = formatPages({
    page_start: detail.page_start,
    page_end: detail.page_end,
    page_label: detail.page_label,
  })
  const table = detail.source_type === 'table' && detail.html ? sanitizeTableHtml(detail.html) : null
  const pdfState = usePdfLinksState()
  const pdfHref = pdfFileHref(detail, pdfState)
  const pdfPage = sourcePage(detail)

  return (
    <div className="mt-2 rounded-lg border border-line bg-surface p-2.5">
      <dl className="space-y-0.5 text-[11px] text-muted">
        <MetaLine label="Документ" value={orNotSpecified(detail.document_title)} />
        <MetaLine label="Год" value={orNotSpecified(detail.year)} />
        <MetaLine label="Issuer" value={orNotSpecified(detail.issuer)} />
        <MetaLine
          label="Раздел"
          value={
            detail.section_path && detail.section_path.length > 0
              ? detail.section_path.join(' / ')
              : NOT_SPECIFIED
          }
        />
        <MetaLine label="Страницы" value={pages} />
        <MetaLine label="Тип" value={detail.source_type ? SOURCE_TYPE_LABEL[detail.source_type] : NOT_SPECIFIED} />
        <MetaLine label="chunk_id" value={detail.chunk_id} />
      </dl>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        {pdfHref ? (
          <a
            href={pdfHref}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost px-2 py-0.5 text-[11px]"
            aria-label={`Открыть PDF документа${pdfPage ? `, стр. ${pdfPage}` : ''}`}
          >
            Открыть PDF{pdfPage ? `, стр. ${pdfPage}` : ''}
          </a>
        ) : (
          <button
            type="button"
            disabled
            className="btn btn-ghost px-2 py-0.5 text-[11px]"
            title={pdfUnavailableReason(detail, pdfState)}
          >
            Открыть PDF
          </button>
        )}
      </div>

      {table ? (
        <div className="mt-2 max-h-80 overflow-auto rounded border border-line p-1 text-[11px] [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-line [&_td]:px-1.5 [&_td]:py-1 [&_th]:border [&_th]:border-line [&_th]:bg-raised [&_th]:px-1.5 [&_th]:py-1">
          {table}
        </div>
      ) : null}

      <div className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded border border-line bg-raised/50 p-2 text-xs text-ink">
        {splitByQuote(detail.text ?? '', quote).map((part, index) =>
          part.match ? (
            <mark key={index} className="rounded bg-amber-200/70 px-0.5 text-ink dark:bg-amber-500/30">
              {part.text}
            </mark>
          ) : (
            <span key={index}>{part.text}</span>
          ),
        )}
      </div>

      <p className="mt-1.5 text-[10px] text-muted">
        Полный текст chunk загружается по /api/evidence/{'{chunk_id}'} и не сохраняется в истории
        сообщений.
      </p>
    </div>
  )
}
