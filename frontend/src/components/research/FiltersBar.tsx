import type { CorpusInfo, FilterSet } from '../../types/api'
import { activeFilterCount } from '../../lib/filters'
import { SlidersHorizontal } from 'lucide-react'

/** Filters are sent to backend with the next message; never applied client-side. */
export function FiltersBar({
  corpus,
  filters,
  onChange,
  disabled,
}: {
  corpus?: CorpusInfo
  filters: FilterSet
  onChange: (filters: FilterSet) => void
  disabled?: boolean
}) {
  const count = activeFilterCount(filters)
  const years = corpus?.years ?? []
  const issuers = corpus?.issuers ?? []
  const documents = corpus?.documents ?? []
  const yearsAvailable = corpus?.years_available !== false && years.length > 0

  const toggleYear = (year: number) => {
    const current = filters.years ?? []
    const next = current.includes(year)
      ? current.filter((item) => item !== year)
      : [...current, year].sort((a, b) => a - b)
    onChange({ ...filters, years: next })
  }

  const toggleIssuer = (issuer: string) => {
    const current = filters.issuers ?? []
    const next = current.includes(issuer)
      ? current.filter((item) => item !== issuer)
      : [...current, issuer]
    onChange({ ...filters, issuers: next })
  }

  const toggleDocument = (docId: string) => {
    const current = filters.allowed_doc_ids ?? []
    const next = current.includes(docId)
      ? current.filter((item) => item !== docId)
      : [...current, docId]
    onChange({ ...filters, allowed_doc_ids: next })
  }

  return (
    <details className="relative" data-testid="filters-bar">
      <summary
        className={`btn btn-ghost relative size-8 cursor-pointer select-none list-none p-0 ${disabled ? 'pointer-events-none opacity-50' : ''}`}
        aria-label="Фильтры запроса"
        title="Фильтры запроса"
      >
        <SlidersHorizontal className="size-4" aria-hidden="true" />
        {count > 0 ? (
          <span className="absolute -right-0.5 -top-0.5 grid size-4 place-items-center rounded-full bg-accent text-[9px] font-semibold text-accent-ink">
            {count}
          </span>
        ) : null}
      </summary>
      <div className="absolute right-0 z-40 mt-1 w-80 rounded-xl border border-line bg-surface p-3 shadow-panel">
        <div className="flex items-center justify-between">
          <p className="panel-title">Параметры запроса</p>
          <button
            type="button"
            className="btn btn-ghost px-1.5 py-0.5 text-[11px]"
            onClick={() => onChange({})}
            disabled={count === 0}
          >
            Сбросить
          </button>
        </div>

        <div className="mt-3">
          <p className="text-xs font-medium">Годы</p>
          {yearsAvailable ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {years.map((year) => {
                const checked = filters.years?.includes(year) ?? false
                return (
                  <label
                    key={year}
                    className={`cursor-pointer rounded-full border px-2 py-0.5 text-xs ${
                      checked ? 'border-accent/40 bg-accent-soft text-accent' : 'border-line bg-raised'
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="sr-only"
                      checked={checked}
                      onChange={() => toggleYear(year)}
                    />
                    {year}
                  </label>
                )
              })}
            </div>
          ) : (
            <p className="mt-1 text-xs text-muted">
              Backend не сообщает годы корпуса — фильтр по годам недоступен.
            </p>
          )}
        </div>

        <div className="mt-3">
          <p className="text-xs font-medium">Issuer</p>
          {issuers.length > 0 ? (
            <ul className="mt-1 space-y-1">
              {issuers.map((issuer) => (
                <li key={issuer}>
                  <label className="flex items-start gap-2 text-xs">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={filters.issuers?.includes(issuer) ?? false}
                      onChange={() => toggleIssuer(issuer)}
                    />
                    <span>{issuer}</span>
                  </label>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-xs text-muted">Backend не передал список issuer.</p>
          )}
        </div>

        <div className="mt-3">
          <p className="text-xs font-medium">Документы</p>
          {documents.length > 0 ? (
            <ul className="mt-1 max-h-40 space-y-1 overflow-y-auto pr-1">
              {documents.map((document) => (
                <li key={document.doc_id}>
                  <label className="flex items-start gap-2 text-xs">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={filters.allowed_doc_ids?.includes(document.doc_id) ?? false}
                      onChange={() => toggleDocument(document.doc_id)}
                    />
                    <span>
                      {document.title ?? document.doc_id}
                      {document.year !== undefined ? ` · ${document.year}` : ''}
                      <span className="ml-1 font-mono text-[10px] text-muted">{document.doc_id}</span>
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-xs text-muted">Backend не передал документы корпуса.</p>
          )}
        </div>

        <p className="mt-3 border-t border-line pt-2 text-[11px] text-muted">
          Фильтры применяются к следующему сообщению. Если backend сузит их, над диалогом появится
          предупреждение.
        </p>
      </div>
    </details>
  )
}
