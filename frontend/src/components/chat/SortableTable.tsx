import { useMemo, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { ArrowDown, ArrowUp, ChevronsUpDown, Download } from 'lucide-react'
import { downloadCsv, toCsv, type CsvColumn, type SortDirection } from '../../lib/analytics'

/**
 * Interactive table for backend analytics: sortable columns, sticky header,
 * keyboard-resizable columns, horizontal scrolling and CSV export of exactly
 * the displayed rows.
 */

export interface TableColumn<T> {
  key: string
  label: string
  align?: 'left' | 'right'
  sortable?: boolean
  sortValue?: (row: T) => string | number | null
  render: (row: T) => ReactNode
  /** Value used by CSV export; omitted columns are not exported. */
  csvValue?: (row: T) => string | number | null | undefined
  minWidth?: number
}

function compareValues(left: string | number | null, right: string | number | null): number {
  if (left === null && right === null) return 0
  if (left === null) return 1
  if (right === null) return -1
  if (typeof left === 'number' && typeof right === 'number') return left - right
  return String(left).localeCompare(String(right), 'ru', { numeric: true })
}

export function SortableTable<T>({
  caption,
  columns,
  rows,
  initialSort,
  csvBaseName,
  emptyMessage = 'Нет строк для отображения.',
  ariaLabel,
  maxHeightClass = 'max-h-80',
}: {
  caption: string
  columns: readonly TableColumn<T>[]
  rows: readonly T[]
  initialSort?: { key: string; direction: SortDirection }
  csvBaseName?: string
  emptyMessage?: string
  ariaLabel?: string
  maxHeightClass?: string
}) {
  const [sort, setSort] = useState<{ key: string; direction: SortDirection } | null>(
    initialSort ?? null,
  )
  const [widths, setWidths] = useState<Record<string, number>>({})
  const [notice, setNotice] = useState<string | null>(null)

  const sortedRows = useMemo(() => {
    if (!sort) return [...rows]
    const column = columns.find((item) => item.key === sort.key)
    if (!column?.sortValue) return [...rows]
    const factor = sort.direction === 'asc' ? 1 : -1
    return [...rows].sort(
      (left, right) => compareValues(column.sortValue!(left), column.sortValue!(right)) * factor,
    )
  }, [rows, sort, columns])

  const toggleSort = (column: TableColumn<T>) => {
    if (!column.sortable) return
    setSort((current) => {
      if (!current || current.key !== column.key) {
        return { key: column.key, direction: 'asc' }
      }
      return { key: column.key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
    })
  }

  const resizeWithKeyboard = (event: KeyboardEvent<HTMLSpanElement>, columnKey: string) => {
    const step = event.key === 'ArrowRight' ? 16 : event.key === 'ArrowLeft' ? -16 : 0
    if (!step) return
    event.preventDefault()
    setWidths((current) => ({
      ...current,
      [columnKey]: Math.max(72, (current[columnKey] ?? 120) + step),
    }))
  }

  const startResize = (event: PointerEvent<HTMLSpanElement>, columnKey: string) => {
    event.preventDefault()
    const header = event.currentTarget.closest('th')
    const startWidth = widths[columnKey] ?? header?.offsetWidth ?? 120
    const startX = event.clientX
    const onMove = (moveEvent: globalThis.PointerEvent) => {
      setWidths((current) => ({
        ...current,
        [columnKey]: Math.max(72, startWidth + moveEvent.clientX - startX),
      }))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const csvColumns: CsvColumn<T>[] = columns
    .filter((column) => Boolean(column.csvValue))
    .map((column) => ({
      key: column.key,
      label: column.label,
      value: (row: T) => column.csvValue?.(row),
    }))

  const exportCsv = () => {
    if (!csvBaseName || csvColumns.length === 0) return
    const csv = toCsv(csvColumns, sortedRows)
    const ok = downloadCsv(`${csvBaseName}.csv`, csv)
    setNotice(
      ok
        ? `Экспортировано строк: ${sortedRows.length}.`
        : 'Браузер не позволяет скачать файл: экспорт CSV недоступен.',
    )
  }

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] text-muted">{caption}</p>
        {csvBaseName && csvColumns.length > 0 ? (
          <button
            type="button"
            className="btn btn-ghost px-2 py-0.5 text-[11px]"
            onClick={exportCsv}
            disabled={sortedRows.length === 0}
          >
            <Download className="size-3" aria-hidden="true" />
            Экспорт CSV
          </button>
        ) : null}
      </div>

      {notice ? (
        <p role="status" className="mb-1 text-[10px] text-muted">
          {notice}
        </p>
      ) : null}

      <div
        role="region"
        tabIndex={0}
        aria-label={ariaLabel ?? caption}
        className={`${maxHeightClass} max-w-full overflow-auto rounded-lg border border-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60`}
      >
        <table className="w-full min-w-[28rem] border-collapse text-xs">
          <caption className="sr-only">{caption}</caption>
          {Object.keys(widths).length > 0 ? (
            <colgroup>
              {columns.map((column) => (
                <col
                  key={column.key}
                  style={
                    widths[column.key]
                      ? { width: `${widths[column.key]}px` }
                      : column.minWidth
                        ? { minWidth: `${column.minWidth}px` }
                        : undefined
                  }
                />
              ))}
            </colgroup>
          ) : null}
          <thead>
            <tr>
              {columns.map((column) => {
                const active = sort?.key === column.key
                const activeDirection = active ? sort?.direction : undefined
                const ariaSort = activeDirection
                  ? activeDirection === 'asc'
                    ? ('ascending' as const)
                    : ('descending' as const)
                  : ('none' as const)
                return (
                  <th
                    key={column.key}
                    scope="col"
                    aria-sort={column.sortable ? ariaSort : undefined}
                    className={`sticky top-0 z-10 border-b border-line bg-raised px-2 py-1.5 font-medium text-muted ${
                      column.align === 'right' ? 'text-right' : 'text-left'
                    }`}
                  >
                    <span className="flex items-center gap-1">
                      {column.sortable ? (
                        <button
                          type="button"
                          onClick={() => toggleSort(column)}
                          aria-label={`Сортировать: ${column.label}`}
                          className="inline-flex items-center gap-1 rounded text-left hover:text-ink"
                        >
                          {column.label}
                          {active ? (
                            activeDirection === 'asc' ? (
                              <ArrowUp className="size-3" aria-hidden="true" />
                            ) : (
                              <ArrowDown className="size-3" aria-hidden="true" />
                            )
                          ) : (
                            <ChevronsUpDown className="size-3 opacity-50" aria-hidden="true" />
                          )}
                        </button>
                      ) : (
                        <span>{column.label}</span>
                      )}
                      <span
                        role="separator"
                        aria-orientation="vertical"
                        aria-label={`Изменить ширину колонки ${column.label}`}
                        tabIndex={0}
                        onPointerDown={(event) => startResize(event, column.key)}
                        onKeyDown={(event) => resizeWithKeyboard(event, column.key)}
                        className="ml-auto h-4 w-1.5 cursor-col-resize rounded bg-line/60 hover:bg-accent/50"
                      />
                    </span>
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {sortedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-2 py-3 text-muted">
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              sortedRows.map((row, rowIndex) => (
                <tr key={rowIndex} className="odd:bg-surface even:bg-raised/30">
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className={`border-b border-line/60 px-2 py-1.5 align-top text-ink ${
                        column.align === 'right' ? 'text-right tabular-nums' : 'text-left'
                      }`}
                    >
                      {column.render(row)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
