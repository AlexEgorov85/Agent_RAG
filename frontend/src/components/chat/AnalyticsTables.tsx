import { useMemo } from 'react'
import { Link2 } from 'lucide-react'
import type { CalculationSeries } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import {
  DIRECTION_LABEL,
  NO_UNIT_LABEL,
  formatFactValue,
  directionLabel,
  seriesLabel,
  sortFactRows,
  type CalculationGroup,
  type FactGroup,
} from '../../lib/analytics'
import { EMPTY_VALUE, formatNumber } from '../../lib/format'
import { SortableTable, type TableColumn } from './SortableTable'

const NOT_CALCULATED = 'не рассчитано backend'

/**
 * Tables built strictly from backend `facts` / `calculations`.
 * A fact row links to its evidence only when the chunk is part of the message
 * evidence snapshot; otherwise the value is shown without a fake link.
 */

function FactValue({
  row,
  evidenceById,
  onOpenEvidence,
}: {
  row: FactGroup['rows'][number]
  evidenceById: Map<string, EvidenceItem>
  onOpenEvidence: (evidenceId: string) => void
}) {
  const formatted = formatFactValue(row.value)
  const item = row.chunkId ? evidenceById.get(row.chunkId) : undefined
  if (item) {
    return (
      <button
        type="button"
        className="inline-flex items-center gap-1 rounded text-left underline decoration-dotted decoration-accent/50 underline-offset-2 hover:text-accent"
        onClick={() => onOpenEvidence(item.evidence_id)}
        aria-label={`Открыть источник значения ${row.metricLabel} за ${row.year}`}
        title={`chunk_id: ${row.chunkId}`}
      >
        {formatted}
        <Link2 className="size-3 text-muted" aria-hidden="true" />
      </button>
    )
  }
  return (
    <span
      title={
        row.chunkId
          ? `chunk_id ${row.chunkId} отсутствует в снимке evidence этого сообщения`
          : 'chunk_id не передан backend'
      }
    >
      {formatted}
    </span>
  )
}

export function FactsTable({
  group,
  evidenceById,
  onOpenEvidence,
}: {
  group: FactGroup
  evidenceById: Map<string, EvidenceItem>
  onOpenEvidence: (evidenceId: string) => void
}) {
  const rows = useMemo(() => sortFactRows(group.rows, 'year', 'asc'), [group.rows])

  const columns: TableColumn<FactGroup['rows'][number]>[] = [
    {
      key: 'metric',
      label: 'Показатель',
      sortable: true,
      sortValue: (row) => row.metricLabel,
      csvValue: (row) => row.metricLabel,
      render: (row) => (
        <span className="block min-w-0">
          <span className="block break-words text-ink">{row.metricLabel}</span>
          {row.metricLabel !== row.metric ? (
            <span className="block font-mono text-[10px] text-muted/70">{row.metric}</span>
          ) : null}
          {row.rowLabel || row.columnLabel ? (
            <span className="block text-[10px] text-muted">
              {[row.rowLabel, row.columnLabel].filter(Boolean).join(' · ')}
            </span>
          ) : null}
        </span>
      ),
    },
    {
      key: 'year',
      label: 'Год',
      sortable: true,
      align: 'right',
      sortValue: (row) => row.year,
      csvValue: (row) => row.year,
      render: (row) => String(row.year),
    },
    {
      key: 'value',
      label: 'Значение',
      sortable: true,
      align: 'right',
      sortValue: (row) => row.value,
      csvValue: (row) => formatFactValue(row.value),
      render: (row) => (
        <FactValue
          row={row}
          evidenceById={evidenceById}
          onOpenEvidence={onOpenEvidence}
        />
      ),
    },
    {
      key: 'unit',
      label: 'Единица',
      csvValue: (row) => row.unitLabel,
      render: (row) => (
        <span className={row.unitLabel === NO_UNIT_LABEL ? 'text-warn' : 'text-muted'}>
          {row.unitLabel}
        </span>
      ),
    },
  ]

  return (
    <div className="space-y-1">
      <SortableTable
        caption={`Факты backend: ${group.metricLabel}, ${group.unitLabel}`}
        ariaLabel={`Таблица фактов: ${group.metricLabel}, ${group.unitLabel}`}
        columns={columns}
        rows={rows}
        initialSort={{ key: 'year', direction: 'asc' }}
        csvBaseName={`facts-${group.metric}-${group.unit ?? 'no-unit'}`}
        emptyMessage="Backend не передал факты для этого показателя."
      />
      {!group.unitSpecified ? (
        <p role="note" className="text-[10px] text-warn">
          Единица не указана: значения показаны как есть, сравнительный блок для них не строится,
          чтобы не смешать разные единицы.
        </p>
      ) : null}
    </div>
  )
}

export function CalculationTable({ group }: { group: CalculationGroup }) {
  const columns: TableColumn<CalculationSeries>[] = [
    {
      key: 'metric',
      label: 'Показатель',
      sortable: true,
      sortValue: (series) => seriesLabel(series),
      csvValue: (series) => seriesLabel(series),
      render: (series) => (
        <span className="block min-w-0">
          <span className="block break-words text-ink">{seriesLabel(series)}</span>
          {series.metric_label ? (
            <span className="block font-mono text-[10px] text-muted/70">{series.metric}</span>
          ) : null}
        </span>
      ),
    },
    {
      key: 'period',
      label: 'Период',
      align: 'right',
      csvValue: (series) => `${series.from_year}–${series.to_year}`,
      render: (series) => (
        <span className="whitespace-nowrap tabular-nums">
          {series.from_year} → {series.to_year}
        </span>
      ),
    },
    {
      key: 'absolute',
      label: 'Абс. изменение',
      align: 'right',
      sortable: true,
      sortValue: (series) =>
        typeof series.absolute_change === 'number' ? series.absolute_change : null,
      csvValue: (series) =>
        typeof series.absolute_change === 'number'
          ? formatNumber(series.absolute_change)
          : EMPTY_VALUE,
      render: (series) =>
        typeof series.absolute_change === 'number' ? formatNumber(series.absolute_change) : EMPTY_VALUE,
    },
    {
      key: 'percent',
      label: 'Изменение, %',
      align: 'right',
      csvValue: (series) =>
        typeof series.percent_change === 'number'
          ? formatNumber(series.percent_change)
          : NOT_CALCULATED,
      render: (series) =>
        typeof series.percent_change === 'number' ? (
          formatNumber(series.percent_change)
        ) : (
          <span className="text-muted">{NOT_CALCULATED}</span>
        ),
    },
    {
      key: 'cagr',
      label: 'CAGR',
      align: 'right',
      csvValue: (series) =>
        typeof series.cagr === 'number' ? formatNumber(series.cagr) : NOT_CALCULATED,
      render: (series) =>
        typeof series.cagr === 'number' ? (
          formatNumber(series.cagr)
        ) : (
          <span className="text-muted">{NOT_CALCULATED}</span>
        ),
    },
    {
      key: 'direction',
      label: 'Направление',
      csvValue: (series) => directionLabel(series.direction),
      render: (series) => (
        <span className="text-muted">
          {DIRECTION_LABEL[series.direction] ?? directionLabel(series.direction)}
        </span>
      ),
    },
  ]

  return (
    <div className="space-y-1">
      <SortableTable
        caption={`Расчёты backend: ${group.unitLabel}. Абсолютное изменение, процент и CAGR рассчитаны backend, не интерфейсом.`}
        ariaLabel={`Таблица расчётов: ${group.unitLabel}`}
        columns={columns}
        rows={group.series}
        initialSort={{ key: 'metric', direction: 'asc' }}
        csvBaseName={`calculations-${group.unit ?? 'no-unit'}`}
        emptyMessage="Backend не передал расчёты для этой единицы."
      />
      {!group.unitSpecified ? (
        <p role="note" className="text-[10px] text-warn">
          Единица не указана: сравнение с другими расчётами не строится.
        </p>
      ) : null}
      {group.series.some((series) => (series.warnings?.length ?? 0) > 0) ? (
        <ul role="status" className="space-y-0.5 text-[10px] text-warn">
          {group.series.flatMap((series) =>
            (series.warnings ?? []).map((warning) => (
              <li key={`${series.metric}:${warning}`}>
                {seriesLabel(series)}: {warning}
              </li>
            )),
          )}
        </ul>
      ) : null}
    </div>
  )
}
