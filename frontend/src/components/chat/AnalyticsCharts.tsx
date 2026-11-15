import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { CalculationSeries } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import {
  buildChartRows,
  chartPointEvidence,
  formatFactValue,
  seriesLabel,
  type ChartRow,
  type ChartSeries,
} from '../../lib/analytics'

/**
 * Recharts views for backend calculation series.
 *
 * Only `points` from backend are plotted: no interpolation and no derived
 * values. A year without a backend point is rendered as a gap (`null`), never
 * as zero. Series are grouped by unit upstream; different units never share a
 * chart. CAGR/percent changes are not drawn because they are computed by backend
 * and already shown in the table.
 */

const SERIES_COLORS = ['#0f6464', '#b45309', '#4b5563', '#0e7490', '#7c3aed', '#9d174d']

interface TooltipEntry {
  dataKey?: string | number
  name?: string | number
  value?: number | string | null
  color?: string
}

function ChartTooltip({
  active,
  payload,
  label,
  unitLabel,
}: {
  active?: boolean
  payload?: TooltipEntry[]
  label?: string | number
  unitLabel: string
}) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="rounded-md border border-line bg-surface px-2 py-1 text-[11px] shadow-sm">
      <p className="font-medium text-ink">Год: {label}</p>
      <ul className="mt-0.5 space-y-0.5">
        {payload.map((entry) => (
          <li key={String(entry.dataKey)} className="text-muted">
            <span
              aria-hidden="true"
              className="mr-1 inline-block size-2 rounded-sm align-middle"
              style={{ backgroundColor: entry.color }}
            />
            {entry.name}:{' '}
            <span className="tabular-nums text-ink">
              {typeof entry.value === 'number' ? formatFactValue(entry.value) : 'нет данных'}
            </span>{' '}
            {entry.value === null || entry.value === undefined ? '' : unitLabel}
          </li>
        ))}
      </ul>
    </div>
  )
}

export interface ChartGroupProps {
  series: CalculationSeries[]
  unitLabel: string
  evidenceById: Map<string, EvidenceItem>
  onOpenEvidence: (evidenceId: string) => void
}

export function CalculationCharts({
  series,
  unitLabel,
  evidenceById,
  onOpenEvidence,
}: ChartGroupProps) {
  const [mode, setMode] = useState<'line' | 'bar'>(series.length > 1 ? 'bar' : 'line')

  const chartSeries: ChartSeries[] = useMemo(
    () =>
      series.map((item, index) => ({
        key: `series-${index}`,
        label: seriesLabel(item),
        points: item.points ?? [],
      })),
    [series],
  )
  const { rows, hasGap } = useMemo(() => buildChartRows(chartSeries), [chartSeries])
  const points = useMemo(() => chartPointEvidence(chartSeries), [chartSeries])

  if (rows.length === 0 || chartSeries.every((item) => item.points.length === 0)) return null

  const labelByKey = new Map(chartSeries.map((item) => [item.key, item.label]))
  const evidenceFor = (key: string, year: number): string[] =>
    points.find((entry) => entry.key === key && entry.year === year)?.evidenceIds ?? []

  const openFromChart = (state: unknown) => {
    const activePayload = (
      state as {
        activePayload?: Array<{ dataKey?: string | number; payload?: ChartRow }>
      } | null
    )?.activePayload
    const entry = activePayload?.[0]
    if (!entry?.dataKey || !entry.payload) return
    const year = Number(entry.payload.year)
    const ids = evidenceFor(String(entry.dataKey), year)
    const first = ids.find((id) => evidenceById.has(id))
    if (first) onOpenEvidence(first)
  }

  const summary = `График: ${chartSeries.map((item) => item.label).join(', ')}. Единица: ${unitLabel}. Годы: ${rows
    .map((row) => row.year)
    .join(', ')}.`

  return (
    <figure className="mt-1 rounded-lg border border-line bg-raised/30 px-2 py-2">
      <figcaption className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] text-muted">
          {chartSeries.map((item) => item.label).join(', ')} · {unitLabel}
        </span>
        <span role="group" aria-label="Тип графика" className="flex items-center gap-1">
          <button
            type="button"
            className="btn btn-ghost px-2 py-0.5 text-[10px]"
            aria-pressed={mode === 'line'}
            onClick={() => setMode('line')}
          >
            Линия
          </button>
          <button
            type="button"
            className="btn btn-ghost px-2 py-0.5 text-[10px]"
            aria-pressed={mode === 'bar'}
            onClick={() => setMode('bar')}
          >
            Столбцы
          </button>
        </span>
      </figcaption>

      <div role="img" aria-label={summary} className="mt-1 h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {mode === 'line' ? (
            <LineChart
              data={rows}
              margin={{ top: 8, right: 12, bottom: 24, left: 8 }}
              onClick={openFromChart}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#d9d6cd" />
              <XAxis
                dataKey="year"
                tick={{ fontSize: 10 }}
                label={{ value: 'Год', position: 'insideBottom', offset: -12, fontSize: 10 }}
              />
              <YAxis
                tick={{ fontSize: 10 }}
                width={64}
                label={{ value: unitLabel, angle: -90, position: 'insideLeft', fontSize: 10 }}
              />
              <Tooltip content={(props) => <ChartTooltip {...(props as unknown as { active?: boolean; payload?: TooltipEntry[]; label?: string | number })} unitLabel={unitLabel} />} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {chartSeries.map((item, index) => (
                <Line
                  key={item.key}
                  type="monotone"
                  dataKey={item.key}
                  name={item.label}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          ) : (
            <BarChart
              data={rows}
              margin={{ top: 8, right: 12, bottom: 24, left: 8 }}
              onClick={openFromChart}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#d9d6cd" />
              <XAxis
                dataKey="year"
                tick={{ fontSize: 10 }}
                label={{ value: 'Год', position: 'insideBottom', offset: -12, fontSize: 10 }}
              />
              <YAxis
                tick={{ fontSize: 10 }}
                width={64}
                label={{ value: unitLabel, angle: -90, position: 'insideLeft', fontSize: 10 }}
              />
              <Tooltip content={(props) => <ChartTooltip {...(props as unknown as { active?: boolean; payload?: TooltipEntry[]; label?: string | number })} unitLabel={unitLabel} />} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {chartSeries.map((item, index) => (
                <Bar
                  key={item.key}
                  dataKey={item.key}
                  name={item.label}
                  fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {hasGap ? (
        <p className="mt-1 text-[10px] text-warn">
          Пропуски в данных backend показаны как разрыв, а не как нулевые значения.
        </p>
      ) : null}

      <details className="mt-1 text-[11px] text-muted">
        <summary className="cursor-pointer select-none">Точные значения и источники</summary>
        <div className="mt-1 max-h-56 overflow-auto rounded border border-line">
          <table className="w-full border-collapse text-[11px]">
            <caption className="sr-only">Точные значения графика и источники</caption>
            <thead>
              <tr>
                <th scope="col" className="border-b border-line bg-raised px-2 py-1 text-left font-medium">
                  Год
                </th>
                {chartSeries.map((item) => (
                  <th
                    key={item.key}
                    scope="col"
                    className="border-b border-line bg-raised px-2 py-1 text-right font-medium"
                  >
                    {item.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.year}>
                  <th scope="row" className="border-b border-line/60 px-2 py-1 text-left font-normal text-muted">
                    {row.year}
                  </th>
                  {chartSeries.map((item) => {
                    const value = row[item.key]
                    const ids = evidenceFor(item.key, row.year).filter((id) => evidenceById.has(id))
                    const numeric = typeof value === 'number' ? value : null
                    return (
                      <td
                        key={item.key}
                        className="border-b border-line/60 px-2 py-1 text-right tabular-nums text-ink"
                      >
                        {numeric === null ? (
                          <span className="text-muted" title="Backend не передал точку за этот год">
                            нет данных
                          </span>
                        ) : ids.length > 0 ? (
                          <button
                            type="button"
                            className="underline decoration-dotted decoration-accent/50 underline-offset-2 hover:text-accent"
                            onClick={() => onOpenEvidence(ids[0] as string)}
                            aria-label={`Открыть источник точки ${labelByKey.get(item.key)} за ${row.year}`}
                          >
                            {formatFactValue(numeric)}
                          </button>
                        ) : (
                          formatFactValue(numeric)
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  )
}
