import type {
  CalculationSeries,
  Fact,
  SeriesPoint,
  StructuralChange,
  StructuralChangeType,
} from '../types/analytics'
import { EMPTY_VALUE, formatNumber } from './format'

/**
 * Pure display shaping for backend analytics (`facts`, `calculations`,
 * `structural_changes`).
 *
 * Rules (spec §6):
 * - values are never recomputed, aggregated, or derived; only formatted;
 * - rows are grouped by `(metric, unit, scale)` so different units are never mixed;
 * - a missing value stays missing (`null`), it is never coerced to 0;
 * - CSV is a serialization of the rows currently displayed.
 */

export const NO_UNIT_LABEL = 'единица не указана'

export type SortDirection = 'asc' | 'desc'

export interface FactRow {
  id: string
  fact: Fact
  metric: string
  metricLabel: string
  year: number
  value: Fact['value']
  unitLabel: string
  chunkId: string
  rowLabel?: string
  columnLabel?: string
}

export interface FactGroup {
  /** Stable key of `(metric, unit, scale)`. */
  key: string
  metric: string
  metricLabel: string
  unit?: string
  scale?: string
  unitLabel: string
  /** `false` when backend omitted the unit: no comparative block may be built. */
  unitSpecified: boolean
  rows: FactRow[]
}

function clean(value: string | undefined): string | undefined {
  if (value === undefined) return undefined
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

export function factUnitLabel(fact: Pick<Fact, 'unit' | 'scale'>): string {
  const unit = clean(fact.unit)
  const scale = clean(fact.scale)
  if (unit && scale) return `${scale} ${unit}`
  if (unit) return unit
  if (scale) return scale
  return NO_UNIT_LABEL
}

/** Unit of a calculation series; empty means "backend did not specify". */
export function calculationUnitLabel(series: Pick<CalculationSeries, 'unit'>): string {
  return clean(series.unit) ?? NO_UNIT_LABEL
}

function factGroupKey(fact: Fact): string {
  return [fact.metric, clean(fact.unit) ?? '', clean(fact.scale) ?? ''].join('\u0000')
}

/**
 * Group facts by `(metric, unit, scale)` preserving backend order inside a group.
 * Groups themselves are sorted by metric label for stable rendering.
 */
export function groupFacts(facts: readonly Fact[]): FactGroup[] {
  const groups = new Map<string, FactGroup>()

  for (const fact of facts) {
    const key = factGroupKey(fact)
    const unitLabel = factUnitLabel(fact)
    const unitSpecified = unitLabel !== NO_UNIT_LABEL
    const metricLabel = clean(fact.metric_label) ?? fact.metric
    let group = groups.get(key)
    if (!group) {
      group = {
        key,
        metric: fact.metric,
        metricLabel,
        unit: clean(fact.unit),
        scale: clean(fact.scale),
        unitLabel,
        unitSpecified,
        rows: [],
      }
      groups.set(key, group)
    }
    group.rows.push({
      id: `${key}:${fact.year}:${fact.chunk_id}:${group.rows.length}`,
      fact,
      metric: fact.metric,
      metricLabel: clean(fact.metric_label) ?? group.metricLabel,
      year: fact.year,
      value: fact.value,
      unitLabel,
      chunkId: fact.chunk_id,
      rowLabel: clean(fact.row_label),
      columnLabel: clean(fact.column_label),
    })
  }

  return [...groups.values()]
    .map((group) => ({
      ...group,
      rows: [...group.rows].sort((a, b) => a.year - b.year || a.metricLabel.localeCompare(b.metricLabel, 'ru')),
    }))
    .sort((a, b) => a.metricLabel.localeCompare(b.metricLabel, 'ru') || a.unitLabel.localeCompare(b.unitLabel, 'ru'))
}

/** Numeric view of a backend value for display sorting only. */
export function numericValue(value: Fact['value']): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value !== 'string') return null
  const normalized = value.replace(/\s|\u00a0/g, '').replace(',', '.')
  if (!/^[-+]?\d+(?:\.\d+)?$/.test(normalized)) return null
  const parsed = Number(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

export function formatFactValue(value: Fact['value']): string {
  if (value === null || value === undefined || value === '') return EMPTY_VALUE
  return formatNumber(value)
}

export function sortFactRows(
  rows: readonly FactRow[],
  key: 'metric' | 'year' | 'value' | 'unit',
  direction: SortDirection,
): FactRow[] {
  const factor = direction === 'asc' ? 1 : -1
  return [...rows].sort((left, right) => {
    if (key === 'year') return (left.year - right.year) * factor
    if (key === 'unit') return left.unitLabel.localeCompare(right.unitLabel, 'ru') * factor
    if (key === 'metric') return left.metricLabel.localeCompare(right.metricLabel, 'ru') * factor

    const leftNumber = numericValue(left.value)
    const rightNumber = numericValue(right.value)
    if (leftNumber !== null && rightNumber !== null) return (leftNumber - rightNumber) * factor
    // Non-numeric values are always last, regardless of direction.
    if (leftNumber !== null) return -1
    if (rightNumber !== null) return 1
    return String(left.value ?? '').localeCompare(String(right.value ?? ''), 'ru') * factor
  })
}

/* -------------------------------------------------------------------------- */
/* Calculations                                                               */
/* -------------------------------------------------------------------------- */

export interface CalculationGroup {
  key: string
  unit?: string
  unitLabel: string
  unitSpecified: boolean
  series: CalculationSeries[]
}

/** Calculations grouped by unit: series with different units never share a chart. */
export function groupCalculations(calculations: readonly CalculationSeries[]): CalculationGroup[] {
  const groups = new Map<string, CalculationGroup>()
  for (const series of calculations) {
    const unitLabel = calculationUnitLabel(series)
    const key = clean(series.unit) ?? '\u0000no-unit'
    let group = groups.get(key)
    if (!group) {
      group = {
        key,
        unit: clean(series.unit),
        unitLabel,
        unitSpecified: unitLabel !== NO_UNIT_LABEL,
        series: [],
      }
      groups.set(key, group)
    }
    group.series.push(series)
  }
  return [...groups.values()]
}

export function seriesLabel(series: Pick<CalculationSeries, 'metric' | 'metric_label'>): string {
  return clean(series.metric_label) ?? series.metric
}

export const DIRECTION_LABEL: Record<CalculationSeries['direction'], string> = {
  increase: 'Рост',
  decrease: 'Снижение',
  flat: 'Без изменений',
}

export function directionLabel(direction: string | undefined): string {
  if (direction === 'increase' || direction === 'decrease' || direction === 'flat') {
    return DIRECTION_LABEL[direction]
  }
  return direction ? String(direction) : EMPTY_VALUE
}

export function hasPoints(series: CalculationSeries): boolean {
  return Array.isArray(series.points) && series.points.length > 0
}

export function seriesWithoutPoints(calculations: readonly CalculationSeries[]): CalculationSeries[] {
  return calculations.filter((series) => !hasPoints(series))
}

/* -------------------------------------------------------------------------- */
/* Charts                                                                     */
/* -------------------------------------------------------------------------- */

export interface ChartSeries {
  key: string
  label: string
  points: SeriesPoint[]
}

/** One row per year; missing years are `null` (a gap, never 0). */
export type ChartRow = { year: number } & Record<string, number | string | null>

export interface ChartData {
  rows: ChartRow[]
  hasGap: boolean
}

function pointYears(points: readonly SeriesPoint[]): number[] {
  return points
    .map((point) => point.year)
    .filter((year) => Number.isFinite(year))
}

/**
 * Build a shared year axis for series of the same unit. Only points returned by
 * backend are placed on the axis; years between them are explicit `null` gaps.
 * `from_year`/`to_year` extend the axis only when they are part of the backend
 * series range.
 */
export function buildChartRows(series: readonly ChartSeries[]): ChartData {
  const years = new Set<number>()
  let from: number | undefined
  let to: number | undefined
  for (const item of series) {
    for (const year of pointYears(item.points)) years.add(year)
  }
  for (const item of series) {
    for (const year of pointYears(item.points)) {
      from = from === undefined ? year : Math.min(from, year)
      to = to === undefined ? year : Math.max(to, year)
    }
  }
  if (from !== undefined && to !== undefined) {
    for (let year = from; year <= to; year += 1) years.add(year)
  }

  const sortedYears = [...years].sort((a, b) => a - b)
  let hasGap = false
  const rows: ChartRow[] = sortedYears.map((year) => {
    const row: ChartRow = { year }
    for (const item of series) {
      const point = item.points.find((candidate) => candidate.year === year)
      if (point && Number.isFinite(point.value)) {
        row[item.key] = point.value
      } else {
        row[item.key] = null
        hasGap = true
      }
    }
    return row
  })

  return { rows, hasGap }
}

export interface ChartPointEvidence {
  key: string
  year: number
  evidenceIds: string[]
}

/** Flat lookup of evidence ids per `(series, year)` for chart interactions. */
export function chartPointEvidence(series: readonly ChartSeries[]): ChartPointEvidence[] {
  const entries: ChartPointEvidence[] = []
  for (const item of series) {
    for (const point of item.points) {
      entries.push({
        key: item.key,
        year: point.year,
        evidenceIds: Array.isArray(point.evidence_ids) ? point.evidence_ids.filter(Boolean) : [],
      })
    }
  }
  return entries
}

/* -------------------------------------------------------------------------- */
/* Structural comparison                                                      */
/* -------------------------------------------------------------------------- */

export const STRUCTURAL_CHANGE_ORDER: StructuralChangeType[] = [
  'added',
  'removed',
  'renamed',
  'restructured',
  'expanded',
  'reduced',
  'stable',
]

const STRUCTURAL_TYPES = new Set<string>(STRUCTURAL_CHANGE_ORDER)

export interface StructuralFilterResult {
  supported: StructuralChange[]
  /** Types the current analyzer does not produce; never rendered as categories. */
  unsupported: string[]
}

/**
 * Split backend changes into types the UI supports. `split`/`merge` and any
 * unknown value are not displayed as categories, because the backend does not
 * support them yet.
 */
export function filterStructuralChanges(changes: readonly StructuralChange[]): StructuralFilterResult {
  const supported: StructuralChange[] = []
  const unsupported = new Set<string>()
  for (const change of changes) {
    const type = String(change.change_type)
    if (STRUCTURAL_TYPES.has(type)) supported.push(change)
    else unsupported.add(type)
  }
  return { supported, unsupported: [...unsupported].sort() }
}

export function structuralTitle(change: StructuralChange): string {
  return (
    clean(change.to_section?.title) ??
    clean(change.from_section?.title) ??
    'раздел без названия'
  )
}

export function structuralPath(change: StructuralChange): string | undefined {
  const path = clean(change.to_section?.section_path) ?? clean(change.from_section?.section_path)
  return path
}

export function sortStructuralChanges(changes: readonly StructuralChange[]): StructuralChange[] {
  const priority = new Map(STRUCTURAL_CHANGE_ORDER.map((type, index) => [type, index]))
  return [...changes].sort((left, right) => {
    if (left.from_year !== right.from_year) return left.from_year - right.from_year
    if (left.to_year !== right.to_year) return left.to_year - right.to_year
    const leftPriority = priority.get(left.change_type) ?? STRUCTURAL_CHANGE_ORDER.length
    const rightPriority = priority.get(right.change_type) ?? STRUCTURAL_CHANGE_ORDER.length
    if (leftPriority !== rightPriority) return leftPriority - rightPriority
    return structuralTitle(left).localeCompare(structuralTitle(right), 'ru')
  })
}

/* -------------------------------------------------------------------------- */
/* CSV                                                                        */
/* -------------------------------------------------------------------------- */

export interface CsvColumn<T> {
  key: string
  label: string
  value: (row: T) => string | number | null | undefined
}

function csvCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return ''
  const text = String(value)
  if (/[",\r\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`
  return text
}

/** Serialize exactly the rows and columns that are on screen. */
export function toCsv<T>(columns: readonly CsvColumn<T>[], rows: readonly T[]): string {
  const header = columns.map((column) => csvCell(column.label)).join(',')
  const body = rows.map((row) => columns.map((column) => csvCell(column.value(row))).join(','))
  return [header, ...body].join('\r\n')
}

export function csvFileName(base: string): string {
  const slug = base
    .toLowerCase()
    .replace(/[^a-zа-я0-9]+/gi, '-')
    .replace(/^-+|-+$/g, '')
  return `${slug || 'analytics'}.csv`
}

/**
 * Browser download of already serialized CSV. Returns `false` when the runtime
 * cannot create object URLs (e.g. a restricted environment), so the caller can
 * say so honestly instead of silently doing nothing.
 */
export function downloadCsv(filename: string, csv: string): boolean {
  if (typeof document === 'undefined') return false
  if (typeof Blob === 'undefined' || typeof URL === 'undefined') return false
  if (typeof URL.createObjectURL !== 'function') return false

  const blob = new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.rel = 'noopener'
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  link.remove()
  if (typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(url)
  return true
}
