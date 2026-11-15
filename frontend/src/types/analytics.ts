/**
 * Analytical contracts (spec §6 and §8).
 *
 * Stage 1–2 UI only stores and forwards these structures. It must not compute
 * percentages, CAGR, totals or coverage: all analytic values come from backend.
 */

export type Unit = string

export interface SeriesPoint {
  year: number
  value: number
  evidence_ids: string[]
}

/** Backend-produced calculation series; never rebuilt on the client. */
export interface CalculationSeries {
  metric: string
  metric_label?: string
  unit: string
  points?: SeriesPoint[]
  from_year: number
  to_year: number
  absolute_change: number
  percent_change?: number | null
  cagr?: number | null
  direction: 'increase' | 'decrease' | 'flat'
  warnings?: string[]
}

export interface Fact {
  metric: string
  metric_label?: string
  year: number
  /** Backend value; `null` is kept as a gap, never replaced with 0. */
  value: string | number | null
  unit?: string
  scale?: string
  chunk_id: string
  row_label?: string
  column_label?: string
}

export type CoverageStatus = 'covered' | 'not_found' | 'year_absent'

export interface CoverageReport {
  status: 'not_required' | 'complete' | 'partial' | 'insufficient'
  expected_years: number[]
  covered_years: number[]
  missing_years: number[]
  metrics: Record<string, Record<string, CoverageStatus>>
  /** Additive backend requirement (stage 3); absent in stage 1–2. */
  cell_evidence?: Record<string, Record<string, string[]>>
  missing_metrics: string[]
  notes: string[]
}

export interface SectionSummary {
  title?: string
  section_path?: string
  level?: number
  page_start?: number
  page_end?: number
}

export type StructuralChangeType =
  | 'stable'
  | 'added'
  | 'removed'
  | 'renamed'
  | 'restructured'
  | 'expanded'
  | 'reduced'

export interface StructuralChange {
  from_year: number
  to_year: number
  from_section?: SectionSummary
  to_section?: SectionSummary
  change_type: StructuralChangeType
  score?: number
  detail?: string
  document_id?: string
  evidence_chunk_ids?: string[]
}

/** `message.analysis` — populated only with what backend actually returned. */
export interface AnalysisResult {
  calculations?: CalculationSeries[]
  facts?: Fact[]
  coverage?: CoverageReport
  structural_changes?: StructuralChange[]
  /** Backend verification payload; rendered as a compact data part. */
  verification?: unknown
}
