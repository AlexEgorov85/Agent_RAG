import type { FilterSet } from '../types/api'

/** Filters are applied by the backend only; these helpers describe them. */

const KEYS: Array<keyof FilterSet> = ['years', 'year_from', 'year_to', 'issuers', 'allowed_doc_ids', 'kinds']

function uniqueSorted(values: Array<string | number> | undefined): string[] {
  if (!values || values.length === 0) return []
  return Array.from(new Set(values.map(String))).sort()
}

export function isEmptyFilters(filters?: FilterSet | null): boolean {
  if (!filters) return true
  return KEYS.every((key) => {
    const value = filters[key]
    if (value === undefined || value === null) return true
    if (Array.isArray(value)) return value.length === 0
    return false
  })
}

/** Drop empty values: the API must not receive empty filter arrays. */
export function compactFilters(filters: FilterSet): FilterSet {
  const compact: FilterSet = {}
  if (filters.years && filters.years.length > 0) compact.years = [...filters.years]
  if (filters.year_from !== undefined) compact.year_from = filters.year_from
  if (filters.year_to !== undefined) compact.year_to = filters.year_to
  if (filters.issuers && filters.issuers.length > 0) compact.issuers = [...filters.issuers]
  if (filters.allowed_doc_ids && filters.allowed_doc_ids.length > 0) {
    compact.allowed_doc_ids = [...filters.allowed_doc_ids]
  }
  if (filters.kinds && filters.kinds.length > 0) compact.kinds = [...filters.kinds]
  return compact
}

export function activeFilterCount(filters?: FilterSet | null): number {
  if (!filters) return 0
  let count = 0
  if (filters.years?.length) count += 1
  if (filters.year_from !== undefined || filters.year_to !== undefined) count += 1
  if (filters.issuers?.length) count += 1
  if (filters.allowed_doc_ids?.length) count += 1
  if (filters.kinds?.length) count += 1
  return count
}

export function filterSummary(filters?: FilterSet | null): string[] {
  if (!filters) return []
  const parts: string[] = []
  if (filters.years?.length) parts.push(`Годы: ${[...filters.years].sort((a, b) => a - b).join(', ')}`)
  if (filters.year_from !== undefined || filters.year_to !== undefined) {
    parts.push(`Период: ${filters.year_from ?? '…'}–${filters.year_to ?? '…'}`)
  }
  if (filters.issuers?.length) parts.push(`Issuer: ${filters.issuers.join(', ')}`)
  if (filters.allowed_doc_ids?.length) parts.push(`Документы: ${filters.allowed_doc_ids.join(', ')}`)
  if (filters.kinds?.length) parts.push(`Тип: ${filters.kinds.join(', ')}`)
  return parts
}

function describeDiff(name: string, requested: Array<string | number> | undefined, effective: Array<string | number> | undefined): string | null {
  const left = uniqueSorted(requested)
  const right = uniqueSorted(effective)
  if (left.join('|') === right.join('|')) return null
  const requestedText = left.length > 0 ? left.join(', ') : 'не заданы'
  const effectiveText = right.length > 0 ? right.join(', ') : 'не переданы'
  return `${name}: запрошено «${requestedText}», применено «${effectiveText}».`
}

/**
 * Server-applied filters may differ from requested ones. The UI shows the
 * concrete difference instead of silently displaying another result (§3.2).
 */
export function diffFilters(requested?: FilterSet | null, effective?: FilterSet | null): string[] {
  if (!effective) return []
  const notices: string[] = []
  const years = describeDiff('Годы', requested?.years, effective.years)
  if (years) notices.push(years)
  const issuers = describeDiff('Issuer', requested?.issuers, effective.issuers)
  if (issuers) notices.push(issuers)
  const documents = describeDiff('Документы', requested?.allowed_doc_ids, effective.allowed_doc_ids)
  if (documents) notices.push(documents)
  const kinds = describeDiff('Типы источников', requested?.kinds, effective.kinds)
  if (kinds) notices.push(kinds)

  const requestedRange = `${requested?.year_from ?? ''}-${requested?.year_to ?? ''}`
  const effectiveRange = `${effective.year_from ?? ''}-${effective.year_to ?? ''}`
  if (requestedRange !== effectiveRange) {
    const format = (from?: number, to?: number): string =>
      from === undefined && to === undefined ? 'не задан' : `${from ?? '…'}–${to ?? '…'}`
    notices.push(
      `Диапазон лет: запрошен «${format(requested?.year_from, requested?.year_to)}», применён «${format(effective.year_from, effective.year_to)}».`,
    )
  }
  return notices
}
