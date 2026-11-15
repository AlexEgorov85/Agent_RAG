import type { EvidenceItem } from '../types/evidence'

/**
 * Display helpers only. Nothing here derives, aggregates or recomputes values.
 * Missing data is rendered honestly: "не указано" for metadata, "—" for numbers.
 */

export const NOT_SPECIFIED = 'не указано'
export const EMPTY_VALUE = '—'

export function orNotSpecified(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return NOT_SPECIFIED
  if (typeof value === 'string' && value.trim() === '') return NOT_SPECIFIED
  return String(value)
}

/** Pages come exclusively from backend fields; never inferred from the quote. */
export function formatPages(item: Pick<EvidenceItem, 'page_start' | 'page_end' | 'page_label'>): string {
  const { page_start: start, page_end: end, page_label: label } = item
  if (start !== undefined && start !== null) {
    if (end !== undefined && end !== null && end !== start) return `стр. ${start}–${end}`
    return `стр. ${start}`
  }
  if (label) return `метка страницы ${label}`
  return NOT_SPECIFIED
}

const dateFormatter = new Intl.DateTimeFormat('ru-RU', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
})

const dateTimeFormatter = new Intl.DateTimeFormat('ru-RU', {
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

export function formatDate(iso: string | undefined | null): string {
  if (!iso) return EMPTY_VALUE
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY_VALUE
  return dateFormatter.format(date)
}

export function formatDateTime(iso: string | undefined | null): string {
  if (!iso) return EMPTY_VALUE
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return EMPTY_VALUE
  return dateTimeFormatter.format(date)
}

export function formatTime(iso: string | undefined | null): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('ru-RU', { hour: '2-digit', minute: '2-digit' }).format(date)
}

/** Formatting of already provided numbers; never used for arithmetic results. */
export function formatNumber(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === '') return EMPTY_VALUE
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return EMPTY_VALUE
    return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 4 }).format(value)
  }
  return value
}

export function formatDuration(ms: number | undefined | null): string {
  if (ms === undefined || ms === null || !Number.isFinite(ms)) return EMPTY_VALUE
  if (ms < 1000) return `${Math.round(ms)} мс`
  return `${(ms / 1000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} с`
}

export function pluralize(count: number, one: string, few: string, many: string): string {
  const mod10 = count % 10
  const mod100 = count % 100
  if (mod10 === 1 && mod100 !== 11) return one
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few
  return many
}

export function previewText(text: string | undefined | null, max = 120): string {
  if (!text) return ''
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}
