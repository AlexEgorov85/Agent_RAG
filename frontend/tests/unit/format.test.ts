import { describe, expect, it } from 'vitest'
import {
  EMPTY_VALUE,
  NOT_SPECIFIED,
  formatDate,
  formatDuration,
  formatNumber,
  formatPages,
  orNotSpecified,
  pluralize,
} from '../../src/lib/format'

describe('format helpers', () => {
  it('отсутствующие строковые поля показываются как «не указано»', () => {
    expect(orNotSpecified(undefined)).toBe(NOT_SPECIFIED)
    expect(orNotSpecified(null)).toBe(NOT_SPECIFIED)
    expect(orNotSpecified('   ')).toBe(NOT_SPECIFIED)
    expect(orNotSpecified(0)).toBe('0')
    expect(orNotSpecified('ПАО Сбербанк')).toBe('ПАО Сбербанк')
  })

  it('отсутствующее число не превращается в ноль', () => {
    expect(formatNumber(undefined)).toBe(EMPTY_VALUE)
    expect(formatNumber(null)).toBe(EMPTY_VALUE)
    expect(formatNumber(0)).toBe('0')
    expect(formatNumber(1508.6)).toMatch(/1\s?508,6/)
  })

  it('страницы берутся только из переданных полей', () => {
    expect(formatPages({ page_start: 12, page_end: 12, page_label: '12' })).toBe('стр. 12')
    expect(formatPages({ page_start: 12, page_end: 15 })).toBe('стр. 12–15')
    expect(formatPages({ page_label: 'iv' })).toBe('метка страницы iv')
    expect(formatPages({})).toBe(NOT_SPECIFIED)
  })

  it('дата и длительность не выдумываются', () => {
    expect(formatDate(undefined)).toBe(EMPTY_VALUE)
    expect(formatDate('не дата')).toBe(EMPTY_VALUE)
    expect(formatDuration(undefined)).toBe(EMPTY_VALUE)
    expect(formatDuration(840)).toBe('840 мс')
    expect(formatDuration(1520)).toMatch(/1,5\s?с/)
  })

  it('склоняет русские существительные', () => {
    expect(pluralize(1, 'источник', 'источника', 'источников')).toBe('источник')
    expect(pluralize(3, 'источник', 'источника', 'источников')).toBe('источника')
    expect(pluralize(12, 'источник', 'источника', 'источников')).toBe('источников')
  })
})
