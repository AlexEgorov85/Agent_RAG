import { describe, expect, it } from 'vitest'
import {
  activeFilterCount,
  compactFilters,
  diffFilters,
  filterSummary,
  isEmptyFilters,
} from '../../src/lib/filters'

describe('filter helpers', () => {
  it('не отправляет пустые фильтры на backend', () => {
    expect(compactFilters({ years: [], issuers: [], allowed_doc_ids: [] })).toEqual({})
    expect(isEmptyFilters(compactFilters({ years: [] }))).toBe(true)
  })

  it('сохраняет непустые значения без изменений', () => {
    const filters = { years: [2023, 2024], issuers: ['ПАО Сбербанк'], kinds: ['table' as const] }
    expect(compactFilters(filters)).toEqual(filters)
    expect(activeFilterCount(filters)).toBe(3)
  })

  it('показывает конкретное отличие применённых фильтров', () => {
    const notices = diffFilters(
      { years: [2023], issuers: ['ПАО Сбербанк'] },
      { years: [2023, 2024], issuers: ['ПАО Сбербанк'] },
    )
    expect(notices).toHaveLength(1)
    expect(notices[0]).toContain('Годы')
    expect(notices[0]).toContain('2024')
  })

  it('не показывает предупреждение, если фильтры совпали', () => {
    expect(diffFilters({ years: [2023] }, { years: [2023] })).toEqual([])
  })

  it('описывает фильтры для панели', () => {
    expect(filterSummary({ years: [2024, 2023], issuers: ['ПАО Сбербанк'] })).toEqual([
      'Годы: 2023, 2024',
      'Issuer: ПАО Сбербанк',
    ])
  })
})
