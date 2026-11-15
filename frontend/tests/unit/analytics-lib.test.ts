import { describe, expect, it, vi } from 'vitest'
import type { CalculationSeries, Fact, StructuralChange } from '../../src/types/analytics'
import {
  buildChartRows,
  chartPointEvidence,
  csvFileName,
  downloadCsv,
  filterStructuralChanges,
  formatFactValue,
  groupCalculations,
  groupFacts,
  numericValue,
  seriesWithoutPoints,
  sortFactRows,
  sortStructuralChanges,
  structuralTitle,
  toCsv,
} from '../../src/lib/analytics'
import { pdfFileHref, pdfUnavailableReason, sourcePage } from '../../src/lib/pdf-links'

function fact(overrides: Partial<Fact> = {}): Fact {
  return {
    metric: 'revenue',
    metric_label: 'Выручка',
    year: 2023,
    value: 1508.6,
    unit: 'млрд руб.',
    chunk_id: 'chunk-1',
    ...overrides,
  }
}

function series(overrides: Partial<CalculationSeries> = {}): CalculationSeries {
  return {
    metric: 'revenue',
    metric_label: 'Выручка',
    unit: 'млрд руб.',
    from_year: 2020,
    to_year: 2023,
    absolute_change: 300,
    percent_change: 24.8,
    cagr: 7.6,
    direction: 'increase',
    ...overrides,
  }
}

describe('facts: grouping by metric and unit', () => {
  it('не смешивает разные единицы одного показателя', () => {
    const groups = groupFacts([
      fact({ year: 2022, value: 1200 }),
      fact({ year: 2023, value: 1508.6 }),
      fact({ year: 2023, value: 1.5086, unit: 'трлн руб.', chunk_id: 'chunk-2' }),
    ])

    expect(groups).toHaveLength(2)
    const billion = groups.find((group) => group.unit === 'млрд руб.')
    const trillion = groups.find((group) => group.unit === 'трлн руб.')
    expect(billion?.rows).toHaveLength(2)
    expect(trillion?.rows).toHaveLength(1)
    expect(billion?.rows.map((row) => row.year)).toEqual([2022, 2023])
    expect(billion?.unitLabel).toBe('млрд руб.')
  })

  it('помечает отсутствующую единицу и не строит сравнение', () => {
    const [group] = groupFacts([fact({ unit: undefined })])
    expect(group?.unitLabel).toBe('единица не указана')
    expect(group?.unitSpecified).toBe(false)
  })

  it('сохраняет scale рядом с unit, не подменяя единицу', () => {
    const [group] = groupFacts([fact({ unit: 'руб.', scale: 'млн' })])
    expect(group?.unitLabel).toBe('млн руб.')
    expect(group?.scale).toBe('млн')
  })

  it('сортирует значения численно, нечисловые — в конце', () => {
    const rows = groupFacts([
      fact({ year: 2021, value: 'н/д', chunk_id: 'c1' }),
      fact({ year: 2022, value: 9.5, chunk_id: 'c2' }),
      fact({ year: 2023, value: 100, chunk_id: 'c3' }),
    ])[0]!.rows

    expect(sortFactRows(rows, 'value', 'desc').map((row) => row.year)).toEqual([2023, 2022, 2021])
    expect(sortFactRows(rows, 'value', 'asc').map((row) => row.year)).toEqual([2022, 2023, 2021])
  })

  it('форматирует числа по ru-RU и никогда не превращает null в 0', () => {
    expect(formatFactValue(1508.6).replace(/\u00a0/g, ' ')).toBe('1 508,6')
    expect(formatFactValue(null)).toBe('—')
    expect(numericValue('1 508,6')).toBe(1508.6)
    expect(numericValue('н/д')).toBeNull()
  })
})

describe('calculations: grouping and chart data', () => {
  it('разделяет расчёты по единицам измерения', () => {
    const groups = groupCalculations([
      series({ metric: 'revenue' }),
      series({ metric: 'profit', unit: 'млн руб.' }),
      series({ metric: 'assets' }),
    ])
    expect(groups).toHaveLength(2)
    expect(groups.find((group) => group.unit === 'млрд руб.')?.series).toHaveLength(2)
    expect(groups.find((group) => group.unit === 'млн руб.')?.series).toHaveLength(1)
  })

  it('находит ряды без points, для которых график не строится', () => {
    const withPoints = series({
      points: [{ year: 2020, value: 100, evidence_ids: ['c1'] }],
    })
    const without = series({ metric: 'profit', points: undefined })
    expect(seriesWithoutPoints([withPoints, without]).map((item) => item.metric)).toEqual(['profit'])
  })

  it('пропуск в ряду — null, а не ноль; ось лет непрерывная', () => {
    const { rows, hasGap } = buildChartRows([
      {
        key: 's1',
        label: 'Выручка',
        points: [
          { year: 2020, value: 100, evidence_ids: ['c1'] },
          { year: 2022, value: 120, evidence_ids: ['c2'] },
        ],
      },
    ])

    expect(rows.map((row) => row.year)).toEqual([2020, 2021, 2022])
    expect(rows[1]?.s1).toBeNull()
    expect(rows[1]?.s1).not.toBe(0)
    expect(hasGap).toBe(true)
  })

  it('объединяет несколько рядов одной единицы по годам без вычислений', () => {
    const { rows } = buildChartRows([
      {
        key: 'a',
        label: 'Выручка',
        points: [{ year: 2020, value: 100, evidence_ids: [] }],
      },
      {
        key: 'b',
        label: 'Прибыль',
        points: [{ year: 2021, value: 20, evidence_ids: [] }],
      },
    ])
    expect(rows).toEqual([
      { year: 2020, a: 100, b: null },
      { year: 2021, a: null, b: 20 },
    ])
  })

  it('возвращает evidence_ids точки для перехода к источнику', () => {
    const entries = chartPointEvidence([
      {
        key: 's1',
        label: 'Выручка',
        points: [{ year: 2023, value: 1508.6, evidence_ids: ['chunk-1'] }],
      },
    ])
    expect(entries).toEqual([{ key: 's1', year: 2023, evidenceIds: ['chunk-1'] }])
  })
})

describe('CSV export', () => {
  function readBlob(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result ?? ''))
      reader.onerror = () => reject(reader.error)
      reader.readAsText(blob)
    })
  }

  function readBytes(blob: Blob): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer))
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(blob)
    })
  }

  interface Row {
    label: string
    value: number | null
  }

  it('сериализует отображаемые строки и экранирует разделители', () => {
    const csv = toCsv<Row>(
      [
        { key: 'label', label: 'Показатель', value: (row) => row.label },
        { key: 'value', label: 'Значение', value: (row) => row.value },
      ],
      [
        { label: 'Выручка, млрд "руб."', value: 1508.6 },
        { label: 'Строка\nс переносом', value: null },
      ],
    )

    expect(csv.split('\r\n')).toHaveLength(3)
    expect(csv).toContain('"Выручка, млрд ""руб."""')
    expect(csv).toContain('"Строка\nс переносом"')
    expect(csv.endsWith(',')).toBe(true)
  })

  it('формирует безопасное имя файла', () => {
    expect(csvFileName('Facts / Выручка (млрд руб.)')).toBe('facts-выручка-млрд-руб.csv')
    expect(csvFileName('***')).toBe('analytics.csv')
  })

  it('скачивает CSV с BOM через object URL', async () => {
    const blobs: Blob[] = []
    const createObjectURL = (blob: Blob) => {
      blobs.push(blob)
      return 'blob:test'
    }
    const originalCreate = URL.createObjectURL
    const originalRevoke = URL.revokeObjectURL
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    URL.createObjectURL = createObjectURL as typeof URL.createObjectURL
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL

    try {
      const ok = downloadCsv('facts.csv', 'a,b\r\n1,2')
      expect(ok).toBe(true)
      expect(click).toHaveBeenCalled()
      const blob = blobs[0]
      expect(blob).toBeDefined()
      const bytes = await readBytes(blob!)
      expect([...bytes.slice(0, 3)]).toEqual([0xef, 0xbb, 0xbf])
      const text = await readBlob(blob!)
      expect(text).toContain('a,b\r\n1,2')
    } finally {
      URL.createObjectURL = originalCreate
      URL.revokeObjectURL = originalRevoke
    }
  })
})

describe('structural changes', () => {
  const change = (overrides: Partial<StructuralChange> = {}): StructuralChange => ({
    from_year: 2022,
    to_year: 2023,
    from_section: { title: 'Стратегия', section_path: 'Отчёт / Стратегия', level: 1 },
    to_section: { title: 'Стратегия', section_path: 'Отчёт / Стратегия', level: 1 },
    change_type: 'stable',
    ...overrides,
  })

  it('отбрасывает split/merge и неизвестные категории', () => {
    const result = filterStructuralChanges([
      change(),
      change({ change_type: 'split' as StructuralChange['change_type'], to_section: { title: 'ESG' } }),
      change({ change_type: 'merge' as StructuralChange['change_type'] }),
    ])
    expect(result.supported).toHaveLength(1)
    expect(result.unsupported.sort()).toEqual(['merge', 'split'])
  })

  it('сортирует по годам, затем по приоритету изменения', () => {
    const sorted = sortStructuralChanges([
      change({ change_type: 'stable' }),
      change({ change_type: 'added', to_section: { title: 'ESG' } }),
      change({ from_year: 2021, to_year: 2022, change_type: 'removed' }),
    ])
    expect(sorted.map((item) => `${item.from_year}-${item.to_year}:${item.change_type}`)).toEqual([
      '2021-2022:removed',
      '2022-2023:added',
      '2022-2023:stable',
    ])
  })

  it('берёт название из to_section, иначе из from_section', () => {
    expect(structuralTitle(change())).toBe('Стратегия')
    expect(
      structuralTitle(change({ to_section: undefined, from_section: { title: 'Риски' } })),
    ).toBe('Риски')
    expect(structuralTitle(change({ to_section: undefined, from_section: undefined }))).toBe(
      'раздел без названия',
    )
  })
})

describe('PDF links', () => {
  it('строит fragment по page_pdf, затем по page_start', () => {
    expect(pdfFileHref({ document_id: 'sber_2023', page_pdf: 7, page_start: 12 }, 'enabled')).toBe(
      '/api/documents/sber_2023/file#page=7',
    )
    expect(pdfFileHref({ document_id: 'sber_2023', page_start: 12 }, 'enabled')).toBe(
      '/api/documents/sber_2023/file#page=12',
    )
  })

  it('без страницы даёт ссылку без fragment', () => {
    expect(pdfFileHref({ document_id: 'sber_2023' }, 'enabled')).toBe(
      '/api/documents/sber_2023/file',
    )
  })

  it('не выдумывает URL без документа или без capability', () => {
    expect(pdfFileHref({ page_start: 12 }, 'enabled')).toBeNull()
    expect(pdfFileHref({ document_id: 'sber_2023', page_start: 12 }, 'disabled')).toBeNull()
    expect(pdfUnavailableReason({ document_id: 'sber_2023' }, 'disabled')).toContain(
      'pdf_page_links',
    )
    expect(pdfUnavailableReason({}, 'enabled')).toContain('document_id')
  })

  it('игнорирует неположительные и нечисловые страницы', () => {
    expect(sourcePage({ page_start: 0 })).toBeUndefined()
    expect(sourcePage({ page_start: Number.NaN })).toBeUndefined()
    expect(sourcePage({ page_start: 12 })).toBe(12)
  })
})
