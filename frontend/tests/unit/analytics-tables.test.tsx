import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { CalculationSeries, Fact } from '../../src/types/analytics'
import { groupCalculations, groupFacts } from '../../src/lib/analytics'
import { CalculationTable, FactsTable } from '../../src/components/chat/AnalyticsTables'
import type { EvidenceItem } from '../../src/types/evidence'

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
    points: [{ year: 2023, value: 1508.6, evidence_ids: ['chunk-1'] }],
    from_year: 2020,
    to_year: 2023,
    absolute_change: 300,
    percent_change: 24.8,
    cagr: 7.6,
    direction: 'increase',
    ...overrides,
  }
}

const evidenceItem: EvidenceItem = {
  evidence_id: 'chunk-1',
  chunk_id: 'chunk-1',
  document_id: 'sber_2023',
  document_title: 'Годовой отчёт 2023',
  excerpt: '',
  has_full_text: true,
}

function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('FactsTable', () => {
  it('рисует отдельные таблицы для разных единиц и sticky-заголовок', () => {
    const groups = groupFacts([
      fact({ year: 2022, value: 1200 }),
      fact({ year: 2023, value: 1508.6 }),
      fact({ year: 2023, value: 1.5, unit: 'трлн руб.', chunk_id: 'chunk-2' }),
    ])

    render(
      <div>
        {groups.map((group) => (
          <FactsTable
            key={group.key}
            group={group}
            evidenceById={new Map()}
            onOpenEvidence={vi.fn()}
          />
        ))}
      </div>,
    )

    expect(screen.getAllByRole('table')).toHaveLength(2)
    expect(screen.getAllByText('млрд руб.').length).toBeGreaterThan(0)
    expect(screen.getAllByText('трлн руб.').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/1.508,6/).length).toBeGreaterThan(0)

    const header = screen.getAllByRole('columnheader')[0]!
    expect(header.className).toContain('sticky')
  })

  it('помечает отсутствие единицы и не смешивает сравнение', () => {
    const [group] = groupFacts([fact({ unit: undefined })])

    render(
      <FactsTable
        group={group!}
        evidenceById={new Map()}
        onOpenEvidence={vi.fn()}
      />,
    )

    expect(screen.getByText('единица не указана')).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent(/сравнительный блок .* не строится/)
  })

  it('сортирует по клику и ведёт от значения к evidence', async () => {
    const onOpenEvidence = vi.fn()
    const [group] = groupFacts([
      fact({ year: 2022, value: 1200, chunk_id: 'chunk-2' }),
      fact({ year: 2023, value: 1508.6 }),
    ])

    render(
      <FactsTable
        group={group!}
        evidenceById={new Map([['chunk-1', evidenceItem]])}
        onOpenEvidence={onOpenEvidence}
      />,
    )

    const table = screen.getByRole('table')
    const yearColumn = () =>
      within(table)
        .getAllByRole('row')
        .slice(1)
        .map((row) => within(row).getAllByRole('cell')[1]?.textContent)

    expect(yearColumn()).toEqual(['2022', '2023'])

    await userEvent.click(screen.getByRole('button', { name: 'Сортировать: Значение' }))
    expect(yearColumn()).toEqual(['2022', '2023'])

    await userEvent.click(screen.getByRole('button', { name: 'Сортировать: Значение' }))
    expect(yearColumn()).toEqual(['2023', '2022'])

    await userEvent.click(
      screen.getByRole('button', { name: 'Открыть источник значения Выручка за 2023' }),
    )
    expect(onOpenEvidence).toHaveBeenCalledWith('chunk-1')
  })

  it('экспортирует CSV только отображаемых строк', async () => {
    const blobs: Blob[] = []
    const originalCreate = URL.createObjectURL
    const originalRevoke = URL.revokeObjectURL
    URL.createObjectURL = ((blob: Blob) => {
      blobs.push(blob)
      return 'blob:facts'
    }) as typeof URL.createObjectURL
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    try {
      const [group] = groupFacts([fact({ year: 2022, value: 1200 })])
      render(
        <FactsTable
          group={group!}
          evidenceById={new Map()}
          onOpenEvidence={vi.fn()}
        />,
      )

      await userEvent.click(screen.getByRole('button', { name: 'Экспорт CSV' }))
      expect(screen.getByRole('status')).toHaveTextContent('Экспортировано строк: 1.')

      const csv = (await readBlob(blobs[0]!)).replace(/\u00a0/g, ' ')
      expect(csv).toContain('Показатель,Год,Значение,Единица')
      expect(csv).toContain('Выручка,2022,1 200,млрд руб.')
    } finally {
      URL.createObjectURL = originalCreate
      URL.revokeObjectURL = originalRevoke
    }
  })

  it('показывает пустое состояние без строк', () => {
    const [group] = groupFacts([])
    render(
      <FactsTable
        group={group ?? { key: 'empty', metric: 'm', metricLabel: 'm', unitLabel: 'руб.', unitSpecified: true, rows: [] }}
        evidenceById={new Map()}
        onOpenEvidence={vi.fn()}
      />,
    )
    expect(screen.getByText('Backend не передал факты для этого показателя.')).toBeInTheDocument()
  })
})

describe('CalculationTable', () => {
  it('показывает расчёты backend и честные «не рассчитано backend»', () => {
    const [group] = groupCalculations([
      series({ percent_change: null, cagr: null, direction: 'flat' }),
    ])

    render(<CalculationTable group={group!} />)

    expect(screen.getAllByText(/CAGR рассчитаны backend/).length).toBeGreaterThan(0)
    expect(screen.getAllByText('не рассчитано backend')).toHaveLength(2)
    expect(screen.getByText('Без изменений')).toBeInTheDocument()
    expect(screen.getByText('2020 → 2023')).toBeInTheDocument()
  })

  it('не смешивает единицы в одной таблице', () => {
    const groups = groupCalculations([
      series(),
      series({ metric: 'profit', unit: 'млн руб.', metric_label: 'Прибыль' }),
    ])
    expect(groups.map((group) => group.unitLabel)).toEqual(['млрд руб.', 'млн руб.'])
  })
})
