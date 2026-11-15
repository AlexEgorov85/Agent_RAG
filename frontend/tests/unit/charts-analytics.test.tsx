import type { ReactNode } from 'react'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { CalculationSeries } from '../../src/types/analytics'
import type { EvidenceItem } from '../../src/types/evidence'
import { CalculationCharts } from '../../src/components/chat/AnalyticsCharts'

vi.mock('recharts', () => {
  const Passthrough = ({ children }: { children?: ReactNode }) => <div>{children}</div>
  const Empty = () => null
  return {
    ResponsiveContainer: Passthrough,
    LineChart: Passthrough,
    BarChart: Passthrough,
    XAxis: Empty,
    YAxis: Empty,
    CartesianGrid: Empty,
    Tooltip: Empty,
    Legend: Empty,
    Line: Empty,
    Bar: Empty,
  }
})

function series(overrides: Partial<CalculationSeries> = {}): CalculationSeries {
  return {
    metric: 'revenue',
    metric_label: 'Выручка',
    unit: 'млрд руб.',
    points: [
      { year: 2020, value: 100, evidence_ids: ['chunk-1'] },
      { year: 2022, value: 120, evidence_ids: [] },
    ],
    from_year: 2020,
    to_year: 2022,
    absolute_change: 20,
    percent_change: 20,
    cagr: 9.5,
    direction: 'increase',
    ...overrides,
  }
}

const evidenceItem: EvidenceItem = {
  evidence_id: 'chunk-1',
  chunk_id: 'chunk-1',
  document_id: 'sber_2023',
  excerpt: '',
  has_full_text: true,
}

const evidenceById = new Map([['chunk-1', evidenceItem]])

describe('CalculationCharts', () => {
  it('строит график только из series/points backend и подписывает единицу', () => {
    render(
      <CalculationCharts
        series={[series()]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={vi.fn()}
      />,
    )

    expect(screen.getByRole('img', { name: /График: Выручка\. Единица: млрд руб\./ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Линия' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Столбцы' })).toHaveAttribute('aria-pressed', 'false')

    const group = screen.getByRole('group', { name: 'Тип графика' })
    expect(within(group).getByRole('button', { name: 'Линия' })).toBeInTheDocument()
  })

  it('переключает тип графика с клавиатуры и мышью', async () => {
    render(
      <CalculationCharts
        series={[series()]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={vi.fn()}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: 'Столбцы' }))
    expect(screen.getByRole('button', { name: 'Столбцы' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Линия' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('показывает пропуск как «нет данных», а не как ноль', () => {
    render(
      <CalculationCharts
        series={[series()]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={vi.fn()}
      />,
    )

    expect(screen.getByText(/Пропуски в данных backend показаны как разрыв/)).toBeInTheDocument()
    const table = screen.getByRole('table')
    expect(within(table).getByText('нет данных')).toBeInTheDocument()
    expect(within(table).queryByText('0')).toBeNull()
  })

  it('ведёт от точки к доказательству, если evidence есть в снимке', async () => {
    const onOpenEvidence = vi.fn()
    render(
      <CalculationCharts
        series={[series()]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={onOpenEvidence}
      />,
    )

    await userEvent.click(
      screen.getByRole('button', { name: 'Открыть источник точки Выручка за 2020' }),
    )
    expect(onOpenEvidence).toHaveBeenCalledWith('chunk-1')
    expect(
      screen.queryByRole('button', { name: 'Открыть источник точки Выручка за 2022' }),
    ).toBeNull()
  })

  it('не строит график без points', () => {
    const { container } = render(
      <CalculationCharts
        series={[series({ points: undefined })]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={vi.fn()}
      />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('группированные столбцы для нескольких показателей одной единицы', () => {
    render(
      <CalculationCharts
        series={[
          series(),
          series({
            metric: 'profit',
            metric_label: 'Прибыль',
            points: [{ year: 2020, value: 30, evidence_ids: [] }],
          }),
        ]}
        unitLabel="млрд руб."
        evidenceById={evidenceById}
        onOpenEvidence={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Столбцы' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('img', { name: /Выручка, Прибыль/ })).toBeInTheDocument()
  })
})
