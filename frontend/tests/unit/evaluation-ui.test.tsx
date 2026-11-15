import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MetricCard, ProgressBar, RunStatusBadge, TestStatusBadge } from '../../src/features/evaluation/components'
import {
  formatDeltaPoints,
  formatDeltaValue,
  formatPercent,
  formatRatio,
  statusLabel,
} from '../../src/features/evaluation/format'
import { EvaluationDisabled, EvaluationPage } from '../../src/features/evaluation/EvaluationPage'
import type { EvaluationCapabilities } from '../../src/types/evaluation'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('Evaluation format helpers', () => {
  it('не подменяет неизмеренную метрику нулём', () => {
    expect(formatPercent(null)).toBe('—')
    expect(formatPercent(undefined)).toBe('—')
    expect(formatPercent(0)).toBe('0%')
    expect(formatRatio(null)).toBe('—')
    expect(formatRatio(0)).toBe('0')
    expect(formatDeltaPoints(null)).toBe('—')
    expect(formatDeltaValue(null)).toBe('—')
  })

  it('форматирует дельты в п.п. и единицах метрики', () => {
    // ru-RU: десятичный разделитель — запятая.
    expect(formatDeltaPoints(0.021)).toBe('+2,1 п.п.')
    expect(formatDeltaPoints(-0.014)).toBe('−1,4 п.п.')
    expect(formatDeltaPoints(0)).toBe('0')
    expect(formatDeltaValue(0.09)).toBe('+0,09')
    expect(formatDeltaValue(-0.05)).toBe('−0,05')
  })

  it('переводит статусы запуска и теста', () => {
    expect(statusLabel('cancel_requested')).toBe('Останавливается…')
    expect(statusLabel('interrupted')).toBe('Прерван')
    expect(statusLabel('unknown_status')).toBe('unknown_status')
  })
})

describe('Evaluation badges and cards', () => {
  it('показывает статус запуска и теста', () => {
    render(
      <div>
        <RunStatusBadge status="completed" />
        <RunStatusBadge status="running" />
        <TestStatusBadge status="completed" pass={false} />
        <TestStatusBadge status="skipped" pass={false} />
      </div>,
    )
    expect(screen.getByText('Завершён')).toBeInTheDocument()
    expect(screen.getByText('Выполняется')).toBeInTheDocument()
    expect(screen.getByText('Выполнен · fail')).toBeInTheDocument()
    expect(screen.getByText('Пропущен')).toBeInTheDocument()
  })

  it('различает «—» и измеренный 0% в карточке', () => {
    render(
      <div>
        <MetricCard label="Измерено" value={formatPercent(0)} context="0/12 eligible" />
        <MetricCard label="Не измерено" value={formatPercent(null)} context="rubric не измерялся" />
      </div>,
    )
    expect(screen.getByTestId('metric-Измерено')).toHaveTextContent('0%')
    expect(screen.getByTestId('metric-Не измерено')).toHaveTextContent('—')
  })

  it('показывает progress без выдуманных значений', () => {
    render(<ProgressBar percent={41.7} completed={5} total={12} currentTestId="medium_002" />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '41.7')
    expect(screen.getByText(/5 \/ 12 пар/)).toBeInTheDocument()
    expect(screen.getByText(/текущий medium_002/)).toBeInTheDocument()
  })
})

describe('Evaluation disabled route', () => {
  it('показывает честный evaluation_disabled, а не пустой экран', () => {
    render(<EvaluationDisabled />)
    expect(screen.getByText(/evaluation_disabled/)).toBeInTheDocument()
    expect(screen.getByText(/EVAL_UI_ENABLED=1/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Вернуться к исследованию' })).toBeInTheDocument()
  })
})

const capabilities: EvaluationCapabilities = {
  contract_version: '1',
  token_streaming: false,
  incremental_evidence: false,
  coverage: true,
  calculations: true,
  structure_comparison: true,
  structure_content_comparison: true,
  graph_panel: false,
  pdf_page_links: false,
  evaluation_ui: true,
  evaluation_runs: true,
  evaluation_sse: false,
  evaluation_cancel: true,
  retrieval_modes: ['hybrid', 'dense', 'keyword'],
  judge: false,
  max_repeat: 5,
  max_queued: 3,
}

function renderEvaluation() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ items: [], total: 0, next_offset: null }),
    })),
  )
  return render(
    <QueryClientProvider client={client}>
      <EvaluationPage capabilities={capabilities} />
    </QueryClientProvider>,
  )
}

describe('Evaluation tabs semantics', () => {
  it('вкладки входят в tablist и управляют подписанным tabpanel', async () => {
    renderEvaluation()
    const tablist = await screen.findByRole('tablist', { name: 'Разделы Evaluation' })
    expect(tablist).toBeInTheDocument()

    const overview = screen.getByRole('tab', { name: 'Обзор' })
    expect(overview).toHaveAttribute('aria-selected', 'true')
    expect(overview).toHaveAttribute('aria-controls', 'eval-tabpanel')
    expect(overview).toHaveAttribute('tabindex', '0')

    const runs = screen.getByRole('tab', { name: 'Запуски' })
    expect(runs).toHaveAttribute('aria-selected', 'false')
    expect(runs).toHaveAttribute('aria-controls', 'eval-tabpanel')
    expect(runs).toHaveAttribute('tabindex', '-1')

    const panel = screen.getByRole('tabpanel')
    expect(panel).toHaveAttribute('id', 'eval-tabpanel')
    expect(panel).toHaveAttribute('aria-labelledby', 'eval-tab-overview')
  })

  it('стрелки перемещают выбор и обновляют aria-labelledby панели', async () => {
    renderEvaluation()
    const overview = await screen.findByRole('tab', { name: 'Обзор' })
    overview.focus()
    await userEvent.keyboard('{ArrowRight}')

    expect(screen.getByRole('tab', { name: 'Запуски' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    expect(screen.getByRole('tab', { name: 'Запуски' })).toHaveAttribute('tabindex', '0')
    expect(screen.getByRole('tab', { name: 'Обзор' })).toHaveAttribute('tabindex', '-1')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'eval-tab-runs')
  })
})
