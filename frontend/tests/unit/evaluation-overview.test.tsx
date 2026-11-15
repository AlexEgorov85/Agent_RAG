import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { OverviewTab } from '../../src/features/evaluation/OverviewTab'
import type {
  EvaluationRun,
  EvaluationRunListItem,
  EvaluationSummary,
} from '../../src/types/evaluation'

const api = vi.hoisted(() => ({
  suites: vi.fn(),
  createRun: vi.fn(),
  listRuns: vi.fn(),
  getRun: vi.fn(),
  listRunTests: vi.fn(),
  getRunTest: vi.fn(),
  getRunTestTrace: vi.fn(),
  getReport: vi.fn(),
  cancelRun: vi.fn(),
  listTests: vi.fn(),
  getTest: vi.fn(),
  compare: vi.fn(),
}))

vi.mock('../../src/features/evaluation/evaluation-api', () => ({ evaluationApi: api }))

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

function summary(overrides: Partial<EvaluationSummary> = {}): EvaluationSummary {
  return {
    pass_rate: 0,
    execution_coverage: 1,
    coverage: {
      tests_selected: 10,
      tests_eligible: 10,
      executed: 10,
      passed: 0,
      skipped: 0,
      errors: 0,
    },
    groups: {
      retrieval: { recall_at_10: null, evidence_coverage: 0.6 },
      answer: { automatic_pass_rate: 0.4, rubric_mean: null },
      reasoning: { pass_rate: 0.3 },
      dialogue: { scenario_success_rate: null },
    },
    critical_errors: [{ code: 'invalid_citation', count: 2, test_ids: ['easy_002'] }],
    problem_tests: [{ id: 'easy_002', reasons: ['invalid_citation'] }],
    efficiency: { mean_latency_ms: 900 },
    limitations: [],
    errors: [],
    ...overrides,
  }
}

function listItem(overrides: Partial<EvaluationRunListItem> = {}): EvaluationRunListItem {
  return {
    id: 'run_last',
    status: 'completed',
    suite: 'smoke',
    split: 'dev',
    retrieval_mode: 'hybrid',
    judge: false,
    repeat: 1,
    created_at: '2026-11-14T12:00:00Z',
    pass_rate: 0,
    execution_coverage: 1,
    tests_total: 10,
    tests_eligible: 10,
    tests_executed: 10,
    tests_passed: 0,
    tests_skipped: 0,
    git_sha: 'abc1234',
    dataset_sha256: 'dataset-1',
    corpus_content_sha256: 'corpus-1',
    has_summary: true,
    ...overrides,
  }
}

function runDetail(id: string, data: EvaluationSummary): EvaluationRun {
  return {
    id,
    status: 'completed',
    suite: 'smoke',
    split: 'dev',
    retrieval_mode: 'hybrid',
    judge: false,
    repeat: 1,
    created_at: '2026-11-14T12:00:00Z',
    progress: { tests_total: 10, tests_completed: 10, percent: 100 },
    dataset: { version: '1.1', sha256: 'dataset-1', tests_total: 10 },
    versions: {
      git_sha: 'abc1234',
      git_dirty: false,
      corpus_content_sha256: 'corpus-1',
      prompt_hashes: {},
    },
    summary: data,
  }
}

const previous = listItem({
  id: 'run_prev',
  created_at: '2026-11-12T12:00:00Z',
  pass_rate: 0.8,
})
const last = listItem()

function renderOverview(runs: EvaluationRunListItem[], overrides: Record<string, unknown> = {}) {
  return render(
    <OverviewTab
      runs={runs}
      isLoading={false}
      error={null}
      onRetry={vi.fn()}
      onStartSmoke={vi.fn()}
      onOpenRuns={vi.fn()}
      onOpenCompare={vi.fn()}
      onOpenTests={vi.fn()}
      {...overrides}
    />,
    { wrapper },
  )
}

describe('OverviewTab', () => {
  it('показывает реальные нули и «—» для неизмеренных метрик, дельта только при сопоставимости', async () => {
    api.getRun.mockImplementation(async (id: string) =>
      runDetail(id, summary(id === 'run_prev' ? { pass_rate: 0.8 } : {})),
    )
    api.compare.mockResolvedValue({
      comparable: false,
      reasons: ['corpus_content_sha256_mismatch'],
      version_differences: [],
      base: { id: 'run_prev' },
      candidate: { id: 'run_last' },
      metrics: [{ key: 'pass_rate', base: 0.8, candidate: 0, delta: null }],
      regressions: [],
      improvements: [],
      test_changes: [],
    })
    renderOverview([last, previous])

    await waitFor(() => {
      expect(screen.getByTestId('metric-Успешные тесты')).toHaveTextContent('0%')
      expect(
        screen.getByText('0/10 eligible · 10/10 executed · coverage 100%'),
      ).toBeInTheDocument()
    })
    expect(screen.getByTestId('metric-Recall@10')).toHaveTextContent('—')
    expect(screen.getByTestId('metric-Multi-turn')).toHaveTextContent('—')
    await waitFor(() => {
      expect(screen.getAllByText('Несопоставимо: смотрите side-by-side').length).toBeGreaterThan(0)
    })
    expect(screen.getByText('invalid_citation')).toBeInTheDocument()
  })

  it('показывает пустое состояние без выдуманных чисел', () => {
    renderOverview([])
    expect(screen.getByText('Запусков ещё не было')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Запустить Smoke/ })).toBeInTheDocument()
  })

  it('показывает progress вместо старых значений при активном запуске', () => {
    api.getRun.mockImplementation(async (id: string) => runDetail(id, summary()))
    const active = listItem({
      id: 'run_active',
      status: 'running',
      pass_rate: null,
      has_summary: false,
      progress: {
        tests_total: 12,
        tests_completed: 5,
        current_test_id: 'medium_002',
        current_turn: 1,
        percent: 41.7,
      },
    })
    renderOverview([active, previous])
    expect(screen.getByText('Текущий запуск')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '41.7')
    expect(screen.getByTestId('metric-Успешные тесты')).toHaveTextContent('—')
  })
})
