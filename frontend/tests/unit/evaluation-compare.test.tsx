import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { CompareTab } from '../../src/features/evaluation/CompareTab'
import type { EvaluationCompareResponse, EvaluationRunListItem } from '../../src/types/evaluation'

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

function runItem(overrides: Partial<EvaluationRunListItem> = {}): EvaluationRunListItem {
  return {
    id: 'run_b',
    status: 'completed',
    suite: 'smoke',
    split: 'dev',
    retrieval_mode: 'hybrid',
    judge: false,
    repeat: 1,
    created_at: '2026-11-14T12:00:00Z',
    pass_rate: 0.9,
    execution_coverage: 1,
    tests_total: 12,
    tests_eligible: 10,
    tests_executed: 12,
    tests_passed: 9,
    tests_skipped: 2,
    duration_ms: 240000,
    git_sha: 'abc1234',
    dataset_sha256: 'dataset-1',
    corpus_content_sha256: 'corpus-1',
    has_summary: true,
    ...overrides,
  }
}

const baseRun = runItem({ id: 'run_a', created_at: '2026-11-12T12:00:00Z', pass_rate: 0.83 })
const candidateRun = runItem({ id: 'run_b' })

function comparePayload(overrides: Partial<EvaluationCompareResponse> = {}): EvaluationCompareResponse {
  return {
    comparable: true,
    reasons: [],
    version_differences: [],
    base: {
      id: 'run_a',
      status: 'completed',
      suite: 'smoke',
      split: 'dev',
      retrieval_mode: 'hybrid',
      git_sha: 'aaa1111',
      dataset_sha256: 'dataset-1',
      corpus_content_sha256: 'corpus-1',
    },
    candidate: {
      id: 'run_b',
      status: 'completed',
      suite: 'smoke',
      split: 'dev',
      retrieval_mode: 'hybrid',
      git_sha: 'bbb2222',
      dataset_sha256: 'dataset-1',
      corpus_content_sha256: 'corpus-1',
    },
    metrics: [
      { key: 'pass_rate', base: 0.83, candidate: 0.87, delta: 0.04 },
      { key: 'recall_at_10', base: 0.82, candidate: null, delta: null },
    ],
    regressions: [
      {
        test_id: 'easy_004',
        retrieval_mode: 'hybrid',
        attempt_index: 0,
        from: 'pass',
        to: 'fail',
        reasons: ['invalid_citation'],
      },
    ],
    improvements: [],
    test_changes: [],
    ...overrides,
  }
}

describe('CompareTab', () => {
  it('показывает not_comparable и не выдаёт дельты как надёжные', async () => {
    api.compare.mockResolvedValue(
      comparePayload({
        comparable: false,
        reasons: ['corpus_content_sha256_mismatch'],
        metrics: [{ key: 'pass_rate', base: 0.83, candidate: 0.87, delta: null }],
        regressions: [],
      }),
    )
    render(
      <CompareTab runs={[baseRun, candidateRun]} onOpenRuns={vi.fn()} />,
      { wrapper },
    )
    expect(await screen.findByText(/not_comparable/)).toBeInTheDocument()
    expect(screen.getByText(/Разный content hash корпуса/)).toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
    expect(screen.queryByText('+4,0 п.п.')).not.toBeInTheDocument()
  })

  it('показывает дельты, регрессии и различия версий', async () => {
    api.compare.mockResolvedValue(
      comparePayload({ version_differences: ['git_sha_differs'] }),
    )
    render(
      <CompareTab runs={[baseRun, candidateRun]} onOpenRuns={vi.fn()} />,
      { wrapper },
    )
    expect(await screen.findByText('+4 п.п.')).toBeInTheDocument()
    expect(screen.getByText('easy_004')).toBeInTheDocument()
    const row = screen.getByText('easy_004').closest('li')
    expect(row?.textContent).toContain('pass')
    expect(row?.textContent).toContain('fail')
    expect(screen.getByText(/Разный git SHA/)).toBeInTheDocument()
    expect(screen.queryByText(/not_comparable/)).not.toBeInTheDocument()
  })

  it('просит два завершённых запуска', () => {
    render(<CompareTab runs={[baseRun]} onOpenRuns={vi.fn()} />, { wrapper })
    expect(screen.getByText('Нужны два завершённых запуска')).toBeInTheDocument()
  })
})
