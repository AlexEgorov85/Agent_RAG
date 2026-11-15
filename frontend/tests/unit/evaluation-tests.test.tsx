import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { TestsTab } from '../../src/features/evaluation/TestsTab'
import { TestDetail } from '../../src/features/evaluation/TestDetail'
import { ApiError } from '../../src/services/api-client'
import type {
  EvaluationTestCatalogResponse,
  EvaluationTestDetail,
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

function catalog(): EvaluationTestCatalogResponse {
  return {
    items: [
      {
        id: 'hard_002',
        suite: 'hard',
        category: 'temporal',
        difficulty: 'hard',
        split: 'dev',
        question: 'Сравни чистую прибыль за 2020–2024 годы.',
        last_run: {
          run_id: 'run_1',
          created_at: '2026-11-14T12:00:00Z',
          retrieval_mode: 'hybrid',
          pass: false,
          status: 'completed',
          critical_errors: [{ code: 'missing_required_year_claimed_complete' }],
          trace_available: true,
        },
      },
      {
        id: 'easy_001',
        suite: 'easy',
        category: 'fact_lookup',
        difficulty: 'easy',
        split: 'dev',
        question: 'Какая выручка за 2023 год?',
        last_run: null,
      },
    ],
    total: 2,
    next_offset: null,
    dataset_available: true,
    run_id: 'run_1',
    retrieval_mode: 'hybrid',
  }
}

function detail(overrides: Partial<EvaluationTestDetail> = {}): EvaluationTestDetail {
  return {
    test: {
      id: 'hard_002',
      suite: 'hard',
      split: 'dev',
      category: 'temporal',
      difficulty: 'hard',
      question: 'Сравни чистую прибыль за 2020–2024 годы.',
      expected_status: 'answer',
      expected: { required_years: [2020, 2021] },
      gold_evidence: { groups: [{ any_of: ['gold-1'] }], min_coverage: 1 },
      reference_answer: '2020: 1, 2021: 2',
    },
    run_id: 'run_1',
    retrieval_mode: 'hybrid',
    attempt_index: 0,
    result: {
      test_id: 'hard_002',
      retrieval_mode: 'hybrid',
      attempt_index: 0,
      suite: 'hard',
      category: 'temporal',
      difficulty: 'hard',
      status: 'completed',
      pass: false,
      metrics: {
        recall_at_10: 0.75,
        evidence_coverage: 0.75,
        loss_stage: 'never_retrieved',
      },
      answer: { automatic_pass: false, fact_accuracy: 0.5 },
      critical_errors: [{ code: 'missing_required_year_claimed_complete' }],
      warnings: [],
      efficiency: { tool_calls: 2, search_calls: 1, latency_ms: 1234 },
      trace_available: true,
    },
    query_plan: {
      intent: 'temporal',
      years: [2020, 2021],
      needs_graph: false,
      answer_mode: 'concise',
    },
    retrieval_calls: [
      {
        call_index: 0,
        tool_name: 'search_reports',
        retrieval_mode: 'hybrid',
        query: 'чистая прибыль 2020 2021',
        stages: {
          dense: [{ chunk_id: 'gold-1', rank: 1 }],
          bm25: null,
          rrf: [],
        },
      },
    ],
    evidence: {
      manifest: ['gold-1'],
      context_chunk_ids: null,
      context_source: 'evidence_manifest',
      coverage: [
        { year: 2020, status: 'covered' },
        { year: 2021, status: 'not_found' },
      ],
      loss_stage: 'never_retrieved',
    },
    answer: {
      status: 'answer',
      text: 'Полный пятилетний ряд без раскрытия пропуска 2021.',
      citations: [{ chunk_id: 'gold-1', year: 2020, page_pdf: 12 }],
      facts: [],
      warnings: [],
      calculations: [],
    },
    evaluation: {
      pass: false,
      critical_errors: [{ code: 'missing_required_year_claimed_complete' }],
      rubric: null,
      automatic_pass: false,
    },
    diagnostics: {
      likely_cause: 'retrieval',
      notes: ['missing_required_year_claimed_complete'],
    },
    trace_available: true,
    trace_url: '/api/evaluation/runs/run_1/tests/hard_002/trace?retrieval_mode=hybrid',
    trace_truncated: false,
    ...overrides,
  }
}

describe('TestsTab and TestDetail', () => {
  it('показывает каталог, not_run и trace_unavailable без пустой таблицы', async () => {
    api.listTests.mockResolvedValue(catalog())
    api.getTest.mockResolvedValue({ test: null, history: [], dataset_available: true })
    api.getRunTest.mockResolvedValue(
      detail({
        trace_available: false,
        trace_url: null,
      }),
    )
    render(<TestsTab onOpenRuns={vi.fn()} />, { wrapper })

    expect(await screen.findByText('hard_002')).toBeInTheDocument()
    expect(screen.getAllByText('not_run').length).toBeGreaterThan(0)
    await userEvent.click(screen.getByRole('button', { name: 'hard_002' }))

    expect(await screen.findByText(/context_chunk_ids недоступен/)).toBeInTheDocument()
    expect(await screen.findByText('trace_unavailable')).toBeInTheDocument()
    expect(screen.getByText('не найдено')).toBeInTheDocument()
    expect(screen.queryByText('год отсутствует в корпусе')).not.toBeInTheDocument()
  })

  it('показывает query plan, retrieval-этапы, ответ и critical errors', async () => {
    api.listTests.mockResolvedValue(catalog())
    api.getTest.mockResolvedValue({ test: null, history: [], dataset_available: true })
    api.getRunTest.mockResolvedValue(detail())
    render(<TestDetail runId="run_1" testId="hard_002" mode="hybrid" />, { wrapper })

    expect(await screen.findByText('Query Plan')).toBeInTheDocument()
    expect(screen.getByText('Dense candidates')).toBeInTheDocument()
    expect(screen.getAllByText(/gold-1/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Полный пятилетний ряд/)).toBeInTheDocument()
    expect(screen.getAllByText('missing_required_year_claimed_complete').length).toBeGreaterThan(0)
    expect(screen.getByText(/Поиск не нашёл доказательство/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Показать trace/ })).toBeInTheDocument()
  })

  it('объясняет ambiguous_test_result для 409', async () => {
    api.getTest.mockResolvedValue({ test: null, history: [], dataset_available: true })
    api.getRunTest.mockRejectedValue(
      new ApiError({
        code: 'ambiguous_test_result',
        message: 'У теста несколько режимов retrieval; укажите retrieval_mode.',
        status: 409,
      }),
    )
    render(<TestDetail runId="run_1" testId="hard_002" mode="hybrid" />, { wrapper })
    expect((await screen.findAllByText(/ambiguous_test_result/)).length).toBeGreaterThan(0)
    expect(screen.getByText(/укажите retrieval_mode/)).toBeInTheDocument()
  })

  it('не выдумывает тесты при недоступном датасете', async () => {
    api.listTests.mockResolvedValue({
      items: [],
      total: 0,
      next_offset: null,
      dataset_available: false,
      run_id: null,
      retrieval_mode: 'hybrid',
    })
    render(<TestsTab onOpenRuns={vi.fn()} />, { wrapper })
    expect(await screen.findByText('Датасет недоступен')).toBeInTheDocument()
  })
})