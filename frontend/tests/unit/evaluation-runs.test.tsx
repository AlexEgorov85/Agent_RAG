import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { RunForm } from '../../src/features/evaluation/RunForm'
import { RunCard } from '../../src/features/evaluation/RunCard'
import type {
  EvaluationCapabilities,
  EvaluationRun,
  EvaluationRunCreateRequest,
  EvaluationRunListItem,
} from '../../src/types/evaluation'

const capabilities: EvaluationCapabilities = {
  contract_version: '1',
  token_streaming: false,
  incremental_evidence: false,
  coverage: true,
  calculations: true,
  structure_comparison: true,
  structure_content_comparison: false,
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

const activeRun: EvaluationRunListItem = {
  id: 'run_active',
  status: 'running',
  suite: 'smoke',
  split: 'dev',
  retrieval_mode: 'hybrid',
  judge: false,
  repeat: 1,
  created_at: '2026-11-14T12:00:00Z',
  tests_total: 12,
  tests_eligible: 0,
  tests_executed: 0,
  tests_passed: 0,
  tests_skipped: 0,
  has_summary: false,
  progress: {
    tests_total: 12,
    tests_completed: 5,
    current_test_id: 'medium_002',
    current_turn: 1,
    percent: 41.7,
  },
}

describe('RunForm', () => {
  it('требует подтверждение holdout и передаёт его в API', async () => {
    const onCreate = vi.fn(async (_payload: EvaluationRunCreateRequest) => ({
      id: 'run_new',
      status: 'queued' as const,
      status_url: '/api/evaluation/runs/run_new',
      created_at: '2026-11-14T12:00:00Z',
    }))
    render(
      <RunForm
        capabilities={capabilities}
        activeRun={null}
        pending={false}
        error={null}
        onCreate={onCreate}
        onCreated={vi.fn()}
      />,
    )
    await userEvent.selectOptions(screen.getByLabelText('Набор тестов'), 'full')
    expect(screen.getByText('Holdout-набор')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Запустить Full/ }))
    expect(await screen.findByText(/Holdout требует явного подтверждения/)).toBeInTheDocument()
    expect(onCreate).not.toHaveBeenCalled()

    await userEvent.click(screen.getByLabelText('Подтверждаю holdout-запуск'))
    await userEvent.click(screen.getByRole('button', { name: /Запустить Full/ }))
    expect(onCreate).toHaveBeenCalledTimes(1)
    expect(onCreate.mock.calls[0]?.[0]).toMatchObject({
      suite: 'full',
      split: 'all',
      confirm_holdout: true,
      judge: false,
    })
  })

  it('блокирует запуск при активном benchmark и объясняет причину', async () => {
    render(
      <RunForm
        capabilities={capabilities}
        activeRun={activeRun}
        pending={false}
        error={null}
        onCreate={vi.fn()}
        onCreated={vi.fn()}
      />,
    )
    expect(screen.getByText(/Уже выполняется запуск run_active/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Запустить Smoke/ })).toBeDisabled()
  })

  it('не позволяет включить недоступного судью', () => {
    render(
      <RunForm
        capabilities={capabilities}
        activeRun={null}
        pending={false}
        error={null}
        onCreate={vi.fn()}
        onCreated={vi.fn()}
      />,
    )
    expect(screen.getByLabelText('LLM-судья')).toBeDisabled()
    expect(screen.getByText(/judge-модель не подключена/)).toBeInTheDocument()
  })
})

function run(overrides: Partial<EvaluationRun> = {}): EvaluationRun {
  return {
    id: 'run_1',
    status: 'cancelled',
    suite: 'smoke',
    split: 'dev',
    retrieval_mode: 'hybrid',
    judge: false,
    repeat: 1,
    created_at: '2026-11-14T12:00:00Z',
    started_at: '2026-11-14T12:00:10Z',
    finished_at: '2026-11-14T12:01:00Z',
    progress: {
      tests_total: 12,
      tests_completed: 4,
      current_test_id: null,
      current_turn: null,
      percent: 33.3,
    },
    dataset: { version: '1.1', sha256: 'dataset-sha', tests_total: 12 },
    versions: {
      git_sha: 'abc1234',
      git_dirty: true,
      corpus_content_sha256: 'corpus-sha',
      corpus_fingerprint: 'index-fp',
      config_fingerprint: 'config-fp',
      llm_model: 'test-llm',
      reranker_model: null,
      judge_model: null,
      prompt_hashes: {},
    },
    summary: {
      pass_rate: 0.5,
      execution_coverage: 0.33,
      coverage: {
        tests_selected: 12,
        tests_eligible: 4,
        executed: 4,
        passed: 2,
        skipped: 0,
        errors: 0,
      },
      groups: {
        retrieval: { recall_at_10: 0.75, evidence_coverage: 0.75 },
        answer: { automatic_pass_rate: 0.5, rubric_mean: null },
        reasoning: { pass_rate: null },
        dialogue: { scenario_success_rate: null },
      },
      critical_errors: [{ code: 'invalid_citation', count: 1, test_ids: ['easy_002'] }],
      problem_tests: [{ id: 'easy_002', reasons: ['invalid_citation'] }],
      efficiency: { mean_latency_ms: 1200 },
      limitations: [],
      errors: [],
    },
    error: { code: 'cancelled', message: 'Запуск отменён.', retryable: false },
    artifacts_url: '/api/evaluation/runs/run_1/report',
    artifacts_available: true,
    ...overrides,
  }
}

describe('RunCard', () => {
  it('показывает частичные результаты и версии для cancelled', () => {
    render(
      <RunCard
        run={run()}
        canCancel
        cancelPending={false}
        onCancel={vi.fn()}
        onOpenTests={vi.fn()}
        onCompare={vi.fn()}
      />,
    )
    expect(screen.getByText('Частичные результаты')).toBeInTheDocument()
    expect(screen.getByText(/abc1234 \(dirty\)/)).toBeInTheDocument()
    expect(screen.getByText('Отменён')).toBeInTheDocument()
    expect(screen.getByText('invalid_citation')).toBeInTheDocument()
  })

  it('показывает ошибку failed и не маскирует её процентом', () => {
    render(
      <RunCard
        run={run({
          id: 'run_failed',
          status: 'failed',
          summary: null,
          error: { code: 'artifact_import_error', message: 'Нет summary', retryable: true },
        })}
        canCancel
        cancelPending={false}
        onCancel={vi.fn()}
        onOpenTests={vi.fn()}
        onCompare={vi.fn()}
      />,
    )
    expect(screen.getByText('Ошибка')).toBeInTheDocument()
    expect(screen.getByText(/artifact_import_error/)).toBeInTheDocument()
    expect(screen.getByText(/summary отсутствует/)).toBeInTheDocument()
  })

  it('предлагает отмену только для активного запуска', () => {
    const { rerender } = render(
      <RunCard
        run={run({ status: 'running', summary: null, error: null })}
        canCancel
        cancelPending={false}
        onCancel={vi.fn()}
        onOpenTests={vi.fn()}
        onCompare={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: /Отменить/ })).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()

    rerender(
      <RunCard
        run={run({ status: 'completed' })}
        canCancel
        cancelPending={false}
        onCancel={vi.fn()}
        onOpenTests={vi.fn()}
        onCompare={vi.fn()}
      />,
    )
    expect(screen.queryByRole('button', { name: /Отменить/ })).not.toBeInTheDocument()
  })
})
