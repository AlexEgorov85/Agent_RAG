/**
 * Evaluation / Benchmark contract types (docs/specs/evaluation-webui.md §3, §8).
 *
 * Every number here is produced by the backend (`eval.runner` artifacts or the
 * Evaluation API). The UI never recomputes metrics, deltas or pass rates; it
 * only renders what the API returned and shows `—` when a value is null.
 */

import type { Capabilities } from './api'

export type EvaluationRunStatus =
  | 'queued'
  | 'running'
  | 'cancel_requested'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export type EvaluationSuite = 'smoke' | 'full' | 'easy' | 'medium' | 'hard' | 'multi_turn'

export type EvaluationSplit = 'dev' | 'holdout' | 'all'

export type RetrievalMode = 'hybrid' | 'dense' | 'keyword'

export type RetrievalModeSelection = RetrievalMode | 'all'

export type EvaluationTestStatus = 'completed' | 'failed' | 'timeout' | 'error' | 'skipped'

/** Capabilities plus the evaluation extension from §8.2. */
export type EvaluationCapabilities = Capabilities & {
  evaluation_runs?: boolean
  max_repeat?: number
  max_queued?: number
}

export interface EvaluationProgress {
  tests_total: number
  tests_completed: number
  current_test_id?: string | null
  current_turn?: number | null
  percent: number
}

export interface EvaluationError {
  code: string
  message: string
  retryable?: boolean
  test_id?: string
}

export interface GroupStats {
  pass_rate?: number | null
  eligible?: number | null
  executed?: number | null
  passed?: number | null
}

/** `summary.json` из spec_benchmark.md §6.2 — читается как есть. */
export interface EvaluationSummary {
  run_id?: string
  started_at?: string
  finished_at?: string
  command?: string
  suite?: string
  split?: string
  judge?: boolean
  fake_service?: boolean
  git?: { sha?: string; dirty?: boolean }
  dataset?: {
    version?: string
    sha256?: string
    tests?: number
    expected_corpus_sha256?: string | null
    suites?: Record<string, string>
  }
  corpus?: {
    content_sha256?: string
    chunk_count?: number
    documents?: string[]
    index_fingerprint?: string | null
    graph_fingerprint?: string | null
  }
  config?: Record<string, unknown>
  config_fingerprint?: string | null
  models?: {
    llm?: string | null
    embeddings?: string | null
    reranker?: string | null
    judge?: string | null
  }
  prompts?: Record<string, string | null>
  capabilities?: Record<string, unknown>
  coverage?: {
    tests_selected?: number
    tests_eligible?: number
    executed?: number
    passed?: number
    failed?: number
    skipped?: number
    unsupported?: number
    timeouts?: number
    errors?: number
    judge_errors?: number
  }
  pass_rate?: number | null
  execution_coverage?: number | null
  pass_at_1?: number | null
  stability?: number | null
  groups?: {
    retrieval?: {
      recall_at_5?: number | null
      recall_at_10?: number | null
      mrr?: number | null
      evidence_coverage?: number | null
      context_coverage?: number | null
    }
    answer?: {
      automatic_pass_rate?: number | null
      rubric_mean?: number | null
      fact_accuracy?: number | null
      citation_validity?: number | null
      citation_coverage?: number | null
      calculation_accuracy?: number | null
      abstention_accuracy?: number | null
      status_accuracy?: number | null
    }
    reasoning?: { pass_rate?: number | null; rubric_mean?: number | null }
    dialogue?: { scenario_success_rate?: number | null }
  }
  by_difficulty?: Record<string, GroupStats>
  by_category?: Record<string, GroupStats>
  by_split?: Record<string, GroupStats>
  by_mode?: Record<string, GroupStats>
  critical_errors?: Array<{ code: string; count: number; test_ids?: string[] }>
  problem_tests?: Array<{ id: string; reasons: string[] }>
  efficiency?: {
    mean_tool_calls?: number | null
    mean_search_calls?: number | null
    mean_latency_ms?: number | null
    llm_calls?: number | null
    llm_concurrency?: number | null
  }
  limitations?: string[]
  errors?: Array<{ code: string; message?: string; test_id?: string }>
}

export interface EvaluationVersions {
  git_sha: string
  git_dirty: boolean
  corpus_content_sha256: string
  corpus_fingerprint?: string | null
  config_fingerprint?: string | null
  llm_model?: string | null
  reranker_model?: string | null
  judge_model?: string | null
  prompt_hashes: Record<string, string>
}

export interface EvaluationDatasetInfo {
  version?: string | null
  sha256?: string | null
  tests_total: number
}

export interface EvaluationRun {
  id: string
  status: EvaluationRunStatus
  suite: EvaluationSuite | string
  split: EvaluationSplit | string
  retrieval_mode: RetrievalModeSelection | string
  judge: boolean
  repeat: number
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  progress: EvaluationProgress
  dataset: EvaluationDatasetInfo
  versions: EvaluationVersions
  summary: EvaluationSummary | null
  error?: EvaluationError | null
  artifacts_url?: string | null
  artifacts_available?: boolean
  notes?: string | null
  duration_ms?: number | null
}

export interface EvaluationRunListItem {
  id: string
  status: EvaluationRunStatus
  suite: string
  split: string
  retrieval_mode: string
  judge: boolean
  repeat: number
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  pass_rate?: number | null
  execution_coverage?: number | null
  tests_total: number
  tests_eligible: number
  tests_executed: number
  tests_passed: number
  tests_skipped: number
  duration_ms?: number | null
  git_sha?: string | null
  git_dirty?: boolean
  dataset_sha256?: string | null
  corpus_content_sha256?: string | null
  progress?: EvaluationProgress
  has_summary: boolean
}

export interface EvaluationRunListResponse {
  items: EvaluationRunListItem[]
  total: number
  next_offset: number | null
}

export interface EvaluationSuiteInfo {
  suite: EvaluationSuite
  tests: number | null
  dev: number | null
  holdout: number | null
  requires_holdout_confirmation: boolean
}

export interface EvaluationSuitesResponse {
  items: EvaluationSuiteInfo[]
  dataset_available: boolean
  dataset_sha256?: string | null
  expected_corpus_sha256?: string | null
  retrieval_modes: RetrievalMode[]
  max_repeat: number
  max_queued: number
}

export interface EvaluationRunCreateRequest {
  suite: EvaluationSuite
  split?: EvaluationSplit
  retrieval_mode?: RetrievalModeSelection
  judge?: boolean
  repeat?: number
  notes?: string
  client_request_id?: string
  confirm_holdout?: boolean
}

export interface EvaluationRunCreated {
  id: string
  status: EvaluationRunStatus
  status_url: string
  created_at: string
  duplicate?: boolean
}

export interface EvaluationTestMetrics {
  recall_at_5?: number | null
  recall_at_10?: number | null
  mrr?: number | null
  retrieval_coverage?: number | null
  evidence_coverage?: number | null
  context_coverage?: number | null
  loss_stage?: string | null
  not_measured?: string[]
}

export interface EvaluationCriticalError {
  code: string
  message?: string
  turn_index?: number
}

export interface EvaluationTestResult {
  test_id: string
  retrieval_mode: RetrievalMode
  attempt_index: number
  suite?: string | null
  category?: string | null
  difficulty?: string | null
  status: EvaluationTestStatus | string
  pass: boolean
  metrics: EvaluationTestMetrics
  answer?: {
    automatic_pass?: boolean | null
    rubric_total?: number | null
    fact_accuracy?: number | null
    citation_validity?: number | null
    citation_coverage?: number | null
    calculation_accuracy?: number | null
  }
  critical_errors: EvaluationCriticalError[]
  warnings?: string[]
  efficiency?: {
    tool_calls?: number
    search_calls?: number
    llm_calls?: number | null
    latency_ms?: number
  }
  trace_path?: string | null
  trace_available: boolean
  duration_ms?: number | null
}

export interface EvaluationRunTestsResponse {
  items: EvaluationTestResult[]
  total: number
  next_offset: number | null
}

export interface EvaluationTraceCitation {
  chunk_id?: string | null
  doc_id?: string | null
  year?: number | null
  issuer?: string | null
  page_pdf?: number | null
  page_label?: string | null
  section_path?: string | null
  quote?: string | null
}

export interface EvaluationQueryPlan {
  intent?: string | null
  years?: number[]
  year_from?: number | null
  year_to?: number | null
  needs_graph?: boolean | null
  answer_mode?: string | null
  [key: string]: unknown
}

export interface EvaluationStageItem {
  chunk_id?: string | null
  rank?: number | null
  score?: number | null
  [key: string]: unknown
}

export interface EvaluationRetrievalCall {
  call_index?: number | null
  turn_index?: number | null
  tool_name?: string | null
  query?: string | null
  retrieval_mode?: string | null
  top_k?: number | null
  counts?: Record<string, number>
  stages?: Record<string, EvaluationStageItem[] | null>
  available?: Record<string, boolean>
  [key: string]: unknown
}

export interface EvaluationEvidenceYear {
  year: number
  status: 'covered' | 'not_found' | 'year_absent' | 'unknown'
  expected?: boolean
}

export interface EvaluationTestDetail {
  test: {
    id: string
    suite?: string | null
    split?: string | null
    category?: string | null
    difficulty?: string | null
    question?: string | null
    turns?: Array<Record<string, unknown>> | null
    expected_status?: string | null
    expected?: Record<string, unknown>
    gold_evidence?: Record<string, unknown>
    reference_answer?: string | null
    checks?: Record<string, unknown>
    tags?: string[]
  }
  run_id: string
  retrieval_mode: RetrievalMode
  attempt_index: number
  result: EvaluationTestResult
  query_plan: EvaluationQueryPlan | null
  retrieval_calls: EvaluationRetrievalCall[]
  evidence: {
    manifest: string[]
    context_chunk_ids: string[] | null
    context_source?: string | null
    coverage: EvaluationEvidenceYear[]
    loss_stage?: string | null
  }
  answer: {
    status?: string | null
    text?: string | null
    citations?: EvaluationTraceCitation[]
    facts?: Array<Record<string, unknown>>
    warnings?: string[]
    calculations?: Array<Record<string, unknown>>
  }
  evaluation: {
    pass: boolean
    critical_errors: EvaluationCriticalError[]
    warnings?: string[]
    rubric?: number | null
    automatic_pass?: boolean | null
  }
  diagnostics: {
    likely_cause?: string | null
    notes: string[]
  }
  trace_available: boolean
  trace_url: string | null
  trace_truncated: boolean
}

export interface EvaluationTrace {
  trace_schema_version?: string
  run_id?: string
  test_id?: string
  retrieval_mode?: string
  attempt_index?: number
  status?: string
  turns?: Array<Record<string, unknown>>
  errors?: Array<Record<string, unknown>>
  truncated?: boolean
  [key: string]: unknown
}

export interface EvaluationTestCatalogItem {
  id: string
  suite?: string | null
  category?: string | null
  difficulty?: string | null
  split?: string | null
  question?: string | null
  last_run: {
    run_id: string
    created_at?: string
    retrieval_mode: RetrievalMode
    pass: boolean
    status: string
    critical_errors: EvaluationCriticalError[]
    trace_available?: boolean
  } | null
}

export interface EvaluationTestCatalogResponse {
  items: EvaluationTestCatalogItem[]
  total: number
  next_offset: number | null
  dataset_available: boolean
  run_id: string | null
  retrieval_mode: RetrievalMode
}

export interface EvaluationTestHistoryEntry {
  run_id: string
  created_at: string
  suite?: string | null
  split?: string | null
  retrieval_mode: RetrievalMode
  attempt_index: number
  status: string
  pass: boolean
  metrics: EvaluationTestMetrics
  critical_errors: EvaluationCriticalError[]
  trace_available: boolean
}

export interface EvaluationTestDefinition {
  test: EvaluationTestCatalogItem | null
  history: EvaluationTestHistoryEntry[]
  dataset_available: boolean
}

export interface EvaluationCompareMeta {
  id: string
  status: string
  suite: string
  split: string
  retrieval_mode: string
  created_at?: string | null
  git_sha?: string | null
  git_dirty?: boolean | null
  dataset_sha256?: string | null
  corpus_content_sha256?: string | null
  corpus_fingerprint?: string | null
  llm_model?: string | null
}

export interface EvaluationMetricDelta {
  key: string
  base: number | null
  candidate: number | null
  delta: number | null
}

export interface EvaluationTestChange {
  test_id: string
  retrieval_mode: string
  attempt_index: number
  from: 'pass' | 'fail'
  to: 'pass' | 'fail'
  reasons: string[]
  difficulty?: string | null
  suite?: string | null
}

export interface EvaluationCompareResponse {
  comparable: boolean
  reasons: string[]
  version_differences: string[]
  base: EvaluationCompareMeta
  candidate: EvaluationCompareMeta
  metrics: EvaluationMetricDelta[]
  regressions: EvaluationTestChange[]
  improvements: EvaluationTestChange[]
  test_changes: EvaluationTestChange[]
}
