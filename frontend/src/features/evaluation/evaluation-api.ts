import { apiRequest, apiRequestRaw } from '../../services/api-client'
import type {
  EvaluationCompareResponse,
  EvaluationRun,
  EvaluationRunCreateRequest,
  EvaluationRunCreated,
  EvaluationRunListResponse,
  EvaluationRunTestsResponse,
  EvaluationSuitesResponse,
  EvaluationTestCatalogResponse,
  EvaluationTestDefinition,
  EvaluationTestDetail,
  EvaluationTrace,
  RetrievalMode,
} from '../../types/evaluation'

export interface RunListParams {
  suite?: string
  status?: string
  split?: string
  retrieval_mode?: string
  limit?: number
  offset?: number
}

export interface RunTestsParams {
  status?: string
  critical?: boolean
  difficulty?: string
  category?: string
  retrieval_mode?: string
  limit?: number
  offset?: number
}

export interface CatalogParams {
  difficulty?: string
  category?: string
  suite?: string
  status?: string
  critical?: boolean
  q?: string
  retrieval_mode?: RetrievalMode
  run_id?: string
  limit?: number
  offset?: number
}

const base = '/api/evaluation'

/** Query-строка для опциональных параметров Evaluation API. */
function query(params: object): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

function runPath(runId: string): string {
  return `${base}/runs/${encodeURIComponent(runId)}`
}

/** Evaluation API: только чтение артефактов и управление очередью (§8). */
export const evaluationApi = {
  suites(): Promise<EvaluationSuitesResponse> {
    return apiRequest<EvaluationSuitesResponse>(`${base}/suites`)
  },

  createRun(payload: EvaluationRunCreateRequest): Promise<EvaluationRunCreated> {
    return apiRequest<EvaluationRunCreated>(`${base}/runs`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },

  listRuns(params: RunListParams = {}): Promise<EvaluationRunListResponse> {
    return apiRequest<EvaluationRunListResponse>(`${base}/runs${query(params)}`)
  },

  getRun(runId: string): Promise<EvaluationRun> {
    return apiRequest<EvaluationRun>(runPath(runId))
  },

  listRunTests(runId: string, params: RunTestsParams = {}): Promise<EvaluationRunTestsResponse> {
    return apiRequest<EvaluationRunTestsResponse>(
      `${runPath(runId)}/tests${query(params)}`,
    )
  },

  getRunTest(
    runId: string,
    testId: string,
    params: { retrieval_mode: RetrievalMode; attempt_index?: number },
  ): Promise<EvaluationTestDetail> {
    return apiRequest<EvaluationTestDetail>(
      `${runPath(runId)}/tests/${encodeURIComponent(testId)}${query(params)}`,
    )
  },

  getRunTestTrace(
    runId: string,
    testId: string,
    params: { retrieval_mode: RetrievalMode; attempt_index?: number },
  ): Promise<EvaluationTrace> {
    return apiRequest<EvaluationTrace>(
      `${runPath(runId)}/tests/${encodeURIComponent(testId)}/trace${query(params)}`,
    )
  },

  async getReport(runId: string): Promise<string> {
    const response = await apiRequestRaw(`${runPath(runId)}/report`, {
      headers: { Accept: 'text/markdown, text/html' },
    })
    if (!response.ok) {
      const { ApiError } = await import('../../services/api-client')
      throw await ApiError.fromResponse(response)
    }
    return response.text()
  },

  cancelRun(runId: string): Promise<{ id: string; status: string; cancel_supported: boolean }> {
    return apiRequest(`${runPath(runId)}/cancel`, { method: 'POST' })
  },

  listTests(params: CatalogParams = {}): Promise<EvaluationTestCatalogResponse> {
    return apiRequest<EvaluationTestCatalogResponse>(`${base}/tests${query(params)}`)
  },

  getTest(testId: string): Promise<EvaluationTestDefinition> {
    return apiRequest<EvaluationTestDefinition>(`${base}/tests/${encodeURIComponent(testId)}`)
  },

  compare(baseRunId: string, candidateRunId: string): Promise<EvaluationCompareResponse> {
    return apiRequest<EvaluationCompareResponse>(
      `${base}/compare${query({ base: baseRunId, candidate: candidateRunId })}`,
    )
  },
}
