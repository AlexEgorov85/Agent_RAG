import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  evaluationApi,
  type CatalogParams,
  type RunListParams,
  type RunTestsParams,
} from './evaluation-api'
import type {
  EvaluationRunCreateRequest,
  RetrievalMode,
} from '../../types/evaluation'
import { isActiveStatus } from './format'

const ROOT = ['evaluation'] as const

export const evaluationKeys = {
  suites: () => [...ROOT, 'suites'] as const,
  runs: (params: RunListParams) => [...ROOT, 'runs', params] as const,
  run: (runId: string) => [...ROOT, 'run', runId] as const,
  runTests: (runId: string, params: RunTestsParams) =>
    [...ROOT, 'run-tests', runId, params] as const,
  runTest: (runId: string, testId: string, mode: RetrievalMode, attempt: number) =>
    [...ROOT, 'run-test', runId, testId, mode, attempt] as const,
  trace: (runId: string, testId: string, mode: RetrievalMode, attempt: number) =>
    [...ROOT, 'trace', runId, testId, mode, attempt] as const,
  catalog: (params: CatalogParams) => [...ROOT, 'catalog', params] as const,
  test: (testId: string) => [...ROOT, 'test', testId] as const,
  compare: (base: string, candidate: string) =>
    [...ROOT, 'compare', base, candidate] as const,
}

export function useEvaluationSuites(enabled = true) {
  return useQuery({
    queryKey: evaluationKeys.suites(),
    queryFn: () => evaluationApi.suites(),
    enabled,
    staleTime: 60_000,
  })
}

/** История запусков; активный запуск обновляется polling'ом (§5.4). */
export function useEvaluationRuns(params: RunListParams = {}, enabled = true) {
  return useQuery({
    queryKey: evaluationKeys.runs(params),
    queryFn: () => evaluationApi.listRuns(params),
    enabled,
    refetchInterval: (query) =>
      query.state.data?.items.some((item) => isActiveStatus(item.status)) ? 2000 : false,
  })
}

export function useEvaluationRun(runId: string | null, enabled = true) {
  return useQuery({
    queryKey: evaluationKeys.run(runId ?? ''),
    queryFn: () => evaluationApi.getRun(runId as string),
    enabled: enabled && Boolean(runId),
    refetchInterval: (query) =>
      query.state.data && isActiveStatus(query.state.data.status) ? 2000 : false,
  })
}

/** Детали нескольких запусков — для динамики и дельт обзора. */
export function useEvaluationRunDetails(runIds: string[]) {
  return useQueries({
    queries: runIds.map((runId) => ({
      queryKey: evaluationKeys.run(runId),
      queryFn: () => evaluationApi.getRun(runId),
      staleTime: 30_000,
    })),
  })
}

export function useEvaluationRunTests(
  runId: string | null,
  params: RunTestsParams = {},
  enabled = true,
) {
  return useQuery({
    queryKey: evaluationKeys.runTests(runId ?? '', params),
    queryFn: () => evaluationApi.listRunTests(runId as string, params),
    enabled: enabled && Boolean(runId),
  })
}

export function useEvaluationRunTest(
  runId: string | null,
  testId: string | null,
  mode: RetrievalMode | null,
  attempt: number,
  enabled = true,
) {
  return useQuery({
    queryKey: evaluationKeys.runTest(runId ?? '', testId ?? '', mode ?? 'hybrid', attempt),
    queryFn: () =>
      evaluationApi.getRunTest(runId as string, testId as string, {
        retrieval_mode: mode as RetrievalMode,
        attempt_index: attempt,
      }),
    enabled: enabled && Boolean(runId) && Boolean(testId) && Boolean(mode),
  })
}

export function useEvaluationTrace(
  runId: string | null,
  testId: string | null,
  mode: RetrievalMode | null,
  attempt: number,
  enabled: boolean,
) {
  return useQuery({
    queryKey: evaluationKeys.trace(runId ?? '', testId ?? '', mode ?? 'hybrid', attempt),
    queryFn: () =>
      evaluationApi.getRunTestTrace(runId as string, testId as string, {
        retrieval_mode: mode as RetrievalMode,
        attempt_index: attempt,
      }),
    enabled: enabled && Boolean(runId) && Boolean(testId) && Boolean(mode),
    retry: false,
  })
}

export function useEvaluationCatalog(params: CatalogParams = {}, enabled = true) {
  return useQuery({
    queryKey: evaluationKeys.catalog(params),
    queryFn: () => evaluationApi.listTests(params),
    enabled,
  })
}

export function useEvaluationTest(testId: string | null, enabled = true) {
  return useQuery({
    queryKey: evaluationKeys.test(testId ?? ''),
    queryFn: () => evaluationApi.getTest(testId as string),
    enabled: enabled && Boolean(testId),
  })
}

export function useEvaluationCompare(base: string | null, candidate: string | null) {
  return useQuery({
    queryKey: evaluationKeys.compare(base ?? '', candidate ?? ''),
    queryFn: () => evaluationApi.compare(base as string, candidate as string),
    enabled: Boolean(base) && Boolean(candidate) && base !== candidate,
  })
}

export function useCreateEvaluationRun() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: EvaluationRunCreateRequest) => evaluationApi.createRun(payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: [...ROOT, 'runs'] })
    },
  })
}

export function useCancelEvaluationRun() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) => evaluationApi.cancelRun(runId),
    onSuccess: (data) => {
      void client.invalidateQueries({ queryKey: [...ROOT, 'runs'] })
      void client.invalidateQueries({ queryKey: evaluationKeys.run(data.id) })
    },
  })
}
