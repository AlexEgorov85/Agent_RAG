import { useEffect, useMemo, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { EmptyState } from '../../components/common/EmptyState'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import { describeApiError } from '../../lib/errors'
import type {
  EvaluationCapabilities,
  EvaluationRunListItem,
} from '../../types/evaluation'
import {
  EMPTY_VALUE,
  formatDuration,
  formatPercent,
  isActiveStatus,
  suiteLabel,
} from './format'
import { RunForm } from './RunForm'
import { RunCard } from './RunCard'
import {
  useCancelEvaluationRun,
  useCreateEvaluationRun,
  useEvaluationRun,
  useEvaluationRuns,
} from './use-evaluation'

export interface RunsTabProps {
  capabilities: EvaluationCapabilities
  onOpenTests: () => void
  onOpenCompare: (baseId: string, candidateId: string) => void
  focusRunId?: string | null
  onFocusHandled?: () => void
}

export function RunsTab({
  capabilities,
  onOpenTests,
  onOpenCompare,
  focusRunId,
  onFocusHandled,
}: RunsTabProps) {
  const [suite, setSuite] = useState('')
  const [status, setStatus] = useState('')
  const [split, setSplit] = useState('')
  const [mode, setMode] = useState('')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(focusRunId ?? null)

  const params = useMemo(
    () => ({
      suite: suite || undefined,
      status: status || undefined,
      split: split || undefined,
      retrieval_mode: mode || undefined,
      limit: 50,
    }),
    [suite, status, split, mode],
  )
  const runs = useEvaluationRuns(params)
  const detail = useEvaluationRun(selectedRunId)
  const createRun = useCreateEvaluationRun()
  const cancelRun = useCancelEvaluationRun()

  const items = runs.data?.items ?? []
  const activeRun = items.find((item) => isActiveStatus(item.status)) ?? null

  useEffect(() => {
    if (focusRunId) {
      setSelectedRunId(focusRunId)
      onFocusHandled?.()
    }
  }, [focusRunId, onFocusHandled])

  useEffect(() => {
    if (selectedRunId && items.some((item) => item.id === selectedRunId)) return
    const first = items[0]
    if (!selectedRunId && first) setSelectedRunId(first.id)
  }, [items, selectedRunId])

  const previousComparable = (run: EvaluationRunListItem) =>
    items.find(
      (item) =>
        item.id !== run.id &&
        item.status === 'completed' &&
        item.suite === run.suite &&
        item.split === run.split &&
        (item.dataset_sha256 ?? '') === (run.dataset_sha256 ?? ''),
    )

  return (
    <div className="space-y-4">
      <RunForm
        capabilities={capabilities}
        activeRun={activeRun}
        pending={createRun.isPending}
        error={createRun.error}
        onCreate={createRun.mutateAsync}
        onCreated={(created) => {
          setSuite('')
          setStatus('')
          setSplit('')
          setMode('')
          setSelectedRunId(created.id)
        }}
      />

      <section className="rounded-xl border border-line bg-surface p-4 shadow-sm">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-ink">История запусков</h2>
            <p className="text-xs text-muted">
              {runs.data ? `${runs.data.total} запусков` : 'Загрузка…'} · сортировка по дате
              создания
            </p>
          </div>
          <button
            type="button"
            className="btn btn-ghost px-2 py-1 text-xs"
            onClick={() => void runs.refetch()}
            disabled={runs.isFetching}
          >
            <RefreshCw className={`size-3.5 ${runs.isFetching ? 'animate-spin' : ''}`} aria-hidden="true" />
            Обновить
          </button>
        </header>

        <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1 text-xs text-muted">
            Suite
            <select className="field" value={suite} onChange={(event) => setSuite(event.target.value)} aria-label="Фильтр suite">
              <option value="">Все</option>
              {['smoke', 'full', 'easy', 'medium', 'hard', 'multi_turn'].map((item) => (
                <option key={item} value={item}>
                  {suiteLabel(item)}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Статус
            <select className="field" value={status} onChange={(event) => setStatus(event.target.value)} aria-label="Фильтр статуса">
              <option value="">Все</option>
              {['queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled', 'interrupted'].map(
                (item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ),
              )}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Split
            <select className="field" value={split} onChange={(event) => setSplit(event.target.value)} aria-label="Фильтр split">
              <option value="">Все</option>
              {['dev', 'holdout', 'all'].map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Режим
            <select className="field" value={mode} onChange={(event) => setMode(event.target.value)} aria-label="Фильтр режима">
              <option value="">Все</option>
              {['hybrid', 'dense', 'keyword', 'all'].map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="mt-3">
          {runs.isLoading ? (
            <div className="flex h-24 items-center justify-center">
              <Spinner label="Загрузка запусков" />
            </div>
          ) : runs.error ? (
            <ErrorState error={runs.error} title="История запусков недоступна" onRetry={() => void runs.refetch()} />
          ) : items.length === 0 ? (
            <EmptyState
              title="Запусков не найдено"
              description="Измените фильтры или создайте новый запуск выше. Значения не подставляются."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] text-left text-xs">
                <thead className="text-muted">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-2 font-medium">Дата</th>
                    <th className="py-2 pr-2 font-medium">Suite</th>
                    <th className="py-2 pr-2 font-medium">Split</th>
                    <th className="py-2 pr-2 font-medium">Mode</th>
                    <th className="py-2 pr-2 font-medium">Pass rate</th>
                    <th className="py-2 pr-2 font-medium">Coverage</th>
                    <th className="py-2 pr-2 font-medium">Длительность</th>
                    <th className="py-2 pr-2 font-medium">Git</th>
                    <th className="py-2 font-medium">Действия</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((run) => (
                    <tr
                      key={run.id}
                      className={`border-b border-line/60 last:border-0 ${
                        selectedRunId === run.id ? 'bg-raised/70' : ''
                      }`}
                    >
                      <td className="py-2 pr-2 whitespace-nowrap">
                        <button
                          type="button"
                          className="text-left font-mono text-[11px] underline decoration-line underline-offset-2 hover:decoration-accent"
                          onClick={() => setSelectedRunId(run.id)}
                        >
                          {new Date(run.created_at).toLocaleString('ru-RU')}
                        </button>
                      </td>
                      <td className="py-2 pr-2">{suiteLabel(run.suite)}</td>
                      <td className="py-2 pr-2">{run.split}</td>
                      <td className="py-2 pr-2 font-mono">{run.retrieval_mode}</td>
                      <td className="py-2 pr-2 tabular-nums">
                        {run.status === 'completed' ? (
                          <>
                            {formatPercent(run.pass_rate)}{' '}
                            <span className="text-muted">
                              ({run.tests_passed}/{run.tests_eligible})
                            </span>
                          </>
                        ) : isActiveStatus(run.status) ? (
                          `${run.progress?.percent ?? 0}%`
                        ) : (
                          <span className="text-muted">{EMPTY_VALUE}</span>
                        )}
                      </td>
                      <td className="py-2 pr-2 tabular-nums">
                        {run.tests_executed}/{run.tests_total}
                      </td>
                      <td className="py-2 pr-2">{formatDuration(run.duration_ms)}</td>
                      <td className="py-2 pr-2 font-mono text-[11px]">
                        {run.git_sha?.slice(0, 7) ?? EMPTY_VALUE}
                        {run.git_dirty ? ' *' : ''}
                      </td>
                      <td className="py-2">
                        <div className="flex flex-wrap gap-1">
                          <button
                            type="button"
                            className="btn btn-ghost px-2 py-1 text-[11px]"
                            onClick={() => setSelectedRunId(run.id)}
                          >
                            Карточка
                          </button>
                          {run.status === 'completed' ? (
                            <button
                              type="button"
                              className="btn btn-ghost px-2 py-1 text-[11px]"
                              onClick={() => {
                                const previous = previousComparable(run)
                                if (previous) onOpenCompare(previous.id, run.id)
                              }}
                              disabled={!previousComparable(run)}
                              title={
                                previousComparable(run)
                                  ? 'Сравнить с предыдущим сопоставимым запуском'
                                  : 'Нет предыдущего сопоставимого запуска'
                              }
                            >
                              Сравнить
                            </button>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {activeRun ? (
            <Banner tone="info" title="Benchmark выполняется">
              {activeRun.id}: {activeRun.progress?.tests_completed ?? 0}/
              {activeRun.progress?.tests_total ?? activeRun.tests_total} пар. Запуски доступны, но
              второй параллельный benchmark не создаётся.
            </Banner>
          ) : null}
        </div>
      </section>

      {detail.isLoading ? (
        <div className="flex h-32 items-center justify-center">
          <Spinner label="Загрузка карточки запуска" />
        </div>
      ) : detail.error ? (
        <ErrorState
          error={detail.error}
          title="Карточка запуска недоступна"
          onRetry={() => void detail.refetch()}
        />
      ) : detail.data ? (
        <RunCard
          run={detail.data}
          canCancel={Boolean(capabilities.evaluation_cancel)}
          cancelPending={cancelRun.isPending}
          onCancel={() => {
            if (!selectedRunId) return
            cancelRun.mutate(selectedRunId)
          }}
          onOpenTests={onOpenTests}
          onCompare={() => {
            const item = items.find((candidate) => candidate.id === detail.data?.id)
            if (!item) return
            const previous = previousComparable(item)
            if (previous) onOpenCompare(previous.id, item.id)
          }}
        />
      ) : null}

      {cancelRun.error ? (
        <Banner tone="error" title="Отмена не выполнена">{describeApiError(cancelRun.error)}</Banner>
      ) : null}
    </div>
  )
}
