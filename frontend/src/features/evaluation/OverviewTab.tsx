import { useMemo, useState } from 'react'
import { AlertTriangle, ArrowRight, Play } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { EmptyState } from '../../components/common/EmptyState'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import type {
  EvaluationRun,
  EvaluationRunListItem,
  EvaluationSummary,
} from '../../types/evaluation'
import {
  EMPTY_VALUE,
  formatDeltaPoints,
  formatDeltaValue,
  formatPercent,
  formatRatio,
  isNumber,
  problemRows,
  suiteLabel,
} from './format'
import { ActiveRunNotice, MetricCard, ProgressBar, RunStatusBadge, SectionCard } from './components'
import { useEvaluationCompare, useEvaluationRunDetails } from './use-evaluation'

type TrendMetric = 'pass_rate' | 'recall_at_10' | 'answer' | 'multi_turn'

const TREND_OPTIONS: Array<{ key: TrendMetric; label: string; kind: 'percent' | 'ratio' }> = [
  { key: 'pass_rate', label: 'Pass rate', kind: 'percent' },
  { key: 'recall_at_10', label: 'Recall@10', kind: 'ratio' },
  { key: 'answer', label: 'Answer quality', kind: 'percent' },
  { key: 'multi_turn', label: 'Multi-turn', kind: 'percent' },
]

function trendValue(summary: EvaluationSummary | null, metric: TrendMetric): number | null {
  if (!summary) return null
  if (metric === 'pass_rate') return isNumber(summary.pass_rate) ? summary.pass_rate : null
  if (metric === 'recall_at_10') {
    const value = summary.groups?.retrieval?.recall_at_10
    return isNumber(value) ? value : null
  }
  if (metric === 'answer') {
    const automatic = summary.groups?.answer?.automatic_pass_rate
    if (isNumber(automatic)) return automatic
    const rubric = summary.groups?.answer?.rubric_mean
    return isNumber(rubric) ? rubric / 10 : null
  }
  const dialogue = summary.groups?.dialogue?.scenario_success_rate
  return isNumber(dialogue) ? dialogue : null
}

function isComparable(a: EvaluationRunListItem, b: EvaluationRunListItem): boolean {
  return (
    a.suite === b.suite &&
    a.split === b.split &&
    (a.dataset_sha256 ?? '') === (b.dataset_sha256 ?? '') &&
    (a.corpus_content_sha256 ?? '') === (b.corpus_content_sha256 ?? '')
  )
}

function TrendChart({
  points,
  metric,
}: {
  points: Array<{ id: string; label: string; value: number | null; meta: string }>
  metric: TrendMetric
}) {
  const kind = TREND_OPTIONS.find((item) => item.key === metric)?.kind ?? 'percent'
  const measured = points.filter((point) => isNumber(point.value))
  if (measured.length < 2) {
    return (
      <EmptyState
        title="Недостаточно данных для динамики"
        description="Нужны минимум два завершённых запуска с измеренной метрикой. Пропуски не изображаются нулями."
      />
    )
  }
  const values = measured.map((point) => point.value as number)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || Math.abs(max) || 1
  const width = 640
  const height = 150
  const padding = 24
  const scaleX = (index: number) =>
    points.length <= 1
      ? padding
      : padding + (index * (width - padding * 2)) / (points.length - 1)
  const scaleY = (value: number) =>
    height - padding - ((value - min + span * 0.15) / (span * 1.3)) * (height - padding * 2)

  const segments: Array<Array<{ x: number; y: number; point: (typeof points)[number] }>> = []
  let current: Array<{ x: number; y: number; point: (typeof points)[number] }> = []
  points.forEach((point, index) => {
    if (!isNumber(point.value)) {
      if (current.length > 0) segments.push(current)
      current = []
      return
    }
    current.push({ x: scaleX(index), y: scaleY(point.value), point })
  })
  if (current.length > 0) segments.push(current)

  const format = (value: number) =>
    kind === 'percent' ? formatPercent(value) : formatRatio(value, 2)

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Динамика качества по завершённым запускам"
        className="h-[150px] w-full min-w-[420px]"
      >
        {segments.map((segment, index) =>
          segment.length > 1 ? (
            <polyline
              key={`segment-${index}`}
              points={segment.map((item) => `${item.x},${item.y}`).join(' ')}
              fill="none"
              className="stroke-accent"
              strokeWidth={2}
            />
          ) : null,
        )}
        {segments.flat().map((item) => (
          <circle key={item.point.id} cx={item.x} cy={item.y} r={3.5} className="fill-accent">
            <title>
              {`${item.point.label} · ${format(item.point.value as number)} · ${item.point.meta}`}
            </title>
          </circle>
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span>Минимум: {format(min)}</span>
        <span>Максимум: {format(max)}</span>
        <span>Точки только реальных запусков; пропуск не соединён прямой.</span>
      </div>
    </div>
  )
}

export interface OverviewTabProps {
  runs: EvaluationRunListItem[]
  isLoading: boolean
  error: unknown
  onRetry: () => void
  onStartSmoke: () => void
  onOpenRuns: () => void
  onOpenCompare: (baseId?: string, candidateId?: string) => void
  onOpenTests: () => void
}

export function OverviewTab({
  runs,
  isLoading,
  error,
  onRetry,
  onStartSmoke,
  onOpenRuns,
  onOpenCompare,
  onOpenTests,
}: OverviewTabProps) {
  const [trendMetric, setTrendMetric] = useState<TrendMetric>('pass_rate')

  const completed = useMemo(
    () => runs.filter((run) => run.status === 'completed' && run.has_summary),
    [runs],
  )
  const active = useMemo(() => runs.find((run) => run.status !== 'completed' && run.status !== 'failed' && run.status !== 'cancelled' && run.status !== 'interrupted'), [runs])

  const last = completed[0]
  const previous = useMemo(() => {
    if (!last) return undefined
    return completed.slice(1).find((run) => isComparable(run, last))
  }, [completed, last])

  const trendIds = useMemo(() => completed.slice(0, 8).map((run) => run.id), [completed])
  const details = useEvaluationRunDetails(trendIds)
  const detailById = useMemo(() => {
    const map = new Map<string, EvaluationRun>()
    details.forEach((query) => {
      if (query.data) map.set(query.data.id, query.data)
    })
    return map
  }, [details])

  const lastDetail = last ? detailById.get(last.id) : undefined
  const previousDetail = previous ? detailById.get(previous.id) : undefined
  const compareQuery = useEvaluationCompare(previous?.id ?? null, last?.id ?? null)
  const compare = compareQuery.data
  const deltaFor = (key: string): number | null => {
    if (!compare || !compare.comparable) return null
    const metric = compare.metrics.find((item) => item.key === key)
    return metric && isNumber(metric.delta) ? metric.delta : null
  }
  const deltaNote = !previous
    ? 'Нет предыдущего сопоставимого запуска'
    : compare && !compare.comparable
      ? 'Несопоставимо: смотрите side-by-side'
      : undefined

  const summary = lastDetail?.summary ?? null
  const coverage = summary?.coverage
  const passRate = isNumber(summary?.pass_rate) ? summary?.pass_rate : null
  const eligible = coverage?.tests_eligible
  const executed = coverage?.executed
  const selected = coverage?.tests_selected
  const passed = coverage?.passed
  const activeRunning = Boolean(active)

  if (isLoading && runs.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center">
        <Spinner label="Загрузка запусков" />
      </div>
    )
  }
  if (error) {
    return <ErrorState error={error} title="История запусков недоступна" onRetry={onRetry} />
  }
  if (runs.length === 0) {
    return (
      <EmptyState
        title="Запусков ещё не было"
        description="Evaluation показывает только реальные прогоны benchmark. Запустите Smoke, чтобы получить первые результаты."
        action={
          <button type="button" className="btn btn-primary" onClick={onStartSmoke}>
            <Play className="size-4" aria-hidden="true" /> Запустить Smoke
          </button>
        }
      />
    )
  }
  if (!last) {
    return (
      <div className="space-y-3">
        <Banner tone="warn" title="Нет завершённых запусков">
          Ни один запуск ещё не завершился успешно. Откройте «Запуски», чтобы посмотреть статус или
          повторить запуск.
        </Banner>
        <button type="button" className="btn" onClick={onOpenRuns}>
          Открыть запуски <ArrowRight className="size-4" aria-hidden="true" />
        </button>
      </div>
    )
  }

  const valueOrDash = (value: string) => (activeRunning ? EMPTY_VALUE : value)
  const activeContext = activeRunning ? 'Текущий запуск выполняется — значения не подменяются старыми' : undefined

  const problems = problemRows(summary?.critical_errors)
  const problemTests = summary?.problem_tests ?? []
  const trendPoints = trendIds
    .map((id) => {
      const detail = detailById.get(id)
      const item = completed.find((run) => run.id === id)
      if (!item) return null
      return {
        id,
        label: item.created_at,
        value: trendValue(detail?.summary ?? null, trendMetric),
        meta: `${suiteLabel(item.suite)} · ${item.split} · ${item.retrieval_mode} · ${
          item.git_sha?.slice(0, 7) ?? 'no-git'
        }`,
      }
    })
    .filter((item): item is NonNullable<typeof item> => item !== null)
    .reverse()

  return (
    <div className="space-y-4">
      {activeRunning && active ? (
        <SectionCard
          title="Текущий запуск"
          description={`${suiteLabel(active.suite)} · ${active.split} · ${active.retrieval_mode}`}
          actions={<RunStatusBadge status={active.status} />}
        >
          <div className="space-y-2">
            <ActiveRunNotice status={active.status} />
            <ProgressBar
              percent={active.progress?.percent ?? 0}
              completed={active.progress?.tests_completed ?? 0}
              total={active.progress?.tests_total ?? active.tests_total}
              currentTestId={active.progress?.current_test_id}
            />
            <button type="button" className="btn btn-ghost px-2 py-1 text-xs" onClick={onOpenRuns}>
              Открыть запуски
            </button>
          </div>
        </SectionCard>
      ) : null}

      {!activeRunning && runs[0] && runs[0].status === 'failed' ? (
        <Banner
          tone="error"
          title="Последний запуск завершился ошибкой"
          action={
            <button type="button" className="btn btn-ghost px-2 py-1 text-xs" onClick={onOpenRuns}>
              Подробности
            </button>
          }
        >
          Запуск {runs[0].id} не завершён. Ниже — значения последнего успешного запуска{' '}
          {last.id}; ошибка не маскируется процентом.
        </Banner>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Успешные тесты"
          value={valueOrDash(passRate === null ? EMPTY_VALUE : formatPercent(passRate))}
          context={
            activeContext ??
            (isNumber(passed) && isNumber(eligible)
              ? `${passed}/${eligible} eligible · ${executed ?? EMPTY_VALUE}/${selected ?? EMPTY_VALUE} executed · coverage ${formatPercent(
                  summary?.execution_coverage,
                )}`
              : 'coverage не указан')
          }
          delta={formatDeltaPoints(deltaFor('pass_rate'))}
          deltaNote={deltaNote}
        />
        <MetricCard
          label="Recall@10"
          value={valueOrDash(formatRatio(summary?.groups?.retrieval?.recall_at_10, 2))}
          context={activeContext ?? 'macro mean по retrieval-тестам'}
          delta={formatDeltaValue(deltaFor('recall_at_10'), 2)}
          deltaNote={deltaNote}
        />
        <MetricCard
          label="Качество ответов"
          value={valueOrDash(
            isNumber(summary?.groups?.answer?.rubric_mean)
              ? `${formatRatio(summary?.groups?.answer?.rubric_mean, 1)} / 10`
              : formatPercent(summary?.groups?.answer?.automatic_pass_rate),
          )}
          context={
            activeContext ??
            (isNumber(summary?.groups?.answer?.rubric_mean)
              ? 'rubric mean (judge/manual)'
              : 'automatic pass rate — rubric не измерялся')
          }
          delta={
            isNumber(summary?.groups?.answer?.rubric_mean)
              ? formatDeltaValue(deltaFor('rubric_mean'), 1)
              : formatDeltaPoints(deltaFor('automatic_pass_rate'))
          }
          deltaNote={deltaNote}
        />
        <MetricCard
          label="Multi-turn"
          value={valueOrDash(formatPercent(summary?.groups?.dialogue?.scenario_success_rate))}
          context={activeContext ?? 'scenario success rate'}
          delta={formatDeltaPoints(deltaFor('dialogue_success_rate'))}
          deltaNote={deltaNote}
        />
      </div>

      <SectionCard
        title="Качество по направлениям"
        description="Четыре независимые полосы; неизмеренная метрика — «—», не 0%."
      >
        <div className="space-y-3">
          {[
            {
              label: 'Retrieval',
              value: summary?.groups?.retrieval?.recall_at_10,
              hint: `Evidence coverage: ${formatRatio(summary?.groups?.retrieval?.evidence_coverage, 2)}`,
            },
            {
              label: 'Answer',
              value: summary?.groups?.answer?.automatic_pass_rate,
              hint: isNumber(summary?.groups?.answer?.rubric_mean)
                ? `Rubric: ${formatRatio(summary?.groups?.answer?.rubric_mean, 1)} / 10`
                : 'Rubric не измерялся',
            },
            {
              label: 'Analysis',
              value: summary?.groups?.reasoning?.pass_rate,
              hint: 'medium/hard reasoning',
            },
            {
              label: 'Dialogue',
              value: summary?.groups?.dialogue?.scenario_success_rate,
              hint: 'multi-turn success',
            },
          ].map((direction) => (
            <div key={direction.label} className="space-y-1">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-medium text-ink">{direction.label}</span>
                <span className="tabular-nums text-muted">
                  {activeRunning ? EMPTY_VALUE : formatPercent(direction.value)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-raised">
                <div
                  className="h-full rounded-full bg-accent/80"
                  style={{
                    width: `${isNumber(direction.value) && !activeRunning ? Math.max(0, Math.min(100, direction.value * 100)) : 0}%`,
                  }}
                />
              </div>
              <p className="text-[11px] text-muted">{direction.hint}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard
        title="Динамика качества"
        description="Только реальные завершённые запуски; пропуск не изображается нулём."
        actions={
          <label className="flex items-center gap-1 text-xs text-muted">
            Метрика
            <select
              className="field w-auto py-1 text-xs"
              value={trendMetric}
              onChange={(event) => setTrendMetric(event.target.value as TrendMetric)}
              aria-label="Метрика динамики"
            >
              {TREND_OPTIONS.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        }
      >
        <TrendChart points={trendPoints} metric={trendMetric} />
      </SectionCard>

      <div className="grid gap-4 lg:grid-cols-2">
        <SectionCard
          title="Последние запуски"
          actions={
            <button type="button" className="btn btn-ghost px-2 py-1 text-xs" onClick={onOpenRuns}>
              Все запуски
            </button>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-muted">
                <tr className="border-b border-line">
                  <th className="py-1.5 pr-2 font-medium">Дата</th>
                  <th className="py-1.5 pr-2 font-medium">Набор</th>
                  <th className="py-1.5 pr-2 font-medium">Режим</th>
                  <th className="py-1.5 pr-2 font-medium">Pass rate</th>
                  <th className="py-1.5 pr-2 font-medium">Coverage</th>
                  <th className="py-1.5 font-medium">Статус</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 5).map((run) => (
                  <tr key={run.id} className="border-b border-line/60 last:border-0">
                    <td className="py-2 pr-2 whitespace-nowrap">
                      {new Date(run.created_at).toLocaleDateString('ru-RU')}
                    </td>
                    <td className="py-2 pr-2">{suiteLabel(run.suite)}</td>
                    <td className="py-2 pr-2 font-mono">{run.retrieval_mode}</td>
                    <td className="py-2 pr-2 tabular-nums">
                      {run.status === 'completed' ? (
                        <>
                          {formatPercent(run.pass_rate)}{' '}
                          <span className="text-muted">
                            ({run.tests_passed}/{run.tests_eligible})
                          </span>
                        </>
                      ) : (
                        <span className="text-muted">progress {run.progress?.percent ?? 0}%</span>
                      )}
                    </td>
                    <td className="py-2 pr-2 tabular-nums">
                      {run.tests_executed}/{run.tests_total}
                    </td>
                    <td className="py-2">
                      <RunStatusBadge status={run.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>

        <SectionCard
          title="Обнаруженные проблемы"
          description="Счётчики из summary: UI не классифицирует ошибки самостоятельно."
          actions={
            <button type="button" className="btn btn-ghost px-2 py-1 text-xs" onClick={onOpenTests}>
              Открыть тесты
            </button>
          }
        >
          {problems.length === 0 && problemTests.length === 0 ? (
            <p className="text-xs text-muted">Critical errors в последнем запуске не зафиксированы.</p>
          ) : (
            <div className="space-y-3">
              {problems.length > 0 ? (
                <table className="w-full text-left text-xs">
                  <thead className="text-muted">
                    <tr className="border-b border-line">
                      <th className="py-1.5 pr-2 font-medium">Проблема</th>
                      <th className="py-1.5 pr-2 font-medium">Кол-во</th>
                      <th className="py-1.5 font-medium">Тесты</th>
                    </tr>
                  </thead>
                  <tbody>
                    {problems.map((problem) => (
                      <tr key={problem.code} className="border-b border-line/60 last:border-0">
                        <td className="py-2 pr-2">{problem.label}</td>
                        <td className="py-2 pr-2 tabular-nums">{problem.count}</td>
                        <td className="py-2 font-mono text-[11px]">
                          {problem.testIds.slice(0, 4).join(', ') || EMPTY_VALUE}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
              {problemTests.length > 0 ? (
                <div className="flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/10 px-2 py-1.5 text-xs">
                  <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warn" aria-hidden="true" />
                  <div>
                    <p className="font-medium text-ink">Problem tests</p>
                    <ul className="mt-1 space-y-0.5 text-muted">
                      {problemTests.slice(0, 5).map((item) => (
                        <li key={item.id}>
                          <span className="font-mono">{item.id}</span> —{' '}
                          {item.reasons.join(', ')}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        <span>
          Последний завершённый запуск: <span className="font-mono">{last.id}</span>
          {previousDetail ? ` · предыдущий: ${previousDetail.id}` : ''}
        </span>
        <button
          type="button"
          className="btn btn-ghost px-2 py-1 text-xs"
          onClick={() => onOpenCompare(previous?.id, last.id)}
          disabled={!previous}
          title={previous ? 'Открыть side-by-side' : 'Нет предыдущего сопоставимого запуска'}
        >
          Сравнить с предыдущим
        </button>
      </div>
    </div>
  )
}
