import { useEffect, useMemo, useState } from 'react'
import { ArrowLeftRight } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { EmptyState } from '../../components/common/EmptyState'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import { ToneBadge } from '../../components/common/ToneBadge'
import type {
  EvaluationRunListItem,
  EvaluationTestChange,
} from '../../types/evaluation'
import {
  COMPARE_REASON_LABEL,
  EMPTY_VALUE,
  VERSION_DIFFERENCE_LABEL,
  formatDeltaMs,
  formatDeltaPoints,
  formatDeltaValue,
  formatMs,
  formatPercent,
  formatRatio,
  isNumber,
  suiteLabel,
} from './format'
import { SectionCard } from './components'
import { useEvaluationCompare } from './use-evaluation'

type MetricKind = 'percent' | 'ratio' | 'ms'

const METRIC_META: Record<string, { label: string; kind: MetricKind }> = {
  pass_rate: { label: 'Pass rate', kind: 'percent' },
  execution_coverage: { label: 'Execution coverage', kind: 'percent' },
  pass_at_1: { label: 'Pass@1', kind: 'percent' },
  recall_at_5: { label: 'Recall@5', kind: 'ratio' },
  recall_at_10: { label: 'Recall@10', kind: 'ratio' },
  mrr: { label: 'MRR', kind: 'ratio' },
  evidence_coverage: { label: 'Evidence Coverage', kind: 'ratio' },
  context_coverage: { label: 'Context Coverage', kind: 'ratio' },
  automatic_pass_rate: { label: 'Answer automatic pass', kind: 'percent' },
  rubric_mean: { label: 'Rubric mean', kind: 'ratio' },
  fact_accuracy: { label: 'Fact accuracy', kind: 'ratio' },
  citation_validity: { label: 'Citation validity', kind: 'ratio' },
  citation_coverage: { label: 'Citation coverage', kind: 'ratio' },
  dialogue_success_rate: { label: 'Multi-turn', kind: 'percent' },
  mean_latency_ms: { label: 'Mean latency', kind: 'ms' },
}

function formatValue(value: number | null, kind: MetricKind): string {
  if (!isNumber(value)) return EMPTY_VALUE
  if (kind === 'percent') return formatPercent(value)
  if (kind === 'ms') return formatMs(value)
  return formatRatio(value, 2)
}

function formatDelta(delta: number | null, kind: MetricKind): string {
  if (!isNumber(delta)) return EMPTY_VALUE
  if (kind === 'percent') return formatDeltaPoints(delta)
  if (kind === 'ms') return formatDeltaMs(delta)
  return formatDeltaValue(delta, 2)
}

function deltaClass(delta: number | null): string {
  if (!isNumber(delta) || Math.abs(delta) < 0.0005) return 'text-muted'
  return delta > 0 ? 'text-emerald-700 dark:text-emerald-300' : 'text-danger'
}

function ChangeTable({
  title,
  rows,
  empty,
}: {
  title: string
  rows: EvaluationTestChange[]
  empty: string
}) {
  const [suite, setSuite] = useState('')
  const [difficulty, setDifficulty] = useState('')
  const filtered = rows.filter(
    (row) =>
      (!suite || row.suite === suite) && (!difficulty || row.difficulty === difficulty),
  )
  const suites = [...new Set(rows.map((row) => row.suite).filter(Boolean))] as string[]
  const difficulties = [...new Set(rows.map((row) => row.difficulty).filter(Boolean))] as string[]

  return (
    <SectionCard
      title={title}
      description={`${rows.length} изменений pass/fail`}
      actions={
        <div className="flex flex-wrap gap-1">
          <select
            className="field w-auto py-1 text-xs"
            value={suite}
            onChange={(event) => setSuite(event.target.value)}
            aria-label={`Фильтр suite: ${title}`}
          >
            <option value="">Все suite</option>
            {suites.map((item) => (
              <option key={item} value={item}>
                {suiteLabel(item)}
              </option>
            ))}
          </select>
          <select
            className="field w-auto py-1 text-xs"
            value={difficulty}
            onChange={(event) => setDifficulty(event.target.value)}
            aria-label={`Фильтр difficulty: ${title}`}
          >
            <option value="">Все difficulty</option>
            {difficulties.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      }
    >
      {rows.length === 0 ? (
        <p className="text-xs text-muted">{empty}</p>
      ) : filtered.length === 0 ? (
        <p className="text-xs text-muted">Под выбранные фильтры изменений нет.</p>
      ) : (
        <ul className="space-y-1 text-xs">
          {filtered.map((row) => (
            <li
              key={`${row.test_id}-${row.retrieval_mode}-${row.attempt_index}`}
              className="flex flex-wrap items-center gap-2 rounded border border-line bg-raised/40 px-2 py-1"
            >
              <span className="font-mono text-[11px]">{row.test_id}</span>
              <ToneBadge tone="neutral">{row.retrieval_mode}</ToneBadge>
              {row.attempt_index > 0 ? (
                <ToneBadge tone="neutral">attempt {row.attempt_index}</ToneBadge>
              ) : null}
              <span className="text-muted">
                {row.from} → {row.to}
              </span>
              {row.reasons.length > 0 ? (
                <span className="font-mono text-[11px] text-muted">{row.reasons.join(', ')}</span>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </SectionCard>
  )
}

export function CompareTab({
  runs,
  initialBase,
  initialCandidate,
  onOpenRuns,
}: {
  runs: EvaluationRunListItem[]
  initialBase?: string | null
  initialCandidate?: string | null
  onOpenRuns: () => void
}) {
  const completed = useMemo(
    () => runs.filter((run) => run.status === 'completed' && run.has_summary),
    [runs],
  )
  const [base, setBase] = useState<string | null>(initialBase ?? null)
  const [candidate, setCandidate] = useState<string | null>(initialCandidate ?? null)

  useEffect(() => {
    if (completed.length === 0) return
    const latest = completed[0]
    if (!latest) return
    setCandidate((current) => current ?? latest.id)
    setBase((current) => {
      if (current) return current
      const previous = completed.find(
        (run) => run.id !== latest.id && run.suite === latest.suite && run.split === latest.split,
      )
      return previous?.id ?? completed[1]?.id ?? null
    })
  }, [completed])

  useEffect(() => {
    if (initialBase) setBase(initialBase)
    if (initialCandidate) setCandidate(initialCandidate)
  }, [initialBase, initialCandidate])

  const compare = useEvaluationCompare(base, candidate)

  if (completed.length < 2) {
    return (
      <EmptyState
        title="Нужны два завершённых запуска"
        description="Сравнение показывает дельты только между реальными прогонами; несопоставимые данные не превращаются в выводы."
        action={
          <button type="button" className="btn" onClick={onOpenRuns}>
            Открыть запуски
          </button>
        }
      />
    )
  }

  const data = compare.data

  return (
    <div className="space-y-4">
      <SectionCard title="Сравнение запусков" description="По умолчанию — предыдущий и последний завершённые запуски с тем же suite/split.">
        <div className="grid gap-3 sm:grid-cols-[1fr_auto_1fr] sm:items-end">
          <label className="space-y-1 text-xs text-muted">
            Базовый запуск
            <select
              className="field"
              value={base ?? ''}
              onChange={(event) => setBase(event.target.value || null)}
              aria-label="Базовый запуск"
            >
              <option value="">—</option>
              {completed.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.id} · {suiteLabel(run.suite)}/{run.split} · {run.retrieval_mode} ·{' '}
                  {formatPercent(run.pass_rate)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => {
              setBase(candidate)
              setCandidate(base)
            }}
            title="Поменять местами"
            aria-label="Поменять местами базовый и сравниваемый запуски"
          >
            <ArrowLeftRight className="size-4" aria-hidden="true" />
          </button>
          <label className="space-y-1 text-xs text-muted">
            Сравниваемый запуск
            <select
              className="field"
              value={candidate ?? ''}
              onChange={(event) => setCandidate(event.target.value || null)}
              aria-label="Сравниваемый запуск"
            >
              <option value="">—</option>
              {completed.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.id} · {suiteLabel(run.suite)}/{run.split} · {run.retrieval_mode} ·{' '}
                  {formatPercent(run.pass_rate)}
                </option>
              ))}
            </select>
          </label>
        </div>
      </SectionCard>

      {compare.isLoading ? (
        <div className="flex h-24 items-center justify-center">
          <Spinner label="Сравнение запусков" />
        </div>
      ) : compare.error ? (
        <ErrorState error={compare.error} title="Сравнение недоступно" onRetry={() => void compare.refetch()} />
      ) : !data ? (
        <EmptyState title="Выберите два разных запуска" description="Дельты считает API; frontend их не пересчитывает." />
      ) : (
        <>
          {!data.comparable ? (
            <Banner tone="warn" title="not_comparable">
              Дельты не выдаются как надёжный вывод: запуски несопоставимы. Причины:{' '}
              {data.reasons.map((reason) => COMPARE_REASON_LABEL[reason] ?? reason).join('; ') ||
                'не указаны'}
              . Side-by-side просмотр разрешён.
            </Banner>
          ) : null}
          {data.version_differences.length > 0 ? (
            <Banner tone="info" title="Различия версий">
              {data.version_differences
                .map((item) => VERSION_DIFFERENCE_LABEL[item] ?? item)
                .join(' · ')}
              . Дельты разрешены: сравниваются две версии, а не «лучшая» и «худшая».
            </Banner>
          ) : null}

          <SectionCard
            title="Метрики"
            description={`${data.base.id} (${data.base.git_sha?.slice(0, 7) ?? 'no-git'}) → ${
              data.candidate.id
            } (${data.candidate.git_sha?.slice(0, 7) ?? 'no-git'})`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-muted">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-2 font-medium">Метрика</th>
                    <th className="py-2 pr-2 text-right font-medium">До</th>
                    <th className="py-2 pr-2 text-right font-medium">После</th>
                    <th className="py-2 text-right font-medium">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {data.metrics.map((metric) => {
                    const meta = METRIC_META[metric.key] ?? { label: metric.key, kind: 'ratio' as MetricKind }
                    return (
                      <tr key={metric.key} className="border-b border-line/60 last:border-0">
                        <td className="py-1.5 pr-2">{meta.label}</td>
                        <td className="py-1.5 pr-2 text-right tabular-nums">
                          {formatValue(metric.base, meta.kind)}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums">
                          {formatValue(metric.candidate, meta.kind)}
                        </td>
                        <td className={`py-1.5 text-right tabular-nums ${deltaClass(metric.delta)}`}>
                          {formatDelta(metric.delta, meta.kind)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {!data.comparable ? (
              <p className="mt-2 text-[11px] text-muted">
                Δ = —, потому что запуски несопоставимы; значения показаны side-by-side.
              </p>
            ) : null}
          </SectionCard>

          <div className="grid gap-4 lg:grid-cols-2">
            <ChangeTable
              title="Регрессии"
              rows={data.regressions}
              empty="Регрессий pass → fail не найдено."
            />
            <ChangeTable
              title="Улучшения"
              rows={data.improvements}
              empty="Улучшений fail → pass не найдено."
            />
          </div>

          <SectionCard
            title="Все изменения статусов"
            description={`${data.test_changes.length} тестов изменили pass/fail между запусками.`}
          >
            {data.test_changes.length === 0 ? (
              <p className="text-xs text-muted">Статусы тестов не менялись.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-muted">
                    <tr className="border-b border-line">
                      <th className="py-1.5 pr-2 font-medium">Тест</th>
                      <th className="py-1.5 pr-2 font-medium">Mode</th>
                      <th className="py-1.5 pr-2 font-medium">Suite</th>
                      <th className="py-1.5 pr-2 font-medium">Difficulty</th>
                      <th className="py-1.5 pr-2 font-medium">Было</th>
                      <th className="py-1.5 pr-2 font-medium">Стало</th>
                      <th className="py-1.5 font-medium">Reasons</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.test_changes.map((row) => (
                      <tr
                        key={`${row.test_id}-${row.retrieval_mode}-${row.attempt_index}`}
                        className="border-b border-line/60 last:border-0"
                      >
                        <td className="py-1.5 pr-2 font-mono text-[11px]">{row.test_id}</td>
                        <td className="py-1.5 pr-2 font-mono">{row.retrieval_mode}</td>
                        <td className="py-1.5 pr-2">{row.suite ?? EMPTY_VALUE}</td>
                        <td className="py-1.5 pr-2">{row.difficulty ?? EMPTY_VALUE}</td>
                        <td className="py-1.5 pr-2">{row.from}</td>
                        <td className="py-1.5 pr-2">
                          <span className={row.to === 'pass' ? 'text-emerald-700 dark:text-emerald-300' : 'text-danger'}>
                            {row.to}
                          </span>
                        </td>
                        <td className="py-1.5 font-mono text-[11px] text-muted">
                          {row.reasons.join(', ') || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </SectionCard>

          <SectionCard title="Версии" description="Различия версий видны, но не скрывают дельты.">
            <div className="grid gap-2 text-xs sm:grid-cols-2">
              {[data.base, data.candidate].map((meta, index) => (
                <div key={meta.id} className="rounded-lg border border-line bg-raised/40 p-2">
                  <p className="font-medium text-ink">
                    {index === 0 ? 'До' : 'После'}: <span className="font-mono">{meta.id}</span>
                  </p>
                  <ul className="mt-1 space-y-0.5 text-muted">
                    <li>git: {meta.git_sha ?? EMPTY_VALUE}{meta.git_dirty ? ' (dirty)' : ''}</li>
                    <li>dataset: {(meta.dataset_sha256 ?? EMPTY_VALUE).slice(0, 12)}</li>
                    <li>corpus content: {(meta.corpus_content_sha256 ?? EMPTY_VALUE).slice(0, 12)}</li>
                    <li>index fingerprint: {(meta.corpus_fingerprint ?? EMPTY_VALUE).slice(0, 12)}</li>
                    <li>model: {meta.llm_model ?? EMPTY_VALUE}</li>
                  </ul>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}
    </div>
  )
}
