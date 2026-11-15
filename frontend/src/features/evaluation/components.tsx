import type { ReactNode } from 'react'
import { ToneBadge, type BadgeTone } from '../../components/common/ToneBadge'
import { isActiveStatus, statusLabel, testStatusLabel, testStatusTone } from './format'
import { EMPTY_VALUE } from '../../lib/format'
import type { EvaluationRunStatus, EvaluationTestStatus } from '../../types/evaluation'

const RUN_TONE: Record<EvaluationRunStatus, BadgeTone> = {
  queued: 'neutral',
  running: 'active',
  cancel_requested: 'warn',
  completed: 'success',
  failed: 'danger',
  cancelled: 'neutral',
  interrupted: 'warn',
}

export function RunStatusBadge({ status }: { status: string }) {
  const tone = RUN_TONE[status as EvaluationRunStatus] ?? 'neutral'
  return (
    <ToneBadge tone={tone} title={`Статус запуска: ${status}`}>
      {statusLabel(status)}
    </ToneBadge>
  )
}

export function TestStatusBadge({
  status,
  pass,
}: {
  status: EvaluationTestStatus | string
  pass: boolean
}) {
  return (
    <ToneBadge tone={testStatusTone(status, pass)} title={`Статус теста: ${status}`}>
      {testStatusLabel(status)}
      {status !== 'skipped' ? ` · ${pass ? 'pass' : 'fail'}` : ''}
    </ToneBadge>
  )
}

export function SectionCard({
  title,
  description,
  actions,
  children,
}: {
  title: string
  description?: string
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="rounded-xl border border-line bg-surface p-4 shadow-sm">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {description ? <p className="mt-0.5 text-xs text-muted">{description}</p> : null}
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </header>
      <div className="mt-3">{children}</div>
    </section>
  )
}

/**
 * Карточка независимой метрики. `value` уже отформатирован API-значением;
 * отсутствие измерения — `—`, а не `0`.
 */
export function MetricCard({
  label,
  value,
  context,
  delta,
  deltaNote,
  unavailable,
}: {
  label: string
  value: string
  context?: ReactNode
  delta?: string
  deltaNote?: string
  unavailable?: string
}) {
  const deltaTone =
    delta && delta !== EMPTY_VALUE
      ? delta.startsWith('−')
        ? 'text-danger'
        : delta.startsWith('+')
          ? 'text-emerald-700 dark:text-emerald-300'
          : 'text-muted'
      : 'text-muted'
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-xl border border-line bg-surface p-4 shadow-sm">
      <p className="panel-title">{label}</p>
      <p className="text-2xl font-semibold tabular-nums text-ink" data-testid={`metric-${label}`}>
        {value}
      </p>
      {context ? <div className="text-xs text-muted">{context}</div> : null}
      {delta !== undefined ? (
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
          <span className={`font-medium tabular-nums ${deltaTone}`} aria-label="Дельта к предыдущему запуску">
            {delta}
          </span>
          {deltaNote ? <span className="text-muted">{deltaNote}</span> : null}
        </div>
      ) : null}
      {unavailable ? <p className="mt-1 text-xs text-muted">{unavailable}</p> : null}
    </div>
  )
}

export function ProgressBar({
  percent,
  completed,
  total,
  currentTestId,
}: {
  percent: number
  completed: number
  total: number
  currentTestId?: string | null
}) {
  const safePercent = Math.max(0, Math.min(100, percent))
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>
          {completed} / {total || '—'} пар (test × mode)
          {currentTestId ? ` · текущий ${currentTestId}` : ''}
        </span>
        <span className="tabular-nums">{safePercent.toLocaleString('ru-RU', { maximumFractionDigits: 1 })}%</span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-raised"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={safePercent}
      >
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${safePercent}%` }} />
      </div>
    </div>
  )
}

export function KeyValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-[0.08em] text-muted">{label}</dt>
      <dd className="truncate font-mono text-xs text-ink" title={typeof value === 'string' ? value : undefined}>
        {value}
      </dd>
    </div>
  )
}

export function ActiveRunNotice({ status }: { status: string }) {
  if (!isActiveStatus(status)) return null
  return (
    <div className="rounded-lg border border-accent/25 bg-accent-soft/50 px-3 py-2 text-xs text-ink">
      {status === 'cancel_requested'
        ? 'Запуск останавливается: worker завершает текущий turn.'
        : 'Выполняется benchmark. Чат может подождать один benchmark-turn.'}
    </div>
  )
}
