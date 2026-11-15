import { useState } from 'react'
import { Download, Square, Tags } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { describeApiError } from '../../lib/errors'
import type { EvaluationRun } from '../../types/evaluation'
import { evaluationApi } from './evaluation-api'
import {
  EMPTY_VALUE,
  errorText,
  formatCoverage,
  formatDuration,
  formatMs,
  formatPercent,
  formatRatio,
  isActiveStatus,
  isNumber,
  suiteLabel,
} from './format'
import { ActiveRunNotice, KeyValue, ProgressBar, RunStatusBadge, SectionCard } from './components'

export function RunCard({
  run,
  canCancel,
  cancelPending,
  onCancel,
  onOpenTests,
  onCompare,
}: {
  run: EvaluationRun
  canCancel: boolean
  cancelPending: boolean
  onCancel: () => void
  onOpenTests: () => void
  onCompare: () => void
}) {
  const [reportError, setReportError] = useState<string | null>(null)
  const [downloading, setDownloading] = useState(false)
  const summary = run.summary
  const coverage = summary?.coverage
  const active = isActiveStatus(run.status)
  const partial = (run.status === 'cancelled' || run.status === 'interrupted') && Boolean(summary)

  const downloadReport = async () => {
    setReportError(null)
    setDownloading(true)
    try {
      const text = await evaluationApi.getReport(run.id)
      const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${run.id}-report.md`
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      setReportError(describeApiError(error))
    } finally {
      setDownloading(false)
    }
  }

  return (
    <SectionCard
      title={`Запуск ${run.id}`}
      description={`${suiteLabel(run.suite)} · ${run.split} · ${run.retrieval_mode} · judge ${
        run.judge ? 'on' : 'off'
      } · repeat ${run.repeat}`}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <RunStatusBadge status={run.status} />
          {canCancel && (run.status === 'queued' || run.status === 'running') ? (
            <button
              type="button"
              className="btn btn-ghost px-2 py-1 text-xs"
              onClick={onCancel}
              disabled={cancelPending}
            >
              <Square className="size-3.5" aria-hidden="true" />
              {cancelPending ? 'Отмена…' : 'Отменить'}
            </button>
          ) : null}
        </div>
      }
    >
      <div className="space-y-4">
        {active ? (
          <div className="space-y-2">
            <ActiveRunNotice status={run.status} />
            <ProgressBar
              percent={run.progress.percent}
              completed={run.progress.tests_completed}
              total={run.progress.tests_total}
              currentTestId={run.progress.current_test_id}
            />
          </div>
        ) : null}

        {run.error ? (
          <Banner tone={run.status === 'failed' ? 'error' : 'warn'} title="Статус запуска">
            {errorText(run.error)}
            {run.error.retryable ? ' · можно повторить запуск' : ''}
          </Banner>
        ) : null}
        {partial ? (
          <Banner tone="warn" title="Частичные результаты">
            Запуск не завершён ({run.status}); сохранены только уже записанные результаты. Артефакты
            не выдаются за полный прогон.
          </Banner>
        ) : null}

        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <KeyValue label="Git SHA" value={`${run.versions.git_sha || EMPTY_VALUE}${run.versions.git_dirty ? ' (dirty)' : ''}`} />
          <KeyValue label="Dataset" value={`${run.dataset.version ?? EMPTY_VALUE} · ${(run.dataset.sha256 ?? EMPTY_VALUE).slice(0, 12)}`} />
          <KeyValue label="Corpus content" value={(run.versions.corpus_content_sha256 || EMPTY_VALUE).slice(0, 12)} />
          <KeyValue label="Index fingerprint" value={(run.versions.corpus_fingerprint || EMPTY_VALUE).slice(0, 12)} />
          <KeyValue label="LLM" value={run.versions.llm_model ?? EMPTY_VALUE} />
          <KeyValue label="Reranker" value={run.versions.reranker_model ?? EMPTY_VALUE} />
          <KeyValue label="Config fingerprint" value={(run.versions.config_fingerprint || EMPTY_VALUE).slice(0, 12)} />
          <KeyValue label="Длительность" value={formatDuration(run.duration_ms)} />
        </dl>

        {summary ? (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-lg border border-line bg-raised/60 p-3">
                <p className="panel-title">Pass rate</p>
                <p className="text-lg font-semibold tabular-nums">{formatPercent(summary.pass_rate)}</p>
                <p className="text-[11px] text-muted">
                  {isNumber(coverage?.passed) && isNumber(coverage?.tests_eligible)
                    ? `${coverage?.passed}/${coverage?.tests_eligible} eligible`
                    : 'eligible не указан'}{' '}
                  · {formatPercent(summary.execution_coverage)} executed
                </p>
              </div>
              <div className="rounded-lg border border-line bg-raised/60 p-3">
                <p className="panel-title">Recall@10</p>
                <p className="text-lg font-semibold tabular-nums">
                  {formatRatio(summary.groups?.retrieval?.recall_at_10, 2)}
                </p>
                <p className="text-[11px] text-muted">
                  Evidence: {formatRatio(summary.groups?.retrieval?.evidence_coverage, 2)}
                </p>
              </div>
              <div className="rounded-lg border border-line bg-raised/60 p-3">
                <p className="panel-title">Answer quality</p>
                <p className="text-lg font-semibold tabular-nums">
                  {isNumber(summary.groups?.answer?.rubric_mean)
                    ? `${formatRatio(summary.groups?.answer?.rubric_mean, 1)} / 10`
                    : formatPercent(summary.groups?.answer?.automatic_pass_rate)}
                </p>
                <p className="text-[11px] text-muted">
                  {isNumber(summary.groups?.answer?.rubric_mean)
                    ? 'rubric mean'
                    : 'automatic pass rate'}
                </p>
              </div>
              <div className="rounded-lg border border-line bg-raised/60 p-3">
                <p className="panel-title">Multi-turn</p>
                <p className="text-lg font-semibold tabular-nums">
                  {formatPercent(summary.groups?.dialogue?.scenario_success_rate)}
                </p>
                <p className="text-[11px] text-muted">scenario success rate</p>
              </div>
            </div>

            <div className="grid gap-2 text-xs text-muted sm:grid-cols-2 lg:grid-cols-4">
              <span>
                Coverage: {formatCoverage({
                  eligible: coverage?.tests_eligible,
                  executed: coverage?.executed,
                  passed: coverage?.passed,
                })}
              </span>
              <span>
                selected: {coverage?.tests_selected ?? EMPTY_VALUE} · skipped:{' '}
                {coverage?.skipped ?? EMPTY_VALUE} · errors: {coverage?.errors ?? EMPTY_VALUE}
              </span>
              <span>Pass@1: {formatPercent(summary.pass_at_1)} · stability: {formatPercent(summary.stability)}</span>
              <span>Средняя latency: {formatMs(summary.efficiency?.mean_latency_ms)}</span>
            </div>

            {(summary.limitations?.length ?? 0) > 0 ? (
              <div className="rounded-lg border border-line bg-raised/50 p-2 text-[11px] text-muted">
                <p className="font-medium text-ink">Ограничения прогона</p>
                <ul className="mt-1 list-disc space-y-0.5 pl-4">
                  {summary.limitations?.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
            ) : null}

            {(summary.errors?.length ?? 0) > 0 ? (
              <div className="rounded-lg border border-danger/30 bg-danger/10 p-2 text-[11px]">
                <p className="font-medium text-ink">Ошибки runner'а</p>
                <ul className="mt-1 space-y-0.5 text-muted">
                  {summary.errors?.map((item, index) => (
                    <li key={`${item.code}-${index}`}>
                      <span className="font-mono">{item.code}</span>
                      {item.test_id ? ` · ${item.test_id}` : ''}
                      {item.message ? ` — ${item.message}` : ''}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {(summary.critical_errors?.length ?? 0) > 0 || (summary.problem_tests?.length ?? 0) > 0 ? (
              <div className="grid gap-2 text-xs lg:grid-cols-2">
                <div>
                  <p className="panel-title">Critical errors</p>
                  {summary.critical_errors?.length ? (
                    <ul className="mt-1 space-y-0.5 text-muted">
                      {summary.critical_errors.map((item) => (
                        <li key={item.code}>
                          <span className="font-mono">{item.code}</span> · {item.count} ·{' '}
                          {(item.test_ids ?? []).slice(0, 3).join(', ') || EMPTY_VALUE}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1 text-muted">не зафиксированы</p>
                  )}
                </div>
                <div>
                  <p className="panel-title">Problem tests</p>
                  {summary.problem_tests?.length ? (
                    <ul className="mt-1 space-y-0.5 text-muted">
                      {summary.problem_tests.slice(0, 6).map((item) => (
                        <li key={item.id}>
                          <span className="font-mono">{item.id}</span> — {item.reasons.join(', ')}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1 text-muted">нет</p>
                  )}
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-xs text-muted">
            {active
              ? 'summary появится после завершения запуска.'
              : 'summary отсутствует: значения не подставляются нулями.'}
          </p>
        )}

        {reportError ? <Banner tone="error">{reportError}</Banner> : null}

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="btn"
            onClick={() => void downloadReport()}
            disabled={downloading}
          >
            <Download className="size-4" aria-hidden="true" />
            {downloading ? 'Загрузка…' : 'Открыть отчёт'}
          </button>
          <button type="button" className="btn" onClick={onOpenTests}>
            <Tags className="size-4" aria-hidden="true" /> Открыть тесты
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={onCompare}
            disabled={run.status !== 'completed'}
            title={run.status === 'completed' ? undefined : 'Сравнение доступно после завершения'}
          >
            Сравнить с предыдущим
          </button>
          {run.notes ? <span className="text-xs text-muted">Комментарий: {run.notes}</span> : null}
        </div>
      </div>
    </SectionCard>
  )
}
