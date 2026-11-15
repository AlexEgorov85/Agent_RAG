import { useState } from 'react'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import { ToneBadge } from '../../components/common/ToneBadge'
import type { RetrievalMode } from '../../types/evaluation'
import {
  CAUSE_LABEL,
  EMPTY_VALUE,
  LOSS_STAGE_LABEL,
  formatPercent,
  formatRatio,
  isNumber,
  testStatusLabel,
} from './format'
import { KeyValue, SectionCard, TestStatusBadge } from './components'
import { CollapsibleTrace, STAGE_LABEL, TraceViewer, useGoldChunkSet } from './TraceViewer'
import {
  useEvaluationRunTest,
  useEvaluationTest,
  useEvaluationTrace,
} from './use-evaluation'
import type { EvaluationEvidenceYear, EvaluationTraceCitation } from '../../types/evaluation'

const COVERAGE_LABEL: Record<EvaluationEvidenceYear['status'], string> = {
  covered: 'найдено',
  not_found: 'не найдено',
  year_absent: 'год отсутствует в корпусе',
  unknown: 'не измерялось',
}

const COVERAGE_TONE: Record<EvaluationEvidenceYear['status'], 'success' | 'warn' | 'neutral'> = {
  covered: 'success',
  not_found: 'warn',
  year_absent: 'neutral',
  unknown: 'neutral',
}

function textValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return EMPTY_VALUE
  if (Array.isArray(value)) return value.map((item) => String(item)).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function TestDetail({
  runId,
  testId,
  mode,
  runCreatedAt,
}: {
  runId: string
  testId: string
  mode: RetrievalMode
  runCreatedAt?: string
}) {
  const [traceRequested, setTraceRequested] = useState(false)
  const detail = useEvaluationRunTest(runId, testId, mode, 0)
  const history = useEvaluationTest(testId)
  const trace = useEvaluationTrace(runId, testId, mode, 0, traceRequested)
  const gold = useGoldChunkSet(detail.data?.test.gold_evidence)

  if (detail.isLoading) {
    return (
      <div className="flex h-40 items-center justify-center">
        <Spinner label="Загрузка результата теста" />
      </div>
    )
  }
  if (detail.error) {
    return (
      <ErrorState
        error={detail.error}
        title="Результат теста недоступен"
        onRetry={() => void detail.refetch()}
      />
    )
  }
  if (!detail.data) return null

  const data = detail.data
  const test = data.test
  const question =
    test.question ??
    (Array.isArray(test.turns) && test.turns.length > 0
      ? String((test.turns[0] as Record<string, unknown>).question ?? '')
      : null)
  const expected = test.expected ?? {}
  const calls = data.retrieval_calls ?? []
  const coverage = data.evidence.coverage ?? []
  const citations = data.answer.citations ?? []

  return (
    <div className="space-y-4">
      <SectionCard
        title={`Тест ${test.id}`}
        description={`${test.difficulty ?? EMPTY_VALUE} · ${test.category ?? EMPTY_VALUE} · suite ${
          test.suite ?? EMPTY_VALUE
        } · run ${runId}${runCreatedAt ? ` · ${new Date(runCreatedAt).toLocaleString('ru-RU')}` : ''}`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <ToneBadge tone="neutral">mode {data.retrieval_mode}</ToneBadge>
            <ToneBadge tone="neutral">attempt {data.attempt_index}</ToneBadge>
            <TestStatusBadge status={data.result.status} pass={data.result.pass} />
          </div>
        }
      >
        <div className="space-y-2">
          <p className="text-sm text-ink">{question ?? 'Вопрос недоступен: датасет не найден.'}</p>
          <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <KeyValue label="expected_status" value={test.expected_status ?? EMPTY_VALUE} />
            <KeyValue label="actual_status" value={data.answer.status ?? EMPTY_VALUE} />
            <KeyValue label="difficulty" value={test.difficulty ?? EMPTY_VALUE} />
            <KeyValue label="split" value={test.split ?? EMPTY_VALUE} />
          </dl>
          {test.reference_answer ? (
            <div className="rounded-lg border border-line bg-raised/50 p-2 text-xs">
              <p className="panel-title">Reference answer</p>
              <p className="mt-1 whitespace-pre-wrap text-ink">{test.reference_answer}</p>
            </div>
          ) : null}
          <div className="grid gap-2 text-xs sm:grid-cols-2">
            <div>
              <p className="panel-title">Expected facts / years / documents</p>
              <ul className="mt-1 space-y-0.5 text-muted">
                <li>required_years: {textValue((expected as Record<string, unknown>).required_years)}</li>
                <li>missing_years: {textValue((expected as Record<string, unknown>).missing_years)}</li>
                <li>
                  required_documents:{' '}
                  {textValue((expected as Record<string, unknown>).required_documents)}
                </li>
                <li>
                  required_facts: {textValue((expected as Record<string, unknown>).required_facts)}
                </li>
              </ul>
            </div>
            <div>
              <p className="panel-title">Gold evidence</p>
              {gold.size === 0 ? (
                <p className="mt-1 text-muted">gold evidence не указан для этого теста.</p>
              ) : (
                <p className="mt-1 flex flex-wrap gap-1 font-mono text-[11px] text-emerald-700 dark:text-emerald-300">
                  {[...gold].map((chunkId) => (
                    <span key={chunkId}>{chunkId}</span>
                  ))}
                </p>
              )}
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Query Plan" description="Flatten AgentRunResult.diagnostics.query_plan; API не создаёт альтернативную диагностику.">
        {data.query_plan ? (
          <dl className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <KeyValue label="intent" value={textValue(data.query_plan.intent)} />
            <KeyValue label="years" value={textValue(data.query_plan.years)} />
            <KeyValue
              label="year range"
              value={`${textValue(data.query_plan.year_from)} — ${textValue(data.query_plan.year_to)}`}
            />
            <KeyValue label="answer_mode" value={textValue(data.query_plan.answer_mode)} />
            <KeyValue label="needs_graph" value={textValue(data.query_plan.needs_graph)} />
            <KeyValue label="filters" value={textValue(data.query_plan.effective_filters)} />
          </dl>
        ) : (
          <p className="text-xs text-muted">Query plan отсутствует в trace.</p>
        )}
      </SectionCard>

      <SectionCard
        title="Retrieval"
        description="Этапы из trace; «не измерялся» — это не ноль. Gold chunks отмечены ✓."
      >
        {calls.length === 0 ? (
          <p className="text-xs text-muted">
            Retrieval calls недоступны: trace отсутствует. Таблица этапов не подменяется пустой.
          </p>
        ) : (
          <div className="space-y-3">
            {calls.map((call, index) => (
              <div key={`${call.call_index ?? index}`} className="space-y-1">
                <p className="text-[11px] text-muted">
                  call {call.call_index ?? index} · {call.tool_name ?? 'search_reports'}
                  {call.retrieval_mode ? ` · ${call.retrieval_mode}` : ''}
                  {call.query ? ` · «${call.query}»` : ''}
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="text-muted">
                      <tr className="border-b border-line">
                        <th className="py-1.5 pr-2 font-medium">Этап</th>
                        <th className="py-1.5 pr-2 font-medium">Count</th>
                        <th className="py-1.5 font-medium">Top chunks</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['dense', 'bm25', 'graph', 'rrf', 'rerank', 'returned', 'context'].map((stage) => {
                        const stageValue = call.stages?.[stage]
                        const measured = stageValue !== undefined
                        const items = Array.isArray(stageValue) ? stageValue : []
                        return (
                          <tr key={stage} className="border-b border-line/60 last:border-0 align-top">
                            <td className="py-1.5 pr-2 whitespace-nowrap">{STAGE_LABEL[stage] ?? stage}</td>
                            <td className="py-1.5 pr-2 tabular-nums">
                              {!measured || stageValue === null ? (
                                <span className="text-muted">не измерялся</span>
                              ) : (
                                items.length
                              )}
                            </td>
                            <td className="py-1.5">
                              {items.length === 0 ? (
                                <span className="text-muted">{measured ? 'пусто' : EMPTY_VALUE}</span>
                              ) : (
                                <span className="flex flex-wrap gap-1 font-mono text-[11px]">
                                  {items.slice(0, 6).map((item, itemIndex) => {
                                    const chunkId = String(item.chunk_id ?? '')
                                    return (
                                      <span
                                        key={`${chunkId}-${itemIndex}`}
                                        className={
                                          gold.has(chunkId)
                                            ? 'font-semibold text-emerald-700 dark:text-emerald-300'
                                            : ''
                                        }
                                        title={gold.has(chunkId) ? 'gold evidence' : undefined}
                                      >
                                        {item.rank ?? itemIndex + 1}. {chunkId}
                                        {gold.has(chunkId) ? ' ✓' : ''}
                                      </span>
                                    )
                                  })}
                                </span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Evidence"
        description="Покрытие по годам: найдено / не найдено / год отсутствует в корпусе."
      >
        <div className="space-y-3 text-xs">
          <div>
            <p className="panel-title">Покрытие по годам</p>
            {coverage.length === 0 ? (
              <p className="mt-1 text-muted">Coverage не измерялся.</p>
            ) : (
              <ul className="mt-1 flex flex-wrap gap-2">
                {coverage.map((row) => (
                  <li key={row.year} className="flex items-center gap-1">
                    <span className="font-mono">{row.year}</span>
                    <ToneBadge tone={COVERAGE_TONE[row.status]} title={`Статус года: ${row.status}`}>
                      {COVERAGE_LABEL[row.status]}
                    </ToneBadge>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <p className="panel-title">Evidence manifest</p>
            {data.evidence.manifest.length === 0 ? (
              <p className="mt-1 text-muted">Манифест пуст или недоступен.</p>
            ) : (
              <p className="mt-1 flex flex-wrap gap-1 font-mono text-[11px]">
                {data.evidence.manifest.map((chunkId) => (
                  <span
                    key={chunkId}
                    className={gold.has(chunkId) ? 'font-semibold text-emerald-700 dark:text-emerald-300' : ''}
                  >
                    {chunkId}
                    {gold.has(chunkId) ? ' ✓gold' : ''}
                  </span>
                ))}
              </p>
            )}
          </div>
          <div>
            <p className="panel-title">Context chunks</p>
            {data.evidence.context_chunk_ids === null ? (
              <p className="mt-1 text-muted">
                context_chunk_ids недоступен; источник coverage:{' '}
                {data.evidence.context_source ?? 'evidence_manifest'}.
              </p>
            ) : (
              <p className="mt-1 flex flex-wrap gap-1 font-mono text-[11px]">
                {data.evidence.context_chunk_ids.map((chunkId) => (
                  <span key={chunkId}>{chunkId}</span>
                ))}
              </p>
            )}
          </div>
          <p className="text-muted">
            Loss stage: {data.evidence.loss_stage ? LOSS_STAGE_LABEL[data.evidence.loss_stage] ?? data.evidence.loss_stage : EMPTY_VALUE}
          </p>
        </div>
      </SectionCard>

      <SectionCard title="Ответ агента" description="Текст и citations; полный контекст — только в trace.">
        <div className="space-y-2 text-xs">
          <p className="whitespace-pre-wrap text-sm text-ink">
            {data.answer.text || 'Ответ отсутствует (abstain или ошибка выполнения).'}
          </p>
          <p className="text-muted">
            Answer status: {data.answer.status ?? EMPTY_VALUE} · warnings:{' '}
            {(data.answer.warnings ?? []).join(', ') || 'нет'}
          </p>
          {citations.length > 0 ? (
            <ul className="space-y-1">
              {citations.map((citation: EvaluationTraceCitation, index) => (
                <li key={`${citation.chunk_id ?? index}`} className="rounded border border-line bg-raised/40 px-2 py-1">
                  <span className="font-mono text-[11px]">{citation.chunk_id ?? EMPTY_VALUE}</span>
                  {citation.year ? ` · ${citation.year}` : ''}
                  {citation.page_pdf ? ` · стр. ${citation.page_pdf}` : ''}
                  {citation.quote ? <p className="mt-1 line-clamp-2 text-muted">{citation.quote}</p> : null}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-muted">Citations отсутствуют.</p>
          )}
          {(data.answer.facts?.length ?? 0) > 0 || (data.answer.calculations?.length ?? 0) > 0 ? (
            <div className="grid gap-2 sm:grid-cols-2">
              <div>
                <p className="panel-title">Facts</p>
                <pre className="mt-1 max-h-32 overflow-auto rounded bg-raised p-2 text-[11px]">
                  {JSON.stringify(data.answer.facts ?? [], null, 2)}
                </pre>
              </div>
              <div>
                <p className="panel-title">Calculations</p>
                <pre className="mt-1 max-h-32 overflow-auto rounded bg-raised p-2 text-[11px]">
                  {JSON.stringify(data.answer.calculations ?? [], null, 2)}
                </pre>
              </div>
            </div>
          ) : null}
        </div>
      </SectionCard>

      <SectionCard title="Оценка" description="Четыре группы независимы; rubric — только при judge/manual.">
        <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="panel-title">Pass</p>
            <p className="text-lg font-semibold">{data.evaluation.pass ? 'pass' : 'fail'}</p>
            <p className="text-muted">automatic: {data.evaluation.automatic_pass === null || data.evaluation.automatic_pass === undefined ? EMPTY_VALUE : data.evaluation.automatic_pass ? 'pass' : 'fail'}</p>
          </div>
          <div>
            <p className="panel-title">Rubric</p>
            <p className="text-lg font-semibold">
              {isNumber(data.evaluation.rubric) ? `${formatRatio(data.evaluation.rubric, 1)} / 10` : EMPTY_VALUE}
            </p>
          </div>
          <div>
            <p className="panel-title">Recall@10</p>
            <p className="text-lg font-semibold">{formatRatio(data.result.metrics.recall_at_10, 2)}</p>
          </div>
          <div>
            <p className="panel-title">Evidence coverage</p>
            <p className="text-lg font-semibold">{formatRatio(data.result.metrics.evidence_coverage, 2)}</p>
          </div>
        </div>
        <dl className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
          <KeyValue label="fact accuracy" value={formatPercent(data.result.answer?.fact_accuracy)} />
          <KeyValue label="citation validity" value={formatRatio(data.result.answer?.citation_validity, 2)} />
          <KeyValue label="citation coverage" value={formatRatio(data.result.answer?.citation_coverage, 2)} />
          <KeyValue label="latency" value={data.result.duration_ms ? `${Math.round(data.result.duration_ms)} мс` : EMPTY_VALUE} />
        </dl>
        <div className="mt-3">
          <p className="panel-title">Critical errors</p>
          {data.evaluation.critical_errors.length === 0 ? (
            <p className="mt-1 text-xs text-muted">Нет critical errors.</p>
          ) : (
            <ul className="mt-1 space-y-0.5 text-xs">
              {data.evaluation.critical_errors.map((item, index) => (
                <li key={`${item.code}-${index}`}>
                  <span className="font-mono">{item.code}</span>
                  {item.message ? ` — ${item.message}` : ''}
                </li>
              ))}
            </ul>
          )}
        </div>
      </SectionCard>

      <SectionCard title="Диагностика" description="likely_cause приходит из evaluator; UI её не вычисляет.">
        <div className="space-y-1 text-xs">
          <p>
            Вероятная причина:{' '}
            <span className="font-medium text-ink">
              {data.diagnostics.likely_cause
                ? CAUSE_LABEL[data.diagnostics.likely_cause] ?? data.diagnostics.likely_cause
                : EMPTY_VALUE}
            </span>
          </p>
          {data.diagnostics.notes.length > 0 ? (
            <p className="text-muted">Коды: {data.diagnostics.notes.join(', ')}</p>
          ) : null}
          {data.result.warnings?.length ? (
            <p className="text-muted">Warnings теста: {data.result.warnings.join(', ')}</p>
          ) : null}
        </div>
      </SectionCard>

      <SectionCard title="Trace" description="Отдельный запрос; без prompt, chain-of-thought и полного текста корпуса.">
        {!data.trace_available ? (
          <CollapsibleTrace
            available={false}
            loading={false}
            error={null}
            trace={undefined}
            gold={gold}
            onLoad={() => {}}
          />
        ) : !traceRequested ? (
          <CollapsibleTrace
            available
            loading={false}
            error={null}
            trace={undefined}
            gold={gold}
            onLoad={() => setTraceRequested(true)}
          />
        ) : trace.isLoading ? (
          <Spinner label="Загрузка trace" />
        ) : trace.error ? (
          <ErrorState error={trace.error} title="Trace не загрузился" onRetry={() => void trace.refetch()} />
        ) : trace.data ? (
          <TraceViewer trace={trace.data} gold={gold} />
        ) : null}
      </SectionCard>

      <SectionCard title="История теста" description="Только завершённые запуски; режимы не смешиваются.">
        {history.isLoading ? (
          <Spinner label="Загрузка истории" />
        ) : history.error ? (
          <p className="text-xs text-muted">История недоступна: {String((history.error as Error).message)}</p>
        ) : (history.data?.history.length ?? 0) === 0 ? (
          <p className="text-xs text-muted">История пуста.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-muted">
                <tr className="border-b border-line">
                  <th className="py-1.5 pr-2 font-medium">Дата</th>
                  <th className="py-1.5 pr-2 font-medium">Run</th>
                  <th className="py-1.5 pr-2 font-medium">Mode</th>
                  <th className="py-1.5 pr-2 font-medium">Статус</th>
                  <th className="py-1.5 pr-2 font-medium">Recall@10</th>
                  <th className="py-1.5 font-medium">Errors</th>
                </tr>
              </thead>
              <tbody>
                {history.data?.history.map((entry) => (
                  <tr key={`${entry.run_id}-${entry.retrieval_mode}-${entry.attempt_index}`} className="border-b border-line/60 last:border-0">
                    <td className="py-1.5 pr-2 whitespace-nowrap">
                      {new Date(entry.created_at).toLocaleDateString('ru-RU')}
                    </td>
                    <td className="py-1.5 pr-2 font-mono text-[11px]">{entry.run_id}</td>
                    <td className="py-1.5 pr-2 font-mono">{entry.retrieval_mode}</td>
                    <td className="py-1.5 pr-2">
                      {entry.pass ? 'pass' : 'fail'} · {testStatusLabel(entry.status)}
                    </td>
                    <td className="py-1.5 pr-2 tabular-nums">{formatRatio(entry.metrics.recall_at_10, 2)}</td>
                    <td className="py-1.5 font-mono text-[11px]">
                      {entry.critical_errors.map((item) => item.code).join(', ') || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>
    </div>
  )
}
