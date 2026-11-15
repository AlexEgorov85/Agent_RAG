import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import type {
  EvaluationRetrievalCall,
  EvaluationStageItem,
  EvaluationTrace,
} from '../../types/evaluation'
import { EMPTY_VALUE, formatRatio, isNumber } from './format'

const STAGE_ORDER = ['dense', 'bm25', 'graph', 'rrf', 'rerank', 'returned', 'context'] as const
export const STAGE_LABEL: Record<string, string> = {
  dense: 'Dense candidates',
  bm25: 'BM25 candidates',
  graph: 'Graph',
  rrf: 'RRF',
  rerank: 'Reranked',
  returned: 'Returned',
  context: 'Context',
}

function stageItems(
  call: EvaluationRetrievalCall,
  stage: string,
): EvaluationStageItem[] | null | undefined {
  const stages = call.stages
  if (!stages || !(stage in stages)) return undefined
  const items = stages[stage]
  return items ?? null
}

function StageTable({ call, gold }: { call: EvaluationRetrievalCall; gold: Set<string> }) {
  return (
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
          {STAGE_ORDER.map((stage) => {
            const items = stageItems(call, stage)
            const available = items !== undefined
            const list = Array.isArray(items) ? items : []
            return (
              <tr key={stage} className="border-b border-line/60 last:border-0 align-top">
                <td className="py-1.5 pr-2 whitespace-nowrap">{STAGE_LABEL[stage] ?? stage}</td>
                <td className="py-1.5 pr-2 tabular-nums">
                  {!available || items === null ? (
                    <span className="text-muted" title="Этап не измерялся">
                      не измерялся
                    </span>
                  ) : (
                    list.length
                  )}
                </td>
                <td className="py-1.5">
                  {list.length === 0 ? (
                    <span className="text-muted">{available ? 'пусто' : EMPTY_VALUE}</span>
                  ) : (
                    <span className="flex flex-wrap gap-1">
                      {list.slice(0, 6).map((item, index) => {
                        const chunkId = String(item.chunk_id ?? '')
                        return (
                          <span
                            key={`${chunkId}-${index}`}
                            className={`font-mono text-[11px] ${
                              gold.has(chunkId) ? 'font-semibold text-emerald-700 dark:text-emerald-300' : ''
                            }`}
                            title={`${chunkId}${gold.has(chunkId) ? ' · gold evidence' : ''}${
                              isNumber(item.score) ? ` · score ${formatRatio(item.score, 3)}` : ''
                            }`}
                          >
                            {item.rank ?? index + 1}. {chunkId}
                            {gold.has(chunkId) ? ' ✓gold' : ''}
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
  )
}

/**
 * Просмотр trace: retrieval stages и ответ. Prompt, chain-of-thought и полный
 * контекст сюда не попадают — их нет в trace-файле (§8.8).
 */
export function TraceViewer({ trace, gold }: { trace: EvaluationTrace; gold: Set<string> }) {
  const turns = Array.isArray(trace.turns) ? trace.turns : []
  return (
    <div className="space-y-3">
      {trace.truncated ? (
        <Banner tone="warn" title="Trace обрезан">
          Полный stage list остаётся в retrieval.csv; здесь показана усечённая версия.
        </Banner>
      ) : null}
      <p className="text-[11px] text-muted">
        trace_schema {trace.trace_schema_version ?? EMPTY_VALUE} · status {trace.status ?? EMPTY_VALUE} ·
        turns {turns.length}
      </p>
      {turns.map((turn, index) => {
        const calls = Array.isArray(turn.retrieval_calls)
          ? (turn.retrieval_calls as EvaluationRetrievalCall[])
          : []
        const answer = (turn.answer ?? {}) as Record<string, unknown>
        return (
          <details key={index} className="rounded-lg border border-line bg-raised/40 p-2" open={index === 0}>
            <summary className="cursor-pointer text-xs font-medium text-ink">
              Turn {Number(turn.turn_index ?? index) + 1}: {String(turn.question ?? '')}
            </summary>
            <div className="mt-2 space-y-3">
              {calls.length === 0 ? (
                <p className="text-xs text-muted">Retrieval calls отсутствуют в trace.</p>
              ) : (
                calls.map((call, callIndex) => (
                  <div key={callIndex} className="space-y-1">
                    <p className="text-[11px] text-muted">
                      call {call.call_index ?? callIndex} · {call.tool_name ?? 'search_reports'}
                      {call.retrieval_mode ? ` · ${call.retrieval_mode}` : ''}
                      {call.query ? ` · «${call.query}»` : ''}
                    </p>
                    <StageTable call={call} gold={gold} />
                  </div>
                ))
              )}
              <div className="text-xs">
                <p className="panel-title">Ответ</p>
                <p className="mt-1 whitespace-pre-wrap text-ink">
                  {typeof answer.text === 'string' && answer.text ? answer.text : EMPTY_VALUE}
                </p>
              </div>
            </div>
          </details>
        )
      })}
    </div>
  )
}

export function CollapsibleTrace({
  available,
  loading,
  error,
  trace,
  gold,
  onLoad,
}: {
  available: boolean
  loading: boolean
  error: unknown
  trace: EvaluationTrace | undefined
  gold: Set<string>
  onLoad: () => void
}) {
  const [open, setOpen] = useState(false)

  const handleToggle = () => {
    const next = !open
    setOpen(next)
    if (next && !trace && !loading) onLoad()
  }

  if (!available) {
    return (
      <Banner tone="warn" title="trace_unavailable">
        Trace для этого результата недоступен (файл не создан или удалён). Таблица этапов не
        подменяется пустой.
      </Banner>
    )
  }

  return (
    <div className="space-y-2">
      <button type="button" className="btn btn-ghost px-2 py-1 text-xs" onClick={handleToggle}>
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        {open ? 'Скрыть trace' : 'Показать trace'}
      </button>
      {open ? (
        loading ? (
          <p className="text-xs text-muted">Загрузка trace…</p>
        ) : error ? (
          <Banner tone="error" title="Trace не загрузился">
            {error instanceof Error ? error.message : 'Ошибка запроса trace.'}
          </Banner>
        ) : trace ? (
          <TraceViewer trace={trace} gold={gold} />
        ) : null
      ) : null}
    </div>
  )
}

/** Множество gold chunk_id из определения теста; только для визуального различия. */
export function useGoldChunkSet(goldEvidence: Record<string, unknown> | undefined): Set<string> {
  return useMemo(() => {
    const result = new Set<string>()
    const groups = goldEvidence?.groups
    if (Array.isArray(groups)) {
      for (const group of groups) {
        if (!group || typeof group !== 'object') continue
        const anyOf = (group as { any_of?: unknown }).any_of
        if (Array.isArray(anyOf)) {
          for (const chunkId of anyOf) {
            if (typeof chunkId === 'string') result.add(chunkId)
          }
        }
      }
    }
    return result
  }, [goldEvidence])
}
