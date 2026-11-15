import { useEffect, useRef, useState } from 'react'
import { ChevronRight, Loader2, Wrench } from 'lucide-react'
import { filterSummary } from '../../lib/filters'
import { formatDuration } from '../../lib/format'
import { toolLabel } from '../../lib/labels'
import type { ExecutionData, StageSnapshot } from '../../features/assistant-runtime/data-parts'

const PHASE_LABEL: Record<string, string> = {
  idle: 'ожидание',
  submitting: 'запрос отправлен',
  streaming: 'выполняется',
  reconnecting: 'переподключение',
  completed: 'завершено',
  abstained: 'без утвердительного ответа',
  failed: 'ошибка',
  cancelled: 'отменено',
  interrupted: 'прервано',
}

const STAGE_ICON: Record<StageSnapshot['status'], string> = {
  pending: '○',
  active: '◉',
  done: '✓',
  failed: '✕',
  skipped: '–',
}

function isActivePhase(phase: string): boolean {
  return phase === 'streaming' || phase === 'submitting' || phase === 'reconnecting'
}

/**
 * Compact, collapsible run trace built only from backend events.
 * It contains stages, tool calls and applied filters — never chain-of-thought.
 */
export function ExecutionTrace({ data }: { data: ExecutionData }) {
  const active = isActivePhase(data.phase)
  const [open, setOpen] = useState(active)
  const previousActive = useRef(active)

  useEffect(() => {
    if (previousActive.current && !active) setOpen(false)
    previousActive.current = active
  }, [active])

  const stages = data.stages.filter((stage) => stage.status !== 'pending')
  const filters = filterSummary(data.effective_filters)
  const toolCount = data.tools.length
  const failed = data.phase === 'failed' || data.phase === 'interrupted'

  return (
    <div className="mb-2 overflow-hidden rounded-xl border border-line/80 bg-raised/40">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-muted hover:bg-raised/70"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <ChevronRight
          className={`size-3.5 shrink-0 transition-transform ${open ? 'rotate-90' : ''}`}
          aria-hidden="true"
        />
        {active ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-accent" aria-hidden="true" />
        ) : (
          <span
            aria-hidden="true"
            className={`size-1.5 shrink-0 rounded-full ${
              failed ? 'bg-danger' : data.phase === 'completed' ? 'bg-emerald-600' : 'bg-muted'
            }`}
          />
        )}
        <span className="font-medium text-ink">Ход выполнения</span>
        <span className="truncate">{PHASE_LABEL[data.phase] ?? data.phase}</span>
        {data.intent ? <span className="hidden truncate sm:inline">· {data.intent}</span> : null}
        {toolCount > 0 ? (
          <span className="ml-auto shrink-0 tabular-nums">
            {toolCount} {toolCount === 1 ? 'инструмент' : 'инстр.'}
          </span>
        ) : null}
      </button>

      {open ? (
        <div className="space-y-2 border-t border-line/70 px-3 py-2">
          {stages.length > 0 ? (
            <ol className="space-y-1">
              {stages.map((stage) => (
                <li key={stage.id} className="flex items-start gap-2 text-[11px]">
                  <span
                    aria-hidden="true"
                    className={
                      stage.status === 'active'
                        ? 'text-accent'
                        : stage.status === 'failed'
                          ? 'text-danger'
                          : 'text-muted'
                    }
                  >
                    {STAGE_ICON[stage.status]}
                  </span>
                  <span className="min-w-0">
                    <span className={stage.status === 'active' ? 'text-ink' : 'text-muted'}>
                      {stage.label}
                    </span>
                    {stage.detail ? <span className="text-muted/80"> — {stage.detail}</span> : null}
                    <span className="ml-1 font-mono text-[10px] text-muted/50">{stage.source}</span>
                  </span>
                </li>
              ))}
            </ol>
          ) : null}

          {data.tools.length > 0 ? (
            <ul className="space-y-1">
              {data.tools.map((tool) => (
                <li key={tool.id} className="text-[11px]">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                    <Wrench className="size-3 text-muted" aria-hidden="true" />
                    <span className="text-ink">{toolLabel(tool.tool_name)}</span>
                    <span className="font-mono text-[10px] text-muted/70">{tool.tool_name}</span>
                    <span
                      className={
                        tool.status === 'success'
                          ? 'text-emerald-700 dark:text-emerald-300'
                          : tool.status === 'error'
                            ? 'text-danger'
                            : 'text-accent'
                      }
                    >
                      {tool.status === 'running'
                        ? 'выполняется'
                        : tool.status === 'success'
                          ? 'успешно'
                          : 'ошибка'}
                    </span>
                    {tool.result_count !== undefined ? (
                      <span className="text-muted">найдено: {tool.result_count}</span>
                    ) : null}
                    {tool.duration_ms !== undefined ? (
                      <span className="text-muted">{formatDuration(tool.duration_ms)}</span>
                    ) : null}
                  </div>
                  {tool.short_arguments ? (
                    <p className="truncate pl-5 font-mono text-[10px] text-muted/60">
                      {Object.entries(tool.short_arguments)
                        .slice(0, 4)
                        .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
                        .join(' · ')}
                    </p>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}

          {filters.length > 0 ? (
            <p className="text-[11px] text-muted">Применённые фильтры: {filters.join('; ')}</p>
          ) : null}

          {data.filter_notices && data.filter_notices.length > 0 ? (
            <div
              role="status"
              className="rounded-md border border-warn/30 bg-warn/10 px-2 py-1.5 text-[11px]"
            >
              <p className="font-medium text-ink">Backend изменил фильтры</p>
              <ul className="mt-0.5 list-disc space-y-0.5 pl-4 text-muted">
                {data.filter_notices.map((notice) => (
                  <li key={notice}>{notice}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {data.resync_reason ? (
            <p className="rounded-md border border-warn/30 bg-warn/10 px-2 py-1.5 text-[11px]">
              Поток событий переподключён (причина: {data.resync_reason}). Данные загружены из REST.
            </p>
          ) : null}

          <p className="flex flex-wrap gap-x-3 font-mono text-[10px] text-muted/60">
            {data.request_id ? <span>request_id: {data.request_id}</span> : null}
            {data.sequence !== undefined ? <span>sequence: {data.sequence}</span> : null}
            {data.duration_ms !== undefined ? (
              <span>длительность: {formatDuration(data.duration_ms)}</span>
            ) : null}
          </p>
        </div>
      ) : null}
    </div>
  )
}
