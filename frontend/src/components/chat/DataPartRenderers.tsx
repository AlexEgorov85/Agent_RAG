import { useMemo } from 'react'
import { RefreshCw } from 'lucide-react'
import type { DataMessagePartProps } from '@assistant-ui/react'
import { ActionBarPrimitive, useAuiState } from '@assistant-ui/react'
import type { AnalysisResult, CoverageReport } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import { COVERAGE_STATUS_LABEL, errorLabel, warningLabel } from '../../lib/labels'
import { usePdfLinksState } from '../../lib/pdf-links'
import { AnalysisView } from './AnalysisView'
import { AssistantCitations } from './Citations'
import { CoverageMatrix } from './CoverageMatrix'
import type {
  AnalysisData,
  AnswerStateData,
  EvidenceData,
  RunErrorData,
} from '../../features/assistant-runtime/data-parts'
import { evidenceOfMessage } from '../../features/assistant-runtime/data-parts'
import { useChatUi } from '../../features/assistant-runtime/chat-ui-context'

/**
 * Recharts lives in a separate chunk (see `AnalysisView`), so the main chat
 * bundle does not grow with the chart library. Tables render from the same
 * backend payload as everything else; no UI is invented while a chunk loads.
 */

export function EvidenceDataPart({ data }: DataMessagePartProps<EvidenceData>) {
  const { openEvidence } = useChatUi()
  const pdfState = usePdfLinksState()
  const items: EvidenceItem[] = data?.items ?? []
  if (items.length === 0) return null

  return (
    <AssistantCitations
      items={items}
      pdfState={pdfState}
      onOpenEvidence={(evidenceId) => openEvidence(items, evidenceId)}
    />
  )
}

export function AnalysisDataPart({ data }: DataMessagePartProps<AnalysisData>) {
  if (!data) return null
  return <AnalysisView analysis={data as AnalysisResult} />
}

export function CoverageDataPart({ data }: DataMessagePartProps<CoverageReport>) {
  const { openEvidence } = useChatUi()
  const message = useAuiState((state) => state.message)
  const evidence = useMemo(() => evidenceOfMessage(message), [message])
  const evidenceById = useMemo(
    () => new Map<string, EvidenceItem>(evidence.map((item) => [item.evidence_id, item])),
    [evidence],
  )

  if (!data) return null
  const label =
    data.status === 'not_required'
      ? 'покрытие не требуется'
      : data.status === 'complete'
        ? 'покрытие полное'
        : data.status === 'partial'
          ? 'покрытие неполное'
          : 'недостаточно доказательств'

  return (
    <section
      aria-label="Покрытие доказательствами"
      className="mt-3 rounded-lg border border-line bg-raised/50 px-3 py-2 text-xs text-muted"
    >
      <p className="font-medium text-ink">Покрытие доказательствами backend: {label}.</p>
      {data.expected_years.length > 0 ? (
        <p className="mt-1">Ожидаемые годы: {data.expected_years.join(', ')}.</p>
      ) : null}
      {data.missing_years.length > 0 ? (
        <p>Годы без покрытия: {data.missing_years.join(', ')}.</p>
      ) : null}
      <p className="mt-1">
        Статусы по показателям: {Object.keys(data.metrics).length > 0 ? Object.keys(data.metrics).length : 'нет'}
        {'. Матрица показывает только статусы backend и не вычисляет покрытие самостоятельно.'}
      </p>

      <CoverageMatrix
        coverage={data}
        evidenceById={evidenceById}
        onOpenEvidence={(evidenceId) => openEvidence(evidence, evidenceId)}
      />

      {data.notes.length > 0 ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          {data.notes.map((note: string) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}

      <p className="sr-only">{Object.values(COVERAGE_STATUS_LABEL).join(', ')}</p>
    </section>
  )
}

export function WarningsDataPart({ data }: DataMessagePartProps<{ codes: string[] }>) {
  const codes: string[] = data?.codes ?? []
  if (codes.length === 0) return null
  return (
    <ul className="mt-2 space-y-0.5" aria-label="Предупреждения">
      {codes.map((code) => {
        const label = warningLabel(code)
        return (
          <li key={code} className="text-[11px] text-muted">
            <span className="mr-1" aria-hidden="true">
              ⚠
            </span>
            {label}
            {label !== code ? (
              <>
                {' '}
                <span className="font-mono text-[10px] text-muted/60">{code}</span>
              </>
            ) : null}
          </li>
        )
      })}
    </ul>
  )
}

export function RunErrorDataPart({ data }: DataMessagePartProps<RunErrorData>) {
  if (!data) return null
  return (
    <div role="alert" className="mt-3 rounded-lg border border-danger/35 bg-danger/10 px-3 py-2 text-xs">
      <p className="font-medium text-ink">
        {errorLabel(data.code)}: {data.message}
      </p>
      <p className="mt-1 flex flex-wrap gap-x-3 text-muted">
        <span>код: {data.code}</span>
        {data.request_id ? <span className="font-mono">request_id: {data.request_id}</span> : null}
        <span>{data.retryable ? 'можно повторить' : 'повтор не поможет'}</span>
      </p>
    </div>
  )
}

const ANSWER_STATE_TEXT: Record<AnswerStateData['state'], string> = {
  abstained:
    'Агент не сформировал утвердительный ответ: переданных доказательств недостаточно. Состояние пришло от backend.',
  cancelled: 'Выполнение отменено. Частичный текст не считается готовым ответом.',
  failed: 'Выполнение завершилось ошибкой backend. Технические детали указаны ниже.',
  interrupted:
    'Выполнение прервано: события этого запроса недоступны после перезапуска API. Повторите вопрос.',
}

export function AnswerStateDataPart({ data }: DataMessagePartProps<AnswerStateData>) {
  if (!data) return null
  const state = data.state as AnswerStateData['state']
  const warn = state === 'abstained' || state === 'interrupted'
  return (
    <div
      role="status"
      className={`mt-3 rounded-lg border px-3 py-2 text-xs text-ink ${
        warn ? 'border-warn/35 bg-warn/10' : 'border-line bg-raised/60'
      }`}
    >
      <p>{ANSWER_STATE_TEXT[state] ?? state}</p>
      {data.detail && state === 'interrupted' ? (
        <p className="mt-1 text-muted">{data.detail}</p>
      ) : null}
      {state === 'interrupted' ? (
        <ActionBarPrimitive.Reload asChild>
          <button type="button" className="btn btn-ghost mt-1.5 px-2 py-0.5 text-[11px]">
            <RefreshCw className="size-3" aria-hidden="true" />
            Повторить запрос
          </button>
        </ActionBarPrimitive.Reload>
      ) : null}
    </div>
  )
}

export function VerificationDataPart({ data }: DataMessagePartProps<unknown>) {
  if (!data) return null
  return (
    <details className="mt-2 rounded-lg border border-line bg-raised/40 px-2 py-1.5 text-[11px] text-muted">
      <summary className="cursor-pointer select-none">Проверка ответа (verification от backend)</summary>
      <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[10px]">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  )
}

/** Shown while the assistant message has no text yet; never imitates typing. */
export function WaitingIndicator() {
  const status = useAuiState((state) => state.message.status)
  const { tokenStreaming } = useChatUi()
  if (!status || status.type !== 'running') return null

  return (
    <div className="mb-3 flex flex-col gap-1 py-1 text-sm text-muted" role="status" aria-live="polite">
      <span className="inline-flex items-center gap-2">
        <span className="inline-flex gap-1" aria-hidden="true">
          <span className="size-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.2s]" />
          <span className="size-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.1s]" />
          <span className="size-1.5 animate-bounce rounded-full bg-muted" />
        </span>
        <span>Агент готовит ответ…</span>
      </span>
      {!tokenStreaming ? (
        <span className="text-[11px] text-muted/70">
          Текст ответа не передаётся по токенам (capabilities.token_streaming = false), поэтому
          ответ появится целиком после run_completed.
        </span>
      ) : null}
    </div>
  )
}
