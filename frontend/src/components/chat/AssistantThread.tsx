import { forwardRef, useEffect, useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from 'react'
import {
  ActionBarPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
} from '@assistant-ui/react'
import { ArrowDown, Check, Copy, Download, RefreshCw } from 'lucide-react'
import { Markdown } from './Markdown'
import { Composer } from './Composer'
import { ExecutionTrace } from './ExecutionTrace'
import { CitationNavProvider } from './Citations'
import {
  AnalysisDataPart,
  AnswerStateDataPart,
  CoverageDataPart,
  EvidenceDataPart,
  RunErrorDataPart,
  VerificationDataPart,
  WaitingIndicator,
  WarningsDataPart,
} from './DataPartRenderers'
import {
  DATA_NAMES,
  evidenceOfMessage,
  getDataPart,
  type ExecutionData,
} from '../../features/assistant-runtime/data-parts'
import { useChatUi } from '../../features/assistant-runtime/chat-ui-context'

export const SUGGESTIONS: Array<{ label: string; prompt: string }> = [
  { label: 'Прибыль за 2023', prompt: 'Какая чистая прибыль за 2023 год?' },
  { label: 'Сравни 2020–2024', prompt: 'Сравни выручку за 2020–2024 годы.' },
  { label: 'Структура отчётов', prompt: 'Сравни структуру годовых отчётов за 2022 и 2023 годы.' },
  { label: 'Риски', prompt: 'Какие ключевые риски указаны в отчёте за 2023 год?' },
  { label: 'Что с 2021?', prompt: 'Что известно о чистой прибыли за 2021 год?' },
]

const IconAction = forwardRef<HTMLButtonElement, ComponentPropsWithoutRef<'button'>>(
  function IconAction({ className = '', children, ...props }, ref) {
    return (
      <button
        ref={ref}
        {...props}
        className={`grid size-7 place-items-center rounded-md text-muted transition-colors hover:bg-raised hover:text-ink disabled:opacity-40 ${className}`}
      >
        {children}
      </button>
    )
  },
)

function CopyAction() {
  const copied = useAuiState((state) => state.message.isCopied)
  return (
    <ActionBarPrimitive.Copy asChild>
      <IconAction type="button" aria-label="Скопировать ответ" title="Скопировать ответ">
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </IconAction>
    </ActionBarPrimitive.Copy>
  )
}

function UserMessage() {
  return (
    <MessagePrimitive.Root
      data-testid="message"
      data-role="user"
      className="fade-up mx-auto flex w-full max-w-[var(--thread-max-width)] justify-end px-2"
    >
      <div className="max-w-[min(34rem,88%)] rounded-2xl bg-raised px-4 py-2 text-[15px] leading-relaxed text-ink">
        <MessagePrimitive.Parts
          components={{
            Text: ({ text }) => <span className="whitespace-pre-wrap break-words">{text}</span>,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  )
}

function RuntimeCancelledNotice() {
  const status = useAuiState((state) => state.message.status)
  const hasAnswerState = useAuiState(
    (state) => getDataPart(state.message, DATA_NAMES.answerState) !== null,
  )
  if (hasAnswerState) return null
  if (!status || status.type !== 'incomplete' || status.reason !== 'cancelled') return null
  return (
    <p role="status" className="mt-3 rounded-lg border border-line bg-raised/60 px-3 py-2 text-xs text-muted">
      Выполнение отменено. Частичный текст не считается готовым ответом.
    </p>
  )
}

function BackendErrorMessage() {
  const hasBackendError = useAuiState(
    (state) => getDataPart(state.message, DATA_NAMES.runError) !== null,
  )
  if (hasBackendError) return null
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="mt-3 rounded-lg border border-danger/35 bg-danger/10 px-3 py-2 text-xs text-danger dark:text-red-200">
        <ErrorPrimitive.Message className="line-clamp-4" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  )
}

function AssistantMessage() {
  const message = useAuiState((state) => state.message)
  const { openEvidence } = useChatUi()
  const evidence = useMemo(() => evidenceOfMessage(message), [message])
  const citationNav = useMemo(
    () => ({
      items: evidence,
      openEvidence: (evidenceId: string) => openEvidence(evidence, evidenceId),
    }),
    [evidence, openEvidence],
  )

  return (
    <MessagePrimitive.Root
      data-testid="message"
      data-role="assistant"
      className="fade-up group/message relative mx-auto w-full max-w-[var(--thread-max-width)] px-2"
    >
      <CitationNavProvider value={citationNav}>
        <div className="min-w-0 text-[15px] leading-relaxed text-ink">
          <MessagePrimitive.Parts
            components={{
              Text: ({ text }) => <Markdown content={text} />,
              Empty: WaitingIndicator,
              data: {
                by_name: {
                  [DATA_NAMES.execution]: ({ data }) => (
                    <ExecutionTrace data={data as ExecutionData} />
                  ),
                  [DATA_NAMES.evidence]: EvidenceDataPart,
                  [DATA_NAMES.analysis]: AnalysisDataPart,
                  [DATA_NAMES.coverage]: CoverageDataPart,
                  [DATA_NAMES.warnings]: WarningsDataPart,
                  [DATA_NAMES.runError]: RunErrorDataPart,
                  [DATA_NAMES.answerState]: AnswerStateDataPart,
                  [DATA_NAMES.verification]: VerificationDataPart,
                },
                Fallback: () => null,
              },
            }}
          />
          <RuntimeCancelledNotice />
          <BackendErrorMessage />
        </div>
      </CitationNavProvider>

      <div className="mt-1 flex items-center gap-0.5 opacity-100 transition-opacity md:opacity-0 md:group-hover/message:opacity-100 md:group-focus-within/message:opacity-100">
        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="not-last"
          className="flex items-center gap-0.5"
        >
          <CopyAction />
          <ActionBarPrimitive.Reload asChild>
            <IconAction type="button" aria-label="Повторить ответ" title="Повторить ответ">
              <RefreshCw className="size-3.5" />
            </IconAction>
          </ActionBarPrimitive.Reload>
          <ActionBarPrimitive.ExportMarkdown asChild>
            <IconAction
              type="button"
              aria-label="Экспорт ответа в Markdown"
              title="Экспорт ответа в Markdown"
            >
              <Download className="size-3.5" />
            </IconAction>
          </ActionBarPrimitive.ExportMarkdown>
        </ActionBarPrimitive.Root>
      </div>
    </MessagePrimitive.Root>
  )
}

function MessageRow() {
  const role = useAuiState((state) => state.message.role)
  if (role === 'user') return <UserMessage />
  if (role === 'assistant') return <AssistantMessage />
  return null
}

/**
 * Screen-reader announcement when a streaming run settles. Text is owned by
 * assistant-ui; this region only reports that the answer is ready, so assistive
 * technology is not left waiting after the token stream stops.
 */
function StreamFinishedAnnouncer() {
  const isRunning = useAuiState((state) => state.thread.isRunning)
  const [announcement, setAnnouncement] = useState('')
  const wasRunning = useRef(false)

  useEffect(() => {
    if (wasRunning.current && !isRunning) setAnnouncement('Ответ сформирован.')
    wasRunning.current = isRunning
  }, [isRunning])

  return (
    <p className="sr-only" role="status" aria-live="polite">
      {announcement}
    </p>
  )
}

function Suggestions() {
  const aui = useAui()
  const isRunning = useAuiState((state) => state.thread.isRunning)

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 px-2" aria-label="Примеры вопросов">
      {SUGGESTIONS.map((suggestion) => (
        <button
          key={suggestion.label}
          type="button"
          disabled={isRunning}
          onClick={() => {
            if (isRunning) return
            aui.thread.append({ content: [{ type: 'text', text: suggestion.prompt }] })
          }}
          className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-muted transition-colors hover:border-accent/40 hover:bg-accent-soft/40 hover:text-ink disabled:opacity-50"
        >
          {suggestion.label}
        </button>
      ))}
    </div>
  )
}

/**
 * Thread layout of the Base demo: 44rem column, welcome + centered composer
 * for an empty research, docked composer and scroll-to-bottom later.
 */
export function AssistantThread({
  modelLabel,
  disabled = false,
  onCreateResearch,
  isCreating = false,
  banners,
  hasEarlier = false,
  isLoadingEarlier = false,
  onLoadEarlier,
}: {
  modelLabel: string
  disabled?: boolean
  onCreateResearch: () => void
  isCreating?: boolean
  banners?: ReactNode
  hasEarlier?: boolean
  isLoadingEarlier?: boolean
  onLoadEarlier?: () => void
}) {
  const isEmpty = useAuiState((state) => state.thread.isEmpty)
  const isRunning = useAuiState((state) => state.thread.isRunning)

  return (
    <section
      className="flex h-full min-h-0 flex-col bg-surface"
      aria-label="Диалог"
      style={{ ['--thread-max-width' as string]: '44rem' }}
    >
      {isEmpty ? null : <h1 className="sr-only">Диалог с агентом</h1>}
      <StreamFinishedAnnouncer />
      <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col">
        <ThreadPrimitive.Viewport
          className="relative flex min-h-0 flex-1 flex-col overflow-y-auto scroll-smooth px-4 pt-3"
          autoScroll
        >
          {banners ? (
            <div className="mx-auto mb-3 w-full max-w-[var(--thread-max-width)] space-y-2">
              {banners}
            </div>
          ) : null}

          {!isEmpty && (hasEarlier || isLoadingEarlier) ? (
            <div className="mx-auto mb-3 w-full max-w-[var(--thread-max-width)]">
              <button
                type="button"
                className="btn w-full text-xs"
                onClick={onLoadEarlier}
                disabled={isLoadingEarlier || isRunning}
              >
                {isLoadingEarlier
                  ? 'Загрузка более ранних сообщений…'
                  : 'Загрузить более ранние сообщения'}
              </button>
            </div>
          ) : null}

          {isEmpty ? (
            <div className="flex flex-1 flex-col items-center justify-center px-1 pb-10">
              <div className="w-full max-w-[var(--thread-max-width)] space-y-5">
                <div className="px-2">
                  <h1 className="text-2xl font-medium tracking-tight text-ink">
                    Чем я могу помочь?
                  </h1>
                  <p className="mt-1 text-sm text-muted">
                    Вопросы по годовым отчётам. Ответы опираются на источники backend, а не на
                    догадки интерфейса.
                  </p>
                </div>

                {disabled ? (
                  <div className="rounded-2xl border border-line bg-raised/40 p-4">
                    <p className="text-sm text-muted">
                      Диалог не выбран. Создайте его, чтобы задать вопрос агенту — история
                      хранится на backend.
                    </p>
                    <button
                      type="button"
                      className="btn btn-primary mt-3"
                      onClick={onCreateResearch}
                      disabled={isCreating}
                    >
                      {isCreating ? 'Создание…' : 'Создать диалог'}
                    </button>
                  </div>
                ) : (
                  <>
                    <Composer modelLabel={modelLabel} />
                    <Suggestions />
                  </>
                )}
              </div>
            </div>
          ) : (
            <>
              <div className="mx-auto w-full max-w-[var(--thread-max-width)] flex-1 space-y-6 pb-4">
                <ThreadPrimitive.Messages>
                  {() => <MessageRow />}
                </ThreadPrimitive.Messages>
              </div>

              <div className="sticky bottom-0 mt-auto bg-surface pb-4 pt-2">
                <div className="relative mx-auto w-full max-w-[var(--thread-max-width)] px-4">
                  <ThreadPrimitive.ScrollToBottom asChild>
                    <button
                      type="button"
                      aria-label="Прокрутить вниз"
                      title="Прокрутить вниз"
                      className="absolute -top-11 left-1/2 z-10 grid size-8 -translate-x-1/2 place-items-center rounded-full border border-line bg-surface text-muted shadow-sm transition-colors hover:text-ink disabled:invisible"
                    >
                      <ArrowDown className="size-4" />
                    </button>
                  </ThreadPrimitive.ScrollToBottom>
                  <Composer modelLabel={modelLabel} />
                </div>
              </div>
            </>
          )}
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </section>
  )
}
