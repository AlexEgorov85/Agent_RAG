import { useMemo, useState, type DragEvent } from 'react'
import { ComposerPrimitive, useAui, useAuiState } from '@assistant-ui/react'
import { ArrowUp, ChevronDown, Cpu, Paperclip, Square } from 'lucide-react'
import { describeApiError } from '../../lib/errors'
import { messageApi } from '../../services/message-api'
import {
  DATA_NAMES,
  getDataPart,
  type ExecutionData,
} from '../../features/assistant-runtime/data-parts'

/**
 * Base-demo composer: rounded shell, attachment + model row, circular send /
 * stop button. Attachments are not supported by the backend, so the button is
 * disabled and a drop is rejected with an explicit notice instead of pretending.
 *
 * Cancellation is best effort: the local stream is always aborted (the adapter
 * also asks backend to cancel), but a backend rejection is surfaced as an alert
 * instead of being swallowed.
 */
export function Composer({
  modelLabel,
  disabled = false,
  placeholder,
}: {
  modelLabel: string
  disabled?: boolean
  placeholder?: string
}) {
  const [dragging, setDragging] = useState(false)
  const [dropNotice, setDropNotice] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const aui = useAui()
  const isRunning = useAuiState((state) => state.thread.isRunning)
  const canSend = useAuiState((state) => state.composer.canSend)
  const isEmpty = useAuiState((state) => state.composer.isEmpty)
  const threadMessages = useAuiState((state) => state.thread.messages)

  const activeRequestId = useMemo(() => {
    for (let index = threadMessages.length - 1; index >= 0; index -= 1) {
      const message = threadMessages[index]
      if (!message || message.role !== 'assistant') continue
      const execution = getDataPart<ExecutionData>(message, DATA_NAMES.execution)
      if (execution?.request_id) return execution.request_id
    }
    return null
  }, [threadMessages])

  const handleCancel = async () => {
    if (cancelling) return
    setCancelling(true)
    setCancelError(null)

    let backendError: string | null = null
    if (activeRequestId) {
      try {
        await messageApi.cancel(activeRequestId)
      } catch (error) {
        backendError = describeApiError(error)
      }
    }

    // Always stop the local stream: a failed backend cancel must not leave the
    // UI spinning forever.
    aui.thread.cancelRun()
    setCancelling(false)
    if (backendError) {
      setCancelError(
        `Backend не подтвердил отмену: ${backendError}. Локальный поток остановлен; результат может появиться на backend после обновления.`,
      )
    }
  }

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (disabled) return
    event.preventDefault()
    setDragging(true)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    setDropNotice('Вложения не поддерживаются backend: файл не отправлен.')
  }

  return (
    <ComposerPrimitive.Root className="relative flex w-full flex-col">
      <div
        data-dragging={dragging}
        onDragOver={onDragOver}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex w-full cursor-text flex-col gap-1.5 rounded-2xl border bg-raised/60 p-2 transition-colors focus-within:border-accent/50 ${
          dragging ? 'border-dashed border-accent/70 bg-accent-soft/40' : 'border-line'
        }`}
      >
        <ComposerPrimitive.Input
          aria-label="Вопрос к агенту"
          placeholder={
            placeholder ??
            (disabled ? 'Сначала создайте диалог' : 'Отправить сообщение…')
          }
          disabled={disabled}
          minRows={1}
          maxRows={8}
          submitMode="enter"
          className="max-h-48 w-full resize-none bg-transparent px-3 py-1.5 text-[15px] leading-6 text-ink outline-none placeholder:text-muted/70 disabled:cursor-not-allowed"
        />

        <div className="flex items-center justify-between gap-1 px-0.5">
          <div className="flex min-w-0 items-center gap-1">
            <button
              type="button"
              disabled
              aria-label="Добавить вложение"
              title="Вложения не поддерживаются backend"
              className="grid size-7 shrink-0 place-items-center rounded-full text-muted/60 disabled:cursor-not-allowed"
            >
              <Paperclip className="size-4" aria-hidden="true" />
            </button>

            <button
              type="button"
              disabled
              title="Модель задаётся конфигурацией backend и не выбирается в этом интерфейсе"
              className="flex min-w-0 items-center gap-1 rounded-full px-2 py-1 text-xs text-muted disabled:cursor-default"
            >
              <Cpu className="size-3.5 shrink-0" aria-hidden="true" />
              <span className="max-w-[10rem] truncate">{modelLabel}</span>
              <ChevronDown className="size-3 shrink-0 opacity-60" aria-hidden="true" />
            </button>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            {isRunning ? (
              <button
                type="button"
                onClick={() => void handleCancel()}
                disabled={cancelling}
                aria-label="Остановить выполнение"
                aria-busy={cancelling}
                title={cancelling ? 'Отмена…' : 'Остановить выполнение'}
                className="grid size-8 place-items-center rounded-full bg-ink text-surface transition-opacity hover:opacity-90 disabled:opacity-60"
              >
                <Square className="size-3.5 fill-current" aria-hidden="true" />
              </button>
            ) : (
              <ComposerPrimitive.Send asChild>
                <button
                  type="button"
                  aria-label="Отправить сообщение"
                  title="Отправить сообщение"
                  disabled={disabled || !canSend || isEmpty}
                  className="grid size-8 place-items-center rounded-full bg-ink text-surface transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  <ArrowUp className="size-4" aria-hidden="true" />
                </button>
              </ComposerPrimitive.Send>
            )}
          </div>
        </div>
      </div>

      {dropNotice ? (
        <p role="status" className="mt-1 px-2 text-[11px] text-warn">
          {dropNotice}{' '}
          <button
            type="button"
            className="underline underline-offset-2"
            onClick={() => setDropNotice(null)}
          >
            Скрыть
          </button>
        </p>
      ) : null}

      {cancelError ? (
        <div role="alert" className="mt-1 flex items-start gap-2 px-2 text-[11px] text-danger">
          <span className="min-w-0 flex-1">{cancelError}</span>
          <button
            type="button"
            className="shrink-0 underline underline-offset-2"
            onClick={() => setCancelError(null)}
          >
            Скрыть
          </button>
        </div>
      ) : null}
    </ComposerPrimitive.Root>
  )
}
