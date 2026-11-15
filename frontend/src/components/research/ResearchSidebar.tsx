import { useEffect, useRef, useState } from 'react'
import {
  Moon,
  Pencil,
  Search,
  SquarePen,
  Sun,
  Trash2,
  X,
} from 'lucide-react'
import type { Research } from '../../types/api'
import { formatDate, previewText } from '../../lib/format'
import { describeApiError } from '../../lib/errors'
import { useUiSettings } from '../../app/ui-settings'
import { ResearchStatusBadge } from '../common/StatusBadge'
import { ConfirmDialog } from '../common/ConfirmDialog'
import { EmptyState } from '../common/EmptyState'
import { ErrorState } from '../common/ErrorState'
import { Spinner } from '../common/Spinner'

function ResearchInitial({ title }: { title: string }) {
  const letter = title.trim().charAt(0).toUpperCase() || 'И'
  return <span className="text-xs font-semibold">{letter}</span>
}

export function ResearchSidebar({
  researches,
  total,
  search,
  onSearchChange,
  isLoading,
  error,
  onRetry,
  selectedId,
  onSelect,
  onCreate,
  isCreating,
  createError,
  onRename,
  onDelete,
  onCloseDrawer,
  collapsed = false,
  corpusName,
}: {
  researches: Research[]
  total: number
  search: string
  onSearchChange: (value: string) => void
  isLoading: boolean
  error: unknown
  onRetry: () => void
  selectedId: string | null
  onSelect: (research: Research) => void
  onCreate: () => void
  isCreating: boolean
  createError?: unknown
  onRename: (researchId: string, title: string) => Promise<void>
  onDelete: (researchId: string) => Promise<void>
  onCloseDrawer?: () => void
  collapsed?: boolean
  corpusName?: string
}) {
  const { theme, toggleTheme } = useUiSettings()
  const [renameTarget, setRenameTarget] = useState<Research | null>(null)
  const [renameTitle, setRenameTitle] = useState('')
  const [renameError, setRenameError] = useState<string | null>(null)
  const [renamePending, setRenamePending] = useState(false)
  const renameInputRef = useRef<HTMLInputElement>(null)

  const [deleteTarget, setDeleteTarget] = useState<Research | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [deletePending, setDeletePending] = useState(false)

  useEffect(() => {
    if (!renameTarget) return
    setRenameTitle(renameTarget.title)
    setRenameError(null)
    const timer = setTimeout(() => renameInputRef.current?.select(), 0)
    return () => clearTimeout(timer)
  }, [renameTarget])

  const submitRename = async () => {
    if (!renameTarget) return
    const title = renameTitle.trim()
    if (title.length < 1 || title.length > 200) {
      setRenameError('Название должно быть от 1 до 200 символов.')
      return
    }
    setRenamePending(true)
    try {
      await onRename(renameTarget.id, title)
      setRenameTarget(null)
    } catch (mutationError) {
      setRenameError(describeApiError(mutationError))
    } finally {
      setRenamePending(false)
    }
  }

  const submitDelete = async () => {
    if (!deleteTarget) return
    setDeletePending(true)
    try {
      await onDelete(deleteTarget.id)
      setDeleteTarget(null)
      setDeleteError(null)
    } catch (mutationError) {
      setDeleteError(describeApiError(mutationError))
    } finally {
      setDeletePending(false)
    }
  }

  return (
    <aside className="flex h-full min-h-0 w-full flex-col gap-2 p-2" aria-label="История диалогов">
      <div className={`flex items-center gap-2 px-1 py-1 ${collapsed ? 'justify-center' : ''}`}>
        <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent text-[11px] font-bold tracking-wide text-accent-ink">
          AR
        </div>
        {!collapsed ? (
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-ink">Agent RAG</p>
            <p className="truncate text-[10px] uppercase tracking-[0.12em] text-muted">
              Диалоги
            </p>
          </div>
        ) : null}
        {onCloseDrawer && !collapsed ? (
          <button
            type="button"
            className="btn btn-ghost ml-auto size-7 p-0"
            onClick={onCloseDrawer}
            aria-label="Закрыть панель диалогов"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        ) : null}
      </div>

      <button
        type="button"
        className={`btn btn-primary gap-2 rounded-lg ${
          collapsed ? 'size-9 justify-center self-center p-0' : 'w-full justify-start px-2.5'
        }`}
        onClick={onCreate}
        disabled={isCreating}
        aria-label="Новый диалог"
        title="Новый диалог"
      >
        <SquarePen className="size-4 shrink-0" aria-hidden="true" />
        {!collapsed ? <span>{isCreating ? 'Создание…' : 'Новый диалог'}</span> : null}
      </button>

      {!collapsed ? (
        <>
          <div className="relative">
            <Search
              className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted"
              aria-hidden="true"
            />
            <label className="sr-only" htmlFor="research-search">
              Поиск по диалогам
            </label>
            <input
              id="research-search"
              className="field py-1.5 pl-7 text-xs"
              type="search"
              placeholder="Поиск диалогов"
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
            />
          </div>
          {createError ? (
            <p role="alert" className="text-[11px] text-danger">
              {describeApiError(createError)}
            </p>
          ) : null}
        </>
      ) : null}

      <div className="scroll-area min-h-0 flex-1 px-0.5 py-0.5">
        {isLoading && researches.length === 0 ? (
          <div className={collapsed ? 'flex justify-center p-2' : 'p-2'}>
            {collapsed ? (
              <Spinner label="" />
            ) : (
              <Spinner label="Загрузка диалогов" />
            )}
          </div>
        ) : error ? (
          collapsed ? null : (
            <div className="p-1">
              <ErrorState error={error} title="История недоступна" onRetry={onRetry} />
            </div>
          )
        ) : researches.length === 0 ? (
          collapsed ? null : (
            <div className="p-1">
              <EmptyState
                title={search ? 'Ничего не найдено' : 'Диалогов пока нет'}
                description={
                  search
                    ? 'Поиск выполняется на backend по названию и тексту сообщений.'
                    : 'Создайте первый диалог, чтобы задать вопрос агенту.'
                }
              />
            </div>
          )
        ) : collapsed ? (
          <ul className="flex flex-col items-center gap-1" aria-label="Список диалогов">
            {researches.slice(0, 12).map((research) => {
              const selected = research.id === selectedId
              return (
                <li key={research.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(research)}
                    title={research.title}
                    aria-label={research.title}
                    aria-current={selected ? 'true' : undefined}
                    className={`grid size-9 place-items-center rounded-lg border transition-colors ${
                      selected
                        ? 'border-accent/40 bg-accent-soft text-accent'
                        : 'border-transparent text-muted hover:bg-raised hover:text-ink'
                    }`}
                  >
                    <ResearchInitial title={research.title} />
                  </button>
                </li>
              )
            })}
          </ul>
        ) : (
          <ul className="space-y-0.5" aria-label="Список диалогов">
            {researches.map((research) => {
              const selected = research.id === selectedId
              return (
                <li key={research.id}>
                  <div
                    className={`group relative rounded-lg border px-2 py-1.5 transition-colors ${
                      selected
                        ? 'border-accent/30 bg-accent-soft/50'
                        : 'border-transparent hover:bg-raised'
                    }`}
                  >
                    <button
                      type="button"
                      className="block w-full text-left"
                      onClick={() => onSelect(research)}
                      aria-current={selected ? 'true' : undefined}
                    >
                      <span className="line-clamp-1 pr-10 text-[13px] font-medium text-ink">
                        {research.title}
                      </span>
                      <span className="mt-0.5 flex flex-wrap items-center gap-1 text-[10px] text-muted">
                        <ResearchStatusBadge status={research.last_status} />
                        <span>{formatDate(research.updated_at)}</span>
                        <span>· {research.messages_count}</span>
                      </span>
                      {research.last_message_preview ? (
                        <span className="mt-0.5 line-clamp-1 block text-[11px] text-muted/80">
                          {previewText(research.last_message_preview, 70)}
                        </span>
                      ) : null}
                    </button>
                    <div className="absolute right-1 top-1 flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                      <button
                        type="button"
                        className="grid size-6 place-items-center rounded text-muted hover:bg-surface hover:text-ink"
                        onClick={() => setRenameTarget(research)}
                        aria-label="Переименовать"
                        title="Переименовать"
                      >
                        <Pencil className="size-3" aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        className="grid size-6 place-items-center rounded text-muted hover:bg-surface hover:text-danger"
                        onClick={() => {
                          setDeleteError(null)
                          setDeleteTarget(research)
                        }}
                        aria-label="Удалить"
                        title="Удалить"
                      >
                        <Trash2 className="size-3" aria-hidden="true" />
                      </button>
                    </div>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <div
        className={`shrink-0 border-t border-line/70 pt-1.5 ${
          collapsed ? 'flex justify-center' : 'flex items-center gap-2 px-1'
        }`}
      >
        <button
          type="button"
          className="btn btn-ghost size-7 p-0"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему'}
          title={theme === 'dark' ? 'Светлая тема' : 'Тёмная тема'}
        >
          {theme === 'dark' ? (
            <Sun className="size-4" aria-hidden="true" />
          ) : (
            <Moon className="size-4" aria-hidden="true" />
          )}
        </button>
        {!collapsed ? (
          <div className="min-w-0 text-[10px] text-muted">
            <p className="truncate">{corpusName ?? 'Корпус не указан'}</p>
            <p>
              {researches.length} из {total} · история на backend
            </p>
          </div>
        ) : null}
      </div>

      {renameTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="rename-dialog-title"
            className="w-full max-w-md rounded-xl border border-line bg-surface p-5 shadow-panel"
          >
            <h2 id="rename-dialog-title" className="text-base font-semibold">
              Переименовать диалог
            </h2>
            <label className="mt-3 block text-xs text-muted" htmlFor="rename-input">
              Название (1–200 символов)
            </label>
            <input
              id="rename-input"
              ref={renameInputRef}
              className="field mt-1"
              value={renameTitle}
              maxLength={200}
              onChange={(event) => setRenameTitle(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void submitRename()
                if (event.key === 'Escape') setRenameTarget(null)
              }}
            />
            {renameError ? (
              <p role="alert" className="mt-2 text-xs text-danger">
                {renameError}
              </p>
            ) : null}
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                className="btn"
                onClick={() => setRenameTarget(null)}
              >
                Отмена
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => void submitRename()}
                disabled={renamePending}
              >
                {renamePending ? 'Сохранение…' : 'Сохранить'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Удалить диалог?"
        description={
          deleteTarget ? (
            <>
              «{deleteTarget.title}» и все сообщения будут удалены на backend. Действие необратимо.
              {deleteError ? <p className="mt-2 text-danger">{deleteError}</p> : null}
            </>
          ) : null
        }
        confirmLabel="Удалить"
        danger
        pending={deletePending}
        onConfirm={() => void submitDelete()}
        onCancel={() => setDeleteTarget(null)}
      />
    </aside>
  )
}
