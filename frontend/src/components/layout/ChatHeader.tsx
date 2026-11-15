import { useEffect, useState, type MouseEvent } from 'react'
import {
  FlaskConical,
  Menu,
  Moon,
  PanelLeft,
  PanelRight,
  Share2,
  Sun,
} from 'lucide-react'
import type { CorpusInfo, FilterSet, Research, ResearchStatus } from '../../types/api'
import { useUiSettings } from '../../app/ui-settings'
import { navigate } from '../../app/routes'
import { describeApiError } from '../../lib/errors'
import { FiltersBar } from '../research/FiltersBar'

const STATUS_DOT: Record<ResearchStatus, string> = {
  idle: 'bg-muted/50',
  running: 'bg-accent animate-pulse',
  completed: 'bg-emerald-600',
  abstained: 'bg-warn',
  failed: 'bg-danger',
  cancelled: 'bg-muted',
}

export function ChatHeader({
  research,
  corpus,
  filters,
  onFiltersChange,
  onRename,
  sidebarCollapsed,
  onToggleSidebar,
  onOpenMobileSidebar,
  onToggleEvidence,
  evidenceOpen,
  evaluationEnabled = false,
}: {
  research: Research | null
  corpus?: CorpusInfo
  filters: FilterSet
  onFiltersChange: (filters: FilterSet) => void
  onRename: (title: string) => Promise<void>
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
  onOpenMobileSidebar: () => void
  onToggleEvidence: () => void
  evidenceOpen: boolean
  /**
   * `capabilities.evaluation_ui`: раздел Evaluation рекламируется только тогда,
   * когда backend действительно его отдаёт. Иначе ссылка скрыта, а прямой
   * `/evaluation` показывает честный `evaluation_disabled`.
   */
  evaluationEnabled?: boolean
}) {
  const { theme, toggleTheme } = useUiSettings()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setEditing(false)
    setError(null)
  }, [research?.id])

  const startEditing = () => {
    if (!research) return
    setDraft(research.title)
    setError(null)
    setEditing(true)
  }

  const commit = async () => {
    if (!research) return
    const title = draft.trim()
    if (!title || title.length > 200) {
      setError('Название: 1–200 символов')
      return
    }
    if (title === research.title) {
      setEditing(false)
      return
    }
    try {
      await onRename(title)
      setEditing(false)
    } catch (mutationError) {
      setError(describeApiError(mutationError))
    }
  }

  /**
   * Обычный <a href> сохраняет семантику ссылки (и открытие в новой вкладке по
   * Ctrl/Cmd+click), но внутри SPA навигация идёт через history API. Выбранный
   * research при этом не теряется: WorkspacePage хранит его id в localStorage.
   */
  const openEvaluation = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.defaultPrevented || event.button !== 0) return
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    navigate('/evaluation')
  }

  return (
    <header className="flex h-12 shrink-0 items-center gap-1.5 px-3">
      <button
        type="button"
        className="btn btn-ghost size-8 shrink-0 p-0 md:hidden"
        onClick={onOpenMobileSidebar}
        aria-label="Открыть список диалогов"
      >
        <Menu className="size-4" aria-hidden="true" />
      </button>

      <button
        type="button"
        className="btn btn-ghost hidden size-8 shrink-0 p-0 md:grid"
        onClick={onToggleSidebar}
        aria-label={sidebarCollapsed ? 'Показать список диалогов' : 'Скрыть список диалогов'}
        title={sidebarCollapsed ? 'Показать список диалогов' : 'Скрыть список диалогов'}
      >
        <PanelLeft className="size-4" aria-hidden="true" />
      </button>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span
          aria-hidden="true"
          className={`size-1.5 shrink-0 rounded-full ${
            research ? STATUS_DOT[research.last_status] : 'bg-muted/40'
          }`}
        />
        {editing && research ? (
          <input
            className="min-w-0 max-w-md flex-1 rounded-md border border-line bg-surface px-2 py-1 text-sm text-ink outline-none focus:border-accent/50"
            value={draft}
            maxLength={200}
            autoFocus
            aria-label="Название диалога"
            onChange={(event) => setDraft(event.target.value)}
            onBlur={() => void commit()}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void commit()
              if (event.key === 'Escape') {
                setEditing(false)
                setError(null)
              }
            }}
          />
        ) : (
          <button
            type="button"
            className="min-w-0 truncate rounded px-1 text-left text-sm font-medium text-ink hover:bg-raised disabled:hover:bg-transparent"
            onClick={startEditing}
            disabled={!research}
            title={research ? 'Переименовать диалог' : 'Диалог не выбран'}
          >
            {research?.title ?? 'Новый диалог'}
          </button>
        )}
        {error ? (
          <span role="alert" className="truncate text-[11px] text-danger">
            {error}
          </span>
        ) : null}
      </div>

      <div className="flex shrink-0 items-center gap-0.5">
        <FiltersBar corpus={corpus} filters={filters} onChange={onFiltersChange} />

        {evaluationEnabled ? (
          <a
            href="/evaluation"
            className="btn btn-ghost h-8 gap-1.5 px-2 text-xs"
            onClick={openEvaluation}
            aria-label="Evaluation"
            title="Открыть Evaluation / Benchmark"
          >
            <FlaskConical className="size-4" aria-hidden="true" />
            <span className="hidden sm:inline">Evaluation</span>
          </a>
        ) : null}

        <button
          type="button"
          className="btn btn-ghost size-8 p-0"
          onClick={onToggleEvidence}
          aria-label={evidenceOpen ? 'Скрыть панель источников' : 'Показать панель источников'}
          aria-pressed={evidenceOpen}
          title="Источники"
        >
          <PanelRight className="size-4" aria-hidden="true" />
        </button>

        <button
          type="button"
          className="btn btn-ghost size-8 p-0"
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

        <button
          type="button"
          disabled
          className="btn btn-ghost size-8 p-0 disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Поделиться"
          title="Поделиться недоступно в этой версии"
        >
          <Share2 className="size-4" aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}
