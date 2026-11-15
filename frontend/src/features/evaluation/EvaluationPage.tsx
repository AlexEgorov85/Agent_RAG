import { useEffect, useState, type KeyboardEvent, type MouseEvent } from 'react'
import { BarChart3, FlaskConical, GitCompareArrows, ListChecks, PlayCircle } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import type { EvaluationCapabilities } from '../../types/evaluation'
import { navigate } from '../../app/routes'
import { OverviewTab } from './OverviewTab'
import { RunsTab } from './RunsTab'
import { TestsTab } from './TestsTab'
import { CompareTab } from './CompareTab'
import { useEvaluationRuns } from './use-evaluation'

type EvaluationTab = 'overview' | 'runs' | 'tests' | 'compare'

/** Один динамический tabpanel: все вкладки управляют им через aria-controls. */
const TAB_PANEL_ID = 'eval-tabpanel'

function tabButtonId(key: EvaluationTab): string {
  return `eval-tab-${key}`
}

/**
 * Возврат в чат без потери выбранного research: WorkspacePage восстанавливает
 * его из localStorage. SPA-переход не перезагружает вкладку.
 */
function openChat(event: MouseEvent<HTMLAnchorElement>) {
  if (event.defaultPrevented || event.button !== 0) return
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  event.preventDefault()
  navigate('/')
}

const TABS: Array<{ key: EvaluationTab; label: string; icon: typeof BarChart3 }> = [
  { key: 'overview', label: 'Обзор', icon: BarChart3 },
  { key: 'runs', label: 'Запуски', icon: PlayCircle },
  { key: 'tests', label: 'Тесты', icon: ListChecks },
  { key: 'compare', label: 'Сравнение', icon: GitCompareArrows },
]

export function EvaluationPage({ capabilities }: { capabilities: EvaluationCapabilities }) {
  const [tab, setTab] = useState<EvaluationTab>('overview')
  const [focusRunId, setFocusRunId] = useState<string | null>(null)
  const [comparePreset, setComparePreset] = useState<{ base?: string; candidate?: string }>({})

  const runs = useEvaluationRuns({ limit: 50 })

  useEffect(() => {
    document.title = 'Evaluation / Benchmark — Agent RAG'
    return () => {
      document.title = 'Agent RAG'
    }
  }, [])

  const openCompare = (base?: string, candidate?: string) => {
    setComparePreset({ base, candidate })
    setTab('compare')
  }

  const openRuns = (runId?: string) => {
    setFocusRunId(runId ?? null)
    setTab('runs')
  }

  /** Keyboard pattern of a tablist: arrows/Home/End move the selected tab. */
  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, key: EvaluationTab) => {
    const index = TABS.findIndex((item) => item.key === key)
    let nextIndex: number | null = null
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % TABS.length
    else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + TABS.length) % TABS.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = TABS.length - 1
    if (nextIndex === null) return
    const next = TABS[nextIndex]
    if (!next) return
    event.preventDefault()
    setTab(next.key)
    document.getElementById(tabButtonId(next.key))?.focus()
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-canvas">
      <header className="shrink-0 border-b border-line bg-surface">
        <div className="mx-auto flex w-full max-w-[1500px] flex-wrap items-center gap-3 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <FlaskConical className="size-4 text-accent" aria-hidden="true" />
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
              Agent RAG
            </span>
          </div>
          <h1 className="text-sm font-semibold text-ink">Evaluation / Benchmark</h1>
          <nav aria-label="Верхняя навигация" className="ml-auto flex items-center gap-1 text-xs">
            <a
              href="/"
              className="btn btn-ghost px-2 py-1 text-xs"
              onClick={openChat}
              aria-label="Research / Chat"
              title="Вернуться к исследованию; выбранный диалог сохранится"
            >
              Research / Chat
            </a>
            <span aria-current="page" className="rounded-md bg-accent-soft px-2 py-1 font-medium text-accent">
              Evaluation
            </span>
          </nav>
        </div>
        <div
          role="tablist"
          aria-label="Разделы Evaluation"
          className="mx-auto flex w-full max-w-[1500px] gap-1 overflow-x-auto px-3"
        >
          {TABS.map((item) => {
            const Icon = item.icon
            const active = tab === item.key
            return (
              <button
                key={item.key}
                id={tabButtonId(item.key)}
                type="button"
                role="tab"
                aria-selected={active}
                aria-controls={TAB_PANEL_ID}
                tabIndex={active ? 0 : -1}
                className={`-mb-px inline-flex items-center gap-1.5 border-b-2 px-3 py-2 text-xs font-medium transition-colors ${
                  active
                    ? 'border-accent text-accent'
                    : 'border-transparent text-muted hover:text-ink'
                }`}
                onClick={() => setTab(item.key)}
                onKeyDown={(event) => handleTabKeyDown(event, item.key)}
              >
                <Icon className="size-3.5" aria-hidden="true" />
                {item.label}
              </button>
            )
          })}
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto px-3 py-4 md:px-6">
        <div
          id={TAB_PANEL_ID}
          role="tabpanel"
          aria-labelledby={tabButtonId(tab)}
          tabIndex={0}
          className="mx-auto w-full max-w-[1500px] space-y-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60"
        >
          {tab === 'overview' ? (
            <OverviewTab
              runs={runs.data?.items ?? []}
              isLoading={runs.isLoading}
              error={runs.error}
              onRetry={() => void runs.refetch()}
              onStartSmoke={() => openRuns()}
              onOpenRuns={() => openRuns()}
              onOpenCompare={openCompare}
              onOpenTests={() => setTab('tests')}
            />
          ) : null}
          {tab === 'runs' ? (
            <RunsTab
              capabilities={capabilities}
              onOpenTests={() => setTab('tests')}
              onOpenCompare={openCompare}
              focusRunId={focusRunId}
              onFocusHandled={() => setFocusRunId(null)}
            />
          ) : null}
          {tab === 'tests' ? <TestsTab onOpenRuns={() => openRuns()} /> : null}
          {tab === 'compare' ? (
            <CompareTab
              runs={runs.data?.items ?? []}
              initialBase={comparePreset.base}
              initialCandidate={comparePreset.candidate}
              onOpenRuns={() => openRuns()}
            />
          ) : null}
        </div>
        {runs.error && tab !== 'overview' ? (
          <div className="mx-auto mt-3 w-full max-w-[1500px]">
            <ErrorState error={runs.error} title="История запусков недоступна" onRetry={() => void runs.refetch()} />
          </div>
        ) : null}
        <p className="mx-auto mt-6 w-full max-w-[1500px] pb-6 text-[11px] text-muted">
          Evaluation читает артефакты benchmark (summary.json, cases.jsonl, traces) и не запускает
          retrieval/LLM во frontend. Неизмеренная метрика показывается как «—», а не 0. Benchmark
          использует тот же AgentService и общий LLM-lock, поэтому чат и оценка не выполняются
          параллельно.
        </p>
      </main>
    </div>
  )
}

/** Честный disabled route: при `evaluation_ui=false` нет пустого экрана. */
export function EvaluationDisabled() {
  return (
    <main className="mx-auto flex min-h-full max-w-2xl flex-col justify-center gap-4 px-6 py-16">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">Agent RAG</p>
      <h1 className="text-2xl font-semibold">Evaluation / Benchmark</h1>
      <Banner tone="warn" title="Раздел Evaluation отключён (evaluation_disabled)">
        Backend сообщает <span className="font-mono">evaluation_ui=false</span>. Включите раздел
        переменной <span className="font-mono">EVAL_UI_ENABLED=1</span> и перезапустите API. Здесь нет
        демонстрационных метрик: показывать вымышленные результаты оценки запрещено.
      </Banner>
      <div className="flex flex-wrap gap-2">
        <button type="button" className="btn btn-primary" onClick={() => navigate('/')}>
          Вернуться к исследованию
        </button>
        <button type="button" className="btn" onClick={() => window.location.reload()}>
          Проверить снова
        </button>
      </div>
    </main>
  )
}

export function EvaluationLoading() {
  return (
    <div className="flex h-full items-center justify-center bg-canvas p-4">
      <div className="rounded-xl border border-line bg-surface p-6">
        <Spinner label="Проверка доступности Evaluation" />
      </div>
    </div>
  )
}
