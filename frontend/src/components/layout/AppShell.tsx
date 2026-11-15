import { useEffect, type ReactNode } from 'react'
import { X } from 'lucide-react'

/**
 * Base-demo shell: muted canvas, collapsible research rail, main chat card,
 * and an evidence drawer overlaying the chat (never a competing third column).
 */
export function AppShell({
  sidebar,
  sidebarCollapsed,
  header,
  children,
  evidence,
  evidenceOpen,
  onCloseEvidence,
  mobileSidebarOpen,
  onCloseMobileSidebar,
  isDesktop,
}: {
  sidebar: ReactNode
  sidebarCollapsed: boolean
  header: ReactNode
  children: ReactNode
  evidence: ReactNode
  evidenceOpen: boolean
  onCloseEvidence: () => void
  mobileSidebarOpen: boolean
  onCloseMobileSidebar: () => void
  isDesktop: boolean
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      if (document.querySelector('[role="dialog"][aria-modal="true"]:not([data-drawer])')) return
      if (evidenceOpen) {
        onCloseEvidence()
        return
      }
      if (mobileSidebarOpen) onCloseMobileSidebar()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [evidenceOpen, mobileSidebarOpen, onCloseEvidence, onCloseMobileSidebar])

  return (
    <div className="flex h-full overflow-hidden bg-canvas md:p-2 md:pl-0">
      {isDesktop ? (
        <aside
          className={`flex shrink-0 flex-col overflow-hidden transition-[width] duration-200 ${
            sidebarCollapsed ? 'w-16' : 'w-[18rem]'
          }`}
          aria-label="Список диалогов"
        >
          {sidebar}
        </aside>
      ) : null}

      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-surface md:rounded-xl md:border md:border-line/70 md:shadow-sm">
        {header}
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>

      {!isDesktop && mobileSidebarOpen ? (
        <div className="fixed inset-0 z-40">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={onCloseMobileSidebar}
            aria-hidden="true"
          />
          <div
            className="drawer-in absolute inset-y-0 left-0 flex w-80 max-w-[88vw] flex-col border-r border-line bg-surface shadow-panel"
            role="dialog"
            aria-modal="true"
            aria-label="История диалогов"
            data-drawer="true"
          >
            {sidebar}
          </div>
        </div>
      ) : null}

      {evidenceOpen ? (
        <div className="fixed inset-0 z-40">
          <div
            className="absolute inset-0 bg-black/25"
            onClick={onCloseEvidence}
            aria-hidden="true"
          />
          <div
            className="drawer-in absolute inset-y-0 right-0 flex w-[28rem] max-w-[94vw] flex-col border-l border-line bg-surface shadow-panel"
            role="dialog"
            aria-modal="true"
            aria-label="Источники и доказательства"
            data-drawer="true"
          >
            <div className="flex items-center justify-between border-b border-line px-3 py-2">
              <h2 className="panel-title">Источники и доказательства</h2>
              <button
                type="button"
                className="btn btn-ghost size-8 p-0"
                onClick={onCloseEvidence}
                aria-label="Закрыть панель источников"
                title="Закрыть"
              >
                <X className="size-4" aria-hidden="true" />
              </button>
            </div>
            <div className="min-h-0 flex-1">{evidence}</div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
