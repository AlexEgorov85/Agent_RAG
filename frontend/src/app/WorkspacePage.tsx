import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useAuiState } from '@assistant-ui/react'
import type {
  Capabilities,
  CorpusInfo,
  FilterSet,
  Research,
} from '../types/api'
import type { EvidenceItem } from '../types/evidence'
import { AppShell } from '../components/layout/AppShell'
import { ChatHeader } from '../components/layout/ChatHeader'
import { ResearchSidebar } from '../components/research/ResearchSidebar'
import { AssistantThread } from '../components/chat/AssistantThread'
import { EvidencePanel } from '../components/evidence/EvidencePanel'
import { Banner } from '../components/common/Banner'
import { ErrorState } from '../components/common/ErrorState'
import { Spinner } from '../components/common/Spinner'
import { useMessageHistory } from '../features/conversations/use-conversation'
import { useCapabilities, useCorpus } from '../features/evidence/use-evidence'
import { readLastResearchId, writeLastResearchId } from '../features/research-history/last-research'
import {
  invalidateResearchData,
  useCreateResearch,
  useDeleteResearch,
  useRenameResearch,
  useResearches,
  useResearch,
} from '../features/research-history/use-researches'
import {
  createResearchChatAdapter,
} from '../features/assistant-runtime/chat-adapter'
import { chatMessageToThreadMessageLike } from '../features/assistant-runtime/chat-mapping'
import { ResearchRuntimeProvider } from '../features/assistant-runtime/ResearchRuntimeProvider'
import { ChatUiProvider } from '../features/assistant-runtime/chat-ui-context'
import {
  latestEvidenceMessage,
  evidenceOfMessage,
} from '../features/assistant-runtime/data-parts'
import { useUiSettings } from '../app/ui-settings'
import { describeApiError, isNotFound } from '../lib/errors'
import { useDebouncedValue } from '../lib/use-debounced'
import { useMediaQuery } from '../lib/use-media-query'
import { isMockMode } from '../services/api-client'
import { MOCK_BANNER } from '../mocks/data'

interface ManualEvidence {
  items: EvidenceItem[]
  evidenceId: string | null
}

export function WorkspacePage() {
  const queryClient = useQueryClient()
  const { sidebarCollapsed, toggleSidebar } = useUiSettings()
  const isDesktop = useMediaQuery('(min-width: 768px)')

  const [selectedResearchId, setSelectedResearchId] = useState<string | null>(() =>
    readLastResearchId(),
  )
  const [search, setSearch] = useState('')
  const debouncedSearch = useDebouncedValue(search)
  const [filters, setFilters] = useState<FilterSet>({})
  const [notice, setNotice] = useState<string | null>(null)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  const filtersRef = useRef<FilterSet>(filters)
  filtersRef.current = filters

  const capabilities = useCapabilities()
  const corpus = useCorpus()
  const researches = useResearches(debouncedSearch)
  const researchDetail = useResearch(selectedResearchId)
  const messages = useMessageHistory(selectedResearchId)
  const [historyEpoch, setHistoryEpoch] = useState(0)
  const createMutation = useCreateResearch()
  const renameMutation = useRenameResearch()
  const deleteMutation = useDeleteResearch()

  const selectResearch = useCallback((researchId: string | null) => {
    setSelectedResearchId(researchId)
    writeLastResearchId(researchId)
  }, [])

  const researchFromList = useMemo(
    () => researches.data?.items.find((item) => item.id === selectedResearchId) ?? null,
    [researches.data, selectedResearchId],
  )
  const research: Research | null = researchDetail.data ?? researchFromList

  /* Restore the last selected research from localStorage; backend validates it. */
  useEffect(() => {
    if (selectedResearchId) return
    if (debouncedSearch) return
    const items = researches.data?.items
    if (!items || items.length === 0) return
    const stored = readLastResearchId()
    const candidate = stored ? items.find((item) => item.id === stored) : items[0]
    if (candidate) selectResearch(candidate.id)
    else if (stored) writeLastResearchId(null)
  }, [researches.data, selectedResearchId, debouncedSearch, selectResearch])

  /* A research deleted in another tab yields 404: return to the list honestly. */
  const detailMissing = isNotFound(researchDetail.error)
  useEffect(() => {
    if (!detailMissing) return
    writeLastResearchId(null)
    setSelectedResearchId(null)
    setNotice(
      'Диалог не найден (код not_found). Возможно, он удалён в другой вкладке — выберите другой в списке.',
    )
    void queryClient.invalidateQueries({ queryKey: ['researches'] })
  }, [detailMissing, queryClient])

  const adapter = useMemo(
    () =>
      createResearchChatAdapter({
        researchId: selectedResearchId,
        getFilters: () => filtersRef.current,
        onSettled: (researchId) => invalidateResearchData(queryClient, researchId),
      }),
    [selectedResearchId, queryClient],
  )

  const initialMessages = useMemo(
    () => messages.messages.map(chatMessageToThreadMessageLike),
    [messages.messages],
  )

  /**
   * Older pages are fetched from `next_before`. The runtime only accepts
   * history on construction, so the epoch forks its key once the page is in
   * the cache and the merged, deduplicated list is rendered in order.
   */
  const handleLoadEarlier = useCallback(async () => {
    if (!messages.hasNextPage || messages.isFetchingNextPage) return
    try {
      const result = await messages.loadEarlier()
      if (result.isSuccess) setHistoryEpoch((value) => value + 1)
    } catch {
      // The failure is reported from the query state below the thread.
    }
  }, [messages.hasNextPage, messages.isFetchingNextPage, messages.loadEarlier])

  const extraCapabilities = capabilities.data as
    | (Capabilities & { model?: string; llm_model?: string; provider?: string })
    | undefined
  const modelLabel =
    extraCapabilities?.llm_model ??
    extraCapabilities?.model ??
    extraCapabilities?.provider ??
    'GigaChat'
  const tokenStreaming = capabilities.data?.token_streaming ?? false

  const capabilitiesError = capabilities.error
  const corpusError = corpus.error
  const mock = isMockMode()

  const banners = (
    <>
      {mock ? (
        <Banner tone="mock" title="Демонстрационные данные (mock)">
          {MOCK_BANNER}
        </Banner>
      ) : null}
      {capabilitiesError ? (
        <Banner tone="error" title="API недоступен">
          {describeApiError(capabilitiesError)} Чат не сможет отправить запрос, пока backend не
          ответит.
        </Banner>
      ) : null}
      {corpusError ? (
        <Banner tone="warn" title="Корпус недоступен">
          {describeApiError(corpusError)} Фильтры по годам, issuer и документам не показываются,
          потому что интерфейс не подставляет значения сам.
        </Banner>
      ) : null}
      {notice ? (
        <Banner
          tone="warn"
          title="Требуется внимание"
          action={
            <button
              type="button"
              className="btn btn-ghost px-2 py-1 text-xs"
              onClick={() => setNotice(null)}
            >
              Закрыть
            </button>
          }
        >
          {notice}
        </Banner>
      ) : null}
    </>
  )

  const handleCreateResearch = async () => {
    setNotice(null)
    try {
      const created = await createMutation.mutateAsync(undefined)
      selectResearch(created.id)
    } catch {
      // Error is rendered in the sidebar from createMutation.error.
    }
  }

  const handleSelectResearch = (next: Research) => {
    setNotice(null)
    selectResearch(next.id)
  }

  const handleRenameResearch = async (researchId: string, title: string) => {
    await renameMutation.mutateAsync({ researchId, title })
  }

  const handleDeleteResearch = async (researchId: string) => {
    await deleteMutation.mutateAsync(researchId)
    if (researchId === selectedResearchId) {
      const remaining = (researches.data?.items ?? []).filter((item) => item.id !== researchId)
      selectResearch(remaining[0]?.id ?? null)
    }
  }

  const sidebar = (
    <ResearchSidebar
      researches={researches.data?.items ?? []}
      total={researches.data?.total ?? 0}
      search={search}
      onSearchChange={setSearch}
      isLoading={researches.isLoading}
      error={researches.error}
      onRetry={() => void researches.refetch()}
      selectedId={selectedResearchId}
      onSelect={handleSelectResearch}
      onCreate={() => void handleCreateResearch()}
      isCreating={createMutation.isPending}
      createError={createMutation.error}
      onRename={handleRenameResearch}
      onDelete={handleDeleteResearch}
      onCloseDrawer={isDesktop ? undefined : () => setMobileSidebarOpen(false)}
      collapsed={isDesktop && sidebarCollapsed}
      corpusName={corpus.data?.name}
    />
  )

  useEffect(() => {
    if (isDesktop) setMobileSidebarOpen(false)
  }, [isDesktop])

  const runtimeKey = `${selectedResearchId ?? 'no-research'}:${historyEpoch}`

  if (selectedResearchId && messages.isLoading && !messages.data) {
    return (
      <div className="flex h-full items-center justify-center bg-canvas p-2">
        <div className="flex w-full max-w-md flex-col items-center gap-3 rounded-xl border border-line bg-surface p-8">
          <Spinner label="Загрузка диалога" />
        </div>
      </div>
    )
  }

  if (selectedResearchId && messages.isError) {
    return (
      <div className="flex h-full items-center justify-center bg-canvas p-4">
        <div className="w-full max-w-lg space-y-3">
          <h1 className="text-lg font-semibold">История диалога недоступна</h1>
          <ErrorState
            error={messages.error}
            title="Сообщения не загрузились"
            onRetry={() => void messages.refetch()}
          />
          <button
            type="button"
            className="btn"
            onClick={() => {
              selectResearch(null)
              void queryClient.invalidateQueries({ queryKey: ['researches'] })
            }}
          >
            Вернуться к списку диалогов
          </button>
        </div>
      </div>
    )
  }

  return (
    <ResearchRuntimeProvider key={runtimeKey} adapter={adapter} initialMessages={initialMessages}>
      <WorkspaceShell
        banners={banners}
        research={research}
        corpus={corpus.data}
        filters={filters}
        onFiltersChange={setFilters}
        onRename={async (title) => {
          if (!selectedResearchId) return
          await handleRenameResearch(selectedResearchId, title)
        }}
        isDesktop={isDesktop}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={toggleSidebar}
        mobileSidebarOpen={mobileSidebarOpen}
        onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
        onCloseMobileSidebar={() => setMobileSidebarOpen(false)}
        sidebar={sidebar}
        modelLabel={modelLabel}
        tokenStreaming={tokenStreaming}
        canChat={Boolean(selectedResearchId)}
        evaluationEnabled={Boolean(capabilities.data?.evaluation_ui)}
        onCreateResearch={() => void handleCreateResearch()}
        isCreating={createMutation.isPending}
        hasEarlier={messages.hasEarlier}
        isLoadingEarlier={messages.isLoadingEarlier}
        onLoadEarlier={() => void handleLoadEarlier()}
      />
    </ResearchRuntimeProvider>
  )
}

function WorkspaceShell({
  banners,
  research,
  corpus,
  filters,
  onFiltersChange,
  onRename,
  isDesktop,
  sidebarCollapsed,
  onToggleSidebar,
  mobileSidebarOpen,
  onOpenMobileSidebar,
  onCloseMobileSidebar,
  sidebar,
  modelLabel,
  tokenStreaming,
  canChat,
  evaluationEnabled,
  onCreateResearch,
  isCreating,
  hasEarlier,
  isLoadingEarlier,
  onLoadEarlier,
}: {
  banners: ReactNode
  research: Research | null
  corpus?: CorpusInfo
  filters: FilterSet
  onFiltersChange: (filters: FilterSet) => void
  onRename: (title: string) => Promise<void>
  isDesktop: boolean
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
  mobileSidebarOpen: boolean
  onOpenMobileSidebar: () => void
  onCloseMobileSidebar: () => void
  sidebar: ReactNode
  modelLabel: string
  tokenStreaming: boolean
  canChat: boolean
  evaluationEnabled: boolean
  onCreateResearch: () => void
  isCreating: boolean
  hasEarlier: boolean
  isLoadingEarlier: boolean
  onLoadEarlier: () => void
}) {
  const [evidenceOpen, setEvidenceOpen] = useState(false)
  const [manualEvidence, setManualEvidence] = useState<ManualEvidence | null>(null)
  const runtimeMessages = useAuiState((state) => state.thread.messages)
  const runActive = useAuiState((state) => state.thread.isRunning)

  const latestWithEvidence = useMemo(
    () => latestEvidenceMessage(runtimeMessages),
    [runtimeMessages],
  )
  const panelEvidence = manualEvidence?.items ?? (latestWithEvidence ? evidenceOfMessage(latestWithEvidence) : [])

  const openEvidence = useCallback((items: EvidenceItem[], evidenceId: string) => {
    setManualEvidence({ items, evidenceId })
    setEvidenceOpen(true)
  }, [])

  const uiContext = useMemo(
    () => ({ tokenStreaming, openEvidence }),
    [tokenStreaming, openEvidence],
  )

  return (
    <ChatUiProvider value={uiContext}>
      <AppShell
        sidebar={sidebar}
        sidebarCollapsed={sidebarCollapsed}
        mobileSidebarOpen={mobileSidebarOpen}
        onCloseMobileSidebar={onCloseMobileSidebar}
        isDesktop={isDesktop}
        header={
          <ChatHeader
            research={research}
            corpus={corpus}
            filters={filters}
            onFiltersChange={onFiltersChange}
            onRename={onRename}
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={onToggleSidebar}
            onOpenMobileSidebar={onOpenMobileSidebar}
            onToggleEvidence={() => setEvidenceOpen((value) => !value)}
            evidenceOpen={evidenceOpen}
            evaluationEnabled={evaluationEnabled}
          />
        }
        evidence={
          <EvidencePanel
            evidence={panelEvidence}
            selectedEvidenceId={manualEvidence?.evidenceId ?? null}
            onSelect={(evidenceId) =>
              setManualEvidence((current) =>
                current ? { ...current, evidenceId } : current,
              )
            }
            runActive={runActive}
          />
        }
        evidenceOpen={evidenceOpen}
        onCloseEvidence={() => setEvidenceOpen(false)}
      >
        <AssistantThread
          banners={banners}
          modelLabel={modelLabel}
          disabled={!canChat}
          onCreateResearch={onCreateResearch}
          isCreating={isCreating}
          hasEarlier={hasEarlier}
          isLoadingEarlier={isLoadingEarlier}
          onLoadEarlier={onLoadEarlier}
        />
      </AppShell>
    </ChatUiProvider>
  )
}
