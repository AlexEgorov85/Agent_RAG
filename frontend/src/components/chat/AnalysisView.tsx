import { lazy, Suspense, useMemo } from 'react'
import { useAuiState } from '@assistant-ui/react'
import type { AnalysisResult } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import { evidenceOfMessage } from '../../features/assistant-runtime/data-parts'
import { useChatUi } from '../../features/assistant-runtime/chat-ui-context'
import {
  groupCalculations,
  groupFacts,
  hasPoints,
  seriesLabel,
} from '../../lib/analytics'
import { pluralize } from '../../lib/format'
import { CalculationTable, FactsTable } from './AnalyticsTables'
import { StructureComparison } from './StructureComparison'

/** Recharts is heavy: it is fetched only when a calculation series has points. */
const CalculationCharts = lazy(async () => {
  const module = await import('./AnalyticsCharts')
  return { default: module.CalculationCharts }
})

/**
 * Stage 3–4 analytics section: backend `facts`, `calculations` and
 * `structural_changes` rendered as tables, charts and a structural comparison.
 * Recharts is imported only through this module, which the chat loads lazily.
 */
export function AnalysisView({ analysis }: { analysis: AnalysisResult }) {
  const message = useAuiState((state) => state.message)
  const { openEvidence } = useChatUi()

  const evidence = useMemo(() => evidenceOfMessage(message), [message])
  const evidenceById = useMemo(
    () => new Map<string, EvidenceItem>(evidence.map((item) => [item.evidence_id, item])),
    [evidence],
  )
  const open = (evidenceId: string) => openEvidence(evidence, evidenceId)

  const facts = analysis.facts ?? []
  const calculations = analysis.calculations ?? []
  const structural = analysis.structural_changes ?? []

  const factGroups = useMemo(() => groupFacts(facts), [facts])
  const calculationGroups = useMemo(() => groupCalculations(calculations), [calculations])

  if (facts.length === 0 && calculations.length === 0 && structural.length === 0) return null

  const summary: string[] = []
  if (facts.length > 0) {
    summary.push(`${facts.length} ${pluralize(facts.length, 'факт', 'факта', 'фактов')}`)
  }
  if (calculations.length > 0) {
    summary.push(`${calculations.length} ${pluralize(calculations.length, 'расчёт', 'расчёта', 'расчётов')}`)
  }
  if (structural.length > 0) {
    summary.push(
      `${structural.length} ${pluralize(structural.length, 'структурное изменение', 'структурных изменения', 'структурных изменений')}`,
    )
  }

  return (
    <section
      aria-label="Аналитические данные backend"
      className="mt-3 space-y-3 rounded-xl border border-line bg-raised/30 px-2.5 py-2.5"
    >
      <header>
        <p className="text-xs font-medium text-ink">
          Backend передал структурированные данные: {summary.join(', ')}.
        </p>
        <p className="mt-0.5 text-[10px] text-muted">
          Значения, проценты и CAGR приходят с backend. Интерфейс форматирует, сортирует и
          группирует строки по единицам измерения, но ничего не пересчитывает.
        </p>
      </header>

      {factGroups.length > 0 ? (
        <div className="space-y-2.5">
          <p className="panel-title">Показатели из отчётов</p>
          {factGroups.map((group) => (
            <FactsTable
              key={group.key}
              group={group}
              evidenceById={evidenceById}
              onOpenEvidence={open}
            />
          ))}
        </div>
      ) : null}

      {calculationGroups.length > 0 ? (
        <div className="space-y-2.5">
          <p className="panel-title">Расчёты backend</p>
          {calculationGroups.map((group) => {
            const withPoints = group.series.filter(hasPoints)
            const withoutPoints = group.series.filter((series) => !hasPoints(series))
            return (
              <div key={group.key} className="space-y-1.5">
                <CalculationTable group={group} />

                {group.unitSpecified && withPoints.length > 0 ? (
                  <Suspense
                    fallback={
                      <p role="status" className="text-[10px] text-muted">
                        Загрузка графика…
                      </p>
                    }
                  >
                    <CalculationCharts
                      series={withPoints}
                      unitLabel={group.unitLabel}
                      evidenceById={evidenceById}
                      onOpenEvidence={open}
                    />
                  </Suspense>
                ) : null}

                {!group.unitSpecified && withPoints.length > 0 ? (
                  <p className="text-[10px] text-warn">
                    График недоступен: backend не указал единицу измерения, а смешивать разные
                    единицы нельзя.
                  </p>
                ) : null}

                {withoutPoints.length > 0 ? (
                  <p className="text-[10px] text-muted">
                    График недоступен для {withoutPoints.map(seriesLabel).join(', ')}: backend не
                    передал `points`. Показана только таблица расчёта, ряд не дорисовывается.
                  </p>
                ) : null}
              </div>
            )
          })}
        </div>
      ) : null}

      <StructureComparison
        changes={structural}
        evidenceById={evidenceById}
        onOpenEvidence={open}
      />
    </section>
  )
}
