import type { CoverageReport, CoverageStatus } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import { COVERAGE_STATUS_LABEL } from '../../lib/labels'

/**
 * Coverage matrix (spec §5.4). Statuses come exclusively from backend
 * `CoverageReport`; the client never computes coverage. `not_found` and
 * `year_absent` are rendered as text labels, so meaning never relies on color.
 */

const CELL_CLASS: Record<CoverageStatus, string> = {
  covered: 'text-ink',
  not_found: 'text-warn',
  year_absent: 'text-muted/80',
}

const CELL_TITLE: Record<CoverageStatus, string> = {
  covered: 'Значение подтверждено источником backend',
  not_found:
    'Год есть в корпусе, но показатель не найден поиском. Это не значит, что показателя нет в отчёте.',
  year_absent: 'Отчёта за этот год нет в корпусе. Значение не считается нулевым.',
}

function evidenceIdsFor(
  coverage: CoverageReport,
  metric: string,
  year: number,
): string[] {
  const row = coverage.cell_evidence?.[metric]
  if (!row) return []
  const ids = row[String(year)]
  return Array.isArray(ids) ? ids.filter((id): id is string => typeof id === 'string') : []
}

export function CoverageMatrix({
  coverage,
  evidenceById,
  onOpenEvidence,
}: {
  coverage: CoverageReport
  evidenceById: Map<string, EvidenceItem>
  onOpenEvidence: (evidenceId: string) => void
}) {
  const years = (coverage.expected_years.length > 0
    ? [...coverage.expected_years]
    : [...new Set(Object.values(coverage.metrics).flatMap((row) => Object.keys(row).map(Number)))]
  ).sort((a, b) => a - b)

  const metrics = Object.keys(coverage.metrics).sort((a, b) => a.localeCompare(b, 'ru'))
  if (years.length === 0 || metrics.length === 0) return null

  return (
    <div
      role="region"
      tabIndex={0}
      aria-label="Матрица покрытия доказательствами"
      className="mt-2 max-h-72 max-w-full overflow-auto rounded-lg border border-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/60"
    >
      <table className="w-full min-w-[26rem] border-collapse text-[11px]">
        <caption className="sr-only">
          Покрытие показателей по годам. Статусы приходят с backend: найдено, не найдено, год
          отсутствует в корпусе.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="sticky left-0 top-0 z-20 border-b border-line bg-raised px-2 py-1 text-left font-medium text-muted">
              Показатель
            </th>
            {years.map((year) => (
              <th
                key={year}
                scope="col"
                className="sticky top-0 z-10 border-b border-line bg-raised px-2 py-1 text-center font-medium text-muted"
              >
                {year}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric}>
              <th scope="row" className="sticky left-0 z-10 border-b border-line/60 bg-surface px-2 py-1 text-left font-normal text-ink">
                {metric}
              </th>
              {years.map((year) => {
                const status = coverage.metrics[metric]?.[String(year)]
                if (!status) {
                  return (
                    <td key={year} className="border-b border-line/60 px-2 py-1 text-center text-muted">
                      нет статуса
                    </td>
                  )
                }
                const ids = evidenceIdsFor(coverage, metric, year)
                const available = ids.filter((id) => evidenceById.has(id))
                const label = COVERAGE_STATUS_LABEL[status]
                return (
                  <td
                    key={year}
                    className={`border-b border-line/60 px-2 py-1 text-center ${CELL_CLASS[status]}`}
                  >
                    {available.length > 0 ? (
                      <button
                        type="button"
                        className="rounded px-1 underline decoration-dotted underline-offset-2 hover:bg-accent-soft/50"
                        onClick={() => onOpenEvidence(available[0] as string)}
                        aria-label={`Открыть доказательство: ${metric}, ${year} — ${label}`}
                        title={`${CELL_TITLE[status]} · evidence: ${ids.join(', ')}`}
                      >
                        {label}
                      </button>
                    ) : (
                      <span title={CELL_TITLE[status]}>{label}</span>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
