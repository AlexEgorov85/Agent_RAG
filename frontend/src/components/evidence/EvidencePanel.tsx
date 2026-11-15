import { useEffect, useRef } from 'react'
import type { EvidenceItem } from '../../types/evidence'
import { pluralize } from '../../lib/format'
import { EmptyState } from '../common/EmptyState'
import { EvidenceCard } from './EvidenceCard'

export function EvidencePanel({
  evidence,
  selectedEvidenceId,
  onSelect,
  runActive = false,
}: {
  evidence: EvidenceItem[]
  selectedEvidenceId: string | null
  onSelect: (evidenceId: string | null) => void
  runActive?: boolean
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const missingFullText = evidence.filter((item) => !item.has_full_text).length

  useEffect(() => {
    if (!selectedEvidenceId) return
    const element = containerRef.current?.querySelector(
      `[data-evidence-id="${selectedEvidenceId.replace(/"/g, '')}"]`,
    )
    if (element && typeof element.scrollIntoView === 'function') {
      element.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedEvidenceId])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-line px-3 py-1.5">
        <p className="text-[11px] text-muted">
          {evidence.length > 0
            ? `${evidence.length} ${pluralize(evidence.length, 'источник', 'источника', 'источников')}`
            : 'нет переданных источников'}
          {missingFullText > 0 ? ` · без полного текста: ${missingFullText}` : ''}
        </p>
      </div>

      <div ref={containerRef} className="scroll-area min-h-0 flex-1 px-2 py-2">
        {evidence.length === 0 ? (
          <EmptyState
            title="Источники не переданы"
            description={
              runActive
                ? 'Ответ ещё формируется. Панель обновится по событиям backend.'
                : 'В этом сообщении backend не передал evidence. Интерфейс не подставляет «похожие» источники.'
            }
          />
        ) : (
          <ul className="space-y-2">
            {evidence.map((item, index) => (
              <EvidenceCard
                key={item.evidence_id}
                evidence={item}
                index={index}
                expanded={selectedEvidenceId === item.evidence_id}
                onToggle={() =>
                  onSelect(selectedEvidenceId === item.evidence_id ? null : item.evidence_id)
                }
              />
            ))}
          </ul>
        )}
      </div>

      <footer className="border-t border-line px-3 py-2 text-[11px] text-muted">
        Отсутствующие поля показываются как «не указано». Год, issuer, страницы и название документа
        берутся только из ответа backend.
      </footer>
    </div>
  )
}
