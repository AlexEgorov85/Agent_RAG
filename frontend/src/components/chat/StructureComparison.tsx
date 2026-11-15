import { useMemo, useState } from 'react'
import type { StructuralChange, StructuralChangeType } from '../../types/analytics'
import type { EvidenceItem } from '../../types/evidence'
import { STRUCTURAL_CHANGE_LABEL } from '../../lib/labels'
import {
  STRUCTURAL_CHANGE_ORDER,
  filterStructuralChanges,
  sortStructuralChanges,
  structuralPath,
  structuralTitle,
} from '../../lib/analytics'
import { NOT_SPECIFIED, orNotSpecified } from '../../lib/format'
import { SortableTable, type TableColumn } from './SortableTable'

/**
 * Structural comparison of annual reports (spec §6.3).
 *
 * Only pairwise `structural_changes` returned by backend are displayed.
 * Categories the analyzer does not produce (split/merge) are never shown; an
 * unknown backend type is reported, not invented. Content comparison is a
 * separate backend capability (`structure_content_comparison=true`) and is not
 * synthesized from the section list.
 */

export function StructureComparison({
  changes,
  evidenceById,
  onOpenEvidence,
}: {
  changes: readonly StructuralChange[]
  evidenceById: Map<string, EvidenceItem>
  onOpenEvidence: (evidenceId: string) => void
}) {
  const [filter, setFilter] = useState<StructuralChangeType | 'all'>('all')
  const { supported, unsupported } = useMemo(() => filterStructuralChanges(changes), [changes])

  const presentTypes = useMemo(() => {
    const present = new Set(supported.map((change) => change.change_type))
    return STRUCTURAL_CHANGE_ORDER.filter((type) => present.has(type))
  }, [supported])

  const rows = useMemo(() => {
    const sorted = sortStructuralChanges(supported)
    return filter === 'all' ? sorted : sorted.filter((change) => change.change_type === filter)
  }, [supported, filter])

  if (changes.length === 0) {
    return (
      <section aria-label="Структурное сравнение" className="mt-2">
        <p className="panel-title">Структура отчётов</p>
        <p className="mt-1 rounded-lg border border-dashed border-line bg-raised/40 px-2.5 py-2 text-[11px] text-muted">
          Backend не передал structural_changes для этого ответа. Интерфейс не выводит «стабильные»
          разделы и не называет разделы удалёнными без данных о структуре.
        </p>
      </section>
    )
  }

  const columns: TableColumn<StructuralChange>[] = [
    {
      key: 'section',
      label: 'Раздел',
      sortable: true,
      sortValue: (change) => structuralTitle(change),
      csvValue: (change) => structuralTitle(change),
      render: (change) => {
        const fromTitle = change.from_section?.title
        const toTitle = change.to_section?.title
        const path = structuralPath(change)
        return (
          <span className="block min-w-0">
            <span className="block break-words text-ink">{structuralTitle(change)}</span>
            {fromTitle && toTitle && fromTitle !== toTitle ? (
              <span className="block text-[10px] text-muted">
                {fromTitle} → {toTitle}
              </span>
            ) : null}
            {path ? <span className="block text-[10px] text-muted">{path}</span> : null}
          </span>
        )
      },
    },
    {
      key: 'years',
      label: 'Годы',
      align: 'right',
      sortable: true,
      sortValue: (change) => change.from_year * 10000 + change.to_year,
      csvValue: (change) => `${change.from_year}–${change.to_year}`,
      render: (change) => (
        <span className="whitespace-nowrap tabular-nums">
          {change.from_year} → {change.to_year}
        </span>
      ),
    },
    {
      key: 'change',
      label: 'Изменение',
      sortable: true,
      sortValue: (change) => STRUCTURAL_CHANGE_LABEL[change.change_type] ?? change.change_type,
      csvValue: (change) => STRUCTURAL_CHANGE_LABEL[change.change_type] ?? change.change_type,
      render: (change) => (
        <span
          className={
            change.change_type === 'stable'
              ? 'text-muted'
              : change.change_type === 'removed'
                ? 'text-warn'
                : 'text-ink'
          }
        >
          {STRUCTURAL_CHANGE_LABEL[change.change_type] ?? change.change_type}
        </span>
      ),
    },
    {
      key: 'detail',
      label: 'Описание',
      csvValue: (change) => change.detail ?? '',
      render: (change) => (
        <span className="block max-w-[24rem] whitespace-pre-wrap break-words text-muted">
          {orNotSpecified(change.detail)}
        </span>
      ),
    },
    {
      key: 'document',
      label: 'Документ',
      csvValue: (change) => change.document_id ?? '',
      render: (change) => (
        <span className="block text-[10px] text-muted">
          <span className="font-mono">{orNotSpecified(change.document_id)}</span>
          {change.to_section?.page_start !== undefined || change.from_section?.page_start !== undefined ? (
            <span className="block">
              стр.{' '}
              {change.to_section?.page_start ??
                change.from_section?.page_start}
              {change.to_section?.page_end !== undefined &&
              change.to_section.page_end !== change.to_section.page_start
                ? `–${change.to_section.page_end}`
                : ''}
            </span>
          ) : (
            <span className="block">{NOT_SPECIFIED}</span>
          )}
        </span>
      ),
    },
    {
      key: 'evidence',
      label: 'Доказательство',
      csvValue: (change) => (change.evidence_chunk_ids ?? []).join(' '),
      render: (change) => {
        const ids = change.evidence_chunk_ids ?? []
        if (ids.length === 0) {
          return <span className="text-[10px] text-muted">не передан backend</span>
        }
        const available = ids.filter((id) => evidenceById.has(id))
        if (available.length === 0) {
          return (
            <span className="text-[10px] text-muted" title={ids.join(', ')}>
              вне снимка evidence
            </span>
          )
        }
        return (
          <span className="flex flex-wrap gap-1">
            {available.map((id, index) => (
              <button
                key={id}
                type="button"
                className="rounded border border-line bg-surface px-1.5 py-0.5 text-[10px] text-ink hover:border-accent/40 hover:bg-accent-soft/40"
                onClick={() => onOpenEvidence(id)}
                aria-label={`Открыть доказательство ${index + 1} для раздела ${structuralTitle(change)}`}
              >
                [{index + 1}]
              </button>
            ))}
          </span>
        )
      },
    },
  ]

  return (
    <section aria-label="Структурное сравнение" className="mt-2 space-y-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="panel-title">Структура отчётов · изменения разделов</p>
        <span role="group" aria-label="Фильтр по типу изменения" className="flex flex-wrap gap-1">
          <button
            type="button"
            className="btn btn-ghost px-2 py-0.5 text-[10px]"
            aria-pressed={filter === 'all'}
            onClick={() => setFilter('all')}
          >
            Все ({supported.length})
          </button>
          {presentTypes.map((type) => (
            <button
              key={type}
              type="button"
              className="btn btn-ghost px-2 py-0.5 text-[10px]"
              aria-pressed={filter === type}
              onClick={() => setFilter(type)}
            >
              {STRUCTURAL_CHANGE_LABEL[type]}
            </button>
          ))}
        </span>
      </div>

      <SortableTable
        caption="Структурные изменения из backend: пара лет, раздел, тип изменения, описание и доказательства."
        ariaLabel="Таблица структурных изменений"
        columns={columns}
        rows={rows}
        initialSort={{ key: 'years', direction: 'asc' }}
        csvBaseName="structural-changes"
        emptyMessage="Backend не вернул изменений выбранного типа."
      />

      <p className="text-[10px] text-muted">
        Сравнение попарное: {supported.length > 0 ? 'годы указаны в каждой строке' : 'пар нет'}.
        «Не обнаружен в более позднем отчёте» не означает, что раздел удалён из содержания.
        Сравнение содержания разделов поддерживается backend
        (capabilities.structure_content_comparison=true) и не синтезируется интерфейсом из списка
        разделов. Разделение и объединение разделов в structural_changes не передаются — интерфейс
        их не показывает.
      </p>
      {unsupported.length > 0 ? (
        <p role="status" className="text-[10px] text-warn">
          Backend вернул неподдерживаемые категории ({unsupported.join(', ')}); они не отображаются
          как изменения, чтобы не выдумывать split/merge.
        </p>
      ) : null}
    </section>
  )
}
