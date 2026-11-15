import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { StructuralChange } from '../../src/types/analytics'
import type { EvidenceItem } from '../../src/types/evidence'
import { StructureComparison } from '../../src/components/chat/StructureComparison'

function change(overrides: Partial<StructuralChange> = {}): StructuralChange {
  return {
    from_year: 2022,
    to_year: 2023,
    from_section: { title: 'Управление рисками', section_path: 'Отчёт / Риски', level: 1 },
    to_section: { title: 'Управление рисками', section_path: 'Отчёт / Риски', level: 1 },
    change_type: 'stable',
    ...overrides,
  }
}

const evidenceItem: EvidenceItem = {
  evidence_id: 'chunk-1',
  chunk_id: 'chunk-1',
  document_id: 'sber_2023',
  excerpt: '',
  has_full_text: true,
}

function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

describe('StructureComparison', () => {
  it('показывает только реально возвращённые изменения и пары лет', () => {
    render(
      <StructureComparison
        changes={[
          change({ change_type: 'stable' }),
          change({ change_type: 'added', to_section: { title: 'ESG' }, to_year: 2023 }),
          change({
            change_type: 'removed',
            to_year: 2023,
            to_section: undefined,
            detail: 'заголовок не найден в отчёте 2023',
          }),
        ]}
        evidenceById={new Map()}
        onOpenEvidence={vi.fn()}
      />,
    )

    const table = screen.getByRole('table')
    expect(within(table).getByText('ESG')).toBeInTheDocument()
    expect(within(table).getByText('Добавлен')).toBeInTheDocument()
    expect(
      within(table).getByText('Не обнаружен в более позднем отчёте'),
    ).toBeInTheDocument()
    expect(within(table).getByText('заголовок не найден в отчёте 2023')).toBeInTheDocument()
    expect(within(table).getAllByText('2022 → 2023').length).toBeGreaterThan(0)
    expect(screen.getByText(/попарное/)).toBeInTheDocument()
  })

  it('не показывает split/merge, даже если backend их прислал', () => {
    render(
      <StructureComparison
        changes={[
          change({ change_type: 'split' as StructuralChange['change_type'] }),
          change({ change_type: 'merge' as StructuralChange['change_type'] }),
          change({ change_type: 'added', to_section: { title: 'ESG' } }),
        ]}
        evidenceById={new Map()}
        onOpenEvidence={vi.fn()}
      />,
    )

    expect(screen.queryByText(/^Разделение$/)).toBeNull()
    expect(screen.queryByText(/^Объединение$/)).toBeNull()
    expect(screen.getByText('ESG')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/merge, split/)
  })

  it('фильтрует по типу изменения', async () => {
    render(
      <StructureComparison
        changes={[
          change({ change_type: 'stable' }),
          change({ change_type: 'renamed', to_section: { title: 'Стратегия' } }),
        ]}
        evidenceById={new Map()}
        onOpenEvidence={vi.fn()}
      />,
    )

    const table = screen.getByRole('table')
    expect(within(table).getByText('Управление рисками')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Переименован' }))
    expect(screen.getByRole('button', { name: 'Переименован' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(within(table).queryByText('Управление рисками')).toBeNull()
    expect(within(table).getByText('Стратегия')).toBeInTheDocument()
  })

  it('ведёт к доказательству только при наличии chunk в снимке', async () => {
    const onOpenEvidence = vi.fn()
    render(
      <StructureComparison
        changes={[
          change({
            change_type: 'renamed',
            evidence_chunk_ids: ['chunk-1', 'chunk-missing'],
            document_id: 'sber_2023',
          }),
          change({ change_type: 'added', to_section: { title: 'ESG' } }),
        ]}
        evidenceById={new Map([['chunk-1', evidenceItem]])}
        onOpenEvidence={onOpenEvidence}
      />,
    )

    await userEvent.click(
      screen.getByRole('button', {
        name: 'Открыть доказательство 1 для раздела Управление рисками',
      }),
    )
    expect(onOpenEvidence).toHaveBeenCalledWith('chunk-1')
    expect(screen.getAllByText('не передан backend').length).toBeGreaterThan(0)
  })

  it('показывает честное пустое состояние без structural_changes', () => {
    render(
      <StructureComparison changes={[]} evidenceById={new Map()} onOpenEvidence={vi.fn()} />,
    )
    expect(
      screen.getByText(/Backend не передал structural_changes для этого ответа/),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table')).toBeNull()
  })

  it('экспортирует структурные изменения в CSV с годами и типом', async () => {
    const blobs: Blob[] = []
    const originalCreate = URL.createObjectURL
    const originalRevoke = URL.revokeObjectURL
    URL.createObjectURL = ((blob: Blob) => {
      blobs.push(blob)
      return 'blob:structure'
    }) as typeof URL.createObjectURL
    URL.revokeObjectURL = (() => {}) as typeof URL.revokeObjectURL
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    try {
      render(
        <StructureComparison
          changes={[change({ change_type: 'added', to_section: { title: 'ESG' } })]}
          evidenceById={new Map()}
          onOpenEvidence={vi.fn()}
        />,
      )

      await userEvent.click(screen.getByRole('button', { name: 'Экспорт CSV' }))
      const csv = await readBlob(blobs[0]!)
      expect(csv).toContain('Раздел,Годы,Изменение,Описание,Документ,Доказательство')
      expect(csv).toContain('ESG,2022–2023,Добавлен')
    } finally {
      URL.createObjectURL = originalCreate
      URL.revokeObjectURL = originalRevoke
    }
  })
})
