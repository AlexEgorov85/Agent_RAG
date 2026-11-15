import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { EvidenceItem } from '../../src/types/evidence'
import {
  AssistantCitations,
  CitationNavProvider,
} from '../../src/components/chat/Citations'
import { Markdown } from '../../src/components/chat/Markdown'
import { EvidenceCard } from '../../src/components/evidence/EvidenceCard'
import { fullEvidence } from './fixtures/messages'

function renderCitations(
  items: EvidenceItem[],
  pdfState: 'unknown' | 'enabled' | 'disabled',
  onOpenEvidence = vi.fn(),
) {
  return render(
    <AssistantCitations
      items={items}
      pdfState={pdfState}
      onOpenEvidence={onOpenEvidence}
    />,
  )
}

const bps2015: EvidenceItem = {
  evidence_id: 'bps-1',
  chunk_id: 'bps-1',
  document_id: 'bps_2015',
  document_title: 'Годовой отчёт 2015',
  issuer: 'БПС-Сбербанк (Беларусь)',
  year: 2015,
  section_title: 'Основные показатели',
  section_path: ['Отчёт', 'Основные показатели'],
  page_start: 4,
  excerpt: '',
  source_type: 'text',
  has_full_text: false,
}

describe('AssistantCitations: явные ссылки на первоисточник', () => {
  it('строит PDF-ссылку с fragment страницы', () => {
    renderCitations([fullEvidence], 'enabled')

    const link = screen.getByRole('link', { name: /Открыть PDF источника 1: Годовой отчёт 2023, стр. 12/ })
    expect(link).toHaveAttribute('href', '/api/documents/sber_2023/file#page=12')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('предпочитает page_pdf из backend', () => {
    const item = { ...fullEvidence, page_pdf: 7 } as unknown as EvidenceItem
    renderCitations([item], 'enabled')

    const link = screen.getByRole('link', { name: /Открыть PDF источника 1/ })
    expect(link).toHaveAttribute('href', '/api/documents/sber_2023/file#page=7')
  })

  it('без страницы ссылается на документ без fragment', () => {
    const item: EvidenceItem = { ...fullEvidence, page_start: undefined }
    renderCitations([item], 'enabled')

    const link = screen.getByRole('link', { name: /Открыть PDF источника 1/ })
    expect(link).toHaveAttribute('href', '/api/documents/sber_2023/file')
    expect(link.getAttribute('href')).not.toContain('#page')
  })

  it('при pdf_page_links=false не выдумывает ссылку, но оставляет Evidence Panel', async () => {
    const onOpenEvidence = vi.fn()
    renderCitations([fullEvidence], 'disabled', onOpenEvidence)

    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText('PDF недоступен')).toBeInTheDocument()

    await userEvent.click(
      screen.getByRole('button', { name: 'Открыть источник 1: Годовой отчёт 2023' }),
    )
    expect(onOpenEvidence).toHaveBeenCalledWith('chunk-1')
  })

  it('показывает [n], год, issuer, документ, страницу и путь раздела', () => {
    renderCitations([bps2015], 'disabled')

    const item = screen.getByRole('listitem')
    expect(item).toHaveTextContent('[1]')
    expect(item).toHaveTextContent('2015')
    expect(item).toHaveTextContent('БПС-Сбербанк (Беларусь)')
    expect(item).toHaveTextContent('Годовой отчёт 2015')
    expect(item).toHaveTextContent('bps_2015')
    expect(item).toHaveTextContent('стр. 4')
    expect(item).toHaveTextContent('Отчёт / Основные показатели')
  })

  it('нумерует citations в порядке снимка evidence', () => {
    renderCitations([fullEvidence, bps2015], 'disabled')
    const items = screen.getAllByRole('listitem')
    expect(items[0]).toHaveTextContent('[1]')
    expect(items[1]).toHaveTextContent('[2]')
  })
})

describe('inline citation markers [n]', () => {
  function renderMarkdown(content: string, items: EvidenceItem[], onOpenEvidence = vi.fn()) {
    return render(
      <CitationNavProvider value={{ items, openEvidence: onOpenEvidence }}>
        <Markdown content={content} />
      </CitationNavProvider>,
    )
  }

  it('делает [1] кнопкой, ведущей к citation/PDF через evidence panel', async () => {
    const onOpenEvidence = vi.fn()
    renderMarkdown('Выручка выросла [1], это подтверждено отчётом.', [fullEvidence], onOpenEvidence)

    const marker = screen.getByRole('button', { name: /Перейти к источнику 1: Годовой отчёт 2023/ })
    expect(marker).toHaveTextContent('[1]')

    await userEvent.click(marker)
    expect(onOpenEvidence).toHaveBeenCalledWith('chunk-1')
  })

  it('оставляет [9] текстом, если citation с таким номером нет', () => {
    renderMarkdown('Неизвестный маркер [9].', [fullEvidence])
    expect(screen.queryByRole('button', { name: /Перейти к источнику 9/ })).toBeNull()
    expect(screen.getByText(/\[9\]/)).toBeInTheDocument()
  })

  it('не трогает обычные markdown-ссылки и код', async () => {
    const onOpenEvidence = vi.fn()
    renderMarkdown(
      'См. [сайт](https://example.com) и `[1]` в коде, а также [1].',
      [fullEvidence],
      onOpenEvidence,
    )

    const external = screen.getByRole('link', { name: 'сайт' })
    expect(external).toHaveAttribute('href', 'https://example.com')
    expect(screen.getByText('[1]', { selector: 'code' })).toBeInTheDocument()

    const markers = screen.getAllByRole('button', { name: /Перейти к источнику 1/ })
    expect(markers).toHaveLength(1)
  })
})

describe('EvidenceCard: PDF fallback', () => {
  it('disabled «Открыть PDF» с объяснением, когда endpoint недоступен', async () => {
    render(
      <EvidenceCard
        evidence={fullEvidence}
        index={0}
        expanded={false}
        onToggle={vi.fn()}
      />,
    )

    const pdfButton = screen.getByRole('button', { name: /Открыть PDF/ })
    expect(pdfButton).toBeDisabled()
    await waitFor(() =>
      expect(pdfButton).toHaveAttribute('title', expect.stringContaining('pdf_page_links')),
    )
  })
})
