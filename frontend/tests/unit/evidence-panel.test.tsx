import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { EvidencePanel } from '../../src/components/evidence/EvidencePanel'
import { fullEvidence, missingEvidence, tableEvidence } from './fixtures/messages'

describe('EvidencePanel', () => {
  it('показывает полные метаданные источника', () => {
    render(
      <EvidencePanel
        evidence={[fullEvidence]}
        selectedEvidenceId={null}
        onSelect={vi.fn()}
      />,
    )
    expect(screen.getByText('Годовой отчёт 2023')).toBeInTheDocument()
    expect(screen.getByText(/2023 · ПАО Сбербанк/)).toBeInTheDocument()
    expect(screen.getByText('Ключевые показатели')).toBeInTheDocument()
    expect(screen.getByText('Стратегия / Ключевые показатели')).toBeInTheDocument()
    expect(screen.getByText('стр. 12')).toBeInTheDocument()
    expect(screen.getByText('sber_2023')).toBeInTheDocument()
    expect(screen.getByText('chunk-1')).toBeInTheDocument()
    expect(screen.getByText(/Чистая прибыль составила 1 508,6/)).toBeInTheDocument()
    expect(screen.getByText('Текст')).toBeInTheDocument()
  })

  it('не выдумывает отсутствующие поля', () => {
    render(
      <EvidencePanel
        evidence={[missingEvidence]}
        selectedEvidenceId={null}
        onSelect={vi.fn()}
      />,
    )
    const card = screen.getByTestId('evidence-card')
    expect(card).toHaveTextContent('не указано')
    expect(card).toHaveTextContent('год не указан')
    expect(card).toHaveTextContent('тип не указан')
    expect(card).toHaveTextContent('Цитата не предоставлена в снимке evidence.')
  })

  it('не показывает кнопку открытия, если полный текст недоступен', () => {
    render(
      <EvidencePanel
        evidence={[tableEvidence]}
        selectedEvidenceId={null}
        onSelect={vi.fn()}
      />,
    )
    const button = screen.getByRole('button', { name: 'Открыть источник' })
    expect(button).toBeDisabled()
    expect(screen.getByText(/has_full_text = false/)).toBeInTheDocument()
    expect(screen.getAllByText('Таблица').length).toBeGreaterThan(0)
  })

  it('показывает пустое состояние без фиктивных источников', () => {
    render(<EvidencePanel evidence={[]} selectedEvidenceId={null} onSelect={vi.fn()} />)
    expect(screen.getByText('Источники не переданы')).toBeInTheDocument()
    expect(screen.getByText(/Интерфейс не подставляет «похожие» источники/)).toBeInTheDocument()
  })

  it('во время выполнения поясняет, что источники ещё формируются', () => {
    render(
      <EvidencePanel evidence={[]} selectedEvidenceId={null} onSelect={vi.fn()} runActive />,
    )
    expect(screen.getByText(/Ответ ещё формируется/)).toBeInTheDocument()
  })
})
