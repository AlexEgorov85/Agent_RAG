import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatHeader } from '../../src/components/layout/ChatHeader'
import { UiSettingsProvider } from '../../src/app/ui-settings'
import type { Research } from '../../src/types/api'

const research: Research = {
  id: 'research-1',
  title: 'Динамика прибыли',
  created_at: '2026-11-12T09:10:00Z',
  updated_at: '2026-11-12T09:12:30Z',
  last_status: 'completed',
  messages_count: 2,
}

function renderHeader(overrides: Partial<Parameters<typeof ChatHeader>[0]> = {}) {
  const props: Parameters<typeof ChatHeader>[0] = {
    research,
    corpus: {
      name: 'Корпус',
      years: [2023],
      documents: [],
      issuers: ['ПАО Сбербанк'],
      years_available: true,
    },
    filters: { years: [2023] },
    onFiltersChange: vi.fn(),
    onRename: vi.fn(async () => {}),
    sidebarCollapsed: false,
    onToggleSidebar: vi.fn(),
    onOpenMobileSidebar: vi.fn(),
    onToggleEvidence: vi.fn(),
    evidenceOpen: false,
    ...overrides,
  }
  render(
    <UiSettingsProvider>
      <ChatHeader {...props} />
    </UiSettingsProvider>,
  )
  return props
}

describe('ChatHeader (Base demo layout)', () => {
  it('показывает название диалога и disabled Share', () => {
    renderHeader()
    expect(screen.getByRole('button', { name: 'Динамика прибыли' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Поделиться' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Показать панель источников' })).not.toBeDisabled()
  })

  it('переключает sidebar и evidence panel', async () => {
    const props = renderHeader()
    await userEvent.click(screen.getByRole('button', { name: 'Скрыть список диалогов' }))
    expect(props.onToggleSidebar).toHaveBeenCalled()
    await userEvent.click(screen.getByRole('button', { name: 'Показать панель источников' }))
    expect(props.onToggleEvidence).toHaveBeenCalled()
  })

  it('переименовывает диалог по Enter', async () => {
    const onRename = vi.fn(async () => {})
    renderHeader({ onRename })
    await userEvent.click(screen.getByRole('button', { name: 'Динамика прибыли' }))
    const input = screen.getByLabelText('Название диалога')
    expect(input).toHaveValue('Динамика прибыли')
    await userEvent.clear(input)
    await userEvent.type(input, 'Новое имя{Enter}')
    await waitFor(() => expect(onRename).toHaveBeenCalledWith('Новое имя'))
  })

  it('показывает фильтры и тему', () => {
    renderHeader()
    expect(screen.getByLabelText('Фильтры запроса')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: 'Включить тёмную тему' }),
    ).toBeInTheDocument()
  })

  it('показывает ссылку Evaluation при evaluation_ui=true', async () => {
    window.history.pushState({}, '', '/')
    renderHeader({ evaluationEnabled: true })
    const link = screen.getByRole('link', { name: 'Evaluation' })
    expect(link).toHaveAttribute('href', '/evaluation')
    await userEvent.click(link)
    expect(window.location.pathname).toBe('/evaluation')
  })

  it('скрывает ссылку Evaluation при evaluation_ui=false', () => {
    renderHeader()
    expect(screen.queryByRole('link', { name: 'Evaluation' })).toBeNull()
  })
})
