import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { ResearchSidebar } from '../../src/components/research/ResearchSidebar'
import { UiSettingsProvider } from '../../src/app/ui-settings'
import type { Research } from '../../src/types/api'

const researches: Research[] = [
  {
    id: 'research-1',
    title: 'Динамика чистой прибыли',
    created_at: '2026-11-12T09:10:00Z',
    updated_at: '2026-11-12T09:12:30Z',
    last_message_preview: 'Какая чистая прибыль за 2023 год?',
    last_status: 'completed',
    messages_count: 2,
  },
  {
    id: 'research-2',
    title: 'Структура отчёта',
    created_at: '2026-11-10T14:00:00Z',
    updated_at: '2026-11-10T14:02:00Z',
    last_status: 'failed',
    messages_count: 4,
  },
]

function renderSidebar(overrides: Partial<Parameters<typeof ResearchSidebar>[0]> = {}) {
  const props: Parameters<typeof ResearchSidebar>[0] = {
    researches,
    total: researches.length,
    search: '',
    onSearchChange: vi.fn(),
    isLoading: false,
    error: null,
    onRetry: vi.fn(),
    selectedId: 'research-1',
    onSelect: vi.fn(),
    onCreate: vi.fn(),
    isCreating: false,
    createError: undefined,
    onRename: vi.fn(async () => {}),
    onDelete: vi.fn(async () => {}),
    ...overrides,
  }
  render(
    <UiSettingsProvider>
      <ResearchSidebar {...props} />
    </UiSettingsProvider>,
  )
  return props
}

describe('ResearchSidebar', () => {
  it('показывает диалоги со статусом и выбирает нужный', async () => {
    const props = renderSidebar()
    expect(screen.getByText('Динамика чистой прибыли')).toBeInTheDocument()
    expect(screen.getByText('Готово')).toBeInTheDocument()
    expect(screen.getByText('Ошибка')).toBeInTheDocument()
    await userEvent.click(screen.getByText('Структура отчёта'))
    expect(props.onSelect).toHaveBeenCalledWith(researches[1])
  })

  it('передаёт поисковый запрос наверх (поиск выполняет backend)', async () => {
    const onSearchChange = vi.fn()
    function SearchHarness() {
      const [value, setValue] = useState('')
      return (
        <UiSettingsProvider>
          <ResearchSidebar
            researches={researches}
            total={researches.length}
            search={value}
            onSearchChange={(next) => {
              setValue(next)
              onSearchChange(next)
            }}
            isLoading={false}
            error={null}
            onRetry={vi.fn()}
            selectedId={null}
            onSelect={vi.fn()}
            onCreate={vi.fn()}
            isCreating={false}
            onRename={vi.fn(async () => {})}
            onDelete={vi.fn(async () => {})}
          />
        </UiSettingsProvider>
      )
    }
    render(<SearchHarness />)
    await userEvent.type(screen.getByLabelText('Поиск по диалогам'), 'прибыль')
    expect(onSearchChange).toHaveBeenLastCalledWith('прибыль')
  })

  it('переименовывает диалог через модальное окно', async () => {
    const onRename = vi.fn(async () => {})
    renderSidebar({ onRename })
    await userEvent.click(screen.getAllByRole('button', { name: 'Переименовать' })[0]!)
    const input = screen.getByLabelText('Название (1–200 символов)')
    await userEvent.clear(input)
    await userEvent.type(input, 'Новое название')
    await userEvent.click(screen.getByRole('button', { name: 'Сохранить' }))
    await waitFor(() =>
      expect(onRename).toHaveBeenCalledWith('research-1', 'Новое название'),
    )
  })

  it('не переименовывает в пустое название', async () => {
    const onRename = vi.fn(async () => {})
    renderSidebar({ onRename })
    await userEvent.click(screen.getAllByRole('button', { name: 'Переименовать' })[0]!)
    const input = screen.getByLabelText('Название (1–200 символов)')
    await userEvent.clear(input)
    await userEvent.click(screen.getByRole('button', { name: 'Сохранить' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Название должно быть от 1 до 200')
    expect(onRename).not.toHaveBeenCalled()
  })

  it('удаляет только после подтверждения', async () => {
    const onDelete = vi.fn(async () => {})
    renderSidebar({ onDelete })
    await userEvent.click(screen.getAllByRole('button', { name: 'Удалить' })[0]!)
    const dialog = screen.getByRole('dialog', { name: 'Удалить диалог?' })
    expect(onDelete).not.toHaveBeenCalled()
    await userEvent.click(within(dialog).getByRole('button', { name: 'Удалить' }))
    await waitFor(() => expect(onDelete).toHaveBeenCalledWith('research-1'))
  })

  it('показывает пустое состояние, когда ничего не найдено', () => {
    renderSidebar({ researches: [], total: 0, search: 'нет такого' })
    expect(screen.getByText('Ничего не найдено')).toBeInTheDocument()
  })

  it('показывает загрузку и ошибку загрузки истории', () => {
    const { unmount } = render(
      <UiSettingsProvider>
        <ResearchSidebar
          researches={[]}
          total={0}
          search=""
          onSearchChange={vi.fn()}
          isLoading
          error={null}
          onRetry={vi.fn()}
          selectedId={null}
          onSelect={vi.fn()}
          onCreate={vi.fn()}
          isCreating={false}
          onRename={vi.fn(async () => {})}
          onDelete={vi.fn(async () => {})}
        />
      </UiSettingsProvider>,
    )
    expect(screen.getByText('Загрузка диалогов')).toBeInTheDocument()
    unmount()

    render(
      <UiSettingsProvider>
        <ResearchSidebar
          researches={[]}
          total={0}
          search=""
          onSearchChange={vi.fn()}
          isLoading={false}
          error={new Error('boom')}
          onRetry={vi.fn()}
          selectedId={null}
          onSelect={vi.fn()}
          onCreate={vi.fn()}
          isCreating={false}
          onRename={vi.fn(async () => {})}
          onDelete={vi.fn(async () => {})}
        />
      </UiSettingsProvider>,
    )
    expect(screen.getByText('История недоступна')).toBeInTheDocument()
  })
})
