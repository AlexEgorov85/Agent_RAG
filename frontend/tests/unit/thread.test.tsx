import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ChatMessage } from '../../src/types/api'
import type { ChatModelAdapter, ChatModelRunOptions, ThreadMessageLike } from '@assistant-ui/react'
import { AssistantThread, SUGGESTIONS } from '../../src/components/chat/AssistantThread'
import { ResearchRuntimeProvider } from '../../src/features/assistant-runtime/ResearchRuntimeProvider'
import { ChatUiProvider } from '../../src/features/assistant-runtime/chat-ui-context'
import { chatMessageToThreadMessageLike } from '../../src/features/assistant-runtime/chat-mapping'
import { assistantMessage, fullEvidence, userMessage } from './fixtures/messages'

function renderThread({
  messages = [],
  adapter,
  disabled = false,
  tokenStreaming = false,
  onCreateResearch = vi.fn(),
  hasEarlier = false,
  isLoadingEarlier = false,
  onLoadEarlier = vi.fn(),
}: {
  messages?: ChatMessage[]
  adapter?: ChatModelAdapter
  disabled?: boolean
  tokenStreaming?: boolean
  onCreateResearch?: () => void
  hasEarlier?: boolean
  isLoadingEarlier?: boolean
  onLoadEarlier?: () => void
} = {}) {
  const openEvidence = vi.fn()
  const initialMessages: ThreadMessageLike[] = messages.map(chatMessageToThreadMessageLike)
  const effectiveAdapter: ChatModelAdapter =
    adapter ??
    {
      async run() {
        return { content: [{ type: 'text', text: 'Ответ агента' }] }
      },
    }

  const view = render(
    <ResearchRuntimeProvider adapter={effectiveAdapter} initialMessages={initialMessages}>
      <ChatUiProvider value={{ tokenStreaming, openEvidence }}>
        <AssistantThread
          modelLabel="GigaChat"
          disabled={disabled}
          onCreateResearch={onCreateResearch}
          hasEarlier={hasEarlier}
          isLoadingEarlier={isLoadingEarlier}
          onLoadEarlier={onLoadEarlier}
        />
      </ChatUiProvider>
    </ResearchRuntimeProvider>,
  )
  return { ...view, openEvidence, onCreateResearch, onLoadEarlier }
}

describe('AssistantThread: Base-demo empty state and suggestions', () => {
  it('показывает welcome, composer и chips проекта', async () => {
    renderThread()
    expect(await screen.findByText('Чем я могу помочь?')).toBeInTheDocument()
    expect(screen.getByLabelText('Вопрос к агенту')).toBeInTheDocument()
    for (const suggestion of SUGGESTIONS) {
      expect(screen.getByRole('button', { name: suggestion.label })).toBeInTheDocument()
    }
    expect(screen.getByText('GigaChat')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Добавить вложение' })).toBeDisabled()
  })

  it('chip отправляет готовый вопрос через runtime', async () => {
    const run = vi.fn(async (_options: ChatModelRunOptions) => ({
      content: [{ type: 'text' as const, text: 'Ответ' }],
    }))
    renderThread({ adapter: { run } })
    const suggestion = SUGGESTIONS[1]!
    await userEvent.click(await screen.findByRole('button', { name: suggestion.label }))
    await waitFor(() => expect(run).toHaveBeenCalled())
    const firstOptions = run.mock.calls[0]?.[0] as unknown as ChatModelRunOptions
    const lastUser = [...firstOptions.messages].reverse().find((message) => message.role === 'user')
    const text = lastUser?.content.flatMap((part) => (part.type === 'text' ? [part.text] : [])).join('')
    expect(text).toBe(suggestion.prompt)
  })

  it('в пустом диалоге composer отправляет по Enter', async () => {
    const run = vi.fn(async () => ({ content: [{ type: 'text' as const, text: 'Ответ' }] }))
    renderThread({ adapter: { run } })
    const input = await screen.findByLabelText('Вопрос к агенту')
    await userEvent.type(input, 'Какая прибыль за 2024 год?')
    await userEvent.keyboard('{Enter}')
    await waitFor(() => expect(run).toHaveBeenCalled())
    expect(await screen.findByText('Какая прибыль за 2024 год?')).toBeInTheDocument()
  })

  it('Shift+Enter оставляет перевод строки и не отправляет', async () => {
    const run = vi.fn(async () => ({ content: [{ type: 'text' as const, text: 'Ответ' }] }))
    renderThread({ adapter: { run } })
    const input = (await screen.findByLabelText('Вопрос к агенту')) as HTMLTextAreaElement
    await userEvent.type(input, 'первая{Shift>}{Enter}{/Shift}вторая')
    expect(run).not.toHaveBeenCalled()
    expect(input.value).toContain('первая\nвторая')
  })

  it('во время выполнения показывает кнопку остановки', async () => {
    let aborted = false
    const run = vi.fn(async (options: ChatModelRunOptions) => {
      await new Promise<void>((resolve) => {
        options.abortSignal.addEventListener('abort', () => {
          aborted = true
          resolve()
        })
      })
      return { content: [{ type: 'text' as const, text: '' }] }
    })
    renderThread({ adapter: { run } })
    const input = await screen.findByLabelText('Вопрос к агенту')
    await userEvent.type(input, 'Долгий вопрос{Enter}')
    const stop = await screen.findByRole('button', { name: 'Остановить выполнение' })
    await userEvent.click(stop)
    await waitFor(() => expect(aborted).toBe(true))
  })

  it('без диалога предлагает создать его вместо composer', async () => {
    const onCreateResearch = vi.fn()
    renderThread({ disabled: true, onCreateResearch })
    expect(await screen.findByText('Чем я могу помочь?')).toBeInTheDocument()
    expect(screen.queryByLabelText('Вопрос к агенту')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Создать диалог' }))
    expect(onCreateResearch).toHaveBeenCalled()
  })
})

describe('AssistantThread: a11y and history pagination', () => {
  it('в непустом диалоге есть h1, а в пустом — приветствие', async () => {
    const { unmount } = renderThread()
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Чем я могу помочь?' }),
    ).toBeInTheDocument()
    unmount()

    renderThread({ messages: [userMessage(), assistantMessage({ evidence: [] })] })
    expect(
      await screen.findByRole('heading', { level: 1, name: 'Диалог с агентом' }),
    ).toBeInTheDocument()
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
  })

  it('кнопка «загрузить раньше» появляется только при наличии next_before', async () => {
    const { unmount } = renderThread({
      messages: [userMessage(), assistantMessage({ evidence: [] })],
    })
    expect(
      screen.queryByRole('button', { name: 'Загрузить более ранние сообщения' }),
    ).not.toBeInTheDocument()
    unmount()

    const onLoadEarlier = vi.fn()
    renderThread({
      messages: [userMessage(), assistantMessage({ evidence: [] })],
      hasEarlier: true,
      onLoadEarlier,
    })
    await userEvent.click(
      await screen.findByRole('button', { name: 'Загрузить более ранние сообщения' }),
    )
    expect(onLoadEarlier).toHaveBeenCalledTimes(1)
  })

  it('во время подгрузки кнопка disabled и честно сообщает о загрузке', async () => {
    renderThread({
      messages: [userMessage(), assistantMessage({ evidence: [] })],
      hasEarlier: true,
      isLoadingEarlier: true,
    })
    const button = await screen.findByRole('button', {
      name: 'Загрузка более ранних сообщений…',
    })
    expect(button).toBeDisabled()
  })

  it('объявляет завершение потокового ответа через aria-live', async () => {
    const run = vi.fn(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 10)
      })
      return { content: [{ type: 'text' as const, text: 'Готово' }] }
    })
    renderThread({ adapter: { run } })
    const input = await screen.findByLabelText('Вопрос к агенту')
    await userEvent.type(input, 'Вопрос{Enter}')
    const announcement = await screen.findByText('Ответ сформирован.')
    expect(announcement).toHaveAttribute('role', 'status')
    expect(announcement).toHaveAttribute('aria-live', 'polite')
  })
})

describe('AssistantThread: message rendering', () => {
  it('рендерит историю, markdown и безопасные ссылки', async () => {
    renderThread({
      messages: [
        userMessage(),
        assistantMessage({
          content:
            '**Прибыль** выросла.\n\n| Год | Значение |\n| --- | --- |\n| 2023 | 1 508,6 |\n\n[отчёт](https://example.com/report)',
          evidence: [],
        }),
      ],
    })
    expect(await screen.findByText(/Какая чистая прибыль/)).toBeInTheDocument()
    expect(screen.getByText('Прибыль')).toBeInTheDocument()
    expect(screen.getByRole('table')).toBeInTheDocument()
    const link = screen.getByRole('link', { name: 'отчёт' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('не исполняет raw HTML из ответа', async () => {
    const { container } = renderThread({
      messages: [
        assistantMessage({
          content: 'Текст\n\n<script>alert(1)</script>\n\n<img src="x" onerror="alert(2)">',
          evidence: [],
        }),
      ],
    })
    await screen.findByText('Текст')
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img[onerror]')).toBeNull()
  })

  it('клик по цитате открывает evidence panel с точным снимком', async () => {
    const message = assistantMessage({ evidence: [fullEvidence] })
    const { openEvidence } = renderThread({ messages: [message] })
    const citation = await screen.findByRole('button', {
      name: /Открыть источник 1: Годовой отчёт 2023/,
    })
    await userEvent.click(citation)
    expect(openEvidence).toHaveBeenCalledWith([fullEvidence], 'chunk-1')
  })

  it('показывает предупреждения, анализ и покрытие как данные backend', async () => {
    renderThread({
      messages: [
        assistantMessage({
          warnings: ['no_evidence_for_period'],
          analysis: {
            facts: [{ metric: 'profit', year: 2023, value: 1508.6, chunk_id: 'chunk-1' }],
            coverage: {
              status: 'partial',
              expected_years: [2022, 2023],
              covered_years: [2023],
              missing_years: [2022],
              metrics: {},
              missing_metrics: [],
              notes: [],
            },
          },
        }),
      ],
    })
    expect(await screen.findByText(/доказательства не найдены/)).toBeInTheDocument()
    expect(screen.getByText(/Backend передал структурированные данные/)).toBeInTheDocument()
    expect(screen.getByText(/Покрытие доказательствами backend: покрытие неполное/)).toBeInTheDocument()
    expect(screen.getByText(/Годы без покрытия: 2022/)).toBeInTheDocument()
  })

  it('переводит известные предупреждения и не дублирует непереведённый код', async () => {
    renderThread({
      messages: [assistantMessage({ warnings: ['no_evidence', 'custom_warning'] })],
    })
    const list = await screen.findByRole('list', { name: 'Предупреждения' })
    expect(list).toHaveTextContent(
      'В доступных материалах не найдено подтверждение для ответа.',
    )
    expect(list.textContent?.match(/custom_warning/g)).toHaveLength(1)
  })

  it('ошибка backend — inline alert с кодом и request_id', async () => {
    renderThread({
      messages: [
        assistantMessage({
          status: 'failed',
          content: '',
          evidence: [],
          error: {
            code: 'backend_unavailable',
            message: 'AgentService недоступен',
            request_id: 'req-42',
            retryable: true,
          },
        }),
      ],
    })
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('backend_unavailable')
    expect(alert).toHaveTextContent('request_id: req-42')
    expect(screen.getByText(/Выполнение завершилось ошибкой backend/)).toBeInTheDocument()
  })

  it('abstained и cancelled показываются честно, без spinner', async () => {
    const { unmount } = renderThread({
      messages: [assistantMessage({ status: 'abstained', content: '', evidence: [] })],
    })
    expect(await screen.findByText(/не сформировал утвердительный ответ/)).toBeInTheDocument()
    expect(screen.queryByText('Агент готовит ответ…')).not.toBeInTheDocument()
    unmount()

    renderThread({
      messages: [assistantMessage({ status: 'cancelled', content: '', evidence: [] })],
    })
    expect(await screen.findByText(/Выполнение отменено/)).toBeInTheDocument()
    expect(screen.queryByText('Агент готовит ответ…')).not.toBeInTheDocument()
  })

  it('action bar содержит Copy, Reload и Export', async () => {
    renderThread({ messages: [assistantMessage({ evidence: [] })] })
    expect(await screen.findByRole('button', { name: 'Скопировать ответ' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Повторить ответ' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Экспорт ответа в Markdown' })).toBeInTheDocument()
  })
})
