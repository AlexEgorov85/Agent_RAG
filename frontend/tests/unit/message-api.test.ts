import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  mergeMessagePages,
  messageApi,
  normalizeMessageList,
  normalizeMessagePage,
  type MessagePage,
} from '../../src/services/message-api'
import type { ChatMessage } from '../../src/types/api'

afterEach(() => {
  vi.unstubAllGlobals()
})

function message(id: string, role: ChatMessage['role'] = 'user'): ChatMessage {
  return {
    id,
    research_id: 'research-1',
    role,
    content: `text ${id}`,
    created_at: '2026-11-14T10:00:00Z',
    status: 'completed',
    evidence: [],
    warnings: [],
  }
}

describe('message page envelope', () => {
  it('читает next_before как курсор предыдущей страницы', () => {
    const page = normalizeMessagePage({
      items: [message('m3'), message('m4')],
      next_before: 'm3',
    })
    expect(page.items.map((item) => item.id)).toEqual(['m3', 'm4'])
    expect(page.nextBefore).toBe('m3')
  })

  it('последняя страница не имеет курсора', () => {
    expect(normalizeMessagePage({ items: [], next_before: null }).nextBefore).toBeNull()
    expect(normalizeMessagePage({ items: [], next_before: '' }).nextBefore).toBeNull()
  })

  it('принимает bare array как одну страницу без продолжения', () => {
    const page = normalizeMessagePage([message('m1')])
    expect(page.items).toHaveLength(1)
    expect(page.nextBefore).toBeNull()
    expect(normalizeMessageList([message('m1')])).toHaveLength(1)
  })

  it('запрашивает предыдущую страницу по курсору before', async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => ({
      ok: true,
      status: 200,
      json: async () => ({ items: [message('m2')], next_before: 'm2' }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    const page = await messageApi.listPage('research-1', { limit: 2, before: 'm3' })
    const url = String(fetchMock.mock.calls[0]?.[0])
    expect(url).toBe('/api/research/research-1/messages?limit=2&before=m3')
    expect(page.items.map((item) => item.id)).toEqual(['m2'])
    expect(page.nextBefore).toBe('m2')
  })
})

describe('mergeMessagePages', () => {
  it('склеивает страницы newest-first в возрастающую историю', () => {
    const pages: MessagePage[] = [
      { items: [message('m3'), message('m4')], nextBefore: 'm3' },
      { items: [message('m1'), message('m2')], nextBefore: null },
    ]
    expect(mergeMessagePages(pages).map((item) => item.id)).toEqual([
      'm1',
      'm2',
      'm3',
      'm4',
    ])
  })

  it('не дублирует сообщения, попавшие в две страницы', () => {
    const pages: MessagePage[] = [
      { items: [message('m3'), message('m4')], nextBefore: 'm3' },
      { items: [message('m1'), message('m2'), message('m3')], nextBefore: null },
    ]
    const merged = mergeMessagePages(pages)
    expect(merged.map((item) => item.id)).toEqual(['m1', 'm2', 'm3', 'm4'])
    expect(new Set(merged.map((item) => item.id)).size).toBe(merged.length)
  })

  it('пустая история остаётся пустой', () => {
    expect(mergeMessagePages([])).toEqual([])
  })
})
