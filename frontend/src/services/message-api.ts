import { apiRequest, queryString } from './api-client'
import type {
  ChatMessage,
  SendMessageRequest,
  SendMessageResponse,
  RequestStatusResponse,
} from '../types/api'

/** One page of `GET /api/research/{id}/messages`, newest page first. */
export interface MessagePage {
  items: ChatMessage[]
  /** Cursor for the page before `items`; `null` means there is no earlier page. */
  nextBefore: string | null
}

/**
 * The endpoint returns `{items, next_before}`; a bare array is still accepted
 * for older mocks. `next_before` is the id of the oldest item in the page.
 */
export function normalizeMessagePage(payload: unknown): MessagePage {
  if (Array.isArray(payload)) {
    return { items: payload as ChatMessage[], nextBefore: null }
  }
  const envelope = payload as { items?: unknown; next_before?: unknown } | null
  const items = Array.isArray(envelope?.items) ? (envelope.items as ChatMessage[]) : []
  const nextBefore =
    typeof envelope?.next_before === 'string' && envelope.next_before
      ? envelope.next_before
      : null
  return { items, nextBefore }
}

/** Backwards-compatible array view of a message page. */
export function normalizeMessageList(payload: unknown): ChatMessage[] {
  return normalizeMessagePage(payload).items
}

/**
 * Merge pages loaded newest-first into one ascending history.
 *
 * Within a page the backend returns messages oldest-first; earlier pages are
 * prepended as a block. Duplicate ids (a run settling while the user pages)
 * are dropped so the thread never renders the same message twice.
 */
export function mergeMessagePages(pages: readonly MessagePage[]): ChatMessage[] {
  const seen = new Set<string>()
  const ordered: ChatMessage[] = []
  for (let index = pages.length - 1; index >= 0; index -= 1) {
    const page = pages[index]
    if (!page) continue
    for (const message of page.items) {
      if (seen.has(message.id)) continue
      seen.add(message.id)
      ordered.push(message)
    }
  }
  return ordered
}

export const messageApi = {
  async listPage(
    researchId: string,
    params: { limit?: number; before?: string } = {},
  ): Promise<MessagePage> {
    const payload = await apiRequest<unknown>(
      `/api/research/${encodeURIComponent(researchId)}/messages${queryString(params)}`,
    )
    return normalizeMessagePage(payload)
  },

  async list(
    researchId: string,
    params: { limit?: number; before?: string } = {},
  ): Promise<ChatMessage[]> {
    return (await messageApi.listPage(researchId, params)).items
  },

  send(researchId: string, request: SendMessageRequest): Promise<SendMessageResponse> {
    return apiRequest<SendMessageResponse>(
      `/api/research/${encodeURIComponent(researchId)}/messages`,
      { method: 'POST', body: JSON.stringify(request) },
    )
  },

  requestStatus(requestId: string): Promise<RequestStatusResponse> {
    return apiRequest<RequestStatusResponse>(`/api/requests/${encodeURIComponent(requestId)}`)
  },

  async cancel(requestId: string): Promise<void> {
    await apiRequest<void>(`/api/requests/${encodeURIComponent(requestId)}/cancel`, {
      method: 'POST',
    })
  },
}
