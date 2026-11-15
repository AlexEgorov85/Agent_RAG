import { useMemo } from 'react'
import { useInfiniteQuery, useMutation } from '@tanstack/react-query'
import { mergeMessagePages, messageApi } from '../../services/message-api'
import type { SendMessageRequest } from '../../types/api'

/** Page size for the backend history; older pages are loaded on demand. */
export const MESSAGE_PAGE_SIZE = 50

/**
 * Message history is loaded newest-page-first and merged into one ascending
 * list. `next_before` from the backend is the cursor for the previous page.
 */
export function useMessageHistory(researchId: string | null) {
  const query = useInfiniteQuery({
    queryKey: ['messages', researchId],
    queryFn: ({ pageParam }) =>
      messageApi.listPage(researchId as string, {
        limit: MESSAGE_PAGE_SIZE,
        before: pageParam ?? undefined,
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.nextBefore,
    enabled: Boolean(researchId),
  })

  const pages = query.data?.pages
  const messages = useMemo(() => mergeMessagePages(pages ?? []), [pages])

  return {
    ...query,
    messages,
    hasEarlier: Boolean(query.hasNextPage),
    isLoadingEarlier: query.isFetchingNextPage,
    loadEarlier: query.fetchNextPage,
  }
}

export function useSendMessage(researchId: string | null) {
  return useMutation({
    mutationFn: (request: SendMessageRequest) =>
      messageApi.send(researchId as string, request),
  })
}
