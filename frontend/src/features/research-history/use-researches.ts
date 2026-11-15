import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { researchApi } from '../../services/research-api'
import type { Research } from '../../types/api'

export function invalidateResearchData(client: QueryClient, researchId?: string | null): void {
  void client.invalidateQueries({ queryKey: ['researches'] })
  if (researchId) {
    void client.invalidateQueries({ queryKey: ['research', researchId] })
    void client.invalidateQueries({ queryKey: ['messages', researchId] })
  }
}

/** Research list is always loaded from backend; search is backend-side too. */
export function useResearches(search: string) {
  return useQuery({
    queryKey: ['researches', search],
    queryFn: () => researchApi.list({ q: search || undefined, limit: 100 }),
  })
}

export function useResearch(researchId: string | null, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ['research', researchId],
    queryFn: () => researchApi.get(researchId as string),
    enabled: Boolean(researchId) && (options.enabled ?? true),
    retry: false,
  })
}

export function useCreateResearch() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (title?: string) => researchApi.create(title),
    onSuccess: (research: Research) => {
      client.setQueryData(['research', research.id], research)
      invalidateResearchData(client)
    },
  })
}

export function useRenameResearch() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ researchId, title }: { researchId: string; title: string }) =>
      researchApi.rename(researchId, title),
    onSuccess: (research: Research) => {
      client.setQueryData(['research', research.id], research)
      invalidateResearchData(client)
    },
  })
}

export function useDeleteResearch() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (researchId: string) => researchApi.remove(researchId),
    onSuccess: (_data, researchId) => {
      client.removeQueries({ queryKey: ['messages', researchId] })
      client.removeQueries({ queryKey: ['research', researchId] })
      invalidateResearchData(client)
    },
  })
}
