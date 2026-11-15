import { useQuery } from '@tanstack/react-query'
import { systemApi } from '../../services/research-api'

/** Full chunk text is loaded on demand and never stored in message history. */
export function useEvidenceDetail(chunkId: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ['evidence', chunkId],
    queryFn: () => systemApi.evidence(chunkId as string),
    enabled: enabled && Boolean(chunkId),
    retry: 1,
    staleTime: 5 * 60_000,
  })
}

export function useCapabilities() {
  return useQuery({
    queryKey: ['capabilities'],
    queryFn: () => systemApi.capabilities(),
    staleTime: 60_000,
  })
}

export function useCorpus() {
  return useQuery({
    queryKey: ['corpus'],
    queryFn: () => systemApi.corpus(),
    staleTime: 5 * 60_000,
  })
}
