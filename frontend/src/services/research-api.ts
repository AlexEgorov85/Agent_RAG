import { apiRequest, queryString } from './api-client'
import type {
  CorpusInfo,
  HealthResponse,
  Capabilities,
  Research,
  ResearchListResponse,
} from '../types/api'
import type { EvidenceDetail } from '../types/evidence'

/** Research CRUD. Backend is the only source of truth for research history. */
export const researchApi = {
  list(params: { q?: string; limit?: number; offset?: number } = {}): Promise<ResearchListResponse> {
    return apiRequest<ResearchListResponse>(`/api/research${queryString(params)}`)
  },

  create(title?: string): Promise<Research> {
    return apiRequest<Research>('/api/research', {
      method: 'POST',
      body: JSON.stringify(title ? { title } : {}),
    })
  },

  get(researchId: string): Promise<Research> {
    return apiRequest<Research>(`/api/research/${encodeURIComponent(researchId)}`)
  },

  rename(researchId: string, title: string): Promise<Research> {
    return apiRequest<Research>(`/api/research/${encodeURIComponent(researchId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    })
  },

  async remove(researchId: string): Promise<void> {
    await apiRequest<void>(`/api/research/${encodeURIComponent(researchId)}`, { method: 'DELETE' })
  },
}

/** Read-only adapter capabilities, corpus and evidence endpoints. */
export const systemApi = {
  health(): Promise<HealthResponse> {
    return apiRequest<HealthResponse>('/api/health')
  },

  capabilities(): Promise<Capabilities> {
    return apiRequest<Capabilities>('/api/capabilities')
  },

  corpus(): Promise<CorpusInfo> {
    return apiRequest<CorpusInfo>('/api/corpus')
  },

  evidence(chunkId: string): Promise<EvidenceDetail> {
    return apiRequest<EvidenceDetail>(`/api/evidence/${encodeURIComponent(chunkId)}`)
  },
}
