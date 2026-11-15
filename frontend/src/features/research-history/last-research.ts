/** The only browser-persisted research state: which research was selected. */
const STORAGE_KEY = 'agent-rag.last-research-id'

export function readLastResearchId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function writeLastResearchId(researchId: string | null): void {
  try {
    if (researchId) localStorage.setItem(STORAGE_KEY, researchId)
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Non-critical preference.
  }
}
