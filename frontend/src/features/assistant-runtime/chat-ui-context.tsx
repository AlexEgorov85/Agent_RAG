import { createContext, useContext } from 'react'
import type { EvidenceItem } from '../../types/evidence'

export interface ChatUiContextValue {
  /** `capabilities.token_streaming`; controls the honest "no token stream" hint. */
  tokenStreaming: boolean
  /** Opens the evidence panel with an exact snapshot received from backend. */
  openEvidence: (items: EvidenceItem[], evidenceId: string) => void
}

const ChatUiContext = createContext<ChatUiContextValue | null>(null)

export const ChatUiProvider = ChatUiContext.Provider

export function useChatUi(): ChatUiContextValue {
  const value = useContext(ChatUiContext)
  if (!value) throw new Error('useChatUi должен использоваться внутри ChatUiProvider')
  return value
}
