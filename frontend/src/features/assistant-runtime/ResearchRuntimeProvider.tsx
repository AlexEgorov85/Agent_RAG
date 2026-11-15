import type { ReactNode } from 'react'
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessageLike,
} from '@assistant-ui/react'

/**
 * assistant-ui runtime for one research.
 *
 * `LocalRuntime` + `ChatModelAdapter` is used because the backend exposes a
 * custom FastAPI SSE contract, not the Vercel AI SDK data-stream protocol.
 * The provider is keyed by research id: switching or reloading a research
 * recreates the runtime with backend history as initial messages.
 */
export function ResearchRuntimeProvider({
  adapter,
  initialMessages,
  children,
}: {
  adapter: ChatModelAdapter
  initialMessages: readonly ThreadMessageLike[]
  children: ReactNode
}) {
  const runtime = useLocalRuntime(adapter, { initialMessages })
  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}
