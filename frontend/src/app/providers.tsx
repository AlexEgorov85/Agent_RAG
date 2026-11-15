import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { UiSettingsProvider } from './ui-settings'

/** Server data cache. Streaming buffers live outside this client (spec §8). */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 0,
      refetchOnWindowFocus: false,
      staleTime: 10_000,
    },
  },
})

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <UiSettingsProvider>{children}</UiSettingsProvider>
    </QueryClientProvider>
  )
}
