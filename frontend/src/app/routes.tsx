import { useEffect, useState } from 'react'

export type AppRoute = 'workspace' | 'evaluation'

function routeFromPath(pathname: string): AppRoute {
  return pathname.startsWith('/evaluation') ? 'evaluation' : 'workspace'
}

/** Tiny history-based routing; the workspace stays the default screen. */
export function useAppRoute(): AppRoute {
  const [route, setRoute] = useState<AppRoute>(() => routeFromPath(window.location.pathname))

  useEffect(() => {
    const onPopState = () => setRoute(routeFromPath(window.location.pathname))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  return route
}

export function navigate(path: string): void {
  window.history.pushState({}, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}
