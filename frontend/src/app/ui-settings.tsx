import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'light' | 'dark'

export interface UiSettings {
  theme: Theme
  sidebarCollapsed: boolean
}

interface UiSettingsContextValue extends UiSettings {
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  toggleSidebar: () => void
}

const STORAGE_KEY = 'agent-rag.ui-settings.v1'

const DEFAULT_SETTINGS: UiSettings = { theme: 'light', sidebarCollapsed: false }

function readSettings(): UiSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_SETTINGS
    const parsed = JSON.parse(raw) as Partial<UiSettings>
    return {
      theme: parsed.theme === 'dark' ? 'dark' : 'light',
      sidebarCollapsed: parsed.sidebarCollapsed === true,
    }
  } catch {
    return DEFAULT_SETTINGS
  }
}

/** UI-only preferences: never server data, never chat messages. */
const UiSettingsContext = createContext<UiSettingsContextValue | null>(null)

export function UiSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<UiSettings>(readSettings)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
    } catch {
      // Storage may be unavailable (private mode); preferences are not critical.
    }
  }, [settings])

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', settings.theme === 'dark')
    root.style.colorScheme = settings.theme
  }, [settings.theme])

  const setTheme = useCallback((theme: Theme) => {
    setSettings((current) => ({ ...current, theme }))
  }, [])

  const toggleTheme = useCallback(() => {
    setSettings((current) => ({ ...current, theme: current.theme === 'dark' ? 'light' : 'dark' }))
  }, [])

  const toggleSidebar = useCallback(() => {
    setSettings((current) => ({ ...current, sidebarCollapsed: !current.sidebarCollapsed }))
  }, [])

  const value = useMemo<UiSettingsContextValue>(
    () => ({ ...settings, setTheme, toggleTheme, toggleSidebar }),
    [settings, setTheme, toggleTheme, toggleSidebar],
  )

  return <UiSettingsContext.Provider value={value}>{children}</UiSettingsContext.Provider>
}

export function useUiSettings(): UiSettingsContextValue {
  const context = useContext(UiSettingsContext)
  if (!context) throw new Error('useUiSettings должен использоваться внутри UiSettingsProvider')
  return context
}
