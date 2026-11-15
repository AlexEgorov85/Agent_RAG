/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** `mock` enables the explicit in-browser fixture backend. */
  readonly VITE_API_MODE?: 'mock' | 'live'
}
