import { defineConfig, devices } from '@playwright/test'

/**
 * E2E runs against a scripted fixture backend; it never calls the Python core.
 *
 * The default development port 8000 may already be occupied by a real FastAPI
 * process, and `reuseExistingServer` would silently accept it. E2E therefore
 * uses dedicated ports and env overrides:
 *
 *   E2E_API_PORT (default 8010) — fixture backend
 *   E2E_UI_PORT  (default 5273) — Vite dev server
 *
 * `VITE_API_PROXY_TARGET` is passed to Vite so the SPA's /api goes to the
 * fixture, never to the process on 8000.
 */
const apiPort = Number(process.env.E2E_API_PORT ?? 8010)
const uiPort = Number(process.env.E2E_UI_PORT ?? 5273)
const apiOrigin = `http://127.0.0.1:${apiPort}`
const uiOrigin = `http://127.0.0.1:${uiPort}`

const useExistingServers = process.env.E2E_REUSE_SERVERS === '1'

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: '**/*.spec.ts',
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: uiOrigin,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: 'node tests/e2e/fixture-server.mjs',
      // Never reuse an unknown process: a real API on this port would make the
      // tests pass against the wrong data or fail with an empty database.
      env: { ...process.env, FIXTURE_PORT: String(apiPort) },
      url: `${apiOrigin}/api/health`,
      reuseExistingServer: useExistingServers,
      timeout: 30_000,
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${uiPort} --strictPort`,
      env: {
        ...process.env,
        VITE_PORT: String(uiPort),
        VITE_API_PROXY_TARGET: apiOrigin,
      },
      url: uiOrigin,
      reuseExistingServer: useExistingServers,
      timeout: 60_000,
    },
  ],
})
