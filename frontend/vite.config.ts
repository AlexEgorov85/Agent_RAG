import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vite dev server proxies /api to the FastAPI adapter (src/api/app.py).
// In production the SPA is served same-origin, so /api stays relative.
//
// The proxy target and port can be overridden from the environment; E2E uses
// this to run against a scripted fixture on non-default ports while a real
// backend keeps the default 8000 free.
const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000'
const devPort = Number(process.env.VITE_PORT ?? 5173)

export default defineConfig({
  plugins: [react()],
  server: {
    port: devPort,
    // With an explicit port (E2E) fail loudly instead of silently moving,
    // because the test baseURL points to this exact port.
    strictPort: Boolean(process.env.VITE_PORT),
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: false,
        // SSE must not be buffered by the proxy.
        ws: false,
      },
    },
  },
  preview: {
    port: 4173,
  },
  build: {
    rollupOptions: {
      output: {
        // Keep the >500 kB single bundle warning away: React, assistant-ui and
        // the markdown pipeline change rarely and cache independently.
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined
          // assistant-ui is the largest runtime dependency: keep its React
          // bindings, the core/store layer and the stream protocol in separate
          // chunks so no single vendor file crosses the 500 kB warning limit.
          if (id.includes('node_modules/@assistant-ui/react/')) return 'vendor-assistant-ui'
          if (id.includes('node_modules/@assistant-ui/')) return 'vendor-assistant-core'
          if (
            id.includes('node_modules/assistant-stream/') ||
            id.includes('node_modules/assistant-cloud/')
          ) {
            return 'vendor-assistant-stream'
          }
          if (
            id.includes('node_modules/radix-ui/') ||
            id.includes('node_modules/@radix-ui/') ||
            id.includes('node_modules/zustand/') ||
            id.includes('node_modules/zod/')
          ) {
            return 'vendor-ui'
          }
          if (
            id.includes('node_modules/react-markdown/') ||
            id.includes('node_modules/remark-') ||
            id.includes('node_modules/rehype-') ||
            id.includes('node_modules/unified/') ||
            id.includes('node_modules/micromark') ||
            id.includes('node_modules/mdast-') ||
            id.includes('node_modules/hast-') ||
            id.includes('node_modules/unist-') ||
            id.includes('node_modules/vfile')
          ) {
            return 'vendor-markdown'
          }
          if (
            id.includes('node_modules/react/') ||
            id.includes('node_modules/react-dom/') ||
            id.includes('node_modules/scheduler/')
          ) {
            return 'vendor-react'
          }
          return undefined
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    css: false,
    restoreMocks: true,
  },
})
