import type { ApiErrorBody } from '../types/api'
import { mockFetch } from '../mocks/mock-transport'

export const API_PREFIX = '/api'

/**
 * Unified API error, spec §7.8.
 * Every failure carries a code; `request_id` is surfaced when the backend sent it.
 */
export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly requestId?: string
  readonly retryable: boolean
  readonly details?: unknown

  constructor(options: {
    code: string
    message: string
    status: number
    requestId?: string
    retryable?: boolean
    details?: unknown
  }) {
    super(options.message)
    this.name = 'ApiError'
    this.code = options.code
    this.status = options.status
    this.requestId = options.requestId
    this.retryable = options.retryable ?? false
    this.details = options.details
  }

  static async fromResponse(response: Response): Promise<ApiError> {
    let body: unknown = null
    try {
      body = await response.json()
    } catch {
      body = null
    }
    const envelope = body as Partial<ApiErrorBody> | null
    const error = envelope?.error
    if (error && typeof error.code === 'string' && typeof error.message === 'string') {
      return new ApiError({
        code: error.code,
        message: error.message,
        status: response.status,
        requestId: typeof error.request_id === 'string' ? error.request_id : undefined,
        retryable: Boolean(error.retryable),
        details: error.details,
      })
    }
    return new ApiError({
      code: statusToCode(response.status),
      message: `Запрос завершился с ошибкой HTTP ${response.status}.`,
      status: response.status,
      retryable: response.status >= 500,
    })
  }

  static network(error: unknown): ApiError {
    const message =
      error instanceof Error && error.message
        ? `Не удалось связаться с API: ${error.message}`
        : 'Не удалось связаться с API.'
    return new ApiError({ code: 'network_error', message, status: 0, retryable: true })
  }
}

function statusToCode(status: number): string {
  switch (status) {
    case 400:
      return 'validation_error'
    case 404:
      return 'not_found'
    case 409:
      return 'request_in_progress'
    case 410:
      return 'request_not_replayable'
    case 413:
      return 'payload_too_large'
    case 422:
      return 'validation_error'
    case 429:
      return 'rate_limited'
    case 503:
      return 'backend_unavailable'
    default:
      return status >= 500 ? 'internal_error' : 'request_failed'
  }
}

/**
 * Mock mode is explicit and opt-in (`VITE_API_MODE=mock`). It exists so the UI
 * can be inspected without a backend; the app renders a persistent banner and
 * mock data is never presented as a real research result.
 */
export function isMockMode(): boolean {
  try {
    return import.meta.env.VITE_API_MODE === 'mock'
  } catch {
    return false
  }
}

function buildUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path
  return path.startsWith('/') ? path : `/${path}`
}

function mergeHeaders(init: RequestInit): Headers {
  const headers = new Headers(init.headers)
  if (!headers.has('Accept')) headers.set('Accept', 'application/json')
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  return headers
}

/** Raw response access for SSE and other non-JSON endpoints. */
export async function apiRequestRaw(path: string, init: RequestInit = {}): Promise<Response> {
  const url = buildUrl(path)
  if (isMockMode()) {
    return mockFetch(url, { ...init, headers: mergeHeaders(init) })
  }
  try {
    return await fetch(url, {
      ...init,
      credentials: 'include',
      headers: mergeHeaders(init),
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw ApiError.network(error)
  }
}

/** JSON request helper; throws `ApiError` for any non-2xx response. */
export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await apiRequestRaw(path, init)
  if (!response.ok) {
    throw await ApiError.fromResponse(response)
  }
  if (response.status === 204) return undefined as T
  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError({
      code: 'invalid_response',
      message: 'API вернул ответ, который не удалось разобрать.',
      status: response.status,
      retryable: true,
    })
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  )
}

export function queryString(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}
