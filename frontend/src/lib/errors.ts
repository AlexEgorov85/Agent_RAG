import { ApiError } from '../services/api-client'

/**
 * Human-readable API failure. The code and request_id are always visible:
 * spec §4.3 forbids console-only errors.
 */
export function describeApiError(error: unknown): string {
  if (error instanceof ApiError) {
    const request = error.requestId ? ` · request_id ${error.requestId}` : ''
    return `${error.message} (код: ${error.code}${request})`
  }
  if (error instanceof Error && error.message) return error.message
  return 'Неизвестная ошибка.'
}

export function errorCode(error: unknown): string | undefined {
  return error instanceof ApiError ? error.code : undefined
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 404 || error.code === 'not_found')
}
