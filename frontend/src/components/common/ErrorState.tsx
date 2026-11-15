import { describeApiError } from '../../lib/errors'
import { errorLabel } from '../../lib/labels'
import { ApiError } from '../../services/api-client'

export function ErrorState({
  error,
  title = 'Не удалось загрузить данные',
  onRetry,
}: {
  error: unknown
  title?: string
  onRetry?: () => void
}) {
  const code = error instanceof ApiError ? error.code : undefined
  const requestId = error instanceof ApiError ? error.requestId : undefined
  return (
    <div role="alert" className="rounded-lg border border-danger/30 bg-danger/10 p-3 text-sm">
      <p className="font-medium text-ink">{title}</p>
      <p className="mt-1 text-muted">{describeApiError(error)}</p>
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted">
        {code ? <span>Код: {errorLabel(code)}</span> : null}
        {requestId ? <span className="font-mono">request_id: {requestId}</span> : null}
        {onRetry ? (
          <button type="button" className="btn btn-ghost px-2 py-1" onClick={onRetry}>
            Повторить
          </button>
        ) : null}
      </div>
    </div>
  )
}
