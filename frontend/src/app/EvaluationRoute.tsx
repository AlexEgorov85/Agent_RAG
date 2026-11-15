import { FlaskConical } from 'lucide-react'
import { ErrorState } from '../components/common/ErrorState'
import { useCapabilities } from '../features/evidence/use-evidence'
import {
  EvaluationDisabled,
  EvaluationLoading,
  EvaluationPage,
} from '../features/evaluation/EvaluationPage'
import type { EvaluationCapabilities } from '../types/evaluation'
import { navigate } from './routes'

/**
 * Evaluation — отдельный route верхнего уровня (specs/evaluation-webui.md §2).
 * Не вкладка чата: composer и пользовательские сообщения здесь отсутствуют.
 */
export function EvaluationRoute() {
  const capabilities = useCapabilities()

  if (capabilities.isLoading && !capabilities.data) return <EvaluationLoading />

  if (capabilities.error) {
    return (
      <main className="mx-auto flex min-h-full max-w-xl flex-col justify-center gap-4 px-6 py-16">
        <div className="flex items-center gap-2">
          <FlaskConical className="size-4 text-accent" aria-hidden="true" />
          <h1 className="text-lg font-semibold">Evaluation / Benchmark</h1>
        </div>
        <p className="text-sm text-muted">
          Доступность раздела определяется флагом <span className="font-mono">evaluation_ui</span> в
          ответе API. Пока API не ответил, показывать расчётные экраны нельзя.
        </p>
        <ErrorState
          error={capabilities.error}
          title="Capabilities API недоступны"
          onRetry={() => void capabilities.refetch()}
        />
        <div>
          <button type="button" className="btn" onClick={() => navigate('/')}>
            Вернуться к исследованию
          </button>
        </div>
      </main>
    )
  }

  const extended = capabilities.data as EvaluationCapabilities | undefined
  if (!extended?.evaluation_ui) return <EvaluationDisabled />

  return <EvaluationPage capabilities={extended} />
}
