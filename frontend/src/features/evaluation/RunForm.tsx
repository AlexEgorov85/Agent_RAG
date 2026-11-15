import { useMemo, useState } from 'react'
import { Play } from 'lucide-react'
import { Banner } from '../../components/common/Banner'
import { describeApiError } from '../../lib/errors'
import { newClientMessageId } from '../../lib/ids'
import type {
  EvaluationCapabilities,
  EvaluationRunCreateRequest,
  EvaluationRunCreated,
  EvaluationRunListItem,
  EvaluationSplit,
  EvaluationSuite,
  RetrievalModeSelection,
} from '../../types/evaluation'
import { suiteLabel } from './format'

const SUITES: EvaluationSuite[] = ['smoke', 'full', 'easy', 'medium', 'hard', 'multi_turn']

interface RunFormProps {
  capabilities: EvaluationCapabilities
  activeRun?: EvaluationRunListItem | null
  pending: boolean
  error: unknown
  onCreate: (payload: EvaluationRunCreateRequest) => Promise<EvaluationRunCreated>
  onCreated: (run: EvaluationRunCreated) => void
}

export function RunForm({ capabilities, activeRun, pending, error, onCreate, onCreated }: RunFormProps) {
  const [suite, setSuite] = useState<EvaluationSuite>('smoke')
  const [split, setSplit] = useState<EvaluationSplit>('dev')
  const [mode, setMode] = useState<RetrievalModeSelection>('hybrid')
  const [judge, setJudge] = useState(false)
  const [repeat, setRepeat] = useState(1)
  const [notes, setNotes] = useState('')
  const [confirmHoldout, setConfirmHoldout] = useState(false)
  const [clientRequestId, setClientRequestId] = useState(() => newClientMessageId())
  const [localError, setLocalError] = useState<string | null>(null)

  const modes = capabilities.retrieval_modes ?? ['hybrid', 'dense', 'keyword']
  const maxRepeat = capabilities.max_repeat ?? 5
  const judgeAvailable = capabilities.judge ?? false
  const needsHoldoutConfirmation = split === 'holdout' || (suite === 'full' && split === 'all')
  const activeId = activeRun?.id

  const changeSuite = (next: EvaluationSuite) => {
    setSuite(next)
    setConfirmHoldout(false)
    setSplit(next === 'smoke' ? 'dev' : next === 'full' ? 'all' : 'dev')
    if (next === 'smoke') setRepeat(1)
  }

  const disabledReason = useMemo(() => {
    if (activeId) {
      return `Уже выполняется запуск ${activeId}. Дождитесь завершения или отмените его.`
    }
    return null
  }, [activeId])

  const submit = async () => {
    setLocalError(null)
    if (needsHoldoutConfirmation && !confirmHoldout) {
      setLocalError('Holdout требует явного подтверждения: отметьте «Подтверждаю holdout-запуск».')
      return
    }
    const payload: EvaluationRunCreateRequest = {
      suite,
      split,
      retrieval_mode: mode,
      judge: judgeAvailable ? judge : false,
      repeat,
      notes: notes.trim() || undefined,
      client_request_id: clientRequestId,
      confirm_holdout: confirmHoldout,
    }
    try {
      const created = await onCreate(payload)
      setClientRequestId(newClientMessageId())
      setConfirmHoldout(false)
      onCreated(created)
    } catch {
      // Ошибка рендерится ниже из `error`/`localError`.
    }
  }

  return (
    <section className="rounded-xl border border-line bg-surface p-4 shadow-sm">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-ink">Новый запуск</h2>
          <p className="text-xs text-muted">
            Benchmark выполняется в фоне worker'ом; HTTP-запрос возвращает 202 и не вызывает LLM.
          </p>
        </div>
      </header>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs text-muted">
          Набор
          <select
            className="field"
            value={suite}
            onChange={(event) => changeSuite(event.target.value as EvaluationSuite)}
            aria-label="Набор тестов"
          >
            {SUITES.map((item) => (
              <option key={item} value={item}>
                {suiteLabel(item)}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs text-muted">
          Split
          <select
            className="field"
            value={split}
            onChange={(event) => {
              setSplit(event.target.value as EvaluationSplit)
              setConfirmHoldout(false)
            }}
            disabled={suite === 'smoke'}
            aria-label="Split"
          >
            <option value="dev">dev</option>
            {suite === 'smoke' ? null : <option value="holdout">holdout</option>}
            {suite === 'smoke' ? null : <option value="all">all</option>}
          </select>
        </label>
        <label className="space-y-1 text-xs text-muted">
          Retrieval mode
          <select
            className="field"
            value={mode}
            onChange={(event) => setMode(event.target.value as RetrievalModeSelection)}
            aria-label="Retrieval mode"
          >
            {modes.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
            <option value="all">all (отдельный результат на режим)</option>
          </select>
        </label>
        <label className="space-y-1 text-xs text-muted">
          Повторы (repeat)
          <input
            type="number"
            min={1}
            max={maxRepeat}
            value={repeat}
            onChange={(event) =>
              setRepeat(Math.max(1, Math.min(maxRepeat, Number(event.target.value) || 1)))
            }
            className="field"
            aria-label="Повторы"
          />
          {repeat > 1 ? (
            <span className="block text-[11px] text-warn">
              Проверка стабильности: каждый повтор сохраняется отдельно.
            </span>
          ) : null}
        </label>
        <label className="space-y-1 text-xs text-muted">
          Комментарий
          <input
            type="text"
            className="field"
            maxLength={500}
            value={notes}
            placeholder="Например: после изменения reranker"
            onChange={(event) => setNotes(event.target.value)}
            aria-label="Комментарий к запуску"
          />
        </label>
        <label className="flex items-start gap-2 self-start pt-5 text-xs text-ink">
          <input
            type="checkbox"
            checked={judge}
            disabled={!judgeAvailable}
            onChange={(event) => setJudge(event.target.checked)}
            aria-label="LLM-судья"
          />
          <span>
            LLM-судья (rubric)
            {!judgeAvailable ? (
              <span className="block text-[11px] text-muted">Недоступен: judge-модель не подключена.</span>
            ) : null}
          </span>
        </label>
      </div>

      {suite === 'smoke' ? (
        <p className="mt-2 text-[11px] text-muted">Smoke всегда выполняется на split=dev.</p>
      ) : null}
      {split === 'holdout' || (suite === 'full' && split === 'all') ? (
        <div className="mt-3 space-y-2">
          <Banner tone="warn" title="Holdout-набор">
            Holdout не используется для настройки. Full/holdout может идти десятки минут; benchmark
            занимает очередь LLM и не запускается параллельно с чатом.
          </Banner>
          <label className="flex items-center gap-2 text-xs text-ink">
            <input
              type="checkbox"
              checked={confirmHoldout}
              onChange={(event) => setConfirmHoldout(event.target.checked)}
              aria-label="Подтверждаю holdout-запуск"
            />
            Подтверждаю holdout-запуск
          </label>
        </div>
      ) : null}

      {disabledReason ? <Banner tone="info" title="Запуск недоступен">{disabledReason}</Banner> : null}
      {localError ? <Banner tone="error">{localError}</Banner> : null}
      {error ? <Banner tone="error" title="Не удалось создать запуск">{describeApiError(error)}</Banner> : null}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void submit()}
          disabled={pending || Boolean(disabledReason)}
          title={disabledReason ?? undefined}
        >
          <Play className="size-4" aria-hidden="true" />
          {pending ? 'Создание…' : `Запустить ${suiteLabel(suite)}`}
        </button>
        <span className="text-[11px] text-muted">
          Очередь ограничена ({capabilities.max_queued ?? 3}); при переполнении API вернёт queue_full.
        </span>
      </div>
    </section>
  )
}
