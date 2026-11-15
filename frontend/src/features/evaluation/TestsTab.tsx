import { useMemo, useState } from 'react'
import { EmptyState } from '../../components/common/EmptyState'
import { ErrorState } from '../../components/common/ErrorState'
import { Spinner } from '../../components/common/Spinner'
import { ToneBadge } from '../../components/common/ToneBadge'
import { useDebouncedValue } from '../../lib/use-debounced'
import type { RetrievalMode } from '../../types/evaluation'
import { EMPTY_VALUE, previewText } from '../../lib/format'
import { testStatusLabel } from './format'
import { useEvaluationCatalog } from './use-evaluation'
import { TestDetail } from './TestDetail'

const CATEGORIES = ['fact_lookup', 'temporal', 'structural', 'multi_hop', 'abstain', 'dialogue']
const DIFFICULTIES = ['easy', 'medium', 'hard', 'multi_turn']

export function TestsTab({ onOpenRuns }: { onOpenRuns: () => void }) {
  const [difficulty, setDifficulty] = useState('')
  const [category, setCategory] = useState('')
  const [suite, setSuite] = useState('')
  const [status, setStatus] = useState('')
  const [critical, setCritical] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const [search, setSearch] = useState('')
  const [selectedTestId, setSelectedTestId] = useState<string | null>(null)
  const debouncedSearch = useDebouncedValue(search)

  const params = useMemo(
    () => ({
      difficulty: difficulty || undefined,
      category: category || undefined,
      suite: suite || undefined,
      status: status || undefined,
      critical: critical === 'with' ? true : critical === 'without' ? false : undefined,
      q: debouncedSearch || undefined,
      retrieval_mode: mode,
      limit: 100,
    }),
    [difficulty, category, suite, status, critical, debouncedSearch, mode],
  )
  const catalog = useEvaluationCatalog(params)
  const items = catalog.data?.items ?? []
  const selected = items.find((item) => item.id === selectedTestId) ?? null

  return (
    <div className="space-y-4">
      <section className="rounded-xl border border-line bg-surface p-4 shadow-sm">
        <header className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-ink">Каталог тестов</h2>
            <p className="text-xs text-muted">
              Последний статус — из последнего завершённого запуска. Режимы не смешиваются в одной
              ячейке.
            </p>
          </div>
          <ToneBadge tone="neutral">режим {mode}</ToneBadge>
        </header>

        <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <label className="space-y-1 text-xs text-muted">
            Поиск
            <input
              type="search"
              className="field"
              value={search}
              placeholder="ID или вопрос"
              onChange={(event) => setSearch(event.target.value)}
              aria-label="Поиск по тестам"
            />
          </label>
          <label className="space-y-1 text-xs text-muted">
            Difficulty
            <select className="field" value={difficulty} onChange={(event) => setDifficulty(event.target.value)} aria-label="Фильтр difficulty">
              <option value="">Все</option>
              {DIFFICULTIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Category
            <select className="field" value={category} onChange={(event) => setCategory(event.target.value)} aria-label="Фильтр category">
              <option value="">Все</option>
              {CATEGORIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Suite
            <select className="field" value={suite} onChange={(event) => setSuite(event.target.value)} aria-label="Фильтр suite">
              <option value="">Все</option>
              {['easy', 'medium', 'hard', 'multi_turn'].map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Последний статус
            <select className="field" value={status} onChange={(event) => setStatus(event.target.value)} aria-label="Фильтр статуса теста">
              <option value="">Все</option>
              <option value="passed">passed</option>
              <option value="failed">failed</option>
              <option value="not_run">not_run</option>
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Critical errors
            <select className="field" value={critical} onChange={(event) => setCritical(event.target.value)} aria-label="Фильтр critical errors">
              <option value="">Все</option>
              <option value="with">есть</option>
              <option value="without">нет</option>
            </select>
          </label>
          <label className="space-y-1 text-xs text-muted">
            Retrieval mode
            <select
              className="field"
              value={mode}
              onChange={(event) => {
                setMode(event.target.value as RetrievalMode)
                setSelectedTestId(null)
              }}
              aria-label="Режим каталога"
            >
              {(['hybrid', 'dense', 'keyword'] as RetrievalMode[]).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>

        <p className="mt-2 text-[11px] text-muted">
          {catalog.data
            ? `${catalog.data.total} тестов · run ${catalog.data.run_id ?? '—'}`
            : 'Загрузка…'}
        </p>

        <div className="mt-3">
          {catalog.isLoading ? (
            <div className="flex h-24 items-center justify-center">
              <Spinner label="Загрузка каталога" />
            </div>
          ) : catalog.error ? (
            <ErrorState error={catalog.error} title="Каталог недоступен" onRetry={() => void catalog.refetch()} />
          ) : !catalog.data?.dataset_available ? (
            <EmptyState
              title="Датасет недоступен"
              description="Каталог строится из eval/datasets. Пока набор не найден, UI не выдумывает тесты."
              action={
                <button type="button" className="btn" onClick={onOpenRuns}>
                  Открыть запуски
                </button>
              }
            />
          ) : items.length === 0 ? (
            <EmptyState title="Тесты не найдены" description="Измените фильтры или режим retrieval." />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-xs">
                <thead className="text-muted">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-2 font-medium">ID</th>
                    <th className="py-2 pr-2 font-medium">Вопрос</th>
                    <th className="py-2 pr-2 font-medium">Difficulty</th>
                    <th className="py-2 pr-2 font-medium">Category</th>
                    <th className="py-2 pr-2 font-medium">Статус</th>
                    <th className="py-2 pr-2 font-medium">Critical</th>
                    <th className="py-2 font-medium">Запуск</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr
                      key={item.id}
                      className={`border-b border-line/60 last:border-0 ${
                        selectedTestId === item.id ? 'bg-raised/70' : ''
                      }`}
                    >
                      <td className="py-2 pr-2 font-mono text-[11px]">
                        <button
                          type="button"
                          className="underline decoration-line underline-offset-2 hover:decoration-accent"
                          onClick={() => setSelectedTestId(item.id)}
                        >
                          {item.id}
                        </button>
                      </td>
                      <td className="max-w-[22rem] py-2 pr-2" title={item.question ?? undefined}>
                        {previewText(item.question, 90) || EMPTY_VALUE}
                      </td>
                      <td className="py-2 pr-2">{item.difficulty ?? EMPTY_VALUE}</td>
                      <td className="py-2 pr-2">{item.category ?? EMPTY_VALUE}</td>
                      <td className="py-2 pr-2">
                        {item.last_run ? (
                          <ToneBadge tone={item.last_run.pass ? 'success' : 'danger'}>
                            {item.last_run.pass ? 'passed' : 'failed'} ·{' '}
                            {testStatusLabel(item.last_run.status)}
                          </ToneBadge>
                        ) : (
                          <ToneBadge tone="neutral">not_run</ToneBadge>
                        )}
                      </td>
                      <td className="py-2 pr-2 font-mono text-[11px]">
                        {item.last_run?.critical_errors.map((error) => error.code).join(', ') || '—'}
                      </td>
                      <td className="py-2 font-mono text-[11px]">
                        {item.last_run?.run_id ?? EMPTY_VALUE}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {!selected ? (
        <EmptyState
          title="Выберите тест"
          description="Детализация покажет query plan, этапы retrieval, evidence, ответ, critical errors и trace."
        />
      ) : !selected.last_run ? (
        <EmptyState
          title="Тест ещё не выполнялся"
          description="Нет результата в последнем завершённом запуске — детализация недоступна без реального прогона."
          action={
            <button type="button" className="btn" onClick={onOpenRuns}>
              Открыть запуски
            </button>
          }
        />
      ) : (
        <TestDetail
          key={`${selected.last_run.run_id}-${selected.id}-${mode}`}
          runId={selected.last_run.run_id}
          testId={selected.id}
          mode={mode}
          runCreatedAt={selected.last_run.created_at}
        />
      )}
    </div>
  )
}
