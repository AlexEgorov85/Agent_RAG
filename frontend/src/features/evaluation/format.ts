/**
 * Display helpers for the Evaluation section.
 *
 * Только форматирование уже полученных чисел: проценты, дельты, подписи.
 * Ничего не агрегируется и не пересчитывается из traces (§4.1, §8.9).
 */

import type {
  EvaluationError,
  EvaluationRunStatus,
  EvaluationSuite,
  EvaluationTestStatus,
  GroupStats,
} from '../../types/evaluation'
import { EMPTY_VALUE } from '../../lib/format'

export { EMPTY_VALUE }

export const RUN_STATUS_LABEL: Record<EvaluationRunStatus, string> = {
  queued: 'В очереди',
  running: 'Выполняется',
  cancel_requested: 'Останавливается…',
  completed: 'Завершён',
  failed: 'Ошибка',
  cancelled: 'Отменён',
  interrupted: 'Прерван',
}

export const TEST_STATUS_LABEL: Record<string, string> = {
  completed: 'Выполнен',
  failed: 'Не выполнен',
  timeout: 'Таймаут',
  error: 'Ошибка',
  skipped: 'Пропущен',
  passed: 'Пройден',
}

export const SUITE_LABEL: Record<EvaluationSuite, string> = {
  smoke: 'Smoke',
  full: 'Full',
  easy: 'Easy',
  medium: 'Medium',
  hard: 'Hard',
  multi_turn: 'Multi-turn',
}

export const SPLIT_LABEL: Record<string, string> = {
  dev: 'dev',
  holdout: 'holdout',
  all: 'все',
}

export const RETRIEVAL_MODE_LABEL: Record<string, string> = {
  hybrid: 'hybrid',
  dense: 'dense',
  keyword: 'keyword',
  all: 'все режимы',
}

export const LOSS_STAGE_LABEL: Record<string, string> = {
  ok: 'Доказательства найдены',
  never_retrieved: 'Не найдено на этапе поиска',
  lost_at_fusion: 'Потеряно при слиянии (RRF)',
  lost_at_rerank: 'Потеряно после rerank',
  lost_at_context: 'Не попало в контекст ответа',
  not_cited: 'Найдено, но не процитировано',
  unsupported_claim: 'Утверждение без подтверждения',
  dialogue_context_lost: 'Потерян контекст диалога',
}

export const CAUSE_LABEL: Record<string, string> = {
  retrieval: 'Поиск не нашёл доказательство',
  fusion: 'Доказательство потеряно при слиянии',
  rerank: 'Доказательство потеряно после rerank',
  context: 'Доказательство не попало в контекст',
  citation: 'Ответ не ссылается на найденное доказательство',
  answer: 'Ошибка в ответе/оценке',
  dialogue: 'Потерян контекст многошагового диалога',
}

export const COMPARE_REASON_LABEL: Record<string, string> = {
  dataset_sha256_mismatch: 'Разный датасет (dataset.sha256)',
  corpus_content_sha256_mismatch: 'Разный content hash корпуса',
  suite_or_split_mismatch: 'Разный набор тестов или split',
  test_set_mismatch: 'Разный состав suite-файлов',
}

export const VERSION_DIFFERENCE_LABEL: Record<string, string> = {
  git_sha_differs: 'Разный git SHA',
  git_dirty_differs: 'Разный флаг dirty',
  llm_model_differs: 'Другая LLM-модель',
  reranker_model_differs: 'Другая reranker-модель',
  judge_model_differs: 'Другая judge-модель',
  retrieval_config_differs: 'Другой retrieval config',
  corpus_index_fingerprint_differs: 'Другой index fingerprint (переиндексация)',
  retrieval_mode_differs: 'Другой retrieval mode',
}

export function statusLabel(status: string): string {
  return RUN_STATUS_LABEL[status as EvaluationRunStatus] ?? status
}

export function testStatusLabel(status: string): string {
  return TEST_STATUS_LABEL[status] ?? status
}

export function suiteLabel(suite: string | null | undefined): string {
  if (!suite) return EMPTY_VALUE
  return SUITE_LABEL[suite as EvaluationSuite] ?? suite
}

/** `null` — это «не измерялось», а не ноль. */
export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function formatPercent(value: unknown, digits = 1): string {
  if (!isNumber(value)) return EMPTY_VALUE
  return `${(value * 100).toLocaleString('ru-RU', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  })}%`
}

export function formatRatio(value: unknown, digits = 2): string {
  if (!isNumber(value)) return EMPTY_VALUE
  return value.toLocaleString('ru-RU', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  })
}

/** Дельта в процентных пунктах (для pass rate/coverage). */
export function formatDeltaPoints(delta: unknown, digits = 1): string {
  if (!isNumber(delta)) return EMPTY_VALUE
  const points = delta * 100
  if (Math.abs(points) < 0.05) return '0'
  const sign = points > 0 ? '+' : '−'
  return `${sign}${Math.abs(points).toLocaleString('ru-RU', {
    maximumFractionDigits: digits,
  })} п.п.`
}

/** Дельта в единицах метрики (Recall@10, rubric и т.п.). */
export function formatDeltaValue(delta: unknown, digits = 2): string {
  if (!isNumber(delta)) return EMPTY_VALUE
  if (Math.abs(delta) < 0.0005) return '0'
  const sign = delta > 0 ? '+' : '−'
  return `${sign}${Math.abs(delta).toLocaleString('ru-RU', {
    maximumFractionDigits: digits,
  })}`
}

/** Дельта latency в миллисекундах → секунды. */
export function formatDeltaMs(delta: unknown): string {
  if (!isNumber(delta)) return EMPTY_VALUE
  const seconds = delta / 1000
  if (Math.abs(seconds) < 0.05) return '0'
  const sign = seconds > 0 ? '+' : '−'
  return `${sign}${Math.abs(seconds).toLocaleString('ru-RU', {
    maximumFractionDigits: 1,
  })} с`
}

export function formatMs(value: unknown): string {
  if (!isNumber(value)) return EMPTY_VALUE
  if (value < 1000) return `${Math.round(value)} мс`
  return `${(value / 1000).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} с`
}

export function formatDuration(ms: number | null | undefined): string {
  if (!isNumber(ms)) return EMPTY_VALUE
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds} с`
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  if (minutes < 60) return `${minutes} мин${rest ? ` ${rest} с` : ''}`
  const hours = Math.floor(minutes / 60)
  return `${hours} ч ${minutes % 60} мин`
}

export function formatCoverage(stats: GroupStats | null | undefined): string {
  if (!stats) return EMPTY_VALUE
  const eligible = stats.eligible
  const executed = stats.executed
  if (!isNumber(eligible) && !isNumber(executed)) return EMPTY_VALUE
  return `${isNumber(eligible) ? eligible : EMPTY_VALUE} eligible · ${
    isNumber(executed) ? executed : EMPTY_VALUE
  } executed`
}

/** Человеческое объяснение кодов ошибок Evaluation API (§8.11). */
export const EVALUATION_ERROR_LABEL: Record<string, string> = {
  evaluation_disabled: 'Раздел Evaluation отключён конфигурацией',
  run_not_found: 'Запуск не найден',
  test_not_found: 'Тест или trace не найден',
  ambiguous_test_result: 'У теста несколько режимов — укажите retrieval_mode',
  invalid_retrieval_mode: 'Недопустимый retrieval mode',
  run_not_finished: 'Запуск ещё не завершён',
  run_in_progress: 'Уже выполняется другой benchmark',
  queue_full: 'Очередь benchmark заполнена',
  invalid_suite: 'Неизвестный suite',
  invalid_split: 'Недопустимый split',
  invalid_repeat: 'repeat должен быть 1–5',
  mode_not_supported: 'Режим retrieval не поддержан backend',
  judge_unavailable: 'LLM-судья недоступен',
  holdout_confirmation_required: 'Нужно подтверждение holdout',
  not_comparable: 'Запуски несопоставимы',
  cancel_not_supported: 'Отмена недоступна',
  report_not_found: 'Отчёт не найден',
  backend_unavailable: 'Backend недоступен',
  network_error: 'Нет связи с API',
  internal_error: 'Внутренняя ошибка backend',
}

export function evaluationErrorLabel(code: string): string {
  return EVALUATION_ERROR_LABEL[code] ?? code
}

/** Проблемы запуска: счётчики берутся из summary, не классифицируются в UI. */
export interface EvaluationProblemRow {
  code: string
  label: string
  count: number
  testIds: string[]
}

export function problemRows(
  critical: Array<{ code: string; count: number; test_ids?: string[] }> | undefined,
): EvaluationProblemRow[] {
  if (!critical) return []
  return critical
    .filter((item) => isNumber(item.count) && item.count > 0)
    .map((item) => ({
      code: item.code,
      label: LOSS_STAGE_LABEL[item.code] ?? item.code,
      count: item.count,
      testIds: item.test_ids ?? [],
    }))
}

export function errorText(error: EvaluationError | null | undefined): string {
  if (!error) return EMPTY_VALUE
  return `${evaluationErrorLabel(error.code)} (${error.code})`
}

export function isActiveStatus(status: EvaluationRunStatus | string): boolean {
  return status === 'queued' || status === 'running' || status === 'cancel_requested'
}

export function isTerminalStatus(status: EvaluationRunStatus | string): boolean {
  return (
    status === 'completed' ||
    status === 'failed' ||
    status === 'cancelled' ||
    status === 'interrupted'
  )
}

export function testStatusTone(
  status: EvaluationTestStatus | string,
  pass: boolean,
): 'success' | 'danger' | 'warn' | 'neutral' | 'active' {
  if (status === 'skipped') return 'neutral'
  if (status === 'error' || status === 'timeout') return 'warn'
  if (pass) return 'success'
  return 'danger'
}
