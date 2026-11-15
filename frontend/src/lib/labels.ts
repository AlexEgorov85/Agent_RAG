import type { MessageStatus, ResearchStatus } from '../types/api'
import type { StructuralChangeType } from '../types/analytics'

/** UI labels for backend codes. Unknown codes fall back to the code itself. */

export const RESEARCH_STATUS_LABEL: Record<ResearchStatus, string> = {
  idle: 'Ожидает',
  running: 'Выполняется',
  completed: 'Готово',
  abstained: 'Без ответа',
  failed: 'Ошибка',
  cancelled: 'Отменено',
}

export const MESSAGE_STATUS_LABEL: Record<MessageStatus, string> = {
  pending: 'Ожидает',
  streaming: 'Выполняется',
  completed: 'Готово',
  abstained: 'Ответ не сформирован',
  failed: 'Ошибка',
  cancelled: 'Отменено',
}

export const TOOL_LABEL: Record<string, string> = {
  search_reports: 'Поиск по отчётам',
  get_structure_tree: 'Структура документа',
  get_entities: 'Сущности графа',
  get_chunk: 'Фрагмент документа',
}

export function toolLabel(toolName: string): string {
  return TOOL_LABEL[toolName] ?? toolName
}

/** Warning codes from the backend, localized on the client (§7.10). */
export const WARNING_LABEL: Record<string, string> = {
  no_evidence: 'В доступных материалах не найдено подтверждение для ответа.',
  no_evidence_for_period: 'За запрошенный период доказательства не найдены.',
  missing_year: 'В доступных материалах отсутствует запрошенный год.',
  missing_metric: 'В отчёте за указанный период показатель не найден.',
  low_confidence: 'Уверенность в ответе низкая — проверьте источники.',
  scope_mismatch: 'Источники соответствуют запросу не полностью.',
  issuer_mismatch: 'Источник относится к другой организации.',
  incomparable_year_excluded:
    'Год исключён из сравнения: отчёт относится к другому эмитенту.',
  incomplete_period: 'Данные за запрошенный период неполные.',
  query_rewrite_fallback: 'Уточнить запрос не удалось — использован исходный текст.',
  chunk_not_authorized: 'Фрагмент документа недоступен для этого запроса.',
  tool_budget_exhausted: 'Лимит обращений к инструментам исчерпан.',
  generation_unavailable: 'Модель не сформировала ответ: генерация недоступна.',
  citation_validation_failed: 'Проверка цитат не удалась: часть ссылок убрана.',
  citation_validation_retry: 'Ответ сформирован повторно после замечаний к цитатам.',
  citation_outside_manifest: 'Часть цитат отсутствует в переданных источниках и убрана.',
  verification_retry: 'Ответ проверен повторно после замечаний.',
  verification_failed: 'Проверка ответа завершилась с замечаниями.',
  verification_warning: 'Проверка ответа нашла замечания — сверьтесь с источниками.',
  agent_run_failed: 'Выполнение агента завершилось ошибкой.',
  graph_unavailable: 'Граф сущностей недоступен.',
  table_row_unit_assumed: 'Единица измерения таблицы восстановлена по контексту.',
  structure_unavailable: 'Структура документа недоступна.',
  structure_scope_limited: 'Структурный анализ ограничен доступными документами.',
  citation_not_found: 'Часть цитат не найдена в индексе.',
  answer_truncated: 'Ответ был сокращён.',
}

export function warningLabel(code: string): string {
  return WARNING_LABEL[code] ?? code
}

export const ERROR_LABEL: Record<string, string> = {
  validation_error: 'Некорректный запрос',
  not_found: 'Объект не найден',
  request_in_progress: 'В этом диалоге уже выполняется запрос',
  request_not_found: 'Запрос не найден',
  request_not_replayable: 'События запроса недоступны после перезапуска API',
  payload_too_large: 'Превышен допустимый размер',
  rate_limited: 'Слишком много запросов',
  backend_unavailable: 'Backend недоступен',
  internal_error: 'Внутренняя ошибка backend',
  network_error: 'Нет связи с API',
  interrupted: 'Выполнение прервано',
  connection_lost: 'Соединение с потоком событий потеряно',
  cancel_failed: 'Отмена не выполнена',
  invalid_response: 'Некорректный ответ API',
}

export function errorLabel(code: string): string {
  return ERROR_LABEL[code] ?? code
}

export const STRUCTURAL_CHANGE_LABEL: Record<StructuralChangeType, string> = {
  stable: 'Без изменений',
  added: 'Добавлен',
  removed: 'Не обнаружен в более позднем отчёте',
  renamed: 'Переименован',
  restructured: 'Перемещён / изменён уровень',
  expanded: 'Объём вырос',
  reduced: 'Объём сократился',
}

export const COVERAGE_STATUS_LABEL = {
  covered: 'Найдено',
  not_found: 'Не найдено',
  year_absent: 'Год отсутствует в корпусе',
} as const

export const SOURCE_TYPE_LABEL = {
  text: 'Текст',
  table: 'Таблица',
} as const

export const ROLE_LABEL = {
  user: 'Вы',
  assistant: 'Агент',
} as const
