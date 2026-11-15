import type { Capabilities, ChatMessage, CorpusInfo, HealthResponse, Research } from '../types/api'
import type { EvidenceDetail, EvidenceItem } from '../types/evidence'

/**
 * Fixture data for the explicit mock mode (`VITE_API_MODE=mock`).
 *
 * This data is hand-written to exercise UI states (full metadata, missing
 * metadata, table source, abstain/cancel/error). It is NOT the RAG core and is
 * never presented as a real research result: the app shows a persistent banner.
 */

export const MOCK_BANNER =
  'MOCK-режим: интерфейс работает на встроенных фикстурах, Python-ядро не подключено.'

export const mockHealth: HealthResponse = { status: 'mock', corpus_ready: true }

export const mockCapabilities: Capabilities = {
  contract_version: 'mock',
  token_streaming: false,
  incremental_evidence: false,
  coverage: false,
  calculations: false,
  structure_comparison: false,
  structure_content_comparison: true,
  graph_panel: false,
  pdf_page_links: false,
  evaluation_ui: false,
  evaluation_sse: false,
  evaluation_cancel: false,
  retrieval_modes: ['hybrid'],
  judge: false,
}

export const mockCorpus: CorpusInfo = {
  name: 'Демонстрационный корпус (mock)',
  years: [2015, 2020, 2022, 2023, 2024],
  documents: [
    { doc_id: 'sber_2023', year: 2023, issuer: 'ПАО Сбербанк', title: 'Годовой отчёт 2023' },
    { doc_id: 'sber_2022', year: 2022, issuer: 'ПАО Сбербанк', title: 'Годовой отчёт 2022' },
    {
      doc_id: 'bps_2015',
      year: 2015,
      issuer: 'БПС-Сбербанк (Беларусь)',
      title: 'Годовой отчёт 2015',
    },
  ],
  issuers: ['ПАО Сбербанк', 'БПС-Сбербанк (Беларусь)'],
  years_available: true,
}

const chunkA = 'sber_2023:00078:c3:5cc6e1f46e2'
const chunkB = 'sber_2023:00112:t1:9a1f0c77b3d4'
const chunkC = 'bps_2015:00042:c1:0d42aa9c18ff'

export const mockEvidence: EvidenceItem[] = [
  {
    evidence_id: chunkA,
    chunk_id: chunkA,
    document_id: 'sber_2023',
    document_title: 'Годовой отчёт 2023',
    issuer: 'ПАО Сбербанк',
    year: 2023,
    section_title: 'Ключевые показатели',
    section_path: ['Стратегия', 'Ключевые показатели'],
    page_start: 12,
    page_label: '12',
    excerpt:
      'Чистая прибыль за 2023 год составила 1 508,6 млрд рублей, увеличившись на 4,0% по сравнению с 2022 годом.',
    source_type: 'text',
    has_full_text: true,
  },
  {
    evidence_id: chunkB,
    chunk_id: chunkB,
    document_id: 'sber_2023',
    issuer: 'ПАО Сбербанк',
    section_title: 'Основные финансовые результаты',
    section_path: ['Результаты', 'Основные финансовые результаты'],
    excerpt:
      '| Показатель | 2023 | 2022 |\n| --- | --- | --- |\n| Чистая прибыль, млрд руб. | 1 508,6 | 1 450,4 |',
    source_type: 'table',
    has_full_text: true,
  },
  {
    evidence_id: chunkC,
    chunk_id: chunkC,
    issuer: 'БПС-Сбербанк (Беларусь)',
    excerpt: '',
    source_type: 'text',
    has_full_text: false,
  },
]

export const mockEvidenceDetails: Record<string, EvidenceDetail> = {
  [chunkA]: {
    chunk_id: chunkA,
    document_id: 'sber_2023',
    document_title: 'Годовой отчёт 2023',
    issuer: 'ПАО Сбербанк',
    year: 2023,
    section_path: ['Стратегия', 'Ключевые показатели'],
    page_start: 12,
    page_label: '12',
    source_type: 'text',
    text:
      'Стратегия\n\nКлючевые показатели\n\nЧистая прибыль за 2023 год составила 1 508,6 млрд рублей, ' +
      'увеличившись на 4,0% по сравнению с 2022 годом. Рентабельность капитала достигла 25,3%. ' +
      'Показатель приведён по данным консолидированной отчётности.',
    html: null,
  },
  [chunkB]: {
    chunk_id: chunkB,
    document_id: 'sber_2023',
    document_title: 'Годовой отчёт 2023',
    issuer: 'ПАО Сбербанк',
    year: 2023,
    section_path: ['Результаты', 'Основные финансовые результаты'],
    source_type: 'table',
    text:
      'Показатель\t2023\t2022\nЧистая прибыль, млрд руб.\t1 508,6\t1 450,4\n' +
      'Операционные доходы, млрд руб.\t3 526,9\t3 128,4',
    html:
      '<table><caption>Основные финансовые результаты</caption><thead><tr>' +
      '<th>Показатель</th><th>2023</th><th>2022</th></tr></thead><tbody>' +
      '<tr><td>Чистая прибыль, млрд руб.</td><td>1 508,6</td><td>1 450,4</td></tr>' +
      '<tr><td>Операционные доходы, млрд руб.</td><td>3 526,9</td><td>3 128,4</td></tr>' +
      '</tbody></table>',
  },
}

export const mockAnswerMarkdown = [
  'По данным годового отчёта за 2023 год **чистая прибыль составила 1 508,6 млрд руб.**',
  '',
  '| Показатель | 2023 | 2022 |',
  '| --- | --- | --- |',
  '| Чистая прибыль, млрд руб. | 1 508,6 | 1 450,4 |',
  '',
  'Динамику и расчёт изменения агент приводит отдельно; в этом ответе расчёт не сформирован.',
].join('\n')

export const seedResearches: Research[] = [
  {
    id: 'research-profit-2024',
    title: 'Динамика чистой прибыли',
    created_at: '2026-11-12T09:10:00Z',
    updated_at: '2026-11-12T09:12:30Z',
    last_message_preview: 'Какая чистая прибыль за 2023 год?',
    last_status: 'completed',
    messages_count: 2,
  },
  {
    id: 'research-structure-2024',
    title: 'Структура отчёта 2022 → 2023',
    created_at: '2026-11-10T14:00:00Z',
    updated_at: '2026-11-10T14:02:00Z',
    last_message_preview: 'Сравни структуру отчётов за 2022 и 2023 годы',
    last_status: 'failed',
    messages_count: 2,
  },
]

export const seedMessages: Record<string, ChatMessage[]> = {
  'research-profit-2024': [
    {
      id: 'msg-user-1',
      research_id: 'research-profit-2024',
      role: 'user',
      content: 'Какая чистая прибыль за 2023 год?',
      created_at: '2026-11-12T09:10:00Z',
      status: 'completed',
      evidence: [],
      warnings: [],
    },
    {
      id: 'msg-assistant-1',
      research_id: 'research-profit-2024',
      role: 'assistant',
      content: mockAnswerMarkdown,
      created_at: '2026-11-12T09:12:30Z',
      status: 'completed',
      request_id: 'mock-request-1',
      effective_filters: { years: [2023] },
      evidence: mockEvidence,
      warnings: ['table_row_unit_assumed'],
    },
  ],
  'research-structure-2024': [
    {
      id: 'msg-user-2',
      research_id: 'research-structure-2024',
      role: 'user',
      content: 'Сравни структуру отчётов за 2022 и 2023 годы',
      created_at: '2026-11-10T14:00:00Z',
      status: 'completed',
      evidence: [],
      warnings: [],
    },
    {
      id: 'msg-assistant-2',
      research_id: 'research-structure-2024',
      role: 'assistant',
      content: '',
      created_at: '2026-11-10T14:02:00Z',
      status: 'failed',
      request_id: 'mock-request-2',
      evidence: [],
      warnings: [],
      error: {
        code: 'backend_unavailable',
        message: 'AgentService недоступен в момент выполнения запроса.',
        request_id: 'mock-request-2',
        retryable: true,
      },
    },
  ],
}

export const mockAbstainAnswer =
  'В переданных фрагментах нет значения чистой прибыли за 2021 год. ' +
  'Год 2021 отсутствует в демонстрационном корпусе, поэтому ответ не сформирован.'
