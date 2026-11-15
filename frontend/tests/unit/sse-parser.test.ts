import { describe, expect, it } from 'vitest'
import { SseParser, parseSseJson } from '../../src/services/sse-parser'

describe('SseParser', () => {
  it('разбирает событие, разбитое на несколько чанков', () => {
    const parser = new SseParser()
    expect(parser.push('id: 1\nevent: ag')).toEqual([])
    const messages = parser.push('ent\ndata: {"a":1}\n\n')
    expect(messages).toEqual([{ id: '1', event: 'agent', data: '{"a":1}' }])
  })

  it('игнорирует heartbeat-комментарии', () => {
    const parser = new SseParser()
    expect(parser.push(': keep-alive\n\n')).toEqual([])
    const messages = parser.push('data: {"type":"run_started"}\n\n')
    expect(messages).toHaveLength(1)
  })

  it('склеивает многострочный data через перевод строки', () => {
    const parser = new SseParser()
    const messages = parser.push('event: agent\ndata: line1\ndata: line2\n\n')
    expect(messages[0]?.data).toBe('line1\nline2')
    expect(messages[0]?.event).toBe('agent')
  })

  it('поддерживает CRLF', () => {
    const parser = new SseParser()
    const messages = parser.push('id: 7\r\ndata: ok\r\n\r\n')
    expect(messages).toEqual([{ id: '7', data: 'ok' }])
  })

  it('не диспатчит событие без data', () => {
    const parser = new SseParser()
    expect(parser.push('event: agent\n\n')).toEqual([])
  })

  it('parseSseJson возвращает null на битом JSON', () => {
    expect(parseSseJson('{oops')).toBeNull()
    expect(parseSseJson('{"ok":true}')).toEqual({ ok: true })
  })
})
