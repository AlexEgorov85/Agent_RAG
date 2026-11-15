import { afterEach, describe, expect, it, vi } from 'vitest'
import { fallbackUuidV4, newClientMessageId } from '../../src/lib/ids'

/** RFC 4122 version 4 UUID as required by the backend `MessageCreate`. */
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('client message ids', () => {
  it('fallback без crypto.randomUUID — валидный UUID v4, а не cm-строка', () => {
    vi.stubGlobal('crypto', {
      getRandomValues: (array: Uint8Array) => {
        for (let index = 0; index < array.length; index += 1) array[index] = index * 7 % 256
      },
    })
    const id = newClientMessageId()
    expect(id).toMatch(UUID_V4)
    expect(id.startsWith('cm-')).toBe(false)
  })

  it('fallback без getRandomValues тоже даёт RFC 4122 v4', () => {
    vi.stubGlobal('crypto', {})
    expect(fallbackUuidV4()).toMatch(UUID_V4)
  })

  it('fallback не повторяется между вызовами', () => {
    vi.stubGlobal('crypto', {})
    const ids = new Set(Array.from({ length: 50 }, () => newClientMessageId()))
    expect(ids.size).toBe(50)
  })

  it('использует crypto.randomUUID, когда он доступен', () => {
    const randomUUID = vi.fn(() => '11111111-1111-4111-8111-111111111111')
    vi.stubGlobal('crypto', { randomUUID })
    expect(newClientMessageId()).toBe('11111111-1111-4111-8111-111111111111')
    expect(randomUUID).toHaveBeenCalledTimes(1)
  })
})
