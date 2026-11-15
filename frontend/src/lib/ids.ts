function randomBytes(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  const cryptoApi = typeof crypto !== 'undefined' ? crypto : undefined
  if (cryptoApi && typeof cryptoApi.getRandomValues === 'function') {
    cryptoApi.getRandomValues(bytes)
    return bytes
  }
  for (let index = 0; index < length; index += 1) {
    bytes[index] = Math.floor(Math.random() * 256)
  }
  return bytes
}

/**
 * RFC 4122 version 4 UUID without `crypto.randomUUID`.
 *
 * The backend validates `client_message_id` as `UUID`, so the fallback must be
 * a real UUID: a `cm-…` string is rejected with 422 in non-secure contexts
 * (plain HTTP on a LAN address) where `crypto.randomUUID` is unavailable.
 */
export function fallbackUuidV4(): string {
  const bytes = randomBytes(16)
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function newClientMessageId(): string {
  const cryptoApi = typeof crypto !== 'undefined' ? crypto : undefined
  if (cryptoApi && typeof cryptoApi.randomUUID === 'function') {
    return cryptoApi.randomUUID()
  }
  return fallbackUuidV4()
}
