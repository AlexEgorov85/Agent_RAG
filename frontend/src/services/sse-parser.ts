/**
 * Minimal SSE parser (WHATWG text/event-stream).
 *
 * Comments (heartbeat `: keep-alive`) produce no messages. Multi-line `data:`
 * fields are joined with `\n` as required by the SSE specification.
 */

export interface SseMessage {
  id?: string
  event?: string
  data: string
}

export class SseParser {
  private buffer = ''
  private dataLines: string[] = []
  private eventName: string | undefined
  private lastId: string | undefined

  /** Feed a decoded chunk; returns all messages completed by this chunk. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk
    const messages: SseMessage[] = []
    let newlineIndex = this.buffer.indexOf('\n')
    while (newlineIndex !== -1) {
      let line = this.buffer.slice(0, newlineIndex)
      this.buffer = this.buffer.slice(newlineIndex + 1)
      if (line.endsWith('\r')) line = line.slice(0, -1)
      const message = this.consumeLine(line)
      if (message) messages.push(message)
      newlineIndex = this.buffer.indexOf('\n')
    }
    return messages
  }

  private consumeLine(line: string): SseMessage | null {
    if (line === '') {
      const message = this.dispatch()
      this.eventName = undefined
      this.dataLines = []
      return message
    }
    if (line.startsWith(':')) return null

    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    let value = separator === -1 ? '' : line.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    switch (field) {
      case 'data':
        this.dataLines.push(value)
        break
      case 'event':
        this.eventName = value
        break
      case 'id':
        if (value) this.lastId = value
        break
      default:
        break
    }
    return null
  }

  private dispatch(): SseMessage | null {
    if (this.dataLines.length === 0) return null
    const message: SseMessage = { data: this.dataLines.join('\n') }
    if (this.eventName !== undefined) message.event = this.eventName
    if (this.lastId !== undefined) message.id = this.lastId
    return message
  }
}

/** Parse a JSON SSE payload without throwing on malformed input. */
export function parseSseJson(data: string): unknown {
  try {
    return JSON.parse(data) as unknown
  } catch {
    return null
  }
}
