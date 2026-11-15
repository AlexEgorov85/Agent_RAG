import { createElement, Fragment, type ReactNode } from 'react'

/**
 * Whitelist sanitizer for table HTML from `/api/evidence`.
 *
 * DOMParser output is converted into React elements; the HTML string is never
 * injected with `dangerouslySetInnerHTML`, so scripts, event handlers, styles
 * and external resources cannot execute.
 */

const ALLOWED_TAGS = new Set([
  'TABLE',
  'THEAD',
  'TBODY',
  'TFOOT',
  'TR',
  'TH',
  'TD',
  'CAPTION',
  'COLGROUP',
  'COL',
  'P',
  'SPAN',
  'STRONG',
  'EM',
  'B',
  'I',
  'BR',
])

const DROPPED_TAGS = new Set(['SCRIPT', 'STYLE', 'IFRAME', 'OBJECT', 'EMBED', 'LINK', 'META', 'IMG', 'SVG'])

const ALLOWED_ATTRIBUTES: Record<string, string[]> = {
  TD: ['colspan', 'rowspan'],
  TH: ['colspan', 'rowspan', 'scope'],
  COL: ['span'],
  COLGROUP: ['span'],
}

/** HTML attribute name → React prop name (React warns on lowercase DOM props). */
const ATTRIBUTE_NAMES: Record<string, string> = {
  colspan: 'colSpan',
  rowspan: 'rowSpan',
}

const TAG_NAMES: Record<string, string> = {
  TABLE: 'table',
  THEAD: 'thead',
  TBODY: 'tbody',
  TFOOT: 'tfoot',
  TR: 'tr',
  TH: 'th',
  TD: 'td',
  CAPTION: 'caption',
  COLGROUP: 'colgroup',
  COL: 'col',
  P: 'p',
  SPAN: 'span',
  STRONG: 'strong',
  EM: 'em',
  B: 'b',
  I: 'i',
  BR: 'br',
}

export function sanitizeTableHtml(html: string): ReactNode {
  if (!html || typeof DOMParser === 'undefined') return null
  const documentNode = new DOMParser().parseFromString(html, 'text/html')
  return convertChildren(documentNode.body, 'sanitized')
}

function convertChildren(parent: Node, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let index = 0
  parent.childNodes.forEach((child) => {
    const converted = convertNode(child, `${keyPrefix}-${index}`)
    index += 1
    if (converted !== null) nodes.push(converted)
  })
  return nodes
}

function convertNode(node: Node, key: string): ReactNode {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? ''
  if (node.nodeType !== Node.ELEMENT_NODE) return null

  const element = node as Element
  const tag = element.tagName.toUpperCase()
  if (DROPPED_TAGS.has(tag)) return null

  const children = convertChildren(element, key)
  if (!ALLOWED_TAGS.has(tag)) return createElement(Fragment, { key }, ...children)

  const allowed = ALLOWED_ATTRIBUTES[tag] ?? []
  const props: Record<string, string> = {}
  for (const attribute of allowed) {
    const value = element.getAttribute(attribute)
    if (value !== null) props[ATTRIBUTE_NAMES[attribute] ?? attribute] = value
  }

  return createElement(TAG_NAMES[tag] ?? tag.toLowerCase(), { key, ...props }, ...children)
}

/** Split text around the quote for highlighting; returns parts with a match flag. */
export function splitByQuote(text: string, quote: string): Array<{ text: string; match: boolean }> {
  const normalizedQuote = quote.trim()
  if (!normalizedQuote) return [{ text, match: false }]
  const index = text.indexOf(normalizedQuote)
  if (index === -1) return [{ text, match: false }]
  const parts: Array<{ text: string; match: boolean }> = []
  if (index > 0) parts.push({ text: text.slice(0, index), match: false })
  parts.push({ text: normalizedQuote, match: true })
  const tail = text.slice(index + normalizedQuote.length)
  if (tail) parts.push({ text: tail, match: false })
  return parts
}
