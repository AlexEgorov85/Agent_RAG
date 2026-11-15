import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import type { Components } from 'react-markdown'
import { CitationMarker, rehypeCitationMarkers } from './Citations'

/**
 * Markdown rendering with sanitization.
 *
 * Raw HTML is never parsed (no `rehype-raw`) and `rehype-sanitize` strips
 * dangerous URLs, so agent output cannot inject HTML or scripts.
 *
 * Citation markers `[n]` are converted into anchors *after* sanitization and
 * rendered as buttons that open the matching evidence snapshot. External links
 * keep `target="_blank"` + `noopener noreferrer`; nothing else is clickable.
 */

const CITATION_HREF = /^#citation-(\d{1,3})$/

const components: Components = {
  a({ href, children, ...props }) {
    const citationMatch = typeof href === 'string' ? CITATION_HREF.exec(href) : null
    if (citationMatch) {
      return <CitationMarker n={Number(citationMatch[1])} />
    }

    const external = typeof href === 'string' && /^https?:\/\//i.test(href)
    return (
      <a
        href={href}
        {...props}
        {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      >
        {children}
      </a>
    )
  },
  table({ children }) {
    return (
      <div className="max-w-full overflow-x-auto">
        <table>{children}</table>
      </div>
    )
  },
}

export function Markdown({ content }: { content: string }) {
  return (
    <div className="md min-w-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize, rehypeCitationMarkers]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
