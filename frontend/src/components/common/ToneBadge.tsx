import type { ReactNode } from 'react'

export type BadgeTone = 'neutral' | 'active' | 'success' | 'warn' | 'danger'

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: 'border-line bg-raised text-muted',
  active: 'border-accent/30 bg-accent-soft text-accent',
  success: 'border-emerald-600/25 bg-emerald-600/10 text-emerald-700 dark:text-emerald-300',
  warn: 'border-warn/30 bg-warn/10 text-warn',
  danger: 'border-danger/30 bg-danger/10 text-danger',
}

export function ToneBadge({
  tone = 'neutral',
  children,
  title,
}: {
  tone?: BadgeTone
  children: ReactNode
  title?: string
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${TONE_CLASSES[tone]}`}
      title={title}
    >
      {children}
    </span>
  )
}
