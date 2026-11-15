import type { ReactNode } from 'react'

export type BannerTone = 'info' | 'warn' | 'error' | 'mock' | 'success'

const TONE: Record<BannerTone, string> = {
  info: 'border-accent/25 bg-accent-soft/60 text-ink',
  warn: 'border-warn/35 bg-warn/10 text-ink',
  error: 'border-danger/35 bg-danger/10 text-ink',
  mock: 'border-fuchsia-500/40 bg-fuchsia-500/10 text-ink',
  success: 'border-emerald-600/25 bg-emerald-600/10 text-ink',
}

export function Banner({
  tone = 'info',
  title,
  children,
  action,
}: {
  tone?: BannerTone
  title?: string
  children?: ReactNode
  action?: ReactNode
}) {
  const role = tone === 'error' ? 'alert' : 'status'
  return (
    <div
      role={role}
      className={`flex flex-wrap items-start gap-2 rounded-lg border px-3 py-2 text-sm ${TONE[tone]}`}
    >
      <div className="min-w-0 flex-1">
        {title ? <p className="font-medium">{title}</p> : null}
        {children ? <div className={title ? 'mt-1 text-muted' : 'text-muted'}>{children}</div> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}
