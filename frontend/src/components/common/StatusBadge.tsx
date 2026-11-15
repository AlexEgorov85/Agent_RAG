import type { MessageStatus, ResearchStatus } from '../../types/api'
import { MESSAGE_STATUS_LABEL, RESEARCH_STATUS_LABEL } from '../../lib/labels'
import { ToneBadge, type BadgeTone } from './ToneBadge'

const RESEARCH_TONE: Record<ResearchStatus, BadgeTone> = {
  idle: 'neutral',
  running: 'active',
  completed: 'success',
  abstained: 'warn',
  failed: 'danger',
  cancelled: 'neutral',
}

const MESSAGE_TONE: Record<MessageStatus, BadgeTone> = {
  pending: 'neutral',
  streaming: 'active',
  completed: 'success',
  abstained: 'warn',
  failed: 'danger',
  cancelled: 'neutral',
}

export function ResearchStatusBadge({ status }: { status: ResearchStatus }) {
  return (
    <ToneBadge tone={RESEARCH_TONE[status] ?? 'neutral'} title={`Статус: ${status}`}>
      {RESEARCH_STATUS_LABEL[status] ?? status}
    </ToneBadge>
  )
}

export function MessageStatusBadge({ status }: { status: MessageStatus }) {
  return (
    <ToneBadge tone={MESSAGE_TONE[status] ?? 'neutral'} title={`Статус сообщения: ${status}`}>
      {MESSAGE_STATUS_LABEL[status] ?? status}
    </ToneBadge>
  )
}
