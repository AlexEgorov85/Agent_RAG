import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SortableTable } from '../../src/components/chat/SortableTable'

describe('SortableTable: keyboard access to the scroll region', () => {
  it('скролл-контейнер фокусируется и показывает видимый focus ring', () => {
    render(
      <SortableTable
        caption="Данные backend"
        ariaLabel="Таблица данных"
        columns={[{ key: 'metric', label: 'Показатель', render: (row) => row.metric }]}
        rows={[{ metric: 'revenue' }]}
      />,
    )
    const region = screen.getByRole('region', { name: 'Таблица данных' })
    expect(region).toHaveAttribute('tabindex', '0')
    expect(region).toHaveClass('focus-visible:ring-2')
    expect(region).toHaveClass('focus-visible:ring-accent/60')
  })
})
