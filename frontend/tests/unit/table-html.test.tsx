import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { sanitizeTableHtml, splitByQuote } from '../../src/lib/table-html'

describe('sanitizeTableHtml', () => {
  it('оставляет таблицу и текст, удаляет скрипты, картинки и обработчики', () => {
    const html =
      '<table><thead><tr><th colspan="2">Показатель</th></tr></thead>' +
      '<tbody><tr><td onclick="alert(1)">Прибыль</td><td>1 508,6</td></tr></tbody></table>' +
      '<script>alert(2)</script><img src="x" onerror="alert(3)"><iframe src="https://evil"></iframe>'

    const { container } = render(<div>{sanitizeTableHtml(html)}</div>)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('iframe')).toBeNull()
    expect(container.querySelector('td')?.getAttribute('onclick')).toBeNull()
    expect(container.querySelector('th')?.getAttribute('colspan')).toBe('2')
    expect(container.textContent).toContain('Прибыль')
    expect(container.textContent).toContain('1 508,6')
  })

  it('не падает на пустом html', () => {
    const { container } = render(<div>{sanitizeTableHtml('')}</div>)
    expect(container.firstElementChild).toBeEmptyDOMElement()
  })
})

describe('splitByQuote', () => {
  it('выделяет цитату в полном тексте', () => {
    const parts = splitByQuote('До. Чистая прибыль выросла. После.', 'Чистая прибыль выросла.')
    expect(parts.map((part) => part.match)).toEqual([false, true, false])
    expect(parts[1]?.text).toBe('Чистая прибыль выросла.')
  })

  it('не выдумывает позицию, если цитаты нет в тексте', () => {
    expect(splitByQuote('Текст без совпадений', 'другая цитата')).toEqual([
      { text: 'Текст без совпадений', match: false },
    ])
  })
})
