import { expect, test } from '@playwright/test'

test.describe('Base-style research chat against fixture backend', () => {
  test('welcome → question → real stages → answer → evidence → reload', async ({ page }, testInfo) => {
    await page.goto('/')

    await expect(page.getByRole('heading', { name: 'Чем я могу помочь?' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Поделиться' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Прибыль за 2023' })).toBeVisible()

    await page.screenshot({ path: testInfo.outputPath('base-welcome.png') })

    await page.getByRole('button', { name: 'Прибыль за 2023' }).click()

    await expect(page.getByText('Ход выполнения')).toBeVisible()
    await expect(page.getByText(/1 508,6 млрд/)).toBeVisible({ timeout: 20_000 })

    await page.screenshot({ path: testInfo.outputPath('base-answer.png') })

    await page.getByRole('button', { name: /Открыть источник 1/ }).click()
    const evidence = page.getByRole('dialog', { name: 'Источники и доказательства' })
    await expect(evidence).toBeVisible()
    await expect(evidence.getByText('Годовой отчёт 2023').first()).toBeVisible()
    await expect(evidence.getByText('стр. 12').first()).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(evidence).toBeHidden()

    // Backend must already expose the terminal row before the client reloads.
    const stored = await page.request.get('/api/research/research-e2e/messages')
    expect(stored.ok()).toBeTruthy()
    const storedBody = (await stored.json()) as {
      items: Array<{ role: string; status: string; content: string; evidence: unknown[] }>
    }
    const storedAssistant = storedBody.items.find((item) => item.role === 'assistant')
    expect(storedAssistant?.status).toBe('completed')
    expect(storedAssistant?.content).toContain('1 508,6')
    expect(storedAssistant?.evidence.length).toBeGreaterThan(0)

    await page.reload()
    await expect(page.getByText(/1 508,6 млрд/)).toBeVisible({ timeout: 20_000 })
  })

  test('backend failure is shown inline with code and request_id', async ({ page }) => {
    await page.goto('/')
    const composer = page.getByLabel('Вопрос к агенту')
    await composer.fill('Покажи ошибку backend')
    await composer.press('Enter')

    const alert = page.getByRole('alert').first()
    await expect(alert).toContainText('backend_unavailable', { timeout: 20_000 })
    await expect(alert).toContainText('request_id')
    await expect(page.getByText('Агент готовит ответ…')).toHaveCount(0)
  })

  test('sidebar collapses and keeps the selected research', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Agent RAG')).toBeVisible()
    await page.getByRole('button', { name: 'Скрыть список диалогов' }).click()
    await expect(page.getByRole('button', { name: 'Показать список диалогов' })).toBeVisible()
    await expect(page.getByText('Agent RAG')).toBeHidden()
    await page.getByRole('button', { name: 'Показать список диалогов' }).click()
    await expect(page.getByText('Agent RAG')).toBeVisible()
  })

  test('mobile: research list opens as a drawer', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/')
    await page.getByRole('button', { name: 'Открыть список диалогов' }).click()
    const drawer = page.getByRole('dialog', { name: 'История диалогов' })
    await expect(drawer).toBeVisible()
    await expect(drawer.getByText('E2E диалог')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(drawer).toBeHidden()
  })

  test('light/dark theme toggles the design tokens', async ({ page }) => {
    await page.goto('/')
    const html = page.locator('html')
    await expect(html).not.toHaveClass(/dark/)
    await page.getByRole('button', { name: 'Включить тёмную тему' }).first().click()
    await expect(html).toHaveClass(/dark/)
    // Let the 150ms color transition finish before the comparison screenshot.
    await page.waitForTimeout(400)
    await page.screenshot({ path: 'test-results/base-dark.png' })
    await page.getByRole('button', { name: 'Включить светлую тему' }).first().click()
    await expect(html).not.toHaveClass(/dark/)
  })
})
