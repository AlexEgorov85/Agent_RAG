import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

/**
 * Evaluation E2E against the scripted fixture backend.
 *
 * The fixture starts with `evaluation_ui=false` (the five chat specs keep that
 * contract). These tests opt in through a fixture-only switch and use EMPTY
 * evaluation collections: no pass rate, summary or trace is fabricated.
 */

const SECOND_RESEARCH_TITLE = 'Второй диалог (E2E)'

async function setEvaluationUi(request: APIRequestContext, enabled: boolean) {
  const response = await request.post('/api/_fixture/evaluation-ui', { data: { enabled } })
  expect(response.ok()).toBeTruthy()
  const body = (await response.json()) as { evaluation_ui: boolean }
  expect(body.evaluation_ui).toBe(enabled)
}

/** Selects a research in the sidebar (not in the header) and waits for it. */
async function selectResearch(page: Page, title: string) {
  const sidebar = page.getByRole('list', { name: 'Список диалогов' })
  await sidebar.getByRole('button', { name: title }).click()
  await expect(page.getByRole('button', { name: title, exact: true })).toBeVisible()
}

async function expectChatWithResearch(page: Page, title: string) {
  await expect(page.getByRole('button', { name: title, exact: true })).toBeVisible()
  await expect(page.getByLabel('Вопрос к агенту')).toBeVisible()
  const selected = page.locator('button[aria-current="true"]')
  await expect(selected).toHaveCount(1)
  await expect(selected).toContainText(title)
}

async function expectChatShell(page: Page) {
  await expect(page.getByLabel('Вопрос к агенту')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Поделиться' })).toBeDisabled()
}

test.describe('Evaluation navigation against fixture backend', () => {
  test.afterEach(async ({ request }) => {
    await setEvaluationUi(request, false)
  })

  test('chat → Evaluation → tabs/empty states → back, selected research preserved', async ({
    page,
  }, testInfo) => {
    await setEvaluationUi(page.request, true)
    const created = await page.request.post('/api/research', {
      data: { title: SECOND_RESEARCH_TITLE },
    })
    expect(created.status()).toBe(201)

    await page.goto('/')
    await expectChatShell(page)
    // The second research is first in the list, so picking the original one
    // proves that back-navigation restores the user's selection, not index 0.
    await selectResearch(page, 'E2E диалог')

    const evaluationLink = page.getByRole('link', { name: 'Evaluation' })
    await expect(evaluationLink).toBeVisible()
    await expect(evaluationLink).toHaveAttribute('href', '/evaluation')

    await page.screenshot({ path: testInfo.outputPath('eval-chat-header.png') })
    await evaluationLink.click()

    await expect(page).toHaveURL(/\/evaluation$/)
    await expect(page.getByRole('heading', { name: 'Evaluation / Benchmark' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Поделиться' })).toHaveCount(0)

    const tabs = page.getByRole('tab')
    await expect(tabs).toHaveCount(4)
    await expect(page.getByRole('tab', { name: 'Обзор' })).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByText('Запусков ещё не было')).toBeVisible()

    // Fixture is explicit about returning no measured runs.
    const runsResponse = await page.request.get('/api/evaluation/runs')
    expect(runsResponse.ok()).toBeTruthy()
    const runsBody = (await runsResponse.json()) as {
      items: unknown[]
      total: number
      fixture?: string
    }
    expect(runsBody.fixture).toBe('empty-evaluation-fixture')
    expect(runsBody.total).toBe(0)
    expect(runsBody.items).toEqual([])

    await page.getByRole('tab', { name: 'Запуски' }).click()
    await expect(page.getByRole('heading', { name: 'Новый запуск' })).toBeVisible()
    await expect(page.getByText('Запусков не найдено')).toBeVisible()

    await page.getByRole('tab', { name: 'Тесты' }).click()
    await expect(page.getByText('Датасет недоступен')).toBeVisible()
    await expect(page.getByText('Выберите тест')).toBeVisible()

    await page.getByRole('tab', { name: 'Сравнение' }).click()
    await expect(page.getByText('Нужны два завершённых запуска')).toBeVisible()

    await page.screenshot({ path: testInfo.outputPath('eval-empty-tabs.png') })

    await page.getByRole('link', { name: 'Research / Chat' }).click()
    await expect(page).toHaveURL(/\/$/)
    await expectChatWithResearch(page, 'E2E диалог')
  })

  test('evaluation_ui=false hides the link and shows honest evaluation_disabled', async ({
    page,
  }, testInfo) => {
    await setEvaluationUi(page.request, false)
    await page.goto('/')
    await expectChatShell(page)
    await selectResearch(page, 'E2E диалог')
    await expect(page.getByRole('link', { name: 'Evaluation' })).toHaveCount(0)

    // Direct URL must not render an empty evaluation screen.
    await page.goto('/evaluation')
    await expect(page.getByRole('heading', { name: 'Evaluation / Benchmark' })).toBeVisible()
    await expect(page.getByText('Раздел Evaluation отключён (evaluation_disabled)')).toBeVisible()
    await expect(page.getByText(/EVAL_UI_ENABLED=1/)).toBeVisible()
    await expect(page.getByRole('tab')).toHaveCount(0)
    await expect(page.getByText('Запусков ещё не было')).toHaveCount(0)

    await page.screenshot({ path: testInfo.outputPath('eval-disabled.png') })

    // The back button returns to the same chat; the selected research is kept.
    await page.getByRole('button', { name: 'Вернуться к исследованию' }).click()
    await expect(page).toHaveURL(/\/$/)
    await expectChatWithResearch(page, 'E2E диалог')
  })
})
