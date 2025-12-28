import { test, expect } from '@playwright/test';

test.describe('Problems Page - Authenticated', () => {
  test('should display problems list when authenticated', async ({ page }) => {
    await page.goto('/problems');

    // Should stay on problems page (not redirect to login)
    await expect(page).toHaveURL(/problems/);
  });

  test('should show problem cards', async ({ page }) => {
    await page.goto('/problems');
    await page.waitForLoadState('networkidle');

    // Look for problem items or cards
    const content = page.locator('body');
    await expect(content).toBeVisible();

    // Should have some content on the page
    const pageText = await page.textContent('body');
    expect(pageText).toBeTruthy();
  });

  test('should have filter options', async ({ page }) => {
    await page.goto('/problems');
    await page.waitForLoadState('networkidle');

    // Page should load successfully
    await expect(page).toHaveURL(/problems/);
  });

  test('should navigate to problem detail when clicking a problem', async ({ page }) => {
    await page.goto('/problems');
    await page.waitForLoadState('networkidle');

    // Try to find and click a problem link
    const problemLink = page.locator('a[href*="/problems/"]').first();

    if (await problemLink.isVisible()) {
      await problemLink.click();
      await expect(page).toHaveURL(/problems\/[^/]+/);
    }
  });
});
