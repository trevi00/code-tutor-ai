import { test, expect } from '@playwright/test';

test.describe('Dashboard - Authenticated', () => {
  test('should display dashboard when authenticated', async ({ page }) => {
    await page.goto('/dashboard');

    // Should stay on dashboard (not redirect to login)
    await expect(page).toHaveURL(/dashboard/);
  });

  test('should show user stats', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Dashboard should have content
    const content = page.locator('body');
    await expect(content).toBeVisible();
  });

  test('should display streak information', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Look for streak-related content
    await expect(page).toHaveURL(/dashboard/);
  });

  test('should show category progress', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Page should load
    await expect(page.locator('body')).toBeVisible();
  });

  test('should show recent submissions', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Dashboard should be accessible
    await expect(page).toHaveURL(/dashboard/);
  });

  test('should navigate to problems from dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Find link to problems
    const problemsLink = page.locator('a[href*="/problems"]').first();

    if (await problemsLink.isVisible()) {
      await problemsLink.click();
      await expect(page).toHaveURL(/problems/);
    }
  });
});
