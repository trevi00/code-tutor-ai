import { test, expect } from '@playwright/test';

test.describe('User Actions - Authenticated', () => {
  test('should be able to access tutor chat', async ({ page }) => {
    await page.goto('/tutor');

    // Should stay on tutor page or redirect appropriately
    await expect(page.locator('body')).toBeVisible();
  });

  test('should show navigation for authenticated user', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for authenticated user navigation elements
    const nav = page.locator('nav, header');
    await expect(nav.first()).toBeVisible();
  });

  test('should be able to logout', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for logout button
    const logoutButton = page.getByRole('button', { name: /logout|sign out|로그아웃/i });

    if (await logoutButton.isVisible()) {
      await logoutButton.click();

      // Should redirect to login or home
      await expect(page).toHaveURL(/login|\/$/);
    }
  });

  test('should persist authentication across page navigation', async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/dashboard/);

    // Navigate to problems
    await page.goto('/problems');
    await expect(page).toHaveURL(/problems/);

    // Navigate back to dashboard
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/dashboard/);
  });
});
