import { test, expect } from '@playwright/test';

test.describe('Problems Page - Protected Route', () => {
  test('should redirect unauthenticated users to login', async ({ page }) => {
    await page.goto('/problems');

    // Unauthenticated users should be redirected
    await expect(page).toHaveURL(/login/);
  });

  test('should show login page when accessing problems without auth', async ({ page }) => {
    await page.goto('/problems');

    // Wait for redirect
    await page.waitForURL(/login/);

    // Should show login form (Korean labels)
    await expect(page.getByLabel('이메일')).toBeVisible();
    await expect(page.getByLabel('비밀번호')).toBeVisible();
  });

  test('should preserve redirect destination after login attempt', async ({ page }) => {
    await page.goto('/problems');

    // Should be redirected to login
    await expect(page).toHaveURL(/login/);

    // Login form should be displayed (Korean)
    await expect(page.getByRole('button', { name: /로그인/i })).toBeVisible();
  });
});
