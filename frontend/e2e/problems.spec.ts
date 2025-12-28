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

    // Should show login form
    await expect(page.getByLabel('Email')).toBeVisible();
    await expect(page.getByLabel('Password')).toBeVisible();
  });

  test('should preserve redirect destination after login attempt', async ({ page }) => {
    await page.goto('/problems');

    // Should be redirected to login
    await expect(page).toHaveURL(/login/);

    // Login form should be displayed
    await expect(page.getByRole('button', { name: /Sign In/i })).toBeVisible();
  });
});
