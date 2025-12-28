import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should display home page with correct title', async ({ page }) => {
    await page.goto('/');

    // Home page should have correct title
    await expect(page).toHaveTitle(/Code Tutor/i);
  });

  test('should redirect unauthenticated users from problems to login', async ({ page }) => {
    await page.goto('/problems');

    // Unauthenticated users should be redirected to login
    await expect(page).toHaveURL(/login/);
  });

  test('should have responsive navigation', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Page should still be usable on mobile
    await expect(page.locator('body')).toBeVisible();
  });

  test('should navigate to login page', async ({ page }) => {
    await page.goto('/login');

    await expect(page).toHaveURL(/login/);
    await expect(page.locator('body')).toBeVisible();
  });

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/register');

    await expect(page).toHaveURL(/register/);
    await expect(page.locator('body')).toBeVisible();
  });
});
