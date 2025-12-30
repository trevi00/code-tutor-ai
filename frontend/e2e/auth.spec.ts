import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should display login page', async ({ page }) => {
    await page.goto('/login');

    // Check login form elements exist (Korean labels)
    await expect(page.getByLabel('이메일')).toBeVisible();
    await expect(page.getByLabel('비밀번호')).toBeVisible();
    await expect(page.getByRole('button', { name: /로그인/i })).toBeVisible();
  });

  test('should display register page', async ({ page }) => {
    await page.goto('/register');

    // Check register form elements exist (Korean labels)
    await expect(page.getByLabel('이메일')).toBeVisible();
    await expect(page.getByLabel('사용자명')).toBeVisible();
    await expect(page.getByLabel('비밀번호', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: /계정 만들기/i })).toBeVisible();
  });

  test('should navigate from login to register', async ({ page }) => {
    await page.goto('/login');

    // Find and click register link (Korean)
    const registerLink = page.getByRole('link', { name: /회원가입/i });
    await registerLink.click();

    await expect(page).toHaveURL(/register/);
  });

  test('should show validation error for empty submit', async ({ page }) => {
    await page.goto('/login');

    // Try to submit empty form - HTML5 validation should prevent
    const submitButton = page.getByRole('button', { name: /로그인/i });
    await submitButton.click();

    // Form should still be on login page (HTML5 required validation)
    await expect(page).toHaveURL(/login/);
  });

  test('should redirect unauthenticated user from protected route', async ({ page }) => {
    await page.goto('/dashboard');

    // Should redirect to login
    await expect(page).toHaveURL(/login/);
  });
});
