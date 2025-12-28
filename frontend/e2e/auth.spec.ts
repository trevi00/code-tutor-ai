import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should display login page', async ({ page }) => {
    await page.goto('/login');

    // Check login form elements exist
    await expect(page.getByLabel('Email')).toBeVisible();
    await expect(page.getByLabel('Password')).toBeVisible();
    await expect(page.getByRole('button', { name: /Sign In/i })).toBeVisible();
  });

  test('should display register page', async ({ page }) => {
    await page.goto('/register');

    // Check register form elements exist
    await expect(page.getByLabel('Email')).toBeVisible();
    await expect(page.getByLabel('Username')).toBeVisible();
    await expect(page.getByLabel('Password', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: /Create Account|Sign Up/i })).toBeVisible();
  });

  test('should navigate from login to register', async ({ page }) => {
    await page.goto('/login');

    // Find and click register link
    const registerLink = page.getByRole('link', { name: /Sign up/i });
    await registerLink.click();

    await expect(page).toHaveURL(/register/);
  });

  test('should show validation error for empty submit', async ({ page }) => {
    await page.goto('/login');

    // Try to submit empty form - HTML5 validation should prevent
    const submitButton = page.getByRole('button', { name: /Sign In/i });
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
