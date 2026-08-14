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
    // Logging out blacklists the access token on the backend. Obtain a
    // dedicated token via the refresh endpoint (its own rate-limit bucket,
    // no login attempt consumed) so the shared storage-state access token
    // used by the other (parallel) authenticated tests is not invalidated.
    await page.goto('/');
    const refreshToken = await page.evaluate(() => localStorage.getItem('refresh_token'));
    if (!refreshToken) {
      throw new Error('No refresh_token in storage state');
    }
    const refreshResp = await page.request.post('http://localhost:8000/api/v1/auth/refresh', {
      data: { refresh_token: refreshToken },
    });
    if (!refreshResp.ok()) {
      throw new Error(`Token refresh for logout test failed: HTTP ${refreshResp.status()}`);
    }
    const tokens = (await refreshResp.json()) as {
      access_token: string;
      refresh_token: string;
    };

    await page.evaluate(([access, refresh]) => {
      localStorage.setItem('access_token', access);
      localStorage.setItem('refresh_token', refresh);
    }, [tokens.access_token, tokens.refresh_token]);
    await page.reload();
    await page.waitForLoadState('networkidle');

    // The logout button lives inside the header user menu (a second one exists
    // in the mobile drawer, so scope to the banner). Open the menu first.
    const banner = page.getByRole('banner');
    await banner.getByRole('button', { name: /e2etestuser/i }).click();

    const logoutButton = banner.getByRole('button', { name: /logout|sign out|로그아웃/i });
    await expect(logoutButton).toBeVisible();
    await logoutButton.click();

    // Should redirect to login or home
    await expect(page).toHaveURL(/login|\/$/);
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
