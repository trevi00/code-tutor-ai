import { test, expect } from '@playwright/test';

test.describe('Manual Browser Test', () => {
  test('should complete full user flow', async ({ page }) => {
    // 1. Go to home page
    await page.goto('/');
    await expect(page).toHaveTitle(/Code Tutor/);
    console.log('✅ Home page loaded');

    // Take screenshot
    await page.screenshot({ path: 'test-results/01-home.png' });

    // 2. Navigate to login (Korean)
    await page.click('text=로그인');
    await expect(page).toHaveURL(/login/);
    console.log('✅ Login page loaded');
    await page.screenshot({ path: 'test-results/02-login.png' });

    // 3. Login with test user
    await page.fill('input[type="email"]', 'e2etest@example.com');
    await page.fill('input[type="password"]', 'TestPassword123!');
    await page.click('button[type="submit"]');

    // Wait for redirect after login (redirects to /problems)
    await page.waitForURL(/problems/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Login successful - redirected to Problems');
    await page.screenshot({ path: 'test-results/03-problems.png' });

    // Verify problems list is visible (Korean)
    await expect(page.locator('text=두 수의 합')).toBeVisible({ timeout: 5000 });
    console.log('✅ Problems list loaded');

    // 4. Go to Dashboard (Korean)
    await page.click('text=대시보드');
    await page.waitForURL(/dashboard/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Dashboard loaded');
    await page.screenshot({ path: 'test-results/04-dashboard.png' });

    // 5. Go to Patterns page (Korean)
    await page.click('text=패턴');
    await page.waitForURL(/patterns/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Patterns page loaded');
    await page.screenshot({ path: 'test-results/05-patterns.png' });

    // Verify patterns list is visible (Korean)
    await expect(page.locator('text=투 포인터')).toBeVisible({ timeout: 5000 });
    console.log('✅ Patterns list loaded');

    // 6. Go to AI Tutor (Korean)
    await page.click('text=AI 튜터');
    await page.waitForURL(/chat/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ AI Tutor page loaded');
    await page.screenshot({ path: 'test-results/06-chat.png' });

    console.log('🎉 All tests passed!');
  });
});
