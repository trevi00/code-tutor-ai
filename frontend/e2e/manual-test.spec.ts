import { test, expect } from '@playwright/test';

test.describe('Manual Browser Test', () => {
  test('should complete full user flow', async ({ page }) => {
    // 1. Go to home page
    await page.goto('/');
    await expect(page).toHaveTitle(/Code Tutor/);
    // Wait past the suspense "로딩 중..." fallback — screenshotting before the
    // first real frame is composited makes CDP captureScreenshot fail under load
    await page.waitForLoadState('networkidle');
    console.log('✅ Home page loaded');

    // Take screenshot
    await page.screenshot({ path: 'test-results/01-home.png' });

    // 2. Navigate to login (Korean)
    await page.click('text=로그인');
    await expect(page).toHaveURL(/login/);
    await page.waitForLoadState('networkidle');
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

    // Verify problems list is visible: search for a known seeded problem
    // (the catalog spans multiple pages, so filter by title first)
    await page.getByPlaceholder('문제 검색...').fill('두 수의 합');
    await expect(page.locator('main').getByText('두 수의 합').first()).toBeVisible({ timeout: 5000 });
    console.log('✅ Problems list loaded');

    const banner = page.getByRole('banner');

    // 4. Go to Dashboard (inside the 분석 nav dropdown — click opens it;
    // hover alone is racy: a transient page reflow fires a spurious
    // mouseleave that closes the menu before the link click lands)
    await banner.getByRole('button', { name: '분석' }).click();
    await banner.getByRole('link', { name: '대시보드' }).click();
    await page.waitForURL(/dashboard/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Dashboard loaded');
    await page.screenshot({ path: 'test-results/04-dashboard.png' });

    // 5. Go to Patterns page (inside the 학습 nav dropdown — click opens it)
    await banner.getByRole('button', { name: '학습' }).click();
    await banner.getByRole('link', { name: '패턴' }).click();
    await page.waitForURL(/patterns/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Patterns page loaded');
    await page.screenshot({ path: 'test-results/05-patterns.png' });

    // Verify patterns list is visible (use category name that exists)
    await expect(page.locator('text=/배열|Array|ARRAY|패턴/i').first()).toBeVisible({ timeout: 5000 });
    console.log('✅ Patterns list loaded');

    // 6. Go to AI Tutor (visible header link)
    await banner.getByRole('link', { name: 'AI 튜터' }).click();
    await page.waitForURL(/chat/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ AI Tutor page loaded');
    await page.screenshot({ path: 'test-results/06-chat.png' });

    console.log('🎉 All tests passed!');
  });
});
