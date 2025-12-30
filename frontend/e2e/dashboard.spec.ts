import { test, expect } from '@playwright/test';

test.describe('Dashboard Page', () => {
  const baseUrl = 'http://localhost:5176';
  const testUser = {
    email: 'fronttest@example.com',
    password: 'Password123',
  };

  test('should display dashboard with stats and activity', async ({ page }) => {
    // 1. Login first
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Navigate to dashboard
    await page.goto(`${baseUrl}/dashboard`);
    console.log('2. Navigated to dashboard');

    // 3. Wait for dashboard to load
    await page.waitForSelector('h1:has-text("대시보드")', { timeout: 10000 });
    console.log('3. Dashboard title visible');

    // 4. Check for stat cards
    const statCards = page.locator('.bg-white.rounded-lg.shadow');
    const cardCount = await statCards.count();
    console.log(`4. Found ${cardCount} stat cards`);
    expect(cardCount).toBeGreaterThanOrEqual(4);

    // 5. Check for "푼 문제" stat
    await expect(page.locator('text=푼 문제')).toBeVisible();
    console.log('5. "푼 문제" stat visible');

    // 6. Check for "총 제출" stat
    await expect(page.locator('text=총 제출')).toBeVisible();
    console.log('6. "총 제출" stat visible');

    // 7. Check for "현재 스트릭" stat
    await expect(page.locator('text=현재 스트릭')).toBeVisible();
    console.log('7. "현재 스트릭" stat visible');

    // 8. Check for category progress section
    await expect(page.locator('text=카테고리별 진행률')).toBeVisible();
    console.log('8. Category progress section visible');

    // 9. Check for recent submissions section
    await expect(page.locator('text=최근 제출')).toBeVisible();
    console.log('9. Recent submissions section visible');

    // 10. Check for quick actions
    await expect(page.locator('text=빠른 시작')).toBeVisible();
    console.log('10. Quick actions section visible');

    // 11. Check quick action links
    await expect(page.locator('text=문제 풀기')).toBeVisible();
    await expect(page.locator('text=AI 튜터')).toBeVisible();
    console.log('11. Quick action links visible');

    // Take screenshot
    await page.screenshot({ path: 'test-results/dashboard.png', fullPage: true });
    console.log('Dashboard test completed successfully!');
  });
});
