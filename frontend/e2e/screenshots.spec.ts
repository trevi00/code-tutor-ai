import { test } from '@playwright/test';

test.describe('Screenshots for README', () => {
  const testUser = {
    email: 'e2etest@example.com',
    password: 'TestPassword123!',
  };
  const screenshotDir = '../docs/screenshots';

  test('capture all main pages', async ({ page }) => {
    test.setTimeout(180000); // 3 minutes for AI response
    // Set viewport for consistent screenshots
    await page.setViewportSize({ width: 1280, height: 800 });

    // 1. Home page
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${screenshotDir}/01-home.png` });
    console.log('1. Home page captured');

    // 2. Login page
    await page.goto('/login');
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${screenshotDir}/02-login.png` });
    console.log('2. Login page captured');

    // 3. Login
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });

    // 4. Dashboard
    await page.goto('/dashboard');
    await page.waitForSelector('h1:has-text("대시보드")', { timeout: 10000 });
    await page.waitForTimeout(1000); // Wait for animations
    await page.screenshot({ path: `${screenshotDir}/03-dashboard.png`, fullPage: true });
    console.log('3. Dashboard captured');

    // 5. Problems list
    await page.goto('/problems');
    await page.waitForSelector('table tbody tr', { timeout: 10000 });
    await page.screenshot({ path: `${screenshotDir}/04-problems.png` });
    console.log('4. Problems list captured');

    // 6. Problem solve page
    await page.click('table tbody tr:first-child a');
    await page.waitForURL(/\/problems\/.*\/solve/);
    await page.waitForSelector('.monaco-editor', { timeout: 15000 });
    await page.waitForTimeout(1000); // Wait for Monaco to fully load
    await page.screenshot({ path: `${screenshotDir}/05-solve.png` });
    console.log('5. Problem solve page captured');

    // 7. Patterns page
    await page.goto('/patterns');
    await page.waitForSelector('text=알고리즘 패턴', { timeout: 10000 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${screenshotDir}/06-patterns.png` });
    console.log('6. Patterns page captured');

    // 8. AI Tutor chat
    await page.goto('/chat');
    await page.waitForSelector('text=AI 튜터', { timeout: 10000 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${screenshotDir}/07-chat.png` });
    console.log('7. AI Tutor chat captured');

    console.log('All screenshots captured!');
  });
});
