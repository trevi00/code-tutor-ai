import { test } from '@playwright/test';

test.describe('Screenshots for README', () => {
  const baseUrl = 'http://localhost:5176';
  const testUser = {
    email: 'fronttest@example.com',
    password: 'Password123',
  };
  const screenshotDir = '../docs/screenshots';

  test('capture all main pages', async ({ page }) => {
    test.setTimeout(180000); // 3 minutes for AI response
    // Set viewport for consistent screenshots
    await page.setViewportSize({ width: 1280, height: 800 });

    // 1. Home page
    await page.goto(baseUrl);
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${screenshotDir}/01-home.png` });
    console.log('1. Home page captured');

    // 2. Login page
    await page.goto(`${baseUrl}/login`);
    await page.waitForLoadState('networkidle');
    await page.screenshot({ path: `${screenshotDir}/02-login.png` });
    console.log('2. Login page captured');

    // 3. Login
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });

    // 4. Dashboard
    await page.goto(`${baseUrl}/dashboard`);
    await page.waitForSelector('h1:has-text("대시보드")', { timeout: 10000 });
    await page.waitForTimeout(1000); // Wait for animations
    await page.screenshot({ path: `${screenshotDir}/03-dashboard.png`, fullPage: true });
    console.log('3. Dashboard captured');

    // 5. Problems list
    await page.goto(`${baseUrl}/problems`);
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

    // 7. AI Tutor chat
    await page.goto(`${baseUrl}/chat`);
    await page.waitForSelector('text=AI Tutor Chat', { timeout: 10000 });
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${screenshotDir}/06-chat.png` });
    console.log('6. AI Tutor chat captured');

    // 8. Send a message to AI
    const inputField = page.locator('input[placeholder*="message"]');
    await inputField.fill('Two Sum 문제 힌트 알려줘');
    await page.screenshot({ path: `${screenshotDir}/07-chat-input.png` });

    // Send and wait for response
    await page.click('button[type="submit"]');
    console.log('7. Waiting for AI response...');

    try {
      await page.waitForFunction(
        () => {
          const loadingDots = document.querySelector('.animate-bounce');
          const pageText = document.body.innerText;
          const hasResponse =
            pageText.includes('hash') ||
            pageText.includes('Hash') ||
            pageText.includes('힌트') ||
            pageText.includes('O(n');
          return !loadingDots && hasResponse;
        },
        { timeout: 90000 }
      );
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${screenshotDir}/08-chat-response.png`, fullPage: true });
      console.log('8. AI response captured');
    } catch {
      console.log('8. AI response timeout, capturing current state');
      await page.screenshot({ path: `${screenshotDir}/08-chat-response.png`, fullPage: true });
    }

    console.log('All screenshots captured!');
  });
});
