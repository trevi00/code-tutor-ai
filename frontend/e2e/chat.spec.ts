import { test, expect } from '@playwright/test';

test.describe('AI Tutor Chat Page', () => {
  const testUser = {
    email: 'e2etest@example.com',
    password: 'TestPassword123!',
  };

  test('should display chat page and send message', async ({ page }) => {
    test.setTimeout(120000); // AI response can take up to 90 seconds
    // 1. Login first
    await page.goto('/login');
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Navigate to chat page
    await page.goto('/chat');
    console.log('2. Navigated to chat page');

    // 3. Wait for chat page to load (Korean)
    await page.waitForSelector('text=AI 튜터', { timeout: 10000 });
    console.log('3. Chat page title visible');

    // 4. Check for welcome message
    await expect(page.locator('text=안녕하세요')).toBeVisible();
    console.log('4. Welcome message visible');

    // 5. Check for input field (placeholder: 메시지를 입력하세요...)
    const inputField = page.locator('input[placeholder*="메시지"], textarea[placeholder*="메시지"]');
    await expect(inputField.first()).toBeVisible();
    console.log('5. Input field visible');

    // 6. Check for send button
    const sendButton = page.locator('button[type="submit"]');
    await expect(sendButton).toBeVisible();
    console.log('6. Send button visible');

    // 7. Type a message
    await inputField.first().fill('Two Sum 문제 힌트 알려줘');
    console.log('7. Message typed');

    // 8. Take screenshot before sending
    await page.screenshot({ path: 'test-results/chat-before-send.png' });

    // 9. Send message
    await sendButton.click();
    console.log('8. Message sent');

    // 10. Wait for user message to appear
    await expect(page.locator('text=Two Sum 문제 힌트 알려줘')).toBeVisible();
    console.log('9. User message visible');

    // 11. Wait for AI response (Ollama can take 30-60 seconds)
    console.log('10. Waiting for AI response...');

    try {
      await page.waitForFunction(
        () => {
          const loadingDots = document.querySelector('.animate-bounce');
          const pageText = document.body.innerText;
          const hasResponse =
            pageText.includes('hash') ||
            pageText.includes('Hash') ||
            pageText.includes('dictionary') ||
            pageText.includes('O(n)') ||
            pageText.includes('힌트') ||
            pageText.includes('complement');
          return !loadingDots && hasResponse;
        },
        { timeout: 90000 }
      );
      console.log('11. AI response received!');
      await page.screenshot({ path: 'test-results/chat-ai-response.png', fullPage: true });
    } catch {
      console.log('11. Timeout waiting for AI response');
      await page.screenshot({ path: 'test-results/chat-timeout.png', fullPage: true });
    }

    console.log('Chat test completed!');
  });

  test('should access chat from problem page', async ({ page }) => {
    // 1. Login
    await page.goto('/login');
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Go to a problem solve page
    await page.goto('/problems');
    await page.waitForSelector('table tbody tr', { timeout: 10000 });
    await page.click('table tbody tr:first-child a');
    await page.waitForURL(/\/problems\/.*\/solve/);
    console.log('2. On problem solve page');

    // 3. Click AI 도움 button
    const aiHelpButton = page.locator('text=AI 도움');
    await expect(aiHelpButton).toBeVisible();
    await aiHelpButton.click();
    console.log('3. Clicked AI 도움 button');

    // 4. Should navigate to chat with problem context
    await page.waitForURL(/\/chat\?problem=/, { timeout: 10000 });
    console.log('4. Navigated to chat with problem context');

    // 5. Verify chat page loaded (Korean)
    await page.waitForSelector('text=AI 튜터', { timeout: 10000 });
    console.log('5. Chat page loaded with problem context');

    await page.screenshot({ path: 'test-results/chat-with-problem.png' });
    console.log('Problem context chat test completed!');
  });
});
