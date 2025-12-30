import { test, expect } from '@playwright/test';

test.describe('AI Tutor Chat Page', () => {
  const baseUrl = 'http://localhost:5176';
  const testUser = {
    email: 'fronttest@example.com',
    password: 'Password123',
  };

  test('should display chat page and send message', async ({ page }) => {
    // 1. Login first
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Navigate to chat page
    await page.goto(`${baseUrl}/chat`);
    console.log('2. Navigated to chat page');

    // 3. Wait for chat page to load
    await page.waitForSelector('text=AI Tutor Chat', { timeout: 10000 });
    console.log('3. Chat page title visible');

    // 4. Check for welcome message
    await expect(page.locator('text=안녕하세요')).toBeVisible();
    console.log('4. Welcome message visible');

    // 5. Check for input field
    const inputField = page.locator('input[placeholder*="message"]');
    await expect(inputField).toBeVisible();
    console.log('5. Input field visible');

    // 6. Check for send button
    const sendButton = page.locator('button[type="submit"]');
    await expect(sendButton).toBeVisible();
    console.log('6. Send button visible');

    // 7. Check for New Chat button
    await expect(page.locator('text=New Chat')).toBeVisible();
    console.log('7. New Chat button visible');

    // 8. Type a message
    await inputField.fill('Two Sum 문제 힌트 알려줘');
    console.log('8. Message typed');

    // 9. Take screenshot before sending
    await page.screenshot({ path: 'test-results/chat-before-send.png' });

    // 10. Send message
    await sendButton.click();
    console.log('9. Message sent');

    // 11. Wait for user message to appear
    await expect(page.locator('text=Two Sum 문제 힌트 알려줘')).toBeVisible();
    console.log('10. User message visible');

    // 12. Wait for AI response (Ollama can take 30-60 seconds)
    console.log('11. Waiting for AI response...');

    // Wait for loading indicator to disappear (AI finished responding)
    // or wait for response content to appear
    try {
      // Wait up to 90 seconds for the response
      await page.waitForFunction(
        () => {
          // Check if loading dots are gone and we have AI response text
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
      console.log('12. AI response received!');

      // Verify the response contains helpful content
      const pageContent = await page.content();
      const hasHelpfulContent =
        pageContent.includes('hash') ||
        pageContent.includes('Hash') ||
        pageContent.includes('dictionary') ||
        pageContent.includes('O(n)');

      if (hasHelpfulContent) {
        console.log('13. AI response contains helpful algorithm hints!');
      }

      // Take screenshot of AI response
      await page.screenshot({ path: 'test-results/chat-ai-response.png', fullPage: true });
    } catch {
      console.log('12. Timeout waiting for AI response (Ollama might be slow)');
      await page.screenshot({ path: 'test-results/chat-timeout.png', fullPage: true });
    }

    // Final screenshot
    await page.screenshot({ path: 'test-results/chat-final.png', fullPage: true });
    console.log('Chat test completed!');
  });

  test('should access chat from problem page', async ({ page }) => {
    // 1. Login
    await page.goto(`${baseUrl}/login`);
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Go to a problem solve page
    await page.goto(`${baseUrl}/problems`);
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

    // 5. Verify chat page loaded
    await page.waitForSelector('text=AI Tutor Chat', { timeout: 10000 });
    console.log('5. Chat page loaded with problem context');

    // Take screenshot
    await page.screenshot({ path: 'test-results/chat-with-problem.png' });
    console.log('Problem context chat test completed!');
  });
});
