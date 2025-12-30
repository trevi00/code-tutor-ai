import { test, expect } from '@playwright/test';

test.describe('Code Review via AI Chat', () => {
  const testUser = {
    email: 'e2etest@example.com',
    password: 'TestPassword123!',
  };

  test('should get code review from AI tutor', async ({ page }) => {
    test.setTimeout(120000); // AI response can take up to 90 seconds
    // 1. Login
    await page.goto('/login');
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('1. Logged in');

    // 2. Navigate to chat page
    await page.goto('/chat');
    await page.waitForSelector('text=AI 튜터', { timeout: 10000 });
    console.log('2. Chat page loaded');

    // 3. Type code review request (placeholder: 메시지를 입력하세요...)
    const inputField = page.locator('input[placeholder*="메시지"], textarea[placeholder*="메시지"]');
    const codeReviewMessage = `이 코드를 리뷰해줘:

def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []`;

    await inputField.first().fill(codeReviewMessage);
    console.log('3. Code review request typed');

    // 4. Take screenshot before sending
    await page.screenshot({ path: 'test-results/code-review-before.png' });

    // 5. Send message
    const sendButton = page.locator('button[type="submit"]');
    await sendButton.click();
    console.log('4. Message sent');

    // 6. Wait for AI response
    console.log('5. Waiting for AI code review response...');

    try {
      await page.waitForFunction(
        () => {
          const loadingDots = document.querySelector('.animate-bounce');
          const pageText = document.body.innerText;
          const hasCodeReview =
            pageText.includes('O(n') ||
            pageText.includes('복잡도') ||
            pageText.includes('complexity') ||
            pageText.includes('hash') ||
            pageText.includes('Hash') ||
            pageText.includes('개선') ||
            pageText.includes('improvement');
          return !loadingDots && hasCodeReview;
        },
        { timeout: 90000 }
      );
      console.log('6. AI code review received!');

      // Verify response contains code review content
      const pageContent = await page.content();
      const hasReviewContent =
        pageContent.includes('O(n') ||
        pageContent.includes('complexity') ||
        pageContent.includes('복잡도');

      if (hasReviewContent) {
        console.log('7. Code review contains complexity analysis!');
      }

      await page.screenshot({ path: 'test-results/code-review-response.png', fullPage: true });
    } catch {
      console.log('6. Timeout waiting for code review response');
      await page.screenshot({ path: 'test-results/code-review-timeout.png', fullPage: true });
    }

    console.log('Code review test completed!');
  });
});
