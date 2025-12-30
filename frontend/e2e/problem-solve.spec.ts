import { test, expect } from '@playwright/test';

test.describe('Problem Solving Flow', () => {
  const testUser = {
    email: 'e2etest@example.com',
    password: 'TestPassword123!',
  };

  test('should login, view problems, and access solve page', async ({ page }) => {
    // 1. Go to login page
    await page.goto('/login');
    console.log('1. Login page loaded');

    // 2. Fill login form
    await page.fill('input[type="email"]', testUser.email);
    await page.fill('input[type="password"]', testUser.password);
    await page.click('button[type="submit"]');
    console.log('2. Login submitted');

    // 3. Wait for redirect
    await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
    console.log('3. Redirected after login');

    // 4. Navigate to problems page
    await page.goto('/problems');
    await expect(page.locator('h1')).toContainText('문제');
    console.log('4. Problems page loaded');

    // 5. Wait for problems to load from API
    await page.waitForSelector('table tbody tr', { timeout: 10000 });
    const problemCount = await page.locator('table tbody tr').count();
    console.log(`5. Found ${problemCount} problems`);
    expect(problemCount).toBeGreaterThan(0);

    // 6. Click on first problem
    const firstProblem = page.locator('table tbody tr:first-child a').first();
    const problemTitle = await firstProblem.textContent();
    console.log(`6. Clicking on problem: ${problemTitle}`);
    await firstProblem.click();

    // 7. Wait for solve page
    await page.waitForURL(/\/problems\/.*\/solve/);
    console.log('7. Solve page loaded');

    // 8. Wait for Monaco editor
    await page.waitForSelector('.monaco-editor', { timeout: 15000 });
    console.log('8. Monaco editor loaded');

    // 9. Verify submit button exists (Korean)
    const submitButton = page.locator('button:has-text("제출")');
    await expect(submitButton).toBeVisible();
    console.log('9. Submit button visible');

    // 10. Verify run button exists (Korean)
    const runButton = page.locator('button:has-text("실행")');
    await expect(runButton).toBeVisible();
    console.log('10. Run button visible');

    // Take screenshot
    await page.screenshot({ path: 'test-results/solve-page.png' });
    console.log('Test completed successfully!');
  });
});
