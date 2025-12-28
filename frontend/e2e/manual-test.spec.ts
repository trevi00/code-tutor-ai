import { test, expect } from '@playwright/test';

test.describe('Manual Browser Test', () => {
  test('should complete full user flow', async ({ page }) => {
    // 1. Go to home page
    await page.goto('/');
    await expect(page).toHaveTitle(/Code Tutor/);
    console.log('✅ Home page loaded');

    // Take screenshot
    await page.screenshot({ path: 'test-results/01-home.png' });

    // 2. Navigate to login (using English text)
    await page.click('text=Login');
    await expect(page).toHaveURL(/login/);
    console.log('✅ Login page loaded');
    await page.screenshot({ path: 'test-results/02-login.png' });

    // 3. Login with demo user
    await page.fill('input[type="email"]', 'demo@example.com');
    await page.fill('input[type="password"]', 'DemoPass123');
    await page.click('button[type="submit"]');

    // Wait for redirect after login (redirects to /problems)
    await page.waitForURL(/problems/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Login successful - redirected to Problems');
    await page.screenshot({ path: 'test-results/03-problems.png' });

    // Verify problems list is visible
    await expect(page.locator('text=Two Sum')).toBeVisible({ timeout: 5000 });
    console.log('✅ Problems list loaded');

    // 4. Go to Dashboard
    await page.click('text=Dashboard');
    await page.waitForURL(/dashboard/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Dashboard loaded');
    await page.screenshot({ path: 'test-results/04-dashboard.png' });

    // 5. Open user menu and go to Profile
    await page.click('button:has-text("demouser")');
    await page.waitForTimeout(500);
    await page.click('text=My Profile');
    await page.waitForURL(/profile/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Profile page loaded');
    await page.screenshot({ path: 'test-results/05-profile.png' });

    // Verify profile shows bio
    await expect(page.locator('text=Hello I am learning to code')).toBeVisible({ timeout: 5000 });
    console.log('✅ Bio is displayed correctly');

    // 6. Open user menu and go to Settings
    await page.click('button:has-text("demouser")');
    await page.waitForTimeout(500);
    await page.click('text=Settings');
    await page.waitForURL(/settings/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Settings page loaded');
    await page.screenshot({ path: 'test-results/06-settings.png' });

    // Verify settings sections are visible
    await expect(page.locator('text=계정 정보')).toBeVisible({ timeout: 5000 });
    console.log('✅ Settings content loaded');

    // 7. Open user menu and go to Submissions
    await page.click('button:has-text("demouser")');
    await page.waitForTimeout(500);
    await page.click('text=Submissions');
    await page.waitForURL(/submissions/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ Submissions page loaded');
    await page.screenshot({ path: 'test-results/07-submissions.png' });

    // 8. Go to AI Tutor
    await page.click('text=AI Tutor');
    await page.waitForURL(/chat/, { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    console.log('✅ AI Tutor page loaded');
    await page.screenshot({ path: 'test-results/08-chat.png' });

    console.log('🎉 All tests passed!');
  });
});
