import { test as setup, expect } from '@playwright/test';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const authFile = join(__dirname, '../.auth/user.json');

setup('authenticate', async ({ page }) => {
  const testEmail = 'e2etest@example.com';
  const testPassword = 'TestPassword123!';
  const testUsername = 'e2etestuser';

  // Step 1: Try to login first (in case user already exists)
  await page.goto('/login');
  await page.waitForLoadState('networkidle');

  // Korean labels
  await page.getByLabel('이메일').fill(testEmail);
  await page.getByLabel('비밀번호').fill(testPassword);
  await page.getByRole('button', { name: /로그인/i }).click();

  // Wait for result
  await page.waitForTimeout(2000);

  // Check if login succeeded
  const currentUrl = page.url();
  if (currentUrl.includes('/problems') || currentUrl.includes('/dashboard')) {
    // Login succeeded, save state and exit
    await page.context().storageState({ path: authFile });
    return;
  }

  // Step 2: Login failed, need to register
  await page.goto('/register');
  await page.waitForLoadState('networkidle');

  await page.getByLabel('이메일').fill(testEmail);
  await page.getByLabel('사용자명').fill(testUsername);
  await page.getByLabel('비밀번호', { exact: true }).fill(testPassword);
  await page.getByLabel('비밀번호 확인').fill(testPassword);

  // Click register button (계정 만들기 = Create Account)
  await page.getByRole('button', { name: /계정 만들기/i }).click();

  // Wait for result - could be success or already exists error
  await page.waitForTimeout(3000);

  // Check if we got an error (user already exists)
  const hasError = await page.locator('text=/already exists|이미 존재/i').isVisible();
  if (hasError) {
    // User already exists, go to login
    console.log('User already exists, proceeding to login');
    await page.goto('/login');
    await page.waitForLoadState('networkidle');
    await page.getByLabel('이메일').fill(testEmail);
    await page.getByLabel('비밀번호').fill(testPassword);
    await page.getByRole('button', { name: /로그인/i }).click();
    await page.waitForURL(/problems|dashboard/, { timeout: 10000 });
    await page.context().storageState({ path: authFile });
    return;
  }

  // Wait for redirect to login after successful registration
  await page.waitForURL(/login/, { timeout: 10000 });

  // Step 3: Now login with the registered user
  await page.waitForLoadState('networkidle');

  await page.getByLabel('이메일').fill(testEmail);
  await page.getByLabel('비밀번호').fill(testPassword);
  await page.getByRole('button', { name: /로그인/i }).click();

  // Wait for redirect to protected route
  await page.waitForURL(/problems|dashboard/, { timeout: 10000 });

  // Save authentication state
  await page.context().storageState({ path: authFile });
});
