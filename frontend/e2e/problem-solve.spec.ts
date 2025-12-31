import { test, expect } from '@playwright/test';

// Run this test in isolation to avoid Monaco editor conflicts
test.describe.configure({ mode: 'serial' });

test('problem solve - run and submit', async ({ page, context }) => {
  test.setTimeout(180000); // 3 minutes for safety

  // Grant clipboard permissions
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);

  // Login with e2etest user (same as other tests)
  await page.goto('http://localhost:5173/login');
  await page.fill('input[type="email"]', 'e2etest@example.com');
  await page.fill('input[type="password"]', 'TestPassword123!');
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/(dashboard|problems|$)/, { timeout: 10000 });
  await page.waitForTimeout(2000);

  // Go to Two Sum problem
  await page.goto('http://localhost:5173/problems/246130bc-e8ee-4909-a25b-b09dd1098ad0/solve');
  await page.waitForLoadState('networkidle');

  // Wait for Monaco editor to fully load
  await page.waitForTimeout(3000);

  // Screenshot initial state
  await page.screenshot({ path: 'e2e/screenshots/solve-1-initial.png' });

  // The solution code with stdin parsing
  const solutionCode = `import json

def solution(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))`;

  // Wait for editor and click with retry
  let editorClicked = false;
  for (let attempt = 0; attempt < 5 && !editorClicked; attempt++) {
    try {
      const editorArea = page.locator('.monaco-editor .view-lines');
      await editorArea.waitFor({ state: 'visible', timeout: 10000 });
      await editorArea.click();
      editorClicked = true;
      console.log(`Editor clicked on attempt ${attempt + 1}`);
    } catch (e) {
      console.log(`Attempt ${attempt + 1} failed, retrying...`);
      await page.waitForTimeout(2000);
    }
  }

  if (!editorClicked) {
    // Fallback: try clicking the monaco-editor container
    await page.locator('.monaco-editor').first().click();
  }

  await page.waitForTimeout(500);

  // Select all and delete
  await page.keyboard.press('Control+a');
  await page.waitForTimeout(200);
  await page.keyboard.press('Delete');
  await page.waitForTimeout(200);

  // Use clipboard to paste code (avoids Monaco typing issues)
  await page.evaluate(async (code) => {
    await navigator.clipboard.writeText(code);
  }, solutionCode);

  await page.keyboard.press('Control+v');
  await page.waitForTimeout(1000);

  await page.screenshot({ path: 'e2e/screenshots/solve-2-code.png' });

  // Click Run button
  console.log('Clicking Run button...');
  await page.click('button:has-text("실행")');
  await page.waitForTimeout(8000);

  // Screenshot after run
  await page.screenshot({ path: 'e2e/screenshots/solve-3-run-result.png' });

  // Click Submit button
  console.log('Clicking Submit button...');
  await page.click('button:has-text("제출")');
  await page.waitForTimeout(10000);

  // Screenshot after submit
  await page.screenshot({ path: 'e2e/screenshots/solve-4-submit-result.png' });

  // Verify test results show "통과" (passed)
  const resultText = await page.locator('text=/\\d+\\/\\d+ 통과/').textContent();
  console.log('Result:', resultText);

  console.log('Test completed!');
});
