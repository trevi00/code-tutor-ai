import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { getProblemIdByTitle } from './helpers';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Reuse the authenticated session from auth.setup.ts instead of logging in
// through the UI in every test (the login endpoint is rate-limited to 5/min).
test.use({ storageState: join(__dirname, '../.auth/user.json') });

// Run tests serially to avoid session conflicts
test.describe.configure({ mode: 'serial' });

// Helper function to solve a problem (ids are random per seed run — resolve by title)
async function solveProblem(page: import('@playwright/test').Page, context: import('@playwright/test').BrowserContext, problemTitle: string, code: string) {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);

  // Go to problem solve page
  const problemId = await getProblemIdByTitle(page.request, problemTitle);
  await page.goto(`http://localhost:5173/problems/${problemId}/solve`);
  await page.waitForLoadState('networkidle');

  // Wait for Monaco editor to fully load
  await page.waitForTimeout(3000);

  // Wait for editor and click with retry
  let editorClicked = false;
  for (let attempt = 0; attempt < 5 && !editorClicked; attempt++) {
    try {
      const editorArea = page.locator('.monaco-editor .view-lines');
      await editorArea.waitFor({ state: 'visible', timeout: 10000 });
      await editorArea.click();
      editorClicked = true;
    } catch {
      console.log(`Editor click attempt ${attempt + 1} failed, retrying...`);
      await page.waitForTimeout(2000);
    }
  }

  if (!editorClicked) {
    // Fallback: try clicking the monaco-editor container
    await page.locator('.monaco-editor').first().click();
  }

  await page.waitForTimeout(300);

  // Select all and delete
  await page.keyboard.press('Control+a');
  await page.waitForTimeout(200);
  await page.keyboard.press('Delete');
  await page.waitForTimeout(200);

  // Paste code using clipboard
  await page.evaluate(async (c: string) => {
    await navigator.clipboard.writeText(c);
  }, code);
  await page.keyboard.press('Control+v');
  await page.waitForTimeout(500);

  // Click Submit button
  await page.click('button:has-text("제출")');
  await page.waitForTimeout(8000);

  // Get result
  const resultText = await page.locator('text=/\\d+\\/\\d+ 통과/').textContent();
  console.log(`Result: ${resultText}`);

  return resultText;
}

test.describe('Problem Types Tests', () => {
  test('STACK - 유효한 괄호', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `def solution(s: str) -> bool:
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != pairs[char]:
                return False
    return len(stack) == 0

if __name__ == "__main__":
    s = input().strip()
    print(solution(s))`;

    const result = await solveProblem(
      page, context,
      '유효한 괄호',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-stack.png' });
    expect(result).toContain('5/5');
  });

  test('BINARY_SEARCH - 이진 탐색', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `import json

def solution(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))`;

    const result = await solveProblem(
      page, context,
      '이진 탐색',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-binary-search.png' });
    expect(result).toContain('3/3');
  });

  test('DYNAMIC_PROGRAMMING - 최대 부분배열 합', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `import json

def solution(nums: list[int]) -> int:
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))`;

    const result = await solveProblem(
      page, context,
      '최대 부분배열 합 (카데인)',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-dp.png' });
    expect(result).toContain('3/3');
  });

  test('TREE - 이진 트리 레벨 순회', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def solution(root: TreeNode) -> list[list[int]]:
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))`;

    const result = await solveProblem(
      page, context,
      '이진 트리 레벨 순회',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-tree.png' });
    expect(result).toContain('3/3');
  });

  test('GREEDY - 회의실 배정', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `import json

def solution(meetings: list[list[int]]) -> int:
    if not meetings:
        return 0
    meetings.sort(key=lambda x: x[1])
    count = 1
    end_time = meetings[0][1]
    for start, end in meetings[1:]:
        if start >= end_time:
            count += 1
            end_time = end
    return count

if __name__ == "__main__":
    meetings = json.loads(input())
    print(solution(meetings))`;

    const result = await solveProblem(
      page, context,
      '회의실 배정 (그리디)',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-greedy.png' });
    expect(result).toContain('2/2');
  });

  test('STRING - 문자열 뒤집기', async ({ page, context }) => {
    test.setTimeout(120000);

    const code = `import json

def solution(s: list[str]) -> list[str]:
    return s[::-1]

if __name__ == "__main__":
    s = json.loads(input())
    result = solution(s)
    print(json.dumps(result))`;

    const result = await solveProblem(
      page, context,
      '문자열 뒤집기',
      code
    );

    await page.screenshot({ path: 'e2e/screenshots/type-string.png' });
    expect(result).toContain('2/2');
  });
});
