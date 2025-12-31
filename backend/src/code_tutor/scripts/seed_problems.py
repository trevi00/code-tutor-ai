"""Seed script for algorithm problems with pattern information"""

import asyncio
from uuid import uuid4

from code_tutor.shared.infrastructure.database import get_session_context, init_db
from code_tutor.learning.infrastructure.models import ProblemModel, TestCaseModel, SubmissionModel
from code_tutor.learning.domain.value_objects import Category, Difficulty
# Import all models for proper table creation
from code_tutor.identity.infrastructure.models import UserModel
from code_tutor.tutor.infrastructure.models import ConversationModel, MessageModel


SEED_PROBLEMS = [
    {
        "title": "두 수의 합",
        "description": """정수 배열 `nums`와 정수 `target`이 주어집니다.

두 수의 합이 `target`이 되는 두 인덱스를 반환하세요.

**각 입력에는 정확히 하나의 해가 존재합니다.**

### 예시
```
입력: nums = [2, 7, 11, 15], target = 9
출력: [0, 1]
설명: nums[0] + nums[1] = 2 + 7 = 9
```

### 제약조건
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- 같은 요소를 두 번 사용할 수 없습니다.""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9",
        "hints": [
            "먼저 브루트포스로 O(n²) 해결법을 생각해보세요.",
            "해시맵을 사용하면 O(n)에 해결할 수 있습니다.",
            "각 숫자의 보수(target - num)를 저장하고 조회하세요.",
        ],
        "pattern_ids": ["two-pointers"],
        "pattern_explanation": "해시맵을 사용해 각 원소를 순회하며 필요한 보수를 O(1)에 조회합니다. 정렬된 배열이라면 투 포인터도 가능합니다.",
        "approach_hint": "각 숫자를 순회하면서 target - num이 이미 해시맵에 있는지 확인하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """import json

def solution(nums: list[int], target: int) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
""",
        "reference_solution": """import json

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
    print(solution(nums, target))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2, 7, 11, 15]\n9", "expected_output": "[0, 1]", "is_sample": True},
            {"input": "[3, 2, 4]\n6", "expected_output": "[1, 2]", "is_sample": True},
            {"input": "[3, 3]\n6", "expected_output": "[0, 1]", "is_sample": False},
        ],
    },
    {
        "title": "유효한 괄호",
        "description": """여는 괄호 `(`, `{`, `[`와 닫는 괄호 `)`, `}`, `]`로만 이루어진 문자열 `s`가 주어집니다.

괄호가 유효하게 짝지어졌는지 판별하세요.

### 유효한 조건
1. 여는 괄호는 같은 종류의 닫는 괄호로 닫혀야 합니다.
2. 여는 괄호는 올바른 순서로 닫혀야 합니다.

### 예시
```
입력: s = "()"
출력: true

입력: s = "()[]{}"
출력: true

입력: s = "(]"
출력: false
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.STACK,
        "constraints": "1 <= s.length <= 10^4",
        "hints": [
            "스택 자료구조를 사용하세요.",
            "여는 괄호는 스택에 push, 닫는 괄호는 pop 후 매칭 확인",
            "스택이 비어있는데 pop을 시도하면 유효하지 않습니다.",
        ],
        "pattern_ids": ["monotonic-stack"],
        "pattern_explanation": "스택을 사용하여 여는 괄호를 저장하고, 닫는 괄호가 나올 때 스택 top과 매칭되는지 확인합니다.",
        "approach_hint": "해시맵으로 괄호 쌍을 정의하고, 스택으로 여는 괄호를 관리하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """def solution(s: str) -> bool:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
""",
        "reference_solution": """def solution(s: str) -> bool:
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
    print(solution(s))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "()", "expected_output": "True", "is_sample": True},
            {"input": "()[]{}", "expected_output": "True", "is_sample": True},
            {"input": "(]", "expected_output": "False", "is_sample": True},
            {"input": "([)]", "expected_output": "False", "is_sample": False},
            {"input": "{[]}", "expected_output": "True", "is_sample": False},
        ],
    },
    {
        "title": "이진 탐색",
        "description": """정렬된 정수 배열 `nums`와 정수 `target`이 주어집니다.

`target`이 배열에 있으면 해당 인덱스를, 없으면 -1을 반환하세요.

**시간 복잡도 O(log n)으로 해결해야 합니다.**

### 예시
```
입력: nums = [-1, 0, 3, 5, 9, 12], target = 9
출력: 4

입력: nums = [-1, 0, 3, 5, 9, 12], target = 2
출력: -1
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.BINARY_SEARCH,
        "constraints": "1 <= nums.length <= 10^4, 배열은 정렬되어 있습니다",
        "hints": [
            "left와 right 포인터를 사용하세요.",
            "중간값(mid)과 target을 비교하세요.",
            "left <= right 조건으로 반복하세요.",
        ],
        "pattern_ids": ["binary-search"],
        "pattern_explanation": "정렬된 배열에서 이진 탐색을 사용합니다. 매 반복마다 탐색 범위를 절반으로 줄여 O(log n)에 찾습니다.",
        "approach_hint": "mid = (left + right) // 2로 중간값을 구하고, target과 비교하여 범위를 좁혀가세요.",
        "time_complexity_hint": "O(log n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """import json

def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
""",
        "reference_solution": """import json

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
    print(solution(nums, target))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[-1, 0, 3, 5, 9, 12]\n9", "expected_output": "4", "is_sample": True},
            {"input": "[-1, 0, 3, 5, 9, 12]\n2", "expected_output": "-1", "is_sample": True},
            {"input": "[5]\n5", "expected_output": "0", "is_sample": False},
        ],
    },
    {
        "title": "최대 부분배열 합 (카데인)",
        "description": """정수 배열 `nums`가 주어질 때, 합이 최대인 연속 부분배열을 찾아 그 합을 반환하세요.

**부분배열은 연속된 원소들로 이루어져야 합니다.**

### 예시
```
입력: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
출력: 6
설명: [4, -1, 2, 1]의 합 = 6

입력: nums = [1]
출력: 1
```

### 힌트
Kadane's Algorithm을 사용하면 O(n)에 해결할 수 있습니다.""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= nums.length <= 10^5, -10^4 <= nums[i] <= 10^4",
        "hints": [
            "각 위치에서 '현재까지의 최대 합'을 계산하세요.",
            "현재 원소를 포함한 새 배열 시작 vs 이전 배열에 현재 원소 추가",
            "current_sum = max(nums[i], current_sum + nums[i])",
        ],
        "pattern_ids": ["dp"],
        "pattern_explanation": "카데인 알고리즘(Kadane's Algorithm)은 동적 프로그래밍의 일종으로, 각 위치에서 '여기서 끝나는 최대 부분배열 합'을 계산합니다.",
        "approach_hint": "음수가 나오면 새로 시작하는 것이 나을 수 있습니다. max(현재 원소, 이전 합 + 현재 원소)를 비교하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
""",
        "reference_solution": """import json

def solution(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[-2, 1, -3, 4, -1, 2, 1, -5, 4]", "expected_output": "6", "is_sample": True},
            {"input": "[1]", "expected_output": "1", "is_sample": True},
            {"input": "[5, 4, -1, 7, 8]", "expected_output": "23", "is_sample": False},
        ],
    },
    {
        "title": "섬의 개수",
        "description": """'1'(육지)과 '0'(물)로 이루어진 2D 그리드가 주어질 때, 섬의 개수를 반환하세요.

섬은 물로 둘러싸여 있으며, 가로나 세로로 인접한 육지를 연결하여 형성됩니다.

### 예시
```
입력:
[
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
출력: 3
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.GRAPH,
        "constraints": "m == grid.length, n == grid[i].length, 1 <= m, n <= 300",
        "hints": [
            "그리드를 순회하면서 '1'을 발견하면 탐색을 시작하세요.",
            "DFS 또는 BFS로 연결된 모든 육지를 방문하세요.",
            "방문한 육지는 '0'으로 변경하여 중복 방문을 방지하세요.",
        ],
        "pattern_ids": ["dfs", "bfs", "matrix-traversal"],
        "pattern_explanation": "DFS/BFS를 사용한 그래프 탐색 문제입니다. 연결된 컴포넌트의 개수를 세는 전형적인 패턴입니다.",
        "approach_hint": "각 셀을 순회하며 육지를 발견하면 DFS로 연결된 모든 육지를 방문(마킹)하고 카운트를 증가시키세요.",
        "time_complexity_hint": "O(m × n)",
        "space_complexity_hint": "O(m × n)",
        "solution_template": """import json

def solution(grid: list[list[str]]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    grid = json.loads(input())
    print(solution(grid))
""",
        "reference_solution": """import json

def solution(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # 방문 표시
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count


if __name__ == "__main__":
    grid = json.loads(input())
    print(solution(grid))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": '[[\"1\",\"1\",\"0\",\"0\",\"0\"],[\"1\",\"1\",\"0\",\"0\",\"0\"],[\"0\",\"0\",\"1\",\"0\",\"0\"],[\"0\",\"0\",\"0\",\"1\",\"1\"]]', "expected_output": "3", "is_sample": True},
            {"input": '[[\"1\",\"1\",\"1\"],[\"0\",\"1\",\"0\"],[\"1\",\"1\",\"1\"]]', "expected_output": "1", "is_sample": True},
        ],
    },
    {
        "title": "최장 증가 부분수열 (LIS)",
        "description": """정수 배열 `nums`가 주어질 때, 가장 긴 **증가하는 부분수열**의 길이를 반환하세요.

부분수열은 원래 배열에서 일부 원소를 삭제(또는 삭제하지 않고)하여 순서를 바꾸지 않고 얻은 수열입니다.

### 예시
```
입력: nums = [10, 9, 2, 5, 3, 7, 101, 18]
출력: 4
설명: 가장 긴 증가 부분수열은 [2, 3, 7, 101] 또는 [2, 3, 7, 18]

입력: nums = [0, 1, 0, 3, 2, 3]
출력: 4
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= nums.length <= 2500, -10^4 <= nums[i] <= 10^4",
        "hints": [
            "dp[i]는 nums[i]로 끝나는 LIS의 길이입니다.",
            "이진 탐색을 사용하면 O(n log n)에 해결할 수 있습니다.",
            "patience sorting 기법을 적용해보세요.",
        ],
        "pattern_ids": ["dp", "binary-search"],
        "pattern_explanation": "DP 해법은 O(n²)이고, 이진 탐색을 사용한 최적화 해법은 O(n log n)입니다. Patience Sorting 개념을 활용합니다.",
        "approach_hint": "각 위치에서 '이 원소로 끝나는 최장 증가 부분수열'을 계산하세요.",
        "time_complexity_hint": "O(n²) 또는 O(n log n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
""",
        "reference_solution": """import json

def solution(nums: list[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[10, 9, 2, 5, 3, 7, 101, 18]", "expected_output": "4", "is_sample": True},
            {"input": "[0, 1, 0, 3, 2, 3]", "expected_output": "4", "is_sample": True},
            {"input": "[7, 7, 7, 7, 7, 7, 7]", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "동전 거스름돈",
        "description": """서로 다른 금액의 동전 배열 `coins`와 총 금액 `amount`가 주어집니다.

금액을 만들 수 있는 **최소 동전 개수**를 반환하세요. 만들 수 없다면 -1을 반환하세요.

### 예시
```
입력: coins = [1, 2, 5], amount = 11
출력: 3
설명: 11 = 5 + 5 + 1

입력: coins = [2], amount = 3
출력: -1
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= coins.length <= 12, 0 <= amount <= 10^4",
        "hints": [
            "dp[i]는 금액 i를 만드는 최소 동전 수입니다.",
            "각 동전에 대해 dp[i] = min(dp[i], dp[i - coin] + 1)",
            "bottom-up DP로 해결하세요.",
        ],
        "pattern_ids": ["dp"],
        "pattern_explanation": "무한 배낭 문제(Unbounded Knapsack) 변형입니다. 각 금액에 대해 최소 동전 수를 저장하며 bottom-up으로 계산합니다.",
        "approach_hint": "dp[0] = 0으로 시작하고, 각 금액 i에 대해 모든 동전을 시도해보세요.",
        "time_complexity_hint": "O(amount × coins)",
        "space_complexity_hint": "O(amount)",
        "solution_template": """import json

def solution(coins: list[int], amount: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    coins = json.loads(input())
    amount = int(input())
    print(solution(coins, amount))
""",
        "reference_solution": """import json

def solution(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


if __name__ == "__main__":
    coins = json.loads(input())
    amount = int(input())
    print(solution(coins, amount))
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1, 2, 5]\n11", "expected_output": "3", "is_sample": True},
            {"input": "[2]\n3", "expected_output": "-1", "is_sample": True},
            {"input": "[1]\n0", "expected_output": "0", "is_sample": False},
        ],
    },
    {
        "title": "이진 트리 레벨 순회",
        "description": """이진 트리가 주어질 때, 노드 값을 **레벨 순서**로 반환하세요.

같은 레벨의 노드는 왼쪽에서 오른쪽으로 순회합니다.

### 예시
```
    3
   / \\
  9  20
    /  \\
   15   7

출력: [[3], [9, 20], [15, 7]]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.TREE,
        "constraints": "0 <= 노드 수 <= 2000",
        "hints": [
            "BFS(너비 우선 탐색)를 사용하세요.",
            "큐를 사용하여 레벨별로 처리하세요.",
            "각 레벨의 노드 수만큼 pop하고 결과에 추가하세요.",
        ],
        "pattern_ids": ["bfs"],
        "pattern_explanation": "BFS는 레벨 순회에 완벽히 적합합니다. 큐를 사용하여 현재 레벨의 모든 노드를 처리하고 다음 레벨로 넘어갑니다.",
        "approach_hint": "큐에 노드를 넣고, 현재 큐의 크기만큼 pop하여 한 레벨을 처리하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 9, 20, null, null, 15, 7]", "expected_output": "[[3], [9, 20], [15, 7]]", "is_sample": True},
            {"input": "[1]", "expected_output": "[[1]]", "is_sample": True},
            {"input": "[]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "회의실 배정 (그리디)",
        "description": """회의 시작/종료 시간 배열이 주어집니다.

한 회의실에서 **최대한 많은 회의**를 진행하려면 몇 개의 회의를 진행할 수 있나요?

### 예시
```
입력: meetings = [[0, 30], [5, 10], [15, 20]]
출력: 2
설명: [5, 10]과 [15, 20] 회의를 진행할 수 있습니다.
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.GREEDY,
        "constraints": "1 <= meetings.length <= 10^4",
        "hints": [
            "종료 시간이 빠른 순으로 정렬하세요.",
            "현재 회의가 이전 회의와 겹치지 않으면 선택하세요.",
            "그리디하게 '가장 빨리 끝나는 회의'를 선택합니다.",
        ],
        "pattern_ids": ["greedy", "merge-intervals"],
        "pattern_explanation": "활동 선택 문제(Activity Selection)는 대표적인 그리디 문제입니다. 종료 시간 기준 정렬 후 겹치지 않는 회의를 선택합니다.",
        "approach_hint": "종료 시간이 빠른 회의부터 선택하면 이후에 더 많은 회의를 배치할 수 있습니다.",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(meetings: list[list[int]]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([[0, 30], [5, 10], [15, 20]]))  # 2
    print(solution([[7, 10], [2, 4]]))              # 2
""",
        "reference_solution": """def solution(meetings: list[list[int]]) -> int:
    if not meetings:
        return 0

    # 종료 시간 기준 정렬
    meetings.sort(key=lambda x: x[1])

    count = 1
    last_end = meetings[0][1]

    for start, end in meetings[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[[0, 30], [5, 10], [15, 20]]", "expected_output": "2", "is_sample": True},
            {"input": "[[7, 10], [2, 4]]", "expected_output": "2", "is_sample": True},
        ],
    },
    {
        "title": "N-Queen (백트래킹)",
        "description": """n×n 체스판에 n개의 퀸을 배치하세요.

퀸은 같은 행, 열, 대각선에 다른 퀸이 없어야 합니다.

### 예시
```
입력: n = 4
출력: 2
설명: 4×4 보드에 4개의 퀸을 배치하는 방법은 2가지입니다.
```""",
        "difficulty": Difficulty.HARD,
        "category": Category.BACKTRACKING,
        "constraints": "1 <= n <= 9",
        "hints": [
            "행 단위로 퀸을 배치해 나가세요.",
            "각 열, 대각선을 집합으로 관리하세요.",
            "유효하지 않으면 백트래킹하세요.",
        ],
        "pattern_ids": ["backtracking"],
        "pattern_explanation": "백트래킹의 전형적인 문제입니다. 조건을 만족하는 배치를 찾고, 불가능하면 되돌아갑니다.",
        "approach_hint": "각 행에 하나씩 퀸을 배치하고, 열/대각선 충돌을 체크하며 진행하세요.",
        "time_complexity_hint": "O(n!)",
        "space_complexity_hint": "O(n)",
        "solution_template": """def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(4))  # 2
    print(solution(1))  # 1
""",
        "reference_solution": """def solution(n: int) -> int:
    count = 0
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "4", "expected_output": "2", "is_sample": True},
            {"input": "1", "expected_output": "1", "is_sample": True},
            {"input": "8", "expected_output": "92", "is_sample": False},
        ],
    },
    {
        "title": "최대 합 슬라이딩 윈도우",
        "description": """정수 배열 `nums`와 정수 `k`가 주어집니다.

길이 `k`인 연속 부분배열 중 합이 최대인 값을 반환하세요.

### 예시
```
입력: nums = [2, 1, 5, 1, 3, 2], k = 3
출력: 9
설명: 길이 3인 부분배열 [5, 1, 3]의 합이 9로 최대입니다.
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "1 <= k <= nums.length <= 10^5",
        "hints": [
            "처음 k개 원소의 합을 먼저 계산하세요.",
            "윈도우를 오른쪽으로 한 칸씩 이동하세요.",
            "왼쪽 원소를 빼고 오른쪽 원소를 더하세요.",
        ],
        "pattern_ids": ["sliding-window"],
        "pattern_explanation": "슬라이딩 윈도우 패턴을 사용합니다. 고정 크기의 윈도우를 이동시키며 합을 O(1)에 업데이트합니다.",
        "approach_hint": "window_sum = window_sum - nums[i-k] + nums[i]로 효율적으로 계산하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int], k: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2, 1, 5, 1, 3, 2], 3))  # 9
""",
        "reference_solution": """def solution(nums: list[int], k: int) -> int:
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2, 1, 5, 1, 3, 2]\n3", "expected_output": "9", "is_sample": True},
            {"input": "[2, 3, 4, 1, 5]\n2", "expected_output": "7", "is_sample": True},
            {"input": "[1]\n1", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "연결 리스트 사이클 감지",
        "description": """연결 리스트의 head가 주어집니다.

연결 리스트에 사이클이 있는지 확인하세요.

사이클이 있으면 True, 없으면 False를 반환하세요.

### 예시
```
입력: head = [3, 2, 0, -4], 마지막 노드가 인덱스 1의 노드를 가리킴
출력: True
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.LINKED_LIST,
        "constraints": "0 <= 노드 수 <= 10^4",
        "hints": [
            "두 포인터를 사용하세요.",
            "느린 포인터는 한 칸, 빠른 포인터는 두 칸씩 이동합니다.",
            "사이클이 있으면 두 포인터가 만나게 됩니다.",
        ],
        "pattern_ids": ["fast-slow-pointers"],
        "pattern_explanation": "Floyd의 사이클 감지 알고리즘입니다. 빠른 포인터와 느린 포인터가 만나면 사이클이 존재합니다.",
        "approach_hint": "토끼와 거북이 알고리즘을 생각해보세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(head: ListNode) -> bool:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(head: ListNode) -> bool:
    if not head or not head.next:
        return False

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 2, 0, -4]\n1", "expected_output": "True", "is_sample": True},
            {"input": "[1, 2]\n0", "expected_output": "True", "is_sample": True},
            {"input": "[1]\n-1", "expected_output": "False", "is_sample": False},
        ],
    },
    {
        "title": "K번째 큰 수",
        "description": """정수 배열 `nums`와 정수 `k`가 주어집니다.

배열에서 `k`번째로 큰 원소를 반환하세요.

### 예시
```
입력: nums = [3, 2, 1, 5, 6, 4], k = 2
출력: 5

입력: nums = [3, 2, 3, 1, 2, 4, 5, 5, 6], k = 4
출력: 4
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.HEAP,
        "constraints": "1 <= k <= nums.length <= 10^5",
        "hints": [
            "정렬하면 O(n log n)에 해결됩니다.",
            "힙을 사용하면 O(n log k)에 해결할 수 있습니다.",
            "크기 k인 최소 힙을 유지하세요.",
        ],
        "pattern_ids": ["top-k-elements"],
        "pattern_explanation": "Top K 패턴의 전형적인 문제입니다. 최소 힙을 사용하면 효율적으로 K번째 큰 수를 찾을 수 있습니다.",
        "approach_hint": "크기 k인 최소 힙을 유지하면, 힙의 root가 k번째 큰 수입니다.",
        "time_complexity_hint": "O(n log k)",
        "space_complexity_hint": "O(k)",
        "solution_template": """def solution(nums: list[int], k: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([3, 2, 1, 5, 6, 4], 2))  # 5
    print(solution([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 4
""",
        "reference_solution": """import heapq


def solution(nums: list[int], k: int) -> int:
    # 크기 k인 최소 힙 유지
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap[0]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 2, 1, 5, 6, 4]\n2", "expected_output": "5", "is_sample": True},
            {"input": "[3, 2, 3, 1, 2, 4, 5, 5, 6]\n4", "expected_output": "4", "is_sample": True},
            {"input": "[1]\n1", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "부분집합 생성",
        "description": """중복 없는 정수 배열 `nums`가 주어집니다.

가능한 모든 부분집합(멱집합)을 반환하세요.

### 예시
```
입력: nums = [1, 2, 3]
출력: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

### 참고
- 부분집합에는 빈 집합도 포함됩니다.
- 순서는 상관없습니다.""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.BACKTRACKING,
        "constraints": "1 <= nums.length <= 10, 모든 원소는 유일합니다.",
        "hints": [
            "각 원소를 포함하거나 포함하지 않는 두 가지 선택이 있습니다.",
            "재귀적으로 모든 조합을 생성하세요.",
            "반복적인 방법도 가능합니다.",
        ],
        "pattern_ids": ["subsets", "backtracking"],
        "pattern_explanation": "부분집합 패턴은 각 원소에 대해 '포함/비포함' 결정을 내리는 방식입니다. 총 2^n개의 부분집합이 생성됩니다.",
        "approach_hint": "기존 부분집합들에 현재 원소를 추가한 새 부분집합들을 만드세요.",
        "time_complexity_hint": "O(2^n)",
        "space_complexity_hint": "O(2^n)",
        "solution_template": """def solution(nums: list[int]) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([1, 2, 3]))  # [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
""",
        "reference_solution": """def solution(nums: list[int]) -> list[list[int]]:
    result = [[]]

    for num in nums:
        result += [curr + [num] for curr in result]

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1, 2, 3]", "expected_output": "[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]", "is_sample": True},
            {"input": "[0]", "expected_output": "[[], [0]]", "is_sample": True},
        ],
    },
    {
        "title": "문자열 뒤집기",
        "description": """문자 배열 `s`가 주어집니다.

배열을 제자리에서 뒤집으세요.

**추가 공간 O(1)만 사용해야 합니다.**

### 예시
```
입력: s = ["h","e","l","l","o"]
출력: ["o","l","l","e","h"]
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.STRING,
        "constraints": "1 <= s.length <= 10^5",
        "hints": [
            "투 포인터를 사용하세요.",
            "양쪽 끝에서 시작하여 가운데로 이동합니다.",
            "두 문자를 스왑하세요.",
        ],
        "pattern_ids": ["two-pointers"],
        "pattern_explanation": "투 포인터 패턴의 기본 예시입니다. 양쪽 끝에서 시작해 중앙으로 이동하며 스왑합니다.",
        "approach_hint": "left와 right 포인터가 만날 때까지 스왑을 반복하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(s: list[str]) -> None:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    arr = ["h", "e", "l", "l", "o"]
    solution(arr)
    print(arr)  # ["o", "l", "l", "e", "h"]
""",
        "reference_solution": """def solution(s: list[str]) -> None:
    left, right = 0, len(s) - 1

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": '["h", "e", "l", "l", "o"]', "expected_output": '["o", "l", "l", "e", "h"]', "is_sample": True},
            {"input": '["H", "a", "n", "n", "a", "h"]', "expected_output": '["h", "a", "n", "n", "a", "H"]', "is_sample": True},
        ],
    },
    {
        "title": "다음 큰 원소",
        "description": """정수 배열 `nums`가 주어집니다.

각 원소에 대해, 오른쪽에 있는 첫 번째 '더 큰' 원소를 찾으세요.
없으면 -1을 반환합니다.

### 예시
```
입력: nums = [4, 5, 2, 25]
출력: [5, 25, 25, -1]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.STACK,
        "constraints": "1 <= nums.length <= 10^4",
        "hints": [
            "스택을 사용하세요.",
            "아직 답을 찾지 못한 인덱스를 스택에 저장하세요.",
            "현재 원소가 스택 top보다 크면 pop하며 답을 기록하세요.",
        ],
        "pattern_ids": ["monotonic-stack"],
        "pattern_explanation": "단조 스택 패턴을 사용합니다. 감소하는 순서를 유지하며, 더 큰 원소가 나오면 스택에서 pop합니다.",
        "approach_hint": "스택에 인덱스를 저장하고, 현재 원소가 더 크면 스택을 pop하며 결과를 채우세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """def solution(nums: list[int]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([4, 5, 2, 25]))  # [5, 25, 25, -1]
""",
        "reference_solution": """def solution(nums: list[int]) -> list[int]:
    result = [-1] * len(nums)
    stack = []

    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[4, 5, 2, 25]", "expected_output": "[5, 25, 25, -1]", "is_sample": True},
            {"input": "[1, 2, 3, 4]", "expected_output": "[2, 3, 4, -1]", "is_sample": True},
            {"input": "[4, 3, 2, 1]", "expected_output": "[-1, -1, -1, -1]", "is_sample": False},
        ],
    },
    # === NEW PROBLEMS TO COVER ALL PATTERNS ===
    {
        "title": "누락된 숫자 찾기",
        "description": """[0, n] 범위의 n개의 서로 다른 숫자를 포함하는 배열 `nums`가 주어집니다.

이 범위에서 배열에 없는 유일한 숫자를 반환하세요.

### 예시
```
입력: nums = [3, 0, 1]
출력: 2
설명: 범위 [0, 3]에서 2가 누락됨

입력: nums = [9,6,4,2,3,5,7,0,1]
출력: 8
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "n == nums.length, 0 <= nums[i] <= n, 모든 숫자는 유일합니다",
        "hints": [
            "수학적 공식 n*(n+1)/2를 사용하세요.",
            "순환 정렬을 사용할 수도 있습니다.",
            "XOR 연산을 사용하는 방법도 있습니다.",
        ],
        "pattern_ids": ["cyclic-sort", "bitwise-xor"],
        "pattern_explanation": "순환 정렬로 각 숫자를 제자리에 배치한 후 누락된 위치를 찾거나, XOR의 자기 상쇄 특성을 활용합니다.",
        "approach_hint": "0부터 n까지의 모든 인덱스와 값을 XOR하면 누락된 숫자만 남습니다.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([3, 0, 1]))  # 2
    print(solution([9,6,4,2,3,5,7,0,1]))  # 8
""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 0, 1]", "expected_output": "2", "is_sample": True},
            {"input": "[9,6,4,2,3,5,7,0,1]", "expected_output": "8", "is_sample": True},
            {"input": "[0]", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "연결 리스트 뒤집기",
        "description": """단일 연결 리스트의 head가 주어집니다.

리스트를 뒤집고 뒤집힌 리스트의 head를 반환하세요.

### 예시
```
입력: head = [1, 2, 3, 4, 5]
출력: [5, 4, 3, 2, 1]
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.LINKED_LIST,
        "constraints": "0 <= 노드 수 <= 5000",
        "hints": [
            "세 개의 포인터를 사용하세요: prev, curr, next.",
            "각 노드의 next 포인터를 이전 노드를 가리키도록 바꾸세요.",
            "반복적 또는 재귀적으로 풀 수 있습니다.",
        ],
        "pattern_ids": ["in-place-reversal"],
        "pattern_explanation": "제자리 뒤집기 패턴입니다. 추가 공간 없이 포인터만 조작하여 리스트를 뒤집습니다.",
        "approach_hint": "현재 노드의 next를 이전 노드로 바꾸고, 포인터들을 한 칸씩 이동하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(head: ListNode) -> ListNode:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(head: ListNode) -> ListNode:
    prev = None
    curr = head

    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    return prev
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]", "expected_output": "[5, 4, 3, 2, 1]", "is_sample": True},
            {"input": "[1, 2]", "expected_output": "[2, 1]", "is_sample": True},
            {"input": "[]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "이진 트리 최대 깊이",
        "description": """이진 트리의 root가 주어질 때, 트리의 최대 깊이를 반환하세요.

최대 깊이는 루트 노드에서 가장 먼 리프 노드까지의 최장 경로에 있는 노드 수입니다.

### 예시
```
    3
   / \\
  9  20
    /  \\
   15   7

출력: 3
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.TREE,
        "constraints": "0 <= 노드 수 <= 10^4",
        "hints": [
            "DFS로 재귀적으로 풀 수 있습니다.",
            "BFS로 레벨별로 탐색할 수도 있습니다.",
            "각 노드에서 왼쪽/오른쪽 서브트리의 최대 깊이 + 1",
        ],
        "pattern_ids": ["tree-dfs"],
        "pattern_explanation": "트리 DFS 패턴입니다. 재귀적으로 각 서브트리의 깊이를 구하고 최대값 + 1을 반환합니다.",
        "approach_hint": "max(left_depth, right_depth) + 1을 반환하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h) where h is tree height",
        "solution_template": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> int:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> int:
    if not root:
        return 0
    return max(solution(root.left), solution(root.right)) + 1
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 9, 20, null, null, 15, 7]", "expected_output": "3", "is_sample": True},
            {"input": "[1, null, 2]", "expected_output": "2", "is_sample": True},
            {"input": "[]", "expected_output": "0", "is_sample": False},
        ],
    },
    {
        "title": "중앙값 찾기 (데이터 스트림)",
        "description": """정수 스트림에서 중앙값을 효율적으로 찾는 클래스를 구현하세요.

- `addNum(num)`: 스트림에 정수 추가
- `findMedian()`: 현재까지의 중앙값 반환

### 예시
```
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3)
findMedian() -> 2.0
```""",
        "difficulty": Difficulty.HARD,
        "category": Category.HEAP,
        "constraints": "-10^5 <= num <= 10^5",
        "hints": [
            "두 개의 힙을 사용하세요.",
            "최대 힙(작은 절반)과 최소 힙(큰 절반)으로 나누세요.",
            "두 힙의 크기를 균형있게 유지하세요.",
        ],
        "pattern_ids": ["two-heaps"],
        "pattern_explanation": "두 힙 패턴을 사용합니다. 작은 절반은 최대 힙, 큰 절반은 최소 힙으로 관리하여 중앙값을 O(1)에 구합니다.",
        "approach_hint": "작은 쪽 최대 힙의 top과 큰 쪽 최소 힙의 top을 이용해 중앙값을 계산하세요.",
        "time_complexity_hint": "addNum: O(log n), findMedian: O(1)",
        "space_complexity_hint": "O(n)",
        "solution_template": """import heapq


class MedianFinder:
    def __init__(self):
        # 여기에 코드를 작성하세요
        pass

    def addNum(self, num: int) -> None:
        pass

    def findMedian(self) -> float:
        pass


# 테스트
if __name__ == "__main__":
    mf = MedianFinder()
    mf.addNum(1)
    mf.addNum(2)
    print(mf.findMedian())  # 1.5
    mf.addNum(3)
    print(mf.findMedian())  # 2.0
""",
        "reference_solution": """import heapq


class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated)
        self.large = []  # min heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))

        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[[1,2],[],[3],[]]", "expected_output": "[null,null,1.5,null,2.0]", "is_sample": True},
        ],
    },
    {
        "title": "K개 정렬 리스트 병합",
        "description": """k개의 정렬된 연결 리스트 배열이 주어집니다.

모든 연결 리스트를 하나의 정렬된 연결 리스트로 병합하세요.

### 예시
```
입력: lists = [[1,4,5],[1,3,4],[2,6]]
출력: [1,1,2,3,4,4,5,6]
```""",
        "difficulty": Difficulty.HARD,
        "category": Category.LINKED_LIST,
        "constraints": "0 <= k <= 10^4, 0 <= lists[i].length <= 500",
        "hints": [
            "최소 힙을 사용하세요.",
            "각 리스트의 첫 번째 노드를 힙에 넣으세요.",
            "힙에서 최솟값을 꺼내고 다음 노드를 넣으세요.",
        ],
        "pattern_ids": ["k-way-merge"],
        "pattern_explanation": "K-way 병합 패턴입니다. 최소 힙을 사용해 K개의 정렬된 입력을 효율적으로 병합합니다.",
        "approach_hint": "힙에 (값, 리스트 인덱스, 노드)를 저장하고, 항상 최솟값을 꺼내 결과에 추가하세요.",
        "time_complexity_hint": "O(N log k)",
        "space_complexity_hint": "O(k)",
        "solution_template": """import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(lists: list[ListNode]) -> ListNode:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def solution(lists: list[ListNode]) -> ListNode:
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    curr = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[[1,4,5],[1,3,4],[2,6]]", "expected_output": "[1,1,2,3,4,4,5,6]", "is_sample": True},
            {"input": "[]", "expected_output": "[]", "is_sample": True},
            {"input": "[[]]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "강의 순서 (위상 정렬)",
        "description": """총 `numCourses`개의 강의가 있습니다.

선수과목 배열 `prerequisites[i] = [a, b]`는 강의 a를 들으려면 먼저 강의 b를 들어야 함을 의미합니다.

모든 강의를 들을 수 있는 순서를 반환하세요. 불가능하면 빈 배열을 반환하세요.

### 예시
```
입력: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
출력: [0,1,2,3] 또는 [0,2,1,3]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.GRAPH,
        "constraints": "1 <= numCourses <= 2000",
        "hints": [
            "위상 정렬을 사용하세요.",
            "진입 차수(in-degree)가 0인 노드부터 시작하세요.",
            "사이클이 있으면 불가능합니다.",
        ],
        "pattern_ids": ["topological-sort"],
        "pattern_explanation": "위상 정렬 패턴입니다. DAG(비순환 방향 그래프)에서 선행 관계를 고려한 순서를 찾습니다.",
        "approach_hint": "진입 차수가 0인 노드를 큐에 넣고, BFS로 처리하세요.",
        "time_complexity_hint": "O(V + E)",
        "space_complexity_hint": "O(V + E)",
        "solution_template": """from collections import deque, defaultdict


def solution(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(4, [[1,0],[2,0],[3,1],[3,2]]))  # [0,1,2,3] 또는 [0,2,1,3]
""",
        "reference_solution": """from collections import deque, defaultdict


def solution(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []

    while queue:
        course = queue.popleft()
        result.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return result if len(result) == numCourses else []
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "4\n[[1,0],[2,0],[3,1],[3,2]]", "expected_output": "[0, 1, 2, 3]", "is_sample": True},
            {"input": "2\n[[1,0]]", "expected_output": "[0, 1]", "is_sample": True},
            {"input": "2\n[[1,0],[0,1]]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "0/1 배낭 문제",
        "description": """n개의 아이템과 용량 W인 배낭이 있습니다.

각 아이템은 무게 `weights[i]`와 가치 `values[i]`를 가집니다.

배낭 용량을 초과하지 않으면서 담을 수 있는 최대 가치를 구하세요.

**각 아이템은 한 번만 선택할 수 있습니다.**

### 예시
```
입력: weights = [1, 2, 3], values = [6, 10, 12], W = 5
출력: 22
설명: 무게 2와 3인 아이템을 선택 (가치 10 + 12 = 22)
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= n <= 100, 1 <= W <= 1000",
        "hints": [
            "2D DP 테이블을 사용하세요.",
            "dp[i][w] = i번째 아이템까지 고려했을 때 용량 w로 얻는 최대 가치",
            "각 아이템을 선택하거나 선택하지 않는 두 경우를 비교하세요.",
        ],
        "pattern_ids": ["0-1-knapsack"],
        "pattern_explanation": "0/1 배낭 패턴입니다. 각 아이템을 선택/비선택하는 결정을 DP로 최적화합니다.",
        "approach_hint": "dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])",
        "time_complexity_hint": "O(n × W)",
        "space_complexity_hint": "O(n × W) 또는 O(W)",
        "solution_template": """def solution(weights: list[int], values: list[int], W: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([1, 2, 3], [6, 10, 12], 5))  # 22
""",
        "reference_solution": """def solution(weights: list[int], values: list[int], W: int) -> int:
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][W]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1, 2, 3]\n[6, 10, 12]\n5", "expected_output": "22", "is_sample": True},
            {"input": "[1, 3, 4, 5]\n[1, 4, 5, 7]\n7", "expected_output": "9", "is_sample": True},
        ],
    },
    {
        "title": "피보나치 수",
        "description": """n번째 피보나치 수를 구하세요.

피보나치 수열: F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)

### 예시
```
입력: n = 4
출력: 3
설명: F(4) = F(3) + F(2) = 2 + 1 = 3

입력: n = 10
출력: 55
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "0 <= n <= 45",
        "hints": [
            "재귀로 풀면 중복 계산이 많습니다.",
            "메모이제이션 또는 DP를 사용하세요.",
            "공간 최적화: 이전 두 값만 저장하면 됩니다.",
        ],
        "pattern_ids": ["fibonacci-numbers", "dp"],
        "pattern_explanation": "피보나치 패턴은 DP의 기본입니다. 이전 두 상태만으로 현재 상태를 계산합니다.",
        "approach_hint": "두 변수 a, b를 사용해 반복적으로 계산하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(4))   # 3
    print(solution(10))  # 55
""",
        "reference_solution": """def solution(n: int) -> int:
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "4", "expected_output": "3", "is_sample": True},
            {"input": "10", "expected_output": "55", "is_sample": True},
            {"input": "0", "expected_output": "0", "is_sample": False},
        ],
    },
    {
        "title": "가장 긴 팰린드롬 부분수열",
        "description": """문자열 `s`가 주어질 때, 가장 긴 팰린드롬 **부분수열**의 길이를 반환하세요.

부분수열은 일부 문자를 삭제하여 얻을 수 있으며, 순서는 유지됩니다.

### 예시
```
입력: s = "bbbab"
출력: 4
설명: 가장 긴 팰린드롬 부분수열은 "bbbb"

입력: s = "cbbd"
출력: 2
설명: 가장 긴 팰린드롬 부분수열은 "bb"
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= s.length <= 1000",
        "hints": [
            "dp[i][j]는 s[i:j+1]의 최장 팰린드롬 부분수열 길이입니다.",
            "s[i] == s[j]이면 dp[i][j] = dp[i+1][j-1] + 2",
            "그렇지 않으면 dp[i][j] = max(dp[i+1][j], dp[i][j-1])",
        ],
        "pattern_ids": ["palindromic-subsequence", "dp"],
        "pattern_explanation": "팰린드롬 부분수열 패턴입니다. 양 끝에서 시작해 중앙으로 접근하며 최적해를 구합니다.",
        "approach_hint": "길이가 짧은 부분 문자열부터 시작해 DP 테이블을 채우세요.",
        "time_complexity_hint": "O(n²)",
        "space_complexity_hint": "O(n²)",
        "solution_template": """def solution(s: str) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution("bbbab"))  # 4
    print(solution("cbbd"))   # 2
""",
        "reference_solution": """def solution(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "bbbab", "expected_output": "4", "is_sample": True},
            {"input": "cbbd", "expected_output": "2", "is_sample": True},
            {"input": "a", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "최장 공통 부분수열 (LCS)",
        "description": """두 문자열 `text1`과 `text2`가 주어질 때, 가장 긴 공통 부분수열의 길이를 반환하세요.

공통 부분수열이 없으면 0을 반환하세요.

### 예시
```
입력: text1 = "abcde", text2 = "ace"
출력: 3
설명: 가장 긴 공통 부분수열은 "ace"

입력: text1 = "abc", text2 = "def"
출력: 0
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= text1.length, text2.length <= 1000",
        "hints": [
            "dp[i][j]는 text1[:i]와 text2[:j]의 LCS 길이입니다.",
            "마지막 문자가 같으면 dp[i][j] = dp[i-1][j-1] + 1",
            "다르면 dp[i][j] = max(dp[i-1][j], dp[i][j-1])",
        ],
        "pattern_ids": ["longest-common-subsequence", "dp"],
        "pattern_explanation": "LCS 패턴은 두 시퀀스 비교 문제의 기본입니다. 2D DP로 최적해를 구합니다.",
        "approach_hint": "각 위치에서 두 문자가 같으면 대각선 + 1, 다르면 위/왼쪽 중 최대값을 선택하세요.",
        "time_complexity_hint": "O(m × n)",
        "space_complexity_hint": "O(m × n)",
        "solution_template": """def solution(text1: str, text2: str) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution("abcde", "ace"))  # 3
    print(solution("abc", "def"))    # 0
""",
        "reference_solution": """def solution(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "abcde\nace", "expected_output": "3", "is_sample": True},
            {"input": "abc\ndef", "expected_output": "0", "is_sample": True},
            {"input": "abc\nabc", "expected_output": "3", "is_sample": False},
        ],
    },
    {
        "title": "단일 숫자 찾기",
        "description": """비어있지 않은 정수 배열이 주어집니다.

모든 요소가 두 번씩 나타나고, **정확히 하나만 한 번** 나타납니다.

그 단일 요소를 찾으세요.

### 예시
```
입력: nums = [2, 2, 1]
출력: 1

입력: nums = [4, 1, 2, 1, 2]
출력: 4
```

**O(1) 추가 공간으로 풀어보세요.**""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "1 <= nums.length <= 3 × 10^4",
        "hints": [
            "XOR 연산의 특성을 활용하세요.",
            "a XOR a = 0, a XOR 0 = a",
            "모든 숫자를 XOR하면?",
        ],
        "pattern_ids": ["bitwise-xor"],
        "pattern_explanation": "비트 XOR 패턴입니다. XOR의 자기 상쇄 특성(a^a=0)을 이용해 유일한 숫자를 찾습니다.",
        "approach_hint": "모든 숫자를 XOR하면 짝수 개인 숫자들은 상쇄되고 단일 숫자만 남습니다.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2, 2, 1]))        # 1
    print(solution([4, 1, 2, 1, 2]))  # 4
""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2, 2, 1]", "expected_output": "1", "is_sample": True},
            {"input": "[4, 1, 2, 1, 2]", "expected_output": "4", "is_sample": True},
            {"input": "[1]", "expected_output": "1", "is_sample": False},
        ],
    },
    {
        "title": "구간 합 쿼리",
        "description": """정수 배열 `nums`와 여러 쿼리가 주어집니다.

각 쿼리 `[left, right]`에 대해 nums[left]부터 nums[right]까지의 합을 구하세요.

### 예시
```
입력: nums = [-2, 0, 3, -5, 2, -1]
쿼리: sumRange(0, 2) -> 1  (-2 + 0 + 3)
쿼리: sumRange(2, 5) -> -1 (3 + -5 + 2 + -1)
쿼리: sumRange(0, 5) -> -3
```

**여러 쿼리에 효율적으로 응답하세요.**""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "1 <= nums.length <= 10^4, 최대 10^4개의 쿼리",
        "hints": [
            "매번 합을 계산하면 O(n×q)입니다.",
            "누적 합(prefix sum)을 미리 계산하세요.",
            "sum(i, j) = prefix[j+1] - prefix[i]",
        ],
        "pattern_ids": ["prefix-sum"],
        "pattern_explanation": "누적 합 패턴입니다. O(n)에 전처리하여 각 구간 합 쿼리를 O(1)에 응답합니다.",
        "approach_hint": "prefix[i] = nums[0] + nums[1] + ... + nums[i-1]로 정의하세요.",
        "time_complexity_hint": "전처리: O(n), 쿼리: O(1)",
        "space_complexity_hint": "O(n)",
        "solution_template": """class NumArray:
    def __init__(self, nums: list[int]):
        # 여기에 코드를 작성하세요
        pass

    def sumRange(self, left: int, right: int) -> int:
        pass


# 테스트
if __name__ == "__main__":
    arr = NumArray([-2, 0, 3, -5, 2, -1])
    print(arr.sumRange(0, 2))  # 1
    print(arr.sumRange(2, 5))  # -1
    print(arr.sumRange(0, 5))  # -3
""",
        "reference_solution": """class NumArray:
    def __init__(self, nums: list[int]):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix[right + 1] - self.prefix[left]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[-2, 0, 3, -5, 2, -1]\n0\n2", "expected_output": "1", "is_sample": True},
            {"input": "[-2, 0, 3, -5, 2, -1]\n2\n5", "expected_output": "-1", "is_sample": True},
            {"input": "[-2, 0, 3, -5, 2, -1]\n0\n5", "expected_output": "-3", "is_sample": False},
        ],
    },
    {
        "title": "최단 경로 (다익스트라)",
        "description": """가중치 그래프와 시작 노드가 주어집니다.

시작 노드에서 다른 모든 노드까지의 최단 거리를 구하세요.

### 예시
```
입력: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]], start = 0
출력: [0, 1, 3, 4]
설명: 0→0: 0, 0→1: 1, 0→1→2: 3, 0→1→2→3: 4
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.GRAPH,
        "constraints": "1 <= n <= 1000",
        "hints": [
            "다익스트라 알고리즘을 사용하세요.",
            "우선순위 큐(최소 힙)를 사용하세요.",
            "방문한 노드는 다시 처리하지 마세요.",
        ],
        "pattern_ids": ["graph-traversal"],
        "pattern_explanation": "그래프 탐색 + 최단 경로 패턴입니다. 다익스트라 알고리즘은 음이 아닌 가중치 그래프에서 최단 경로를 찾습니다.",
        "approach_hint": "거리가 가장 짧은 노드부터 처리하고, 인접 노드의 거리를 갱신하세요.",
        "time_complexity_hint": "O((V + E) log V)",
        "space_complexity_hint": "O(V + E)",
        "solution_template": """import heapq
from collections import defaultdict


def solution(n: int, edges: list[list[int]], start: int) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(4, [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]], 0))  # [0, 1, 3, 4]
""",
        "reference_solution": """import heapq
from collections import defaultdict


def solution(n: int, edges: list[list[int]], start: int) -> list[int]:
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return [d if d != float('inf') else -1 for d in dist]
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "4\n[[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]]\n0", "expected_output": "[0, 1, 3, 4]", "is_sample": True},
        ],
    },
    {
        "title": "계단 오르기",
        "description": """n개의 계단을 오르려고 합니다.

한 번에 1개 또는 2개의 계단을 오를 수 있습니다.

정상에 도달하는 방법의 수를 구하세요.

### 예시
```
입력: n = 2
출력: 2
설명: 1+1 또는 2

입력: n = 3
출력: 3
설명: 1+1+1, 1+2, 2+1
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= n <= 45",
        "hints": [
            "n번째 계단에 도달하는 방법은?",
            "n-1번째에서 1계단 오르거나, n-2번째에서 2계단 오르거나",
            "피보나치 수열과 비슷합니다.",
        ],
        "pattern_ids": ["fibonacci-numbers", "dp"],
        "pattern_explanation": "피보나치 패턴의 응용입니다. dp[n] = dp[n-1] + dp[n-2]",
        "approach_hint": "f(n) = f(n-1) + f(n-2)로 재귀 관계를 세우세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(2))  # 2
    print(solution(3))  # 3
    print(solution(5))  # 8
""",
        "reference_solution": """def solution(n: int) -> int:
    if n <= 2:
        return n

    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b

    return b
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "2", "expected_output": "2", "is_sample": True},
            {"input": "3", "expected_output": "3", "is_sample": True},
            {"input": "5", "expected_output": "8", "is_sample": False},
        ],
    },
    {
        "title": "정렬된 배열 합치기",
        "description": """두 정렬된 정수 배열 `nums1`과 `nums2`가 주어집니다.

두 배열을 하나의 정렬된 배열로 합치세요.

`nums1`에는 두 배열을 합칠 충분한 공간이 있습니다.

### 예시
```
입력: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
출력: [1,2,2,3,5,6]
```

**제자리(in-place)로 해결하세요.**""",
        "difficulty": Difficulty.EASY,
        "category": Category.ARRAY,
        "constraints": "nums1.length == m + n",
        "hints": [
            "뒤에서부터 채우세요.",
            "세 개의 포인터를 사용하세요.",
            "두 배열의 끝에서 시작해 더 큰 값을 뒤에 배치하세요.",
        ],
        "pattern_ids": ["two-pointers"],
        "pattern_explanation": "투 포인터 패턴의 변형입니다. 뒤에서부터 채워 기존 데이터를 덮어쓰지 않습니다.",
        "approach_hint": "i = m-1, j = n-1, k = m+n-1부터 시작해 더 큰 값을 nums1[k]에 배치하세요.",
        "time_complexity_hint": "O(m + n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    nums1 = [1, 2, 3, 0, 0, 0]
    solution(nums1, 3, [2, 5, 6], 3)
    print(nums1)  # [1, 2, 2, 3, 5, 6]
""",
        "reference_solution": """def solution(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    i, j, k = m - 1, n - 1, m + n - 1

    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1,2,3,0,0,0]\n3\n[2,5,6]\n3", "expected_output": "[1, 2, 2, 3, 5, 6]", "is_sample": True},
            {"input": "[1]\n1\n[]\n0", "expected_output": "[1]", "is_sample": True},
        ],
    },
    {
        "title": "부분배열의 최대 곱",
        "description": """정수 배열 `nums`가 주어질 때, 곱이 최대인 연속 부분배열을 찾아 그 곱을 반환하세요.

### 예시
```
입력: nums = [2, 3, -2, 4]
출력: 6
설명: [2, 3]의 곱 = 6

입력: nums = [-2, 0, -1]
출력: 0
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= nums.length <= 2 × 10^4",
        "hints": [
            "음수 × 음수 = 양수가 될 수 있습니다.",
            "최대값과 최소값을 동시에 추적하세요.",
            "현재 원소가 음수면 최대/최소가 뒤바뀝니다.",
        ],
        "pattern_ids": ["dp"],
        "pattern_explanation": "DP 패턴의 변형입니다. 음수 처리를 위해 최대값과 최소값을 모두 추적합니다.",
        "approach_hint": "max_ending과 min_ending을 동시에 유지하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2, 3, -2, 4]))  # 6
    print(solution([-2, 0, -1]))    # 0
""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    max_prod = min_prod = result = nums[0]

    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod

        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2, 3, -2, 4]", "expected_output": "6", "is_sample": True},
            {"input": "[-2, 0, -1]", "expected_output": "0", "is_sample": True},
            {"input": "[-2, 3, -4]", "expected_output": "24", "is_sample": False},
        ],
    },
    {
        "title": "회전된 배열에서 검색",
        "description": """오름차순 정렬된 배열이 한 지점에서 회전되었습니다.

예: [0,1,2,4,5,6,7]이 [4,5,6,7,0,1,2]로 회전

이 배열에서 `target`을 찾아 인덱스를 반환하세요. 없으면 -1을 반환하세요.

### 예시
```
입력: nums = [4,5,6,7,0,1,2], target = 0
출력: 4

입력: nums = [4,5,6,7,0,1,2], target = 3
출력: -1
```

**O(log n) 시간 복잡도로 해결하세요.**""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.BINARY_SEARCH,
        "constraints": "1 <= nums.length <= 5000, 모든 값은 유일합니다",
        "hints": [
            "이진 탐색을 응용하세요.",
            "mid를 기준으로 한쪽은 정렬되어 있습니다.",
            "정렬된 부분에 target이 있는지 확인하세요.",
        ],
        "pattern_ids": ["binary-search"],
        "pattern_explanation": "이진 탐색 패턴의 변형입니다. 회전된 배열에서도 절반씩 탐색 범위를 줄일 수 있습니다.",
        "approach_hint": "mid 기준 왼쪽 또는 오른쪽 중 정렬된 부분을 찾고, target이 그 범위에 있는지 확인하세요.",
        "time_complexity_hint": "O(log n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([4,5,6,7,0,1,2], 0))  # 4
    print(solution([4,5,6,7,0,1,2], 3))  # -1
""",
        "reference_solution": """def solution(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[4,5,6,7,0,1,2]\n0", "expected_output": "4", "is_sample": True},
            {"input": "[4,5,6,7,0,1,2]\n3", "expected_output": "-1", "is_sample": True},
            {"input": "[1]\n0", "expected_output": "-1", "is_sample": False},
        ],
    },
    {
        "title": "이진 트리의 모든 경로",
        "description": """이진 트리의 root가 주어질 때, 루트에서 리프까지의 모든 경로를 반환하세요.

### 예시
```
    1
   / \\
  2   3
   \\
    5

출력: ["1->2->5", "1->3"]
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.TREE,
        "constraints": "1 <= 노드 수 <= 100",
        "hints": [
            "DFS를 사용하세요.",
            "현재 경로를 추적하며 리프 노드에서 결과에 추가하세요.",
            "백트래킹 또는 문자열 전달 방식으로 구현하세요.",
        ],
        "pattern_ids": ["tree-dfs", "backtracking"],
        "pattern_explanation": "트리 DFS + 경로 추적 패턴입니다. 모든 루트-리프 경로를 탐색합니다.",
        "approach_hint": "재귀 호출 시 현재 경로를 전달하고, 리프에서 결과에 추가하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[str]:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[str]:
    if not root:
        return []

    result = []

    def dfs(node, path):
        if not node.left and not node.right:
            result.append(path)
            return

        if node.left:
            dfs(node.left, path + "->" + str(node.left.val))
        if node.right:
            dfs(node.right, path + "->" + str(node.right.val))

    dfs(root, str(root.val))
    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1, 2, 3, null, 5]", "expected_output": "[\"1->2->5\", \"1->3\"]", "is_sample": True},
            {"input": "[1]", "expected_output": "[\"1\"]", "is_sample": True},
        ],
    },
    {
        "title": "가장 긴 무중복 부분 문자열",
        "description": """문자열 `s`가 주어질 때, 중복 문자가 없는 가장 긴 부분 문자열의 길이를 반환하세요.

### 예시
```
입력: s = "abcabcbb"
출력: 3
설명: "abc"가 가장 긴 무중복 부분 문자열

입력: s = "bbbbb"
출력: 1

입력: s = "pwwkew"
출력: 3
설명: "wke"가 정답. "pwke"는 부분 문자열이 아닌 부분수열임.
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.STRING,
        "constraints": "0 <= s.length <= 5 × 10^4",
        "hints": [
            "슬라이딩 윈도우를 사용하세요.",
            "집합 또는 해시맵으로 현재 윈도우의 문자를 추적하세요.",
            "중복이 발견되면 왼쪽 포인터를 이동하세요.",
        ],
        "pattern_ids": ["sliding-window"],
        "pattern_explanation": "가변 크기 슬라이딩 윈도우 패턴입니다. 조건(중복 없음)을 만족하는 최대 윈도우를 찾습니다.",
        "approach_hint": "오른쪽으로 확장하다 중복이 나오면 왼쪽에서 제거하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(min(n, 문자 집합 크기))",
        "solution_template": """def solution(s: str) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution("abcabcbb"))  # 3
    print(solution("bbbbb"))     # 1
    print(solution("pwwkew"))    # 3
""",
        "reference_solution": """def solution(s: str) -> int:
    char_index = {}
    max_length = 0
    left = 0

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "abcabcbb", "expected_output": "3", "is_sample": True},
            {"input": "bbbbb", "expected_output": "1", "is_sample": True},
            {"input": "pwwkew", "expected_output": "3", "is_sample": False},
        ],
    },
    {
        "title": "구간 병합",
        "description": """구간 배열이 주어집니다. 겹치는 모든 구간을 병합하세요.

### 예시
```
입력: intervals = [[1,3],[2,6],[8,10],[15,18]]
출력: [[1,6],[8,10],[15,18]]
설명: [1,3]과 [2,6]이 겹쳐서 [1,6]으로 병합

입력: intervals = [[1,4],[4,5]]
출력: [[1,5]]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.ARRAY,
        "constraints": "1 <= intervals.length <= 10^4",
        "hints": [
            "시작점 기준으로 정렬하세요.",
            "이전 구간과 현재 구간이 겹치면 병합하세요.",
            "겹치지 않으면 새 구간을 결과에 추가하세요.",
        ],
        "pattern_ids": ["merge-intervals"],
        "pattern_explanation": "구간 병합 패턴입니다. 정렬 후 순차적으로 겹치는 구간을 합칩니다.",
        "approach_hint": "current[1] >= intervals[i][0]이면 병합, 아니면 새로 추가하세요.",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """def solution(intervals: list[list[int]]) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([[1,3],[2,6],[8,10],[15,18]]))  # [[1,6],[8,10],[15,18]]
    print(solution([[1,4],[4,5]]))                  # [[1,5]]
""",
        "reference_solution": """def solution(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[[1,3],[2,6],[8,10],[15,18]]", "expected_output": "[[1, 6], [8, 10], [15, 18]]", "is_sample": True},
            {"input": "[[1,4],[4,5]]", "expected_output": "[[1, 5]]", "is_sample": True},
        ],
    },
    {
        "title": "조합 합",
        "description": """서로 다른 정수 배열 `candidates`와 목표값 `target`이 주어집니다.

합이 `target`이 되는 모든 유일한 조합을 반환하세요.

**같은 숫자를 여러 번 사용할 수 있습니다.**

### 예시
```
입력: candidates = [2,3,6,7], target = 7
출력: [[2,2,3],[7]]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.BACKTRACKING,
        "constraints": "1 <= candidates.length <= 30, 2 <= target <= 40",
        "hints": [
            "백트래킹을 사용하세요.",
            "각 숫자를 0번, 1번, 2번... 사용하는 경우를 탐색하세요.",
            "합이 target을 초과하면 가지치기하세요.",
        ],
        "pattern_ids": ["backtracking", "unbounded-knapsack"],
        "pattern_explanation": "백트래킹 + 무한 배낭 패턴입니다. 같은 요소를 여러 번 사용할 수 있는 조합 탐색입니다.",
        "approach_hint": "현재 숫자를 계속 사용하거나 다음 숫자로 넘어가세요.",
        "time_complexity_hint": "O(N^(T/M))",
        "space_complexity_hint": "O(T/M)",
        "solution_template": """def solution(candidates: list[int], target: int) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2,3,6,7], 7))  # [[2,2,3],[7]]
""",
        "reference_solution": """def solution(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2,3,6,7]\n7", "expected_output": "[[2, 2, 3], [7]]", "is_sample": True},
            {"input": "[2,3,5]\n8", "expected_output": "[[2, 2, 2, 2], [2, 3, 3], [3, 5]]", "is_sample": True},
        ],
    },
    {
        "title": "점프 게임",
        "description": """비음수 정수 배열 `nums`가 주어집니다.

처음에 첫 번째 인덱스에 있습니다. 각 요소는 해당 위치에서 앞으로 점프할 수 있는 최대 거리입니다.

마지막 인덱스에 도달할 수 있는지 판단하세요.

### 예시
```
입력: nums = [2,3,1,1,4]
출력: true
설명: 0→1→4 또는 0→2→3→4

입력: nums = [3,2,1,0,4]
출력: false
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.GREEDY,
        "constraints": "1 <= nums.length <= 10^4, 0 <= nums[i] <= 10^5",
        "hints": [
            "도달할 수 있는 가장 먼 위치를 추적하세요.",
            "그리디하게 매 위치에서 최대 도달 거리를 갱신하세요.",
            "현재 위치가 도달 가능 범위를 벗어나면 실패입니다.",
        ],
        "pattern_ids": ["greedy"],
        "pattern_explanation": "그리디 패턴입니다. 각 위치에서 가능한 최대 도달 거리를 갱신합니다.",
        "approach_hint": "max_reach = max(max_reach, i + nums[i])를 유지하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int]) -> bool:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2,3,1,1,4]))  # True
    print(solution([3,2,1,0,4]))  # False
""",
        "reference_solution": """def solution(nums: list[int]) -> bool:
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= len(nums) - 1:
            return True

    return True
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[2,3,1,1,4]", "expected_output": "True", "is_sample": True},
            {"input": "[3,2,1,0,4]", "expected_output": "False", "is_sample": True},
        ],
    },
    {
        "title": "이진 트리 지그재그 레벨 순회",
        "description": """이진 트리가 주어질 때, 노드 값을 지그재그 레벨 순서로 반환하세요.

첫 레벨은 왼쪽→오른쪽, 두 번째 레벨은 오른쪽→왼쪽, 세 번째는 왼쪽→오른쪽...

### 예시
```
    3
   / \\
  9  20
    /  \\
   15   7

출력: [[3], [20, 9], [15, 7]]
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.TREE,
        "constraints": "0 <= 노드 수 <= 2000",
        "hints": [
            "BFS로 레벨 순회하세요.",
            "짝수 레벨은 왼쪽→오른쪽, 홀수 레벨은 오른쪽→왼쪽",
            "레벨별로 방향을 토글하세요.",
        ],
        "pattern_ids": ["tree-bfs", "bfs"],
        "pattern_explanation": "트리 BFS 패턴의 변형입니다. 레벨별로 순서를 교대로 바꿉니다.",
        "approach_hint": "레벨 번호에 따라 결과를 reverse하거나, deque의 양쪽 삽입을 활용하세요.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "solution_template": """from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass
""",
        "reference_solution": """from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def solution(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level = deque()
        for _ in range(len(queue)):
            node = queue.popleft()
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[3, 9, 20, null, null, 15, 7]", "expected_output": "[[3], [20, 9], [15, 7]]", "is_sample": True},
            {"input": "[1]", "expected_output": "[[1]]", "is_sample": True},
            {"input": "[]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "모든 중복 찾기",
        "description": """길이 n인 정수 배열이 주어지고, 모든 정수는 [1, n] 범위에 있습니다.

일부 요소는 두 번 나타나고 일부는 한 번 나타납니다.

두 번 나타나는 모든 정수를 찾으세요.

### 예시
```
입력: nums = [4,3,2,7,8,2,3,1]
출력: [2, 3]
```

**추가 공간 O(1)로 해결하세요.**""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.ARRAY,
        "constraints": "n == nums.length, 1 <= nums[i] <= n",
        "hints": [
            "각 숫자가 인덱스 역할을 할 수 있습니다.",
            "방문한 인덱스의 값을 음수로 표시하세요.",
            "이미 음수인 곳을 다시 방문하면 중복입니다.",
        ],
        "pattern_ids": ["cyclic-sort"],
        "pattern_explanation": "순환 정렬 패턴의 응용입니다. 값 자체를 인덱스로 사용해 제자리에서 중복을 찾습니다.",
        "approach_hint": "nums[abs(num)-1]을 음수로 만들고, 이미 음수면 중복입니다.",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "solution_template": """def solution(nums: list[int]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([4,3,2,7,8,2,3,1]))  # [2, 3]
""",
        "reference_solution": """def solution(nums: list[int]) -> list[int]:
    result = []

    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            result.append(abs(num))
        else:
            nums[idx] = -nums[idx]

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[4,3,2,7,8,2,3,1]", "expected_output": "[2, 3]", "is_sample": True},
            {"input": "[1,1,2]", "expected_output": "[1]", "is_sample": True},
            {"input": "[1]", "expected_output": "[]", "is_sample": False},
        ],
    },
    {
        "title": "K쌍의 최소 합",
        "description": """두 정수 배열 `nums1`과 `nums2` (오름차순 정렬)와 정수 `k`가 주어집니다.

(u, v) 쌍의 합 u + v가 가장 작은 k개의 쌍을 반환하세요.

### 예시
```
입력: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
출력: [[1,2],[1,4],[1,6]]
설명: 합이 작은 순서: (1,2)=3, (1,4)=5, (1,6)=7
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.HEAP,
        "constraints": "1 <= nums1.length, nums2.length <= 10^5, 1 <= k <= 10^4",
        "hints": [
            "최소 힙을 사용하세요.",
            "가장 작은 합은 (nums1[0], nums2[0])입니다.",
            "(i, j)를 pop하면 (i+1, j)와 (i, j+1)을 push하세요.",
        ],
        "pattern_ids": ["k-way-merge", "top-k-elements"],
        "pattern_explanation": "K-way 병합 패턴과 Top-K 패턴의 조합입니다. 힙을 사용해 k개의 최소 합 쌍을 찾습니다.",
        "approach_hint": "중복을 피하기 위해 방문한 (i, j)를 추적하세요.",
        "time_complexity_hint": "O(k log k)",
        "space_complexity_hint": "O(k)",
        "solution_template": """import heapq


def solution(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([1,7,11], [2,4,6], 3))  # [[1,2],[1,4],[1,6]]
""",
        "reference_solution": """import heapq


def solution(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    if not nums1 or not nums2:
        return []

    result = []
    heap = [(nums1[0] + nums2[0], 0, 0)]
    visited = {(0, 0)}

    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])

        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
            visited.add((i + 1, j))

        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))

    return result
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "[1,7,11]\n[2,4,6]\n3", "expected_output": "[[1, 2], [1, 4], [1, 6]]", "is_sample": True},
            {"input": "[1,1,2]\n[1,2,3]\n2", "expected_output": "[[1, 1], [1, 1]]", "is_sample": True},
        ],
    },
]


async def seed_problems(force: bool = False):
    """Seed the database with initial problems"""
    await init_db()

    async with get_session_context() as session:
        from sqlalchemy import select, func, delete

        if force:
            # Delete all existing problems and related data
            await session.execute(delete(TestCaseModel))
            await session.execute(delete(ProblemModel))
            await session.commit()
            print("Cleared existing problems.")

        # Check if problems already exist
        result = await session.execute(select(func.count()).select_from(ProblemModel))
        count = result.scalar()

        if count > 0 and not force:
            print(f"Database already has {count} problems. Use --force to reset.")
            return

        # Insert problems
        for problem_data in SEED_PROBLEMS:
            test_cases = problem_data["test_cases"]
            problem_id = uuid4()

            problem = ProblemModel(
                id=problem_id,
                title=problem_data["title"],
                description=problem_data["description"],
                difficulty=problem_data["difficulty"],
                category=problem_data["category"],
                constraints=problem_data["constraints"],
                hints=problem_data["hints"],
                solution_template=problem_data["solution_template"],
                reference_solution=problem_data["reference_solution"],
                time_limit_ms=problem_data["time_limit_ms"],
                memory_limit_mb=problem_data["memory_limit_mb"],
                is_published=True,
                # Pattern fields
                pattern_ids=problem_data.get("pattern_ids", []),
                pattern_explanation=problem_data.get("pattern_explanation", ""),
                approach_hint=problem_data.get("approach_hint", ""),
                time_complexity_hint=problem_data.get("time_complexity_hint", ""),
                space_complexity_hint=problem_data.get("space_complexity_hint", ""),
            )
            session.add(problem)

            # Add test cases
            for i, tc in enumerate(test_cases):
                test_case = TestCaseModel(
                    id=uuid4(),
                    problem_id=problem_id,
                    input_data=tc["input"],
                    expected_output=tc["expected_output"],
                    is_sample=tc["is_sample"],
                    order=i + 1,
                )
                session.add(test_case)

        await session.commit()
        print(f"Successfully seeded {len(SEED_PROBLEMS)} problems with pattern information.")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    asyncio.run(seed_problems(force=force))
