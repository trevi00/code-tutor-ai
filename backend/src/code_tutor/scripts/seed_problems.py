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
        "solution_template": """def solution(nums: list[int], target: int) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([2, 7, 11, 15], 9))  # [0, 1]
    print(solution([3, 2, 4], 6))        # [1, 2]
""",
        "reference_solution": """def solution(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
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


# 테스트
if __name__ == "__main__":
    print(solution("()"))       # True
    print(solution("()[]{}"))   # True
    print(solution("(]"))       # False
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
        "solution_template": """def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([-1, 0, 3, 5, 9, 12], 9))   # 4
    print(solution([-1, 0, 3, 5, 9, 12], 2))   # -1
""",
        "reference_solution": """def solution(nums: list[int], target: int) -> int:
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
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6
    print(solution([1]))                               # 1
    print(solution([5, 4, -1, 7, 8]))                  # 23
""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum
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
        "solution_template": """def solution(grid: list[list[str]]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    print(solution(grid))  # 3
""",
        "reference_solution": """def solution(grid: list[list[str]]) -> int:
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
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
    print(solution([0, 1, 0, 3, 2, 3]))             # 4
    print(solution([7, 7, 7, 7, 7, 7, 7]))          # 1
""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
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
        "solution_template": """def solution(coins: list[int], amount: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution([1, 2, 5], 11))  # 3
    print(solution([2], 3))          # -1
    print(solution([1], 0))          # 0
""",
        "reference_solution": """def solution(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
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
