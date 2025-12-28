"""Seed script for algorithm problems"""

import asyncio
from uuid import uuid4

from code_tutor.shared.infrastructure.database import get_session_context, init_db
from code_tutor.learning.infrastructure.models import ProblemModel, TestCaseModel
from code_tutor.learning.domain.value_objects import Category, Difficulty


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
        "hints": ["해시맵을 사용하면 O(n)에 해결할 수 있습니다.", "각 숫자의 보수(complement)를 저장하세요."],
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
        "id": str(uuid4()),
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
        "hints": ["스택을 사용하세요.", "여는 괄호는 push, 닫는 괄호는 pop 후 매칭 확인"],
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
        "id": str(uuid4()),
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
        "hints": ["중간값과 target을 비교하세요.", "left와 right 포인터를 사용하세요."],
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
        "id": str(uuid4()),
        "title": "피보나치 수",
        "description": """정수 `n`이 주어질 때, n번째 피보나치 수를 반환하세요.

피보나치 수열은 다음과 같이 정의됩니다:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) (n > 1)

### 예시
```
입력: n = 2
출력: 1 (F(2) = F(1) + F(0) = 1 + 0 = 1)

입력: n = 4
출력: 3 (F(4) = F(3) + F(2) = 2 + 1 = 3)
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "0 <= n <= 30",
        "hints": ["메모이제이션을 사용하면 효율적입니다.", "반복문으로도 해결할 수 있습니다."],
        "solution_template": """def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


# 테스트
if __name__ == "__main__":
    print(solution(2))   # 1
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
            {"input": "2", "expected_output": "1", "is_sample": True},
            {"input": "4", "expected_output": "3", "is_sample": True},
            {"input": "10", "expected_output": "55", "is_sample": False},
            {"input": "0", "expected_output": "0", "is_sample": False},
        ],
    },
    {
        "id": str(uuid4()),
        "title": "최대 부분배열 합",
        "description": """정수 배열 `nums`가 주어질 때, 합이 최대인 연속 부분배열을 찾아 그 합을 반환하세요.

**부분배열은 연속된 원소들로 이루어져야 합니다.**

### 예시
```
입력: nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
출력: 6
설명: [4, -1, 2, 1]의 합 = 6

입력: nums = [1]
출력: 1
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= nums.length <= 10^5, -10^4 <= nums[i] <= 10^4",
        "hints": ["Kadane's Algorithm을 사용하세요.", "현재 위치에서의 최대 합을 갱신하세요."],
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
        "id": str(uuid4()),
        "title": "연결 리스트 뒤집기",
        "description": """단일 연결 리스트의 헤드가 주어질 때, 리스트를 뒤집어 반환하세요.

### 예시
```
입력: [1, 2, 3, 4, 5]
출력: [5, 4, 3, 2, 1]
```

### 노드 정의
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.LINKED_LIST,
        "constraints": "0 <= 노드 개수 <= 5000",
        "hints": ["세 개의 포인터를 사용하세요: prev, curr, next", "재귀로도 해결할 수 있습니다."],
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
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

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
        "id": str(uuid4()),
        "title": "이진 트리 최대 깊이",
        "description": """이진 트리가 주어질 때, 최대 깊이를 반환하세요.

최대 깊이는 루트 노드에서 가장 먼 리프 노드까지의 경로에 있는 노드 수입니다.

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
        "hints": ["재귀를 사용하세요.", "BFS로도 해결할 수 있습니다."],
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

    left_depth = solution(root.left)
    right_depth = solution(root.right)

    return max(left_depth, right_depth) + 1
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
        "id": str(uuid4()),
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
        "hints": ["DFS 또는 BFS를 사용하세요.", "방문한 육지는 표시하세요."],
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
        "id": str(uuid4()),
        "title": "동전 교환",
        "description": """서로 다른 금액의 동전 배열 `coins`와 총 금액 `amount`가 주어집니다.

금액을 만들 수 있는 최소 동전 개수를 반환하세요. 만들 수 없다면 -1을 반환하세요.

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
        "hints": ["동적 프로그래밍을 사용하세요.", "dp[i]는 금액 i를 만드는 최소 동전 수입니다."],
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
        "id": str(uuid4()),
        "title": "LRU 캐시",
        "description": """최근 사용 빈도가 가장 낮은(LRU) 항목을 제거하는 캐시를 구현하세요.

### 메서드
- `LRUCache(capacity)`: 주어진 용량으로 캐시 초기화
- `get(key)`: 키가 있으면 값 반환, 없으면 -1 반환
- `put(key, value)`: 키-값 쌍 삽입, 용량 초과 시 LRU 항목 제거

모든 연산은 O(1) 시간 복잡도여야 합니다.

### 예시
```python
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1)       # 1 반환
cache.put(3, 3)    # 키 2 제거
cache.get(2)       # -1 반환 (제거됨)
```""",
        "difficulty": Difficulty.HARD,
        "category": Category.DESIGN,
        "constraints": "1 <= capacity <= 3000, 0 <= key, value <= 10^4",
        "hints": ["OrderedDict를 사용하면 쉽게 구현할 수 있습니다.", "해시맵 + 이중 연결 리스트로 직접 구현할 수도 있습니다."],
        "solution_template": """class LRUCache:
    def __init__(self, capacity: int):
        # 여기에 코드를 작성하세요
        pass

    def get(self, key: int) -> int:
        pass

    def put(self, key: int, value: int) -> None:
        pass


# 테스트
if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))   # 1
    cache.put(3, 3)
    print(cache.get(2))   # -1
""",
        "reference_solution": """from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "test_cases": [
            {"input": "LRUCache(2);put(1,1);put(2,2);get(1);put(3,3);get(2)", "expected_output": "1,-1", "is_sample": True},
        ],
    },
]


async def seed_problems():
    """Seed the database with initial problems"""
    await init_db()

    async with get_session_context() as session:
        # Check if problems already exist
        from sqlalchemy import select, func
        result = await session.execute(select(func.count()).select_from(ProblemModel))
        count = result.scalar()

        if count > 0:
            print(f"Database already has {count} problems. Skipping seed.")
            return

        # Insert problems - make a copy to avoid mutating original
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

        print(f"Successfully seeded {len(SEED_PROBLEMS)} problems with test cases.")


if __name__ == "__main__":
    asyncio.run(seed_problems())
