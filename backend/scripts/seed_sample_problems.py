"""Seed script for sample problems using SQLAlchemy.

Creates a small set of sample problems for testing and demonstration.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select

from code_tutor.shared.infrastructure.database import init_db, get_session_context
from code_tutor.learning.infrastructure.models import ProblemModel, TestCaseModel
from code_tutor.learning.domain.value_objects import Difficulty, Category


# ============== Sample Problems ==============

PROBLEMS_DATA = [
    # Easy - Array
    {
        "title": "Two Sum",
        "description": """정수 배열 `nums`와 목표값 `target`이 주어집니다.
두 수의 합이 `target`이 되는 두 수의 인덱스를 반환하세요.

### 입력
- 정수 배열 `nums` (2 ≤ len(nums) ≤ 10^4)
- 목표값 `target`

### 출력
- 두 수의 인덱스 리스트 [i, j] (i < j)

### 예제
```
입력: nums = [2, 7, 11, 15], target = 9
출력: [0, 1]
설명: nums[0] + nums[1] = 2 + 7 = 9
```

### 제약
- 각 입력에 대해 정확히 하나의 해가 존재합니다.
- 같은 요소를 두 번 사용할 수 없습니다.""",
        "difficulty": Difficulty.EASY,
        "category": Category.HASH_TABLE,
        "constraints": "2 <= len(nums) <= 10^4, -10^9 <= nums[i] <= 10^9",
        "hints": ["해시맵을 사용하면 O(n)에 풀 수 있습니다", "각 숫자를 순회하면서 target - num이 해시맵에 있는지 확인"],
        "solution_template": """def solution(nums: list[int], target: int) -> list[int]:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []""",
        "pattern_ids": ["hash-map", "two-sum"],
        "pattern_explanation": "해시맵을 사용하여 O(n) 시간 복잡도로 해결합니다. 각 숫자를 순회하면서 target - num이 이미 본 숫자 중에 있는지 확인합니다.",
        "approach_hint": "Hash Map",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "is_sample": True},
            {"input": "[3, 2, 4]\n6", "output": "[1, 2]", "is_sample": True},
            {"input": "[3, 3]\n6", "output": "[0, 1]", "is_sample": False},
            {"input": "[1, 5, 3, 7, 2]\n9", "output": "[3, 4]", "is_sample": False},
        ]
    },
    # Easy - String
    {
        "title": "문자열 뒤집기",
        "description": """주어진 문자열을 뒤집어서 반환하세요.

### 입력
- 문자열 `s` (1 ≤ len(s) ≤ 10^5)

### 출력
- 뒤집힌 문자열

### 예제
```
입력: "hello"
출력: "olleh"
```""",
        "difficulty": Difficulty.EASY,
        "category": Category.STRING,
        "constraints": "1 <= len(s) <= 10^5",
        "hints": ["파이썬 슬라이싱 활용", "s[::-1]"],
        "solution_template": """def solution(s: str) -> str:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(s: str) -> str:
    return s[::-1]""",
        "pattern_ids": ["string-manipulation"],
        "pattern_explanation": "파이썬의 슬라이싱 기능을 활용하면 한 줄로 해결할 수 있습니다.",
        "approach_hint": "Slicing",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "hello", "output": "olleh", "is_sample": True},
            {"input": "Python", "output": "nohtyP", "is_sample": True},
            {"input": "a", "output": "a", "is_sample": False},
        ]
    },
    # Medium - Two Pointers
    {
        "title": "컨테이너에 담기는 물",
        "description": """n개의 수직선이 주어집니다. 각 선 i는 좌표 (i, 0)에서 (i, height[i])까지 이어집니다.
두 선과 x축으로 이루어진 컨테이너에 담을 수 있는 물의 최대 양을 구하세요.

### 입력
- 높이 배열 `height` (2 ≤ len(height) ≤ 10^5)

### 출력
- 담을 수 있는 물의 최대 양

### 예제
```
입력: [1, 8, 6, 2, 5, 4, 8, 3, 7]
출력: 49
설명: 인덱스 1(높이 8)과 인덱스 8(높이 7) 사이에 7 * 7 = 49의 물을 담을 수 있습니다.
```""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.TWO_POINTERS,
        "constraints": "2 <= len(height) <= 10^5, 0 <= height[i] <= 10^4",
        "hints": ["양 끝에서 시작하는 투 포인터", "더 낮은 쪽의 포인터를 이동"],
        "solution_template": """def solution(height: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water""",
        "pattern_ids": ["two-pointers"],
        "pattern_explanation": "양 끝에서 시작하는 투 포인터를 사용합니다. 더 낮은 높이의 포인터를 이동시키면서 최대 넓이를 갱신합니다.",
        "approach_hint": "Two Pointers",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[1, 8, 6, 2, 5, 4, 8, 3, 7]", "output": "49", "is_sample": True},
            {"input": "[1, 1]", "output": "1", "is_sample": True},
            {"input": "[4, 3, 2, 1, 4]", "output": "16", "is_sample": False},
        ]
    },
    # Medium - Binary Search
    {
        "title": "회전 정렬 배열 탐색",
        "description": """오름차순으로 정렬된 배열이 어떤 피벗을 기준으로 회전되었습니다.
배열에서 target 값의 인덱스를 찾아 반환하세요. 없으면 -1을 반환합니다.

### 입력
- 회전된 정렬 배열 `nums` (1 ≤ len(nums) ≤ 5000)
- 찾을 값 `target`

### 출력
- target의 인덱스 또는 -1

### 예제
```
입력: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
출력: 4
```

### 제약
- 배열의 모든 값은 고유합니다.
- O(log n) 시간 복잡도로 해결해야 합니다.""",
        "difficulty": Difficulty.MEDIUM,
        "category": Category.BINARY_SEARCH,
        "constraints": "1 <= len(nums) <= 5000, -10^4 <= nums[i] <= 10^4",
        "hints": ["이진 탐색 변형", "왼쪽/오른쪽 중 정렬된 부분 찾기"],
        "solution_template": """def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # 왼쪽 부분이 정렬됨
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 오른쪽 부분이 정렬됨
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1""",
        "pattern_ids": ["binary-search", "rotated-array"],
        "pattern_explanation": "회전된 배열에서 이진 탐색을 적용합니다. mid를 기준으로 왼쪽/오른쪽 중 정렬된 부분을 찾고, target이 그 범위에 있는지 확인합니다.",
        "approach_hint": "Modified Binary Search",
        "time_complexity_hint": "O(log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[4, 5, 6, 7, 0, 1, 2]\n0", "output": "4", "is_sample": True},
            {"input": "[4, 5, 6, 7, 0, 1, 2]\n3", "output": "-1", "is_sample": True},
            {"input": "[1]\n0", "output": "-1", "is_sample": False},
            {"input": "[3, 1]\n1", "output": "1", "is_sample": False},
        ]
    },
    # Hard - Dynamic Programming
    {
        "title": "가장 긴 증가하는 부분 수열",
        "description": """정수 배열 `nums`가 주어질 때, 가장 긴 **순 증가** 부분 수열의 길이를 반환하세요.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 2500)

### 출력
- 가장 긴 증가하는 부분 수열의 길이

### 예제
```
입력: [10, 9, 2, 5, 3, 7, 101, 18]
출력: 4
설명: 가장 긴 증가 부분 수열은 [2, 3, 7, 101]로 길이는 4입니다.
```

### 힌트
- DP로 O(n²), 이진탐색으로 O(n log n)에 풀 수 있습니다.""",
        "difficulty": Difficulty.HARD,
        "category": Category.DYNAMIC_PROGRAMMING,
        "constraints": "1 <= len(nums) <= 2500, -10^4 <= nums[i] <= 10^4",
        "hints": ["dp[i] = nums[i]로 끝나는 LIS 길이", "이진 탐색 + dp 배열로 최적화 가능"],
        "solution_template": """def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(nums: list[int]) -> int:
    import bisect

    # O(n log n) 풀이
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)""",
        "pattern_ids": ["lis", "binary-search-dp"],
        "pattern_explanation": "LIS(Longest Increasing Subsequence) 패턴입니다. O(n²) DP 또는 O(n log n) 이진 탐색 최적화로 풀 수 있습니다.",
        "approach_hint": "Binary Search + DP",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[10, 9, 2, 5, 3, 7, 101, 18]", "output": "4", "is_sample": True},
            {"input": "[0, 1, 0, 3, 2, 3]", "output": "4", "is_sample": True},
            {"input": "[7, 7, 7, 7, 7]", "output": "1", "is_sample": False},
            {"input": "[1, 3, 6, 7, 9, 4, 10, 5, 6]", "output": "6", "is_sample": False},
        ]
    },
    # Hard - Graph
    {
        "title": "단어 사다리",
        "description": """시작 단어 `beginWord`에서 목표 단어 `endWord`로 변환하는 **최단 변환 횟수**를 구하세요.

변환 규칙:
- 한 번에 한 글자만 바꿀 수 있습니다.
- 변환된 단어는 `wordList`에 있어야 합니다.

### 입력
- 시작 단어 `beginWord`
- 목표 단어 `endWord`
- 단어 리스트 `wordList`

### 출력
- 최단 변환 횟수 (변환 불가능하면 0)

### 예제
```
입력: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
출력: 5
설명: "hit" -> "hot" -> "dot" -> "dog" -> "cog" (5단어)
```""",
        "difficulty": Difficulty.HARD,
        "category": Category.BFS,
        "constraints": "1 <= len(beginWord) <= 10, endWord.length == beginWord.length",
        "hints": ["BFS로 최단 경로 탐색", "각 위치별로 a-z 변환 시도"],
        "solution_template": """def solution(beginWord: str, endWord: str, wordList: list[str]) -> int:
    # 여기에 코드를 작성하세요
    pass""",
        "reference_solution": """def solution(beginWord: str, endWord: str, wordList: list[str]) -> int:
    from collections import deque

    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, steps = queue.popleft()

        if word == endWord:
            return steps

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))

    return 0""",
        "pattern_ids": ["bfs", "word-ladder"],
        "pattern_explanation": "BFS를 사용하여 최단 경로를 찾습니다. 각 단어에서 한 글자씩 변경하여 wordList에 있는 단어로 변환 가능한지 확인합니다.",
        "approach_hint": "BFS",
        "time_complexity_hint": "O(M² × N)",
        "space_complexity_hint": "O(M × N)",
        "test_cases": [
            {"input": "hit\ncog\n[\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"]", "output": "5", "is_sample": True},
            {"input": "hit\ncog\n[\"hot\",\"dot\",\"dog\",\"lot\",\"log\"]", "output": "0", "is_sample": True},
            {"input": "a\nc\n[\"a\",\"b\",\"c\"]", "output": "2", "is_sample": False},
        ]
    },
]


async def seed_sample_problems():
    """Seed sample problems."""
    print("Seeding Sample Problems...")

    await init_db()

    async with get_session_context() as session:
        created = 0
        skipped = 0

        for problem_data in PROBLEMS_DATA:
            # Check if problem already exists
            result = await session.execute(
                select(ProblemModel).where(ProblemModel.title == problem_data["title"])
            )
            if result.scalar_one_or_none():
                print(f"  [SKIP] {problem_data['title']} - already exists")
                skipped += 1
                continue

            # Create problem
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
                pattern_ids=problem_data["pattern_ids"],
                pattern_explanation=problem_data["pattern_explanation"],
                approach_hint=problem_data["approach_hint"],
                time_complexity_hint=problem_data["time_complexity_hint"],
                space_complexity_hint=problem_data["space_complexity_hint"],
                is_published=True,
            )
            session.add(problem)
            await session.flush()

            # Create test cases
            for order, tc in enumerate(problem_data["test_cases"], 1):
                test_case = TestCaseModel(
                    id=uuid4(),
                    problem_id=problem_id,
                    input_data=tc["input"],
                    expected_output=tc["output"],
                    is_sample=tc["is_sample"],
                    order=order,
                )
                session.add(test_case)

            print(f"  [ADD] {problem_data['title']} ({problem_data['difficulty'].value})")
            created += 1

        await session.commit()

    print(f"\nProblems seeding complete: {created} created, {skipped} skipped")
    return created, skipped


if __name__ == "__main__":
    asyncio.run(seed_sample_problems())
