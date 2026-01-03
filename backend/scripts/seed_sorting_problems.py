"""Phase 2: 정렬 알고리즘 문제 시딩 스크립트 (8문제)

알고리즘: 계수 정렬, 기수 정렬, 셸 정렬, 트리 정렬, 외부 정렬
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"

PROBLEMS = [
    # ============== EASY (2문제) ==============
    {
        "title": "계수 정렬 기초",
        "description": """0 이상 100 이하의 정수로 이루어진 배열을 **계수 정렬(Counting Sort)**로 정렬하세요.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^5)
- 모든 원소: 0 ≤ nums[i] ≤ 100

### 출력
- 오름차순으로 정렬된 배열

### 예제
```
입력: [4, 2, 2, 8, 3, 3, 1]
출력: [1, 2, 2, 3, 3, 4, 8]
```

### 힌트
- 비교 기반 정렬이 아닌 값의 범위를 이용한 정렬입니다.
- 시간복잡도: O(n + k), k는 값의 범위""",
        "difficulty": "easy",
        "category": "sorting",
        "constraints": "1 <= len(nums) <= 10^5, 0 <= nums[i] <= 100",
        "hints": ["크기 101의 카운트 배열 생성", "각 값의 등장 횟수를 센다", "카운트 배열을 순회하며 결과 생성"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    if not nums:
        return []

    max_val = 100
    count = [0] * (max_val + 1)

    for num in nums:
        count[num] += 1

    result = []
    for i in range(max_val + 1):
        result.extend([i] * count[i])

    return result""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["counting-sort"],
        "pattern_explanation": "계수 정렬은 값의 범위가 제한될 때 O(n+k) 시간에 정렬하는 비교 기반이 아닌 정렬입니다. 각 값의 등장 횟수를 세어 카운트 배열에 저장하고, 이를 순회하여 결과를 생성합니다.",
        "approach_hint": "카운트 배열에 빈도수 저장 후 재구성",
        "time_complexity_hint": "O(n + k)",
        "space_complexity_hint": "O(k)",
        "test_cases": [
            {"input": "[4, 2, 2, 8, 3, 3, 1]", "output": "[1, 2, 2, 3, 3, 4, 8]", "is_sample": True},
            {"input": "[0, 0, 0]", "output": "[0, 0, 0]", "is_sample": False},
            {"input": "[100, 50, 0]", "output": "[0, 50, 100]", "is_sample": False},
        ]
    },
    {
        "title": "셸 정렬 구현",
        "description": """정수 배열을 **셸 정렬(Shell Sort)**로 정렬하세요.

셸 정렬은 삽입 정렬의 개선 버전으로, 일정 간격(gap)으로 떨어진 원소들을 먼저 정렬합니다.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^4)

### 출력
- 오름차순으로 정렬된 배열

### 예제
```
입력: [12, 34, 54, 2, 3]
출력: [2, 3, 12, 34, 54]
```

### 셸 정렬 과정
1. gap = n // 2로 시작
2. gap 간격으로 떨어진 원소들을 삽입 정렬
3. gap을 절반으로 줄이며 반복
4. gap이 1이 되면 일반 삽입 정렬로 마무리""",
        "difficulty": "easy",
        "category": "sorting",
        "constraints": "1 <= len(nums) <= 10^4",
        "hints": ["gap을 n//2부터 시작해서 절반씩 줄인다", "각 gap에서 삽입 정렬 수행", "gap이 1일 때 최종 정렬"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    n = len(nums)
    nums = nums.copy()
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > temp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = temp
        gap //= 2

    return nums""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["shell-sort"],
        "pattern_explanation": "셸 정렬은 삽입 정렬의 개선 버전입니다. 멀리 떨어진 원소들을 먼저 교환하여 삽입 정렬의 약점(거의 정렬된 경우에만 효율적)을 보완합니다. gap 시퀀스에 따라 성능이 달라집니다.",
        "approach_hint": "gap을 줄여가며 삽입 정렬 반복",
        "time_complexity_hint": "O(n log n) ~ O(n^2)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[12, 34, 54, 2, 3]", "output": "[2, 3, 12, 34, 54]", "is_sample": True},
            {"input": "[5, 4, 3, 2, 1]", "output": "[1, 2, 3, 4, 5]", "is_sample": True},
            {"input": "[1]", "output": "[1]", "is_sample": False},
        ]
    },

    # ============== MEDIUM (3문제) ==============
    {
        "title": "정수 기수 정렬",
        "description": """양의 정수 배열을 **기수 정렬(Radix Sort)**로 정렬하세요.

기수 정렬은 각 자릿수를 기준으로 정렬하는 방식입니다.

### 입력
- 양의 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^5)
- 모든 원소: 0 ≤ nums[i] ≤ 10^9

### 출력
- 오름차순으로 정렬된 배열

### 예제
```
입력: [170, 45, 75, 90, 802, 24, 2, 66]
출력: [2, 24, 45, 66, 75, 90, 170, 802]
```

### 기수 정렬 과정
1. 최대값의 자릿수만큼 반복
2. 각 반복에서 해당 자릿수를 기준으로 안정 정렬(계수 정렬) 수행""",
        "difficulty": "medium",
        "category": "sorting",
        "constraints": "1 <= len(nums) <= 10^5, 0 <= nums[i] <= 10^9",
        "hints": ["최대값의 자릿수 계산", "각 자릿수마다 계수 정렬 적용", "10진수 기준 0~9 버킷 사용"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    if not nums:
        return []

    nums = nums.copy()
    max_val = max(nums)
    exp = 1

    while max_val // exp > 0:
        # 현재 자릿수 기준 계수 정렬
        count = [0] * 10
        output = [0] * len(nums)

        for num in nums:
            index = (num // exp) % 10
            count[index] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(len(nums) - 1, -1, -1):
            index = (nums[i] // exp) % 10
            output[count[index] - 1] = nums[i]
            count[index] -= 1

        nums = output
        exp *= 10

    return nums""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["radix-sort"],
        "pattern_explanation": "기수 정렬은 각 자릿수별로 안정 정렬(주로 계수 정렬)을 수행합니다. 10진수 정수의 경우 0~9 버킷을 사용합니다. 시간복잡도는 O(d*(n+k))이며, d는 자릿수, k는 기수(10)입니다.",
        "approach_hint": "자릿수별로 계수 정렬 반복",
        "time_complexity_hint": "O(d * (n + k))",
        "space_complexity_hint": "O(n + k)",
        "test_cases": [
            {"input": "[170, 45, 75, 90, 802, 24, 2, 66]", "output": "[2, 24, 45, 66, 75, 90, 170, 802]", "is_sample": True},
            {"input": "[1, 10, 100, 1000]", "output": "[1, 10, 100, 1000]", "is_sample": False},
            {"input": "[999, 99, 9]", "output": "[9, 99, 999]", "is_sample": False},
        ]
    },
    {
        "title": "음수 포함 계수 정렬",
        "description": """-1000부터 1000 사이의 정수를 **계수 정렬**로 정렬하세요.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^5)
- -1000 ≤ nums[i] ≤ 1000

### 출력
- 오름차순으로 정렬된 배열

### 예제
```
입력: [4, -2, 2, -8, 3, -3, 1]
출력: [-8, -3, -2, 1, 2, 3, 4]
```

### 힌트
- 음수를 처리하려면 오프셋을 사용하세요.""",
        "difficulty": "medium",
        "category": "sorting",
        "constraints": "-1000 <= nums[i] <= 1000",
        "hints": ["오프셋 1000을 더해 0~2000 범위로 변환", "계수 정렬 수행", "결과에서 오프셋을 다시 뺀다"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    if not nums:
        return []

    OFFSET = 1000
    count = [0] * 2001  # -1000 ~ 1000

    for num in nums:
        count[num + OFFSET] += 1

    result = []
    for i in range(2001):
        result.extend([i - OFFSET] * count[i])

    return result""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["counting-sort"],
        "pattern_explanation": "음수를 포함한 계수 정렬은 오프셋을 사용하여 모든 값을 양수 범위로 변환합니다. 예: -1000~1000을 0~2000으로 변환. 정렬 후 오프셋을 다시 빼서 원래 값을 복원합니다.",
        "approach_hint": "오프셋으로 음수를 양수 인덱스로 변환",
        "time_complexity_hint": "O(n + k)",
        "space_complexity_hint": "O(k)",
        "test_cases": [
            {"input": "[4, -2, 2, -8, 3, -3, 1]", "output": "[-8, -3, -2, 1, 2, 3, 4]", "is_sample": True},
            {"input": "[-1000, 1000, 0]", "output": "[-1000, 0, 1000]", "is_sample": False},
            {"input": "[-5, -5, -5]", "output": "[-5, -5, -5]", "is_sample": False},
        ]
    },
    {
        "title": "최적 셸 정렬",
        "description": """**Hibbard 간격 시퀀스**를 사용하여 셸 정렬을 구현하세요.

Hibbard 간격: 1, 3, 7, 15, 31, ... (2^k - 1)

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^4)

### 출력
- 오름차순으로 정렬된 배열과 사용된 간격 시퀀스를 튜플로 반환

### 예제
```
입력: [12, 34, 54, 2, 3, 9, 7, 1]
출력: ([1, 2, 3, 7, 9, 12, 34, 54], [7, 3, 1])
```

### 왜 Hibbard 간격인가?
- 기본 n/2 간격보다 최악 시간복잡도가 개선됨
- O(n^(3/2))로 개선""",
        "difficulty": "medium",
        "category": "sorting",
        "constraints": "1 <= len(nums) <= 10^4",
        "hints": ["2^k - 1 형태의 간격 생성", "n보다 작은 가장 큰 간격부터 시작", "각 간격에서 삽입 정렬"],
        "solution_template": "def solution(nums: list) -> tuple:\n    pass",
        "reference_solution": """def solution(nums: list) -> tuple:
    n = len(nums)
    nums = nums.copy()

    # Hibbard 간격 시퀀스 생성: 2^k - 1
    gaps = []
    k = 1
    while (2 ** k - 1) < n:
        gaps.append(2 ** k - 1)
        k += 1
    gaps.reverse()

    used_gaps = []
    for gap in gaps:
        used_gaps.append(gap)
        for i in range(gap, n):
            temp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > temp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = temp

    return (nums, used_gaps)""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["shell-sort"],
        "pattern_explanation": "Hibbard 간격(2^k - 1)을 사용하면 셸 정렬의 최악 시간복잡도가 O(n^2)에서 O(n^(3/2))로 개선됩니다. 다른 간격 시퀀스(Sedgewick, Knuth 등)도 연구되어 있습니다.",
        "approach_hint": "2^k - 1 간격 시퀀스 사용",
        "time_complexity_hint": "O(n^(3/2))",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[12, 34, 54, 2, 3, 9, 7, 1]", "output": "([1, 2, 3, 7, 9, 12, 34, 54], [7, 3, 1])", "is_sample": True},
            {"input": "[5, 4, 3, 2, 1]", "output": "([1, 2, 3, 4, 5], [3, 1])", "is_sample": False},
        ]
    },

    # ============== HARD (3문제) ==============
    {
        "title": "문자열 기수 정렬",
        "description": """같은 길이의 문자열 배열을 **MSD(Most Significant Digit) 기수 정렬**로 정렬하세요.

### 입력
- 문자열 배열 `strs` (모든 문자열 길이 동일, 소문자만 포함)

### 출력
- 사전순으로 정렬된 배열

### 예제
```
입력: ["cat", "bat", "cab", "abc", "bac"]
출력: ["abc", "bac", "bat", "cab", "cat"]
```

### MSD 기수 정렬
- 가장 왼쪽(Most Significant) 자릿수부터 정렬
- 각 자릿수에서 26개 버킷(a-z) 사용""",
        "difficulty": "hard",
        "category": "sorting",
        "constraints": "모든 문자열 길이 동일, 소문자만",
        "hints": ["26개 버킷(a-z) 사용", "왼쪽 자릿수부터 정렬", "재귀적으로 각 버킷 정렬"],
        "solution_template": "def solution(strs: list) -> list:\n    pass",
        "reference_solution": """def solution(strs: list) -> list:
    if not strs:
        return []

    def msd_sort(arr, d, length):
        if len(arr) <= 1 or d >= length:
            return arr

        buckets = [[] for _ in range(26)]
        for s in arr:
            idx = ord(s[d]) - ord('a')
            buckets[idx].append(s)

        result = []
        for bucket in buckets:
            if bucket:
                result.extend(msd_sort(bucket, d + 1, length))
        return result

    str_len = len(strs[0])
    return msd_sort(strs, 0, str_len)""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["radix-sort", "msd-radix-sort"],
        "pattern_explanation": "MSD(Most Significant Digit) 기수 정렬은 가장 중요한 자릿수(문자열의 경우 첫 문자)부터 정렬합니다. 각 자릿수에서 버킷을 만들고 재귀적으로 정렬합니다. 문자열 정렬에 적합합니다.",
        "approach_hint": "첫 문자부터 26개 버킷으로 분류 후 재귀",
        "time_complexity_hint": "O(n * k)",
        "space_complexity_hint": "O(n + k)",
        "test_cases": [
            {"input": '["cat", "bat", "cab", "abc", "bac"]', "output": '["abc", "bac", "bat", "cab", "cat"]', "is_sample": True},
            {"input": '["aaa", "aab", "aba", "baa"]', "output": '["aaa", "aab", "aba", "baa"]', "is_sample": False},
        ]
    },
    {
        "title": "BST 기반 트리 정렬",
        "description": """이진 탐색 트리(BST)를 사용하여 정수 배열을 정렬하세요.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^4)

### 출력
- 오름차순으로 정렬된 배열

### 트리 정렬 과정
1. 모든 원소를 BST에 삽입
2. BST를 중위 순회(inorder traversal)하여 정렬된 결과 획득

### 예제
```
입력: [5, 3, 7, 2, 4, 6, 8]
출력: [2, 3, 4, 5, 6, 7, 8]
```""",
        "difficulty": "hard",
        "category": "sorting",
        "constraints": "1 <= len(nums) <= 10^4",
        "hints": ["BST 노드 클래스 정의", "삽입 함수 구현", "중위 순회로 정렬된 결과 추출"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def insert(root, val):
        if root is None:
            return TreeNode(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)
        return root

    def inorder(root, result):
        if root:
            inorder(root.left, result)
            result.append(root.val)
            inorder(root.right, result)

    if not nums:
        return []

    root = None
    for num in nums:
        root = insert(root, num)

    result = []
    inorder(root, result)
    return result""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-sort", "bst"],
        "pattern_explanation": "트리 정렬은 BST의 특성을 활용합니다. BST에 모든 원소를 삽입하면 중위 순회 시 정렬된 순서로 방문됩니다. 평균 O(n log n)이지만 불균형 트리에서는 O(n^2)입니다.",
        "approach_hint": "BST 삽입 후 중위 순회",
        "time_complexity_hint": "O(n log n) 평균, O(n^2) 최악",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[5, 3, 7, 2, 4, 6, 8]", "output": "[2, 3, 4, 5, 6, 7, 8]", "is_sample": True},
            {"input": "[1, 2, 3, 4, 5]", "output": "[1, 2, 3, 4, 5]", "is_sample": False},
            {"input": "[5, 5, 5]", "output": "[5, 5, 5]", "is_sample": False},
        ]
    },
    {
        "title": "외부 정렬 시뮬레이션",
        "description": """메모리에 한 번에 `k`개의 원소만 올릴 수 있을 때, 대용량 배열을 정렬하세요.

외부 정렬의 핵심:
1. 배열을 k개씩 나누어 각각 정렬 (Run 생성)
2. 정렬된 Run들을 병합 (K-way Merge)

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^4)
- 메모리 제한 `k` (2 ≤ k ≤ len(nums))

### 출력
- 정렬된 배열

### 예제
```
입력: nums = [5, 3, 8, 1, 2, 7, 4, 6], k = 3
출력: [1, 2, 3, 4, 5, 6, 7, 8]
```

### 과정
1. Run 생성: [3,5,8], [1,2,7], [4,6]
2. K-way Merge로 병합""",
        "difficulty": "hard",
        "category": "sorting",
        "constraints": "2 <= k <= len(nums)",
        "hints": ["배열을 k개씩 나눠 각각 정렬", "힙을 사용한 K-way Merge", "각 Run의 첫 원소를 힙에 넣고 병합"],
        "solution_template": "def solution(nums: list, k: int) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, k: int) -> list:
    import heapq

    if not nums:
        return []

    # 1. Run 생성: k개씩 나누어 정렬
    runs = []
    for i in range(0, len(nums), k):
        run = sorted(nums[i:i+k])
        runs.append(run)

    # 2. K-way Merge using min-heap
    heap = []
    for run_idx, run in enumerate(runs):
        if run:
            heapq.heappush(heap, (run[0], run_idx, 0))

    result = []
    while heap:
        val, run_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        if elem_idx + 1 < len(runs[run_idx]):
            next_val = runs[run_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, run_idx, elem_idx + 1))

    return result""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["external-sort", "k-way-merge"],
        "pattern_explanation": "외부 정렬은 메모리보다 큰 데이터를 정렬할 때 사용합니다. 데이터를 청크로 나눠 정렬(Run 생성)하고, 힙을 사용한 K-way Merge로 병합합니다. 실제로는 디스크 I/O를 고려해야 합니다.",
        "approach_hint": "Run 생성 + 힙 기반 K-way Merge",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[5, 3, 8, 1, 2, 7, 4, 6]\n3", "output": "[1, 2, 3, 4, 5, 6, 7, 8]", "is_sample": True},
            {"input": "[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]\n4", "output": "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", "is_sample": False},
        ]
    },
]


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    added = 0
    skipped = 0

    for problem in PROBLEMS:
        cursor.execute("SELECT id FROM problems WHERE title = ?", (problem["title"],))
        if cursor.fetchone():
            print(f"  [건너뜀] 이미 존재: {problem['title']}")
            skipped += 1
            continue

        problem_id = uuid.uuid4().hex

        cursor.execute("""
            INSERT INTO problems (
                id, title, description, difficulty, category, constraints,
                hints, solution_template, reference_solution,
                time_limit_ms, memory_limit_mb, is_published,
                pattern_ids, pattern_explanation, approach_hint,
                time_complexity_hint, space_complexity_hint,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_id,
            problem["title"],
            problem["description"],
            problem["difficulty"],
            problem["category"],
            problem["constraints"],
            json.dumps(problem["hints"], ensure_ascii=False),
            problem["solution_template"],
            problem["reference_solution"],
            problem["time_limit_ms"],
            problem["memory_limit_mb"],
            1,
            json.dumps(problem.get("pattern_ids", []), ensure_ascii=False),
            problem.get("pattern_explanation", ""),
            problem.get("approach_hint", ""),
            problem.get("time_complexity_hint", ""),
            problem.get("space_complexity_hint", ""),
            now,
            now
        ))

        for i, tc in enumerate(problem["test_cases"]):
            tc_id = uuid.uuid4().hex
            cursor.execute("""
                INSERT INTO test_cases (id, problem_id, input_data, expected_output, is_sample, "order", created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tc_id,
                problem_id,
                tc["input"],
                tc["output"],
                1 if tc.get("is_sample", False) else 0,
                i,
                now
            ))

        print(f"  [추가됨] {problem['title']} ({problem['difficulty']})")
        added += 1

    conn.commit()
    conn.close()

    print(f"\n{'='*50}")
    print(f"Phase 2 정렬 알고리즘 시딩 완료")
    print(f"  - 추가: {added}개")
    print(f"  - 건너뜀: {skipped}개")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("Phase 2: 정렬 알고리즘 문제 시딩 시작...")
    main()
