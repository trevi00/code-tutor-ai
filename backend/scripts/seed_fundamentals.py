"""Phase 1: 기초 자료구조 & 테크닉 문제 시딩 스크립트 (12문제)

카테고리: 문자열, 리스트, 딕셔너리, 해시맵, Prefix Sum, 소수 판별
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

# 상위 디렉토리의 DB 사용
DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"

PROBLEMS = [
    # ============== EASY (5문제) ==============
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
```

### 힌트
- 슬라이싱을 활용하거나 투 포인터를 사용할 수 있습니다.""",
        "difficulty": "easy",
        "category": "string",
        "constraints": "1 <= len(s) <= 10^5",
        "hints": ["파이썬 슬라이싱 s[::-1] 활용", "투 포인터로 양 끝에서 스왑"],
        "solution_template": "def solution(s: str) -> str:\n    pass",
        "reference_solution": """def solution(s: str) -> str:
    return s[::-1]""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["string-manipulation"],
        "pattern_explanation": "문자열 뒤집기는 가장 기본적인 문자열 조작 패턴입니다. 파이썬에서는 슬라이싱 s[::-1]로 간단히 해결할 수 있고, 투 포인터를 사용하면 in-place로 O(1) 공간에 처리할 수 있습니다.",
        "approach_hint": "슬라이싱 또는 투 포인터",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "hello", "output": "olleh", "is_sample": True},
            {"input": "Python", "output": "nohtyP", "is_sample": True},
            {"input": "a", "output": "a", "is_sample": False},
            {"input": "abcdefg", "output": "gfedcba", "is_sample": False},
        ]
    },
    {
        "title": "리스트 회전",
        "description": """정수 리스트를 오른쪽으로 `k`칸 회전시키세요.

### 입력
- 정수 리스트 `nums` (1 ≤ len(nums) ≤ 10^5)
- 회전 횟수 `k` (0 ≤ k ≤ 10^9)

### 출력
- 오른쪽으로 k칸 회전된 리스트

### 예제
```
입력: nums = [1, 2, 3, 4, 5], k = 2
출력: [4, 5, 1, 2, 3]
```

### 설명
- 1회전: [5, 1, 2, 3, 4]
- 2회전: [4, 5, 1, 2, 3]""",
        "difficulty": "easy",
        "category": "array",
        "constraints": "1 <= len(nums) <= 10^5, 0 <= k <= 10^9",
        "hints": ["k가 len(nums)보다 클 수 있으므로 k %= len(nums)", "슬라이싱으로 간단히 해결 가능"],
        "solution_template": "def solution(nums: list, k: int) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, k: int) -> list:
    n = len(nums)
    k = k % n
    return nums[-k:] + nums[:-k] if k else nums""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["array-manipulation"],
        "pattern_explanation": "리스트 회전은 슬라이싱을 활용한 배열 조작 패턴입니다. k가 배열 길이보다 클 수 있으므로 k %= n으로 정규화하는 것이 핵심입니다. nums[-k:] + nums[:-k]로 한 줄에 해결 가능합니다.",
        "approach_hint": "슬라이싱 활용",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]\n2", "output": "[4, 5, 1, 2, 3]", "is_sample": True},
            {"input": "[1, 2, 3]\n1", "output": "[3, 1, 2]", "is_sample": True},
            {"input": "[1, 2, 3]\n0", "output": "[1, 2, 3]", "is_sample": False},
            {"input": "[1, 2, 3]\n6", "output": "[1, 2, 3]", "is_sample": False},
        ]
    },
    {
        "title": "빈도수 계산",
        "description": """문자열에서 각 문자의 등장 횟수를 딕셔너리로 반환하세요.

### 입력
- 문자열 `s` (알파벳 소문자로만 구성)

### 출력
- 각 문자의 빈도수를 담은 딕셔너리 (알파벳 순 정렬)

### 예제
```
입력: "banana"
출력: {"a": 3, "b": 1, "n": 2}
```""",
        "difficulty": "easy",
        "category": "hash_table",
        "constraints": "1 <= len(s) <= 10^5",
        "hints": ["collections.Counter 활용", "딕셔너리 컴프리헨션 사용"],
        "solution_template": "def solution(s: str) -> dict:\n    pass",
        "reference_solution": """def solution(s: str) -> dict:
    from collections import Counter
    return dict(sorted(Counter(s).items()))""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hash-frequency"],
        "pattern_explanation": "빈도수 계산은 해시맵(딕셔너리)의 가장 기본적인 활용입니다. collections.Counter를 사용하면 한 줄로 해결할 수 있습니다. 문자/숫자 등장 횟수를 O(n)에 계산할 수 있는 핵심 패턴입니다.",
        "approach_hint": "해시맵/Counter 활용",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(k) - k는 고유 문자 수",
        "test_cases": [
            {"input": "banana", "output": '{"a": 3, "b": 1, "n": 2}', "is_sample": True},
            {"input": "hello", "output": '{"e": 1, "h": 1, "l": 2, "o": 1}', "is_sample": True},
            {"input": "aaa", "output": '{"a": 3}', "is_sample": False},
        ]
    },
    {
        "title": "구간 합 구하기",
        "description": """정수 배열과 쿼리가 주어집니다. 각 쿼리는 `[left, right]` 구간의 합을 구합니다.

### 입력
- 정수 배열 `nums` (1 ≤ len(nums) ≤ 10^4)
- 쿼리 배열 `queries` (각 쿼리는 [left, right], 0-indexed)

### 출력
- 각 쿼리에 대한 구간 합 리스트

### 예제
```
입력: nums = [1, 2, 3, 4, 5], queries = [[0, 2], [1, 4], [0, 4]]
출력: [6, 14, 15]
```

### 설명
- [0, 2]: 1 + 2 + 3 = 6
- [1, 4]: 2 + 3 + 4 + 5 = 14
- [0, 4]: 1 + 2 + 3 + 4 + 5 = 15""",
        "difficulty": "easy",
        "category": "array",
        "constraints": "1 <= len(nums) <= 10^4, 1 <= len(queries) <= 10^3",
        "hints": ["Prefix Sum 배열을 미리 계산", "prefix[r+1] - prefix[l]로 O(1) 쿼리"],
        "solution_template": "def solution(nums: list, queries: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, queries: list) -> list:
    # Prefix Sum 계산
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    # 각 쿼리 처리
    result = []
    for left, right in queries:
        result.append(prefix[right + 1] - prefix[left])
    return result""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["prefix-sum"],
        "pattern_explanation": "Prefix Sum(누적합)은 구간 합을 O(1)에 계산할 수 있게 해주는 핵심 테크닉입니다. prefix[i]는 0부터 i-1까지의 합을 저장하고, 구간 [l, r]의 합은 prefix[r+1] - prefix[l]로 계산합니다.",
        "approach_hint": "Prefix Sum 전처리",
        "time_complexity_hint": "O(n + q) - n: 배열 크기, q: 쿼리 수",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]\n[[0, 2], [1, 4], [0, 4]]", "output": "[6, 14, 15]", "is_sample": True},
            {"input": "[5, -1, 3, 2]\n[[0, 1], [2, 3]]", "output": "[4, 5]", "is_sample": True},
            {"input": "[10]\n[[0, 0]]", "output": "[10]", "is_sample": False},
        ]
    },
    {
        "title": "소수 판별",
        "description": """주어진 정수 `n`이 소수인지 판별하세요.

### 입력
- 정수 `n` (1 ≤ n ≤ 10^6)

### 출력
- 소수이면 True, 아니면 False

### 예제
```
입력: 17
출력: True

입력: 1
출력: False
```

### 힌트
- 1은 소수가 아닙니다.
- 2보다 큰 짝수는 소수가 아닙니다.""",
        "difficulty": "easy",
        "category": "math",
        "constraints": "1 <= n <= 10^6",
        "hints": ["sqrt(n)까지만 확인하면 됨", "2와 3을 먼저 처리하고 6k±1 형태로 최적화"],
        "solution_template": "def solution(n: int) -> bool:\n    pass",
        "reference_solution": """def solution(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["prime-check"],
        "pattern_explanation": "소수 판별은 수학 문제의 기초입니다. n의 약수는 sqrt(n) 이하에서만 찾으면 되므로 O(sqrt(n))에 판별 가능합니다. 2와 3을 먼저 처리하고 6k±1 형태만 검사하면 더 최적화됩니다.",
        "approach_hint": "sqrt(n)까지 나눗셈 검사",
        "time_complexity_hint": "O(sqrt(n))",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "17", "output": "True", "is_sample": True},
            {"input": "1", "output": "False", "is_sample": True},
            {"input": "2", "output": "True", "is_sample": False},
            {"input": "100", "output": "False", "is_sample": False},
            {"input": "97", "output": "True", "is_sample": False},
        ]
    },

    # ============== MEDIUM (5문제) ==============
    {
        "title": "문자열 압축",
        "description": """연속으로 반복되는 문자를 압축하세요.

### 입력
- 문자열 `s` (알파벳 대소문자로 구성)

### 출력
- 압축된 문자열 (압축된 결과가 원본보다 길면 원본 반환)

### 예제
```
입력: "aabcccccaaa"
출력: "a2b1c5a3"

입력: "abcdef"
출력: "abcdef"  # 압축 결과 "a1b1c1d1e1f1"이 더 길므로 원본 반환
```""",
        "difficulty": "medium",
        "category": "string",
        "constraints": "1 <= len(s) <= 10^5",
        "hints": ["연속 문자 카운트", "StringBuilder 패턴 사용", "결과 길이 비교"],
        "solution_template": "def solution(s: str) -> str:\n    pass",
        "reference_solution": """def solution(s: str) -> str:
    if not s:
        return s

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(s[i-1] + str(count))
            count = 1

    result.append(s[-1] + str(count))
    compressed = ''.join(result)

    return compressed if len(compressed) < len(s) else s""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["string-manipulation", "run-length-encoding"],
        "pattern_explanation": "문자열 압축은 Run-Length Encoding(RLE)의 기본 형태입니다. 연속된 동일 문자를 세어 '문자+개수' 형태로 변환합니다. 결과가 원본보다 길면 원본을 반환하는 조건 처리가 핵심입니다.",
        "approach_hint": "Run-Length Encoding",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "aabcccccaaa", "output": "a2b1c5a3", "is_sample": True},
            {"input": "abcdef", "output": "abcdef", "is_sample": True},
            {"input": "aaa", "output": "a3", "is_sample": False},
            {"input": "a", "output": "a", "is_sample": False},
        ]
    },
    {
        "title": "중복 제거 및 정렬",
        "description": """정수 리스트에서 중복을 제거하고 오름차순으로 정렬하세요.

### 입력
- 정수 리스트 `nums`

### 출력
- 중복 제거 후 정렬된 리스트

### 예제
```
입력: [3, 1, 2, 3, 4, 1, 2, 5]
출력: [1, 2, 3, 4, 5]
```""",
        "difficulty": "medium",
        "category": "hash_table",
        "constraints": "1 <= len(nums) <= 10^5, -10^9 <= nums[i] <= 10^9",
        "hints": ["set을 사용하면 O(n)으로 중복 제거 가능", "sorted() 사용"],
        "solution_template": "def solution(nums: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list) -> list:
    return sorted(set(nums))""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hash-set", "sorting"],
        "pattern_explanation": "해시 셋(set)은 O(1) 평균 시간에 중복 검사를 수행합니다. set()으로 중복을 제거하고 sorted()로 정렬하면 간단히 해결됩니다. 파이썬의 sorted(set(nums))는 매우 자주 사용되는 패턴입니다.",
        "approach_hint": "해시 셋으로 중복 제거 후 정렬",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[3, 1, 2, 3, 4, 1, 2, 5]", "output": "[1, 2, 3, 4, 5]", "is_sample": True},
            {"input": "[5, 5, 5, 5]", "output": "[5]", "is_sample": True},
            {"input": "[1]", "output": "[1]", "is_sample": False},
        ]
    },
    {
        "title": "애너그램 그룹핑",
        "description": """문자열 리스트가 주어지면, 애너그램끼리 그룹핑하세요.

애너그램: 글자를 재배열하면 같아지는 문자열 (예: "eat", "tea", "ate")

### 입력
- 문자열 리스트 `strs`

### 출력
- 애너그램 그룹 리스트 (각 그룹 내 문자열은 알파벳순, 그룹은 첫 번째 문자열 기준 알파벳순)

### 예제
```
입력: ["eat", "tea", "tan", "ate", "nat", "bat"]
출력: [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
```""",
        "difficulty": "medium",
        "category": "hash_table",
        "constraints": "1 <= len(strs) <= 10^4, 0 <= len(strs[i]) <= 100",
        "hints": ["정렬된 문자열을 키로 사용", "defaultdict(list) 활용"],
        "solution_template": "def solution(strs: list) -> list:\n    pass",
        "reference_solution": """def solution(strs: list) -> list:
    from collections import defaultdict

    anagrams = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        anagrams[key].append(s)

    result = [sorted(group) for group in anagrams.values()]
    return sorted(result, key=lambda x: x[0])""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hash-grouping", "anagram"],
        "pattern_explanation": "애너그램 그룹핑은 해시맵을 이용한 그룹핑의 대표적인 예입니다. 정렬된 문자열을 키로 사용하면 같은 문자 구성을 가진 단어들이 동일한 키에 매핑됩니다. defaultdict(list)를 활용하면 코드가 간결해집니다.",
        "approach_hint": "정렬된 문자열을 해시 키로 사용",
        "time_complexity_hint": "O(n * k log k) - k는 문자열 최대 길이",
        "space_complexity_hint": "O(n * k)",
        "test_cases": [
            {"input": '["eat", "tea", "tan", "ate", "nat", "bat"]', "output": '[["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]', "is_sample": True},
            {"input": '[""]', "output": '[[""]]', "is_sample": False},
            {"input": '["a"]', "output": '[["a"]]', "is_sample": False},
        ]
    },
    {
        "title": "2D 구간 합",
        "description": """2차원 배열에서 특정 영역의 합을 빠르게 구하세요.

### 입력
- 2차원 정수 배열 `matrix` (m x n)
- 쿼리 배열 `queries` (각 쿼리는 [r1, c1, r2, c2] 형태)

### 출력
- 각 쿼리에 대한 영역 합 리스트

### 예제
```
입력:
matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
queries = [[0, 0, 1, 1], [1, 1, 2, 2]]

출력: [12, 28]
```

### 설명
- [0,0,1,1]: 1+2+4+5 = 12
- [1,1,2,2]: 5+6+8+9 = 28""",
        "difficulty": "medium",
        "category": "array",
        "constraints": "1 <= m, n <= 200, 1 <= queries <= 10^4",
        "hints": ["2D Prefix Sum 배열 생성", "포함-배제 원리 적용"],
        "solution_template": "def solution(matrix: list, queries: list) -> list:\n    pass",
        "reference_solution": """def solution(matrix: list, queries: list) -> list:
    if not matrix or not matrix[0]:
        return []

    m, n = len(matrix), len(matrix[0])

    # 2D Prefix Sum 계산
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            prefix[i+1][j+1] = (prefix[i][j+1] + prefix[i+1][j]
                               - prefix[i][j] + matrix[i][j])

    # 쿼리 처리
    result = []
    for r1, c1, r2, c2 in queries:
        total = (prefix[r2+1][c2+1] - prefix[r1][c2+1]
                - prefix[r2+1][c1] + prefix[r1][c1])
        result.append(total)

    return result""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["prefix-sum-2d"],
        "pattern_explanation": "2D Prefix Sum은 1D 누적합을 2차원으로 확장한 것입니다. prefix[i][j]는 (0,0)부터 (i-1,j-1)까지의 합을 저장합니다. 영역 합은 포함-배제 원리로 O(1)에 계산: prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]",
        "approach_hint": "2D Prefix Sum 전처리 + 포함-배제",
        "time_complexity_hint": "O(m*n + q)",
        "space_complexity_hint": "O(m*n)",
        "test_cases": [
            {"input": "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n[[0, 0, 1, 1], [1, 1, 2, 2]]", "output": "[12, 28]", "is_sample": True},
            {"input": "[[1, 2], [3, 4]]\n[[0, 0, 0, 0], [0, 0, 1, 1]]", "output": "[1, 10]", "is_sample": False},
        ]
    },
    {
        "title": "N번째 소수",
        "description": """N번째 소수를 구하세요.

### 입력
- 정수 `n` (1 ≤ n ≤ 10^4)

### 출력
- n번째 소수

### 예제
```
입력: 6
출력: 13

설명: 처음 6개 소수는 2, 3, 5, 7, 11, 13
```""",
        "difficulty": "medium",
        "category": "math",
        "constraints": "1 <= n <= 10^4",
        "hints": ["에라토스테네스의 체 사용", "충분히 큰 범위로 체를 생성"],
        "solution_template": "def solution(n: int) -> int:\n    pass",
        "reference_solution": """def solution(n: int) -> int:
    # n번째 소수의 상한 추정 (n >= 6일 때 n*ln(n) + n*ln(ln(n)))
    import math
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 100

    # 에라토스테네스의 체
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    # n번째 소수 찾기
    count = 0
    for i in range(2, limit + 1):
        if sieve[i]:
            count += 1
            if count == n:
                return i

    return -1""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["sieve-of-eratosthenes"],
        "pattern_explanation": "에라토스테네스의 체는 특정 범위 내 모든 소수를 효율적으로 찾는 알고리즘입니다. 2부터 시작해 각 소수의 배수를 지워나갑니다. n번째 소수의 상한은 n*ln(n) + n*ln(ln(n))로 추정할 수 있습니다.",
        "approach_hint": "에라토스테네스의 체",
        "time_complexity_hint": "O(n log log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "6", "output": "13", "is_sample": True},
            {"input": "1", "output": "2", "is_sample": True},
            {"input": "100", "output": "541", "is_sample": False},
            {"input": "1000", "output": "7919", "is_sample": False},
        ]
    },

    # ============== HARD (2문제) ==============
    {
        "title": "문자열 패턴 매칭 (KMP)",
        "description": """텍스트에서 패턴이 등장하는 모든 시작 인덱스를 찾으세요.

### 입력
- 텍스트 `text` (1 ≤ len(text) ≤ 10^6)
- 패턴 `pattern` (1 ≤ len(pattern) ≤ 10^4)

### 출력
- 패턴이 등장하는 시작 인덱스 리스트 (0-indexed)

### 예제
```
입력: text = "ABABDABACDABABCABAB", pattern = "ABABC"
출력: [10]

입력: text = "AAAAAA", pattern = "AA"
출력: [0, 1, 2, 3, 4]
```

### 힌트
- 단순 매칭은 O(n*m)이지만 KMP는 O(n+m)입니다.""",
        "difficulty": "hard",
        "category": "string",
        "constraints": "1 <= len(text) <= 10^6, 1 <= len(pattern) <= 10^4",
        "hints": ["Failure Function(부분 일치 테이블) 계산", "불일치 시 패턴의 적절한 위치로 점프"],
        "solution_template": "def solution(text: str, pattern: str) -> list:\n    pass",
        "reference_solution": """def solution(text: str, pattern: str) -> list:
    def compute_lps(pattern):
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)

    result = []
    i = j = 0

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return result""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["kmp-algorithm", "string-matching"],
        "pattern_explanation": "KMP 알고리즘은 패턴 매칭을 O(n+m)에 수행합니다. 핵심은 LPS(Longest Proper Prefix which is also Suffix) 배열입니다. 불일치 시 패턴의 적절한 위치로 점프하여 불필요한 비교를 건너뜁니다. 문자열 알고리즘의 필수 기법입니다.",
        "approach_hint": "KMP 알고리즘 - LPS 배열 활용",
        "time_complexity_hint": "O(n + m)",
        "space_complexity_hint": "O(m)",
        "test_cases": [
            {"input": "ABABDABACDABABCABAB\nABABC", "output": "[10]", "is_sample": True},
            {"input": "AAAAAA\nAA", "output": "[0, 1, 2, 3, 4]", "is_sample": True},
            {"input": "ABC\nD", "output": "[]", "is_sample": False},
        ]
    },
    {
        "title": "해시맵 그래프 연결",
        "description": """사전 형태로 주어진 관계를 그래프로 변환하고, 두 노드 사이의 연결 여부를 확인하세요.

### 입력
- 관계 사전 `relations`: {"A": ["B", "C"], "B": ["D"], ...}
- 시작 노드 `start`
- 도착 노드 `end`

### 출력
- start에서 end까지 경로가 존재하면 True, 아니면 False

### 예제
```
입력:
relations = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
start = "A"
end = "D"
출력: True

입력:
relations = {"A": ["B"], "B": [], "C": ["D"], "D": []}
start = "A"
end = "D"
출력: False
```

### 힌트
- 해시맵을 인접 리스트 그래프로 해석하세요.""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "노드 수 <= 10^4",
        "hints": ["딕셔너리가 곧 인접 리스트", "BFS 또는 DFS로 경로 탐색", "방문 처리 필수"],
        "solution_template": "def solution(relations: dict, start: str, end: str) -> bool:\n    pass",
        "reference_solution": """def solution(relations: dict, start: str, end: str) -> bool:
    from collections import deque

    if start == end:
        return True

    if start not in relations:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()

        for neighbor in relations.get(node, []):
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hash-to-graph", "bfs"],
        "pattern_explanation": "해시맵(딕셔너리)은 그래프의 인접 리스트 표현과 동일합니다. 키는 노드, 값은 인접 노드 리스트입니다. 이 변환을 이해하면 해시맵 기반 데이터를 그래프 알고리즘(BFS/DFS)으로 처리할 수 있습니다.",
        "approach_hint": "딕셔너리 → 인접 리스트 그래프 → BFS 탐색",
        "time_complexity_hint": "O(V + E)",
        "space_complexity_hint": "O(V)",
        "test_cases": [
            {"input": '{"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}\nA\nD', "output": "True", "is_sample": True},
            {"input": '{"A": ["B"], "B": [], "C": ["D"], "D": []}\nA\nD', "output": "False", "is_sample": True},
            {"input": '{"A": []}\nA\nA', "output": "True", "is_sample": False},
        ]
    },
]


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    added = 0
    skipped = 0

    for problem in PROBLEMS:
        # 중복 확인
        cursor.execute("SELECT id FROM problems WHERE title = ?", (problem["title"],))
        if cursor.fetchone():
            print(f"  [건너뜀] 이미 존재: {problem['title']}")
            skipped += 1
            continue

        problem_id = uuid.uuid4().hex

        # 문제 삽입
        cursor.execute("""
            INSERT INTO problems (
                id, title, description, difficulty, category, constraints,
                hints, solution_template, reference_solution,
                time_limit_ms, memory_limit_mb, is_published,
                pattern_ids, pattern_explanation, approach_hint, time_complexity_hint, space_complexity_hint,
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
            1,  # is_published
            json.dumps(problem.get("pattern_ids", []), ensure_ascii=False),
            problem.get("pattern_explanation", ""),
            problem.get("approach_hint", ""),
            problem.get("time_complexity_hint", ""),
            problem.get("space_complexity_hint", ""),
            now,
            now
        ))

        # 테스트 케이스 삽입
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
    print(f"Phase 1 기초 자료구조 & 테크닉 시딩 완료")
    print(f"  - 추가: {added}개")
    print(f"  - 건너뜀: {skipped}개")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("Phase 1: 기초 자료구조 & 테크닉 문제 시딩 시작...")
    main()
