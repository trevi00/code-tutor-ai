# Pattern 14: Backtracking (백트래킹)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(k^n) 또는 O(n!) |
| **공간복잡도** | O(n) - 재귀 깊이 |
| **선행 지식** | 재귀, DFS |

## 정의

**Backtracking (백트래킹)**은 가능한 모든 경우를 탐색하되, 유망하지 않은 경로는 **가지치기(pruning)**하여 효율적으로 해를 찾는 기법입니다.

## 핵심 아이디어

```
모든 선택지 탐색 + 되돌리기

1. 선택 (Choose)
2. 탐색 (Explore)
3. 되돌리기 (Un-choose)

예: 순열 [1, 2, 3]
           []
    /      |      \
   [1]    [2]    [3]
  /  \   /  \   /  \
[1,2][1,3]... (계속)
```

## 백트래킹 vs DFS

| 특성 | DFS | 백트래킹 |
|------|-----|---------|
| 목적 | 모든 노드 방문 | 조건 만족하는 해 찾기 |
| 가지치기 | 없음 | **있음** (핵심!) |
| 상태 복원 | 선택적 | **필수** |

## 백트래킹 문제 유형

| 유형 | 설명 | 대표 문제 |
|------|------|----------|
| 순열 | 순서가 있는 모든 배열 | Permutations |
| 조합 | 순서 없는 선택 | Combinations |
| 부분집합 | 모든 부분집합 | Subsets |
| N-Queens | 조건부 배치 | N-Queens |
| 스도쿠 | 제약 만족 | Sudoku Solver |

---

## 템플릿 코드

### 템플릿 1: 순열 (Permutations)

```python
def permutations(nums: list) -> list:
    """
    모든 순열 생성

    Time: O(n! × n), Space: O(n)
    """
    result = []

    def backtrack(path, remaining):
        # 기저 조건
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            # 선택
            path.append(remaining[i])

            # 탐색
            backtrack(path, remaining[:i] + remaining[i+1:])

            # 되돌리기
            path.pop()

    backtrack([], nums)
    return result
```

### 템플릿 2: 조합 (Combinations)

```python
def combinations(n: int, k: int) -> list:
    """
    n개 중 k개 선택하는 모든 조합

    Time: O(C(n,k) × k), Space: O(k)
    """
    result = []

    def backtrack(start, path):
        # 기저 조건
        if len(path) == k:
            result.append(path[:])
            return

        # 가지치기: 남은 숫자가 부족하면 중단
        remaining = n - start + 1
        needed = k - len(path)
        if remaining < needed:
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)  # i+1부터 (중복 방지)
            path.pop()

    backtrack(1, [])
    return result
```

### 템플릿 3: 부분집합 (Subsets)

```python
def subsets(nums: list) -> list:
    """
    모든 부분집합 생성

    Time: O(2^n × n), Space: O(n)
    """
    result = []

    def backtrack(start, path):
        # 모든 상태가 유효한 부분집합
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### 템플릿 4: 중복 있는 순열

```python
def permute_unique(nums: list) -> list:
    """
    중복 숫자가 있을 때 유일한 순열들

    Time: O(n! × n), Space: O(n)
    """
    result = []
    nums.sort()  # 정렬 필수!
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # 이미 사용됨
            if used[i]:
                continue

            # 중복 건너뛰기
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result
```

### 템플릿 5: 중복 있는 부분집합

```python
def subsets_with_dup(nums: list) -> list:
    """
    중복 숫자가 있을 때 유일한 부분집합들

    Time: O(2^n × n), Space: O(n)
    """
    result = []
    nums.sort()  # 정렬 필수!

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            # 같은 레벨에서 중복 건너뛰기
            if i > start and nums[i] == nums[i-1]:
                continue

            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### 템플릿 6: 조합 합 (Combination Sum)

```python
def combination_sum(candidates: list, target: int) -> list:
    """
    합이 target이 되는 모든 조합 (중복 사용 가능)

    Time: O(n^(target/min)), Space: O(target/min)
    """
    result = []

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            path.append(candidates[i])
            # 같은 숫자 재사용 가능: i (i+1 아님)
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result
```

### 템플릿 7: N-Queens

```python
def solve_n_queens(n: int) -> list:
    """
    N-Queens 문제 모든 해

    Time: O(n!), Space: O(n)
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]

    # 공격 가능 위치 추적
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            # 가지치기: 공격 가능 위치면 건너뛰기
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # 선택
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            # 탐색
            backtrack(row + 1)

            # 되돌리기
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

### 템플릿 8: 스도쿠 솔버

```python
def solve_sudoku(board: list) -> None:
    """
    스도쿠 풀이

    Time: O(9^(빈칸)), Space: O(81)
    """
    def is_valid(row, col, num):
        # 행 검사
        if num in board[row]:
            return False

        # 열 검사
        for r in range(9):
            if board[r][col] == num:
                return False

        # 3x3 박스 검사
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num

                            if backtrack():
                                return True

                            board[row][col] = '.'

                    return False  # 가지치기

        return True  # 모든 칸 채움

    backtrack()
```

### 템플릿 9: 괄호 생성

```python
def generate_parenthesis(n: int) -> list:
    """
    n쌍의 유효한 괄호 조합

    Time: O(4^n / √n), Space: O(n)
    """
    result = []

    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return

        # 여는 괄호 추가
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()

        # 닫는 괄호 추가 (열린 괄호보다 적을 때만)
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result
```

### 템플릿 10: 단어 검색

```python
def word_search(board: list, word: str) -> bool:
    """
    2D 보드에서 단어 존재 여부

    Time: O(m × n × 4^L), Space: O(L)
    """
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True

        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False

        # 방문 표시
        temp = board[r][c]
        board[r][c] = '#'

        # 4방향 탐색
        found = (backtrack(r+1, c, idx+1) or
                 backtrack(r-1, c, idx+1) or
                 backtrack(r, c+1, idx+1) or
                 backtrack(r, c-1, idx+1))

        # 되돌리기
        board[r][c] = temp

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True

    return False
```

---

## 예제 문제

### 문제 1: 순열 (Medium)

**입력/출력 예시**
```
입력: nums = [1, 2, 3]
출력: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**풀이**
```python
def permute(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result
```

---

### 문제 2: 조합 (Medium)

**입력/출력 예시**
```
입력: n = 4, k = 2
출력: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

**풀이**
```python
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result
```

---

### 문제 3: 부분집합 (Medium)

**입력/출력 예시**
```
입력: nums = [1, 2, 3]
출력: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**풀이**
```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

---

### 문제 4: N-Queens (Hard)

**입력/출력 예시**
```
입력: n = 4
출력: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

**풀이**
```python
def solveNQueens(n):
    result = []
    cols, diag1, diag2 = set(), set(), set()
    board = [['.' for _ in range(n)] for _ in range(n)]

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue

            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

---

### 문제 5: 단어 검색 (Medium)

**입력/출력 예시**
```
입력: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
      word = "ABCCED"
출력: true
```

**풀이**
```python
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[idx]:
            return False

        temp = board[r][c]
        board[r][c] = '#'

        found = (backtrack(r+1, c, idx+1) or backtrack(r-1, c, idx+1) or
                 backtrack(r, c+1, idx+1) or backtrack(r, c-1, idx+1))

        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
```

---

## Editorial (풀이 전략)

### Step 1: 백트래킹 3단계

```python
def backtrack(state):
    if 종료조건:
        결과저장
        return

    for choice in choices:
        # 1. 선택
        state.add(choice)

        # 2. 탐색
        backtrack(new_state)

        # 3. 되돌리기
        state.remove(choice)
```

### Step 2: 가지치기 (Pruning)

```python
# 조합: 시작 인덱스로 중복 방지
for i in range(start, n):  # start부터

# 중복 요소: 정렬 + 건너뛰기
if i > start and nums[i] == nums[i-1]:
    continue

# N-Queens: 공격 가능 위치 체크
if col in cols or (row-col) in diag1:
    continue
```

### Step 3: 문제별 패턴

| 문제 | start 인덱스 | 중복 사용 |
|------|-------------|----------|
| 순열 | 없음 (remaining) | X |
| 조합 | i + 1 | X |
| 부분집합 | i + 1 | X |
| 조합 합 | i (같은 숫자 재사용) | O |

---

## 자주 하는 실수

### 1. 되돌리기 누락
```python
# ❌ pop 안 함
path.append(num)
backtrack(path)
# path.pop()  누락!

# ✅ 반드시 되돌리기
path.append(num)
backtrack(path)
path.pop()
```

### 2. 복사 안 함
```python
# ❌ 참조 추가 (나중에 변경됨)
result.append(path)

# ✅ 복사본 추가
result.append(path[:])
# 또는 result.append(list(path))
```

### 3. 중복 처리 누락
```python
# ❌ 중복 숫자 있는데 처리 안 함
for i in range(len(nums)):
    path.append(nums[i])
    backtrack(path)
    path.pop()

# ✅ 정렬 + 건너뛰기
nums.sort()
for i in range(len(nums)):
    if i > 0 and nums[i] == nums[i-1]:
        continue
    ...
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 46 | Permutations | Medium |
| 47 | Permutations II | Medium |
| 77 | Combinations | Medium |
| 78 | Subsets | Medium |
| 90 | Subsets II | Medium |
| 39 | Combination Sum | Medium |
| 40 | Combination Sum II | Medium |
| 22 | Generate Parentheses | Medium |
| 79 | Word Search | Medium |
| 51 | N-Queens | Hard |
| 37 | Sudoku Solver | Hard |
| 131 | Palindrome Partitioning | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 15649 | N과 M (1) | Silver 3 | 순열 기본 필수 |
| 15650 | N과 M (2) | Silver 3 | 조합 기본 필수 |
| 15651 | N과 M (3) | Silver 3 | 중복 순열 |
| 15652 | N과 M (4) | Silver 3 | 중복 조합 |
| 9663 | N-Queen | Gold 4 | 필수 |
| 2580 | 스도쿠 | Gold 4 | 필수 |
| 14889 | 스타트와 링크 | Silver 1 | 조합 (삼성) |
| 15686 | 치킨 배달 | Gold 5 | 조합 (삼성) |
| 1182 | 부분수열의 합 | Silver 2 | 부분집합 |
| 6603 | 로또 | Silver 2 | 조합 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 모음사전 | Level 2 | 순열 |
| 피로도 | Level 2 | 순열 |
| N-Queen | Level 2 | N-Queen |
| 양과 늑대 | Level 3 | 상태 백트래킹 (카카오) |
| 사라지는 발판 | Level 3 | 게임 이론 + 백트래킹 (카카오) |

---

## 임베딩용 키워드

```
backtracking, 백트래킹, permutation, 순열,
combination, 조합, subset, 부분집합,
pruning, 가지치기, N-Queens, sudoku, 스도쿠,
word search, 단어 검색, generate parentheses, 괄호 생성,
N과 M, 스타트와 링크, 치킨 배달, BOJ, 삼성, 프로그래머스
```
