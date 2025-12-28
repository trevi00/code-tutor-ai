# Pattern 21: Bitmask DP (비트마스킹 DP)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Hard |
| **빈출도** | ⭐⭐⭐⭐ (삼성 SW 역량) |
| **시간복잡도** | O(2^n × n) ~ O(2^n × n²) |
| **공간복잡도** | O(2^n) |
| **선행 지식** | DP, 비트 연산 |

## 정의

**비트마스킹 DP**는 부분집합의 상태를 **비트마스크**로 표현하여 DP를 수행하는 기법입니다. 방문 여부, 선택 여부 등을 비트로 관리합니다.

## 핵심 아이디어

```
n = 4인 경우, 각 원소의 선택 여부를 4비트로 표현

0000 (0)  = {} 공집합
0001 (1)  = {0}
0010 (2)  = {1}
0011 (3)  = {0, 1}
...
1111 (15) = {0, 1, 2, 3} 전체집합

dp[mask] = mask 상태에서의 최적값
```

## 비트 연산 기초

| 연산 | 코드 | 설명 |
|------|------|------|
| i번째 비트 확인 | `mask & (1 << i)` | i가 집합에 있는지 |
| i번째 비트 켜기 | `mask | (1 << i)` | i를 집합에 추가 |
| i번째 비트 끄기 | `mask & ~(1 << i)` | i를 집합에서 제거 |
| 비트 토글 | `mask ^ (1 << i)` | i 상태 반전 |
| 켜진 비트 수 | `bin(mask).count('1')` | 집합 크기 |

---

## 템플릿 코드

### 템플릿 1: 외판원 순회 (TSP)

```python
def tsp(dist: list) -> int:
    """
    외판원 순회 문제 (Traveling Salesman Problem)
    모든 도시를 한 번씩 방문하고 시작점으로 돌아오는 최소 비용

    Time: O(2^n × n²)
    Space: O(2^n × n)
    """
    n = len(dist)
    INF = float('inf')

    # dp[mask][i] = mask 상태로 i번 도시에 있을 때 최소 비용
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 0번 도시에서 시작

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue

            for next_city in range(n):
                # 이미 방문했으면 스킵
                if mask & (1 << next_city):
                    continue

                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]
                dp[new_mask][next_city] = min(dp[new_mask][next_city], new_cost)

    # 모든 도시 방문 후 0번으로 복귀
    full_mask = (1 << n) - 1
    result = INF

    for last in range(1, n):
        if dp[full_mask][last] < INF and dist[last][0] < INF:
            result = min(result, dp[full_mask][last] + dist[last][0])

    return result if result < INF else -1
```

### 템플릿 2: 부분집합 DP (최적 분할)

```python
def min_partition_cost(arr: list, cost_func) -> int:
    """
    배열을 부분집합들로 분할할 때 최소 비용
    cost_func(subset_mask) = 해당 부분집합의 비용

    Time: O(3^n)
    """
    n = len(arr)
    full_mask = (1 << n) - 1

    # 각 부분집합의 비용 미리 계산
    subset_cost = [0] * (1 << n)
    for mask in range(1 << n):
        subset_cost[mask] = cost_func(mask, arr)

    # dp[mask] = mask를 분할하는 최소 비용
    dp = [float('inf')] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        # mask의 모든 부분집합 순회
        subset = mask
        while subset > 0:
            remain = mask ^ subset
            dp[mask] = min(dp[mask], dp[remain] + subset_cost[subset])
            subset = (subset - 1) & mask

    return dp[full_mask]
```

### 템플릿 3: 비트 순회 (부분집합 열거)

```python
def enumerate_subsets(mask: int):
    """
    mask의 모든 부분집합 열거

    Time: O(2^k), k = mask의 비트 수
    """
    subset = mask
    while subset > 0:
        print(bin(subset))
        subset = (subset - 1) & mask

    # 공집합 포함 시
    subset = mask
    while True:
        print(bin(subset))
        if subset == 0:
            break
        subset = (subset - 1) & mask


def enumerate_supersets(mask: int, n: int):
    """
    mask를 포함하는 모든 상위집합 열거 (0 ~ 2^n-1 범위)
    """
    superset = mask
    full = (1 << n) - 1

    while superset <= full:
        print(bin(superset))
        superset = (superset + 1) | mask
```

### 템플릿 4: 할당 문제 (Assignment Problem)

```python
def min_assignment_cost(cost: list) -> int:
    """
    n명의 사람을 n개의 작업에 배정하는 최소 비용
    cost[i][j] = i번 사람이 j번 작업을 할 때 비용

    Time: O(2^n × n)
    """
    n = len(cost)
    INF = float('inf')

    # dp[mask] = mask 작업들이 할당된 상태에서 최소 비용
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        # 현재까지 할당된 사람 수 = 할당된 작업 수
        person = bin(mask).count('1')

        for job in range(n):
            if mask & (1 << job):
                continue

            new_mask = mask | (1 << job)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[person][job])

    return dp[(1 << n) - 1]
```

### 템플릿 5: 해밀턴 경로

```python
def hamiltonian_path_count(adj: list) -> int:
    """
    모든 정점을 정확히 한 번 방문하는 경로 수

    Time: O(2^n × n²)
    """
    n = len(adj)

    # dp[mask][i] = mask 방문, i에서 끝나는 경로 수
    dp = [[0] * n for _ in range(1 << n)]

    # 시작점 초기화
    for i in range(n):
        dp[1 << i][i] = 1

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == 0:
                continue

            for next_v in range(n):
                if mask & (1 << next_v):
                    continue
                if not adj[last][next_v]:
                    continue

                new_mask = mask | (1 << next_v)
                dp[new_mask][next_v] += dp[mask][last]

    full_mask = (1 << n) - 1
    return sum(dp[full_mask])
```

### 템플릿 6: SOS DP (Sum over Subsets)

```python
def sos_dp(arr: list) -> list:
    """
    Sum over Subsets DP
    result[mask] = sum(arr[subset]) for all subset of mask

    Time: O(2^n × n)
    """
    n = len(arr).bit_length() - 1  # arr 크기가 2^n 가정
    dp = arr[:]

    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]

    return dp
```

---

## 예제 문제

### 문제 1: 외판원 순회 (BOJ 2098) - Gold 1

**문제 설명**
N개의 도시를 모두 방문하고 시작 도시로 돌아오는 최소 비용을 구하세요.

**입력/출력 예시**
```
입력: dist = [
    [0, 2, 5, 7],
    [2, 0, 8, 3],
    [5, 8, 0, 1],
    [7, 3, 1, 0]
]
출력: 12 (0 → 1 → 3 → 2 → 0: 2+3+1+5=11 또는 다른 경로)
```

**풀이**
```python
def solution(dist):
    return tsp(dist)
```

---

### 문제 2: 스도미노쿠 (BOJ 4574) - Gold 2

**문제 설명**
스도쿠와 도미노가 합쳐진 퍼즐. 비트마스킹으로 사용 가능한 숫자/도미노 추적.

---

### 문제 3: 달이 차오른다, 가자. (BOJ 1194) - Gold 1

**문제 설명**
미로에서 열쇠를 모아 문을 열면서 탈출. 가진 열쇠 상태를 비트마스크로.

**풀이 (BFS + 비트마스크)**
```python
from collections import deque

def solve(maze: list) -> int:
    rows, cols = len(maze), len(maze[0])

    # 시작점 찾기
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == '0':
                start = (r, c)
                break

    # BFS: (r, c, 열쇠상태)
    visited = set()
    queue = deque([(start[0], start[1], 0, 0)])  # r, c, keys, dist
    visited.add((start[0], start[1], 0))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c, keys, dist = queue.popleft()

        if maze[r][c] == '1':
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                cell = maze[nr][nc]
                new_keys = keys

                if cell == '#':
                    continue

                # 열쇠 획득 (a-f)
                if 'a' <= cell <= 'f':
                    new_keys = keys | (1 << (ord(cell) - ord('a')))

                # 문 확인 (A-F)
                if 'A' <= cell <= 'F':
                    door = ord(cell) - ord('A')
                    if not (keys & (1 << door)):
                        continue

                if (nr, nc, new_keys) not in visited:
                    visited.add((nr, nc, new_keys))
                    queue.append((nr, nc, new_keys, dist + 1))

    return -1
```

---

### 문제 4: 발전소 (BOJ 1102) - Gold 1

**문제 설명**
N개의 발전소 중 일부가 고장났습니다. 최소 비용으로 P개 이상 작동시키세요.

**풀이**
```python
def power_plant(n: int, cost: list, status: str, p: int) -> int:
    INF = float('inf')

    # 초기 상태
    init_mask = 0
    for i, s in enumerate(status):
        if s == 'Y':
            init_mask |= (1 << i)

    init_count = bin(init_mask).count('1')

    if init_count >= p:
        return 0

    if init_count == 0:
        return -1

    # dp[mask] = mask 상태까지 가는 최소 비용
    dp = [INF] * (1 << n)
    dp[init_mask] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue

        if bin(mask).count('1') >= p:
            continue

        # 작동 중인 발전소에서 고장난 발전소 수리
        for on in range(n):
            if not (mask & (1 << on)):
                continue

            for off in range(n):
                if mask & (1 << off):
                    continue

                new_mask = mask | (1 << off)
                dp[new_mask] = min(dp[new_mask], dp[mask] + cost[on][off])

    result = min(dp[mask] for mask in range(1 << n)
                 if bin(mask).count('1') >= p)

    return result if result < INF else -1
```

---

### 문제 5: 공주님을 구해라! (비트마스크 BFS)

**풀이 패턴**
```python
# 상태: (위치, 아이템 보유 상태)
# visited[r][c][item_mask] = 방문 여부
visited = [[[False] * (1 << k) for _ in range(cols)] for _ in range(rows)]
```

---

## Editorial (풀이 전략)

### Step 1: 비트마스크 사용 판단

| 키워드 | 비트마스크 적용 |
|--------|---------------|
| n ≤ 20 | ✅ (2^20 ≈ 10^6) |
| 부분집합 | ✅ |
| 방문 상태 | ✅ |
| 선택/미선택 | ✅ |
| 순열 문제 | ✅ (TSP 등) |

### Step 2: 상태 정의

```python
# TSP
dp[mask][last] = mask 방문, last에서 끝날 때 최소 비용

# 할당 문제
dp[mask] = mask 작업이 완료된 상태의 최소 비용

# 그리드 + 아이템
dp[r][c][mask] = (r,c)에서 mask 아이템을 가진 상태
```

### Step 3: 비트 연산 패턴

```python
# i번째 원소 포함 여부
if mask & (1 << i):
    # 포함됨

# i번째 원소 추가
new_mask = mask | (1 << i)

# 부분집합 순회
subset = mask
while subset > 0:
    # process subset
    subset = (subset - 1) & mask
```

---

## 자주 하는 실수

### 1. 비트 연산 우선순위
```python
# ❌ & 보다 == 가 우선순위 높음
if mask & (1 << i) == 0:  # 틀림!

# ✅ 괄호 필수
if (mask & (1 << i)) == 0:
# 또는
if not (mask & (1 << i)):
```

### 2. 인덱스 범위
```python
# ❌ n이 20 이상이면 int 범위 초과 가능
mask = 1 << 32  # 문제 없음 (Python)

# ✅ 다른 언어에서는 long 사용
```

### 3. 초기 상태 설정
```python
# ❌ 시작점 초기화 누락
dp = [[INF] * n for _ in range(1 << n)]
# dp[1][0] = 0 빠짐!

# ✅ 시작점 초기화
dp[1][0] = 0  # 0번에서 시작
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 847 | Shortest Path Visiting All Nodes | Hard |
| LeetCode | 943 | Find the Shortest Superstring | Hard |
| LeetCode | 1879 | Minimum XOR Sum | Hard |
| BOJ | 2098 | 외판원 순회 | Gold 1 |
| BOJ | 1102 | 발전소 | Gold 1 |
| BOJ | 1194 | 달이 차오른다, 가자. | Gold 1 |
| BOJ | 17182 | 우주 탐사선 | Gold 2 |
| BOJ | 18119 | 단어 암기 | Gold 4 |
| BOJ | 11723 | 집합 | Silver 5 |

---

## 임베딩용 키워드

```
bitmask dp, 비트마스킹 DP, traveling salesman, 외판원 순회, TSP,
subset, 부분집합, bit manipulation, 비트 연산, hamiltonian path,
assignment problem, 할당 문제, SOS DP, sum over subsets,
삼성 SW 역량, state compression, 상태 압축
```
