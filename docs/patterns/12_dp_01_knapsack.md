# Pattern 12: 0/1 Knapsack (0/1 배낭 문제)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(n × W) |
| **공간복잡도** | O(W) 최적화 시 |
| **선행 지식** | 동적 프로그래밍 기초 |

## 정의

**0/1 Knapsack**은 각 아이템을 **선택하거나(1) 선택하지 않거나(0)** 둘 중 하나만 가능한 배낭 문제입니다. 부분 선택이 불가능합니다.

## 핵심 아이디어

```
아이템: [(무게=2, 가치=3), (무게=3, 가치=4), (무게=4, 가치=5)]
배낭 용량: 5

dp[i][w] = i번째 아이템까지 고려했을 때, 용량 w로 얻을 수 있는 최대 가치

선택지:
1. i번째 아이템 선택 안 함: dp[i-1][w]
2. i번째 아이템 선택: dp[i-1][w-weight[i]] + value[i]

dp[i][w] = max(선택 안 함, 선택)
```

## 0/1 Knapsack 변형들

| 변형 | 설명 | 대표 문제 |
|------|------|----------|
| 기본 배낭 | 최대 가치 | Knapsack |
| 부분집합 합 | 합이 target인 부분집합 존재? | Subset Sum |
| 동일 분할 | 두 부분집합의 합이 같은가? | Equal Partition |
| 개수 세기 | 합이 target인 경우의 수 | Count Subsets |
| 최소 차이 | 두 부분집합 합의 최소 차이 | Minimum Difference |

---

## 템플릿 코드

### 템플릿 1: 기본 0/1 배낭 (2D)

```python
def knapsack_01(weights: list, values: list, capacity: int) -> int:
    """
    0/1 배낭 문제 (2D DP)

    Time: O(n × W), Space: O(n × W)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 선택 안 함
            dp[i][w] = dp[i-1][w]

            # 선택 (무게가 충분하면)
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                              dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]
```

### 템플릿 2: 0/1 배낭 (1D 공간 최적화)

```python
def knapsack_01_optimized(weights: list, values: list, capacity: int) -> int:
    """
    0/1 배낭 문제 (1D DP)

    Time: O(n × W), Space: O(W)
    """
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # 역순으로 순회 (0/1 배낭의 핵심!)
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

### 템플릿 3: 부분집합 합 (Subset Sum)

```python
def subset_sum(nums: list, target: int) -> bool:
    """
    합이 target인 부분집합 존재 여부

    Time: O(n × target), Space: O(target)
    """
    dp = [False] * (target + 1)
    dp[0] = True  # 빈 집합의 합은 0

    for num in nums:
        # 역순으로 순회
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    return dp[target]
```

### 템플릿 4: 동일 분할 (Equal Partition)

```python
def can_partition(nums: list) -> bool:
    """
    배열을 합이 같은 두 부분집합으로 나눌 수 있는가?

    Time: O(n × sum/2), Space: O(sum/2)
    """
    total = sum(nums)

    # 홀수면 불가능
    if total % 2 != 0:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    return dp[target]
```

### 템플릿 5: 부분집합 합 경우의 수

```python
def count_subsets_with_sum(nums: list, target: int) -> int:
    """
    합이 target인 부분집합의 개수

    Time: O(n × target), Space: O(target)
    """
    dp = [0] * (target + 1)
    dp[0] = 1  # 빈 집합

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] += dp[t - num]

    return dp[target]
```

### 템플릿 6: 최소 부분집합 합 차이

```python
def min_subset_sum_diff(nums: list) -> int:
    """
    두 부분집합 합의 최소 차이

    Time: O(n × sum/2), Space: O(sum/2)
    """
    total = sum(nums)
    target = total // 2

    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    # 가능한 최대 S1 찾기 (S1 <= total/2)
    for s1 in range(target, -1, -1):
        if dp[s1]:
            s2 = total - s1
            return s2 - s1

    return total
```

### 템플릿 7: Target Sum (+/- 조합)

```python
def find_target_sum_ways(nums: list, target: int) -> int:
    """
    +/- 기호를 붙여 target을 만드는 경우의 수

    P - N = target
    P + N = total
    => P = (target + total) / 2

    Time: O(n × P), Space: O(P)
    """
    total = sum(nums)

    # 불가능한 경우
    if total < abs(target) or (total + target) % 2 != 0:
        return 0

    p = (total + target) // 2

    dp = [0] * (p + 1)
    dp[0] = 1

    for num in nums:
        for t in range(p, num - 1, -1):
            dp[t] += dp[t - num]

    return dp[p]
```

---

## 예제 문제

### 문제 1: 0/1 배낭 (Medium)

**문제 설명**
n개의 아이템(무게, 가치)과 배낭 용량이 주어질 때, 최대 가치를 구하세요.

**입력/출력 예시**
```
입력: weights = [1, 2, 3], values = [6, 10, 12], capacity = 5
출력: 22
설명: 아이템 1과 2 선택 (무게 1+2=3, 가치 6+10+12... 아이템 2,3 선택: 10+12=22)
```

**풀이**
```python
def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

---

### 문제 2: 분할 가능 여부 (Medium)

**문제 설명**
양의 정수 배열을 합이 같은 두 부분집합으로 나눌 수 있는지 확인하세요.

**입력/출력 예시**
```
입력: nums = [1, 5, 11, 5]
출력: true
설명: [1, 5, 5]와 [11], 둘 다 합이 11
```

**풀이**
```python
def can_partition(nums: list) -> bool:
    total = sum(nums)
    if total % 2:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    return dp[target]
```

---

### 문제 3: Target Sum (Medium)

**문제 설명**
배열의 각 숫자에 +/- 기호를 붙여 target을 만드는 경우의 수를 구하세요.

**입력/출력 예시**
```
입력: nums = [1, 1, 1, 1, 1], target = 3
출력: 5
설명: -1+1+1+1+1, +1-1+1+1+1, +1+1-1+1+1, +1+1+1-1+1, +1+1+1+1-1
```

**풀이**
```python
def find_target_sum_ways(nums: list, target: int) -> int:
    total = sum(nums)

    if total < abs(target) or (total + target) % 2:
        return 0

    p = (total + target) // 2
    dp = [0] * (p + 1)
    dp[0] = 1

    for num in nums:
        for t in range(p, num - 1, -1):
            dp[t] += dp[t - num]

    return dp[p]
```

---

### 문제 4: 최소 차이 부분집합 (Hard)

**문제 설명**
배열을 두 부분집합으로 나눌 때, 합의 차이가 최소가 되도록 하세요.

**입력/출력 예시**
```
입력: nums = [1, 6, 11, 5]
출력: 1
설명: [1, 5, 6]=12와 [11]=11, 차이=1
```

**풀이**
```python
def minimum_difference(nums: list) -> int:
    total = sum(nums)
    target = total // 2

    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    for s1 in range(target, -1, -1):
        if dp[s1]:
            return total - 2 * s1

    return total
```

---

### 문제 5: 부분집합 합 개수 (Medium)

**문제 설명**
합이 target인 부분집합의 개수를 구하세요.

**입력/출력 예시**
```
입력: nums = [1, 2, 3, 3], target = 6
출력: 3
설명: [1,2,3], [1,2,3], [3,3]
```

**풀이**
```python
def count_subsets(nums: list, target: int) -> int:
    dp = [0] * (target + 1)
    dp[0] = 1

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] += dp[t - num]

    return dp[target]
```

---

## Editorial (풀이 전략)

### Step 1: 0/1 배낭 패턴 인식

| 특징 | 0/1 배낭 |
|------|---------|
| 선택 | 선택 O / 선택 X |
| 반복 | 각 아이템 1번만 |
| 순회 방향 | **역순** (중요!) |

### Step 2: 역순 순회의 이유

```python
# ❌ 정순 순회 (같은 아이템 중복 사용)
for w in range(weight, capacity + 1):
    dp[w] = max(dp[w], dp[w - weight] + value)

# ✅ 역순 순회 (0/1 배낭)
for w in range(capacity, weight - 1, -1):
    dp[w] = max(dp[w], dp[w - weight] + value)
```

### Step 3: 문제 변환

```
부분집합 합 = 배낭 (무게=값, 가치=1)
동일 분할 = 부분집합 합 (target = total/2)
Target Sum = 부분집합 합 (target = (total+S)/2)
```

---

## 자주 하는 실수

### 1. 순회 방향 오류
```python
# ❌ 0/1 배낭인데 정순 순회
for w in range(weights[i], capacity + 1):
    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

# ✅ 역순 순회
for w in range(capacity, weights[i] - 1, -1):
    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

### 2. 초기값 설정 오류
```python
# ❌ 개수 세기인데 dp[0] = 0
dp = [0] * (target + 1)

# ✅ 빈 집합 = 1가지
dp[0] = 1
```

### 3. Edge Case 누락
```python
# ❌ total이 홀수인 경우 체크 안 함
def can_partition(nums):
    target = sum(nums) // 2
    ...

# ✅ 불가능 케이스 먼저 처리
if sum(nums) % 2 != 0:
    return False
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 416 | Partition Equal Subset Sum | Medium |
| 494 | Target Sum | Medium |
| 474 | Ones and Zeroes | Medium |
| 1049 | Last Stone Weight II | Medium |
| 879 | Profitable Schemes | Hard |
| 956 | Tallest Billboard | Hard |
| 2787 | Ways to Express Integer as Sum | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 12865 | 평범한 배낭 | Gold 5 | 기본 배낭 필수 |
| 1535 | 안녕 | Silver 2 | 배낭 응용 |
| 7579 | 앱 | Gold 3 | 배낭 변형 |
| 2629 | 양팔저울 | Gold 3 | 부분집합 합 |
| 1450 | 냅색문제 | Gold 1 | Meet in the Middle |
| 1943 | 동전 분배 | Gold 1 | 가능 여부 |
| 2098 | 외판원 순회 | Gold 1 | 비트마스크 DP |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 타겟 넘버 | Level 2 | DFS/DP |
| 정수 삼각형 | Level 3 | 경로 DP |
| 등굣길 | Level 3 | 그리드 DP |

---

## 임베딩용 키워드

```
0/1 knapsack, 배낭 문제, subset sum, 부분집합 합,
partition, 분할, target sum, 타겟 합,
dynamic programming, DP, 동적 프로그래밍,
reverse iteration, 역순 순회,
평범한 배낭, 앱, 양팔저울, BOJ 12865, 프로그래머스
```
