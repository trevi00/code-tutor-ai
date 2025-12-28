# Pattern 13: Unbounded Knapsack (무한 배낭 문제)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(n × W) |
| **공간복잡도** | O(W) |
| **선행 지식** | 0/1 배낭 문제 |

## 정의

**Unbounded Knapsack**은 각 아이템을 **무한히 선택**할 수 있는 배낭 문제입니다. 0/1 배낭과 달리 같은 아이템을 여러 번 사용할 수 있습니다.

## 핵심 아이디어

```
0/1 배낭:        역순 순회 (각 아이템 1번만)
무한 배낭:       정순 순회 (같은 아이템 여러 번)

예: 동전 [1, 2, 5], target = 5
정순 순회로 dp[w]를 계산하면 dp[w - coin]이 이미 갱신된 값을 사용
→ 같은 동전 중복 사용 가능
```

## 0/1 vs Unbounded 비교

| 특성 | 0/1 배낭 | 무한 배낭 |
|------|---------|----------|
| 아이템 사용 | 1번만 | 무한 |
| 순회 방향 | **역순** | **정순** |
| 대표 문제 | 부분집합 합 | 동전 교환 |

## Unbounded Knapsack 변형들

| 변형 | 설명 | 대표 문제 |
|------|------|----------|
| 최소 개수 | 합을 만드는 최소 아이템 수 | Coin Change |
| 경우의 수 | 합을 만드는 방법 수 | Coin Change II |
| 막대 자르기 | 최대 수익으로 자르기 | Rod Cutting |
| 리본 자르기 | 최대 조각 수로 자르기 | Ribbon Cut |
| 완전 제곱수 | 최소 제곱수 합 | Perfect Squares |

---

## 템플릿 코드

### 템플릿 1: 기본 무한 배낭

```python
def unbounded_knapsack(weights: list, values: list, capacity: int) -> int:
    """
    무한 배낭 - 최대 가치

    Time: O(n × W), Space: O(W)
    """
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # 정순 순회 (무한 배낭의 핵심!)
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

### 템플릿 2: 동전 교환 - 최소 개수

```python
def coin_change_min(coins: list, amount: int) -> int:
    """
    합이 amount가 되는 최소 동전 개수

    Time: O(n × amount), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

### 템플릿 3: 동전 교환 - 경우의 수

```python
def coin_change_ways(coins: list, amount: int) -> int:
    """
    합이 amount가 되는 조합의 수 (순서 무관)

    Time: O(n × amount), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    # 동전별로 순회 → 조합 (순서 무관)
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]

    return dp[amount]
```

### 템플릿 4: 동전 교환 - 순열의 수

```python
def coin_change_permutations(coins: list, amount: int) -> int:
    """
    합이 amount가 되는 순열의 수 (순서 고려)

    Time: O(n × amount), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1

    # 금액별로 순회 → 순열 (순서 고려)
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                dp[a] += dp[a - coin]

    return dp[amount]
```

### 템플릿 5: 막대 자르기

```python
def rod_cutting(prices: list, length: int) -> int:
    """
    막대를 잘라서 최대 수익

    prices[i] = 길이 i+1의 가격
    Time: O(n × length), Space: O(length)
    """
    dp = [0] * (length + 1)

    for i in range(1, length + 1):
        for cut in range(1, i + 1):
            if cut <= len(prices):
                dp[i] = max(dp[i], dp[i - cut] + prices[cut - 1])

    return dp[length]
```

### 템플릿 6: 완전 제곱수

```python
def num_squares(n: int) -> int:
    """
    n을 만드는 최소 완전제곱수 개수

    Time: O(n × √n), Space: O(n)
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1

    return dp[n]
```

### 템플릿 7: 리본 자르기 - 최대 조각

```python
def max_ribbon_cut(lengths: list, total: int) -> int:
    """
    리본을 주어진 길이로 자를 때 최대 조각 수

    Time: O(n × total), Space: O(total)
    """
    dp = [float('-inf')] * (total + 1)
    dp[0] = 0

    for length in lengths:
        for t in range(length, total + 1):
            dp[t] = max(dp[t], dp[t - length] + 1)

    return dp[total] if dp[total] > 0 else -1
```

### 템플릿 8: Word Break

```python
def word_break(s: str, wordDict: list) -> bool:
    """
    문자열을 사전 단어들로 분할 가능한가?

    Time: O(n² × m), Space: O(n)
    """
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[len(s)]
```

---

## 예제 문제

### 문제 1: 동전 교환 (Medium)

**문제 설명**
주어진 동전들로 amount를 만드는 최소 동전 개수를 구하세요.

**입력/출력 예시**
```
입력: coins = [1, 2, 5], amount = 11
출력: 3
설명: 5 + 5 + 1 = 11
```

**풀이**
```python
def coin_change(coins: list, amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

---

### 문제 2: 동전 교환 II (Medium)

**문제 설명**
주어진 동전들로 amount를 만드는 조합의 수를 구하세요.

**입력/출력 예시**
```
입력: amount = 5, coins = [1, 2, 5]
출력: 4
설명: 5, 2+2+1, 2+1+1+1, 1+1+1+1+1
```

**풀이**
```python
def change(amount: int, coins: list) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]

    return dp[amount]
```

---

### 문제 3: 완전 제곱수 (Medium)

**문제 설명**
n을 완전 제곱수들의 합으로 나타낼 때, 최소 개수를 구하세요.

**입력/출력 예시**
```
입력: n = 12
출력: 3
설명: 4 + 4 + 4 = 12
```

**풀이**
```python
def num_squares(n: int) -> int:
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1

    return dp[n]
```

---

### 문제 4: 조합 총합 IV (Medium)

**문제 설명**
배열의 숫자들을 사용하여 target을 만드는 순열의 수를 구하세요.

**입력/출력 예시**
```
입력: nums = [1, 2, 3], target = 4
출력: 7
설명: (1,1,1,1), (1,1,2), (1,2,1), (1,3), (2,1,1), (2,2), (3,1)
```

**풀이**
```python
def combination_sum4(nums: list, target: int) -> int:
    dp = [0] * (target + 1)
    dp[0] = 1

    # 순열: 금액별 → 숫자별 순회
    for t in range(1, target + 1):
        for num in nums:
            if num <= t:
                dp[t] += dp[t - num]

    return dp[target]
```

---

### 문제 5: 단어 분할 (Medium)

**문제 설명**
문자열 s를 사전의 단어들로 분할할 수 있는지 확인하세요.

**입력/출력 예시**
```
입력: s = "leetcode", wordDict = ["leet", "code"]
출력: true
설명: "leet" + "code"
```

**풀이**
```python
def word_break(s: str, wordDict: list) -> bool:
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[len(s)]
```

---

### 문제 6: 막대 자르기 (Medium)

**문제 설명**
길이 n인 막대를 자를 때 최대 수익을 구하세요.

**입력/출력 예시**
```
입력: prices = [2, 6, 7, 10, 13], n = 5
출력: 13
설명: 길이 2 + 길이 3 = 6 + 7 = 13
```

**풀이**
```python
def max_profit(prices: list, n: int) -> int:
    dp = [0] * (n + 1)

    for length in range(1, n + 1):
        for cut in range(1, min(length, len(prices)) + 1):
            dp[length] = max(dp[length], dp[length - cut] + prices[cut - 1])

    return dp[n]
```

---

## Editorial (풀이 전략)

### Step 1: 조합 vs 순열 구분

```python
# 조합 (순서 무관): 동전별 → 금액별
for coin in coins:
    for a in range(coin, amount + 1):
        dp[a] += dp[a - coin]

# 순열 (순서 고려): 금액별 → 동전별
for a in range(1, amount + 1):
    for coin in coins:
        if coin <= a:
            dp[a] += dp[a - coin]
```

### Step 2: 최솟값 vs 개수 초기화

```python
# 최솟값 문제: inf로 초기화
dp = [float('inf')] * (n + 1)
dp[0] = 0

# 개수 세기 문제: 0으로 초기화, dp[0] = 1
dp = [0] * (n + 1)
dp[0] = 1
```

### Step 3: 불가능 케이스 처리

```python
# 최솟값 문제
return dp[amount] if dp[amount] != float('inf') else -1

# 최댓값 문제
return dp[total] if dp[total] > float('-inf') else -1
```

---

## 자주 하는 실수

### 1. 순회 순서 혼동
```python
# ❌ 조합을 구하는데 순열 순회
for a in range(1, amount + 1):
    for coin in coins:
        dp[a] += dp[a - coin]  # 순열!

# ✅ 조합은 동전별 먼저
for coin in coins:
    for a in range(coin, amount + 1):
        dp[a] += dp[a - coin]  # 조합!
```

### 2. 0/1 배낭과 혼동
```python
# ❌ 무한 배낭인데 역순 순회
for w in range(capacity, weight - 1, -1):
    dp[w] = max(dp[w], dp[w - weight] + value)

# ✅ 무한 배낭은 정순 순회
for w in range(weight, capacity + 1):
    dp[w] = max(dp[w], dp[w - weight] + value)
```

### 3. 초기값 오류
```python
# ❌ 최소 개수인데 0으로 초기화
dp = [0] * (amount + 1)

# ✅ inf로 초기화 후 dp[0] = 0
dp = [float('inf')] * (amount + 1)
dp[0] = 0
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 322 | Coin Change | Medium |
| 518 | Coin Change II | Medium |
| 279 | Perfect Squares | Medium |
| 377 | Combination Sum IV | Medium |
| 139 | Word Break | Medium |
| 983 | Minimum Cost For Tickets | Medium |
| 1449 | Form Largest Integer With Digits | Hard |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 2293 | 동전 1 | Gold 5 | 경우의 수 (필수) |
| 2294 | 동전 2 | Gold 5 | 최소 개수 (필수) |
| 9084 | 동전 | Gold 5 | 동전 응용 |
| 1699 | 제곱수의 합 | Silver 2 | Perfect Squares |
| 11726 | 2×n 타일링 | Silver 3 | 기본 DP |
| 11727 | 2×n 타일링 2 | Silver 3 | 타일링 변형 |
| 9461 | 파도반 수열 | Silver 3 | 수열 DP |
| 15988 | 1, 2, 3 더하기 3 | Silver 2 | 조합 DP |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 거스름돈 | Level 3 | 동전 교환 |
| N으로 표현 | Level 3 | DP (카카오) |
| 멀리 뛰기 | Level 2 | 피보나치 변형 |

---

## 임베딩용 키워드

```
unbounded knapsack, 무한 배낭, coin change, 동전 교환,
combination sum, 조합 합, perfect squares, 완전 제곱수,
word break, 단어 분할, rod cutting, 막대 자르기,
forward iteration, 정순 순회, DP, 동적 프로그래밍,
동전 1, 동전 2, 타일링, BOJ 2293, 프로그래머스
```
