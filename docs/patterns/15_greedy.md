# Pattern 15: Greedy (탐욕 알고리즘)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | 보통 O(n) 또는 O(n log n) |
| **공간복잡도** | O(1) ~ O(n) |
| **선행 지식** | 정렬, 기본 자료구조 |

## 정의

**Greedy (탐욕 알고리즘)**는 매 순간 **지역적 최적 선택**을 반복하여 전역 최적해를 구하는 기법입니다. 항상 최적해를 보장하지는 않지만, 특정 문제에서는 최적해를 보장합니다.

## 핵심 아이디어

```
매 단계에서 가장 좋아 보이는 선택을 함
→ 이전 선택을 번복하지 않음
→ 빠르지만 항상 최적은 아님

예: 거스름돈 문제 (500, 100, 50, 10원)
800원 거슬러 주기:
1. 500원 1개 (남은 300원)
2. 100원 3개 (남은 0원)
→ 총 4개 (최적!)
```

## Greedy vs DP

| 특성 | Greedy | DP |
|------|--------|-----|
| 선택 | 지역 최적 | 모든 경우 |
| 시간 | 빠름 | 느림 |
| 최적해 | **특정 조건**에서만 | 항상 |
| 구현 | 간단 | 복잡 |

## Greedy가 적용되는 조건

1. **탐욕 선택 속성**: 지역 최적 선택이 전역 최적으로 이어짐
2. **최적 부분 구조**: 부분 문제의 최적해가 전체 최적해에 포함

---

## 템플릿 코드

### 템플릿 1: 회의실 배정 (Activity Selection)

```python
def max_meetings(meetings: list) -> int:
    """
    최대 회의 수 (종료 시간 기준 정렬)

    Time: O(n log n), Space: O(1)
    """
    # 종료 시간 기준 정렬
    meetings.sort(key=lambda x: x[1])

    count = 0
    last_end = 0

    for start, end in meetings:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

### 템플릿 2: 점프 게임 (Jump Game)

```python
def can_jump(nums: list) -> bool:
    """
    마지막 인덱스에 도달 가능한가?

    Time: O(n), Space: O(1)
    """
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)

    return True
```

### 템플릿 3: 점프 게임 II (최소 점프 수)

```python
def min_jumps(nums: list) -> int:
    """
    마지막 인덱스까지 최소 점프 수

    Time: O(n), Space: O(1)
    """
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

    return jumps
```

### 템플릿 4: 가스 스테이션

```python
def can_complete_circuit(gas: list, cost: list) -> int:
    """
    순환 가능한 시작 인덱스 찾기

    Time: O(n), Space: O(1)
    """
    total_tank = 0
    curr_tank = 0
    start = 0

    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]

        # 현재 시작점에서 불가능
        if curr_tank < 0:
            start = i + 1
            curr_tank = 0

    return start if total_tank >= 0 else -1
```

### 템플릿 5: 배 태우기 (Two Sum Pairing)

```python
def num_rescue_boats(people: list, limit: int) -> int:
    """
    최소 보트 수 (각 보트 최대 2명, 무게 제한)

    Time: O(n log n), Space: O(1)
    """
    people.sort()
    left, right = 0, len(people) - 1
    boats = 0

    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1

    return boats
```

### 템플릿 6: 풍선 터뜨리기

```python
def find_min_arrow_shots(points: list) -> int:
    """
    모든 풍선을 터뜨리는 최소 화살 수

    Time: O(n log n), Space: O(1)
    """
    if not points:
        return 0

    # 끝점 기준 정렬
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        # 현재 화살로 터뜨릴 수 없음
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows
```

### 템플릿 7: 분배 문제 (Assign Cookies)

```python
def find_content_children(greed: list, cookies: list) -> int:
    """
    만족시킬 수 있는 최대 아이 수

    Time: O(n log n + m log m), Space: O(1)
    """
    greed.sort()
    cookies.sort()

    child = 0
    cookie = 0

    while child < len(greed) and cookie < len(cookies):
        if cookies[cookie] >= greed[child]:
            child += 1
        cookie += 1

    return child
```

### 템플릿 8: 주식 매매 II (무제한 거래)

```python
def max_profit(prices: list) -> int:
    """
    무제한 거래로 최대 이익

    Time: O(n), Space: O(1)
    """
    profit = 0

    for i in range(1, len(prices)):
        # 상승하는 모든 구간 더하기
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]

    return profit
```

### 템플릿 9: 작업 스케줄링 (Deadline & Profit)

```python
def job_scheduling(jobs: list) -> int:
    """
    마감일과 이익이 주어진 작업 스케줄링
    jobs = [(deadline, profit), ...]

    Time: O(n² × n), Space: O(n)
    """
    # 이익 기준 내림차순 정렬
    jobs.sort(key=lambda x: x[1], reverse=True)

    max_deadline = max(j[0] for j in jobs)
    slots = [False] * (max_deadline + 1)
    total_profit = 0

    for deadline, profit in jobs:
        # 가장 늦은 가능한 슬롯 찾기
        for slot in range(deadline, 0, -1):
            if not slots[slot]:
                slots[slot] = True
                total_profit += profit
                break

    return total_profit
```

### 템플릿 10: 문자열 분할 (Partition Labels)

```python
def partition_labels(s: str) -> list:
    """
    각 문자가 최대 한 부분에만 나타나도록 분할

    Time: O(n), Space: O(1)
    """
    # 각 문자의 마지막 위치
    last = {c: i for i, c in enumerate(s)}

    result = []
    start = 0
    end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])

        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result
```

---

## 예제 문제

### 문제 1: 점프 게임 (Medium)

**문제 설명**
배열의 각 원소는 해당 위치에서 점프할 수 있는 최대 거리입니다. 마지막 인덱스에 도달할 수 있는지 확인하세요.

**입력/출력 예시**
```
입력: nums = [2,3,1,1,4]
출력: true
설명: 0→1→4 또는 0→2→3→4
```

**풀이**
```python
def canJump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True
```

---

### 문제 2: 점프 게임 II (Medium)

**문제 설명**
마지막 인덱스에 도달하는 최소 점프 수를 구하세요.

**입력/출력 예시**
```
입력: nums = [2,3,1,1,4]
출력: 2
설명: 0→1→4 (2번 점프)
```

**풀이**
```python
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest

    return jumps
```

---

### 문제 3: 주식 매매 II (Medium)

**문제 설명**
주식 가격 배열이 주어질 때, 무제한 매매로 최대 이익을 구하세요.

**입력/출력 예시**
```
입력: prices = [7,1,5,3,6,4]
출력: 7
설명: 1에 사서 5에 팔고(+4), 3에 사서 6에 팔기(+3) = 7
```

**풀이**
```python
def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit
```

---

### 문제 4: 가스 스테이션 (Medium)

**문제 설명**
N개의 가스 스테이션이 원형으로 있습니다. 시계 방향으로 한 바퀴 돌 수 있는 시작 인덱스를 찾으세요.

**입력/출력 예시**
```
입력: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
출력: 3
설명: 3번 스테이션에서 시작하면 한 바퀴 가능
```

**풀이**
```python
def canCompleteCircuit(gas, cost):
    total_tank = 0
    curr_tank = 0
    start = 0

    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]

        if curr_tank < 0:
            start = i + 1
            curr_tank = 0

    return start if total_tank >= 0 else -1
```

---

### 문제 5: 회의실 배정 (Medium)

**문제 설명**
회의 시작/종료 시간이 주어질 때, 최대한 많은 회의를 배정하세요.

**입력/출력 예시**
```
입력: meetings = [[1,4],[2,3],[3,5],[7,9]]
출력: 3
설명: [2,3], [3,5], [7,9] 선택
```

**풀이**
```python
def maxMeetings(meetings):
    meetings.sort(key=lambda x: x[1])
    count = 0
    last_end = 0

    for start, end in meetings:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

---

### 문제 6: 문자열 분할 (Medium)

**문제 설명**
문자열을 가능한 많은 부분으로 분할하되, 각 문자가 최대 한 부분에만 나타나도록 하세요.

**입력/출력 예시**
```
입력: s = "ababcbacadefegdehijhklij"
출력: [9, 7, 8]
설명: "ababcbaca", "defegde", "hijhklij"
```

**풀이**
```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result
```

---

## Editorial (풀이 전략)

### Step 1: Greedy 적용 가능 판단

```
✅ Greedy 적용 가능:
- 정렬 후 선택이 명확함
- 지역 최적이 전역 최적으로 이어짐
- 선택을 번복할 필요 없음

❌ Greedy 불가 (DP 필요):
- 동전 교환 (일부 동전 조합)
- 0/1 배낭
- 최장 공통 부분수열
```

### Step 2: 정렬 기준 선택

| 문제 | 정렬 기준 |
|------|----------|
| 회의실 배정 | 종료 시간 |
| 풍선 터뜨리기 | 끝점 |
| 작업 스케줄링 | 이익 (내림차순) |
| 배 태우기 | 무게 |

### Step 3: 일반적인 패턴

```python
# 패턴 1: 정렬 후 순차 선택
items.sort(key=...)
for item in items:
    if 조건:
        선택

# 패턴 2: 양 끝에서 접근
items.sort()
left, right = 0, len(items) - 1
while left <= right:
    ...

# 패턴 3: 최대 도달 범위 추적
max_reach = 0
for i, val in enumerate(arr):
    if i > max_reach:
        return False
    max_reach = max(max_reach, i + val)
```

---

## 자주 하는 실수

### 1. Greedy 적용 불가 문제에 적용
```python
# ❌ 동전 교환에 Greedy (반례 존재)
# 동전: [1, 3, 4], 금액: 6
# Greedy: 4+1+1 = 3개
# 최적: 3+3 = 2개

# ✅ DP 사용
dp = [float('inf')] * (amount + 1)
```

### 2. 정렬 기준 오류
```python
# ❌ 회의실 배정에 시작 시간 정렬
meetings.sort(key=lambda x: x[0])

# ✅ 종료 시간 정렬
meetings.sort(key=lambda x: x[1])
```

### 3. Edge Case 누락
```python
# ❌ 빈 배열 체크 안 함
def findMin(arr):
    return min(arr)  # 빈 배열이면 에러

# ✅ 체크 후 처리
if not arr:
    return 0
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 55 | Jump Game | Medium |
| 45 | Jump Game II | Medium |
| 122 | Best Time to Buy and Sell Stock II | Medium |
| 134 | Gas Station | Medium |
| 435 | Non-overlapping Intervals | Medium |
| 452 | Minimum Number of Arrows | Medium |
| 763 | Partition Labels | Medium |
| 881 | Boats to Save People | Medium |
| 455 | Assign Cookies | Easy |
| 621 | Task Scheduler | Medium |
| 1029 | Two City Scheduling | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1931 | 회의실 배정 | Silver 1 | 필수 |
| 11047 | 동전 0 | Silver 4 | 기본 |
| 1541 | 잃어버린 괄호 | Silver 2 | 수식 그리디 |
| 13305 | 주유소 | Silver 3 | 그리디 |
| 1744 | 수 묶기 | Gold 4 | 정렬 + 그리디 |
| 2212 | 센서 | Gold 5 | 정렬 + 그리디 |
| 1202 | 보석 도둑 | Gold 2 | 우선순위 큐 + 그리디 |
| 2839 | 설탕 배달 | Bronze 1 | 기본 |
| 11000 | 강의실 배정 | Gold 5 | 우선순위 큐 |
| 1339 | 단어 수학 | Gold 4 | 그리디 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 체육복 | Level 1 | 기본 |
| 조이스틱 | Level 2 | 그리디 |
| 큰 수 만들기 | Level 2 | 스택 + 그리디 |
| 구명보트 | Level 2 | 투 포인터 + 그리디 |
| 단속카메라 | Level 3 | 구간 그리디 |
| 섬 연결하기 | Level 3 | MST (크루스칼) |

---

## 임베딩용 키워드

```
greedy, 탐욕 알고리즘, activity selection, 회의실 배정,
jump game, 점프 게임, gas station, 가스 스테이션,
interval scheduling, 구간 스케줄링, partition labels,
local optimal, 지역 최적, stock trading, 주식 매매,
동전 0, 잃어버린 괄호, 큰 수 만들기, BOJ 1931, 프로그래머스
```
