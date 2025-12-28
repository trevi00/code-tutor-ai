# Pattern 03: Fast & Slow Pointers (빠른/느린 포인터)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Medium |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(n) |
| **공간복잡도** | O(1) |
| **선행 지식** | 링크드리스트, 포인터 |

## 정의

**Fast & Slow Pointers** (또는 Floyd's Tortoise and Hare Algorithm)는 두 개의 포인터가 **서로 다른 속도**로 이동하는 기법입니다. 주로 **사이클 감지**와 **중간 지점 찾기**에 사용됩니다.

## 핵심 아이디어

```
토끼와 거북이 경주:
- 거북이(slow): 한 번에 1칸 이동
- 토끼(fast): 한 번에 2칸 이동

사이클이 있으면:
→ 토끼가 거북이를 따라잡음 (만남)

사이클이 없으면:
→ 토끼가 먼저 끝에 도달
```

### 시각화: 사이클 감지

```
1 → 2 → 3 → 4 → 5
              ↓
        7 ← 6 ←

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3
Step 3: slow=3, fast=5
Step 4: slow=4, fast=7
Step 5: slow=5, fast=4
Step 6: slow=6, fast=6 ← 만남! (사이클 존재)
```

### 시각화: 중간 노드 찾기

```
1 → 2 → 3 → 4 → 5 → None

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3
Step 3: slow=3, fast=5
Step 4: fast.next = None → 종료

slow = 3 (중간 노드!)
```

## 언제 사용하는가?

### ✅ 사용해야 하는 경우
- 링크드리스트 **사이클 감지**
- 사이클 **시작점** 찾기
- 링크드리스트 **중간 노드** 찾기
- 링크드리스트가 **팰린드롬**인지 확인
- 배열에서 **중복 숫자** 찾기 (특수 조건)
- **행복한 숫자** 문제

### ❌ 사용하지 말아야 하는 경우
- 배열에서 일반적인 검색 (→ Binary Search)
- 두 요소의 합 찾기 (→ Two Pointers)
- 연속 구간 문제 (→ Sliding Window)

---

## 템플릿 코드

### 링크드리스트 노드 정의

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### 템플릿 1: 사이클 감지

```python
def has_cycle(head: ListNode) -> bool:
    """
    링크드리스트에 사이클이 있는지 확인

    Args:
        head: 리스트의 헤드 노드

    Returns:
        사이클 있으면 True

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next         # 1칸 이동
        fast = fast.next.next    # 2칸 이동

        if slow == fast:         # 만남 = 사이클
            return True

    return False  # fast가 끝에 도달 = 사이클 없음
```

### 템플릿 2: 사이클 시작점 찾기

```python
def detect_cycle_start(head: ListNode) -> ListNode:
    """
    사이클 시작 노드 찾기

    수학적 증명:
    - slow가 이동한 거리: D + K (D: 시작→사이클, K: 사이클 내 이동)
    - fast가 이동한 거리: 2(D + K)
    - fast는 사이클을 N바퀴 더 돌았음: 2(D + K) = D + K + N*C (C: 사이클 길이)
    - 정리하면: D + K = N*C → D = N*C - K
    - 따라서 시작점에서 D만큼, 만남점에서 D만큼 이동하면 사이클 시작점에서 만남

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None

    slow = fast = head

    # 1단계: 만나는 지점 찾기
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 사이클 없음

    # 2단계: 시작점에서 다시 출발
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next  # 이제 둘 다 1칸씩

    return slow  # 사이클 시작점
```

### 템플릿 3: 중간 노드 찾기

```python
def find_middle(head: ListNode) -> ListNode:
    """
    링크드리스트의 중간 노드 찾기
    (짝수 개면 두 번째 중간)

    Args:
        head: 리스트의 헤드 노드

    Returns:
        중간 노드

    Time: O(n), Space: O(1)
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # slow가 중간에 위치

# 첫 번째 중간을 원하면 (짝수 개일 때)
def find_middle_first(head: ListNode) -> ListNode:
    slow = fast = head

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

### 템플릿 4: 행복한 숫자 (Happy Number)

```python
def is_happy(n: int) -> bool:
    """
    행복한 숫자인지 확인
    (각 자릿수의 제곱 합을 반복하면 1이 되는 숫자)

    예: 19 → 1² + 9² = 82 → 8² + 2² = 68 → ... → 1 (행복!)
        2 → 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 (사이클!)

    Time: O(log n), Space: O(1)
    """
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit ** 2
            num //= 10
        return total

    slow = n
    fast = get_next(n)

    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))

    return fast == 1
```

---

## 예제 문제

### 문제 1: 링크드리스트 사이클 감지 (Easy)

**문제 설명**
링크드리스트가 주어집니다. 리스트에 사이클이 있으면 `true`, 없으면 `false`를 반환하세요.

**입력/출력 예시**
```
입력: head = [3,2,0,-4], pos = 1 (노드 1이 사이클 시작)
출력: true

입력: head = [1,2], pos = -1 (사이클 없음)
출력: false
```

**풀이**
```python
def has_cycle(head: ListNode) -> bool:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 2: 사이클 시작점 찾기 (Medium)

**문제 설명**
링크드리스트에 사이클이 있으면 사이클 시작 노드를 반환하세요. 없으면 `null`을 반환하세요.

**풀이**
```python
def detect_cycle(head: ListNode) -> ListNode:
    if not head or not head.next:
        return None

    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # 시작점 찾기
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

    return None
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 3: 중간 노드 (Easy)

**문제 설명**
링크드리스트의 중간 노드를 반환하세요. 노드가 두 개면 두 번째 중간 노드를 반환하세요.

**입력/출력 예시**
```
입력: [1,2,3,4,5]
출력: 노드 3

입력: [1,2,3,4,5,6]
출력: 노드 4 (두 번째 중간)
```

**풀이**
```python
def middle_node(head: ListNode) -> ListNode:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 4: 팰린드롬 링크드리스트 (Easy)

**문제 설명**
링크드리스트가 팰린드롬인지 확인하세요.

**입력/출력 예시**
```
입력: [1,2,2,1]
출력: true

입력: [1,2]
출력: false
```

**풀이**
```python
def is_palindrome(head: ListNode) -> bool:
    if not head or not head.next:
        return True

    # 1단계: 중간 찾기
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # 2단계: 후반부 뒤집기
    second_half = reverse_list(slow.next)

    # 3단계: 비교
    first_half = head
    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True

def reverse_list(head: ListNode) -> ListNode:
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 5: 행복한 숫자 (Easy)

**문제 설명**
숫자 `n`이 주어집니다. 각 자릿수의 제곱 합을 반복했을 때 1이 되면 "행복한 숫자"입니다.

**입력/출력 예시**
```
입력: n = 19
출력: true
설명: 1² + 9² = 82 → 8² + 2² = 68 → 6² + 8² = 100 → 1² + 0² + 0² = 1

입력: n = 2
출력: false (무한 사이클)
```

**풀이**
```python
def is_happy(n: int) -> bool:
    def sum_of_squares(num):
        total = 0
        while num:
            digit = num % 10
            total += digit ** 2
            num //= 10
        return total

    slow = n
    fast = sum_of_squares(n)

    while fast != 1 and slow != fast:
        slow = sum_of_squares(slow)
        fast = sum_of_squares(sum_of_squares(fast))

    return fast == 1
```

**복잡도**: 시간 O(log n), 공간 O(1)

---

### 문제 6: 중복 숫자 찾기 (Medium)

**문제 설명**
`n+1`개의 정수를 담은 배열이 있습니다. 각 정수는 `1`부터 `n` 사이입니다. 중복된 숫자를 찾으세요. 배열을 수정하지 않고, O(1) 공간으로 해결하세요.

**입력/출력 예시**
```
입력: nums = [1,3,4,2,2]
출력: 2

입력: nums = [3,1,3,4,2]
출력: 3
```

**풀이 (Floyd's Algorithm)**
```python
def find_duplicate(nums: list) -> int:
    # 배열을 링크드리스트처럼 해석
    # nums[i] = 다음 인덱스
    # 중복이 있으면 사이클 발생

    slow = fast = nums[0]

    # 1단계: 사이클 내 만남점 찾기
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # 2단계: 사이클 시작점 = 중복 숫자
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
```

**복잡도**: 시간 O(n), 공간 O(1)

---

## Editorial (풀이 전략)

### Step 1: 사이클 여부 판단
- 링크드리스트/수열이 **끝이 없이 반복**될 가능성?
- 숫자 연산이 **같은 값으로 돌아올** 가능성?

### Step 2: 포인터 속도 설정
```python
slow = slow.next       # 1칸
fast = fast.next.next  # 2칸
```

### Step 3: 종료 조건

| 상황 | 조건 |
|------|------|
| 사이클 있음 | slow == fast |
| 사이클 없음 | fast == None 또는 fast.next == None |

### Step 4: 사이클 시작점 찾기 (필요 시)
```python
# 만남 후, slow를 처음으로
slow = head
while slow != fast:
    slow = slow.next
    fast = fast.next  # 둘 다 1칸씩
```

---

## 자주 하는 실수

### 1. fast.next 체크 누락
```python
# ❌ NoneType 에러 발생
while fast:
    fast = fast.next.next  # fast.next가 None이면 에러!

# ✅ 올바른 코드
while fast and fast.next:
    fast = fast.next.next
```

### 2. 사이클 시작점 찾기에서 속도 실수
```python
# ❌ 잘못된 코드
while slow != fast:
    slow = slow.next
    fast = fast.next.next  # 여전히 2칸! (틀림)

# ✅ 올바른 코드
while slow != fast:
    slow = slow.next
    fast = fast.next  # 둘 다 1칸씩
```

### 3. 초기 위치 실수
```python
# ❌ slow와 fast가 처음부터 같으면 바로 사이클 감지
slow = fast = head
if slow == fast:  # 항상 True!
    return True

# ✅ 루프 내에서 비교
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:  # 이동 후 비교
        return True
```

---

## 관련 패턴

| 패턴 | 관계 |
|------|------|
| **Two Pointers** | 같은 속도로 다른 시작점 vs 다른 속도로 같은 시작점 |
| **In-place LinkedList Reversal** | 팰린드롬 검사 시 함께 사용 |

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 141 | Linked List Cycle | Easy |
| 142 | Linked List Cycle II | Medium |
| 876 | Middle of the Linked List | Easy |
| 234 | Palindrome Linked List | Easy |
| 202 | Happy Number | Easy |
| 287 | Find the Duplicate Number | Medium |
| 457 | Circular Array Loop | Medium |
| 143 | Reorder List | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 17103 | 골드바흐 파티션 | Silver 2 | 투 포인터 응용 |
| 1484 | 다이어트 | Gold 4 | 투 포인터 |
| 2018 | 수들의 합 5 | Silver 5 | 연속 수열 |
| 1253 | 좋다 | Gold 4 | 투 포인터 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 연속된 부분 수열의 합 | Level 2 | 투 포인터 기본 |

---

## 임베딩용 키워드

```
fast slow pointers, 빠른 느린 포인터, Floyd's algorithm, 플로이드 알고리즘,
tortoise hare, 거북이 토끼, 사이클 감지, cycle detection,
링크드리스트 중간, middle node, 팰린드롬, palindrome linked list,
행복한 숫자, happy number, 중복 숫자, duplicate number,
BOJ, 백준, 프로그래머스
```
