# Pattern 06: In-place LinkedList Reversal (링크드리스트 뒤집기)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Medium |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(n) |
| **공간복잡도** | O(1) |
| **선행 지식** | 링크드리스트, 포인터 |

## 정의

**In-place LinkedList Reversal**은 추가 메모리 없이 링크드리스트의 노드 연결 방향을 뒤집는 기법입니다.

## 핵심 아이디어

```
원본:  1 → 2 → 3 → 4 → None
결과:  4 → 3 → 2 → 1 → None

핵심: 각 노드의 next 포인터를 이전 노드로 변경
```

### 시각화

```
Step 0: prev=None, curr=1
        None ← 1    2 → 3 → 4

Step 1: prev=1, curr=2
        None ← 1 ← 2    3 → 4

Step 2: prev=2, curr=3
        None ← 1 ← 2 ← 3    4

Step 3: prev=3, curr=4
        None ← 1 ← 2 ← 3 ← 4

return prev (= 4)
```

---

## 템플릿 코드

### 노드 정의

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### 템플릿 1: 전체 뒤집기

```python
def reverse_list(head: ListNode) -> ListNode:
    """
    링크드리스트 전체 뒤집기

    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head

    while curr:
        next_temp = curr.next  # 다음 노드 저장
        curr.next = prev       # 방향 뒤집기
        prev = curr            # prev 전진
        curr = next_temp       # curr 전진

    return prev  # 새 헤드
```

### 템플릿 2: 부분 뒤집기 (m ~ n)

```python
def reverse_between(head: ListNode, m: int, n: int) -> ListNode:
    """
    m번째부터 n번째까지만 뒤집기

    Time: O(n), Space: O(1)
    """
    if not head or m == n:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    # 1. m-1 위치로 이동
    for _ in range(m - 1):
        prev = prev.next

    # 2. 뒤집기 시작점
    curr = prev.next

    # 3. n-m번 뒤집기
    for _ in range(n - m):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

### 템플릿 3: K개씩 뒤집기

```python
def reverse_k_group(head: ListNode, k: int) -> ListNode:
    """
    K개씩 그룹으로 뒤집기

    Time: O(n), Space: O(1)
    """
    # K개 있는지 확인
    def has_k_nodes(node, k):
        count = 0
        while node and count < k:
            node = node.next
            count += 1
        return count == k

    # K개 뒤집기
    def reverse_k(head, k):
        prev, curr = None, head
        for _ in range(k):
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        return prev, head, curr  # 새헤드, 새꼬리, 다음시작

    dummy = ListNode(0)
    dummy.next = head
    prev_group_end = dummy

    while has_k_nodes(prev_group_end.next, k):
        new_head, new_tail, next_start = reverse_k(prev_group_end.next, k)
        prev_group_end.next = new_head
        new_tail.next = next_start
        prev_group_end = new_tail

    return dummy.next
```

### 템플릿 4: 번갈아 뒤집기

```python
def reverse_alternate_k(head: ListNode, k: int) -> ListNode:
    """
    K개 뒤집고, K개 건너뛰고, 반복

    Time: O(n), Space: O(1)
    """
    if not head or k <= 1:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    curr = head

    while curr:
        # K개 뒤집기
        tail = curr
        prev_node = None
        count = 0

        while curr and count < k:
            next_temp = curr.next
            curr.next = prev_node
            prev_node = curr
            curr = next_temp
            count += 1

        prev.next = prev_node
        tail.next = curr
        prev = tail

        # K개 건너뛰기
        count = 0
        while curr and count < k:
            prev = curr
            curr = curr.next
            count += 1

    return dummy.next
```

---

## 예제 문제

### 문제 1: 링크드리스트 뒤집기 (Easy)

**문제 설명**
링크드리스트를 뒤집어서 반환하세요.

**입력/출력 예시**
```
입력: [1,2,3,4,5]
출력: [5,4,3,2,1]
```

**풀이**
```python
def reverse_list(head: ListNode) -> ListNode:
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 2: 부분 뒤집기 (Medium)

**문제 설명**
m번째부터 n번째 노드까지만 뒤집으세요. (1-indexed)

**입력/출력 예시**
```
입력: [1,2,3,4,5], m = 2, n = 4
출력: [1,4,3,2,5]
```

**풀이**
```python
def reverse_between(head: ListNode, m: int, n: int) -> ListNode:
    dummy = ListNode(0, head)
    prev = dummy

    for _ in range(m - 1):
        prev = prev.next

    curr = prev.next
    for _ in range(n - m):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 3: K개씩 뒤집기 (Hard)

**문제 설명**
K개씩 그룹으로 뒤집으세요. 남은 노드가 K개 미만이면 그대로 둡니다.

**입력/출력 예시**
```
입력: [1,2,3,4,5], k = 2
출력: [2,1,4,3,5]

입력: [1,2,3,4,5], k = 3
출력: [3,2,1,4,5]
```

**풀이**
```python
def reverse_k_group(head: ListNode, k: int) -> ListNode:
    def count_nodes(node):
        count = 0
        while node:
            count += 1
            node = node.next
        return count

    total = count_nodes(head)
    dummy = ListNode(0, head)
    prev_tail = dummy

    while total >= k:
        # K개 뒤집기
        prev, curr = None, prev_tail.next
        for _ in range(k):
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp

        # 연결
        tail = prev_tail.next
        prev_tail.next = prev
        tail.next = curr
        prev_tail = tail
        total -= k

    return dummy.next
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 4: 리스트 재정렬 (Medium)

**문제 설명**
L0 → L1 → L2 → ... → Ln-1 → Ln 형태를
L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ... 형태로 재정렬하세요.

**입력/출력 예시**
```
입력: [1,2,3,4]
출력: [1,4,2,3]

입력: [1,2,3,4,5]
출력: [1,5,2,4,3]
```

**풀이**
```python
def reorder_list(head: ListNode) -> None:
    if not head or not head.next:
        return

    # 1. 중간 찾기
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # 2. 후반부 뒤집기
    second = slow.next
    slow.next = None
    prev = None
    while second:
        next_temp = second.next
        second.next = prev
        prev = second
        second = next_temp

    # 3. 병합
    first, second = head, prev
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2
```

**복잡도**: 시간 O(n), 공간 O(1)

---

## Editorial (풀이 전략)

### Step 1: Dummy Node 사용
```python
dummy = ListNode(0)
dummy.next = head
# ... 처리 후
return dummy.next
```
→ 헤드가 바뀌는 경우 유용

### Step 2: 세 포인터 패턴
```python
prev = None
curr = head
while curr:
    next_temp = curr.next  # 1. 다음 저장
    curr.next = prev       # 2. 뒤집기
    prev = curr            # 3. prev 전진
    curr = next_temp       # 4. curr 전진
```

### Step 3: 그룹 뒤집기
```
1. 그룹 존재 확인
2. 그룹 뒤집기
3. 이전 그룹과 연결
4. 다음 그룹으로 이동
```

---

## 자주 하는 실수

### 1. next 저장 안 함
```python
# ❌ curr.next가 사라짐
curr.next = prev
curr = curr.next  # None!

# ✅ 먼저 저장
next_temp = curr.next
curr.next = prev
curr = next_temp
```

### 2. 반환 값 실수
```python
# ❌ 원래 head 반환
return head  # 원래 head는 이제 꼬리!

# ✅ prev 반환
return prev  # 새 head
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 206 | Reverse Linked List | Easy |
| 92 | Reverse Linked List II | Medium |
| 25 | Reverse Nodes in k-Group | Hard |
| 143 | Reorder List | Medium |
| 234 | Palindrome Linked List | Easy |
| 24 | Swap Nodes in Pairs | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1406 | 에디터 | Silver 2 | 연결 리스트 (스택) |
| 5397 | 키로거 | Silver 2 | 연결 리스트 (스택) |
| 1158 | 요세푸스 문제 | Silver 4 | 원형 리스트 |
| 28279 | 덱 2 | Silver 4 | 양방향 리스트 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 표 편집 | Level 3 | 연결 리스트 (카카오) |
| 기능개발 | Level 2 | 큐 기반 |

---

## 임베딩용 키워드

```
reverse linked list, 링크드리스트 뒤집기, in-place reversal,
reverse between, 부분 뒤집기, reverse k group, K개씩 뒤집기,
reorder list, 재정렬, swap pairs, 쌍 교환, dummy node,
에디터, 키로거, 표 편집, BOJ, 프로그래머스
```
