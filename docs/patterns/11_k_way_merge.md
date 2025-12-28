# Pattern 11: K-way Merge (K개 병합)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐ (중간) |
| **시간복잡도** | O(n log k) |
| **공간복잡도** | O(k) |
| **선행 지식** | 힙, 정렬된 데이터 |

## 정의

**K-way Merge**는 K개의 정렬된 리스트/배열을 **하나의 정렬된 결과**로 병합하는 기법입니다. Min Heap을 사용하여 효율적으로 처리합니다.

## 핵심 아이디어

```
List1: [1, 4, 7]
List2: [2, 5, 8]
List3: [3, 6, 9]

Heap: 각 리스트의 첫 요소 → [1, 2, 3]
Pop 1 → 결과: [1], Heap: [2, 3, 4]
Pop 2 → 결과: [1, 2], Heap: [3, 4, 5]
...
결과: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 템플릿 코드

### 템플릿 1: K개 정렬된 리스트 병합

```python
import heapq

def merge_k_sorted_lists(lists: list) -> list:
    """
    K개 정렬된 리스트 병합

    Time: O(n log k), Space: O(k)
    """
    result = []
    heap = []

    # 각 리스트의 첫 요소 추가
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 해당 리스트의 다음 요소 추가
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
```

### 템플릿 2: K개 정렬된 링크드리스트 병합

```python
import heapq

def merge_k_lists(lists: list) -> ListNode:
    """
    K개 정렬된 링크드리스트 병합

    Time: O(n log k), Space: O(k)
    """
    heap = []

    # 각 리스트의 헤드 추가
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next

        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

### 템플릿 3: 가장 작은 범위

```python
import heapq

def smallest_range(nums: list) -> list:
    """
    K개 리스트에서 모든 리스트의 요소를 포함하는 가장 작은 범위

    Time: O(n log k), Space: O(k)
    """
    heap = []
    max_val = float('-inf')

    # 각 리스트의 첫 요소
    for i, lst in enumerate(nums):
        heapq.heappush(heap, (lst[0], i, 0))
        max_val = max(max_val, lst[0])

    result = [float('-inf'), float('inf')]

    while len(heap) == len(nums):
        min_val, list_idx, elem_idx = heapq.heappop(heap)

        # 범위 업데이트
        if max_val - min_val < result[1] - result[0]:
            result = [min_val, max_val]

        # 다음 요소 추가
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            max_val = max(max_val, next_val)

    return result
```

---

## 예제 문제

### 문제 1: K개 정렬된 리스트 병합 (Hard)

**문제 설명**
K개의 정렬된 링크드리스트를 하나의 정렬된 링크드리스트로 병합하세요.

**입력/출력 예시**
```
입력: lists = [[1,4,5],[1,3,4],[2,6]]
출력: [1,1,2,3,4,4,5,6]
```

**풀이**
```python
def mergeKLists(lists: List[ListNode]) -> ListNode:
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

---

### 문제 2: 정렬된 행렬에서 K번째 수 (Medium)

**입력/출력 예시**
```
입력: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
출력: 13
```

**풀이**
```python
def kthSmallest(matrix: List[List[int]], k: int) -> int:
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)

    for _ in range(k):
        val, row, col = heapq.heappop(heap)
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col+1], row, col+1))

    return val
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 23 | Merge K Sorted Lists | Hard |
| 378 | Kth Smallest in Sorted Matrix | Medium |
| 632 | Smallest Range Covering K Lists | Hard |
| 373 | Find K Pairs with Smallest Sums | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 2751 | 수 정렬하기 2 | Silver 5 | 기본 정렬 |
| 10814 | 나이순 정렬 | Silver 5 | 안정 정렬 |
| 11650 | 좌표 정렬하기 | Silver 5 | 다중 조건 정렬 |
| 1764 | 듣보잡 | Silver 4 | 정렬 + 이분탐색 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 가장 큰 수 | Level 2 | 커스텀 정렬 |
| H-Index | Level 2 | 정렬 + 이분탐색 |
| 파일명 정렬 | Level 2 | 다중 조건 정렬 (카카오) |

---

## 임베딩용 키워드

```
k-way merge, K개 병합, merge k sorted, K개 정렬 병합,
min heap, 최소 힙, sorted lists, 정렬된 리스트,
smallest range, 가장 작은 범위,
수 정렬하기, 가장 큰 수, 파일명 정렬, BOJ, 프로그래머스
```
