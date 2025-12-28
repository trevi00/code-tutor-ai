# Pattern 10: Top K Elements (상위 K개 요소)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(n log k) |
| **공간복잡도** | O(k) |
| **선행 지식** | 힙 (Heap), 우선순위 큐 |

## 정의

**Top K Elements**는 배열에서 **K개의 가장 큰/작은** 요소를 찾거나, **K번째 큰/작은** 요소를 찾는 기법입니다.

## 핵심 아이디어

### 정렬 vs 힙

| 방법 | 시간복잡도 | 공간복잡도 | 언제 사용? |
|------|-----------|-----------|-----------|
| 정렬 | O(n log n) | O(n) | K가 n에 가까울 때 |
| 힙 | O(n log k) | O(k) | K가 작을 때 ✅ |
| Quick Select | O(n) 평균 | O(1) | K번째만 필요할 때 |

### Min Heap vs Max Heap

```
Top K 최댓값 찾기: Min Heap 사용 (크기 K 유지)
- 가장 작은 것을 버림 → 큰 것 K개 남음

Top K 최솟값 찾기: Max Heap 사용 (크기 K 유지)
- 가장 큰 것을 버림 → 작은 것 K개 남음
```

---

## 템플릿 코드

### 템플릿 1: K번째 큰 수

```python
import heapq

def find_kth_largest(nums: list, k: int) -> int:
    """
    K번째로 큰 수 찾기

    Time: O(n log k), Space: O(k)
    """
    # Min Heap으로 크기 K 유지
    heap = []

    for num in nums:
        heapq.heappush(heap, num)

        if len(heap) > k:
            heapq.heappop(heap)  # 가장 작은 것 제거

    return heap[0]  # K번째 큰 수
```

### 템플릿 2: Top K 빈도수

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list, k: int) -> list:
    """
    가장 빈번한 K개 숫자

    Time: O(n log k), Space: O(n)
    """
    count = Counter(nums)

    # Min Heap (빈도수 기준)
    heap = []

    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))

        if len(heap) > k:
            heapq.heappop(heap)

    return [item[1] for item in heap]
```

### 템플릿 3: 가장 가까운 K개 점

```python
import heapq

def k_closest(points: list, k: int) -> list:
    """
    원점에서 가장 가까운 K개 점

    Time: O(n log k), Space: O(k)
    """
    # Max Heap (거리의 음수로)
    heap = []

    for x, y in points:
        dist = -(x*x + y*y)  # 음수로 Max Heap 효과
        heapq.heappush(heap, (dist, [x, y]))

        if len(heap) > k:
            heapq.heappop(heap)

    return [item[1] for item in heap]
```

### 템플릿 4: 정렬된 배열들에서 K번째 작은 수

```python
import heapq

def kth_smallest_in_matrix(matrix: list, k: int) -> int:
    """
    정렬된 2D 행렬에서 K번째 작은 수

    Time: O(k log n), Space: O(n)
    """
    n = len(matrix)

    # (값, 행, 열)로 Min Heap
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)

    for _ in range(k - 1):
        val, row, col = heapq.heappop(heap)

        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))

    return heapq.heappop(heap)[0]
```

### 템플릿 5: Quick Select (O(n) 평균)

```python
import random

def quick_select(nums: list, k: int) -> int:
    """
    K번째 큰 수 (Quick Select)

    Time: O(n) 평균, O(n²) 최악
    Space: O(1)
    """
    k = len(nums) - k  # K번째 큰 = (n-k)번째 작은

    def partition(left, right):
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]

        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[i], nums[store_idx] = nums[store_idx], nums[i]
                store_idx += 1

        nums[store_idx], nums[right] = nums[right], nums[store_idx]
        return store_idx

    left, right = 0, len(nums) - 1

    while True:
        pivot_idx = partition(left, right)

        if pivot_idx == k:
            return nums[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
```

---

## 예제 문제

### 문제 1: 배열에서 K번째 큰 수 (Medium)

**문제 설명**
정렬되지 않은 배열에서 K번째로 큰 수를 찾으세요.

**입력/출력 예시**
```
입력: nums = [3,2,1,5,6,4], k = 2
출력: 5
```

**풀이**
```python
def find_kth_largest(nums: list, k: int) -> int:
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

---

### 문제 2: Top K 빈도 요소 (Medium)

**문제 설명**
배열에서 가장 빈번하게 나타나는 K개의 요소를 반환하세요.

**입력/출력 예시**
```
입력: nums = [1,1,1,2,2,3], k = 2
출력: [1, 2]
```

**풀이**
```python
def top_k_frequent(nums: list, k: int) -> list:
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

### 문제 3: K개의 가장 가까운 점 (Medium)

**문제 설명**
원점 (0, 0)에서 가장 가까운 K개의 점을 반환하세요.

**입력/출력 예시**
```
입력: points = [[1,3],[-2,2]], k = 1
출력: [[-2,2]]
설명: 거리 √10 vs √8, [-2,2]가 더 가까움
```

**풀이**
```python
def k_closest(points: list, k: int) -> list:
    heap = []
    for x, y in points:
        dist = -(x*x + y*y)
        if len(heap) < k:
            heapq.heappush(heap, (dist, [x, y]))
        elif dist > heap[0][0]:
            heapq.heapreplace(heap, (dist, [x, y]))
    return [p[1] for p in heap]
```

---

### 문제 4: 정렬된 행렬에서 K번째 수 (Medium)

**문제 설명**
각 행과 열이 정렬된 n×n 행렬에서 K번째로 작은 수를 찾으세요.

**입력/출력 예시**
```
입력: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
출력: 13
```

**풀이**
```python
def kth_smallest(matrix: list, k: int) -> int:
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

### 문제 5: K개 리스트 병합 (Hard)

**문제 설명**
K개의 정렬된 링크드리스트를 하나로 병합하세요.

**풀이**
```python
def merge_k_lists(lists: list) -> ListNode:
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

## Editorial (풀이 전략)

### Step 1: 문제 유형 파악

| 유형 | 힙 종류 | 크기 |
|------|--------|-----|
| K개 최댓값 | Min Heap | K |
| K개 최솟값 | Max Heap (음수) | K |
| K번째 큰 값 | Min Heap | K |
| K번째 작은 값 | Max Heap | K |

### Step 2: Python heapq 사용법

```python
import heapq

# Min Heap (기본)
heapq.heappush(heap, item)
heapq.heappop(heap)

# Max Heap (음수 트릭)
heapq.heappush(heap, -item)
-heapq.heappop(heap)

# nlargest/nsmallest
heapq.nlargest(k, items)
heapq.nsmallest(k, items)
```

### Step 3: 튜플 비교

```python
# (우선순위, 값) 형태로 저장
heapq.heappush(heap, (priority, value))

# 우선순위가 같으면 두 번째 값으로 비교
# 비교 불가능한 객체면 인덱스 추가
heapq.heappush(heap, (priority, index, object))
```

---

## 자주 하는 실수

### 1. 힙 방향 혼동
```python
# ❌ K개 최댓값인데 Max Heap
heap = []
for num in nums:
    heapq.heappush(heap, -num)  # 틀림!
    if len(heap) > k:
        heapq.heappop(heap)

# ✅ Min Heap으로 작은 것 제거
for num in nums:
    heapq.heappush(heap, num)
    if len(heap) > k:
        heapq.heappop(heap)
```

### 2. 빈 힙에서 pop
```python
# ❌ 힙이 비어있으면 에러
result = heapq.heappop(heap)

# ✅ 체크 후 pop
if heap:
    result = heapq.heappop(heap)
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 215 | Kth Largest Element | Medium |
| 347 | Top K Frequent Elements | Medium |
| 973 | K Closest Points to Origin | Medium |
| 378 | Kth Smallest in Sorted Matrix | Medium |
| 23 | Merge K Sorted Lists | Hard |
| 703 | Kth Largest Element in a Stream | Easy |
| 692 | Top K Frequent Words | Medium |
| 767 | Reorganize String | Medium |
| 295 | Find Median from Data Stream | Hard |
| 373 | Find K Pairs with Smallest Sums | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 11279 | 최대 힙 | Silver 2 | 기본 |
| 1927 | 최소 힙 | Silver 2 | 기본 |
| 11286 | 절댓값 힙 | Silver 1 | 커스텀 힙 |
| 1655 | 가운데를 말해요 | Gold 2 | 두 개의 힙 |
| 7662 | 이중 우선순위 큐 | Gold 4 | 양방향 힙 |
| 1715 | 카드 정렬하기 | Gold 4 | 그리디 + 힙 |
| 2075 | N번째 큰 수 | Gold 5 | Top K |
| 13975 | 파일 합치기 3 | Gold 4 | 허프만 코딩 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 더 맵게 | Level 2 | 기본 힙 |
| 디스크 컨트롤러 | Level 3 | 우선순위 큐 |
| 이중우선순위큐 | Level 3 | 양방향 힙 |
| 베스트앨범 | Level 3 | 정렬 + 그룹화 |

---

## 임베딩용 키워드

```
top k elements, 상위 K개, kth largest, K번째 큰,
heap, 힙, priority queue, 우선순위 큐, min heap, max heap,
k frequent, K개 빈도, k closest, K개 가까운,
quick select, 퀵 셀렉트, nlargest, nsmallest,
가운데를 말해요, 더 맵게, 디스크 컨트롤러, BOJ, 프로그래머스
```
