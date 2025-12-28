# Pattern 20: Segment Tree (세그먼트 트리)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Hard |
| **빈출도** | ⭐⭐⭐ (삼성 고급) |
| **시간복잡도** | O(log n) per query/update |
| **공간복잡도** | O(n) |
| **선행 지식** | 트리, 재귀, 구간 연산 |

## 정의

**세그먼트 트리(Segment Tree)**는 배열의 **구간 쿼리**(합, 최소, 최대 등)와 **점 업데이트**를 O(log n)에 처리하는 자료구조입니다.

## 핵심 아이디어

```
배열: [1, 3, 5, 7, 9, 11]

세그먼트 트리:
           36 [0-5]
          /        \
     9 [0-2]      27 [3-5]
     /    \       /    \
  4[0-1] 5[2] 16[3-4] 11[5]
  /   \       /   \
1[0] 3[1]   7[3] 9[4]

구간 합 쿼리 [1, 4]:
arr[1] + arr[2] + arr[3] + arr[4] = 3 + 5 + 7 + 9 = 24

O(log n)에 처리!
```

## Segment Tree vs 누적합 vs Fenwick Tree

| 연산 | 누적합 | Fenwick | Segment Tree |
|------|-------|---------|--------------|
| 구간 쿼리 | O(1) | O(log n) | O(log n) |
| 점 업데이트 | O(n) ❌ | O(log n) | O(log n) |
| 구간 업데이트 | O(n) | O(log n) | O(log n) (Lazy) |
| 구현 복잡도 | 쉬움 | 중간 | 어려움 |

---

## 템플릿 코드

### 템플릿 1: 구간 합 세그먼트 트리

```python
class SumSegmentTree:
    """
    구간 합 세그먼트 트리

    Time: O(log n) per operation
    Space: O(n)
    """
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node + 1, start, mid)
            self._build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx: int, val: int):
        """점 업데이트: arr[idx] = val"""
        self._update(0, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node + 1, start, mid, idx, val)
            else:
                self._update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, left: int, right: int) -> int:
        """구간 합 쿼리: sum(arr[left:right+1])"""
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right) -> int:
        if right < start or end < left:
            return 0  # 범위 밖
        if left <= start and end <= right:
            return self.tree[node]  # 완전히 포함

        mid = (start + end) // 2
        left_sum = self._query(2 * node + 1, start, mid, left, right)
        right_sum = self._query(2 * node + 2, mid + 1, end, left, right)
        return left_sum + right_sum
```

### 템플릿 2: 구간 최솟값 세그먼트 트리

```python
class MinSegmentTree:
    """
    구간 최솟값 세그먼트 트리
    """
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node + 1, start, mid)
            self._build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def update(self, idx: int, val: int):
        self._update(0, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node + 1, start, mid, idx, val)
            else:
                self._update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def query(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right) -> int:
        if right < start or end < left:
            return float('inf')
        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return min(
            self._query(2 * node + 1, start, mid, left, right),
            self._query(2 * node + 2, mid + 1, end, left, right)
        )
```

### 템플릿 3: Lazy Propagation (구간 업데이트)

```python
class LazySegmentTree:
    """
    Lazy Propagation 세그먼트 트리
    구간 업데이트 O(log n)
    """
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2 * node + 1, start, mid)
            self._build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def _push_down(self, node, start, end):
        """Lazy 값을 자식에게 전파"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2

            # 현재 노드 업데이트
            self.tree[node] += self.lazy[node] * (end - start + 1)

            # 자식에게 전파
            if start != end:
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]

            self.lazy[node] = 0

    def range_update(self, left: int, right: int, val: int):
        """구간 업데이트: arr[left:right+1] += val"""
        self._range_update(0, 0, self.n - 1, left, right, val)

    def _range_update(self, node, start, end, left, right, val):
        self._push_down(node, start, end)

        if right < start or end < left:
            return

        if left <= start and end <= right:
            self.lazy[node] += val
            self._push_down(node, start, end)
            return

        mid = (start + end) // 2
        self._range_update(2 * node + 1, start, mid, left, right, val)
        self._range_update(2 * node + 2, mid + 1, end, left, right, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right) -> int:
        self._push_down(node, start, end)

        if right < start or end < left:
            return 0

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return (self._query(2 * node + 1, start, mid, left, right) +
                self._query(2 * node + 2, mid + 1, end, left, right))
```

### 템플릿 4: 펜윅 트리 (Binary Indexed Tree)

```python
class FenwickTree:
    """
    펜윅 트리 (Binary Indexed Tree, BIT)
    구간 합 + 점 업데이트
    구현이 세그먼트 트리보다 간단

    Time: O(log n) per operation
    Space: O(n)
    """
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    def update(self, i: int, delta: int):
        """점 업데이트: arr[i] += delta"""
        i += 1  # 1-indexed로 변환
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 다음 노드

    def prefix_sum(self, i: int) -> int:
        """접두사 합: sum(arr[0:i+1])"""
        i += 1
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & (-i)  # 부모 노드
        return result

    def range_sum(self, left: int, right: int) -> int:
        """구간 합: sum(arr[left:right+1])"""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    @classmethod
    def from_array(cls, arr: list) -> 'FenwickTree':
        """배열로부터 펜윅 트리 생성"""
        tree = cls(len(arr))
        for i, val in enumerate(arr):
            tree.update(i, val)
        return tree
```

### 템플릿 5: 2D 펜윅 트리

```python
class FenwickTree2D:
    """
    2D 펜윅 트리
    2D 구간 합 쿼리
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, r: int, c: int, delta: int):
        r += 1
        while r <= self.rows:
            c_idx = c + 1
            while c_idx <= self.cols:
                self.tree[r][c_idx] += delta
                c_idx += c_idx & (-c_idx)
            r += r & (-r)

    def prefix_sum(self, r: int, c: int) -> int:
        """(0,0)부터 (r,c)까지의 합"""
        r += 1
        result = 0
        while r > 0:
            c_idx = c + 1
            while c_idx > 0:
                result += self.tree[r][c_idx]
                c_idx -= c_idx & (-c_idx)
            r -= r & (-r)
        return result

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """(r1,c1)부터 (r2,c2)까지의 합"""
        return (self.prefix_sum(r2, c2)
                - self.prefix_sum(r1 - 1, c2)
                - self.prefix_sum(r2, c1 - 1)
                + self.prefix_sum(r1 - 1, c1 - 1))
```

---

## 예제 문제

### 문제 1: 구간 합 구하기 (BOJ 2042) - Gold 1

**문제 설명**
N개의 수가 있을 때, 중간에 수의 변경이 일어나면서 구간의 합을 구하세요.

**입력/출력 예시**
```
입력: arr = [1, 2, 3, 4, 5]
      queries = [(update, 2, 6), (sum, 1, 4), (update, 4, 2)]
출력: 15 (2 + 6 + 3 + 4)
```

**풀이**
```python
def solve(arr: list, queries: list) -> list:
    tree = SumSegmentTree(arr)
    results = []

    for query in queries:
        if query[0] == 'update':
            tree.update(query[1], query[2])
        else:
            results.append(tree.query(query[1], query[2]))

    return results
```

---

### 문제 2: 최솟값 찾기 (BOJ 10868) - Gold 1

**문제 설명**
N개의 정수와 M개의 쿼리가 주어질 때, 각 쿼리 구간의 최솟값을 구하세요.

**풀이**
```python
def solve(arr: list, queries: list) -> list:
    tree = MinSegmentTree(arr)
    return [tree.query(a - 1, b - 1) for a, b in queries]
```

---

### 문제 3: 구간 합 구하기 2 (BOJ 10999) - Platinum 4

**문제 설명**
구간에 값을 더하는 업데이트와 구간 합 쿼리를 처리하세요.

**풀이 (Lazy Propagation)**
```python
def solve(arr: list, queries: list) -> list:
    tree = LazySegmentTree(arr)
    results = []

    for query in queries:
        if query[0] == 1:  # 구간 업데이트
            tree.range_update(query[1] - 1, query[2] - 1, query[3])
        else:  # 구간 합
            results.append(tree.query(query[1] - 1, query[2] - 1))

    return results
```

---

### 문제 4: 수열과 쿼리 (BOJ 시리즈) - Platinum

**문제 설명**
다양한 구간 쿼리 문제들 (합, 최소, 최대, XOR 등)

---

## Editorial (풀이 전략)

### Step 1: 사용 시기 판단

| 조건 | 자료구조 |
|------|---------|
| 업데이트 없음 | 누적합 |
| 점 업데이트만 | 펜윅 트리 (간단) |
| 구간 업데이트 | Lazy Segment Tree |
| 최소/최대 쿼리 | 세그먼트 트리 |

### Step 2: 트리 크기

```python
# 안전한 크기: 4 * n
self.tree = [0] * (4 * n)

# 더 정확한 크기: 2^(ceil(log2(n)) + 1)
import math
size = 1 << (math.ceil(math.log2(n)) + 1)
```

### Step 3: 인덱싱

```python
# 0-indexed
왼쪽 자식: 2 * node + 1
오른쪽 자식: 2 * node + 2

# 1-indexed
왼쪽 자식: 2 * node
오른쪽 자식: 2 * node + 1
```

---

## 자주 하는 실수

### 1. 트리 크기 부족
```python
# ❌ 2 * n은 부족할 수 있음
self.tree = [0] * (2 * n)

# ✅ 4 * n으로 안전하게
self.tree = [0] * (4 * n)
```

### 2. 범위 체크 순서
```python
# ❌ 포함 체크가 먼저
if left <= start and end <= right:
    return self.tree[node]
if right < start or end < left:  # 늦음!
    return 0

# ✅ 범위 밖 체크가 먼저
if right < start or end < left:
    return 0
if left <= start and end <= right:
    return self.tree[node]
```

### 3. Lazy 전파 누락
```python
# ❌ 쿼리 시 lazy 처리 안 함
def _query(...):
    if ...:
        return self.tree[node]

# ✅ 쿼리 전에 lazy 전파
def _query(...):
    self._push_down(node, start, end)  # 먼저!
    if ...:
        return self.tree[node]
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 307 | Range Sum Query - Mutable | Medium |
| LeetCode | 315 | Count of Smaller Numbers After Self | Hard |
| LeetCode | 493 | Reverse Pairs | Hard |
| BOJ | 2042 | 구간 합 구하기 | Gold 1 |
| BOJ | 10868 | 최솟값 | Gold 1 |
| BOJ | 2357 | 최솟값과 최댓값 | Gold 1 |
| BOJ | 10999 | 구간 합 구하기 2 | Platinum 4 |
| BOJ | 11505 | 구간 곱 구하기 | Gold 1 |

---

## 임베딩용 키워드

```
segment tree, 세그먼트 트리, fenwick tree, 펜윅 트리, BIT,
range query, 구간 쿼리, point update, 점 업데이트,
lazy propagation, 레이지 프로파게이션, range update, 구간 업데이트,
range sum, 구간 합, range minimum, 구간 최솟값
```
