# Pattern 24: Minimum Spanning Tree (최소 스패닝 트리)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(E log E) |
| **공간복잡도** | O(V + E) |
| **선행 지식** | 그래프, Union-Find, 힙 |

## 정의

**최소 스패닝 트리(MST)**는 가중치 그래프에서 **모든 정점을 연결**하면서 **간선 가중치 합이 최소**인 트리입니다.

## 핵심 아이디어

```
그래프:          MST:
  2                2
A---B            A---B
|\ /|              |
3|X |4          3  |
|/ \|              |
C---D            C   D
  5                (A-C 또는 B-D로 연결)

MST 조건:
- V개 정점 → V-1개 간선
- 모든 정점 연결
- 사이클 없음
- 가중치 합 최소
```

## 크루스칼 vs 프림

| 특성 | 크루스칼 | 프림 |
|------|---------|------|
| 접근 | 간선 중심 | 정점 중심 |
| 정렬 | 간선 정렬 O(E log E) | 힙 사용 |
| 자료구조 | Union-Find | 힙 + 방문 배열 |
| 적합 | 희소 그래프 | 밀집 그래프 |
| 구현 | 상대적 쉬움 | 상대적 복잡 |

---

## 템플릿 코드

### 템플릿 1: 크루스칼 알고리즘

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal(n: int, edges: list) -> tuple:
    """
    크루스칼 알고리즘

    edges: [(u, v, weight), ...]
    Time: O(E log E)
    Space: O(V)

    Returns: (MST 가중치 합, MST 간선 리스트)
    """
    # 간선 가중치 기준 정렬
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))

            if len(mst_edges) == n - 1:
                break

    # 연결 그래프인지 확인
    if len(mst_edges) != n - 1:
        return -1, []  # MST 불가능

    return mst_weight, mst_edges
```

### 템플릿 2: 프림 알고리즘

```python
import heapq
from collections import defaultdict

def prim(n: int, edges: list, start: int = 0) -> tuple:
    """
    프림 알고리즘

    Time: O(E log V)
    Space: O(V + E)

    Returns: (MST 가중치 합, MST 간선 리스트)
    """
    # 인접 리스트 구성
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))

    visited = [False] * n
    heap = [(0, start, -1)]  # (가중치, 현재 노드, 이전 노드)
    mst_weight = 0
    mst_edges = []

    while heap and len(mst_edges) < n:
        w, u, prev = heapq.heappop(heap)

        if visited[u]:
            continue

        visited[u] = True
        mst_weight += w

        if prev != -1:
            mst_edges.append((prev, u, w))

        for weight, v in graph[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v, u))

    if len(mst_edges) != n - 1:
        return -1, []

    return mst_weight, mst_edges
```

### 템플릿 3: 최소 스패닝 트리 비용만

```python
def mst_cost(n: int, edges: list) -> int:
    """MST 가중치 합만 반환"""
    edges.sort(key=lambda x: x[2])

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    cost = 0
    count = 0

    for u, v, w in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            cost += w
            count += 1
            if count == n - 1:
                break

    return cost if count == n - 1 else -1
```

### 템플릿 4: 2차 최소 스패닝 트리

```python
def second_mst(n: int, edges: list) -> int:
    """
    두 번째로 작은 MST 가중치

    Time: O(E² log E)
    """
    # 1차 MST 구하기
    mst_cost, mst_edges = kruskal(n, edges)

    if mst_cost == -1:
        return -1

    mst_set = set((min(u, v), max(u, v)) for u, v, _ in mst_edges)

    second_cost = float('inf')

    # MST의 각 간선을 제외하고 MST 재구성
    for u, v, w in mst_edges:
        excluded = (min(u, v), max(u, v))

        # excluded 제외하고 MST 계산
        filtered_edges = [(a, b, c) for a, b, c in edges
                          if (min(a, b), max(a, b)) != excluded]

        cost, _ = kruskal(n, filtered_edges)

        if cost != -1:
            second_cost = min(second_cost, cost)

    return second_cost if second_cost != float('inf') else -1
```

### 템플릿 5: 최대 스패닝 트리

```python
def max_spanning_tree(n: int, edges: list) -> int:
    """최대 스패닝 트리"""
    # 가중치 내림차순 정렬
    edges.sort(key=lambda x: -x[2])

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    cost = 0
    count = 0

    for u, v, w in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            cost += w
            count += 1
            if count == n - 1:
                break

    return cost if count == n - 1 else -1
```

### 템플릿 6: MST + 추가 간선 최소 비용

```python
def mst_with_mandatory_edge(n: int, edges: list, mandatory: tuple) -> int:
    """
    특정 간선을 반드시 포함하는 MST

    mandatory: (u, v, w)
    """
    u, v, w = mandatory

    # 먼저 mandatory 간선 연결
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    parent[find(u)] = find(v)
    cost = w
    count = 1

    # 나머지 간선 처리
    edges.sort(key=lambda x: x[2])

    for a, b, c in edges:
        if (a, b, c) == mandatory:
            continue

        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb
            cost += c
            count += 1
            if count == n - 1:
                break

    return cost if count == n - 1 else -1
```

---

## 예제 문제

### 문제 1: 네트워크 연결 (BOJ 1922) - Gold 4

**문제 설명**
N개의 컴퓨터를 모두 연결하는 최소 비용.

**입력/출력 예시**
```
입력: n = 6, edges = [(1,2,5), (1,3,4), (2,3,2), (2,4,7), (3,4,6), (3,5,11), (4,5,3), (4,6,8), (5,6,8)]
출력: 23
```

**풀이**
```python
def solution(n: int, edges: list) -> int:
    # 0-indexed로 변환
    edges = [(u - 1, v - 1, w) for u, v, w in edges]
    return mst_cost(n, edges)
```

---

### 문제 2: 섬 연결하기 (프로그래머스 Level 3)

**문제 설명**
n개의 섬을 모두 연결하는 최소 비용.

**입력/출력 예시**
```
입력: n = 4, costs = [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]]
출력: 4
```

**풀이**
```python
def solution(n: int, costs: list) -> int:
    edges = [(u, v, w) for u, v, w in costs]
    return mst_cost(n, edges)
```

---

### 문제 3: 최소 스패닝 트리 (BOJ 1197) - Gold 4

**문제 설명**
MST의 가중치 합 구하기.

**풀이**
```python
def solution(v: int, e: int, edges: list) -> int:
    # 1-indexed → 0-indexed
    edges = [(a - 1, b - 1, c) for a, b, c in edges]
    return mst_cost(v, edges)
```

---

### 문제 4: 도시 분할 계획 (BOJ 1647) - Gold 4

**문제 설명**
마을을 두 개로 분할하고, 각 마을 내 연결 비용 최소화.

**풀이 (MST에서 가장 큰 간선 제거)**
```python
def solution(n: int, edges: list) -> int:
    edges = [(a - 1, b - 1, c) for a, b, c in edges]
    edges.sort(key=lambda x: x[2])

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    cost = 0
    max_edge = 0
    count = 0

    for u, v, w in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            cost += w
            max_edge = max(max_edge, w)
            count += 1
            if count == n - 1:
                break

    # 가장 큰 간선 제거 = 두 마을 분리
    return cost - max_edge
```

---

### 문제 5: 별자리 만들기 (BOJ 4386) - Gold 3

**문제 설명**
2D 좌표의 별들을 모두 연결하는 최소 비용 (거리 = 비용).

**풀이**
```python
import math

def solution(stars: list) -> float:
    n = len(stars)

    # 모든 쌍의 거리 계산
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = stars[i]
            x2, y2 = stars[j]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            edges.append((i, j, dist))

    return mst_cost(n, edges)
```

---

### 문제 6: 행성 연결 (BOJ 16398) - Gold 4

**문제 설명**
N개의 행성을 연결하는 최소 비용 (인접 행렬).

**풀이**
```python
def solution(n: int, cost_matrix: list) -> int:
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, cost_matrix[i][j]))

    return mst_cost(n, edges)
```

---

## Editorial (풀이 전략)

### Step 1: MST 사용 판단

| 키워드 | MST 적용 |
|--------|---------|
| 모든 정점 연결 | ✅ |
| 최소 비용 | ✅ |
| 사이클 없음 | ✅ |
| 트리 구조 | ✅ |

### Step 2: 알고리즘 선택

```
희소 그래프 (E ≈ V): 크루스칼
밀집 그래프 (E ≈ V²): 프림

보통 크루스칼이 구현 쉬움
```

### Step 3: 변형 문제

| 변형 | 접근법 |
|------|-------|
| 두 그룹 분할 | MST - 최대 간선 |
| K개 그룹 | MST - 상위 K-1 간선 |
| 특정 간선 포함 | 해당 간선 먼저 연결 |
| 2차 MST | 각 간선 제외 후 재계산 |

---

## 자주 하는 실수

### 1. 연결 그래프 확인 누락
```python
# ❌ MST 구성 불가능 체크 안 함
return cost

# ✅ 간선 수 확인
if count != n - 1:
    return -1
return cost
```

### 2. 인덱스 변환
```python
# ❌ 1-indexed 그대로 사용
edges = [(u, v, w) for ...]

# ✅ 0-indexed로 변환
edges = [(u - 1, v - 1, w) for ...]
```

### 3. 양방향 간선 중복
```python
# ❌ 양방향 간선을 두 번 추가
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))
# edges에 (u, v)와 (v, u)가 모두 있으면 중복!

# ✅ 입력 형태 확인
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 1135 | Connecting Cities With Minimum Cost | Medium |
| LeetCode | 1584 | Min Cost to Connect All Points | Medium |
| LeetCode | 1489 | Find Critical and Pseudo-Critical Edges | Hard |
| BOJ | 1197 | 최소 스패닝 트리 | Gold 4 |
| BOJ | 1922 | 네트워크 연결 | Gold 4 |
| BOJ | 1647 | 도시 분할 계획 | Gold 4 |
| BOJ | 4386 | 별자리 만들기 | Gold 3 |
| BOJ | 1368 | 물대기 | Gold 2 |
| 프로그래머스 | - | 섬 연결하기 | Level 3 |

---

## 임베딩용 키워드

```
minimum spanning tree, 최소 스패닝 트리, MST, kruskal, 크루스칼,
prim, 프림, union-find, 유니온 파인드, greedy, 탐욕,
network connection, 네트워크 연결, 섬 연결하기, 도시 분할
```
