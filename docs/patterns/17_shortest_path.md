# Pattern 17: Shortest Path (최단 경로)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(E log V) ~ O(V³) |
| **공간복잡도** | O(V²) |
| **선행 지식** | 그래프, 힙, BFS |

## 정의

**최단 경로(Shortest Path)** 알고리즘은 그래프에서 두 정점 사이의 최소 비용/거리 경로를 찾는 알고리즘입니다.

## 알고리즘 비교

| 알고리즘 | 용도 | 시간복잡도 | 음수 가중치 |
|---------|------|-----------|------------|
| **다익스트라** | 단일 출발점 | O(E log V) | ❌ |
| **벨만-포드** | 단일 출발점 + 음수 | O(VE) | ✅ |
| **플로이드-워셜** | 모든 쌍 | O(V³) | ✅ |
| **BFS** | 무가중치 그래프 | O(V + E) | - |
| **0-1 BFS** | 가중치 0 또는 1 | O(V + E) | - |

---

## 템플릿 코드

### 템플릿 1: 다익스트라 (Dijkstra) - 기본

```python
import heapq
from collections import defaultdict

def dijkstra(graph: dict, start: int, n: int) -> list:
    """
    다익스트라 알고리즘 - 단일 출발점 최단 경로

    graph: {node: [(neighbor, weight), ...]}
    Time: O(E log V)
    Space: O(V)
    """
    dist = [float('inf')] * n
    dist[start] = 0

    # (거리, 노드) 최소 힙
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)

        # 이미 처리된 노드 스킵
        if d > dist[u]:
            continue

        for v, w in graph[u]:
            new_dist = dist[u] + w

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist
```

### 템플릿 2: 다익스트라 - 경로 추적

```python
def dijkstra_with_path(graph: dict, start: int, end: int, n: int) -> tuple:
    """
    다익스트라 + 실제 경로 반환

    Returns: (최단 거리, 경로 리스트)
    """
    dist = [float('inf')] * n
    dist[start] = 0
    parent = [-1] * n

    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist[u]:
            continue

        if u == end:
            break

        for v, w in graph[u]:
            new_dist = dist[u] + w

            if new_dist < dist[v]:
                dist[v] = new_dist
                parent[v] = u
                heapq.heappush(heap, (new_dist, v))

    # 경로 복원
    if dist[end] == float('inf'):
        return -1, []

    path = []
    curr = end
    while curr != -1:
        path.append(curr)
        curr = parent[curr]

    return dist[end], path[::-1]
```

### 템플릿 3: 다익스트라 - 2D 그리드

```python
def dijkstra_grid(grid: list, start: tuple, end: tuple) -> int:
    """
    2D 그리드에서 최단 경로 (가중치 있는 경우)

    Time: O(RC log RC)
    """
    rows, cols = len(grid), len(grid[0])
    dist = [[float('inf')] * cols for _ in range(rows)]

    sr, sc = start
    dist[sr][sc] = grid[sr][sc]

    heap = [(grid[sr][sc], sr, sc)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while heap:
        d, r, c = heapq.heappop(heap)

        if (r, c) == end:
            return d

        if d > dist[r][c]:
            continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = dist[r][c] + grid[nr][nc]

                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    heapq.heappush(heap, (new_dist, nr, nc))

    return dist[end[0]][end[1]]
```

### 템플릿 4: 벨만-포드 (Bellman-Ford)

```python
def bellman_ford(edges: list, n: int, start: int) -> list:
    """
    벨만-포드 알고리즘 - 음수 가중치 허용

    edges: [(u, v, weight), ...]
    Time: O(VE)
    Space: O(V)

    Returns: 거리 배열 (음수 사이클 있으면 None)
    """
    dist = [float('inf')] * n
    dist[start] = 0

    # V-1번 반복
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True

        if not updated:  # 조기 종료
            break

    # 음수 사이클 검사
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # 음수 사이클 존재

    return dist
```

### 템플릿 5: 플로이드-워셜 (Floyd-Warshall)

```python
def floyd_warshall(graph: list, n: int) -> list:
    """
    플로이드-워셜 알고리즘 - 모든 쌍 최단 경로

    graph: 인접 행렬 (graph[i][j] = i에서 j로 가는 가중치, 없으면 inf)
    Time: O(V³)
    Space: O(V²)
    """
    INF = float('inf')

    # 거리 행렬 초기화
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u in range(n):
        for v, w in graph[u]:
            dist[u][v] = w

    # 중간 경유지 k를 거쳐가는 경우
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

### 템플릿 6: 플로이드-워셜 - 경로 추적

```python
def floyd_warshall_with_path(n: int, edges: list) -> tuple:
    """
    플로이드-워셜 + 경로 복원

    Returns: (거리 행렬, 다음 노드 행렬)
    """
    INF = float('inf')

    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = w
        next_node[u][v] = v

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node


def get_path(next_node: list, u: int, v: int) -> list:
    """i에서 j로 가는 경로 복원"""
    if next_node[u][v] is None:
        return []

    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)

    return path
```

### 템플릿 7: 0-1 BFS

```python
from collections import deque

def bfs_01(graph: dict, start: int, n: int) -> list:
    """
    0-1 BFS - 가중치가 0 또는 1인 그래프

    Time: O(V + E)
    Space: O(V)
    """
    dist = [float('inf')] * n
    dist[start] = 0

    dq = deque([start])

    while dq:
        u = dq.popleft()

        for v, w in graph[u]:
            new_dist = dist[u] + w

            if new_dist < dist[v]:
                dist[v] = new_dist

                if w == 0:
                    dq.appendleft(v)  # 가중치 0이면 앞에
                else:
                    dq.append(v)  # 가중치 1이면 뒤에

    return dist
```

### 템플릿 8: K번째 최단 경로

```python
def kth_shortest_path(graph: dict, start: int, end: int, n: int, k: int) -> int:
    """
    K번째 최단 경로

    Time: O(KE log V)
    """
    count = [0] * n
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        count[u] += 1

        if count[u] == k and u == end:
            return d

        if count[u] > k:
            continue

        for v, w in graph[u]:
            heapq.heappush(heap, (d + w, v))

    return -1
```

---

## 예제 문제

### 문제 1: 배달 (프로그래머스 Level 2) - Medium

**문제 설명**
1번 마을에서 출발하여 K 시간 이내로 배달 가능한 마을 수를 구하세요.

**입력/출력 예시**
```
입력: N = 5, roads = [[1,2,1],[2,3,3],[5,2,2],[1,4,2],[5,3,1],[5,4,2]], K = 3
출력: 4
설명: 1번에서 3시간 내 도달 가능: 1, 2, 4, 5
```

**풀이**
```python
def delivery(N: int, roads: list, K: int) -> int:
    graph = defaultdict(list)

    for a, b, c in roads:
        graph[a].append((b, c))
        graph[b].append((a, c))

    dist = dijkstra(graph, 1, N + 1)

    return sum(1 for d in dist[1:] if d <= K)
```

---

### 문제 2: 합승 택시 요금 (카카오 2021) - Hard

**문제 설명**
A, B 두 사람이 택시를 합승할 때 최소 요금을 구하세요.

**입력/출력 예시**
```
입력: n = 6, s = 4, a = 6, b = 2,
      fares = [[4,1,10],[3,5,24],[5,6,2],[3,1,41],[5,1,24],[4,6,50],[2,4,66],[2,3,22],[1,6,25]]
출력: 82
```

**풀이**
```python
def taxi_fare(n: int, s: int, a: int, b: int, fares: list) -> int:
    INF = float('inf')

    # 플로이드-워셜로 모든 쌍 최단 거리 계산
    dist = [[INF] * (n + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dist[i][i] = 0

    for u, v, w in fares:
        dist[u][v] = w
        dist[v][u] = w

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # 합승 지점 i에서 헤어지는 경우
    answer = INF
    for i in range(1, n + 1):
        # s → i (합승) + i → a + i → b
        answer = min(answer, dist[s][i] + dist[i][a] + dist[i][b])

    return answer
```

---

### 문제 3: 최소비용 구하기 (BOJ 1916) - Medium

**문제 설명**
N개의 도시와 M개의 버스 노선이 있을 때, 출발 도시에서 도착 도시까지 최소 비용을 구하세요.

**풀이**
```python
def min_cost(n: int, buses: list, start: int, end: int) -> int:
    graph = defaultdict(list)

    for a, b, c in buses:
        graph[a].append((b, c))

    dist = dijkstra(graph, start, n + 1)

    return dist[end]
```

---

### 문제 4: 타임머신 (BOJ 11657) - Hard

**문제 설명**
음수 가중치가 있는 그래프에서 1번에서 모든 정점까지 최단 거리를 구하세요. 음수 사이클이 있으면 -1을 출력합니다.

**풀이**
```python
def time_machine(n: int, edges: list) -> list:
    result = bellman_ford(edges, n + 1, 1)

    if result is None:
        return [-1]  # 음수 사이클

    return [r if r != float('inf') else -1 for r in result[1:]]
```

---

### 문제 5: 경로 찾기 (BOJ 11403) - Medium

**문제 설명**
모든 정점 쌍 (i, j)에 대해 i에서 j로 가는 경로가 있는지 확인하세요.

**풀이**
```python
def find_path(n: int, graph: list) -> list:
    # 플로이드-워셜 변형 (경로 존재 여부만)
    reachable = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if graph[i][j]:
                reachable[i][j] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if reachable[i][k] and reachable[k][j]:
                    reachable[i][j] = True

    return [[1 if reachable[i][j] else 0 for j in range(n)] for i in range(n)]
```

---

### 문제 6: 알고스팟 (BOJ 1261) - Medium

**문제 설명**
0은 빈 방, 1은 벽입니다. 벽을 부수면서 (1,1)에서 (N,M)까지 가는 최소 벽 부수기 횟수를 구하세요.

**풀이 (0-1 BFS)**
```python
def algospot(grid: list) -> int:
    n, m = len(grid), len(grid[0])
    dist = [[float('inf')] * m for _ in range(n)]
    dist[0][0] = 0

    dq = deque([(0, 0)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while dq:
        r, c = dq.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < m:
                cost = grid[nr][nc]  # 0 또는 1

                if dist[r][c] + cost < dist[nr][nc]:
                    dist[nr][nc] = dist[r][c] + cost

                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))

    return dist[n-1][m-1]
```

---

## Editorial (풀이 전략)

### Step 1: 알고리즘 선택

```
Q1: 음수 가중치가 있는가?
    Yes → 벨만-포드 또는 플로이드-워셜
    No → 다익스트라

Q2: 단일 출발점 vs 모든 쌍?
    단일 → 다익스트라/벨만-포드
    모든 쌍 → 플로이드-워셜

Q3: 가중치가 0 또는 1뿐인가?
    Yes → 0-1 BFS (더 빠름)

Q4: 가중치가 없는가?
    Yes → 일반 BFS
```

### Step 2: 그래프 표현

```python
# 인접 리스트 (다익스트라, 벨만-포드)
graph = defaultdict(list)
for u, v, w in edges:
    graph[u].append((v, w))

# 인접 행렬 (플로이드-워셜)
dist = [[INF] * n for _ in range(n)]
for u, v, w in edges:
    dist[u][v] = w
```

### Step 3: 주의사항

| 알고리즘 | 주의사항 |
|---------|---------|
| 다익스트라 | 음수 가중치 ❌ |
| 벨만-포드 | O(VE)로 느림 |
| 플로이드 | O(V³), V ≤ 500 정도 |
| 0-1 BFS | 가중치 0,1만 |

---

## 자주 하는 실수

### 1. 다익스트라에서 중복 방문
```python
# ❌ 이미 처리된 노드 재방문
while heap:
    d, u = heappop(heap)
    for v, w in graph[u]:
        ...

# ✅ 스킵 조건 추가
while heap:
    d, u = heappop(heap)
    if d > dist[u]:  # 이미 더 짧은 경로로 처리됨
        continue
    ...
```

### 2. 플로이드-워셜 순서 오류
```python
# ❌ k가 안쪽 루프
for i in range(n):
    for j in range(n):
        for k in range(n):  # 틀림!
            ...

# ✅ k가 가장 바깥 루프
for k in range(n):
    for i in range(n):
        for j in range(n):
            ...
```

### 3. 인덱스 범위
```python
# ❌ 1-indexed 그래프인데 0-indexed 배열
dist = [INF] * n  # 0 ~ n-1

# ✅ n+1 크기로 생성
dist = [INF] * (n + 1)  # 1 ~ n
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 | 알고리즘 |
|--------|---|-------|-------|---------|
| LeetCode | 743 | Network Delay Time | Medium | 다익스트라 |
| LeetCode | 787 | Cheapest Flights Within K Stops | Medium | 벨만-포드 |
| LeetCode | 1334 | Find the City | Medium | 플로이드 |
| LeetCode | 1368 | Minimum Cost to Make at Least One Valid Path | Hard | 0-1 BFS |
| BOJ | 1753 | 최단경로 | Gold 4 | 다익스트라 |
| BOJ | 1916 | 최소비용 구하기 | Gold 5 | 다익스트라 |
| BOJ | 11657 | 타임머신 | Gold 4 | 벨만-포드 |
| BOJ | 11404 | 플로이드 | Gold 4 | 플로이드 |
| BOJ | 1261 | 알고스팟 | Gold 4 | 0-1 BFS |
| 프로그래머스 | - | 배달 | Level 2 | 다익스트라 |
| 프로그래머스 | - | 합승 택시 요금 | Level 3 | 플로이드 |

---

## 임베딩용 키워드

```
shortest path, 최단 경로, dijkstra, 다익스트라,
bellman-ford, 벨만포드, floyd-warshall, 플로이드 워셜,
negative weight, 음수 가중치, single source, 단일 출발점,
all pairs, 모든 쌍, 0-1 BFS, 배달, 합승 택시
```
