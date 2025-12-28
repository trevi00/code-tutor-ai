# Pattern 16: Union-Find (유니온 파인드 / 서로소 집합)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (카카오 매년 출제) |
| **시간복잡도** | O(α(n)) ≈ O(1) |
| **공간복잡도** | O(n) |
| **선행 지식** | 트리, 그래프 기초 |

## 정의

**Union-Find (Disjoint Set Union, DSU)**는 서로소 집합들을 효율적으로 관리하는 자료구조입니다. **합치기(Union)**와 **찾기(Find)** 연산을 거의 상수 시간에 수행합니다.

## 핵심 아이디어

```
초기 상태: 각 원소가 자기 자신을 부모로
[0] [1] [2] [3] [4]

Union(0, 1): 0과 1을 합침
  1
  |
  0     [2] [3] [4]

Union(2, 3): 2와 3을 합침
  1       3
  |       |
  0       2     [4]

Union(1, 3): 두 그룹 합침
      3
    / | \
   1  2  4
   |
   0

Find(0) → 3 (루트 찾기)
```

## 최적화 기법

| 기법 | 설명 | 효과 |
|------|------|------|
| **경로 압축** | Find 시 루트로 직접 연결 | O(log n) → O(α(n)) |
| **랭크 기반 합치기** | 작은 트리를 큰 트리에 붙임 | 트리 높이 최소화 |

---

## 템플릿 코드

### 템플릿 1: 기본 Union-Find

```python
class UnionFind:
    """
    기본 Union-Find (경로 압축 + 랭크 최적화)

    Time: O(α(n)) per operation (거의 상수)
    Space: O(n)
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """루트 찾기 + 경로 압축"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 경로 압축
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """두 집합 합치기 (이미 같은 집합이면 False)"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # 랭크 기반 합치기
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        """같은 집합인지 확인"""
        return self.find(x) == self.find(y)
```

### 템플릿 2: 집합 크기 추적

```python
class UnionFindWithSize:
    """
    집합 크기를 추적하는 Union-Find
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n  # 각 집합의 크기
        self.count = n  # 집합의 개수

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # 작은 집합을 큰 집합에 붙임
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.count -= 1

        return True

    def get_size(self, x: int) -> int:
        """x가 속한 집합의 크기"""
        return self.size[self.find(x)]

    def get_count(self) -> int:
        """집합의 총 개수"""
        return self.count
```

### 템플릿 3: 가중치 Union-Find (Weighted)

```python
class WeightedUnionFind:
    """
    간선에 가중치가 있는 Union-Find
    용도: A/B = k 형태의 관계 처리
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.weight = [1.0] * n  # 부모까지의 가중치

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] *= self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def union(self, x: int, y: int, w: float) -> bool:
        """x / y = w 관계 설정"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # weight[x] * ? = weight[y] * w
        self.parent[root_x] = root_y
        self.weight[root_x] = w * self.weight[y] / self.weight[x]

        return True

    def query(self, x: int, y: int) -> float:
        """x / y 값 계산"""
        if self.find(x) != self.find(y):
            return -1.0
        return self.weight[x] / self.weight[y]
```

### 템플릿 4: 2D 그리드 Union-Find

```python
class GridUnionFind:
    """
    2D 그리드용 Union-Find
    좌표 (r, c) → 1D 인덱스로 변환
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.parent = list(range(rows * cols))
        self.rank = [0] * (rows * cols)
        self.count = 0  # 활성화된 셀 수

    def _index(self, r: int, c: int) -> int:
        return r * self.cols + c

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        idx1 = self._index(r1, c1)
        idx2 = self._index(r2, c2)

        root1 = self.find(idx1)
        root2 = self.find(idx2)

        if root1 == root2:
            return False

        if self.rank[root1] < self.rank[root2]:
            root1, root2 = root2, root1

        self.parent[root2] = root1
        if self.rank[root1] == self.rank[root2]:
            self.rank[root1] += 1

        self.count -= 1
        return True

    def add_land(self, r: int, c: int):
        """새 땅 추가 (섬 문제)"""
        self.count += 1

    def connected(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        return self.find(self._index(r1, c1)) == self.find(self._index(r2, c2))
```

### 템플릿 5: 온라인 연결성 (동적 그래프)

```python
def num_islands_ii(m: int, n: int, positions: list) -> list:
    """
    섬의 개수 II - 동적으로 땅 추가

    Time: O(k × α(m×n)), k = positions 수
    Space: O(m × n)
    """
    uf = GridUnionFind(m, n)
    grid = [[0] * n for _ in range(m)]
    result = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for r, c in positions:
        if grid[r][c] == 1:  # 이미 땅
            result.append(uf.count)
            continue

        grid[r][c] = 1
        uf.add_land(r, c)

        # 인접한 땅과 연결
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                uf.union(r, c, nr, nc)

        result.append(uf.count)

    return result
```

---

## 예제 문제

### 문제 1: 네트워크 연결 (카카오 2019) - Medium

**문제 설명**
n개의 컴퓨터와 연결 정보가 주어질 때, 네트워크의 개수를 구하세요.

**입력/출력 예시**
```
입력: n = 3, connections = [[0,1], [1,2]]
출력: 1

입력: n = 3, connections = [[0,1]]
출력: 2
```

**풀이**
```python
def count_networks(n: int, connections: list) -> int:
    uf = UnionFind(n)

    for a, b in connections:
        uf.union(a, b)

    # 루트의 개수 = 네트워크 개수
    return len(set(uf.find(i) for i in range(n)))
```

---

### 문제 2: 섬 연결하기 (프로그래머스 Level 3) - Medium

**문제 설명**
n개의 섬을 모두 연결하는 최소 비용을 구하세요. (MST + Union-Find)

**입력/출력 예시**
```
입력: n = 4, costs = [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]]
출력: 4
설명: 0-1, 1-3, 0-2 연결 (비용 1+1+2=4)
```

**풀이**
```python
def min_cost_to_connect(n: int, costs: list) -> int:
    # 크루스칼 알고리즘 (MST)
    costs.sort(key=lambda x: x[2])  # 비용 기준 정렬

    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0

    for a, b, cost in costs:
        if uf.union(a, b):  # 사이클이 아니면
            total_cost += cost
            edges_used += 1

            if edges_used == n - 1:  # 모든 섬 연결됨
                break

    return total_cost
```

---

### 문제 3: 친구 네트워크 (BOJ 4195) - Medium

**문제 설명**
두 사람이 친구가 될 때마다 그 친구 네트워크의 크기를 출력하세요.

**입력/출력 예시**
```
입력:
Fred Barney
Barney Betty
Betty Wilma

출력:
2  (Fred-Barney)
3  (Fred-Barney-Betty)
4  (Fred-Barney-Betty-Wilma)
```

**풀이**
```python
def friend_network(friendships: list) -> list:
    parent = {}
    size = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            size[x] = 1
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            if size[root_x] < size[root_y]:
                root_x, root_y = root_y, root_x
            parent[root_y] = root_x
            size[root_x] += size[root_y]
        return size[root_x]

    result = []
    for a, b in friendships:
        result.append(union(a, b))

    return result
```

---

### 문제 4: 등산 코스 (카카오 2022) - Hard

**문제 설명**
여러 출발점에서 여러 도착점으로 가는 경로 중, 가장 높은 간선 가중치가 최소인 경로를 찾으세요.

**풀이 (Union-Find + 정렬)**
```python
def climbing_course(n: int, paths: list, gates: list, summits: list) -> list:
    # 간선을 가중치 순으로 정렬
    paths.sort(key=lambda x: x[2])

    gate_set = set(gates)
    summit_set = set(summits)

    uf = UnionFind(n + 1)

    for a, b, w in paths:
        # 정상은 합치지 않음 (도착점이므로)
        if a in summit_set or b in summit_set:
            continue
        uf.union(a, b)

    # 출발점과 연결된 정상 찾기
    result = [float('inf'), float('inf')]

    for a, b, w in paths:
        if a in summit_set:
            a, b = b, a
        if b in summit_set:
            for gate in gates:
                if uf.connected(a, gate):
                    if w < result[1] or (w == result[1] and b < result[0]):
                        result = [b, w]

    return result
```

---

### 문제 5: 사이클 판별 (BOJ 20040) - Medium

**문제 설명**
무방향 그래프에서 간선을 추가하다가 처음으로 사이클이 생기는 시점을 찾으세요.

**입력/출력 예시**
```
입력: n = 3, edges = [[0,1], [1,2], [2,0]]
출력: 3 (3번째 간선에서 사이클 발생)
```

**풀이**
```python
def detect_cycle(n: int, edges: list) -> int:
    uf = UnionFind(n)

    for i, (a, b) in enumerate(edges, 1):
        if not uf.union(a, b):  # 이미 연결됨 → 사이클!
            return i

    return 0  # 사이클 없음
```

---

### 문제 6: 여행 경로 (Evaluate Division) - Medium

**문제 설명**
a/b = k 형태의 방정식들이 주어질 때, 쿼리 x/y를 계산하세요.

**입력/출력 예시**
```
입력: equations = [["a","b"],["b","c"]], values = [2.0, 3.0]
      queries = [["a","c"],["b","a"]]
출력: [6.0, 0.5]
설명: a/b=2, b/c=3 → a/c=6, b/a=0.5
```

**풀이**
```python
def calc_equation(equations: list, values: list, queries: list) -> list:
    # 변수를 인덱스로 매핑
    var_to_idx = {}
    idx = 0

    for a, b in equations:
        if a not in var_to_idx:
            var_to_idx[a] = idx
            idx += 1
        if b not in var_to_idx:
            var_to_idx[b] = idx
            idx += 1

    uf = WeightedUnionFind(idx)

    for (a, b), val in zip(equations, values):
        uf.union(var_to_idx[a], var_to_idx[b], val)

    result = []
    for x, y in queries:
        if x not in var_to_idx or y not in var_to_idx:
            result.append(-1.0)
        else:
            result.append(uf.query(var_to_idx[x], var_to_idx[y]))

    return result
```

---

## Editorial (풀이 전략)

### Step 1: Union-Find 사용 시기 판단

| 키워드 | Union-Find 적용 |
|--------|----------------|
| 연결 요소 개수 | ✅ |
| 그룹/집합 관리 | ✅ |
| 사이클 판별 | ✅ |
| 동적 연결성 | ✅ |
| MST (크루스칼) | ✅ |

### Step 2: 최적화 적용

```python
# 1. 경로 압축 (필수!)
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 재귀적 압축
    return parent[x]

# 2. 랭크/크기 기반 합치기 (권장)
if rank[root_x] < rank[root_y]:
    root_x, root_y = root_y, root_x
parent[root_y] = root_x
```

### Step 3: 문제 유형별 접근

| 유형 | 추가 기능 |
|------|----------|
| 연결 요소 수 | `count` 변수 |
| 집합 크기 | `size` 배열 |
| 가중치 관계 | `weight` 배열 |
| 2D 그리드 | 좌표 → 인덱스 변환 |

---

## 자주 하는 실수

### 1. 경로 압축 누락
```python
# ❌ 단순 find (O(n))
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x

# ✅ 경로 압축 (O(α(n)))
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
```

### 2. Union 시 루트 비교 누락
```python
# ❌ 직접 합침 (잘못됨)
def union(x, y):
    parent[x] = y

# ✅ 루트끼리 합침
def union(x, y):
    root_x, root_y = find(x), find(y)
    if root_x != root_y:
        parent[root_y] = root_x
```

### 3. 인덱스 범위 오류
```python
# ❌ 0-indexed인데 1-indexed로 접근
uf = UnionFind(n)
uf.union(1, n)  # IndexError!

# ✅ 범위 확인 또는 n+1로 생성
uf = UnionFind(n + 1)
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 547 | Number of Provinces | Medium |
| LeetCode | 684 | Redundant Connection | Medium |
| LeetCode | 721 | Accounts Merge | Medium |
| LeetCode | 399 | Evaluate Division | Medium |
| LeetCode | 305 | Number of Islands II | Hard |
| LeetCode | 1319 | Number of Operations to Make Network Connected | Medium |
| BOJ | 1717 | 집합의 표현 | Gold 5 |
| BOJ | 4195 | 친구 네트워크 | Gold 2 |
| BOJ | 20040 | 사이클 게임 | Gold 4 |
| 프로그래머스 | - | 네트워크 | Level 3 |
| 프로그래머스 | - | 섬 연결하기 | Level 3 |

---

## 임베딩용 키워드

```
union find, 유니온 파인드, disjoint set, 서로소 집합, DSU,
connected components, 연결 요소, cycle detection, 사이클 판별,
path compression, 경로 압축, rank, 랭크, kruskal, 크루스칼,
network, 네트워크, 친구 네트워크, 집합의 표현
```
