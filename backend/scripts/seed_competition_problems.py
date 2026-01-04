"""대회 수준 알고리즘 문제 시딩 스크립트 (25문제)

타겟:
- 코드포스 Div2 수준 (8문제)
- ICPC/IOI 수준 (6문제)
- 삼성 SW 역량테스트 (6문제)
- 카카오/네이버 코테 (5문제)
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"

PROBLEMS = [
    # ============== 코드포스 Div2 수준 (8문제) ==============
    {
        "title": "Segment Tree Range Update",
        "description": """배열이 주어지고, 구간 업데이트와 구간 합 쿼리를 처리하세요.

### 입력
- 정수 배열 `arr` (1 ≤ len(arr) ≤ 10^5)
- 쿼리 리스트 `queries`:
  - `[1, l, r, val]`: arr[l:r+1]에 val을 더함
  - `[2, l, r]`: arr[l:r+1]의 합을 반환

### 출력
- 타입 2 쿼리에 대한 결과 리스트

### 예제
```
입력:
arr = [1, 2, 3, 4, 5]
queries = [[1, 0, 2, 10], [2, 0, 4], [1, 2, 4, 5], [2, 2, 4]]
출력: [45, 27]

설명:
- [1,0,2,10]: [11,12,13,4,5]
- [2,0,4]: 11+12+13+4+5 = 45
- [1,2,4,5]: [11,12,18,9,10]
- [2,2,4]: 18+9+10 = 37
```

### 힌트
- Lazy Propagation을 사용한 세그먼트 트리가 필요합니다.""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n <= 10^5, 1 <= q <= 10^5",
        "hints": ["세그먼트 트리의 각 노드에 lazy 값 저장", "쿼리/업데이트 시 lazy 전파"],
        "solution_template": "def solution(arr: list, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(arr: list, queries: list) -> list:
    n = len(arr)
    tree = [0] * (4 * n)
    lazy = [0] * (4 * n)

    def build(node, start, end):
        if start == end:
            tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            build(2*node, start, mid)
            build(2*node+1, mid+1, end)
            tree[node] = tree[2*node] + tree[2*node+1]

    def push_down(node, start, end):
        if lazy[node] != 0:
            mid = (start + end) // 2
            tree[2*node] += lazy[node] * (mid - start + 1)
            tree[2*node+1] += lazy[node] * (end - mid)
            lazy[2*node] += lazy[node]
            lazy[2*node+1] += lazy[node]
            lazy[node] = 0

    def update(node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            tree[node] += val * (end - start + 1)
            lazy[node] += val
            return
        push_down(node, start, end)
        mid = (start + end) // 2
        update(2*node, start, mid, l, r, val)
        update(2*node+1, mid+1, end, l, r, val)
        tree[node] = tree[2*node] + tree[2*node+1]

    def query(node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        push_down(node, start, end)
        mid = (start + end) // 2
        return query(2*node, start, mid, l, r) + query(2*node+1, mid+1, end, l, r)

    build(1, 0, n-1)
    result = []
    for q in queries:
        if q[0] == 1:
            update(1, 0, n-1, q[1], q[2], q[3])
        else:
            result.append(query(1, 0, n-1, q[1], q[2]))
    return result''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["segment-tree", "lazy-propagation"],
        "pattern_explanation": "Lazy Propagation은 구간 업데이트를 O(log n)에 처리하는 기법입니다. 업데이트 값을 즉시 전파하지 않고 lazy 배열에 저장했다가, 해당 노드를 방문할 때만 자식에게 전파합니다. 구간 합, 구간 최솟값/최댓값 등 다양한 구간 쿼리에 적용 가능합니다.",
        "approach_hint": "Lazy Propagation 세그먼트 트리",
        "time_complexity_hint": "O((n + q) log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]\n[[1, 0, 2, 10], [2, 0, 4], [1, 2, 4, 5], [2, 2, 4]]", "output": "[45, 37]", "is_sample": True},
            {"input": "[0, 0, 0]\n[[1, 0, 2, 1], [2, 0, 2]]", "output": "[3]", "is_sample": True},
            {"input": "[1, 1, 1, 1]\n[[2, 0, 3], [1, 1, 2, 5], [2, 0, 3]]", "output": "[4, 14]", "is_sample": False},
        ]
    },
    {
        "title": "상태 다익스트라",
        "description": """그래프에서 최단 경로를 찾되, 최대 K개의 간선을 무료로 사용할 수 있습니다.

### 입력
- 정점 수 `n`, 무료 간선 수 `k`
- 간선 리스트 `edges`: [[u, v, w], ...] (양방향)
- 시작점 `start`, 도착점 `end`

### 출력
- start에서 end까지의 최소 비용 (도달 불가시 -1)

### 예제
```
입력:
n = 4, k = 1
edges = [[0,1,10], [1,2,20], [0,2,50], [2,3,5]]
start = 0, end = 3
출력: 15

설명: 0→1(10) + 1→2(무료) + 2→3(5) = 15
```

### 힌트
- 상태를 (노드, 남은 무료 간선 수)로 확장하세요.""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= n <= 10^4, 0 <= k <= 10, edges <= 10^5",
        "hints": ["(node, remaining_free)를 상태로", "3차원 DP: dist[node][used_free]"],
        "solution_template": "def solution(n: int, k: int, edges: list, start: int, end: int) -> int:\n    pass",
        "reference_solution": '''def solution(n: int, k: int, edges: list, start: int, end: int) -> int:
    import heapq
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # dist[node][free_used] = 최소 비용
    INF = float('inf')
    dist = [[INF] * (k + 1) for _ in range(n)]
    dist[start][0] = 0

    # (cost, node, free_used)
    pq = [(0, start, 0)]

    while pq:
        cost, node, free_used = heapq.heappop(pq)

        if node == end:
            return cost

        if cost > dist[node][free_used]:
            continue

        for next_node, weight in graph[node]:
            # 정상 이동
            new_cost = cost + weight
            if new_cost < dist[next_node][free_used]:
                dist[next_node][free_used] = new_cost
                heapq.heappush(pq, (new_cost, next_node, free_used))

            # 무료 간선 사용
            if free_used < k:
                if cost < dist[next_node][free_used + 1]:
                    dist[next_node][free_used + 1] = cost
                    heapq.heappush(pq, (cost, next_node, free_used + 1))

    return min(dist[end]) if min(dist[end]) != INF else -1''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["dijkstra", "state-space"],
        "pattern_explanation": "상태 공간 다익스트라는 기본 다익스트라에 추가 상태를 결합한 것입니다. (노드, 사용한 특수 능력 수)를 하나의 상태로 보고, 각 상태에 대해 최단 거리를 관리합니다. 무료 통행권, 텔레포트 횟수 등 제한된 특수 능력 문제에 자주 사용됩니다.",
        "approach_hint": "상태 확장 다익스트라",
        "time_complexity_hint": "O((V*K) * log(V*K))",
        "space_complexity_hint": "O(V*K)",
        "test_cases": [
            {"input": "4\n1\n[[0,1,10], [1,2,20], [0,2,50], [2,3,5]]\n0\n3", "output": "15", "is_sample": True},
            {"input": "3\n0\n[[0,1,10], [1,2,20]]\n0\n2", "output": "30", "is_sample": True},
            {"input": "3\n2\n[[0,1,100], [1,2,100]]\n0\n2", "output": "0", "is_sample": False},
        ]
    },
    {
        "title": "트리 DP - 서브트리 합",
        "description": """루트 트리가 주어집니다. 각 노드에서 서브트리의 모든 노드 값의 합을 구하세요.

### 입력
- 노드 값 리스트 `values` (0-indexed)
- 간선 리스트 `edges`: [[parent, child], ...]
- 루트 노드 `root`

### 출력
- 각 노드의 서브트리 합 리스트

### 예제
```
입력:
values = [1, 2, 3, 4, 5]
edges = [[0,1], [0,2], [1,3], [1,4]]
root = 0

      0(1)
     / \\
   1(2)  2(3)
   / \\
 3(4) 4(5)

출력: [15, 11, 3, 4, 5]

설명:
- 노드 0: 1+2+3+4+5 = 15
- 노드 1: 2+4+5 = 11
- 노드 2: 3
- 노드 3: 4
- 노드 4: 5
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n <= 10^5",
        "hints": ["DFS로 리프부터 루트 방향으로 계산", "자식들의 서브트리 합 + 자신의 값"],
        "solution_template": "def solution(values: list, edges: list, root: int) -> list:\n    pass",
        "reference_solution": '''def solution(values: list, edges: list, root: int) -> list:
    from collections import defaultdict

    n = len(values)
    children = defaultdict(list)

    for parent, child in edges:
        children[parent].append(child)

    subtree_sum = [0] * n

    def dfs(node):
        subtree_sum[node] = values[node]
        for child in children[node]:
            dfs(child)
            subtree_sum[node] += subtree_sum[child]

    dfs(root)
    return subtree_sum''',
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-dp", "dfs"],
        "pattern_explanation": "트리 DP는 DFS를 활용해 트리의 각 노드에서 서브트리 정보를 계산하는 기법입니다. 보통 리프 노드부터 시작해 루트 방향으로 값을 합쳐 올립니다. 서브트리 합, 서브트리 크기, 서브트리 내 최댓값 등 다양한 문제에 적용됩니다.",
        "approach_hint": "후위 순회 DFS로 서브트리 합 계산",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]\n[[0,1], [0,2], [1,3], [1,4]]\n0", "output": "[15, 11, 3, 4, 5]", "is_sample": True},
            {"input": "[10]\n[]\n0", "output": "[10]", "is_sample": True},
            {"input": "[1, 1, 1]\n[[0,1], [1,2]]\n0", "output": "[3, 2, 1]", "is_sample": False},
        ]
    },
    {
        "title": "비트마스크 DP - 외판원 문제",
        "description": """N개의 도시와 도시 간 이동 비용이 주어집니다. 0번 도시에서 시작해 모든 도시를 정확히 한 번씩 방문하고 다시 0번으로 돌아오는 최소 비용을 구하세요.

### 입력
- 비용 행렬 `cost[i][j]`: i에서 j로 이동 비용 (0이면 이동 불가)

### 출력
- 최소 비용 (불가능하면 -1)

### 예제
```
입력:
cost = [
  [0, 10, 15, 20],
  [10, 0, 35, 25],
  [15, 35, 0, 30],
  [20, 25, 30, 0]
]
출력: 80

경로: 0 → 1 → 3 → 2 → 0 = 10 + 25 + 30 + 15 = 80
```

### 힌트
- 방문한 도시 집합을 비트마스크로 표현하세요.""",
        "difficulty": "hard",
        "category": "dynamic_programming",
        "constraints": "2 <= n <= 15",
        "hints": ["dp[mask][i] = mask 도시들을 방문하고 i에 있을 때 최소 비용", "마지막에 0으로 돌아가는 비용 추가"],
        "solution_template": "def solution(cost: list) -> int:\n    pass",
        "reference_solution": '''def solution(cost: list) -> int:
    n = len(cost)
    INF = float('inf')

    # dp[mask][i] = mask에 포함된 도시들을 방문하고 현재 i에 있을 때 최소 비용
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 0번 도시에서 시작

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue

            for v in range(n):
                if mask & (1 << v):  # 이미 방문
                    continue
                if cost[u][v] == 0 and u != v:  # 이동 불가
                    continue

                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + cost[u][v])

    # 모든 도시 방문 후 0으로 돌아가기
    full_mask = (1 << n) - 1
    result = INF
    for u in range(n):
        if dp[full_mask][u] != INF and cost[u][0] != 0:
            result = min(result, dp[full_mask][u] + cost[u][0])

    return result if result != INF else -1''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bitmask-dp", "tsp"],
        "pattern_explanation": "비트마스크 DP는 집합의 상태를 비트로 표현하여 DP를 수행합니다. N개의 원소 집합은 2^N 가지 상태로 표현됩니다. 외판원 문제(TSP)에서 mask는 방문한 도시 집합을, i는 현재 위치를 나타내어 dp[mask][i]로 상태를 정의합니다.",
        "approach_hint": "비트마스크로 방문 상태 표현 + DP",
        "time_complexity_hint": "O(n^2 * 2^n)",
        "space_complexity_hint": "O(n * 2^n)",
        "test_cases": [
            {"input": "[[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]", "output": "80", "is_sample": True},
            {"input": "[[0, 1], [1, 0]]", "output": "2", "is_sample": True},
            {"input": "[[0, 1, 0], [1, 0, 1], [0, 1, 0]]", "output": "-1", "is_sample": False},
        ]
    },
    {
        "title": "2D Fenwick Tree",
        "description": """2차원 배열에서 점 업데이트와 구간 합 쿼리를 처리하세요.

### 입력
- 2D 배열 크기 `n x m`
- 쿼리 리스트:
  - `[1, r, c, delta]`: matrix[r][c] += delta
  - `[2, r1, c1, r2, c2]`: (r1,c1)부터 (r2,c2)까지 합

### 출력
- 타입 2 쿼리 결과 리스트

### 예제
```
입력:
n = 3, m = 3
queries = [[1, 0, 0, 5], [1, 1, 1, 3], [2, 0, 0, 1, 1]]
출력: [8]
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n, m <= 1000, q <= 10^5",
        "hints": ["1D Fenwick Tree를 2차원으로 확장", "update와 query 모두 이중 루프"],
        "solution_template": "def solution(n: int, m: int, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(n: int, m: int, queries: list) -> list:
    tree = [[0] * (m + 1) for _ in range(n + 1)]

    def update(r, c, delta):
        i = r + 1
        while i <= n:
            j = c + 1
            while j <= m:
                tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(r, c):
        total = 0
        i = r + 1
        while i > 0:
            j = c + 1
            while j > 0:
                total += tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total

    def range_sum(r1, c1, r2, c2):
        return (prefix_sum(r2, c2)
                - prefix_sum(r1-1, c2)
                - prefix_sum(r2, c1-1)
                + prefix_sum(r1-1, c1-1))

    result = []
    for q in queries:
        if q[0] == 1:
            update(q[1], q[2], q[3])
        else:
            result.append(range_sum(q[1], q[2], q[3], q[4]))
    return result''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["fenwick-tree", "2d-data-structure"],
        "pattern_explanation": "2D Fenwick Tree(Binary Indexed Tree)는 1D BIT를 2차원으로 확장한 것입니다. 점 업데이트 O(log n * log m), 구간 합 O(log n * log m)에 처리합니다. 2D 누적합보다 메모리 효율적이고 업데이트가 가능합니다.",
        "approach_hint": "2차원 BIT - 행과 열 모두 이진 인덱싱",
        "time_complexity_hint": "O(q * log n * log m)",
        "space_complexity_hint": "O(n * m)",
        "test_cases": [
            {"input": "3\n3\n[[1, 0, 0, 5], [1, 1, 1, 3], [2, 0, 0, 1, 1]]", "output": "[8]", "is_sample": True},
            {"input": "2\n2\n[[1, 0, 0, 1], [1, 1, 1, 1], [2, 0, 0, 1, 1]]", "output": "[2]", "is_sample": True},
        ]
    },
    {
        "title": "Binary Lifting LCA",
        "description": """트리가 주어졌을 때, 두 노드의 최소 공통 조상(LCA)을 구하세요.

### 입력
- 부모 배열 `parent` (parent[i]는 노드 i의 부모, 루트는 -1)
- 쿼리 리스트 `queries`: [[u, v], ...]

### 출력
- 각 쿼리에 대한 LCA

### 예제
```
입력:
parent = [-1, 0, 0, 1, 1, 2, 2]
queries = [[3, 5], [4, 6], [3, 6]]

      0
     / \\
    1   2
   / \\ / \\
  3  4 5  6

출력: [0, 0, 0]
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n <= 10^5, 1 <= q <= 10^5",
        "hints": ["Binary Lifting: 2^k번째 조상 미리 계산", "깊이 맞춘 후 같이 올라가기"],
        "solution_template": "def solution(parent: list, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(parent: list, queries: list) -> list:
    n = len(parent)
    LOG = 17  # log2(10^5) ≈ 17

    # 깊이 계산
    depth = [0] * n
    root = -1
    for i in range(n):
        if parent[i] == -1:
            root = i
            continue
        d = 0
        curr = i
        while parent[curr] != -1:
            curr = parent[curr]
            d += 1
        depth[i] = d

    # Binary Lifting 테이블
    up = [[-1] * n for _ in range(LOG)]
    up[0] = parent[:]
    up[0] = [p if p != -1 else i for i, p in enumerate(parent)]

    for k in range(1, LOG):
        for i in range(n):
            up[k][i] = up[k-1][up[k-1][i]]

    def lca(u, v):
        # 깊이 맞추기
        if depth[u] < depth[v]:
            u, v = v, u

        diff = depth[u] - depth[v]
        for k in range(LOG):
            if diff & (1 << k):
                u = up[k][u]

        if u == v:
            return u

        # 같이 올라가기
        for k in range(LOG - 1, -1, -1):
            if up[k][u] != up[k][v]:
                u = up[k][u]
                v = up[k][v]

        return up[0][u]

    return [lca(u, v) for u, v in queries]''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["binary-lifting", "lca"],
        "pattern_explanation": "Binary Lifting은 2^k번째 조상을 미리 계산해두어 조상 쿼리를 O(log n)에 처리하는 기법입니다. up[k][v]는 노드 v의 2^k번째 조상을 의미합니다. LCA를 구할 때는 먼저 깊이를 맞추고, 두 노드가 만날 때까지 같이 올라갑니다.",
        "approach_hint": "Binary Lifting 전처리 + LCA 쿼리",
        "time_complexity_hint": "O((n + q) * log n)",
        "space_complexity_hint": "O(n * log n)",
        "test_cases": [
            {"input": "[-1, 0, 0, 1, 1, 2, 2]\n[[3, 5], [4, 6], [3, 6]]", "output": "[0, 0, 0]", "is_sample": True},
            {"input": "[-1, 0, 1, 2]\n[[3, 1], [2, 0]]", "output": "[1, 0]", "is_sample": True},
        ]
    },
    {
        "title": "Mo's Algorithm",
        "description": """배열이 주어지고, 여러 구간에 대해 서로 다른 원소의 개수를 구하세요.

### 입력
- 정수 배열 `arr`
- 쿼리 리스트 `queries`: [[l, r], ...]

### 출력
- 각 쿼리에 대한 구간 내 서로 다른 원소 수

### 예제
```
입력:
arr = [1, 2, 1, 3, 1, 2, 1]
queries = [[0, 2], [1, 4], [2, 6]]
출력: [2, 3, 3]

설명:
- [0,2]: {1, 2, 1} → 2가지
- [1,4]: {2, 1, 3, 1} → 3가지
- [2,6]: {1, 3, 1, 2, 1} → 3가지
```

### 힌트
- 쿼리를 재정렬하여 포인터 이동을 최소화하세요.""",
        "difficulty": "hard",
        "category": "array",
        "constraints": "1 <= n <= 10^5, 1 <= q <= 10^5",
        "hints": ["블록 크기 sqrt(n)으로 쿼리 정렬", "현재 구간에서 원소 추가/제거"],
        "solution_template": "def solution(arr: list, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(arr: list, queries: list) -> list:
    import math

    n = len(arr)
    q = len(queries)
    block_size = max(1, int(math.sqrt(n)))

    # 쿼리에 인덱스 추가 후 정렬
    indexed_queries = [(l, r, i) for i, (l, r) in enumerate(queries)]
    indexed_queries.sort(key=lambda x: (x[0] // block_size, x[1]))

    count = {}
    distinct = 0
    cur_l, cur_r = 0, -1
    result = [0] * q

    def add(idx):
        nonlocal distinct
        val = arr[idx]
        count[val] = count.get(val, 0) + 1
        if count[val] == 1:
            distinct += 1

    def remove(idx):
        nonlocal distinct
        val = arr[idx]
        count[val] -= 1
        if count[val] == 0:
            distinct -= 1

    for l, r, i in indexed_queries:
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        while cur_l < l:
            remove(cur_l)
            cur_l += 1

        result[i] = distinct

    return result''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["mos-algorithm", "sqrt-decomposition"],
        "pattern_explanation": "Mo's Algorithm은 오프라인 구간 쿼리를 O((N + Q) * sqrt(N))에 처리합니다. 쿼리를 블록 단위로 정렬하여 투 포인터 이동을 최소화합니다. 구간에 원소 추가/제거가 O(1)일 때 효과적입니다. 구간 내 distinct count, 구간 빈도수 등에 사용됩니다.",
        "approach_hint": "sqrt 분할 정렬 + 투 포인터",
        "time_complexity_hint": "O((N + Q) * sqrt(N))",
        "space_complexity_hint": "O(N)",
        "test_cases": [
            {"input": "[1, 2, 1, 3, 1, 2, 1]\n[[0, 2], [1, 4], [2, 6]]", "output": "[2, 3, 3]", "is_sample": True},
            {"input": "[1, 1, 1]\n[[0, 2]]", "output": "[1]", "is_sample": True},
        ]
    },
    {
        "title": "Heavy-Light Decomposition",
        "description": """트리에서 경로 쿼리를 처리하세요. 두 노드 사이 경로의 최댓값을 구합니다.

### 입력
- 노드 값 `values`
- 간선 리스트 `edges`
- 루트 `root`
- 쿼리 리스트 `queries`: [[u, v], ...]

### 출력
- 각 쿼리에 대한 경로 최댓값

### 예제
```
입력:
values = [1, 5, 2, 3, 4]
edges = [[0,1], [0,2], [1,3], [1,4]]
root = 0
queries = [[3, 4], [2, 4]]
출력: [5, 5]
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n <= 10^5, 1 <= q <= 10^5",
        "hints": ["Heavy edge로 체인 분할", "각 체인에 세그먼트 트리 적용"],
        "solution_template": "def solution(values: list, edges: list, root: int, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(values: list, edges: list, root: int, queries: list) -> list:
    from collections import defaultdict

    n = len(values)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    parent = [-1] * n
    depth = [0] * n
    subtree_size = [1] * n
    heavy = [-1] * n

    # DFS1: 서브트리 크기, 부모, 깊이, heavy child 계산
    def dfs1(v, p, d):
        parent[v] = p
        depth[v] = d
        max_size = 0
        for u in adj[v]:
            if u != p:
                dfs1(u, v, d + 1)
                subtree_size[v] += subtree_size[u]
                if subtree_size[u] > max_size:
                    max_size = subtree_size[u]
                    heavy[v] = u

    dfs1(root, -1, 0)

    # DFS2: 체인 분할
    chain_head = [0] * n
    pos = [0] * n
    pos_to_val = [0] * n
    cur_pos = [0]

    def dfs2(v, h):
        chain_head[v] = h
        pos[v] = cur_pos[0]
        pos_to_val[cur_pos[0]] = values[v]
        cur_pos[0] += 1

        if heavy[v] != -1:
            dfs2(heavy[v], h)

        for u in adj[v]:
            if u != parent[v] and u != heavy[v]:
                dfs2(u, u)

    dfs2(root, root)

    # 세그먼트 트리 (최댓값)
    tree = [0] * (4 * n)

    def build(node, start, end):
        if start == end:
            tree[node] = pos_to_val[start]
        else:
            mid = (start + end) // 2
            build(2*node, start, mid)
            build(2*node+1, mid+1, end)
            tree[node] = max(tree[2*node], tree[2*node+1])

    def query_tree(node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        return max(query_tree(2*node, start, mid, l, r),
                   query_tree(2*node+1, mid+1, end, l, r))

    build(1, 0, n-1)

    def path_max(u, v):
        result = 0
        while chain_head[u] != chain_head[v]:
            if depth[chain_head[u]] < depth[chain_head[v]]:
                u, v = v, u
            result = max(result, query_tree(1, 0, n-1, pos[chain_head[u]], pos[u]))
            u = parent[chain_head[u]]

        if depth[u] > depth[v]:
            u, v = v, u
        result = max(result, query_tree(1, 0, n-1, pos[u], pos[v]))
        return result

    return [path_max(u, v) for u, v in queries]''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hld", "segment-tree"],
        "pattern_explanation": "Heavy-Light Decomposition은 트리를 체인으로 분할하여 경로 쿼리를 효율적으로 처리합니다. 각 노드에서 가장 큰 서브트리로 가는 간선을 heavy edge로 지정하고, heavy edge들이 연결된 체인에 세그먼트 트리를 적용합니다. 경로 쿼리는 O(log^2 N)에 처리됩니다.",
        "approach_hint": "HLD로 체인 분할 + 세그먼트 트리",
        "time_complexity_hint": "O(n + q * log^2 n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 5, 2, 3, 4]\n[[0,1], [0,2], [1,3], [1,4]]\n0\n[[3, 4], [2, 4]]", "output": "[5, 5]", "is_sample": True},
            {"input": "[10, 20, 30]\n[[0,1], [1,2]]\n0\n[[0, 2]]", "output": "[30]", "is_sample": True},
        ]
    },

    # ============== ICPC/IOI 수준 (6문제) ==============
    {
        "title": "Convex Hull Trick",
        "description": """DP 점화식: dp[i] = min(dp[j] + b[j] * a[i]) for j < i

a 배열은 순증가, b 배열은 순감소일 때, 이를 O(n)에 계산하세요.

### 입력
- 배열 `a` (순증가)
- 배열 `b` (순감소)
- 초기값 dp[0] = 0

### 출력
- dp 배열

### 예제
```
입력:
a = [1, 2, 3, 4]
b = [4, 3, 2, 1]
출력: [0, 4, 6, 6]

설명:
dp[1] = dp[0] + b[0]*a[1] = 0 + 4*1 = 4
dp[2] = min(dp[0]+b[0]*a[2], dp[1]+b[1]*a[2]) = min(12, 10) = ... 최솟값 선택
```""",
        "difficulty": "hard",
        "category": "dynamic_programming",
        "constraints": "1 <= n <= 10^5",
        "hints": ["직선 y = b[j]*x + dp[j] 형태로 변환", "Convex Hull로 최적 직선 유지"],
        "solution_template": "def solution(a: list, b: list) -> list:\n    pass",
        "reference_solution": '''def solution(a: list, b: list) -> list:
    n = len(a)
    dp = [0] * n

    # 직선: y = m*x + c where m = b[j], c = dp[j]
    lines = []  # (slope, intercept)

    def bad(l1, l2, l3):
        # l2가 필요없는지 체크 (l1-l3 교점이 l1-l2 교점 왼쪽에 있으면 l2 불필요)
        m1, c1 = l1
        m2, c2 = l2
        m3, c3 = l3
        # (c2-c1)/(m1-m2) >= (c3-c1)/(m1-m3)
        return (c2 - c1) * (m1 - m3) >= (c3 - c1) * (m1 - m2)

    ptr = 0

    for i in range(n):
        if i > 0:
            # x = a[i]에서 최소 y 찾기
            while ptr + 1 < len(lines):
                m1, c1 = lines[ptr]
                m2, c2 = lines[ptr + 1]
                if m1 * a[i] + c1 > m2 * a[i] + c2:
                    ptr += 1
                else:
                    break

            if lines:
                m, c = lines[ptr]
                dp[i] = m * a[i] + c

        # 새 직선 추가: y = b[i]*x + dp[i]
        new_line = (b[i], dp[i])
        while len(lines) >= 2 and bad(lines[-2], lines[-1], new_line):
            lines.pop()
            if ptr >= len(lines):
                ptr = max(0, len(lines) - 1)
        lines.append(new_line)

    return dp''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["convex-hull-trick", "dp-optimization"],
        "pattern_explanation": "Convex Hull Trick은 dp[i] = min(dp[j] + b[j]*a[i]) 형태의 DP를 최적화합니다. 각 j를 기울기 b[j], y절편 dp[j]인 직선으로 보고, x=a[i]에서 최소 y값을 주는 직선을 찾습니다. 기울기가 정렬되어 있으면 O(n)에 해결 가능합니다.",
        "approach_hint": "직선들의 하한 껍질(Lower Hull) 유지",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4]\n[4, 3, 2, 1]", "output": "[0, 4, 6, 6]", "is_sample": True},
            {"input": "[1, 2]\n[2, 1]", "output": "[0, 2]", "is_sample": True},
        ]
    },
    {
        "title": "Centroid Decomposition",
        "description": """트리에서 거리가 K인 노드 쌍의 개수를 구하세요.

### 입력
- 간선 리스트 `edges` (가중치 1)
- 거리 `k`

### 출력
- 거리가 정확히 K인 노드 쌍의 수

### 예제
```
입력:
edges = [[0,1], [1,2], [1,3], [3,4]]
k = 2

    0
    |
    1
   / \\
  2   3
      |
      4

출력: 4
쌍: (0,2), (0,3), (2,3), (1,4)
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= n <= 10^4, 1 <= k <= n",
        "hints": ["Centroid에서 모든 서브트리까지 거리 계산", "같은 서브트리 내 쌍은 제외"],
        "solution_template": "def solution(edges: list, k: int) -> int:\n    pass",
        "reference_solution": '''def solution(edges: list, k: int) -> int:
    from collections import defaultdict

    n = len(edges) + 1
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    removed = [False] * n
    result = [0]

    def get_subtree_size(v, parent):
        size = 1
        for u in adj[v]:
            if u != parent and not removed[u]:
                size += get_subtree_size(u, v)
        return size

    def get_centroid(v, parent, tree_size):
        size = 1
        is_centroid = True
        for u in adj[v]:
            if u != parent and not removed[u]:
                subtree = get_subtree_size(u, v)
                size += subtree
                if subtree > tree_size // 2:
                    is_centroid = False
        if tree_size - size > tree_size // 2:
            is_centroid = False
        if is_centroid:
            return v
        for u in adj[v]:
            if u != parent and not removed[u]:
                c = get_centroid(u, v, tree_size)
                if c != -1:
                    return c
        return -1

    def get_distances(v, parent, dist, distances):
        distances.append(dist)
        for u in adj[v]:
            if u != parent and not removed[u]:
                get_distances(u, v, dist + 1, distances)

    def count_pairs(distances):
        count = defaultdict(int)
        pairs = 0
        for d in distances:
            if k - d in count:
                pairs += count[k - d]
            count[d] += 1
        return pairs

    def solve(v):
        tree_size = get_subtree_size(v, -1)
        centroid = get_centroid(v, -1, tree_size)

        removed[centroid] = True

        # 전체 거리에서 쌍 계산
        all_distances = [0]  # centroid 자신
        for u in adj[centroid]:
            if not removed[u]:
                get_distances(u, centroid, 1, all_distances)

        result[0] += count_pairs(all_distances)

        # 같은 서브트리 내 쌍 제거
        for u in adj[centroid]:
            if not removed[u]:
                sub_distances = [0]
                get_distances(u, centroid, 1, sub_distances)
                result[0] -= count_pairs(sub_distances)

        # 서브트리 재귀
        for u in adj[centroid]:
            if not removed[u]:
                solve(u)

    if n > 0:
        solve(0)
    return result[0]''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["centroid-decomposition", "divide-and-conquer"],
        "pattern_explanation": "Centroid Decomposition은 트리를 중심(centroid)을 기준으로 분할하여 경로 문제를 해결합니다. 각 centroid에서 모든 서브트리까지의 거리를 계산하고, 조건을 만족하는 경로를 셉니다. 같은 서브트리 내 경로는 중복이므로 제거합니다. 재귀적으로 O(n log n)에 해결합니다.",
        "approach_hint": "Centroid에서 거리 계산 + 같은 서브트리 쌍 제외",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[[0,1], [1,2], [1,3], [3,4]]\n2", "output": "4", "is_sample": True},
            {"input": "[[0,1], [1,2]]\n1", "output": "2", "is_sample": True},
            {"input": "[[0,1]]\n1", "output": "1", "is_sample": False},
        ]
    },
    {
        "title": "Maximum Flow (Dinic)",
        "description": """방향 그래프에서 소스 s에서 싱크 t로의 최대 유량을 구하세요.

### 입력
- 정점 수 `n`
- 간선 리스트 `edges`: [[u, v, capacity], ...]
- 소스 `s`, 싱크 `t`

### 출력
- 최대 유량

### 예제
```
입력:
n = 4
edges = [[0,1,10], [0,2,10], [1,2,2], [1,3,4], [2,3,10]]
s = 0, t = 3
출력: 14
```""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= n <= 500, edges <= 5000",
        "hints": ["BFS로 레벨 그래프 구성", "DFS로 blocking flow 찾기"],
        "solution_template": "def solution(n: int, edges: list, s: int, t: int) -> int:\n    pass",
        "reference_solution": '''def solution(n: int, edges: list, s: int, t: int) -> int:
    from collections import deque, defaultdict

    # 인접 리스트 (u -> [(v, capacity, rev_idx), ...])
    graph = defaultdict(list)

    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for u, v, cap in edges:
        add_edge(u, v, cap)

    def bfs():
        level = [-1] * n
        level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, cap, _ in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level[t] >= 0, level

    def dfs(u, t, f, level, iter_):
        if u == t:
            return f
        while iter_[u] < len(graph[u]):
            v, cap, rev = graph[u][iter_[u]]
            if cap > 0 and level[v] == level[u] + 1:
                d = dfs(v, t, min(f, cap), level, iter_)
                if d > 0:
                    graph[u][iter_[u]][1] -= d
                    graph[v][rev][1] += d
                    return d
            iter_[u] += 1
        return 0

    flow = 0
    while True:
        found, level = bfs()
        if not found:
            break
        iter_ = [0] * n
        while True:
            f = dfs(s, t, float('inf'), level, iter_)
            if f == 0:
                break
            flow += f

    return flow''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["max-flow", "dinic"],
        "pattern_explanation": "Dinic 알고리즘은 최대 유량을 O(V^2 * E)에 계산합니다. BFS로 레벨 그래프를 만들고, DFS로 블로킹 플로우를 찾습니다. 유량 네트워크에서 소스에서 싱크로 보낼 수 있는 최대 유량을 구합니다. 이분 매칭, 최소 컷 문제에도 사용됩니다.",
        "approach_hint": "Dinic: BFS 레벨링 + DFS 블로킹 플로우",
        "time_complexity_hint": "O(V^2 * E)",
        "space_complexity_hint": "O(V + E)",
        "test_cases": [
            {"input": "4\n[[0,1,10], [0,2,10], [1,2,2], [1,3,4], [2,3,10]]\n0\n3", "output": "14", "is_sample": True},
            {"input": "2\n[[0,1,5]]\n0\n1", "output": "5", "is_sample": True},
        ]
    },
    {
        "title": "Suffix Array with LCP",
        "description": """문자열의 접미사 배열(Suffix Array)과 LCP 배열을 구하세요.

### 입력
- 문자열 `s`

### 출력
- 접미사 배열 (인덱스 리스트)
- LCP 배열

### 예제
```
입력: "banana"
접미사들:
0: banana
1: anana
2: nana
3: ana
4: na
5: a

정렬 후 순서: 5, 3, 1, 0, 4, 2
LCP: [1, 3, 0, 0, 2]

출력:
suffix_array = [5, 3, 1, 0, 4, 2]
lcp = [1, 3, 0, 0, 2]
```""",
        "difficulty": "hard",
        "category": "string",
        "constraints": "1 <= len(s) <= 10^5",
        "hints": ["O(n log n) 접미사 배열 구성", "Kasai 알고리즘으로 LCP 계산"],
        "solution_template": "def solution(s: str) -> tuple:\n    pass",
        "reference_solution": '''def solution(s: str) -> tuple:
    n = len(s)
    if n == 0:
        return [], []

    # Suffix Array O(n log n)
    suffix_array = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n
    k = 1

    while k < n:
        def compare(i, j):
            if rank[i] != rank[j]:
                return rank[i] - rank[j]
            ri = rank[i + k] if i + k < n else -1
            rj = rank[j + k] if j + k < n else -1
            return ri - rj

        from functools import cmp_to_key
        suffix_array.sort(key=cmp_to_key(compare))

        tmp[suffix_array[0]] = 0
        for i in range(1, n):
            tmp[suffix_array[i]] = tmp[suffix_array[i-1]]
            if compare(suffix_array[i-1], suffix_array[i]) < 0:
                tmp[suffix_array[i]] += 1

        rank = tmp[:]
        k *= 2

    # LCP Array (Kasai)
    lcp = [0] * (n - 1)
    inv_suffix = [0] * n
    for i, sa in enumerate(suffix_array):
        inv_suffix[sa] = i

    k = 0
    for i in range(n):
        if inv_suffix[i] == 0:
            k = 0
            continue
        j = suffix_array[inv_suffix[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[inv_suffix[i] - 1] = k
        if k > 0:
            k -= 1

    return suffix_array, lcp''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["suffix-array", "lcp"],
        "pattern_explanation": "Suffix Array는 문자열의 모든 접미사를 사전순 정렬한 배열입니다. O(n log n)에 구성할 수 있습니다. LCP(Longest Common Prefix) 배열은 인접한 접미사 쌍의 공통 접두사 길이를 저장합니다. Kasai 알고리즘으로 O(n)에 계산합니다. 문자열 검색, 패턴 매칭에 활용됩니다.",
        "approach_hint": "Doubling 기법 Suffix Array + Kasai LCP",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "banana", "output": "([5, 3, 1, 0, 4, 2], [1, 3, 0, 0, 2])", "is_sample": True},
            {"input": "aaa", "output": "([2, 1, 0], [1, 2])", "is_sample": True},
        ]
    },
    {
        "title": "Line Sweep + Segment Tree",
        "description": """2D 평면에 N개의 직사각형이 주어집니다. 직사각형들로 덮인 총 면적을 구하세요 (겹치는 부분은 한 번만).

### 입력
- 직사각형 리스트 `rectangles`: [[x1, y1, x2, y2], ...]

### 출력
- 덮인 총 면적

### 예제
```
입력:
rectangles = [[0,0,2,2], [1,1,3,3]]
출력: 7

설명: 2x2 + 2x2 - 겹침 1x1 = 7
```""",
        "difficulty": "hard",
        "category": "simulation",
        "constraints": "1 <= n <= 10^4, 0 <= 좌표 <= 10^6",
        "hints": ["x 좌표로 이벤트 정렬", "y 구간을 세그먼트 트리로 관리"],
        "solution_template": "def solution(rectangles: list) -> int:\n    pass",
        "reference_solution": '''def solution(rectangles: list) -> int:
    if not rectangles:
        return 0

    # 좌표 압축
    y_coords = set()
    events = []  # (x, type, y1, y2) - type: 1=시작, -1=끝

    for x1, y1, x2, y2 in rectangles:
        y_coords.add(y1)
        y_coords.add(y2)
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))

    y_coords = sorted(y_coords)
    y_to_idx = {y: i for i, y in enumerate(y_coords)}
    m = len(y_coords) - 1

    # 세그먼트 트리: count, covered_length
    count = [0] * (4 * m)
    covered = [0] * (4 * m)

    def update(node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            count[node] += val
        else:
            mid = (start + end) // 2
            update(2*node, start, mid, l, r, val)
            update(2*node+1, mid+1, end, l, r, val)

        if count[node] > 0:
            covered[node] = y_coords[end + 1] - y_coords[start]
        elif start == end:
            covered[node] = 0
        else:
            covered[node] = covered[2*node] + covered[2*node+1]

    events.sort()
    total_area = 0
    prev_x = events[0][0]

    for x, typ, y1, y2 in events:
        total_area += covered[1] * (x - prev_x)
        idx1, idx2 = y_to_idx[y1], y_to_idx[y2]
        update(1, 0, m - 1, idx1, idx2 - 1, typ)
        prev_x = x

    return total_area''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["line-sweep", "segment-tree", "coordinate-compression"],
        "pattern_explanation": "Line Sweep은 이벤트를 정렬하여 순차 처리하는 기법입니다. 직사각형 면적 문제에서는 x좌표로 스윕하면서, y구간의 덮인 길이를 세그먼트 트리로 관리합니다. 좌표 압축으로 좌표 범위를 줄이고, 각 x 구간에서 덮인 y길이 × x너비로 면적을 누적합니다.",
        "approach_hint": "x 스윕 + y 구간 세그먼트 트리 + 좌표 압축",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[[0,0,2,2], [1,1,3,3]]", "output": "7", "is_sample": True},
            {"input": "[[0,0,1,1]]", "output": "1", "is_sample": True},
            {"input": "[[0,0,2,2], [0,0,2,2]]", "output": "4", "is_sample": False},
        ]
    },
    {
        "title": "FFT 기반 다항식 곱셈",
        "description": """두 다항식의 곱을 FFT를 사용하여 계산하세요.

### 입력
- 다항식 계수 배열 `a`: a[i]는 x^i의 계수
- 다항식 계수 배열 `b`

### 출력
- 곱셈 결과 계수 배열

### 예제
```
입력:
a = [1, 2, 1]  # 1 + 2x + x^2 = (1+x)^2
b = [1, 1]     # 1 + x
출력: [1, 3, 3, 1]  # (1+x)^3 = 1 + 3x + 3x^2 + x^3
```""",
        "difficulty": "hard",
        "category": "math",
        "constraints": "len(a), len(b) <= 10^5",
        "hints": ["FFT로 O(n log n)에 다항식 곱셈", "결과는 len(a) + len(b) - 1 크기"],
        "solution_template": "def solution(a: list, b: list) -> list:\n    pass",
        "reference_solution": '''def solution(a: list, b: list) -> list:
    import cmath

    def fft(x, inverse=False):
        n = len(x)
        if n == 1:
            return x

        even = fft(x[0::2], inverse)
        odd = fft(x[1::2], inverse)

        angle = 2 * cmath.pi / n * (-1 if inverse else 1)
        w = 1
        wn = cmath.exp(1j * angle)

        result = [0] * n
        for i in range(n // 2):
            result[i] = even[i] + w * odd[i]
            result[i + n // 2] = even[i] - w * odd[i]
            w *= wn

        return result

    # 결과 크기
    n = len(a) + len(b) - 1
    # 2의 거듭제곱으로 확장
    size = 1
    while size < n:
        size *= 2

    # 패딩
    fa = [complex(x) for x in a] + [0] * (size - len(a))
    fb = [complex(x) for x in b] + [0] * (size - len(b))

    # FFT
    fa = fft(fa)
    fb = fft(fb)

    # 곱셈
    fc = [fa[i] * fb[i] for i in range(size)]

    # Inverse FFT
    result = fft(fc, inverse=True)
    result = [round(x.real / size) for x in result]

    # 불필요한 0 제거
    while len(result) > n:
        result.pop()

    return result''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["fft", "polynomial-multiplication"],
        "pattern_explanation": "FFT(Fast Fourier Transform)는 다항식 곱셈을 O(n log n)에 수행합니다. 다항식을 점 값 표현으로 변환(FFT)하고, 점별로 곱한 후, 역변환(IFFT)합니다. 큰 수 곱셈, 문자열 매칭(와일드카드) 등에 활용됩니다.",
        "approach_hint": "FFT → 점별 곱셈 → IFFT",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 1]\n[1, 1]", "output": "[1, 3, 3, 1]", "is_sample": True},
            {"input": "[1, 0, 1]\n[1, 1]", "output": "[1, 1, 1, 1]", "is_sample": True},
        ]
    },

    # ============== 삼성 SW 역량테스트 (6문제) ==============
    {
        "title": "다중 상태 BFS - 열쇠와 문",
        "description": """N×M 격자에서 시작점 S에서 도착점 E까지 최단 거리를 구하세요. 문(A-F)은 해당 열쇠(a-f)가 있어야 통과 가능합니다.

### 입력
- 격자 `grid`:
  - `.`: 빈 칸
  - `#`: 벽
  - `S`: 시작점
  - `E`: 도착점
  - `a-f`: 열쇠
  - `A-F`: 문

### 출력
- 최단 거리 (도달 불가시 -1)

### 예제
```
입력:
grid = [
  "S.a",
  ".A.",
  "..E"
]
출력: 5
```""",
        "difficulty": "medium",
        "category": "graph",
        "constraints": "1 <= N, M <= 100",
        "hints": ["상태: (x, y, 가진 열쇠들)", "열쇠는 비트마스크로 표현"],
        "solution_template": "def solution(grid: list) -> int:\n    pass",
        "reference_solution": '''def solution(grid: list) -> int:
    from collections import deque

    n, m = len(grid), len(grid[0])

    # 시작점, 도착점 찾기
    start = end = None
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'E':
                end = (i, j)

    # BFS: (x, y, keys)
    # keys: 6비트 (a=0, b=1, ..., f=5)
    visited = set()
    queue = deque([(start[0], start[1], 0, 0)])  # x, y, keys, dist
    visited.add((start[0], start[1], 0))

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    while queue:
        x, y, keys, dist = queue.popleft()

        if (x, y) == end:
            return dist

        for d in range(4):
            nx, ny = x + dx[d], y + dy[d]

            if 0 <= nx < n and 0 <= ny < m:
                cell = grid[nx][ny]

                if cell == '#':
                    continue

                new_keys = keys

                # 열쇠
                if 'a' <= cell <= 'f':
                    new_keys |= (1 << (ord(cell) - ord('a')))

                # 문
                if 'A' <= cell <= 'F':
                    door_bit = 1 << (ord(cell) - ord('A'))
                    if not (keys & door_bit):
                        continue

                if (nx, ny, new_keys) not in visited:
                    visited.add((nx, ny, new_keys))
                    queue.append((nx, ny, new_keys, dist + 1))

    return -1''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bfs", "state-space", "bitmask"],
        "pattern_explanation": "다중 상태 BFS는 위치 외에 추가 상태(열쇠 보유 현황)를 함께 관리합니다. 열쇠 6개를 비트마스크로 표현하면 최대 64가지 상태가 됩니다. visited는 (x, y, keys) 3차원으로 관리하여 같은 위치라도 다른 열쇠 상태면 재방문합니다.",
        "approach_hint": "3차원 BFS: (x, y, 열쇠 비트마스크)",
        "time_complexity_hint": "O(N * M * 2^K)",
        "space_complexity_hint": "O(N * M * 2^K)",
        "test_cases": [
            {"input": '["S.a", ".A.", "..E"]', "output": "5", "is_sample": True},
            {"input": '["S.E"]', "output": "2", "is_sample": True},
            {"input": '["SAE"]', "output": "-1", "is_sample": False},
        ]
    },
    {
        "title": "격자 시뮬레이션 - 모래성",
        "description": """N×M 격자에서 모래가 쌓여 있습니다. 각 칸의 모래는 인접 8방향 빈 칸 수만큼 깎입니다. 모래가 0이 되면 빈 칸이 됩니다. 안정화될 때까지 시뮬레이션하세요.

### 입력
- 모래 격자 `sand` (0은 빈 칸)

### 출력
- 안정화까지 걸린 턴 수

### 예제
```
입력:
sand = [
  [0, 0, 0, 0, 0],
  [0, 9, 9, 9, 0],
  [0, 9, 9, 9, 0],
  [0, 9, 9, 9, 0],
  [0, 0, 0, 0, 0]
]
출력: 3
```""",
        "difficulty": "medium",
        "category": "simulation",
        "constraints": "1 <= N, M <= 500",
        "hints": ["가장자리만 변화 가능", "큐로 변화 가능한 칸만 관리"],
        "solution_template": "def solution(sand: list) -> int:\n    pass",
        "reference_solution": '''def solution(sand: list) -> int:
    from collections import deque

    n, m = len(sand), len(sand[0])
    grid = [row[:] for row in sand]

    # 8방향
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    def count_empty(x, y):
        count = 0
        for d in range(8):
            nx, ny = x + dx[d], y + dy[d]
            if 0 <= nx < n and 0 <= ny < m:
                if grid[nx][ny] == 0:
                    count += 1
            else:
                count += 1  # 경계 밖도 빈 칸 취급
        return count

    turns = 0

    while True:
        # 이번 턴에 변할 칸들 찾기
        to_remove = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] > 0:
                    empty = count_empty(i, j)
                    if grid[i][j] <= empty:
                        to_remove.append((i, j))

        if not to_remove:
            break

        for x, y in to_remove:
            grid[x][y] = 0

        turns += 1

    return turns''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["simulation", "grid"],
        "pattern_explanation": "격자 시뮬레이션은 각 턴마다 규칙에 따라 격자 상태를 업데이트합니다. 동시 업데이트가 필요한 경우, 먼저 변화할 위치를 모두 찾고 한꺼번에 적용합니다. 변화가 없을 때까지 반복하여 안정화 상태에 도달합니다.",
        "approach_hint": "동시 업데이트 시뮬레이션",
        "time_complexity_hint": "O(T * N * M)",
        "space_complexity_hint": "O(N * M)",
        "test_cases": [
            {"input": "[[0,0,0,0,0],[0,9,9,9,0],[0,9,9,9,0],[0,9,9,9,0],[0,0,0,0,0]]", "output": "3", "is_sample": True},
            {"input": "[[0,0,0],[0,1,0],[0,0,0]]", "output": "1", "is_sample": True},
        ]
    },
    {
        "title": "완전탐색 + 가지치기 - N-Queens 변형",
        "description": """N×N 체스판에 N개의 퀸과 M개의 비숍을 배치하세요. 퀸은 가로, 세로, 대각선으로 공격, 비숍은 대각선으로만 공격합니다. 서로 공격하지 않는 배치 수를 구하세요.

### 입력
- 체스판 크기 `n`
- 비숍 수 `m`

### 출력
- 가능한 배치 수

### 예제
```
입력: n = 4, m = 0
출력: 2 (기본 N-Queens)

입력: n = 4, m = 1
출력: ? (퀸 4개 + 비숍 1개)
```""",
        "difficulty": "hard",
        "category": "backtracking",
        "constraints": "1 <= n <= 10, 0 <= m <= 4",
        "hints": ["퀸 먼저 배치 (N-Queens)", "남은 칸에 비숍 배치"],
        "solution_template": "def solution(n: int, m: int) -> int:\n    pass",
        "reference_solution": '''def solution(n: int, m: int) -> int:
    count = [0]

    # 퀸 공격 범위
    col = [False] * n
    diag1 = [False] * (2 * n)
    diag2 = [False] * (2 * n)

    def place_queens(row, queen_positions):
        if row == n:
            # 퀸 배치 완료, 비숍 배치
            place_bishops(queen_positions, 0, 0, [])
            return

        for c in range(n):
            if col[c] or diag1[row - c + n] or diag2[row + c]:
                continue
            col[c] = diag1[row - c + n] = diag2[row + c] = True
            queen_positions.append((row, c))
            place_queens(row + 1, queen_positions)
            queen_positions.pop()
            col[c] = diag1[row - c + n] = diag2[row + c] = False

    def place_bishops(queens, idx, placed, bishops):
        if placed == m:
            count[0] += 1
            return

        queen_set = set(queens)
        queen_diags = set()
        for qr, qc in queens:
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                r, c = qr, qc
                while 0 <= r < n and 0 <= c < n:
                    queen_diags.add((r, c))
                    r += dr
                    c += dc

        # 비숍 간 대각선 충돌
        bishop_diags = set()
        for br, bc in bishops:
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                r, c = br, bc
                while 0 <= r < n and 0 <= c < n:
                    bishop_diags.add((r, c))
                    r += dr
                    c += dc

        for pos in range(idx, n * n):
            r, c = pos // n, pos % n
            if (r, c) in queen_set:
                continue
            if (r, c) in queen_diags:
                continue
            if (r, c) in bishop_diags:
                continue

            bishops.append((r, c))
            place_bishops(queens, pos + 1, placed + 1, bishops)
            bishops.pop()

    place_queens(0, [])
    return count[0]''',
        "time_limit_ms": 10000,
        "memory_limit_mb": 256,
        "pattern_ids": ["backtracking", "n-queens", "pruning"],
        "pattern_explanation": "백트래킹은 가능한 모든 경우를 탐색하되, 불가능한 분기는 조기에 가지치기합니다. N-Queens에서는 열, 두 대각선의 사용 여부를 추적하여 충돌을 O(1)에 검사합니다. 복합 문제에서는 단계별로 제약 조건을 검사하여 탐색 공간을 줄입니다.",
        "approach_hint": "퀸 배치 백트래킹 + 비숍 배치 백트래킹",
        "time_complexity_hint": "O(N! * C(N*N, M))",
        "space_complexity_hint": "O(N)",
        "test_cases": [
            {"input": "4\n0", "output": "2", "is_sample": True},
            {"input": "1\n0", "output": "1", "is_sample": True},
        ]
    },
    {
        "title": "다중 시작점 BFS",
        "description": """N×M 격자에 여러 불(F)이 있습니다. 불은 매 초마다 상하좌우로 퍼집니다. 사람(P)이 출구(E)까지 도달할 수 있는 최소 시간을 구하세요.

### 입력
- 격자 `grid`:
  - `.`: 빈 칸
  - `#`: 벽
  - `F`: 불
  - `P`: 사람 (1명)
  - `E`: 출구

### 출력
- 최소 시간 (도달 불가시 -1)

### 예제
```
입력:
grid = [
  "P..E",
  ".F..",
  "...."
]
출력: 3
```""",
        "difficulty": "medium",
        "category": "graph",
        "constraints": "1 <= N, M <= 1000",
        "hints": ["불 먼저 BFS로 각 칸 도달 시간 계산", "사람 BFS에서 불보다 빨리 도달해야 함"],
        "solution_template": "def solution(grid: list) -> int:\n    pass",
        "reference_solution": '''def solution(grid: list) -> int:
    from collections import deque

    n, m = len(grid), len(grid[0])
    INF = float('inf')

    fire_time = [[INF] * m for _ in range(n)]
    person_start = None
    exit_pos = None
    fire_starts = []

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 'P':
                person_start = (i, j)
            elif grid[i][j] == 'E':
                exit_pos = (i, j)
            elif grid[i][j] == 'F':
                fire_starts.append((i, j))
                fire_time[i][j] = 0

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    # 불 BFS
    queue = deque(fire_starts)
    for x, y in fire_starts:
        fire_time[x][y] = 0

    while queue:
        x, y = queue.popleft()
        for d in range(4):
            nx, ny = x + dx[d], y + dy[d]
            if 0 <= nx < n and 0 <= ny < m:
                if grid[nx][ny] != '#' and fire_time[nx][ny] == INF:
                    fire_time[nx][ny] = fire_time[x][y] + 1
                    queue.append((nx, ny))

    # 사람 BFS
    visited = [[False] * m for _ in range(n)]
    queue = deque([(person_start[0], person_start[1], 0)])
    visited[person_start[0]][person_start[1]] = True

    while queue:
        x, y, t = queue.popleft()

        if (x, y) == exit_pos:
            return t

        for d in range(4):
            nx, ny = x + dx[d], y + dy[d]
            if 0 <= nx < n and 0 <= ny < m:
                if not visited[nx][ny] and grid[nx][ny] != '#':
                    # 사람이 도착할 때(t+1) 불이 아직 안 왔어야 함
                    if t + 1 < fire_time[nx][ny]:
                        visited[nx][ny] = True
                        queue.append((nx, ny, t + 1))

    return -1''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bfs", "multi-source"],
        "pattern_explanation": "다중 시작점 BFS는 여러 시작점에서 동시에 BFS를 수행합니다. 불의 전파처럼 여러 소스에서 퍼지는 문제에 사용됩니다. 먼저 장애물(불)의 각 칸 도달 시간을 계산하고, 이동자가 장애물보다 빨리 도착해야 하는 조건으로 탐색합니다.",
        "approach_hint": "불 BFS 먼저 → 사람 BFS에서 조건 확인",
        "time_complexity_hint": "O(N * M)",
        "space_complexity_hint": "O(N * M)",
        "test_cases": [
            {"input": '["P..E", ".F..", "...."]', "output": "3", "is_sample": True},
            {"input": '["PE"]', "output": "1", "is_sample": True},
            {"input": '["PFE"]', "output": "-1", "is_sample": False},
        ]
    },
    {
        "title": "상태 기계 시뮬레이션 - 로봇 청소기",
        "description": """로봇 청소기가 N×M 격자를 청소합니다. 로봇은 현재 방향을 유지하며 다음 규칙으로 동작합니다:

1. 현재 칸이 청소되지 않았으면 청소
2. 4방향 중 청소되지 않은 빈 칸이 있으면:
   - 반시계 방향으로 90도 회전
   - 앞이 청소되지 않은 빈 칸이면 전진
3. 4방향 모두 청소됐거나 벽이면:
   - 후진 가능하면 후진 (방향 유지)
   - 후진 불가면 종료

### 입력
- 격자 `grid` (0: 빈 칸, 1: 벽)
- 시작 위치 `(r, c)`, 방향 `d` (0:북, 1:동, 2:남, 3:서)

### 출력
- 청소한 칸 수

### 예제
```
입력:
grid = [
  [1, 1, 1, 1, 1],
  [1, 0, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 0, 0, 0, 1],
  [1, 1, 1, 1, 1]
]
r = 1, c = 1, d = 0
출력: 7
```""",
        "difficulty": "hard",
        "category": "simulation",
        "constraints": "3 <= N, M <= 50",
        "hints": ["상태: (위치, 방향)", "반시계 회전: (d + 3) % 4"],
        "solution_template": "def solution(grid: list, r: int, c: int, d: int) -> int:\n    pass",
        "reference_solution": '''def solution(grid: list, r: int, c: int, d: int) -> int:
    n, m = len(grid), len(grid[0])
    cleaned = [[False] * m for _ in range(n)]

    # 방향: 0=북, 1=동, 2=남, 3=서
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]

    count = 0

    while True:
        # 1. 현재 칸 청소
        if not cleaned[r][c]:
            cleaned[r][c] = True
            count += 1

        # 2. 4방향 확인
        found = False
        for _ in range(4):
            d = (d + 3) % 4  # 반시계 회전
            nr, nc = r + dx[d], c + dy[d]
            if 0 <= nr < n and 0 <= nc < m:
                if grid[nr][nc] == 0 and not cleaned[nr][nc]:
                    r, c = nr, nc
                    found = True
                    break

        if not found:
            # 3. 후진 시도
            back_d = (d + 2) % 4
            br, bc = r + dx[back_d], c + dy[back_d]
            if 0 <= br < n and 0 <= bc < m and grid[br][bc] == 0:
                r, c = br, bc
            else:
                break  # 종료

    return count''',
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["simulation", "state-machine"],
        "pattern_explanation": "상태 기계 시뮬레이션은 현재 상태와 규칙에 따라 다음 상태로 전이하는 과정을 구현합니다. 로봇 청소기 문제에서 상태는 (위치, 방향)이며, 주어진 규칙을 정확히 순서대로 적용해야 합니다. 방향 전환, 이동 조건을 명확히 구현하는 것이 핵심입니다.",
        "approach_hint": "규칙 순서대로 정확히 구현",
        "time_complexity_hint": "O(N * M)",
        "space_complexity_hint": "O(N * M)",
        "test_cases": [
            {"input": "[[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]\n1\n1\n0", "output": "7", "is_sample": True},
            {"input": "[[1,1,1],[1,0,1],[1,1,1]]\n1\n1\n0", "output": "1", "is_sample": True},
        ]
    },
    {
        "title": "백트래킹 - 스도쿠 풀기",
        "description": """9×9 스도쿠 퍼즐을 풀어 완성하세요. 빈 칸은 0으로 표시됩니다.

### 입력
- 9×9 스도쿠 격자 `board` (0은 빈 칸)

### 출력
- 완성된 스도쿠 격자 (불가능하면 원본 그대로)

### 예제
```
입력:
board = [
  [5,3,0,0,7,0,0,0,0],
  [6,0,0,1,9,5,0,0,0],
  [0,9,8,0,0,0,0,6,0],
  ...
]
출력: 완성된 스도쿠
```""",
        "difficulty": "hard",
        "category": "backtracking",
        "constraints": "9x9 스도쿠",
        "hints": ["행, 열, 3x3 박스의 숫자 사용 여부 추적", "빈 칸에 1-9 시도"],
        "solution_template": "def solution(board: list) -> list:\n    pass",
        "reference_solution": '''def solution(board: list) -> list:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty = []

    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                num = board[i][j]
                rows[i].add(num)
                cols[j].add(num)
                boxes[(i // 3) * 3 + j // 3].add(num)
            else:
                empty.append((i, j))

    def solve(idx):
        if idx == len(empty):
            return True

        i, j = empty[idx]
        box_idx = (i // 3) * 3 + j // 3

        for num in range(1, 10):
            if num in rows[i] or num in cols[j] or num in boxes[box_idx]:
                continue

            board[i][j] = num
            rows[i].add(num)
            cols[j].add(num)
            boxes[box_idx].add(num)

            if solve(idx + 1):
                return True

            board[i][j] = 0
            rows[i].remove(num)
            cols[j].remove(num)
            boxes[box_idx].remove(num)

        return False

    solve(0)
    return board''',
        "time_limit_ms": 5000,
        "memory_limit_mb": 256,
        "pattern_ids": ["backtracking", "constraint-satisfaction"],
        "pattern_explanation": "스도쿠는 제약 충족 문제(CSP)의 대표적 예입니다. 각 빈 칸에 1-9를 시도하되, 행/열/박스 제약을 set으로 O(1) 검사합니다. 불가능한 분기는 조기에 백트랙하여 탐색 공간을 줄입니다.",
        "approach_hint": "행/열/박스 제약 + 백트래킹",
        "time_complexity_hint": "O(9^빈칸수) 최악, 실제로는 가지치기로 빠름",
        "space_complexity_hint": "O(81)",
        "test_cases": [
            {"input": "[[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]", "output": "[[5,3,4,6,7,8,9,1,2],[6,7,2,1,9,5,3,4,8],[1,9,8,3,4,2,5,6,7],[8,5,9,7,6,1,4,2,3],[4,2,6,8,5,3,7,9,1],[7,1,3,9,2,4,8,5,6],[9,6,1,5,3,7,2,8,4],[2,8,7,4,1,9,6,3,5],[3,4,5,2,8,6,1,7,9]]", "is_sample": True},
        ]
    },

    # ============== 카카오/네이버 코테 (5문제) ==============
    {
        "title": "효율적 문자열 파싱",
        "description": """중첩된 괄호 문자열을 파싱하여 각 레벨의 내용을 추출하세요.

### 입력
- 문자열 `s`: 중첩된 괄호 포함

### 출력
- 레벨별 내용 리스트

### 예제
```
입력: "a(b(c)d)e"
출력: {0: "ae", 1: "bd", 2: "c"}

설명:
레벨 0: a, e
레벨 1: b, d
레벨 2: c
```""",
        "difficulty": "medium",
        "category": "string",
        "constraints": "1 <= len(s) <= 10^4",
        "hints": ["스택으로 현재 깊이 추적", "각 깊이별 문자 수집"],
        "solution_template": "def solution(s: str) -> dict:\n    pass",
        "reference_solution": '''def solution(s: str) -> dict:
    from collections import defaultdict

    result = defaultdict(list)
    depth = 0

    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        else:
            result[depth].append(c)

    return {k: ''.join(v) for k, v in result.items()}''',
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["string-parsing", "stack"],
        "pattern_explanation": "괄호 문자열 파싱은 스택 또는 깊이 카운터로 현재 중첩 레벨을 추적합니다. '('면 깊이 증가, ')'면 감소하며, 각 깊이에서 만나는 문자를 수집합니다. 카카오 코테에서 자주 출제되는 문자열 처리 유형입니다.",
        "approach_hint": "깊이 카운터 + 레벨별 문자 수집",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": '"a(b(c)d)e"', "output": "{0: 'ae', 1: 'bd', 2: 'c'}", "is_sample": True},
            {"input": '"(a)"', "output": "{1: 'a'}", "is_sample": True},
        ]
    },
    {
        "title": "Floyd-Warshall 경로 카운팅",
        "description": """N개의 도시와 도로가 있습니다. 두 도시 사이의 최단 경로의 개수를 구하세요.

### 입력
- 도시 수 `n`
- 도로 리스트 `roads`: [[u, v, w], ...] (양방향)
- 쿼리 리스트 `queries`: [[start, end], ...]

### 출력
- 각 쿼리에 대한 최단 경로 개수 (모듈로 10^9+7)

### 예제
```
입력:
n = 4
roads = [[0,1,1], [1,2,1], [0,2,2], [2,3,1]]
queries = [[0, 3]]
출력: [2]

설명: 0→1→2→3 (길이 3), 0→2→3 (길이 3) → 2개
```""",
        "difficulty": "medium",
        "category": "graph",
        "constraints": "1 <= n <= 200",
        "hints": ["Floyd-Warshall로 최단 거리 계산", "동시에 최단 경로 개수도 업데이트"],
        "solution_template": "def solution(n: int, roads: list, queries: list) -> list:\n    pass",
        "reference_solution": '''def solution(n: int, roads: list, queries: list) -> list:
    MOD = 10**9 + 7
    INF = float('inf')

    dist = [[INF] * n for _ in range(n)]
    count = [[0] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        count[i][i] = 1

    for u, v, w in roads:
        if w < dist[u][v]:
            dist[u][v] = dist[v][u] = w
            count[u][v] = count[v][u] = 1
        elif w == dist[u][v]:
            count[u][v] = (count[u][v] + 1) % MOD
            count[v][u] = count[u][v]

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                new_dist = dist[i][k] + dist[k][j]
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist
                    count[i][j] = (count[i][k] * count[k][j]) % MOD
                elif new_dist == dist[i][j] and dist[i][k] != INF and dist[k][j] != INF:
                    count[i][j] = (count[i][j] + count[i][k] * count[k][j]) % MOD

    result = []
    for start, end in queries:
        if dist[start][end] == INF:
            result.append(0)
        else:
            result.append(count[start][end])

    return result''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["floyd-warshall", "path-counting"],
        "pattern_explanation": "Floyd-Warshall은 모든 쌍 최단 경로를 O(n^3)에 계산합니다. 경로 개수도 함께 계산하려면: 더 짧은 경로 발견 시 개수 갱신, 같은 거리의 경로 발견 시 개수 누적합니다. count[i][j] = count[i][k] * count[k][j]로 곱셈 원리를 적용합니다.",
        "approach_hint": "Floyd-Warshall + 경로 개수 동시 계산",
        "time_complexity_hint": "O(n^3)",
        "space_complexity_hint": "O(n^2)",
        "test_cases": [
            {"input": "4\n[[0,1,1], [1,2,1], [0,2,2], [2,3,1]]\n[[0, 3]]", "output": "[2]", "is_sample": True},
            {"input": "3\n[[0,1,1], [1,2,1]]\n[[0, 2]]", "output": "[1]", "is_sample": True},
        ]
    },
    {
        "title": "좌표 압축",
        "description": """2D 평면에 N개의 직사각형이 있습니다. 직사각형들이 덮는 서로 다른 영역의 개수를 구하세요.

### 입력
- 직사각형 리스트 `rectangles`: [[x1, y1, x2, y2], ...]

### 출력
- 덮인 영역들을 각각 카운트 (겹침에 따라)

### 예제
```
입력:
rectangles = [[0,0,2,2], [1,1,3,3]]
출력:
영역별 직사각형 수: {1: 6, 2: 1}
(1개로 덮인 영역 6칸, 2개로 덮인 영역 1칸)
```""",
        "difficulty": "medium",
        "category": "simulation",
        "constraints": "1 <= n <= 100, 좌표 <= 10^6",
        "hints": ["x, y 좌표 각각 압축", "압축된 격자에서 카운트"],
        "solution_template": "def solution(rectangles: list) -> dict:\n    pass",
        "reference_solution": '''def solution(rectangles: list) -> dict:
    from collections import defaultdict

    # 좌표 수집
    x_coords = set()
    y_coords = set()

    for x1, y1, x2, y2 in rectangles:
        x_coords.add(x1)
        x_coords.add(x2)
        y_coords.add(y1)
        y_coords.add(y2)

    xs = sorted(x_coords)
    ys = sorted(y_coords)

    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}

    # 압축된 격자
    grid = [[0] * (len(ys) - 1) for _ in range(len(xs) - 1)]

    for x1, y1, x2, y2 in rectangles:
        for i in range(x_idx[x1], x_idx[x2]):
            for j in range(y_idx[y1], y_idx[y2]):
                grid[i][j] += 1

    # 결과 집계
    result = defaultdict(int)
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if grid[i][j] > 0:
                area = (xs[i+1] - xs[i]) * (ys[j+1] - ys[j])
                result[grid[i][j]] += area

    return dict(result)''',
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["coordinate-compression", "sweep"],
        "pattern_explanation": "좌표 압축은 큰 좌표 범위를 작은 인덱스로 변환합니다. 직사각형의 모든 x, y 좌표를 수집하고 정렬하여 인덱스를 부여합니다. 압축된 격자는 원래 좌표 간격을 유지하므로, 각 셀의 실제 면적을 계산하여 결과를 도출합니다.",
        "approach_hint": "좌표 압축 → 격자 카운트 → 면적 계산",
        "time_complexity_hint": "O(n^2 * m) where m = 압축된 좌표 수",
        "space_complexity_hint": "O(m^2)",
        "test_cases": [
            {"input": "[[0,0,2,2], [1,1,3,3]]", "output": "{1: 6, 2: 1}", "is_sample": True},
            {"input": "[[0,0,1,1]]", "output": "{1: 1}", "is_sample": True},
        ]
    },
    {
        "title": "이벤트 기반 스케줄링",
        "description": """N개의 작업이 있고, 각 작업은 시작 시간, 처리 시간, 우선순위가 있습니다. 한 번에 하나의 작업만 처리 가능할 때, 각 작업의 완료 시간을 구하세요.

규칙:
- 현재 시간에 시작 가능한 작업 중 우선순위가 가장 높은(숫자가 작은) 것을 선택
- 우선순위가 같으면 먼저 도착한 것 선택
- 처리 중인 작업은 중단 불가 (Non-preemptive)

### 입력
- 작업 리스트 `jobs`: [[arrive, process, priority], ...]

### 출력
- 각 작업의 완료 시간 리스트

### 예제
```
입력:
jobs = [[0, 3, 2], [1, 2, 1], [2, 1, 3]]
출력: [3, 5, 6]

설명:
t=0: job0 시작
t=3: job0 완료, job1 시작 (더 높은 우선순위)
t=5: job1 완료, job2 시작
t=6: job2 완료
```""",
        "difficulty": "medium",
        "category": "simulation",
        "constraints": "1 <= n <= 10^4",
        "hints": ["우선순위 큐로 대기 작업 관리", "시간순 이벤트 처리"],
        "solution_template": "def solution(jobs: list) -> list:\n    pass",
        "reference_solution": '''def solution(jobs: list) -> list:
    import heapq

    n = len(jobs)
    indexed_jobs = [(arrive, process, priority, i) for i, (arrive, process, priority) in enumerate(jobs)]
    indexed_jobs.sort()  # 도착 시간순

    result = [0] * n
    time = 0
    job_idx = 0
    pq = []  # (priority, arrive, process, original_idx)

    while job_idx < n or pq:
        # 현재 시간까지 도착한 작업 추가
        while job_idx < n and indexed_jobs[job_idx][0] <= time:
            arrive, process, priority, idx = indexed_jobs[job_idx]
            heapq.heappush(pq, (priority, arrive, process, idx))
            job_idx += 1

        if pq:
            priority, arrive, process, idx = heapq.heappop(pq)
            time += process
            result[idx] = time
        elif job_idx < n:
            # 대기 중인 작업 없으면 다음 작업 도착까지 점프
            time = indexed_jobs[job_idx][0]

    return result''',
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["priority-queue", "event-driven", "scheduling"],
        "pattern_explanation": "이벤트 기반 스케줄링은 시간순으로 이벤트를 처리합니다. 우선순위 큐로 대기 작업을 관리하고, 현재 시간에 선택 가능한 최적 작업을 O(log n)에 추출합니다. 카카오/네이버 코테에서 자주 출제되는 유형입니다.",
        "approach_hint": "도착 시간 정렬 + 우선순위 큐 스케줄링",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[[0, 3, 2], [1, 2, 1], [2, 1, 3]]", "output": "[3, 5, 6]", "is_sample": True},
            {"input": "[[0, 1, 1]]", "output": "[1]", "is_sample": True},
        ]
    },
    {
        "title": "해시 기반 문자열 그룹핑",
        "description": """문자열 리스트에서 k-gram이 같은 문자열들을 그룹핑하세요.

k-gram: 연속된 k개 문자의 집합 (순서 무시)

### 입력
- 문자열 리스트 `words`
- k-gram 크기 `k`

### 출력
- 같은 k-gram을 가진 문자열 그룹들

### 예제
```
입력:
words = ["abc", "bca", "cab", "xyz", "zyx"]
k = 2

2-gram:
"abc": {ab, bc}
"bca": {bc, ca}
"cab": {ca, ab}
"xyz": {xy, yz}
"zyx": {zy, yx}

그룹: [["abc", "cab"], ["bca"], ["xyz"], ["zyx"]]
```""",
        "difficulty": "medium",
        "category": "hash_table",
        "constraints": "1 <= n <= 10^4, 1 <= k <= len(word)",
        "hints": ["k-gram 집합을 frozenset으로", "frozenset을 해시 키로 사용"],
        "solution_template": "def solution(words: list, k: int) -> list:\n    pass",
        "reference_solution": '''def solution(words: list, k: int) -> list:
    from collections import defaultdict

    def get_kgrams(word, k):
        if len(word) < k:
            return frozenset()
        return frozenset(word[i:i+k] for i in range(len(word) - k + 1))

    groups = defaultdict(list)

    for word in words:
        key = get_kgrams(word, k)
        groups[key].append(word)

    return list(groups.values())''',
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["hash-grouping", "n-gram"],
        "pattern_explanation": "k-gram 그룹핑은 문자열의 부분 문자열 패턴을 기반으로 분류합니다. k-gram 집합을 frozenset으로 만들어 해시 가능하게 하고, 이를 딕셔너리 키로 사용하여 O(n*m) (m=문자열 길이)에 그룹핑합니다. 유사 문서 탐지, 표절 검사에 활용됩니다.",
        "approach_hint": "k-gram frozenset을 해시 키로 그룹핑",
        "time_complexity_hint": "O(n * m)",
        "space_complexity_hint": "O(n * m)",
        "test_cases": [
            {"input": '["abc", "bca", "cab", "xyz", "zyx"]\n2', "output": '[["abc", "cab"], ["bca"], ["xyz"], ["zyx"]]', "is_sample": True},
            {"input": '["aa", "aa"]\n1', "output": '[["aa", "aa"]]', "is_sample": True},
        ]
    },
]


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()

    added = 0
    skipped = 0

    for problem in PROBLEMS:
        cursor.execute("SELECT id FROM problems WHERE title = ?", (problem["title"],))
        if cursor.fetchone():
            print(f"  [건너뜀] 이미 존재: {problem['title']}")
            skipped += 1
            continue

        problem_id = uuid.uuid4().hex

        cursor.execute("""
            INSERT INTO problems (
                id, title, description, difficulty, category, constraints,
                hints, solution_template, reference_solution,
                time_limit_ms, memory_limit_mb, is_published,
                pattern_ids, pattern_explanation, approach_hint, time_complexity_hint, space_complexity_hint,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            problem_id,
            problem["title"],
            problem["description"],
            problem["difficulty"],
            problem["category"],
            problem["constraints"],
            json.dumps(problem["hints"], ensure_ascii=False),
            problem["solution_template"],
            problem["reference_solution"],
            problem["time_limit_ms"],
            problem["memory_limit_mb"],
            1,
            json.dumps(problem.get("pattern_ids", []), ensure_ascii=False),
            problem.get("pattern_explanation", ""),
            problem.get("approach_hint", ""),
            problem.get("time_complexity_hint", ""),
            problem.get("space_complexity_hint", ""),
            now,
            now
        ))

        for i, tc in enumerate(problem["test_cases"]):
            tc_id = uuid.uuid4().hex
            cursor.execute("""
                INSERT INTO test_cases (id, problem_id, input_data, expected_output, is_sample, "order", created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tc_id,
                problem_id,
                tc["input"],
                tc["output"],
                1 if tc.get("is_sample", False) else 0,
                i,
                now
            ))

        print(f"  [추가됨] {problem['title']} ({problem['difficulty']})")
        added += 1

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"대회 수준 알고리즘 문제 시딩 완료")
    print(f"  - 추가: {added}개")
    print(f"  - 건너뜀: {skipped}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("대회 수준 알고리즘 문제 시딩 시작...")
    print("  - 코드포스 Div2: 8문제")
    print("  - ICPC/IOI: 6문제")
    print("  - 삼성 SW 역량테스트: 6문제")
    print("  - 카카오/네이버 코테: 5문제")
    print()
    main()
