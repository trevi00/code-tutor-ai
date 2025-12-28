# Pattern 08: DFS (깊이 우선 탐색)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(V + E) |
| **공간복잡도** | O(V) - 재귀 스택 |
| **선행 지식** | 재귀, 그래프/트리 |

## 정의

**DFS (Depth-First Search, 깊이 우선 탐색)**는 한 경로를 끝까지 탐색한 후 다른 경로를 탐색하는 알고리즘입니다. **재귀** 또는 **스택**을 사용합니다.

## 핵심 아이디어

```
        1
       / \
      2   3
     / \
    4   5

DFS 순서: 1 → 2 → 4 → 5 → 3 (깊이 우선)
BFS 순서: 1 → 2 → 3 → 4 → 5 (너비 우선)
```

## 트리 순회 방식

```
     1
    / \
   2   3

전위 (Preorder):  루트 → 왼쪽 → 오른쪽  [1, 2, 3]
중위 (Inorder):   왼쪽 → 루트 → 오른쪽  [2, 1, 3]
후위 (Postorder): 왼쪽 → 오른쪽 → 루트  [2, 3, 1]
```

---

## 템플릿 코드

### 템플릿 1: 트리 DFS (재귀)

```python
def tree_dfs(root: TreeNode) -> list:
    """
    트리 전위 순회

    Time: O(n), Space: O(h)
    """
    result = []

    def dfs(node):
        if not node:
            return

        result.append(node.val)  # 전위: 처리 먼저
        dfs(node.left)
        dfs(node.right)
        # 중위: left → 처리 → right
        # 후위: left → right → 처리

    dfs(root)
    return result
```

### 템플릿 2: 트리 DFS (스택)

```python
def tree_dfs_iterative(root: TreeNode) -> list:
    """
    스택을 이용한 전위 순회

    Time: O(n), Space: O(h)
    """
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # 오른쪽 먼저 (스택이므로 나중에 처리됨)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result
```

### 템플릿 3: 그래프 DFS

```python
def graph_dfs(graph: dict, start: int) -> list:
    """
    그래프 DFS

    Time: O(V + E), Space: O(V)
    """
    visited = set()
    result = []

    def dfs(node):
        if node in visited:
            return

        visited.add(node)
        result.append(node)

        for neighbor in graph[node]:
            dfs(neighbor)

    dfs(start)
    return result
```

### 템플릿 4: 2D 그리드 DFS

```python
def grid_dfs(grid: list, r: int, c: int) -> int:
    """
    그리드 DFS (섬 크기 계산 예시)

    Time: O(m*n), Space: O(m*n)
    """
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        # 경계 체크 + 방문 체크
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == 0):
            return 0

        grid[r][c] = 0  # 방문 표시
        size = 1

        # 4방향 탐색
        size += dfs(r + 1, c)
        size += dfs(r - 1, c)
        size += dfs(r, c + 1)
        size += dfs(r, c - 1)

        return size

    return dfs(r, c)
```

### 템플릿 5: 경로 찾기 (백트래킹)

```python
def find_all_paths(graph: dict, start: int, end: int) -> list:
    """
    모든 경로 찾기 (백트래킹)

    Time: O(2^V * V), Space: O(V)
    """
    all_paths = []

    def dfs(node, path):
        if node == end:
            all_paths.append(path[:])
            return

        for neighbor in graph[node]:
            if neighbor not in path:  # 사이클 방지
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()  # 백트래킹

    dfs(start, [start])
    return all_paths
```

---

## 예제 문제

### 문제 1: 이진 트리 최대 깊이 (Easy)

**문제 설명**
이진 트리의 최대 깊이를 구하세요.

**입력/출력 예시**
```
입력:     3
         / \
        9  20
           / \
          15  7
출력: 3
```

**풀이**
```python
def max_depth(root: TreeNode) -> int:
    if not root:
        return 0

    return 1 + max(max_depth(root.left), max_depth(root.right))
```

---

### 문제 2: 섬의 개수 (Medium)

**문제 설명**
2D 그리드에서 '1'로 이루어진 섬의 개수를 구하세요.

**입력/출력 예시**
```
입력: [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
출력: 3
```

**풀이**
```python
def num_islands(grid: list) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # 방문 표시
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count
```

---

### 문제 3: 경로 합 (Easy)

**문제 설명**
루트에서 리프까지 경로 합이 targetSum인지 확인하세요.

**입력/출력 예시**
```
입력: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
출력: true (5 → 4 → 11 → 2 = 22)
```

**풀이**
```python
def has_path_sum(root: TreeNode, targetSum: int) -> bool:
    if not root:
        return False

    # 리프 노드 체크
    if not root.left and not root.right:
        return root.val == targetSum

    remaining = targetSum - root.val
    return (has_path_sum(root.left, remaining) or
            has_path_sum(root.right, remaining))
```

---

### 문제 4: 모든 경로 (Medium)

**문제 설명**
그래프에서 시작점부터 끝점까지 모든 경로를 찾으세요.

**풀이**
```python
def all_paths_source_target(graph: list) -> list:
    result = []
    target = len(graph) - 1

    def dfs(node, path):
        if node == target:
            result.append(path[:])
            return

        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()

    dfs(0, [0])
    return result
```

---

### 문제 5: 순열 (Medium)

**문제 설명**
주어진 숫자들의 모든 순열을 구하세요.

**입력/출력 예시**
```
입력: nums = [1,2,3]
출력: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**풀이**
```python
def permute(nums: list) -> list:
    result = []

    def dfs(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            path.append(remaining[i])
            dfs(path, remaining[:i] + remaining[i+1:])
            path.pop()

    dfs([], nums)
    return result
```

---

## Editorial (풀이 전략)

### Step 1: 재귀 vs 스택 선택
- **간단한 로직**: 재귀 (코드 짧음)
- **깊이 제한**: 스택 (스택 오버플로우 방지)

### Step 2: 방문 처리
```python
# 그래프: visited 셋 사용
visited = set()
if node in visited:
    return
visited.add(node)

# 그리드: 값 변경으로 표시
grid[r][c] = 0  # 또는 '#'
```

### Step 3: 백트래킹 필요 시
```python
path.append(choice)
dfs(next_state)
path.pop()  # 되돌리기
```

---

## 자주 하는 실수

### 1. 기저 조건 누락
```python
# ❌ 무한 재귀
def dfs(node):
    dfs(node.left)

# ✅ 기저 조건
def dfs(node):
    if not node:
        return
    dfs(node.left)
```

### 2. 방문 표시 안 함
```python
# ❌ 무한 루프 (그래프)
for neighbor in graph[node]:
    dfs(neighbor)

# ✅ 방문 체크
if neighbor not in visited:
    visited.add(neighbor)
    dfs(neighbor)
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 104 | Maximum Depth of Binary Tree | Easy |
| 112 | Path Sum | Easy |
| 200 | Number of Islands | Medium |
| 113 | Path Sum II | Medium |
| 46 | Permutations | Medium |
| 78 | Subsets | Medium |
| 79 | Word Search | Medium |
| 130 | Surrounded Regions | Medium |
| 695 | Max Area of Island | Medium |
| 437 | Path Sum III | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1260 | DFS와 BFS | Silver 2 | 기본 필수 |
| 2606 | 바이러스 | Silver 3 | 연결 요소 |
| 11724 | 연결 요소의 개수 | Silver 2 | 연결 요소 |
| 2667 | 단지번호붙이기 | Silver 1 | 그리드 DFS |
| 1987 | 알파벳 | Gold 4 | 백트래킹 |
| 10026 | 적록색약 | Gold 5 | 그리드 + 조건 |
| 14889 | 스타트와 링크 | Silver 1 | 백트래킹 (삼성) |
| 15649 | N과 M (1) | Silver 3 | 순열 기본 |
| 15650 | N과 M (2) | Silver 3 | 조합 기본 |
| 9466 | 텀 프로젝트 | Gold 3 | 사이클 탐지 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 타겟 넘버 | Level 2 | DFS 기본 |
| 네트워크 | Level 3 | 연결 요소 |
| 여행경로 | Level 3 | 오일러 경로 (카카오) |
| 단어 변환 | Level 3 | DFS/BFS |
| 양과 늑대 | Level 3 | 상태 DFS (카카오) |

---

## 임베딩용 키워드

```
DFS, depth first search, 깊이 우선 탐색, recursion, 재귀,
preorder, inorder, postorder, 전위, 중위, 후위 순회,
backtracking, 백트래킹, path finding, 경로 찾기,
tree traversal, 트리 순회, graph traversal, 그래프 탐색,
연결 요소, 단지번호, 바이러스, N과 M, BOJ, 삼성, 프로그래머스
```
