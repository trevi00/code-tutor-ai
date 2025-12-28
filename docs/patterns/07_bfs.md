# Pattern 07: BFS (너비 우선 탐색)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(V + E) |
| **공간복잡도** | O(V) |
| **선행 지식** | 큐, 그래프/트리 |

## 정의

**BFS (Breadth-First Search, 너비 우선 탐색)**는 시작점에서 가까운 노드부터 탐색하는 알고리즘입니다. **큐(Queue)**를 사용하며, **레벨 순서**로 탐색합니다.

## 핵심 아이디어

```
레벨 0:        1
             / \
레벨 1:     2   3
           / \   \
레벨 2:   4   5   6

탐색 순서: 1 → 2 → 3 → 4 → 5 → 6
```

### BFS vs DFS

| 특성 | BFS | DFS |
|------|-----|-----|
| 탐색 순서 | 레벨별 (가까운 곳 먼저) | 깊이별 (끝까지 먼저) |
| 자료구조 | 큐 (Queue) | 스택/재귀 |
| 최단 경로 | ✅ 보장 | ❌ 보장 안 됨 |
| 메모리 | O(너비) | O(깊이) |

---

## 템플릿 코드

### 템플릿 1: 트리 레벨 순회

```python
from collections import deque

def level_order(root: TreeNode) -> list:
    """
    트리 레벨 순회

    Time: O(n), Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

### 템플릿 2: 그래프 BFS (최단 경로)

```python
from collections import deque

def shortest_path(graph: dict, start: int, end: int) -> int:
    """
    무가중치 그래프 최단 경로

    Time: O(V + E), Space: O(V)
    """
    if start == end:
        return 0

    visited = {start}
    queue = deque([(start, 0)])  # (노드, 거리)

    while queue:
        node, distance = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return distance + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return -1  # 경로 없음
```

### 템플릿 3: 2D 그리드 BFS

```python
from collections import deque

def grid_bfs(grid: list, start: tuple, end: tuple) -> int:
    """
    2D 그리드 최단 경로

    Time: O(m*n), Space: O(m*n)
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    queue = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        r, c, dist = queue.popleft()

        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != '#' and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1
```

### 템플릿 4: 다중 시작점 BFS

```python
from collections import deque

def multi_source_bfs(grid: list) -> list:
    """
    여러 시작점에서 동시에 BFS (거리 계산)

    예: 모든 0에서 가장 가까운 1까지의 거리
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()

    # 모든 시작점 추가
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                queue.append((r, c))
            else:
                grid[r][c] = float('inf')

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] > grid[r][c] + 1):
                grid[nr][nc] = grid[r][c] + 1
                queue.append((nr, nc))

    return grid
```

---

## 예제 문제

### 문제 1: 이진 트리 레벨 순회 (Medium)

**문제 설명**
이진 트리를 레벨별로 순회하여 반환하세요.

**입력/출력 예시**
```
입력:     3
         / \
        9  20
           / \
          15  7
출력: [[3],[9,20],[15,7]]
```

**풀이**
```python
def level_order(root: TreeNode) -> list:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result
```

---

### 문제 2: 이진 트리 지그재그 순회 (Medium)

**문제 설명**
지그재그로 레벨 순회하세요 (홀수 레벨은 역순).

**입력/출력 예시**
```
입력:     3
         / \
        9  20
           / \
          15  7
출력: [[3],[20,9],[15,7]]
```

**풀이**
```python
def zigzag_level_order(root: TreeNode) -> list:
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level = deque()
        for _ in range(len(queue)):
            node = queue.popleft()

            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result
```

---

### 문제 3: 썩은 오렌지 (Medium)

**문제 설명**
2D 그리드에서 썩은 오렌지(2)가 인접한 신선한 오렌지(1)를 1분마다 썩게 합니다. 모든 오렌지가 썩는데 걸리는 시간을 구하세요.

**입력/출력 예시**
```
입력: [[2,1,1],[1,1,0],[0,1,1]]
출력: 4
```

**풀이**
```python
def oranges_rotting(grid: list) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # 초기 상태
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    minutes = 0

    while queue:
        r, c, minutes = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, minutes + 1))

    return minutes if fresh == 0 else -1
```

---

### 문제 4: 단어 사다리 (Hard)

**문제 설명**
시작 단어에서 끝 단어까지 한 글자씩 바꿔서 도달하는 최단 경로 길이를 구하세요.

**입력/출력 예시**
```
입력: beginWord = "hit", endWord = "cog"
      wordList = ["hot","dot","dog","lot","log","cog"]
출력: 5
설명: hit → hot → dot → dog → cog
```

**풀이**
```python
def ladder_length(beginWord: str, endWord: str, wordList: list) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])

    while queue:
        word, length = queue.popleft()

        if word == endWord:
            return length

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]

                if new_word in word_set:
                    word_set.remove(new_word)
                    queue.append((new_word, length + 1))

    return 0
```

---

## Editorial (풀이 전략)

### Step 1: BFS 사용 조건
- **최단 거리/경로** 필요
- **레벨별 처리** 필요
- **가까운 것부터** 탐색

### Step 2: 기본 구조
```python
queue = deque([start])
visited = {start}

while queue:
    node = queue.popleft()
    # 처리

    for neighbor in get_neighbors(node):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

### Step 3: 레벨 구분이 필요하면
```python
while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        # 같은 레벨 노드들
```

---

## 자주 하는 실수

### 1. visited 체크 시점
```python
# ❌ 꺼낼 때 체크 → 중복 방문
node = queue.popleft()
if node in visited:
    continue
visited.add(node)

# ✅ 넣을 때 체크
if neighbor not in visited:
    visited.add(neighbor)
    queue.append(neighbor)
```

### 2. 거리 추적 누락
```python
# ❌ 거리 정보 없음
queue.append(node)

# ✅ 거리 함께 저장
queue.append((node, distance + 1))
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 102 | Binary Tree Level Order | Medium |
| 103 | Zigzag Level Order | Medium |
| 200 | Number of Islands | Medium |
| 994 | Rotting Oranges | Medium |
| 127 | Word Ladder | Hard |
| 542 | 01 Matrix | Medium |
| 1091 | Shortest Path in Binary Matrix | Medium |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1260 | DFS와 BFS | Silver 2 | 기본 필수 |
| 2178 | 미로 탐색 | Silver 1 | 최단거리 기본 |
| 7576 | 토마토 | Gold 5 | 다중 시작점 필수 |
| 7569 | 토마토 (3D) | Gold 5 | 3차원 BFS |
| 2206 | 벽 부수고 이동하기 | Gold 3 | 상태 BFS |
| 1697 | 숨바꼭질 | Silver 1 | 1차원 BFS |
| 13549 | 숨바꼭질 3 | Gold 5 | 0-1 BFS |
| 16234 | 인구 이동 | Gold 4 | 시뮬레이션 + BFS |
| 14502 | 연구소 | Gold 4 | 브루트포스 + BFS (삼성) |
| 16236 | 아기 상어 | Gold 3 | 시뮬레이션 + BFS (삼성) |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 게임 맵 최단거리 | Level 2 | 기본 BFS |
| 타겟 넘버 | Level 2 | BFS/DFS |
| 네트워크 | Level 3 | BFS/DFS |
| 단어 변환 | Level 3 | BFS (카카오) |
| 퍼즐 조각 채우기 | Level 3 | BFS + 구현 |

---

## 임베딩용 키워드

```
BFS, breadth first search, 너비 우선 탐색, level order, 레벨 순회,
shortest path, 최단 경로, queue, 큐, grid traversal, 그리드 탐색,
rotting oranges, word ladder, 단어 사다리, multi-source BFS,
토마토, 미로 탐색, 아기 상어, 연구소, 숨바꼭질, BOJ 7576, 삼성, 프로그래머스
```
