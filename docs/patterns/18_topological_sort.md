# Pattern 18: Topological Sort (위상 정렬)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(V + E) |
| **공간복잡도** | O(V) |
| **선행 지식** | 그래프, BFS/DFS, 진입차수 |

## 정의

**위상 정렬(Topological Sort)**은 **방향 비순환 그래프(DAG)**에서 모든 간선 (u, v)에 대해 u가 v보다 앞에 오도록 정점을 정렬하는 알고리즘입니다.

## 핵심 아이디어

```
선수과목 관계:
A → B (A를 들어야 B를 들을 수 있음)
A → C
B → D
C → D

가능한 수강 순서:
A → B → C → D ✅
A → C → B → D ✅
B → A → ... ❌ (A가 B의 선수과목)

핵심: 진입차수가 0인 노드부터 제거
```

## 두 가지 접근법

| 방법 | 구현 | 장점 |
|------|------|------|
| **Kahn's Algorithm (BFS)** | 진입차수 기반 | 사이클 검출 쉬움 |
| **DFS** | 후위 순회 역순 | 간단 |

---

## 템플릿 코드

### 템플릿 1: Kahn's Algorithm (BFS 기반)

```python
from collections import deque, defaultdict

def topological_sort_kahn(n: int, edges: list) -> list:
    """
    Kahn's Algorithm - BFS 기반 위상 정렬

    Time: O(V + E)
    Space: O(V)

    Returns: 정렬된 순서 (사이클 있으면 빈 리스트)
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    # 그래프 구성
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # 진입차수 0인 노드로 시작
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            in_degree[v] -= 1

            if in_degree[v] == 0:
                queue.append(v)

    # 모든 노드를 방문했는지 확인 (사이클 검사)
    if len(result) != n:
        return []  # 사이클 존재

    return result
```

### 템플릿 2: DFS 기반 위상 정렬

```python
def topological_sort_dfs(n: int, edges: list) -> list:
    """
    DFS 기반 위상 정렬

    Time: O(V + E)
    Space: O(V)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    visited = [0] * n  # 0: 미방문, 1: 방문 중, 2: 완료
    result = []
    has_cycle = False

    def dfs(u):
        nonlocal has_cycle

        if has_cycle:
            return

        visited[u] = 1  # 방문 중

        for v in graph[u]:
            if visited[v] == 1:  # 사이클!
                has_cycle = True
                return
            if visited[v] == 0:
                dfs(v)

        visited[u] = 2  # 완료
        result.append(u)

    for i in range(n):
        if visited[i] == 0:
            dfs(i)

    if has_cycle:
        return []

    return result[::-1]  # 역순
```

### 템플릿 3: 사이클 검출만

```python
def has_cycle(n: int, edges: list) -> bool:
    """
    DAG인지 확인 (사이클 존재 여부)

    Time: O(V + E)
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    count = 0

    while queue:
        u = queue.popleft()
        count += 1

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return count != n  # True면 사이클 있음
```

### 템플릿 4: 모든 위상 정렬 순서 (백트래킹)

```python
def all_topological_sorts(n: int, edges: list) -> list:
    """
    가능한 모든 위상 정렬 순서

    Time: O(V! × V) 최악
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    all_results = []

    def backtrack(path, in_deg):
        if len(path) == n:
            all_results.append(path[:])
            return

        for u in range(n):
            if in_deg[u] == 0 and u not in path:
                # 선택
                path.append(u)
                for v in graph[u]:
                    in_deg[v] -= 1

                backtrack(path, in_deg)

                # 되돌리기
                path.pop()
                for v in graph[u]:
                    in_deg[v] += 1

    backtrack([], in_degree[:])
    return all_results
```

### 템플릿 5: 작업 완료 시간 (선후 관계 + 시간)

```python
def task_completion_time(n: int, times: list, deps: list) -> list:
    """
    각 작업의 완료 시간 계산

    times[i]: i번 작업 소요 시간
    deps: [(선행, 후행), ...]

    Time: O(V + E)
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in deps:
        graph[u].append(v)
        in_degree[v] += 1

    # 각 작업의 시작 가능 시간
    start_time = [0] * n
    end_time = [0] * n

    queue = deque()

    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            end_time[i] = times[i]

    while queue:
        u = queue.popleft()

        for v in graph[u]:
            # v의 시작 시간 = 모든 선행 작업 완료 후
            start_time[v] = max(start_time[v], end_time[u])
            in_degree[v] -= 1

            if in_degree[v] == 0:
                end_time[v] = start_time[v] + times[v]
                queue.append(v)

    return end_time
```

### 템플릿 6: 사전순 위상 정렬

```python
import heapq

def lexicographic_topological_sort(n: int, edges: list) -> list:
    """
    사전순으로 가장 앞선 위상 정렬

    힙을 사용하여 진입차수 0인 노드 중 가장 작은 것 선택

    Time: O((V + E) log V)
    """
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # 최소 힙 사용
    heap = [i for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)

    result = []

    while heap:
        u = heapq.heappop(heap)
        result.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(heap, v)

    if len(result) != n:
        return []

    return result
```

---

## 예제 문제

### 문제 1: 수업 순서 (Course Schedule) - Medium

**문제 설명**
n개의 수업과 선수과목 관계가 주어질 때, 모든 수업을 들을 수 있는지 확인하세요.

**입력/출력 예시**
```
입력: n = 2, prerequisites = [[1,0]]
출력: true
설명: 0 → 1 순서로 수강 가능

입력: n = 2, prerequisites = [[1,0],[0,1]]
출력: false
설명: 사이클 존재 (0 ↔ 1)
```

**풀이**
```python
def can_finish(numCourses: int, prerequisites: list) -> bool:
    return not has_cycle(numCourses, prerequisites)
```

---

### 문제 2: 수업 순서 II (Course Schedule II) - Medium

**문제 설명**
모든 수업을 들을 수 있는 순서를 반환하세요.

**입력/출력 예시**
```
입력: n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
출력: [0,1,2,3] 또는 [0,2,1,3]
```

**풀이**
```python
def find_order(numCourses: int, prerequisites: list) -> list:
    # prerequisites: [후행, 선행] 형태
    edges = [(pre, course) for course, pre in prerequisites]
    return topological_sort_kahn(numCourses, edges)
```

---

### 문제 3: 줄 세우기 (BOJ 2252) - Gold 3

**문제 설명**
N명의 학생을 키 순서대로 줄 세우기. 일부 학생 쌍의 키 비교 결과가 주어집니다.

**입력/출력 예시**
```
입력: N = 3, comparisons = [(1,3), (2,3)]
출력: 1 2 3 또는 2 1 3
```

**풀이**
```python
def line_up(n: int, comparisons: list) -> list:
    return topological_sort_kahn(n + 1, comparisons)[1:]  # 1-indexed
```

---

### 문제 4: 작업 (BOJ 2056) - Gold 4

**문제 설명**
N개의 작업, 각 작업의 소요 시간과 선행 작업이 주어질 때, 모든 작업을 완료하는 최소 시간을 구하세요.

**입력/출력 예시**
```
입력: times = [5, 1, 3, 6, 1, 8, 4]
      deps = [(1,2), (1,3), (2,4), (2,5), (3,6), (4,7), (5,7), (6,7)]
출력: 23
```

**풀이**
```python
def min_total_time(n: int, times: list, deps: list) -> int:
    end_times = task_completion_time(n, times, deps)
    return max(end_times)
```

---

### 문제 5: 외계인의 사전 (Alien Dictionary) - Hard

**문제 설명**
외계어 단어 목록이 사전순으로 정렬되어 있습니다. 알파벳 순서를 찾으세요.

**입력/출력 예시**
```
입력: words = ["wrt", "wrf", "er", "ett", "rftt"]
출력: "wertf"
설명: w < e < r < t < f
```

**풀이**
```python
def alien_order(words: list) -> str:
    # 모든 문자 수집
    chars = set(''.join(words))
    n = len(chars)

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    graph = defaultdict(list)
    in_degree = [0] * n

    # 인접한 단어 쌍에서 순서 추출
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # 예외: 앞 단어가 더 긴데 접두사가 같으면 불가능
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                u, v = char_to_idx[c1], char_to_idx[c2]
                graph[u].append(v)
                in_degree[v] += 1
                break

    # 위상 정렬
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(idx_to_char[u])

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result) != n:
        return ""  # 사이클

    return ''.join(result)
```

---

### 문제 6: 게임 개발 (BOJ 1516) - Gold 3

**문제 설명**
N개의 건물, 각 건물의 건설 시간과 선행 건물이 주어질 때, 각 건물을 완성하는 최소 시간을 구하세요.

**풀이**
```python
def game_development(n: int, build_info: list) -> list:
    """
    build_info[i] = (건설시간, [선행건물들])
    """
    times = [info[0] for info in build_info]

    deps = []
    for i, info in enumerate(build_info):
        for pre in info[1]:
            deps.append((pre - 1, i))  # 0-indexed

    return task_completion_time(n, times, deps)
```

---

## Editorial (풀이 전략)

### Step 1: 위상 정렬 사용 조건

| 키워드 | 위상 정렬 적용 |
|--------|---------------|
| 선수과목/선행작업 | ✅ |
| 순서 결정 | ✅ |
| 의존성 관계 | ✅ |
| DAG (비순환) | ✅ |
| 사이클 검출 | ✅ |

### Step 2: 알고리즘 선택

```
Q1: 사이클 검출이 필요한가?
    Yes → Kahn's Algorithm (BFS)

Q2: 사전순 정렬이 필요한가?
    Yes → 힙 사용

Q3: 모든 순서가 필요한가?
    Yes → 백트래킹

Q4: 작업 시간 계산이 필요한가?
    Yes → 진입차수 + 시간 누적
```

### Step 3: 구현 패턴

```python
# 기본 패턴
1. 그래프 구성 + 진입차수 계산
2. 진입차수 0인 노드 큐에 추가
3. 큐에서 꺼내면서 인접 노드 진입차수 감소
4. 진입차수 0이 되면 큐에 추가
5. 결과 길이 확인 (사이클 검사)
```

---

## 자주 하는 실수

### 1. 간선 방향 혼동
```python
# ❌ 선행 → 후행 반대로
for course, pre in prerequisites:
    graph[course].append(pre)  # 틀림!

# ✅ 선행 → 후행
for course, pre in prerequisites:
    graph[pre].append(course)
```

### 2. 사이클 검사 누락
```python
# ❌ 사이클 체크 안 함
return result

# ✅ 모든 노드 방문 확인
if len(result) != n:
    return []  # 사이클 존재
return result
```

### 3. 중복 간선 처리
```python
# ❌ 중복 간선 시 진입차수 중복 증가
for u, v in edges:
    graph[u].append(v)
    in_degree[v] += 1

# ✅ set으로 중복 제거 또는 확인
seen = set()
for u, v in edges:
    if (u, v) not in seen:
        seen.add((u, v))
        graph[u].append(v)
        in_degree[v] += 1
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 207 | Course Schedule | Medium |
| LeetCode | 210 | Course Schedule II | Medium |
| LeetCode | 269 | Alien Dictionary | Hard |
| LeetCode | 310 | Minimum Height Trees | Medium |
| LeetCode | 802 | Find Eventual Safe States | Medium |
| BOJ | 2252 | 줄 세우기 | Gold 3 |
| BOJ | 1516 | 게임 개발 | Gold 3 |
| BOJ | 2056 | 작업 | Gold 4 |
| BOJ | 1766 | 문제집 | Gold 2 |
| BOJ | 2623 | 음악프로그램 | Gold 3 |

---

## 임베딩용 키워드

```
topological sort, 위상 정렬, DAG, 방향 비순환 그래프,
kahn's algorithm, 칸 알고리즘, in-degree, 진입차수,
course schedule, 수업 순서, prerequisite, 선수과목,
task scheduling, 작업 스케줄링, dependency, 의존성
```
