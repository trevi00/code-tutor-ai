"""Algorithm Pattern Knowledge Base for RAG System"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# 25 Algorithm Patterns as defined in PRD
ALGORITHM_PATTERNS = [
    {
        "id": "two-pointers",
        "name": "Two Pointers",
        "name_ko": "투 포인터",
        "description": "Use two pointers to traverse array from different positions",
        "description_ko": "배열을 두 포인터로 다른 위치에서 순회하는 기법",
        "use_cases": [
            "정렬된 배열에서 두 수의 합 찾기",
            "중복 제거하기",
            "가장 많은 물을 담는 컨테이너",
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "example_code": """def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []""",
        "keywords": ["pointer", "sorted", "pair", "sum", "opposite direction"],
    },
    {
        "id": "sliding-window",
        "name": "Sliding Window",
        "name_ko": "슬라이딩 윈도우",
        "description": "Maintain a window that slides through the array",
        "description_ko": "배열을 통해 슬라이딩하는 윈도우를 유지하는 기법",
        "use_cases": [
            "최대 합 부분배열",
            "가장 긴 부분 문자열",
            "최소 윈도우 부분 문자열",
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(k) where k is window size",
        "example_code": """def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i-k] + nums[i]
        max_sum = max(max_sum, window_sum)
    return max_sum""",
        "keywords": ["window", "subarray", "substring", "consecutive", "contiguous"],
    },
    {
        "id": "fast-slow-pointers",
        "name": "Fast & Slow Pointers",
        "name_ko": "빠른/느린 포인터",
        "description": "Two pointers moving at different speeds",
        "description_ko": "다른 속도로 이동하는 두 포인터 기법",
        "use_cases": ["순환 감지", "연결 리스트 중간 노드 찾기", "행복한 숫자"],
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "example_code": """def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False""",
        "keywords": ["cycle", "linked list", "middle", "tortoise", "hare"],
    },
    {
        "id": "merge-intervals",
        "name": "Merge Intervals",
        "name_ko": "구간 병합",
        "description": "Merge overlapping intervals",
        "description_ko": "겹치는 구간들을 병합하는 기법",
        "use_cases": ["구간 병합하기", "구간 삽입하기", "회의실 배정"],
        "time_complexity": "O(n log n)",
        "space_complexity": "O(n)",
        "example_code": """def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged""",
        "keywords": ["interval", "overlap", "merge", "range", "meeting"],
    },
    {
        "id": "cyclic-sort",
        "name": "Cyclic Sort",
        "name_ko": "순환 정렬",
        "description": "Sort array with numbers in range [1, n]",
        "description_ko": "[1, n] 범위의 숫자 배열을 정렬하는 기법",
        "use_cases": ["누락된 숫자 찾기", "중복 숫자 찾기", "모든 중복 숫자 찾기"],
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "example_code": """def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1
    return nums""",
        "keywords": ["missing", "duplicate", "range", "1 to n", "in-place"],
    },
    {
        "id": "linked-list-reversal",
        "name": "In-place Reversal of Linked List",
        "name_ko": "연결 리스트 뒤집기",
        "description": "Reverse linked list in-place",
        "description_ko": "연결 리스트를 제자리에서 뒤집는 기법",
        "use_cases": ["연결 리스트 뒤집기", "부분 리스트 뒤집기", "리스트 회전"],
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "example_code": """def reverse_list(head):
    prev, current = None, head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev""",
        "keywords": ["reverse", "linked list", "in-place", "sublist"],
    },
    {
        "id": "bfs",
        "name": "Breadth-First Search",
        "name_ko": "너비 우선 탐색",
        "description": "Level-by-level traversal using queue",
        "description_ko": "큐를 사용한 레벨별 순회",
        "use_cases": ["레벨 순서 순회", "최단 경로", "최소 깊이"],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """from collections import deque
def bfs(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result""",
        "keywords": ["queue", "level", "shortest path", "tree", "graph"],
    },
    {
        "id": "dfs",
        "name": "Depth-First Search",
        "name_ko": "깊이 우선 탐색",
        "description": "Explore as deep as possible before backtracking",
        "description_ko": "백트래킹 전에 가능한 깊이 탐색",
        "use_cases": ["트리 순회", "경로 찾기", "연결 요소"],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """def dfs(root):
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return result""",
        "keywords": ["stack", "recursion", "backtrack", "tree", "graph"],
    },
    {
        "id": "two-heaps",
        "name": "Two Heaps",
        "name_ko": "두 개의 힙",
        "description": "Use min-heap and max-heap together",
        "description_ko": "최소 힙과 최대 힙을 함께 사용하는 기법",
        "use_cases": ["중앙값 찾기", "슬라이딩 윈도우 중앙값", "IPO 문제"],
        "time_complexity": "O(log n) per operation",
        "space_complexity": "O(n)",
        "example_code": """import heapq
class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated)
        self.large = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))""",
        "keywords": ["heap", "median", "priority queue", "stream"],
    },
    {
        "id": "subsets",
        "name": "Subsets",
        "name_ko": "부분집합",
        "description": "Generate all subsets of a set",
        "description_ko": "집합의 모든 부분집합 생성",
        "use_cases": ["모든 부분집합", "중복 포함 부분집합", "문자 조합"],
        "time_complexity": "O(2^n)",
        "space_complexity": "O(2^n)",
        "example_code": """def subsets(nums):
    result = [[]]
    for num in nums:
        result += [curr + [num] for curr in result]
    return result""",
        "keywords": ["subset", "power set", "combination", "backtracking"],
    },
    {
        "id": "binary-search",
        "name": "Binary Search",
        "name_ko": "이진 탐색",
        "description": "Divide and conquer on sorted array",
        "description_ko": "정렬된 배열에서 분할 정복",
        "use_cases": ["정렬된 배열에서 검색", "범위 검색", "피크 요소 찾기"],
        "time_complexity": "O(log n)",
        "space_complexity": "O(1)",
        "example_code": """def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
        "keywords": ["sorted", "search", "divide", "half", "logarithmic"],
    },
    {
        "id": "top-k-elements",
        "name": "Top K Elements",
        "name_ko": "상위 K개 요소",
        "description": "Find top/bottom K elements using heap",
        "description_ko": "힙을 사용해 상위/하위 K개 요소 찾기",
        "use_cases": ["K번째 큰 수", "상위 K개 빈출 요소", "K개의 가장 가까운 점"],
        "time_complexity": "O(n log k)",
        "space_complexity": "O(k)",
        "example_code": """import heapq
def top_k_frequent(nums, k):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
    return heapq.nlargest(k, count.keys(), key=count.get)""",
        "keywords": ["heap", "kth", "top", "frequent", "priority"],
    },
    {
        "id": "k-way-merge",
        "name": "K-way Merge",
        "name_ko": "K-방향 병합",
        "description": "Merge K sorted lists using heap",
        "description_ko": "힙을 사용해 K개의 정렬된 리스트 병합",
        "use_cases": [
            "K개의 정렬된 리스트 병합",
            "행렬에서 K번째 작은 수",
            "가장 작은 범위",
        ],
        "time_complexity": "O(n log k)",
        "space_complexity": "O(k)",
        "example_code": """import heapq
def merge_k_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][elem_idx+1], list_idx, elem_idx+1))
    return result""",
        "keywords": ["merge", "sorted lists", "heap", "matrix"],
    },
    {
        "id": "dp",
        "name": "Dynamic Programming",
        "name_ko": "동적 프로그래밍",
        "description": "Solve problems by breaking into subproblems",
        "description_ko": "문제를 하위 문제로 분해하여 해결",
        "use_cases": ["피보나치", "배낭 문제", "최장 부분 수열"],
        "time_complexity": "Problem dependent",
        "space_complexity": "Problem dependent",
        "example_code": """def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]""",
        "keywords": ["memoization", "tabulation", "optimal", "subproblem", "state"],
    },
    {
        "id": "topological-sort",
        "name": "Topological Sort",
        "name_ko": "위상 정렬",
        "description": "Linear ordering of vertices in DAG",
        "description_ko": "방향 비순환 그래프의 선형 정렬",
        "use_cases": ["수강 순서", "빌드 순서", "외계인 사전"],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """from collections import deque
def topological_sort(n, edges):
    graph = {i: [] for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == n else []""",
        "keywords": ["DAG", "dependency", "order", "course", "prerequisite"],
    },
    {
        "id": "union-find",
        "name": "Union-Find (Disjoint Set)",
        "name_ko": "유니온-파인드",
        "description": "Track elements in disjoint sets",
        "description_ko": "분리 집합의 요소 추적",
        "use_cases": ["연결 요소", "중복 연결", "계정 병합"],
        "time_complexity": "O(α(n)) per operation",
        "space_complexity": "O(n)",
        "example_code": """class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True""",
        "keywords": ["disjoint", "connected", "component", "group", "cluster"],
    },
    {
        "id": "trie",
        "name": "Trie (Prefix Tree)",
        "name_ko": "트라이",
        "description": "Tree structure for string operations",
        "description_ko": "문자열 연산을 위한 트리 구조",
        "use_cases": ["자동완성", "단어 검색", "접두사 매칭"],
        "time_complexity": "O(m) where m is word length",
        "space_complexity": "O(n * m)",
        "example_code": """class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end""",
        "keywords": ["prefix", "autocomplete", "word", "dictionary", "string"],
    },
    {
        "id": "backtracking",
        "name": "Backtracking",
        "name_ko": "백트래킹",
        "description": "Build solution incrementally and backtrack",
        "description_ko": "점진적으로 솔루션을 구축하고 백트래킹",
        "use_cases": ["N-퀸", "스도쿠 풀이", "순열"],
        "time_complexity": "Problem dependent (often exponential)",
        "space_complexity": "O(n)",
        "example_code": """def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result""",
        "keywords": ["permutation", "combination", "constraint", "explore", "prune"],
    },
    {
        "id": "greedy",
        "name": "Greedy Algorithm",
        "name_ko": "그리디 알고리즘",
        "description": "Make locally optimal choice at each step",
        "description_ko": "각 단계에서 국소 최적 선택",
        "use_cases": ["활동 선택", "허프만 코딩", "점프 게임"],
        "time_complexity": "Problem dependent",
        "space_complexity": "Problem dependent",
        "example_code": """def jump_game(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True""",
        "keywords": ["optimal", "local", "choice", "minimum", "maximum"],
    },
    {
        "id": "monotonic-stack",
        "name": "Monotonic Stack",
        "name_ko": "단조 스택",
        "description": "Stack maintaining monotonic order",
        "description_ko": "단조 순서를 유지하는 스택",
        "use_cases": ["다음 큰 요소", "가장 큰 직사각형", "일일 온도"],
        "time_complexity": "O(n)",
        "space_complexity": "O(n)",
        "example_code": """def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)
    return result""",
        "keywords": ["stack", "next greater", "previous smaller", "histogram"],
    },
    {
        "id": "bit-manipulation",
        "name": "Bit Manipulation",
        "name_ko": "비트 조작",
        "description": "Operations using binary representation",
        "description_ko": "이진 표현을 사용한 연산",
        "use_cases": ["단일 숫자", "2의 거듭제곱", "비트 세기"],
        "time_complexity": "O(1) or O(log n)",
        "space_complexity": "O(1)",
        "example_code": """def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0""",
        "keywords": ["xor", "and", "or", "shift", "binary"],
    },
    {
        "id": "graph-coloring",
        "name": "Graph Coloring",
        "name_ko": "그래프 색칠",
        "description": "Color graph vertices with constraints",
        "description_ko": "제약 조건에 따라 그래프 정점 색칠",
        "use_cases": ["이분 그래프 확인", "M-색칠", "스도쿠 검증"],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """def is_bipartite(graph):
    n = len(graph)
    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        queue = [start]
        color[start] = 0
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True""",
        "keywords": ["bipartite", "color", "partition", "chromatic"],
    },
    {
        "id": "segment-tree",
        "name": "Segment Tree",
        "name_ko": "세그먼트 트리",
        "description": "Tree for range queries and updates",
        "description_ko": "범위 쿼리와 업데이트를 위한 트리",
        "use_cases": ["범위 합 쿼리", "범위 최솟값", "범위 업데이트"],
        "time_complexity": "O(log n) per query/update",
        "space_complexity": "O(n)",
        "example_code": """class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def update(self, i, val):
        i += self.n
        self.tree[i] = val
        while i > 1:
            i //= 2
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def query(self, l, r):
        result = 0
        l += self.n
        r += self.n
        while l < r:
            if l % 2 == 1:
                result += self.tree[l]
                l += 1
            if r % 2 == 1:
                r -= 1
                result += self.tree[r]
            l //= 2
            r //= 2
        return result""",
        "keywords": ["range", "query", "update", "sum", "minimum"],
    },
    {
        "id": "matrix-traversal",
        "name": "Matrix Traversal",
        "name_ko": "행렬 순회",
        "description": "Traverse 2D matrix in various patterns",
        "description_ko": "다양한 패턴으로 2D 행렬 순회",
        "use_cases": ["나선형 순회", "대각선 순회", "섬 문제"],
        "time_complexity": "O(m * n)",
        "space_complexity": "O(1) to O(m * n)",
        "example_code": """def spiral_order(matrix):
    if not matrix:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    return result""",
        "keywords": ["matrix", "spiral", "diagonal", "island", "flood fill"],
    },
    # ===== 새로 추가된 패턴들 (Phase 7) =====
    {
        "id": "prefix-sum",
        "name": "Prefix Sum",
        "name_ko": "누적 합",
        "description": "Precompute cumulative sums for range queries",
        "description_ko": "범위 쿼리를 위한 누적 합 미리 계산",
        "use_cases": [
            "구간 합 쿼리",
            "부분 배열 합",
            "평균 계산",
            "차이 배열",
        ],
        "time_complexity": "O(n) 전처리, O(1) 쿼리",
        "space_complexity": "O(n)",
        "example_code": """def prefix_sum(nums):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    return prefix

def range_sum(prefix, left, right):
    # [left, right] 구간 합 (0-indexed)
    return prefix[right + 1] - prefix[left]""",
        "keywords": ["cumulative", "range sum", "subarray", "prefix", "precompute"],
    },
    {
        "id": "prefix-sum-2d",
        "name": "2D Prefix Sum",
        "name_ko": "2차원 누적 합",
        "description": "Precompute cumulative sums for 2D range queries",
        "description_ko": "2차원 범위 쿼리를 위한 누적 합 미리 계산",
        "use_cases": [
            "2D 구간 합",
            "부분 행렬 합",
            "영역 쿼리",
        ],
        "time_complexity": "O(m*n) 전처리, O(1) 쿼리",
        "space_complexity": "O(m*n)",
        "example_code": """def build_2d_prefix(matrix):
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            prefix[i+1][j+1] = (prefix[i][j+1] + prefix[i+1][j]
                               - prefix[i][j] + matrix[i][j])
    return prefix

def query_2d(prefix, r1, c1, r2, c2):
    return (prefix[r2+1][c2+1] - prefix[r1][c2+1]
            - prefix[r2+1][c1] + prefix[r1][c1])""",
        "keywords": ["matrix sum", "2d range", "submatrix", "area", "region"],
    },
    {
        "id": "sieve-of-eratosthenes",
        "name": "Sieve of Eratosthenes",
        "name_ko": "에라토스테네스의 체",
        "description": "Efficiently find all primes up to n",
        "description_ko": "n까지의 모든 소수를 효율적으로 찾기",
        "use_cases": [
            "소수 생성",
            "소수 판별",
            "N번째 소수 찾기",
            "소인수분해",
        ],
        "time_complexity": "O(n log log n)",
        "space_complexity": "O(n)",
        "example_code": """def sieve_of_eratosthenes(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]""",
        "keywords": ["prime", "sieve", "factor", "number theory", "primality"],
    },
    {
        "id": "kmp-algorithm",
        "name": "KMP String Matching",
        "name_ko": "KMP 문자열 매칭",
        "description": "Efficient string pattern matching using failure function",
        "description_ko": "실패 함수를 사용한 효율적인 문자열 패턴 매칭",
        "use_cases": [
            "문자열 검색",
            "패턴 매칭",
            "부분 문자열 찾기",
            "반복 패턴 감지",
        ],
        "time_complexity": "O(n + m)",
        "space_complexity": "O(m)",
        "example_code": """def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length > 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    lps = build_lps(pattern)
    i = j = 0
    positions = []
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                positions.append(i - j)
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1
    return positions""",
        "keywords": ["string match", "pattern", "substring", "failure function", "lps"],
    },
    {
        "id": "counting-sort",
        "name": "Counting Sort",
        "name_ko": "계수 정렬",
        "description": "Sort by counting occurrences of each element",
        "description_ko": "각 요소의 출현 횟수를 세어 정렬",
        "use_cases": [
            "제한된 범위의 정수 정렬",
            "빈도 기반 정렬",
            "안정 정렬 필요 시",
        ],
        "time_complexity": "O(n + k) where k is range",
        "space_complexity": "O(k)",
        "example_code": """def counting_sort(arr):
    if not arr:
        return arr
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val + 1
    count = [0] * range_size

    for num in arr:
        count[num - min_val] += 1

    result = []
    for i, cnt in enumerate(count):
        result.extend([i + min_val] * cnt)
    return result""",
        "keywords": ["counting", "integer sort", "linear sort", "frequency", "bucket"],
    },
    {
        "id": "radix-sort",
        "name": "Radix Sort",
        "name_ko": "기수 정렬",
        "description": "Sort by processing individual digits",
        "description_ko": "개별 자릿수를 처리하여 정렬",
        "use_cases": [
            "정수 정렬",
            "문자열 정렬",
            "고정 길이 키 정렬",
        ],
        "time_complexity": "O(d * (n + k)) where d is digits, k is base",
        "space_complexity": "O(n + k)",
        "example_code": """def radix_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    for i in range(n):
        arr[i] = output[i]""",
        "keywords": ["digit", "base", "lsd", "msd", "linear sort"],
    },
    {
        "id": "shell-sort",
        "name": "Shell Sort",
        "name_ko": "셸 정렬",
        "description": "Generalized insertion sort with gap sequence",
        "description_ko": "갭 시퀀스를 사용한 일반화된 삽입 정렬",
        "use_cases": [
            "중간 크기 배열 정렬",
            "거의 정렬된 배열",
            "제자리 정렬 필요 시",
        ],
        "time_complexity": "O(n log^2 n) to O(n^2)",
        "space_complexity": "O(1)",
        "example_code": """def shell_sort(arr):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr""",
        "keywords": ["gap sequence", "insertion sort", "in-place", "diminishing increment"],
    },
    {
        "id": "fenwick-tree",
        "name": "Fenwick Tree (BIT)",
        "name_ko": "펜윅 트리",
        "description": "Binary indexed tree for range queries and point updates",
        "description_ko": "범위 쿼리와 포인트 업데이트를 위한 이진 인덱스 트리",
        "use_cases": [
            "범위 합 쿼리",
            "포인트 업데이트",
            "역전 카운트",
            "누적 빈도",
        ],
        "time_complexity": "O(log n) per operation",
        "space_complexity": "O(n)",
        "example_code": """class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        i += 1  # 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 다음 노드로 이동

    def prefix_sum(self, i):
        i += 1  # 1-indexed
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & (-i)  # 부모 노드로 이동
        return result

    def range_sum(self, left, right):
        return self.prefix_sum(right) - self.prefix_sum(left - 1)""",
        "keywords": ["binary indexed tree", "bit", "range query", "point update", "inversion"],
    },
    {
        "id": "avl-tree",
        "name": "AVL Tree",
        "name_ko": "AVL 트리",
        "description": "Self-balancing binary search tree",
        "description_ko": "자가 균형 이진 탐색 트리",
        "use_cases": [
            "정렬된 데이터 유지",
            "빠른 삽입/삭제/검색",
            "균형 잡힌 트리 필요 시",
        ],
        "time_complexity": "O(log n) per operation",
        "space_complexity": "O(n)",
        "example_code": """class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None
        self.height = 1

class AVLTree:
    def get_height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right))
        return x

    def rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def insert(self, root, val):
        if not root:
            return AVLNode(val)
        if val < root.val:
            root.left = self.insert(root.left, val)
        else:
            root.right = self.insert(root.right, val)

        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)

        # LL, RR, LR, RL 케이스 처리
        if balance > 1 and val < root.left.val:
            return self.rotate_right(root)
        if balance < -1 and val > root.right.val:
            return self.rotate_left(root)
        if balance > 1 and val > root.left.val:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)
        if balance < -1 and val < root.right.val:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)
        return root""",
        "keywords": ["balanced bst", "rotation", "self-balancing", "height balanced"],
    },
    {
        "id": "lazy-segment-tree",
        "name": "Lazy Segment Tree",
        "name_ko": "레이지 세그먼트 트리",
        "description": "Segment tree with lazy propagation for range updates",
        "description_ko": "범위 업데이트를 위한 지연 전파 세그먼트 트리",
        "use_cases": [
            "범위 업데이트 + 범위 쿼리",
            "구간에 값 더하기",
            "구간 최솟값/최댓값 업데이트",
        ],
        "time_complexity": "O(log n) per operation",
        "space_complexity": "O(n)",
        "example_code": """class LazySegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.build(arr, 1, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2*node, start, mid)
            self.build(arr, 2*node+1, mid+1, end)
            self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def push_down(self, node, start, end):
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            self.tree[2*node] += (mid - start + 1) * self.lazy[node]
            self.tree[2*node+1] += (end - mid) * self.lazy[node]
            self.lazy[2*node] += self.lazy[node]
            self.lazy[2*node+1] += self.lazy[node]
            self.lazy[node] = 0

    def range_update(self, node, start, end, l, r, val):
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.tree[node] += (end - start + 1) * val
            self.lazy[node] += val
            return
        self.push_down(node, start, end)
        mid = (start + end) // 2
        self.range_update(2*node, start, mid, l, r, val)
        self.range_update(2*node+1, mid+1, end, l, r, val)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]""",
        "keywords": ["lazy propagation", "range update", "segment tree", "interval"],
    },
    {
        "id": "dijkstra",
        "name": "Dijkstra's Algorithm",
        "name_ko": "다익스트라 알고리즘",
        "description": "Find shortest path from source to all vertices",
        "description_ko": "출발점에서 모든 정점까지의 최단 경로 찾기",
        "use_cases": [
            "최단 경로",
            "네트워크 지연 시간",
            "GPS 네비게이션",
            "라우팅",
        ],
        "time_complexity": "O((V + E) log V)",
        "space_complexity": "O(V)",
        "example_code": """import heapq
def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]  # (거리, 노드)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    return dist""",
        "keywords": ["shortest path", "weighted graph", "priority queue", "single source"],
    },
    {
        "id": "prim-mst",
        "name": "Prim's MST Algorithm",
        "name_ko": "프림 MST 알고리즘",
        "description": "Find minimum spanning tree using greedy approach",
        "description_ko": "그리디 접근법으로 최소 신장 트리 찾기",
        "use_cases": [
            "최소 신장 트리",
            "네트워크 연결",
            "클러스터링",
        ],
        "time_complexity": "O((V + E) log V)",
        "space_complexity": "O(V)",
        "example_code": """import heapq
def prim_mst(graph, n):
    visited = [False] * n
    mst_cost = 0
    mst_edges = []
    pq = [(0, 0, -1)]  # (weight, node, parent)

    while pq and len(mst_edges) < n:
        weight, u, parent = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        mst_cost += weight
        if parent != -1:
            mst_edges.append((parent, u, weight))

        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (w, v, u))

    return mst_cost, mst_edges""",
        "keywords": ["minimum spanning tree", "mst", "greedy", "connected graph"],
    },
    {
        "id": "kruskal-mst",
        "name": "Kruskal's MST Algorithm",
        "name_ko": "크루스칼 MST 알고리즘",
        "description": "Find MST by sorting edges and using union-find",
        "description_ko": "간선 정렬과 유니온-파인드를 사용한 MST 찾기",
        "use_cases": [
            "최소 신장 트리",
            "네트워크 비용 최소화",
            "희소 그래프",
        ],
        "time_complexity": "O(E log E)",
        "space_complexity": "O(V)",
        "example_code": """def kruskal_mst(n, edges):
    # edges: [(weight, u, v), ...]
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    edges.sort()  # 가중치 기준 정렬
    mst_cost = 0
    mst_edges = []

    for weight, u, v in edges:
        if union(u, v):
            mst_cost += weight
            mst_edges.append((u, v, weight))
            if len(mst_edges) == n - 1:
                break

    return mst_cost, mst_edges""",
        "keywords": ["minimum spanning tree", "mst", "union find", "edge sorting"],
    },
    {
        "id": "tree-traversal",
        "name": "Tree Traversal",
        "name_ko": "트리 순회",
        "description": "Traverse tree nodes in specific orders",
        "description_ko": "특정 순서로 트리 노드 순회",
        "use_cases": [
            "전위/중위/후위 순회",
            "트리 직렬화",
            "표현식 트리",
            "트리 복사",
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(h) where h is height",
        "example_code": """def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

def level_order(root):
    from collections import deque
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result""",
        "keywords": ["preorder", "inorder", "postorder", "level order", "traversal"],
    },
    {
        "id": "lca",
        "name": "Lowest Common Ancestor",
        "name_ko": "최소 공통 조상",
        "description": "Find lowest common ancestor of two nodes in a tree",
        "description_ko": "트리에서 두 노드의 최소 공통 조상 찾기",
        "use_cases": [
            "트리에서 최소 공통 조상",
            "거리 쿼리",
            "경로 쿼리",
        ],
        "time_complexity": "O(n) naive, O(log n) with preprocessing",
        "space_complexity": "O(n)",
        "example_code": """def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right

# Binary Lifting for O(log n) queries
def build_lca(n, parent):
    LOG = 20
    up = [[0] * n for _ in range(LOG)]
    up[0] = parent
    for k in range(1, LOG):
        for v in range(n):
            if up[k-1][v] != -1:
                up[k][v] = up[k-1][up[k-1][v]]
            else:
                up[k][v] = -1
    return up""",
        "keywords": ["ancestor", "tree", "binary lifting", "path", "distance"],
    },
    {
        "id": "interval-scheduling",
        "name": "Interval Scheduling",
        "name_ko": "구간 스케줄링",
        "description": "Select maximum non-overlapping intervals",
        "description_ko": "겹치지 않는 최대 구간 선택",
        "use_cases": [
            "회의실 배정",
            "작업 스케줄링",
            "이벤트 선택",
        ],
        "time_complexity": "O(n log n)",
        "space_complexity": "O(1)",
        "example_code": """def max_meetings(intervals):
    # 종료 시간 기준 정렬
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = float('-inf')

    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish

    return count

def min_meeting_rooms(intervals):
    import heapq
    intervals.sort(key=lambda x: x[0])
    rooms = []  # 각 방의 종료 시간

    for start, end in intervals:
        if rooms and rooms[0] <= start:
            heapq.heappop(rooms)
        heapq.heappush(rooms, end)

    return len(rooms)""",
        "keywords": ["scheduling", "interval", "meeting room", "greedy", "non-overlapping"],
    },
    # ============== Advanced Data Structures ==============
    {
        "id": "sparse-table",
        "name": "Sparse Table",
        "name_ko": "스파스 테이블",
        "description": "A data structure for answering range minimum/maximum queries in O(1) time after O(n log n) preprocessing. Works only for idempotent operations like min, max, gcd.",
        "description_ko": "O(n log n) 전처리 후 O(1) 시간에 구간 최솟값/최댓값 쿼리를 처리하는 자료구조입니다. min, max, gcd 같은 멱등 연산에만 사용 가능합니다.",
        "use_cases": [
            "정적 RMQ (Range Minimum Query)",
            "정적 구간 GCD 쿼리",
            "LCA 전처리",
        ],
        "time_complexity": "전처리 O(n log n), 쿼리 O(1)",
        "space_complexity": "O(n log n)",
        "example_code": """import math

class SparseTable:
    def __init__(self, arr):
        n = len(arr)
        k = int(math.log2(n)) + 1
        self.sparse = [[0] * k for _ in range(n)]
        self.log = [0] * (n + 1)

        # 로그 테이블 전처리
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1

        # 스파스 테이블 구축
        for i in range(n):
            self.sparse[i][0] = arr[i]

        j = 1
        while (1 << j) <= n:
            i = 0
            while i + (1 << j) - 1 < n:
                self.sparse[i][j] = min(
                    self.sparse[i][j - 1],
                    self.sparse[i + (1 << (j - 1))][j - 1]
                )
                i += 1
            j += 1

    def query(self, l, r):
        # O(1) 구간 최솟값 쿼리
        j = self.log[r - l + 1]
        return min(self.sparse[l][j], self.sparse[r - (1 << j) + 1][j])""",
        "keywords": ["sparse table", "RMQ", "range query", "O(1) query", "idempotent"],
    },
    {
        "id": "sqrt-decomposition",
        "name": "Square Root Decomposition",
        "name_ko": "제곱근 분할법",
        "description": "Divides array into sqrt(n) blocks for efficient range queries and updates. Balances between preprocessing and query time.",
        "description_ko": "배열을 sqrt(n)개의 블록으로 나누어 효율적인 구간 쿼리와 업데이트를 처리합니다. 전처리와 쿼리 시간의 균형을 맞춥니다.",
        "use_cases": [
            "구간 합/최솟값 쿼리",
            "Mo's Algorithm 기반",
            "오프라인 쿼리 처리",
        ],
        "time_complexity": "쿼리 O(√n), 업데이트 O(1) 또는 O(√n)",
        "space_complexity": "O(n)",
        "example_code": """import math

class SqrtDecomposition:
    def __init__(self, arr):
        self.arr = arr
        self.n = len(arr)
        self.block_size = int(math.ceil(math.sqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * self.num_blocks

        # 블록 합 초기화
        for i in range(self.n):
            self.blocks[i // self.block_size] += arr[i]

    def update(self, idx, val):
        block_idx = idx // self.block_size
        self.blocks[block_idx] += val - self.arr[idx]
        self.arr[idx] = val

    def query(self, l, r):
        # 구간 [l, r] 합
        result = 0
        while l <= r:
            if l % self.block_size == 0 and l + self.block_size - 1 <= r:
                # 완전한 블록
                result += self.blocks[l // self.block_size]
                l += self.block_size
            else:
                result += self.arr[l]
                l += 1
        return result""",
        "keywords": ["sqrt decomposition", "block", "range query", "Mo's algorithm"],
    },
    {
        "id": "persistent-segment-tree",
        "name": "Persistent Segment Tree",
        "name_ko": "퍼시스턴트 세그먼트 트리",
        "description": "Segment tree that preserves all historical versions after updates. Each update creates a new root while sharing unchanged nodes.",
        "description_ko": "업데이트 후에도 모든 이전 버전을 보존하는 세그먼트 트리입니다. 각 업데이트는 변경되지 않은 노드를 공유하며 새 루트를 생성합니다.",
        "use_cases": [
            "K번째 수 찾기",
            "버전 관리가 필요한 구간 쿼리",
            "2D 쿼리 처리",
        ],
        "time_complexity": "업데이트 O(log n), 쿼리 O(log n)",
        "space_complexity": "O(n log n)",
        "example_code": """class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class PersistentSegTree:
    def __init__(self, n):
        self.n = n
        self.roots = []  # 버전별 루트

    def build(self, arr, node, start, end):
        if start == end:
            return Node(arr[start])
        mid = (start + end) // 2
        left = self.build(arr, None, start, mid)
        right = self.build(arr, None, mid + 1, end)
        return Node(left.val + right.val, left, right)

    def update(self, node, start, end, idx, val):
        if start == end:
            return Node(val)
        mid = (start + end) // 2
        if idx <= mid:
            new_left = self.update(node.left, start, mid, idx, val)
            return Node(new_left.val + node.right.val, new_left, node.right)
        else:
            new_right = self.update(node.right, mid + 1, end, idx, val)
            return Node(node.left.val + new_right.val, node.left, new_right)

    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return node.val
        mid = (start + end) // 2
        return (self.query(node.left, start, mid, l, r) +
                self.query(node.right, mid + 1, end, l, r))""",
        "keywords": ["persistent", "segment tree", "version control", "k-th smallest"],
    },
    {
        "id": "treap",
        "name": "Treap",
        "name_ko": "트립",
        "description": "Randomized BST that maintains both BST property (by key) and heap property (by random priority). Supports split and merge operations.",
        "description_ko": "키에 대해 BST 속성, 랜덤 우선순위에 대해 힙 속성을 유지하는 랜덤화된 이진 검색 트리입니다. 분할과 병합 연산을 지원합니다.",
        "use_cases": [
            "동적 순서 통계",
            "구간에 대한 삽입/삭제",
            "암묵적 키 배열",
        ],
        "time_complexity": "기대값 O(log n)",
        "space_complexity": "O(n)",
        "example_code": """import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()
        self.left = None
        self.right = None
        self.size = 1

def get_size(node):
    return node.size if node else 0

def update(node):
    if node:
        node.size = 1 + get_size(node.left) + get_size(node.right)
    return node

def split(node, key):
    if not node:
        return None, None
    if node.key <= key:
        left, right = split(node.right, key)
        node.right = left
        return update(node), right
    else:
        left, right = split(node.left, key)
        node.left = right
        return left, update(node)

def merge(left, right):
    if not left or not right:
        return left or right
    if left.priority > right.priority:
        left.right = merge(left.right, right)
        return update(left)
    else:
        right.left = merge(left, right.left)
        return update(right)

def insert(root, key):
    left, right = split(root, key)
    new_node = TreapNode(key)
    return merge(merge(left, new_node), right)""",
        "keywords": ["treap", "randomized BST", "split", "merge", "order statistics"],
    },
    {
        "id": "link-cut-tree",
        "name": "Link-Cut Tree",
        "name_ko": "링크컷 트리",
        "description": "Dynamic tree data structure supporting link, cut, and path queries. Uses splay trees to represent preferred paths.",
        "description_ko": "link, cut, 경로 쿼리를 지원하는 동적 트리 자료구조입니다. 선호 경로를 스플레이 트리로 표현합니다.",
        "use_cases": [
            "동적 트리 연결/끊기",
            "경로 쿼리",
            "최소 공통 조상 (동적)",
        ],
        "time_complexity": "분할 상환 O(log n)",
        "space_complexity": "O(n)",
        "example_code": """class LCTNode:
    def __init__(self, val=0):
        self.val = val
        self.sum = val
        self.rev = False
        self.parent = None
        self.children = [None, None]

def is_root(x):
    return not x.parent or (x.parent.children[0] != x and x.parent.children[1] != x)

def push(x):
    if x.rev:
        x.children[0], x.children[1] = x.children[1], x.children[0]
        for c in x.children:
            if c:
                c.rev ^= True
        x.rev = False

def pull(x):
    x.sum = x.val
    for c in x.children:
        if c:
            x.sum += c.sum

def rotate(x):
    p = x.parent
    g = p.parent
    d = 1 if p.children[1] == x else 0
    push(p)
    push(x)

    if not is_root(p):
        g.children[1 if g.children[1] == p else 0] = x
    x.parent = g
    p.children[d] = x.children[1 - d]
    if x.children[1 - d]:
        x.children[1 - d].parent = p
    x.children[1 - d] = p
    p.parent = x
    pull(p)
    pull(x)

def splay(x):
    while not is_root(x):
        p = x.parent
        if not is_root(p):
            g = p.parent
            if (g.children[1] == p) == (p.children[1] == x):
                rotate(p)
            else:
                rotate(x)
        rotate(x)

def access(x):
    splay(x)
    x.children[1] = None
    pull(x)
    while x.parent:
        p = x.parent
        splay(p)
        p.children[1] = x
        pull(p)
        splay(x)

def link(x, y):
    access(x)
    access(y)
    x.parent = y""",
        "keywords": ["link-cut tree", "dynamic tree", "splay", "path query", "LCA"],
    },
    # ============== Graph Algorithms ==============
    {
        "id": "bellman-ford",
        "name": "Bellman-Ford Algorithm",
        "name_ko": "벨만-포드 알고리즘",
        "description": "Single-source shortest path algorithm that handles negative edge weights. Can detect negative cycles.",
        "description_ko": "음수 가중치 간선을 처리할 수 있는 단일 출발점 최단 경로 알고리즘입니다. 음수 사이클을 감지할 수 있습니다.",
        "use_cases": [
            "음수 가중치 그래프",
            "음수 사이클 탐지",
            "거리 차이 제약 (SPFA)",
        ],
        "time_complexity": "O(VE)",
        "space_complexity": "O(V)",
        "example_code": """def bellman_ford(n, edges, src):
    # edges: [(u, v, w), ...]
    dist = [float('inf')] * n
    dist[src] = 0

    # V-1번 완화
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # 음수 사이클 검사
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # 음수 사이클 존재

    return dist

# SPFA (최적화 버전)
from collections import deque

def spfa(n, adj, src):
    dist = [float('inf')] * n
    dist[src] = 0
    in_queue = [False] * n
    count = [0] * n  # 큐 진입 횟수

    q = deque([src])
    in_queue[src] = True

    while q:
        u = q.popleft()
        in_queue[u] = False

        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return None  # 음수 사이클

    return dist""",
        "keywords": ["bellman-ford", "negative weight", "negative cycle", "SPFA", "shortest path"],
    },
    {
        "id": "floyd-warshall",
        "name": "Floyd-Warshall Algorithm",
        "name_ko": "플로이드-워셜 알고리즘",
        "description": "All-pairs shortest path algorithm. Uses dynamic programming to find shortest paths between all pairs of vertices.",
        "description_ko": "모든 쌍 최단 경로 알고리즘입니다. 동적 프로그래밍을 사용하여 모든 정점 쌍 사이의 최단 경로를 찾습니다.",
        "use_cases": [
            "모든 쌍 최단 거리",
            "그래프 추이적 폐쇄",
            "경로 존재 여부 확인",
        ],
        "time_complexity": "O(V³)",
        "space_complexity": "O(V²)",
        "example_code": """def floyd_warshall(n, edges):
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]

    # 자기 자신까지 거리는 0
    for i in range(n):
        dist[i][i] = 0

    # 간선 초기화
    for u, v, w in edges:
        dist[u][v] = w
        next_node[u][v] = v

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node

def reconstruct_path(u, v, next_node):
    if next_node[u][v] is None:
        return []
    path = [u]
    while u != v:
        u = next_node[u][v]
        path.append(u)
    return path""",
        "keywords": ["floyd-warshall", "all pairs", "shortest path", "transitive closure"],
    },
    {
        "id": "articulation-bridges",
        "name": "Articulation Points and Bridges",
        "name_ko": "단절점과 단절선",
        "description": "Find critical vertices (articulation points) and edges (bridges) whose removal disconnects the graph using DFS.",
        "description_ko": "DFS를 사용하여 그래프를 끊어지게 만드는 중요한 정점(단절점)과 간선(단절선)을 찾습니다.",
        "use_cases": [
            "네트워크 취약점 분석",
            "그래프 연결성",
            "이중 연결 요소",
        ],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """def find_articulation_and_bridges(n, adj):
    discovery = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    articulation_points = set()
    bridges = []
    time = [0]

    def dfs(u):
        children = 0
        discovery[u] = low[u] = time[0]
        time[0] += 1

        for v in adj[u]:
            if discovery[v] == -1:  # 방문 안 한 정점
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # 단절점 조건
                if parent[u] == -1 and children > 1:
                    articulation_points.add(u)
                if parent[u] != -1 and low[v] >= discovery[u]:
                    articulation_points.add(u)

                # 단절선 조건
                if low[v] > discovery[u]:
                    bridges.append((u, v))
            elif v != parent[u]:
                low[u] = min(low[u], discovery[v])

    for i in range(n):
        if discovery[i] == -1:
            dfs(i)

    return articulation_points, bridges""",
        "keywords": ["articulation point", "bridge", "cut vertex", "biconnected", "DFS"],
    },
    {
        "id": "2-sat",
        "name": "2-SAT",
        "name_ko": "2-SAT",
        "description": "Solve boolean satisfiability problems with clauses of exactly 2 literals using SCC decomposition.",
        "description_ko": "정확히 2개의 리터럴을 가진 절로 이루어진 불리언 만족성 문제를 SCC 분해를 사용하여 해결합니다.",
        "use_cases": [
            "논리 제약 조건 만족",
            "스케줄링 문제",
            "그래프 2-색칠 가능성",
        ],
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "example_code": """def solve_2sat(n, clauses):
    # n: 변수 개수 (0 ~ n-1)
    # clauses: [(a, b), ...] where a, b are literals
    # literal i = variable i is True
    # literal ~i (= i + n) = variable i is False

    adj = [[] for _ in range(2 * n)]
    radj = [[] for _ in range(2 * n)]

    def neg(x):
        return x + n if x < n else x - n

    # 그래프 구축: (a OR b) => (~a -> b) AND (~b -> a)
    for a, b in clauses:
        adj[neg(a)].append(b)
        adj[neg(b)].append(a)
        radj[b].append(neg(a))
        radj[a].append(neg(b))

    # Kosaraju's SCC
    order = []
    visited = [False] * (2 * n)

    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    def dfs2(u, comp):
        scc[u] = comp
        for v in radj[u]:
            if scc[v] == -1:
                dfs2(v, comp)

    for i in range(2 * n):
        if not visited[i]:
            dfs1(i)

    scc = [-1] * (2 * n)
    comp = 0
    for u in reversed(order):
        if scc[u] == -1:
            dfs2(u, comp)
            comp += 1

    # 만족 가능성 검사
    for i in range(n):
        if scc[i] == scc[i + n]:
            return None  # 불가능

    # 해 구성
    return [scc[i] > scc[i + n] for i in range(n)]""",
        "keywords": ["2-SAT", "boolean satisfiability", "SCC", "implication graph"],
    },
    # ============== DP Patterns ==============
    {
        "id": "knapsack-01",
        "name": "0/1 Knapsack",
        "name_ko": "0/1 배낭 문제",
        "description": "Classic DP problem: maximize value of items that fit in a knapsack. Each item can be taken at most once.",
        "description_ko": "배낭에 들어갈 수 있는 물건의 가치를 최대화하는 클래식 DP 문제입니다. 각 물건은 최대 한 번만 선택할 수 있습니다.",
        "use_cases": [
            "자원 할당 최적화",
            "부분집합 합",
            "예산 제약 선택",
        ],
        "time_complexity": "O(nW)",
        "space_complexity": "O(W) (공간 최적화)",
        "example_code": """def knapsack_01(weights, values, capacity):
    n = len(weights)
    # dp[w] = 무게 w 이하일 때 최대 가치
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 역순으로 순회 (각 물건 한 번만 사용)
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

# 선택한 물건 역추적
def knapsack_with_items(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w],
                              dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    # 역추적
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], selected[::-1]""",
        "keywords": ["knapsack", "0/1", "subset", "optimization", "DP"],
    },
    {
        "id": "lis-binary-search",
        "name": "LIS with Binary Search",
        "name_ko": "이분 탐색 LIS",
        "description": "Find Longest Increasing Subsequence in O(n log n) using binary search with auxiliary array.",
        "description_ko": "보조 배열과 이분 탐색을 사용하여 O(n log n)에 최장 증가 부분 수열을 찾습니다.",
        "use_cases": [
            "최장 증가/감소 수열",
            "박스 쌓기",
            "체인 문제",
        ],
        "time_complexity": "O(n log n)",
        "space_complexity": "O(n)",
        "example_code": """import bisect

def lis_length(arr):
    # tails[i] = 길이가 i+1인 IS의 가장 작은 마지막 원소
    tails = []

    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)

def lis_with_sequence(arr):
    n = len(arr)
    tails = []
    prev = [-1] * n
    indices = []  # tails의 각 위치에 해당하는 원본 인덱스

    for i, num in enumerate(arr):
        pos = bisect.bisect_left(tails, num)
        if pos > 0:
            prev[i] = indices[pos - 1]
        if pos == len(tails):
            tails.append(num)
            indices.append(i)
        else:
            tails[pos] = num
            indices[pos] = i

    # LIS 역추적
    lis = []
    idx = indices[-1]
    while idx != -1:
        lis.append(arr[idx])
        idx = prev[idx]

    return lis[::-1]""",
        "keywords": ["LIS", "longest increasing subsequence", "binary search", "patience sorting"],
    },
    {
        "id": "bitmask-dp",
        "name": "Bitmask DP",
        "name_ko": "비트마스크 DP",
        "description": "DP using bitmask to represent set states. Efficient for problems with small set sizes (n ≤ 20).",
        "description_ko": "집합 상태를 비트마스크로 표현하는 DP입니다. 작은 집합 크기(n ≤ 20)의 문제에 효율적입니다.",
        "use_cases": [
            "외판원 문제 (TSP)",
            "집합 커버",
            "할당 문제",
        ],
        "time_complexity": "O(2^n × n)",
        "space_complexity": "O(2^n)",
        "example_code": """def tsp(dist):
    n = len(dist)
    INF = float('inf')
    # dp[mask][i] = mask에 속한 도시를 방문하고 i에서 끝날 때 최소 비용
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 시작점 0

    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            if not (mask & (1 << last)):
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue  # 이미 방문
                new_mask = mask | (1 << next_city)
                dp[new_mask][next_city] = min(
                    dp[new_mask][next_city],
                    dp[mask][last] + dist[last][next_city]
                )

    # 모든 도시 방문 후 시작점으로 복귀
    full_mask = (1 << n) - 1
    answer = min(dp[full_mask][i] + dist[i][0] for i in range(n))
    return answer

# SOS DP (Sum over Subsets)
def sos_dp(arr):
    n = len(arr).bit_length()
    dp = arr.copy()

    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]

    return dp""",
        "keywords": ["bitmask", "TSP", "subset", "state compression", "SOS DP"],
    },
    {
        "id": "interval-dp",
        "name": "Interval DP",
        "name_ko": "구간 DP",
        "description": "DP over intervals [i, j]. Solve by combining smaller subintervals. Common for parenthesization problems.",
        "description_ko": "구간 [i, j]에 대한 DP입니다. 더 작은 부분 구간을 결합하여 해결합니다. 괄호화 문제에 흔히 사용됩니다.",
        "use_cases": [
            "행렬 체인 곱셈",
            "최적 BST",
            "팰린드롬 분할",
        ],
        "time_complexity": "O(n³) 또는 O(n² log n)",
        "space_complexity": "O(n²)",
        "example_code": """def matrix_chain_multiplication(dims):
    # dims[i] = i번째 행렬의 행 수, dims[i+1] = 열 수
    n = len(dims) - 1
    # dp[i][j] = i~j번 행렬 곱의 최소 연산 횟수
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):  # 구간 길이
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):  # 분할점
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

def min_palindrome_partitions(s):
    n = len(s)
    # is_pal[i][j] = s[i:j+1]이 팰린드롬인지
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for i in range(n - 1):
        is_pal[i][i+1] = (s[i] == s[i+1])
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = is_pal[i+1][j-1] and s[i] == s[j]

    # dp[i] = s[0:i+1]의 최소 분할 수
    dp = list(range(n))
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_pal[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n-1]""",
        "keywords": ["interval DP", "range DP", "matrix chain", "palindrome", "parenthesization"],
    },
    {
        "id": "digit-dp",
        "name": "Digit DP",
        "name_ko": "자릿수 DP",
        "description": "Count numbers in a range satisfying certain digit conditions. Process digits from most significant.",
        "description_ko": "특정 자릿수 조건을 만족하는 범위 내 숫자를 셉니다. 가장 큰 자릿수부터 처리합니다.",
        "use_cases": [
            "특정 자릿수 합을 가진 수 세기",
            "특정 숫자를 포함/제외하는 수",
            "범위 내 조건 만족 수 세기",
        ],
        "time_complexity": "O(자릿수 × 상태 수)",
        "space_complexity": "O(자릿수 × 상태 수)",
        "example_code": """def count_digit_sum(n, target_sum):
    # 1부터 n까지 자릿수 합이 target_sum인 수의 개수
    digits = [int(d) for d in str(n)]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, digit_sum, tight, started):
        if pos == len(digits):
            return 1 if started and digit_sum == target_sum else 0

        limit = digits[pos] if tight else 9
        result = 0

        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_started = started or (d > 0)
            new_sum = digit_sum + d if new_started else 0
            if new_sum <= target_sum:
                result += dp(pos + 1, new_sum, new_tight, new_started)

        return result

    return dp(0, 0, True, False)

def count_without_digit(n, forbidden):
    # 1부터 n까지 forbidden 자릿수를 포함하지 않는 수
    digits = [int(d) for d in str(n)]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos, tight, started):
        if pos == len(digits):
            return 1 if started else 0

        limit = digits[pos] if tight else 9
        result = 0

        for d in range(0, limit + 1):
            if d == forbidden and started:
                continue
            if d == forbidden and not started and d > 0:
                continue

            new_tight = tight and (d == limit)
            new_started = started or (d > 0)
            result += dp(pos + 1, new_tight, new_started)

        return result

    return dp(0, True, False)""",
        "keywords": ["digit DP", "counting", "range query", "digit sum"],
    },
    # ============== String Algorithms ==============
    {
        "id": "rabin-karp",
        "name": "Rabin-Karp Algorithm",
        "name_ko": "라빈-카프 알고리즘",
        "description": "String matching using rolling hash. Efficient for multiple pattern search or plagiarism detection.",
        "description_ko": "롤링 해시를 사용한 문자열 매칭 알고리즘입니다. 다중 패턴 검색이나 표절 탐지에 효율적입니다.",
        "use_cases": [
            "다중 패턴 매칭",
            "가장 긴 중복 부분 문자열",
            "표절 탐지",
        ],
        "time_complexity": "평균 O(n + m), 최악 O(nm)",
        "space_complexity": "O(1)",
        "example_code": """def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    if m > n:
        return []

    BASE = 256
    MOD = 10**9 + 7

    # 패턴 해시 계산
    pattern_hash = 0
    text_hash = 0
    h = pow(BASE, m - 1, MOD)

    for i in range(m):
        pattern_hash = (pattern_hash * BASE + ord(pattern[i])) % MOD
        text_hash = (text_hash * BASE + ord(text[i])) % MOD

    matches = []

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # 해시 충돌 확인
            if text[i:i+m] == pattern:
                matches.append(i)

        if i < n - m:
            # 롤링 해시 업데이트
            text_hash = (text_hash - ord(text[i]) * h) % MOD
            text_hash = (text_hash * BASE + ord(text[i + m])) % MOD
            text_hash = (text_hash + MOD) % MOD

    return matches

# 다중 패턴 버전
def rabin_karp_multi(text, patterns):
    pattern_hashes = {}
    BASE, MOD = 256, 10**9 + 7

    for p in patterns:
        h = 0
        for c in p:
            h = (h * BASE + ord(c)) % MOD
        pattern_hashes.setdefault(len(p), {})[h] = p

    results = {p: [] for p in patterns}
    # ... (각 패턴 길이별로 롤링 해시 적용)
    return results""",
        "keywords": ["rabin-karp", "rolling hash", "string matching", "fingerprint"],
    },
    {
        "id": "z-algorithm",
        "name": "Z Algorithm",
        "name_ko": "Z 알고리즘",
        "description": "Compute Z-array where Z[i] is the length of longest substring starting at i that matches prefix. O(n) time.",
        "description_ko": "Z[i]가 i에서 시작하는 가장 긴 접두사 일치 부분 문자열 길이인 Z 배열을 계산합니다. O(n) 시간.",
        "use_cases": [
            "패턴 매칭",
            "주기 찾기",
            "가장 긴 공통 접두사",
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(n)",
        "example_code": """def z_function(s):
    n = len(s)
    z = [0] * n
    z[0] = n

    l, r = 0, 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def pattern_matching_z(text, pattern):
    # pattern + "$" + text
    combined = pattern + "$" + text
    z = z_function(combined)
    m = len(pattern)

    matches = []
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            matches.append(i - m - 1)

    return matches

def longest_prefix_suffix(s):
    # 가장 긴 proper prefix = suffix
    z = z_function(s)
    n = len(s)

    for length in range(n - 1, 0, -1):
        if z[n - length] == length:
            return length
    return 0""",
        "keywords": ["Z algorithm", "Z function", "prefix matching", "pattern search"],
    },
    {
        "id": "aho-corasick",
        "name": "Aho-Corasick Algorithm",
        "name_ko": "아호-코라식 알고리즘",
        "description": "Efficient multi-pattern string matching using trie with failure links. Finds all patterns in one pass.",
        "description_ko": "실패 링크가 있는 트라이를 사용한 효율적인 다중 패턴 문자열 매칭입니다. 한 번의 순회로 모든 패턴을 찾습니다.",
        "use_cases": [
            "다중 키워드 검색",
            "금칙어 필터링",
            "DNA 서열 매칭",
        ],
        "time_complexity": "O(n + m + z), z=매칭 수",
        "space_complexity": "O(Σm × 알파벳 크기)",
        "example_code": """from collections import deque, defaultdict

class AhoCorasick:
    def __init__(self):
        self.goto = [{}]  # goto 함수
        self.fail = [0]   # failure 함수
        self.output = [[]]  # 각 상태의 출력 패턴

    def add_pattern(self, pattern, idx):
        state = 0
        for c in pattern:
            if c not in self.goto[state]:
                self.goto[state][c] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][c]
        self.output[state].append(idx)

    def build(self):
        q = deque()
        for c, next_state in self.goto[0].items():
            q.append(next_state)

        while q:
            state = q.popleft()
            for c, next_state in self.goto[state].items():
                q.append(next_state)

                # failure 링크 계산
                fail_state = self.fail[state]
                while fail_state and c not in self.goto[fail_state]:
                    fail_state = self.fail[fail_state]

                self.fail[next_state] = self.goto[fail_state].get(c, 0)
                self.output[next_state] += self.output[self.fail[next_state]]

    def search(self, text):
        state = 0
        results = []

        for i, c in enumerate(text):
            while state and c not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(c, 0)

            for pattern_idx in self.output[state]:
                results.append((i, pattern_idx))

        return results""",
        "keywords": ["aho-corasick", "multi-pattern", "trie", "failure link", "automaton"],
    },
    {
        "id": "manacher",
        "name": "Manacher's Algorithm",
        "name_ko": "매나커 알고리즘",
        "description": "Find all palindromic substrings in O(n) time. Uses symmetry of palindromes to avoid redundant comparisons.",
        "description_ko": "O(n) 시간에 모든 팰린드롬 부분 문자열을 찾습니다. 회문의 대칭성을 이용해 중복 비교를 피합니다.",
        "use_cases": [
            "최장 팰린드롬 부분 문자열",
            "팰린드롬 개수 세기",
            "팰린드롬 분할",
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(n)",
        "example_code": """def manacher(s):
    # 문자열 변환: "abc" -> "#a#b#c#"
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = i 중심 팰린드롬 반경

    center = right = 0

    for i in range(n):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])

        # 확장 시도
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0 and
               t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1

        # 경계 업데이트
        if i + p[i] > right:
            center, right = i, i + p[i]

    return p

def longest_palindrome(s):
    p = manacher(s)
    max_len = max(p)
    center = p.index(max_len)

    # 원본 문자열에서 위치 계산
    start = (center - max_len) // 2
    return s[start:start + max_len]

def count_palindromic_substrings(s):
    p = manacher(s)
    # p[i]가 홀수 인덱스면 홀수 길이 팰린드롬
    # p[i]가 짝수 인덱스면 짝수 길이 팰린드롬
    return sum((r + 1) // 2 for r in p)""",
        "keywords": ["manacher", "palindrome", "longest palindrome", "linear time"],
    },
]


class PatternKnowledgeBase:
    """
    Knowledge base for algorithm patterns.
    Provides pattern lookup, embedding, and search functionality.
    """

    def __init__(self, data_path: Path | None = None):
        self.data_path = Path(data_path) if data_path else None
        self._patterns = ALGORITHM_PATTERNS.copy()
        self._embeddings = None
        self._text_embedder = None
        self._code_embedder = None

    @property
    def patterns(self) -> list[dict]:
        """Get all patterns"""
        return self._patterns

    def get_pattern(self, pattern_id: str) -> dict | None:
        """Get a pattern by ID"""
        for pattern in self._patterns:
            if pattern["id"] == pattern_id:
                return pattern
        return None

    def get_patterns_by_keyword(self, keyword: str) -> list[dict]:
        """Find patterns containing a keyword"""
        keyword = keyword.lower()
        results = []
        for pattern in self._patterns:
            keywords = [k.lower() for k in pattern.get("keywords", [])]
            if keyword in keywords or keyword in pattern["name"].lower():
                results.append(pattern)
        return results

    def build_embeddings(self, text_embedder=None, code_embedder=None):
        """
        Build embeddings for all patterns.

        Args:
            text_embedder: TextEmbedder instance for text embeddings
            code_embedder: CodeEmbedder instance for code embeddings
        """
        from code_tutor.ml.embeddings import CodeEmbedder, TextEmbedder

        self._text_embedder = text_embedder or TextEmbedder()
        self._code_embedder = code_embedder or CodeEmbedder()

        # Build text embeddings from descriptions
        texts = []
        for pattern in self._patterns:
            text = (
                f"{pattern['name']}. {pattern['description']}. "
                f"Use cases: {', '.join(pattern['use_cases'])}. "
                f"Keywords: {', '.join(pattern['keywords'])}"
            )
            texts.append(text)

        self._embeddings = {
            "text": self._text_embedder.embed_batch(texts),
            "code": self._code_embedder.embed_batch(
                [p["example_code"] for p in self._patterns]
            ),
        }

        logger.info(f"Built embeddings for {len(self._patterns)} patterns")

    def find_similar_by_text(
        self, query: str, top_k: int = 3, threshold: float = 0.5
    ) -> list[dict]:
        """
        Find similar patterns by text query.

        Args:
            query: Natural language query
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of patterns with similarity scores
        """
        if self._embeddings is None:
            self.build_embeddings()

        query_emb = self._text_embedder.embed(query)
        similarities = query_emb @ self._embeddings["text"].T

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({**self._patterns[idx], "similarity": score})

        return results

    def find_similar_by_code(
        self,
        code: str,
        language: str = "python",
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Find similar patterns by code similarity.

        Args:
            code: Code snippet
            language: Programming language
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of patterns with similarity scores
        """
        if self._embeddings is None:
            self.build_embeddings()

        code_emb = self._code_embedder.embed(code, language)
        similarities = code_emb @ self._embeddings["code"].T

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({**self._patterns[idx], "similarity": score})

        return results

    def save(self, path: Path | None = None):
        """Save patterns to JSON file"""
        save_path = Path(path) if path else self.data_path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self._patterns, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self._patterns)} patterns to {save_path}")

    def load(self, path: Path | None = None):
        """Load patterns from JSON file"""
        load_path = Path(path) if path else self.data_path
        if load_path is None or not load_path.exists():
            logger.warning(f"Pattern file not found: {load_path}")
            return

        with open(load_path, encoding="utf-8") as f:
            self._patterns = json.load(f)

        logger.info(f"Loaded {len(self._patterns)} patterns from {load_path}")

    def to_documents(self) -> list[dict]:
        """
        Convert patterns to document format for vector store.

        Returns:
            List of documents with content and metadata
        """
        documents = []
        for pattern in self._patterns:
            # Create comprehensive document content with Korean emphasis
            # Put Korean name first for better Korean query matching
            content = f"""{pattern["name_ko"]} {pattern["name"]} 알고리즘 패턴

{pattern["description_ko"]}
{pattern["description"]}

사용 사례: {", ".join(pattern["use_cases"])}

시간 복잡도: {pattern["time_complexity"]}
공간 복잡도: {pattern["space_complexity"]}

키워드: {pattern["name_ko"]}, {pattern["name"]}, {", ".join(pattern["keywords"])}
"""
            documents.append(
                {
                    "id": pattern["id"],
                    "content": content,
                    "metadata": {
                        "name": pattern["name"],
                        "name_ko": pattern["name_ko"],
                        "type": "algorithm_pattern",
                    },
                }
            )

        return documents
