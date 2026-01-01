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
