"""Phase 3-6: 트리/그리디/DFS-BFS 문제 통합 시딩 스크립트

Phase 3: 트리 자료구조 (10문제)
Phase 4: 그리디 심화 (10문제)
Phase 5: 트리 DFS/BFS (10문제)
Phase 6: 그래프 DFS/BFS (12문제)

총 42개 문제 추가
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"

# ============== Phase 3: 트리 자료구조 ==============
TREE_STRUCTURE_PROBLEMS = [
    {
        "title": "BST 삽입 구현",
        "description": """이진 탐색 트리(BST)에 값을 삽입하는 함수를 구현하세요.

### 입력
- 트리의 루트와 삽입할 값

### BST 규칙
- 왼쪽 서브트리: 현재 노드보다 작은 값
- 오른쪽 서브트리: 현재 노드보다 큰 값

### 출력
- 삽입 후 트리의 중위 순회 결과

### 예제
```
입력: root = [4, 2, 6], val = 3
출력: [2, 3, 4, 6]
```""",
        "difficulty": "easy",
        "category": "tree",
        "constraints": "0 <= 노드 수 <= 10^4",
        "hints": ["재귀적으로 올바른 위치 탐색", "값이 작으면 왼쪽, 크면 오른쪽"],
        "solution_template": "def solution(nodes: list, val: int) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list, val: int) -> list:
    class TreeNode:
        def __init__(self, v):
            self.val = v
            self.left = self.right = None

    def insert(root, v):
        if not root:
            return TreeNode(v)
        if v < root.val:
            root.left = insert(root.left, v)
        else:
            root.right = insert(root.right, v)
        return root

    def build(arr):
        if not arr:
            return None
        root = TreeNode(arr[0])
        for v in arr[1:]:
            insert(root, v)
        return root

    def inorder(node, res):
        if node:
            inorder(node.left, res)
            res.append(node.val)
            inorder(node.right, res)

    root = build(nodes)
    insert(root, val)
    result = []
    inorder(root, result)
    return result""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bst-insertion"],
        "pattern_explanation": "BST 삽입은 트리 구조의 기초입니다. 값을 비교하며 왼쪽/오른쪽으로 이동하여 적절한 위치에 노드를 추가합니다.",
        "approach_hint": "재귀적 BST 삽입",
        "time_complexity_hint": "O(h) - h는 트리 높이",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[4, 2, 6]\n3", "output": "[2, 3, 4, 6]", "is_sample": True},
            {"input": "[5]\n3", "output": "[3, 5]", "is_sample": False},
        ]
    },
    {
        "title": "BST 검색",
        "description": """BST에서 특정 값을 검색하세요.

### 입력
- BST의 중위 순회 결과 `nodes`
- 검색할 값 `target`

### 출력
- 값이 존재하면 True, 없으면 False

### 예제
```
입력: nodes = [2, 3, 4, 6], target = 3
출력: True
```""",
        "difficulty": "easy",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 10^4",
        "hints": ["BST는 정렬된 배열과 유사", "이진 탐색 활용 가능"],
        "solution_template": "def solution(nodes: list, target: int) -> bool:\n    pass",
        "reference_solution": """def solution(nodes: list, target: int) -> bool:
    left, right = 0, len(nodes) - 1
    while left <= right:
        mid = (left + right) // 2
        if nodes[mid] == target:
            return True
        elif nodes[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bst-search", "binary-search"],
        "pattern_explanation": "BST의 중위 순회는 정렬된 배열입니다. 따라서 이진 탐색으로 O(log n)에 검색할 수 있습니다.",
        "approach_hint": "중위 순회 = 정렬 배열 → 이진 탐색",
        "time_complexity_hint": "O(log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[2, 3, 4, 6]\n3", "output": "True", "is_sample": True},
            {"input": "[2, 3, 4, 6]\n5", "output": "False", "is_sample": True},
        ]
    },
    {
        "title": "완전 이진 트리 판별",
        "description": """주어진 트리가 완전 이진 트리인지 판별하세요.

완전 이진 트리: 마지막 레벨을 제외한 모든 레벨이 완전히 채워지고, 마지막 레벨은 왼쪽부터 채워진 트리

### 입력
- 레벨 순회로 주어진 트리 `nodes` (None은 빈 노드)

### 출력
- 완전 이진 트리면 True, 아니면 False

### 예제
```
입력: [1, 2, 3, 4, 5]
출력: True

입력: [1, 2, 3, None, 4]
출력: False
```""",
        "difficulty": "easy",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 100",
        "hints": ["None을 만난 후에는 다른 노드가 없어야 함", "BFS로 레벨 순회"],
        "solution_template": "def solution(nodes: list) -> bool:\n    pass",
        "reference_solution": """def solution(nodes: list) -> bool:
    if not nodes:
        return True

    found_null = False
    for node in nodes:
        if node is None:
            found_null = True
        else:
            if found_null:
                return False
    return True""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["complete-binary-tree"],
        "pattern_explanation": "완전 이진 트리는 레벨 순회에서 None이 나온 후에 다른 노드가 없어야 합니다.",
        "approach_hint": "레벨 순회에서 None 이후 검사",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]", "output": "True", "is_sample": True},
            {"input": "[1, 2, 3, None, 4]", "output": "False", "is_sample": True},
        ]
    },
    {
        "title": "BST 유효성 검사",
        "description": """주어진 트리가 유효한 BST인지 검사하세요.

### 입력
- 전위 순회로 주어진 트리 `preorder`

### 출력
- 유효한 BST면 True, 아니면 False

### 예제
```
입력: [5, 2, 1, 4, 7]
출력: True (5를 루트로 왼쪽에 2,1,4 오른쪽에 7)
```""",
        "difficulty": "medium",
        "category": "tree",
        "constraints": "1 <= len(preorder) <= 10^4",
        "hints": ["스택을 사용해 전위 순회 검증", "현재 하한값 추적"],
        "solution_template": "def solution(preorder: list) -> bool:\n    pass",
        "reference_solution": """def solution(preorder: list) -> bool:
    stack = []
    lower_bound = float('-inf')

    for val in preorder:
        if val < lower_bound:
            return False

        while stack and val > stack[-1]:
            lower_bound = stack.pop()

        stack.append(val)

    return True""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bst-validation"],
        "pattern_explanation": "BST 유효성은 스택으로 검증할 수 있습니다. 오른쪽 서브트리로 이동할 때 하한값이 갱신됩니다.",
        "approach_hint": "스택 + 하한값 추적",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[5, 2, 1, 4, 7]", "output": "True", "is_sample": True},
            {"input": "[5, 2, 6, 4, 7]", "output": "False", "is_sample": False},
        ]
    },
    {
        "title": "세그먼트 트리 구간합",
        "description": """세그먼트 트리를 사용해 구간 합을 구하세요.

### 입력
- 정수 배열 `nums`
- 쿼리 배열 `queries` (각 쿼리는 [left, right])

### 출력
- 각 쿼리의 구간 합 리스트

### 예제
```
입력: nums = [1, 3, 5, 7, 9], queries = [[0, 2], [1, 4]]
출력: [9, 24]
```""",
        "difficulty": "medium",
        "category": "tree",
        "constraints": "1 <= len(nums) <= 10^4",
        "hints": ["세그먼트 트리 빌드 O(n)", "쿼리 O(log n)"],
        "solution_template": "def solution(nums: list, queries: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, queries: list) -> list:
    n = len(nums)
    tree = [0] * (4 * n)

    def build(node, start, end):
        if start == end:
            tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            build(2*node, start, mid)
            build(2*node+1, mid+1, end)
            tree[node] = tree[2*node] + tree[2*node+1]

    def query(node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        return query(2*node, start, mid, l, r) + query(2*node+1, mid+1, end, l, r)

    build(1, 0, n-1)
    return [query(1, 0, n-1, l, r) for l, r in queries]""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["segment-tree"],
        "pattern_explanation": "세그먼트 트리는 구간 쿼리를 O(log n)에 처리합니다. 각 노드는 해당 구간의 합(또는 최대/최소)을 저장합니다.",
        "approach_hint": "세그먼트 트리 빌드 + 쿼리",
        "time_complexity_hint": "O(n + q log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 3, 5, 7, 9]\n[[0, 2], [1, 4]]", "output": "[9, 24]", "is_sample": True},
        ]
    },
    {
        "title": "펜윅 트리 (BIT)",
        "description": """펜윅 트리(Binary Indexed Tree)로 구간 합과 업데이트를 구현하세요.

### 입력
- 정수 배열 `nums`
- 연산 배열 `operations` ([type, ...])
  - type=0: [0, i, val] → nums[i] += val
  - type=1: [1, l, r] → sum(nums[l:r+1])

### 출력
- type=1 연산의 결과 리스트

### 예제
```
입력: nums = [1, 2, 3, 4], ops = [[1, 0, 2], [0, 1, 3], [1, 0, 2]]
출력: [6, 9]
```""",
        "difficulty": "medium",
        "category": "tree",
        "constraints": "1 <= len(nums) <= 10^5",
        "hints": ["i & (-i)로 최하위 비트 추출", "업데이트: i += i & (-i)", "쿼리: i -= i & (-i)"],
        "solution_template": "def solution(nums: list, operations: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, operations: list) -> list:
    n = len(nums)
    bit = [0] * (n + 1)

    def update(i, delta):
        i += 1
        while i <= n:
            bit[i] += delta
            i += i & (-i)

    def prefix_sum(i):
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    def range_sum(l, r):
        return prefix_sum(r) - (prefix_sum(l-1) if l > 0 else 0)

    for i, v in enumerate(nums):
        update(i, v)

    result = []
    for op in operations:
        if op[0] == 0:
            update(op[1], op[2])
        else:
            result.append(range_sum(op[1], op[2]))
    return result""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["fenwick-tree", "bit"],
        "pattern_explanation": "펜윅 트리는 세그먼트 트리보다 구현이 간단하고 상수가 작습니다. 비트 연산으로 부모/자식 관계를 계산합니다.",
        "approach_hint": "i & (-i)로 최하위 비트 활용",
        "time_complexity_hint": "O((n + q) log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4]\n[[1, 0, 2], [0, 1, 3], [1, 0, 2]]", "output": "[6, 9]", "is_sample": True},
        ]
    },
    {
        "title": "AVL 트리 높이",
        "description": """BST 삽입 순서가 주어질 때, AVL 트리로 구성했을 경우의 높이를 구하세요.

AVL 트리: 모든 노드에서 왼쪽과 오른쪽 서브트리의 높이 차가 1 이하인 BST

### 입력
- 삽입할 값들의 배열 `values`

### 출력
- AVL 트리의 높이 (루트의 높이 = 1)

### 예제
```
입력: [10, 20, 30, 40, 50]
출력: 3

일반 BST면 높이 5지만, AVL 회전으로 균형을 맞춰 높이 3
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= len(values) <= 10^4",
        "hints": ["AVL 삽입 후 회전", "LL, RR, LR, RL 4가지 회전"],
        "solution_template": "def solution(values: list) -> int:\n    pass",
        "reference_solution": """def solution(values: list) -> int:
    class Node:
        def __init__(self, v):
            self.val = v
            self.left = self.right = None
            self.height = 1

    def height(n):
        return n.height if n else 0

    def balance(n):
        return height(n.left) - height(n.right) if n else 0

    def update_height(n):
        n.height = 1 + max(height(n.left), height(n.right))

    def rotate_right(y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        update_height(y)
        update_height(x)
        return x

    def rotate_left(x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        update_height(x)
        update_height(y)
        return y

    def insert(root, val):
        if not root:
            return Node(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)

        update_height(root)
        b = balance(root)

        if b > 1 and val < root.left.val:
            return rotate_right(root)
        if b < -1 and val > root.right.val:
            return rotate_left(root)
        if b > 1 and val > root.left.val:
            root.left = rotate_left(root.left)
            return rotate_right(root)
        if b < -1 and val < root.right.val:
            root.right = rotate_right(root.right)
            return rotate_left(root)

        return root

    root = None
    for v in values:
        root = insert(root, v)
    return height(root)""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["avl-tree"],
        "pattern_explanation": "AVL 트리는 삽입/삭제 시 회전으로 균형을 유지합니다. 4가지 회전(LL, RR, LR, RL)을 통해 높이 차이를 1 이하로 유지합니다.",
        "approach_hint": "균형 인수 체크 후 회전",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[10, 20, 30, 40, 50]", "output": "3", "is_sample": True},
            {"input": "[1]", "output": "1", "is_sample": False},
        ]
    },
    {
        "title": "트라이 자동완성",
        "description": """트라이(Trie)를 사용해 자동완성 기능을 구현하세요.

### 입력
- 단어 리스트 `words`
- 접두사 `prefix`

### 출력
- 해당 접두사로 시작하는 모든 단어 (사전순)

### 예제
```
입력: words = ["apple", "app", "application", "banana"], prefix = "app"
출력: ["app", "apple", "application"]
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= len(words) <= 10^4",
        "hints": ["트라이 구조 구축", "DFS로 prefix 노드에서 모든 단어 수집"],
        "solution_template": "def solution(words: list, prefix: str) -> list:\n    pass",
        "reference_solution": """def solution(words: list, prefix: str) -> list:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()

    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    node = root
    for char in prefix:
        if char not in node.children:
            return []
        node = node.children[char]

    result = []
    def dfs(n, path):
        if n.is_end:
            result.append(prefix + path)
        for c in sorted(n.children.keys()):
            dfs(n.children[c], path + c)

    dfs(node, "")
    return result""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["trie", "autocomplete"],
        "pattern_explanation": "트라이는 문자열 집합을 효율적으로 저장하고 검색하는 트리 구조입니다. 각 노드는 문자를 나타내고, 루트에서 리프까지의 경로가 단어를 형성합니다.",
        "approach_hint": "트라이 구축 + DFS 수집",
        "time_complexity_hint": "O(총 문자 수)",
        "space_complexity_hint": "O(총 문자 수)",
        "test_cases": [
            {"input": '["apple", "app", "application", "banana"]\napp', "output": '["app", "apple", "application"]', "is_sample": True},
        ]
    },
    {
        "title": "레이지 세그먼트 트리",
        "description": """구간 업데이트와 구간 쿼리를 지원하는 레이지 세그먼트 트리를 구현하세요.

### 입력
- 정수 배열 `nums`
- 연산 배열 `operations`
  - [0, l, r, val]: 구간 [l, r]에 val 더하기
  - [1, l, r]: 구간 [l, r] 합 쿼리

### 출력
- 쿼리 결과 리스트

### 예제
```
입력: nums = [1, 2, 3, 4], ops = [[1, 0, 2], [0, 0, 2, 5], [1, 0, 2]]
출력: [6, 21]
```""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= len(nums) <= 10^5",
        "hints": ["lazy 배열로 미뤄진 업데이트 저장", "쿼리/업데이트 시 propagate"],
        "solution_template": "def solution(nums: list, operations: list) -> list:\n    pass",
        "reference_solution": """def solution(nums: list, operations: list) -> list:
    n = len(nums)
    tree = [0] * (4 * n)
    lazy = [0] * (4 * n)

    def build(node, start, end):
        if start == end:
            tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            build(2*node, start, mid)
            build(2*node+1, mid+1, end)
            tree[node] = tree[2*node] + tree[2*node+1]

    def propagate(node, start, end):
        if lazy[node] != 0:
            tree[node] += lazy[node] * (end - start + 1)
            if start != end:
                lazy[2*node] += lazy[node]
                lazy[2*node+1] += lazy[node]
            lazy[node] = 0

    def update(node, start, end, l, r, val):
        propagate(node, start, end)
        if r < start or end < l:
            return
        if l <= start and end <= r:
            lazy[node] += val
            propagate(node, start, end)
            return
        mid = (start + end) // 2
        update(2*node, start, mid, l, r, val)
        update(2*node+1, mid+1, end, l, r, val)
        tree[node] = tree[2*node] + tree[2*node+1]

    def query(node, start, end, l, r):
        propagate(node, start, end)
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        return query(2*node, start, mid, l, r) + query(2*node+1, mid+1, end, l, r)

    build(1, 0, n-1)
    result = []
    for op in operations:
        if op[0] == 0:
            update(1, 0, n-1, op[1], op[2], op[3])
        else:
            result.append(query(1, 0, n-1, op[1], op[2]))
    return result""",
        "time_limit_ms": 5000,
        "memory_limit_mb": 512,
        "pattern_ids": ["lazy-segment-tree"],
        "pattern_explanation": "레이지 세그먼트 트리는 구간 업데이트를 O(log n)에 처리합니다. 업데이트를 즉시 전파하지 않고 lazy 배열에 저장해뒀다가 필요할 때 전파합니다.",
        "approach_hint": "lazy 배열 + propagate 함수",
        "time_complexity_hint": "O((n + q) log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4]\n[[1, 0, 2], [0, 0, 2, 5], [1, 0, 2]]", "output": "[6, 21]", "is_sample": True},
        ]
    },
    {
        "title": "레드-블랙 트리 색상 검증",
        "description": """주어진 트리가 레드-블랙 트리의 속성을 만족하는지 검증하세요.

레드-블랙 트리 속성:
1. 모든 노드는 빨강 또는 검정
2. 루트는 검정
3. 빨간 노드의 자식은 모두 검정
4. 루트에서 모든 리프까지 검정 노드 수가 동일

### 입력
- 노드 배열 `nodes` (각 노드: [val, color, left_idx, right_idx])
- color: 0=검정, 1=빨강

### 출력
- 유효하면 True, 아니면 False""",
        "difficulty": "hard",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["루트 색상 검사", "빨강 노드의 자식 검사", "검정 높이 일관성 검사"],
        "solution_template": "def solution(nodes: list) -> bool:\n    pass",
        "reference_solution": """def solution(nodes: list) -> bool:
    if not nodes:
        return True

    if nodes[0][1] != 0:
        return False

    def check(idx):
        if idx == -1:
            return 1

        val, color, left, right = nodes[idx]

        if color == 1:
            if left != -1 and nodes[left][1] == 1:
                return -1
            if right != -1 and nodes[right][1] == 1:
                return -1

        left_bh = check(left)
        right_bh = check(right)

        if left_bh == -1 or right_bh == -1 or left_bh != right_bh:
            return -1

        return left_bh + (1 if color == 0 else 0)

    return check(0) != -1""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["red-black-tree"],
        "pattern_explanation": "레드-블랙 트리는 AVL보다 회전이 적어 삽입/삭제가 빠릅니다. 색상 규칙으로 균형을 보장합니다.",
        "approach_hint": "DFS로 모든 속성 검증",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[[10, 0, 1, 2], [5, 1, -1, -1], [15, 1, -1, -1]]", "output": "True", "is_sample": True},
        ]
    },
]

# ============== Phase 4: 그리디 심화 ==============
GREEDY_PROBLEMS = [
    {
        "title": "동전 최대 수집",
        "description": """N개의 동전이 일렬로 놓여 있습니다. 각 동전의 가치가 주어질 때, K개의 동전을 선택해 최대 가치를 얻으세요.

### 입력
- 동전 가치 배열 `coins`
- 선택할 동전 수 `k`

### 출력
- 최대 가치 합

### 예제
```
입력: coins = [5, 3, 1, 4, 2], k = 3
출력: 12 (5 + 4 + 3)
```""",
        "difficulty": "easy",
        "category": "greedy",
        "constraints": "1 <= k <= len(coins) <= 10^5",
        "hints": ["가장 큰 k개 선택", "정렬 후 상위 k개 합산"],
        "solution_template": "def solution(coins: list, k: int) -> int:\n    pass",
        "reference_solution": """def solution(coins: list, k: int) -> int:
    return sum(sorted(coins, reverse=True)[:k])""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["greedy-max-selection"],
        "pattern_explanation": "최대/최소 선택 문제에서 그리디는 정렬 후 상위/하위 k개를 선택하는 것입니다.",
        "approach_hint": "정렬 후 상위 k개 선택",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[5, 3, 1, 4, 2]\n3", "output": "12", "is_sample": True},
        ]
    },
    {
        "title": "최소 대기 시간",
        "description": """N명의 고객이 대기 중입니다. 각 고객의 서비스 시간이 주어질 때, 총 대기 시간을 최소화하세요.

### 입력
- 서비스 시간 배열 `times`

### 출력
- 최소 총 대기 시간

### 예제
```
입력: [3, 1, 2]
출력: 4

순서 [1, 2, 3]: 0 + 1 + 3 = 4
```""",
        "difficulty": "easy",
        "category": "greedy",
        "constraints": "1 <= len(times) <= 10^5",
        "hints": ["짧은 작업 먼저 (SJF)", "앞 사람의 시간이 뒤 사람들 대기에 영향"],
        "solution_template": "def solution(times: list) -> int:\n    pass",
        "reference_solution": """def solution(times: list) -> int:
    times = sorted(times)
    total_wait = 0
    cumulative = 0
    for i, t in enumerate(times[:-1]):
        cumulative += t
        total_wait += cumulative
    return total_wait""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["greedy-scheduling"],
        "pattern_explanation": "대기 시간 최소화는 짧은 작업 먼저(SJF) 원칙을 따릅니다. 앞 사람의 시간이 뒤 사람 모두의 대기에 추가됩니다.",
        "approach_hint": "짧은 작업 먼저 (SJF)",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[3, 1, 2]", "output": "4", "is_sample": True},
            {"input": "[5, 5, 5]", "output": "15", "is_sample": False},
        ]
    },
    {
        "title": "회의실 배정",
        "description": """회의 시간표가 주어질 때, 최대 몇 개의 회의를 진행할 수 있는지 구하세요.

### 입력
- 회의 배열 `meetings` ([시작, 끝])

### 출력
- 최대 회의 수

### 예제
```
입력: [[1, 4], [2, 3], [3, 5], [4, 6]]
출력: 3 ([2,3], [3,5] 또는 [2,3], [4,6])
```""",
        "difficulty": "easy",
        "category": "greedy",
        "constraints": "1 <= len(meetings) <= 10^5",
        "hints": ["끝나는 시간 기준 정렬", "이전 회의 끝 시간 이후에 시작하는 회의 선택"],
        "solution_template": "def solution(meetings: list) -> int:\n    pass",
        "reference_solution": """def solution(meetings: list) -> int:
    meetings.sort(key=lambda x: x[1])
    count = 0
    end = 0
    for start, finish in meetings:
        if start >= end:
            count += 1
            end = finish
    return count""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["interval-scheduling"],
        "pattern_explanation": "활동 선택 문제는 끝나는 시간 기준으로 정렬 후 겹치지 않는 것을 선택합니다.",
        "approach_hint": "끝나는 시간 기준 정렬",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[[1, 4], [2, 3], [3, 5], [4, 6]]", "output": "3", "is_sample": True},
        ]
    },
    {
        "title": "작업 스케줄링",
        "description": """각 작업의 마감시간과 이익이 주어집니다. 각 작업은 1단위 시간이 걸립니다. 최대 이익을 구하세요.

### 입력
- 작업 배열 `jobs` ([마감시간, 이익])

### 출력
- 최대 이익

### 예제
```
입력: [[2, 100], [1, 19], [2, 27], [1, 25], [3, 15]]
출력: 142 (작업 0, 2, 4 선택)
```""",
        "difficulty": "medium",
        "category": "greedy",
        "constraints": "1 <= len(jobs) <= 10^4",
        "hints": ["이익 기준 내림차순 정렬", "마감 직전 슬롯에 배치"],
        "solution_template": "def solution(jobs: list) -> int:\n    pass",
        "reference_solution": """def solution(jobs: list) -> int:
    jobs.sort(key=lambda x: -x[1])
    max_deadline = max(j[0] for j in jobs)
    slots = [False] * (max_deadline + 1)
    total = 0

    for deadline, profit in jobs:
        for slot in range(deadline, 0, -1):
            if not slots[slot]:
                slots[slot] = True
                total += profit
                break

    return total""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["job-scheduling"],
        "pattern_explanation": "작업 스케줄링은 이익이 높은 것부터 가능한 늦은 슬롯에 배치합니다.",
        "approach_hint": "이익 기준 정렬 + 슬롯 배치",
        "time_complexity_hint": "O(n * d)",
        "space_complexity_hint": "O(d)",
        "test_cases": [
            {"input": "[[2, 100], [1, 19], [2, 27], [1, 25], [3, 15]]", "output": "142", "is_sample": True},
        ]
    },
    {
        "title": "분할 가능 배낭",
        "description": """물건을 일부만 담을 수 있는 배낭 문제입니다. 최대 가치를 구하세요.

### 입력
- 물건 배열 `items` ([가치, 무게])
- 배낭 용량 `capacity`

### 출력
- 최대 가치 (소수점 2자리)

### 예제
```
입력: items = [[60, 10], [100, 20], [120, 30]], capacity = 50
출력: 240.00
```""",
        "difficulty": "medium",
        "category": "greedy",
        "constraints": "1 <= len(items) <= 10^4",
        "hints": ["무게당 가치 계산", "무게당 가치 기준 내림차순 정렬"],
        "solution_template": "def solution(items: list, capacity: int) -> float:\n    pass",
        "reference_solution": """def solution(items: list, capacity: int) -> float:
    items = [(v, w, v/w) for v, w in items]
    items.sort(key=lambda x: -x[2])

    total_value = 0
    remaining = capacity

    for value, weight, ratio in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += ratio * remaining
            break

    return round(total_value, 2)""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["fractional-knapsack"],
        "pattern_explanation": "분할 가능 배낭은 무게당 가치가 높은 것부터 담습니다. 0-1 배낭과 달리 그리디로 최적해를 구할 수 있습니다.",
        "approach_hint": "무게당 가치 기준 그리디",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[[60, 10], [100, 20], [120, 30]]\n50", "output": "240.0", "is_sample": True},
        ]
    },
    {
        "title": "허프만 인코딩",
        "description": """문자 빈도가 주어질 때, 허프만 코드의 총 비트 수를 구하세요.

### 입력
- 빈도 배열 `freqs`

### 출력
- 총 인코딩 비트 수

### 예제
```
입력: [5, 9, 12, 13, 16, 45]
출력: 224
```""",
        "difficulty": "medium",
        "category": "greedy",
        "constraints": "2 <= len(freqs) <= 10^4",
        "hints": ["최소 힙 사용", "가장 작은 두 개를 합쳐 새 노드 생성"],
        "solution_template": "def solution(freqs: list) -> int:\n    pass",
        "reference_solution": """def solution(freqs: list) -> int:
    import heapq
    heap = freqs[:]
    heapq.heapify(heap)
    total = 0

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = a + b
        total += merged
        heapq.heappush(heap, merged)

    return total""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["huffman-encoding"],
        "pattern_explanation": "허프만 코딩은 빈도가 낮은 것부터 합칩니다. 힙으로 가장 작은 두 개를 반복적으로 합칩니다.",
        "approach_hint": "최소 힙으로 두 개씩 합치기",
        "time_complexity_hint": "O(n log n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[5, 9, 12, 13, 16, 45]", "output": "224", "is_sample": True},
        ]
    },
    {
        "title": "네트워크 최단 경로 (다익스트라 기초)",
        "description": """가중치 그래프에서 시작점으로부터 모든 노드까지의 최단 거리를 구하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges` ([from, to, weight])
- 시작점 `start`

### 출력
- 각 노드까지의 최단 거리 리스트 (도달 불가면 -1)

### 예제
```
입력: n=4, edges=[[0,1,1],[0,2,4],[1,2,2],[2,3,1]], start=0
출력: [0, 1, 3, 4]
```""",
        "difficulty": "medium",
        "category": "greedy",
        "constraints": "1 <= n <= 1000",
        "hints": ["우선순위 큐 사용", "최소 거리 노드부터 처리"],
        "solution_template": "def solution(n: int, edges: list, start: int) -> list:\n    pass",
        "reference_solution": """def solution(n: int, edges: list, start: int) -> list:
    import heapq
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return [d if d != float('inf') else -1 for d in dist]""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["dijkstra"],
        "pattern_explanation": "다익스트라 알고리즘은 그리디하게 최단 거리 노드를 선택합니다. 우선순위 큐로 O((V+E)log V) 시간에 동작합니다.",
        "approach_hint": "우선순위 큐 + 최단 거리 갱신",
        "time_complexity_hint": "O((V+E) log V)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "4\n[[0,1,1],[0,2,4],[1,2,2],[2,3,1]]\n0", "output": "[0, 1, 3, 4]", "is_sample": True},
        ]
    },
    {
        "title": "최소 신장 트리 (프림)",
        "description": """프림 알고리즘으로 최소 신장 트리의 가중치 합을 구하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges` ([from, to, weight])

### 출력
- MST 가중치 합

### 예제
```
입력: n=4, edges=[[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]]
출력: 4 (간선 0-1, 1-2, 2-3 선택)
```""",
        "difficulty": "hard",
        "category": "greedy",
        "constraints": "1 <= n <= 10^4",
        "hints": ["시작점에서 시작", "연결된 간선 중 최소 가중치 선택"],
        "solution_template": "def solution(n: int, edges: list) -> int:\n    pass",
        "reference_solution": """def solution(n: int, edges: list) -> int:
    import heapq
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))

    visited = [False] * n
    heap = [(0, 0)]
    total = 0
    count = 0

    while heap and count < n:
        w, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        count += 1

        for weight, v in graph[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v))

    return total if count == n else -1""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["prim-mst"],
        "pattern_explanation": "프림 알고리즘은 하나의 정점에서 시작해 MST를 점진적으로 확장합니다. 힙으로 최소 가중치 간선을 선택합니다.",
        "approach_hint": "시작점에서 최소 간선 선택 확장",
        "time_complexity_hint": "O(E log V)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "4\n[[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]]", "output": "4", "is_sample": True},
        ]
    },
    {
        "title": "작업 스케줄러",
        "description": """작업과 쿨다운이 주어질 때, 모든 작업을 완료하는 최소 시간을 구하세요.

### 입력
- 작업 배열 `tasks` (대문자 알파벳)
- 쿨다운 `n`

### 출력
- 최소 시간

### 예제
```
입력: tasks = ["A","A","A","B","B","B"], n = 2
출력: 8

A -> B -> idle -> A -> B -> idle -> A -> B
```""",
        "difficulty": "hard",
        "category": "greedy",
        "constraints": "1 <= len(tasks) <= 10^4",
        "hints": ["가장 빈번한 작업 기준", "(max_count - 1) * (n + 1) + 최대 빈도 작업 수"],
        "solution_template": "def solution(tasks: list, n: int) -> int:\n    pass",
        "reference_solution": """def solution(tasks: list, n: int) -> int:
    from collections import Counter
    counts = Counter(tasks)
    max_count = max(counts.values())
    max_count_tasks = sum(1 for c in counts.values() if c == max_count)

    result = (max_count - 1) * (n + 1) + max_count_tasks
    return max(result, len(tasks))""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["task-scheduler"],
        "pattern_explanation": "가장 빈번한 작업이 전체 시간을 결정합니다. 그 사이에 다른 작업을 배치하거나 idle을 넣습니다.",
        "approach_hint": "최대 빈도 기준 계산",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": '["A","A","A","B","B","B"]\n2', "output": "8", "is_sample": True},
        ]
    },
    {
        "title": "사탕 배분",
        "description": """어린이들의 평점에 따라 사탕을 배분합니다. 이웃보다 높은 평점이면 더 많은 사탕을 받아야 합니다.

### 입력
- 평점 배열 `ratings`

### 출력
- 최소 사탕 수

### 예제
```
입력: [1, 0, 2]
출력: 5 (사탕: [2, 1, 2])
```""",
        "difficulty": "hard",
        "category": "greedy",
        "constraints": "1 <= len(ratings) <= 10^4",
        "hints": ["왼쪽 → 오른쪽 순회", "오른쪽 → 왼쪽 순회", "두 값 중 최대 선택"],
        "solution_template": "def solution(ratings: list) -> int:\n    pass",
        "reference_solution": """def solution(ratings: list) -> int:
    n = len(ratings)
    candies = [1] * n

    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1

    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)

    return sum(candies)""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["two-pass-greedy"],
        "pattern_explanation": "양방향 그리디: 왼쪽에서 오른쪽, 오른쪽에서 왼쪽으로 두 번 순회하며 조건을 만족시킵니다.",
        "approach_hint": "양방향 순회로 조건 만족",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 0, 2]", "output": "5", "is_sample": True},
            {"input": "[1, 2, 2]", "output": "4", "is_sample": False},
        ]
    },
]

# ============== Phase 5: 트리 DFS/BFS ==============
TREE_TRAVERSAL_PROBLEMS = [
    {
        "title": "트리 중위 순회",
        "description": """이진 트리의 중위 순회 결과를 반환하세요.

### 입력
- 레벨 순서로 주어진 트리 `nodes` (None은 빈 노드)

### 출력
- 중위 순회 결과

### 예제
```
입력: [1, 2, 3, 4, 5]
출력: [4, 2, 5, 1, 3]
```""",
        "difficulty": "easy",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["왼쪽 → 루트 → 오른쪽", "재귀 또는 스택 사용"],
        "solution_template": "def solution(nodes: list) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list) -> list:
    def inorder(i):
        if i >= len(nodes) or nodes[i] is None:
            return
        inorder(2*i + 1)
        result.append(nodes[i])
        inorder(2*i + 2)

    result = []
    inorder(0)
    return result""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-inorder"],
        "pattern_explanation": "중위 순회는 왼쪽-루트-오른쪽 순서입니다. BST의 경우 정렬된 순서가 됩니다.",
        "approach_hint": "왼쪽 → 루트 → 오른쪽",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]", "output": "[4, 2, 5, 1, 3]", "is_sample": True},
        ]
    },
    {
        "title": "트리 레벨 합계",
        "description": """각 레벨의 노드 합을 구하세요.

### 입력
- 레벨 순서로 주어진 트리 `nodes`

### 출력
- 각 레벨의 합 리스트

### 예제
```
입력: [1, 2, 3, 4, 5, 6, 7]
출력: [1, 5, 22]
```""",
        "difficulty": "easy",
        "category": "bfs",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["BFS로 레벨별 처리", "각 레벨에서 합산"],
        "solution_template": "def solution(nodes: list) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list) -> list:
    from collections import deque
    if not nodes or nodes[0] is None:
        return []

    result = []
    queue = deque([0])

    while queue:
        level_sum = 0
        level_size = len(queue)

        for _ in range(level_size):
            i = queue.popleft()
            if i < len(nodes) and nodes[i] is not None:
                level_sum += nodes[i]
                left, right = 2*i + 1, 2*i + 2
                if left < len(nodes):
                    queue.append(left)
                if right < len(nodes):
                    queue.append(right)

        if level_sum > 0:
            result.append(level_sum)

    return result""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-bfs-level"],
        "pattern_explanation": "BFS로 레벨별 처리하면서 각 레벨의 합을 구합니다.",
        "approach_hint": "BFS 레벨 순회",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5, 6, 7]", "output": "[1, 5, 22]", "is_sample": True},
        ]
    },
    {
        "title": "리프 노드 개수",
        "description": """트리의 리프 노드 개수를 구하세요.

### 입력
- 레벨 순서로 주어진 트리 `nodes`

### 출력
- 리프 노드 개수

### 예제
```
입력: [1, 2, 3, 4, 5]
출력: 3 (노드 3, 4, 5)
```""",
        "difficulty": "easy",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["자식이 없는 노드 카운트", "왼쪽, 오른쪽 자식 인덱스 확인"],
        "solution_template": "def solution(nodes: list) -> int:\n    pass",
        "reference_solution": """def solution(nodes: list) -> int:
    count = 0
    n = len(nodes)

    for i in range(n):
        if nodes[i] is None:
            continue
        left, right = 2*i + 1, 2*i + 2
        has_left = left < n and nodes[left] is not None
        has_right = right < n and nodes[right] is not None
        if not has_left and not has_right:
            count += 1

    return count""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-leaf-count"],
        "pattern_explanation": "리프 노드는 자식이 없는 노드입니다. 각 노드의 자식 존재 여부를 확인합니다.",
        "approach_hint": "자식 없는 노드 카운트",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": "[1, 2, 3, 4, 5]", "output": "3", "is_sample": True},
        ]
    },
    {
        "title": "최소 공통 조상 (LCA)",
        "description": """두 노드의 최소 공통 조상을 찾으세요.

### 입력
- 레벨 순서 트리 `nodes`
- 두 노드 값 `p`, `q`

### 출력
- LCA의 값

### 예제
```
입력: nodes = [3, 5, 1, 6, 2, 0, 8], p = 5, q = 1
출력: 3
```""",
        "difficulty": "medium",
        "category": "tree",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["각 노드의 조상 경로 추적", "두 경로의 첫 교차점"],
        "solution_template": "def solution(nodes: list, p: int, q: int) -> int:\n    pass",
        "reference_solution": """def solution(nodes: list, p: int, q: int) -> int:
    def find_index(val):
        for i, v in enumerate(nodes):
            if v == val:
                return i
        return -1

    def get_ancestors(idx):
        ancestors = []
        while idx >= 0:
            ancestors.append(nodes[idx])
            idx = (idx - 1) // 2 if idx > 0 else -1
        return ancestors

    p_ancestors = set(get_ancestors(find_index(p)))
    q_path = get_ancestors(find_index(q))

    for ancestor in q_path:
        if ancestor in p_ancestors:
            return ancestor

    return nodes[0]""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["lca"],
        "pattern_explanation": "LCA는 두 노드의 조상 경로를 구해 첫 교차점을 찾습니다.",
        "approach_hint": "조상 경로 교차점 찾기",
        "time_complexity_hint": "O(h)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[3, 5, 1, 6, 2, 0, 8]\n5\n1", "output": "3", "is_sample": True},
        ]
    },
    {
        "title": "트리 경로 합",
        "description": """루트에서 리프까지의 경로 중 합이 target인 경로가 있는지 확인하세요.

### 입력
- 트리 `nodes`
- 목표 합 `target`

### 출력
- 존재하면 True, 없으면 False

### 예제
```
입력: nodes = [5, 4, 8, 11, None, 13, 4], target = 22
출력: True (5 → 4 → 11 = 20... 예제 수정 필요)
```""",
        "difficulty": "medium",
        "category": "dfs",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["DFS로 경로 탐색", "현재까지의 합 전달"],
        "solution_template": "def solution(nodes: list, target: int) -> bool:\n    pass",
        "reference_solution": """def solution(nodes: list, target: int) -> bool:
    if not nodes or nodes[0] is None:
        return False

    def dfs(i, current_sum):
        if i >= len(nodes) or nodes[i] is None:
            return False

        current_sum += nodes[i]
        left, right = 2*i + 1, 2*i + 2
        is_leaf = (left >= len(nodes) or nodes[left] is None) and \\
                  (right >= len(nodes) or nodes[right] is None)

        if is_leaf:
            return current_sum == target

        return dfs(left, current_sum) or dfs(right, current_sum)

    return dfs(0, 0)""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["path-sum"],
        "pattern_explanation": "DFS로 루트에서 리프까지 경로를 탐색하며 합을 누적합니다.",
        "approach_hint": "DFS + 누적합",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[5, 4, 8, 11, None, 13, 4]\n22", "output": "True", "is_sample": True},
        ]
    },
    {
        "title": "트리 직렬화",
        "description": """트리를 문자열로 직렬화하고 역직렬화하세요.

### 입력
- 레벨 순서 트리 `nodes`

### 출력
- 직렬화 후 역직렬화한 결과가 원본과 같으면 True

### 예제
```
입력: [1, 2, 3, None, None, 4, 5]
출력: True
```""",
        "difficulty": "medium",
        "category": "tree",
        "constraints": "len(nodes) <= 1000",
        "hints": ["None을 특수 문자로 표현", "BFS로 레벨 순회"],
        "solution_template": "def solution(nodes: list) -> bool:\n    pass",
        "reference_solution": """def solution(nodes: list) -> bool:
    def serialize(nodes):
        return ','.join('N' if n is None else str(n) for n in nodes)

    def deserialize(s):
        return [None if x == 'N' else int(x) for x in s.split(',')]

    serialized = serialize(nodes)
    deserialized = deserialize(serialized)
    return deserialized == nodes""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-serialization"],
        "pattern_explanation": "트리 직렬화는 트리를 문자열로 변환하고 복원하는 것입니다. None을 특수 문자로 표현합니다.",
        "approach_hint": "레벨 순회 + None 처리",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, None, None, 4, 5]", "output": "True", "is_sample": True},
        ]
    },
    {
        "title": "트리 오른쪽 뷰",
        "description": """트리를 오른쪽에서 봤을 때 보이는 노드들을 반환하세요.

### 입력
- 레벨 순서 트리 `nodes`

### 출력
- 오른쪽에서 보이는 노드 값 리스트

### 예제
```
입력: [1, 2, 3, None, 5, None, 4]
출력: [1, 3, 4]
```""",
        "difficulty": "medium",
        "category": "bfs",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["BFS로 레벨 순회", "각 레벨의 마지막 노드 선택"],
        "solution_template": "def solution(nodes: list) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list) -> list:
    from collections import deque
    if not nodes or nodes[0] is None:
        return []

    result = []
    queue = deque([0])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            idx = queue.popleft()
            if idx < len(nodes) and nodes[idx] is not None:
                if i == level_size - 1:
                    result.append(nodes[idx])
                left, right = 2*idx + 1, 2*idx + 2
                if left < len(nodes) and nodes[left] is not None:
                    queue.append(left)
                if right < len(nodes) and nodes[right] is not None:
                    queue.append(right)

    return result""",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-right-view"],
        "pattern_explanation": "각 레벨의 마지막 노드가 오른쪽에서 보입니다. BFS로 레벨별 처리합니다.",
        "approach_hint": "BFS 각 레벨의 마지막 노드",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[1, 2, 3, None, 5, None, 4]", "output": "[1, 3, 4]", "is_sample": True},
        ]
    },
    {
        "title": "거리 K인 노드들",
        "description": """타겟 노드에서 거리 K인 모든 노드를 찾으세요.

### 입력
- 트리 `nodes`
- 타겟 값 `target`
- 거리 `k`

### 출력
- 거리 K인 노드 값 리스트

### 예제
```
입력: nodes = [3, 5, 1, 6, 2, 0, 8], target = 5, k = 2
출력: [7, 4, 1] (7, 4는 서브트리, 1은 부모 방향)
```""",
        "difficulty": "hard",
        "category": "bfs",
        "constraints": "1 <= len(nodes) <= 500",
        "hints": ["트리를 그래프로 변환", "타겟에서 BFS로 거리 K 탐색"],
        "solution_template": "def solution(nodes: list, target: int, k: int) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list, target: int, k: int) -> list:
    from collections import deque, defaultdict

    graph = defaultdict(list)
    n = len(nodes)

    for i in range(n):
        if nodes[i] is None:
            continue
        left, right = 2*i + 1, 2*i + 2
        if left < n and nodes[left] is not None:
            graph[nodes[i]].append(nodes[left])
            graph[nodes[left]].append(nodes[i])
        if right < n and nodes[right] is not None:
            graph[nodes[i]].append(nodes[right])
            graph[nodes[right]].append(nodes[i])

    visited = {target}
    queue = deque([target])
    distance = 0

    while queue:
        if distance == k:
            return list(queue)
        size = len(queue)
        for _ in range(size):
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        distance += 1

    return []""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-to-graph-bfs"],
        "pattern_explanation": "트리를 무방향 그래프로 변환 후 BFS로 거리 K인 노드를 찾습니다.",
        "approach_hint": "트리 → 그래프 + BFS",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(n)",
        "test_cases": [
            {"input": "[3, 5, 1, 6, 2, 0, 8]\n5\n2", "output": "[1, 6, 2]", "is_sample": True},
        ]
    },
    {
        "title": "이진 트리 최대 경로 합",
        "description": """이진 트리에서 최대 경로 합을 구하세요. 경로는 연결된 노드들의 시퀀스입니다.

### 입력
- 트리 `nodes`

### 출력
- 최대 경로 합

### 예제
```
입력: [-10, 9, 20, None, None, 15, 7]
출력: 42 (15 + 20 + 7)
```""",
        "difficulty": "hard",
        "category": "dfs",
        "constraints": "1 <= len(nodes) <= 1000",
        "hints": ["각 노드에서 최대 기여 계산", "왼쪽+노드+오른쪽 vs 경로 확장"],
        "solution_template": "def solution(nodes: list) -> int:\n    pass",
        "reference_solution": """def solution(nodes: list) -> int:
    if not nodes or nodes[0] is None:
        return 0

    max_sum = [float('-inf')]

    def dfs(i):
        if i >= len(nodes) or nodes[i] is None:
            return 0

        left = max(0, dfs(2*i + 1))
        right = max(0, dfs(2*i + 2))

        max_sum[0] = max(max_sum[0], left + nodes[i] + right)

        return nodes[i] + max(left, right)

    dfs(0)
    return max_sum[0]""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-max-path-sum"],
        "pattern_explanation": "각 노드에서 왼쪽+노드+오른쪽 경로와 부모로 확장하는 경로 중 최대를 추적합니다.",
        "approach_hint": "DFS + 각 노드에서 최대 경로 계산",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[-10, 9, 20, None, None, 15, 7]", "output": "42", "is_sample": True},
        ]
    },
    {
        "title": "트리 가지치기",
        "description": """0으로만 구성된 서브트리를 제거하세요.

### 입력
- 트리 `nodes` (0 또는 1 값)

### 출력
- 가지치기 후 남은 노드 값 리스트 (레벨 순서)

### 예제
```
입력: [1, None, 0, 0, 1]
출력: [1, None, 0, None, 1]
```""",
        "difficulty": "hard",
        "category": "dfs",
        "constraints": "1 <= len(nodes) <= 200",
        "hints": ["후위 순회로 처리", "서브트리가 모두 0이면 제거"],
        "solution_template": "def solution(nodes: list) -> list:\n    pass",
        "reference_solution": """def solution(nodes: list) -> list:
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = self.right = None

    def build(i):
        if i >= len(nodes) or nodes[i] is None:
            return None
        node = TreeNode(nodes[i])
        node.left = build(2*i + 1)
        node.right = build(2*i + 2)
        return node

    def prune(node):
        if not node:
            return None
        node.left = prune(node.left)
        node.right = prune(node.right)
        if node.val == 0 and not node.left and not node.right:
            return None
        return node

    def to_list(root):
        from collections import deque
        if not root:
            return []
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)
        while result and result[-1] is None:
            result.pop()
        return result

    root = build(0)
    pruned = prune(root)
    return to_list(pruned)""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["tree-pruning"],
        "pattern_explanation": "후위 순회로 자식을 먼저 처리하고, 서브트리가 모두 0이면 노드를 제거합니다.",
        "approach_hint": "후위 순회 + 조건부 제거",
        "time_complexity_hint": "O(n)",
        "space_complexity_hint": "O(h)",
        "test_cases": [
            {"input": "[1, None, 0, None, None, 0, 1]", "output": "[1, None, 0, None, 1]", "is_sample": True},
        ]
    },
]

# ============== Phase 6: 그래프 DFS/BFS ==============
GRAPH_PROBLEMS = [
    {
        "title": "그래프 연결성 확인",
        "description": """무방향 그래프가 연결 그래프인지 확인하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges`

### 출력
- 연결 그래프면 True, 아니면 False

### 예제
```
입력: n = 4, edges = [[0,1],[1,2],[2,3]]
출력: True
```""",
        "difficulty": "easy",
        "category": "graph",
        "constraints": "1 <= n <= 10^4",
        "hints": ["DFS/BFS로 모든 노드 방문 가능한지 확인"],
        "solution_template": "def solution(n: int, edges: list) -> bool:\n    pass",
        "reference_solution": """def solution(n: int, edges: list) -> bool:
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(0)
    return len(visited) == n""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["graph-connectivity"],
        "pattern_explanation": "0번 노드에서 DFS/BFS로 모든 노드에 도달할 수 있으면 연결 그래프입니다.",
        "approach_hint": "DFS로 방문 노드 수 확인",
        "time_complexity_hint": "O(V+E)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "4\n[[0,1],[1,2],[2,3]]", "output": "True", "is_sample": True},
            {"input": "4\n[[0,1],[2,3]]", "output": "False", "is_sample": False},
        ]
    },
    {
        "title": "이분 그래프 판별",
        "description": """그래프가 이분 그래프인지 판별하세요.

이분 그래프: 모든 간선이 서로 다른 집합의 노드를 연결

### 입력
- 노드 수 `n`
- 간선 배열 `edges`

### 출력
- 이분 그래프면 True, 아니면 False

### 예제
```
입력: n = 4, edges = [[0,1],[0,3],[1,2],[2,3]]
출력: True
```""",
        "difficulty": "easy",
        "category": "bfs",
        "constraints": "1 <= n <= 10^4",
        "hints": ["BFS로 2가지 색으로 칠하기", "인접 노드는 다른 색"],
        "solution_template": "def solution(n: int, edges: list) -> bool:\n    pass",
        "reference_solution": """def solution(n: int, edges: list) -> bool:
    from collections import defaultdict, deque

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    color = [-1] * n

    for start in range(n):
        if color[start] != -1:
            continue
        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False

    return True""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["bipartite-check"],
        "pattern_explanation": "BFS로 노드를 2가지 색으로 칠합니다. 인접 노드가 같은 색이면 이분 그래프가 아닙니다.",
        "approach_hint": "2색 칠하기로 검증",
        "time_complexity_hint": "O(V+E)",
        "space_complexity_hint": "O(V)",
        "test_cases": [
            {"input": "4\n[[0,1],[0,3],[1,2],[2,3]]", "output": "True", "is_sample": True},
            {"input": "3\n[[0,1],[1,2],[0,2]]", "output": "False", "is_sample": False},
        ]
    },
    {
        "title": "섬의 개수",
        "description": """0과 1로 구성된 2D 그리드에서 섬의 개수를 구하세요. (1은 땅, 0은 물)

### 입력
- 2D 그리드 `grid`

### 출력
- 섬의 개수

### 예제
```
입력: [[1,1,0,0],[0,1,1,0],[0,0,0,1]]
출력: 2
```""",
        "difficulty": "medium",
        "category": "dfs",
        "constraints": "1 <= m, n <= 300",
        "hints": ["각 육지에서 DFS로 연결된 땅 방문", "방문한 땅은 표시"],
        "solution_template": "def solution(grid: list) -> int:\n    pass",
        "reference_solution": """def solution(grid: list) -> int:
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0:
            return
        grid[i][j] = 0
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                count += 1
                dfs(i, j)

    return count""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["flood-fill"],
        "pattern_explanation": "각 미방문 육지에서 DFS/BFS로 연결된 모든 땅을 방문합니다. 시작점마다 섬 카운트를 증가시킵니다.",
        "approach_hint": "DFS 플러드 필",
        "time_complexity_hint": "O(m*n)",
        "space_complexity_hint": "O(m*n)",
        "test_cases": [
            {"input": "[[1,1,0,0],[0,1,1,0],[0,0,0,1]]", "output": "2", "is_sample": True},
        ]
    },
    {
        "title": "단어 사다리",
        "description": """시작 단어에서 끝 단어로 한 글자씩 바꿔서 도달하는 최단 경로 길이를 구하세요.

### 입력
- 시작 단어 `begin`
- 끝 단어 `end`
- 단어 리스트 `word_list`

### 출력
- 최단 변환 길이 (불가능하면 0)

### 예제
```
입력: begin = "hit", end = "cog", word_list = ["hot","dot","dog","lot","log","cog"]
출력: 5 (hit → hot → dot → dog → cog)
```""",
        "difficulty": "medium",
        "category": "bfs",
        "constraints": "1 <= word_list.length <= 5000",
        "hints": ["BFS로 최단 경로 탐색", "각 위치마다 a-z로 변경 시도"],
        "solution_template": "def solution(begin: str, end: str, word_list: list) -> int:\n    pass",
        "reference_solution": """def solution(begin: str, end: str, word_list: list) -> int:
    from collections import deque

    word_set = set(word_list)
    if end not in word_set:
        return 0

    queue = deque([(begin, 1)])
    visited = {begin}

    while queue:
        word, length = queue.popleft()

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word == end:
                    return length + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))

    return 0""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["word-ladder-bfs"],
        "pattern_explanation": "BFS로 최단 변환 경로를 찾습니다. 각 단어에서 한 글자씩 바꿔 유효한 단어를 찾습니다.",
        "approach_hint": "BFS + 글자 변경 시도",
        "time_complexity_hint": "O(M^2 * N)",
        "space_complexity_hint": "O(M * N)",
        "test_cases": [
            {"input": 'hit\ncog\n["hot","dot","dog","lot","log","cog"]', "output": "5", "is_sample": True},
        ]
    },
    {
        "title": "강의 순서 (위상 정렬)",
        "description": """선수과목 관계가 주어질 때, 모든 강의를 들을 수 있는 순서를 구하세요.

### 입력
- 강의 수 `n`
- 선수과목 배열 `prerequisites` ([과목, 선수과목])

### 출력
- 수강 순서 (불가능하면 빈 리스트)

### 예제
```
입력: n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
출력: [0, 1, 2, 3] 또는 [0, 2, 1, 3]
```""",
        "difficulty": "medium",
        "category": "graph",
        "constraints": "1 <= n <= 2000",
        "hints": ["진입 차수 0인 노드부터 처리", "처리 후 연결된 노드의 진입 차수 감소"],
        "solution_template": "def solution(n: int, prerequisites: list) -> list:\n    pass",
        "reference_solution": """def solution(n: int, prerequisites: list) -> list:
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * n

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == n else []""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["topological-sort"],
        "pattern_explanation": "위상 정렬은 DAG에서 의존성 순서를 정합니다. 진입 차수 0인 노드부터 BFS로 처리합니다.",
        "approach_hint": "Kahn 알고리즘 (BFS)",
        "time_complexity_hint": "O(V+E)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "4\n[[1,0],[2,0],[3,1],[3,2]]", "output": "[0, 1, 2, 3]", "is_sample": True},
        ]
    },
    {
        "title": "연결 요소 (유니온 파인드)",
        "description": """유니온 파인드로 연결 요소의 개수를 구하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges`

### 출력
- 연결 요소 개수

### 예제
```
입력: n = 5, edges = [[0,1],[1,2],[3,4]]
출력: 2
```""",
        "difficulty": "medium",
        "category": "graph",
        "constraints": "1 <= n <= 10^5",
        "hints": ["Union-Find 자료구조 사용", "경로 압축과 랭크 최적화"],
        "solution_template": "def solution(n: int, edges: list) -> int:\n    pass",
        "reference_solution": """def solution(n: int, edges: list) -> int:
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for u, v in edges:
        union(u, v)

    return len(set(find(i) for i in range(n)))""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["union-find"],
        "pattern_explanation": "Union-Find는 집합의 합집합과 찾기 연산을 거의 O(1)에 수행합니다. 경로 압축과 랭크 최적화가 핵심입니다.",
        "approach_hint": "Union-Find + 경로 압축",
        "time_complexity_hint": "O(E * α(N))",
        "space_complexity_hint": "O(N)",
        "test_cases": [
            {"input": "5\n[[0,1],[1,2],[3,4]]", "output": "2", "is_sample": True},
        ]
    },
    {
        "title": "나이트의 이동",
        "description": """체스판에서 나이트가 시작 위치에서 목표 위치로 이동하는 최소 이동 횟수를 구하세요.

### 입력
- 체스판 크기 `n`
- 시작 위치 `start` [r, c]
- 목표 위치 `target` [r, c]

### 출력
- 최소 이동 횟수

### 예제
```
입력: n = 8, start = [0, 0], target = [7, 7]
출력: 6
```""",
        "difficulty": "hard",
        "category": "bfs",
        "constraints": "1 <= n <= 300",
        "hints": ["나이트의 8가지 이동", "BFS로 최단 경로"],
        "solution_template": "def solution(n: int, start: list, target: list) -> int:\n    pass",
        "reference_solution": """def solution(n: int, start: list, target: list) -> int:
    from collections import deque

    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

    if start == target:
        return 0

    visited = [[False]*n for _ in range(n)]
    queue = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True

    while queue:
        r, c, dist = queue.popleft()

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                if [nr, nc] == target:
                    return dist + 1
                visited[nr][nc] = True
                queue.append((nr, nc, dist + 1))

    return -1""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["knight-bfs", "implicit-graph"],
        "pattern_explanation": "암시적 그래프에서 BFS로 최단 경로를 찾습니다. 나이트는 8가지 방향으로 이동합니다.",
        "approach_hint": "BFS + 8방향 이동",
        "time_complexity_hint": "O(n^2)",
        "space_complexity_hint": "O(n^2)",
        "test_cases": [
            {"input": "8\n[0, 0]\n[7, 7]", "output": "6", "is_sample": True},
        ]
    },
    {
        "title": "외계어 사전",
        "description": """외계어 사전의 단어 순서에서 알파벳 순서를 추론하세요.

### 입력
- 단어 리스트 `words` (외계어 사전순)

### 출력
- 알파벳 순서 문자열 (불가능하면 빈 문자열)

### 예제
```
입력: ["wrt","wrf","er","ett","rftt"]
출력: "wertf"
```""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= words.length <= 100",
        "hints": ["인접 단어 비교로 순서 관계 추출", "위상 정렬로 순서 결정"],
        "solution_template": "def solution(words: list) -> str:\n    pass",
        "reference_solution": """def solution(words: list) -> str:
    from collections import defaultdict, deque

    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))

        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""

        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)
        for next_c in graph[c]:
            in_degree[next_c] -= 1
            if in_degree[next_c] == 0:
                queue.append(next_c)

    return ''.join(result) if len(result) == len(in_degree) else """ ,
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["alien-dictionary", "topological-sort"],
        "pattern_explanation": "인접 단어를 비교해 문자 간 순서 관계를 그래프로 구성하고 위상 정렬합니다.",
        "approach_hint": "순서 관계 추출 + 위상 정렬",
        "time_complexity_hint": "O(C)",
        "space_complexity_hint": "O(1)",
        "test_cases": [
            {"input": '["wrt","wrf","er","ett","rftt"]', "output": '"wertf"', "is_sample": True},
        ]
    },
    {
        "title": "MST 크루스칼",
        "description": """크루스칼 알고리즘으로 최소 신장 트리의 가중치 합을 구하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges` ([from, to, weight])

### 출력
- MST 가중치 합 (연결 불가면 -1)

### 예제
```
입력: n = 4, edges = [[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]]
출력: 4
```""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= n <= 10^5",
        "hints": ["간선을 가중치 순 정렬", "Union-Find로 사이클 검사"],
        "solution_template": "def solution(n: int, edges: list) -> int:\n    pass",
        "reference_solution": """def solution(n: int, edges: list) -> int:
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    edges.sort(key=lambda x: x[2])
    total = 0
    count = 0

    for u, v, w in edges:
        if union(u, v):
            total += w
            count += 1
            if count == n - 1:
                break

    return total if count == n - 1 else -1""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["kruskal-mst", "union-find"],
        "pattern_explanation": "크루스칼은 간선을 가중치 순으로 정렬하고, 사이클을 만들지 않는 간선만 선택합니다.",
        "approach_hint": "간선 정렬 + Union-Find",
        "time_complexity_hint": "O(E log E)",
        "space_complexity_hint": "O(V)",
        "test_cases": [
            {"input": "4\n[[0,1,1],[0,2,4],[1,2,2],[1,3,5],[2,3,1]]", "output": "4", "is_sample": True},
        ]
    },
    {
        "title": "최단 경로 (다익스트라 응용)",
        "description": """가중치 그래프에서 특정 중간 노드를 경유하는 최단 경로를 구하세요.

### 입력
- 노드 수 `n`
- 간선 배열 `edges`
- 시작점 `start`, 중간점 `via`, 끝점 `end`

### 출력
- start → via → end 최단 경로 길이 (-1은 불가)

### 예제
```
입력: n=5, edges=[[0,1,1],[1,2,2],[0,3,4],[3,4,1],[2,4,3]], start=0, via=2, end=4
출력: 6 (0→1→2→4)
```""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= n <= 10^4",
        "hints": ["다익스트라 두 번 실행", "start→via + via→end"],
        "solution_template": "def solution(n: int, edges: list, start: int, via: int, end: int) -> int:\n    pass",
        "reference_solution": """def solution(n: int, edges: list, start: int, via: int, end: int) -> int:
    import heapq
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def dijkstra(src):
        dist = [float('inf')] * n
        dist[src] = 0
        heap = [(0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))

        return dist

    dist_start = dijkstra(start)
    dist_via = dijkstra(via)

    result = dist_start[via] + dist_via[end]
    return result if result < float('inf') else -1""",
        "time_limit_ms": 3000,
        "memory_limit_mb": 256,
        "pattern_ids": ["dijkstra-via"],
        "pattern_explanation": "경유지를 지나는 최단 경로는 다익스트라를 두 번 실행하여 구합니다.",
        "approach_hint": "다익스트라 2회",
        "time_complexity_hint": "O((V+E) log V)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "5\n[[0,1,1],[1,2,2],[0,3,4],[3,4,1],[2,4,3]]\n0\n2\n4", "output": "6", "is_sample": True},
        ]
    },
    {
        "title": "네트워크 딜레이",
        "description": """네트워크에서 신호가 모든 노드에 도달하는 최소 시간을 구하세요.

### 입력
- 간선 배열 `times` ([from, to, time])
- 노드 수 `n`
- 시작 노드 `k`

### 출력
- 모든 노드 도달 시간 (불가능하면 -1)

### 예제
```
입력: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
출력: 2
```""",
        "difficulty": "hard",
        "category": "graph",
        "constraints": "1 <= n <= 100",
        "hints": ["다익스트라로 최단 거리 계산", "모든 노드 중 최대 거리 반환"],
        "solution_template": "def solution(times: list, n: int, k: int) -> int:\n    pass",
        "reference_solution": """def solution(times: list, n: int, k: int) -> int:
    import heapq
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {i: float('inf') for i in range(1, n+1)}
    dist[k] = 0
    heap = [(0, k)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    max_dist = max(dist.values())
    return max_dist if max_dist < float('inf') else -1""",
        "time_limit_ms": 2000,
        "memory_limit_mb": 256,
        "pattern_ids": ["dijkstra", "network-delay"],
        "pattern_explanation": "네트워크 딜레이는 시작점에서 모든 노드까지의 최단 거리 중 최대값입니다.",
        "approach_hint": "다익스트라 + 최대 거리",
        "time_complexity_hint": "O((V+E) log V)",
        "space_complexity_hint": "O(V+E)",
        "test_cases": [
            {"input": "[[2,1,1],[2,3,1],[3,4,1]]\n4\n2", "output": "2", "is_sample": True},
        ]
    },
]

ALL_PROBLEMS = (
    TREE_STRUCTURE_PROBLEMS +
    GREEDY_PROBLEMS +
    TREE_TRAVERSAL_PROBLEMS +
    GRAPH_PROBLEMS
)


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    added = 0
    skipped = 0

    print(f"총 {len(ALL_PROBLEMS)}개 문제 시딩 시작...")

    for problem in ALL_PROBLEMS:
        cursor.execute("SELECT id FROM problems WHERE title = ?", (problem["title"],))
        if cursor.fetchone():
            skipped += 1
            continue

        problem_id = uuid.uuid4().hex

        cursor.execute("""
            INSERT INTO problems (
                id, title, description, difficulty, category, constraints,
                hints, solution_template, reference_solution,
                time_limit_ms, memory_limit_mb, is_published,
                pattern_ids, pattern_explanation, approach_hint,
                time_complexity_hint, space_complexity_hint,
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

        added += 1
        if added % 10 == 0:
            print(f"  ... {added}개 추가됨")

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"Phase 3-6 통합 시딩 완료")
    print(f"  - 추가: {added}개")
    print(f"  - 건너뜀: {skipped}개")
    print(f"  - 전체: {added + skipped}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
