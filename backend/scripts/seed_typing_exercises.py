"""기존 템플릿을 타이핑 연습으로 변환하는 스크립트

기존 code_templates 테이블에서 템플릿을 가져와서
typing_exercises 테이블에 추가합니다.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "codetutor_v2.db"


# 핵심 템플릿 목록 (타이핑 연습에 적합한 것들)
CORE_TEMPLATES = [
    # Two Pointers
    {
        "title": "Two Pointer 기본",
        "source_code": """def two_pointer(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # 조건에 따라 포인터 이동
        if condition(left, right):
            left += 1
        else:
            right -= 1

    return result""",
        "category": "template",
        "difficulty": "easy",
        "description": "배열의 양 끝에서 시작하는 투 포인터 기본 템플릿",
    },
    # Sliding Window
    {
        "title": "슬라이딩 윈도우 기본",
        "source_code": """def sliding_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)

    return max_sum""",
        "category": "template",
        "difficulty": "easy",
        "description": "고정 크기 슬라이딩 윈도우 템플릿",
    },
    # Binary Search
    {
        "title": "이진 탐색 기본",
        "source_code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1""",
        "category": "template",
        "difficulty": "easy",
        "description": "정렬된 배열에서 값을 찾는 이진 탐색 템플릿",
    },
    # DFS
    {
        "title": "DFS 재귀",
        "source_code": """def dfs(graph, node, visited):
    visited.add(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited""",
        "category": "template",
        "difficulty": "easy",
        "description": "그래프 깊이 우선 탐색 재귀 템플릿",
    },
    # BFS
    {
        "title": "BFS 큐",
        "source_code": """from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited""",
        "category": "template",
        "difficulty": "easy",
        "description": "그래프 너비 우선 탐색 큐 템플릿",
    },
    # 문자열 메서드 모음
    {
        "title": "문자열 메서드 모음",
        "source_code": """# 문자열 기본 메서드
s = "Hello, World!"

# 길이
length = len(s)

# 대소문자 변환
upper = s.upper()
lower = s.lower()
title = s.title()

# 분할과 결합
words = s.split(", ")
joined = ", ".join(words)

# 검색과 치환
pos = s.find("World")
replaced = s.replace("World", "Python")

# 공백 제거
stripped = s.strip()""",
        "category": "method",
        "difficulty": "easy",
        "description": "파이썬 문자열 기본 메서드 모음",
    },
    # 리스트 메서드 모음
    {
        "title": "리스트 메서드 모음",
        "source_code": """# 리스트 기본 메서드
arr = [1, 2, 3, 4, 5]

# 추가
arr.append(6)
arr.insert(0, 0)
arr.extend([7, 8])

# 삭제
arr.pop()
arr.remove(0)

# 정렬
arr.sort()
arr.reverse()

# 검색
idx = arr.index(3)
count = arr.count(3)

# 복사
copy = arr.copy()
copy = arr[:]""",
        "category": "method",
        "difficulty": "easy",
        "description": "파이썬 리스트 기본 메서드 모음",
    },
    # 딕셔너리 메서드 모음
    {
        "title": "딕셔너리 메서드 모음",
        "source_code": """# 딕셔너리 기본 메서드
d = {"a": 1, "b": 2, "c": 3}

# 접근
val = d.get("a", 0)
keys = d.keys()
values = d.values()
items = d.items()

# 추가/수정
d["d"] = 4
d.update({"e": 5})

# 삭제
d.pop("a")
d.clear()

# defaultdict
from collections import defaultdict
dd = defaultdict(int)
dd["count"] += 1""",
        "category": "method",
        "difficulty": "easy",
        "description": "파이썬 딕셔너리 기본 메서드 모음",
    },
    # 세그먼트 트리
    {
        "title": "세그먼트 트리 구간합",
        "source_code": """def build_segment_tree(arr):
    n = len(arr)
    tree = [0] * (2 * n)

    # 리프 노드 초기화
    for i in range(n):
        tree[n + i] = arr[i]

    # 내부 노드 구축
    for i in range(n - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]

    return tree

def query(tree, n, left, right):
    result = 0
    left += n
    right += n + 1

    while left < right:
        if left % 2 == 1:
            result += tree[left]
            left += 1
        if right % 2 == 1:
            right -= 1
            result += tree[right]
        left //= 2
        right //= 2

    return result""",
        "category": "algorithm",
        "difficulty": "hard",
        "description": "Bottom-up 세그먼트 트리 구간합 쿼리",
    },
    # 펜윅 트리
    {
        "title": "펜윅 트리 (BIT)",
        "source_code": """class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total

    def range_sum(self, left, right):
        return self.prefix_sum(right) - self.prefix_sum(left - 1)""",
        "category": "algorithm",
        "difficulty": "hard",
        "description": "Binary Indexed Tree 구현",
    },
]


def seed_typing_exercises():
    """타이핑 연습 데이터 시딩"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 이미 존재하는 연습 확인
    cursor.execute("SELECT title FROM typing_exercises")
    existing = {row[0] for row in cursor.fetchall()}

    added = 0
    skipped = 0

    for template in CORE_TEMPLATES:
        if template["title"] in existing:
            print(f"  [SKIP] {template['title']} - 이미 존재")
            skipped += 1
            continue

        exercise_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        cursor.execute("""
            INSERT INTO typing_exercises (
                id, title, source_code, language, category, difficulty,
                description, required_completions, is_published,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exercise_id,
            template["title"],
            template["source_code"],
            "python",
            template["category"],
            template["difficulty"],
            template["description"],
            5,  # required_completions
            1,  # is_published
            now,
            now
        ))

        print(f"  [ADD] {template['title']}")
        added += 1

    conn.commit()
    conn.close()

    print(f"\n완료: {added}개 추가, {skipped}개 건너뜀")
    return added, skipped


if __name__ == "__main__":
    print("=" * 50)
    print("타이핑 연습 데이터 시딩")
    print("=" * 50)
    seed_typing_exercises()
