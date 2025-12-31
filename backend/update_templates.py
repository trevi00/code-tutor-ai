"""Update all problem templates with stdin parsing"""

import sqlite3
import json

DB_PATH = "codetutor_v2.db"

# Template updates for each problem
# Format: (problem_title, solution_template, reference_solution)
UPDATES = [
    # 1. 두 수의 합 - Already updated

    # 2. 유효한 괄호
    ("유효한 괄호",
'''def solution(s: str) -> bool:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
''',
'''def solution(s: str) -> bool:
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != pairs[char]:
                return False

    return len(stack) == 0


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
'''),

    # 3. 이진 탐색
    ("이진 탐색",
'''import json

def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
''',
'''import json

def solution(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
'''),

    # 4. 최대 부분배열 합 (카데인)
    ("최대 부분배열 합 (카데인)",
'''import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> int:
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 5. 섬의 개수
    ("섬의 개수",
'''import json

def solution(grid: list[list[str]]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    grid = json.loads(input())
    print(solution(grid))
''',
'''import json

def solution(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count


if __name__ == "__main__":
    grid = json.loads(input())
    print(solution(grid))
'''),

    # 6. 최장 증가 부분수열 (LIS)
    ("최장 증가 부분수열 (LIS)",
'''import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 7. 동전 거스름돈
    ("동전 거스름돈",
'''import json

def solution(coins: list[int], amount: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    coins = json.loads(input())
    amount = int(input())
    print(solution(coins, amount))
''',
'''import json

def solution(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


if __name__ == "__main__":
    coins = json.loads(input())
    amount = int(input())
    print(solution(coins, amount))
'''),

    # 8. 이진 트리 레벨 순회 - Tree structure, special handling
    ("이진 트리 레벨 순회",
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
''',
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
'''),

    # 9. 회의실 배정 (그리디)
    ("회의실 배정 (그리디)",
'''import json

def solution(intervals: list[list[int]]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    intervals = json.loads(input())
    print(solution(intervals))
''',
'''import json

def solution(intervals: list[list[int]]) -> int:
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])
    count = 1
    end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            count += 1
            end = intervals[i][1]

    return count


if __name__ == "__main__":
    intervals = json.loads(input())
    print(solution(intervals))
'''),

    # 10. N-Queen (백트래킹)
    ("N-Queen (백트래킹)",
'''def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
''',
'''def solution(n: int) -> int:
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == row - i:
                return False
        return True

    def backtrack(row):
        if row == n:
            return 1
        count = 0
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                count += backtrack(row + 1)
                board[row] = -1
        return count

    board = [-1] * n
    return backtrack(0)


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
'''),

    # 11. 최대 합 슬라이딩 윈도우
    ("최대 합 슬라이딩 윈도우",
'''import json

def solution(nums: list[int], k: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    k = int(input())
    print(solution(nums, k))
''',
'''import json

def solution(nums: list[int], k: int) -> int:
    if len(nums) < k:
        return 0

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum


if __name__ == "__main__":
    nums = json.loads(input())
    k = int(input())
    print(solution(nums, k))
'''),

    # 12. 연결 리스트 사이클 감지
    ("연결 리스트 사이클 감지",
'''import json

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(values, pos):
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos >= 0:
        nodes[-1].next = nodes[pos]
    return nodes[0]


def solution(head: ListNode) -> bool:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    pos = int(input())
    head = build_list(values, pos)
    print(solution(head))
''',
'''import json

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(values, pos):
    if not values:
        return None
    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos >= 0:
        nodes[-1].next = nodes[pos]
    return nodes[0]


def solution(head: ListNode) -> bool:
    if not head:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False


if __name__ == "__main__":
    values = json.loads(input())
    pos = int(input())
    head = build_list(values, pos)
    print(solution(head))
'''),

    # 13. K번째 큰 수
    ("K번째 큰 수",
'''import json

def solution(nums: list[int], k: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    k = int(input())
    print(solution(nums, k))
''',
'''import json
import heapq

def solution(nums: list[int], k: int) -> int:
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]


if __name__ == "__main__":
    nums = json.loads(input())
    k = int(input())
    print(solution(nums, k))
'''),

    # 14. 부분집합 생성
    ("부분집합 생성",
'''import json

def solution(nums: list[int]) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 15. 문자열 뒤집기
    ("문자열 뒤집기",
'''def solution(s: str) -> str:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
''',
'''def solution(s: str) -> str:
    chars = list(s)
    left, right = 0, len(chars) - 1

    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1

    return ''.join(chars)


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
'''),

    # 16. 다음 큰 원소
    ("다음 큰 원소",
'''import json

def solution(nums: list[int]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 17. 누락된 숫자 찾기
    ("누락된 숫자 찾기",
'''import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 18. 연결 리스트 뒤집기
    ("연결 리스트 뒤집기",
'''import json

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for v in values[1:]:
        current.next = ListNode(v)
        current = current.next
    return head


def list_to_array(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def solution(head: ListNode) -> ListNode:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    head = build_list(values)
    result = solution(head)
    print(list_to_array(result))
''',
'''import json

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for v in values[1:]:
        current.next = ListNode(v)
        current = current.next
    return head


def list_to_array(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def solution(head: ListNode) -> ListNode:
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev


if __name__ == "__main__":
    values = json.loads(input())
    head = build_list(values)
    result = solution(head)
    print(list_to_array(result))
'''),

    # 19. 이진 트리 최대 깊이
    ("이진 트리 최대 깊이",
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
''',
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> int:
    if not root:
        return 0
    return 1 + max(solution(root.left), solution(root.right))


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
'''),

    # 20. 중앙값 찾기 (데이터 스트림) - Special: class-based
    ("중앙값 찾기 (데이터 스트림)",
'''import json
import heapq

class MedianFinder:
    def __init__(self):
        # 여기에 코드를 작성하세요
        pass

    def addNum(self, num: int) -> None:
        pass

    def findMedian(self) -> float:
        pass


if __name__ == "__main__":
    operations = json.loads(input())
    values = json.loads(input())

    finder = MedianFinder()
    results = []
    for op, val in zip(operations, values):
        if op == "addNum":
            finder.addNum(val[0])
            results.append(None)
        elif op == "findMedian":
            results.append(finder.findMedian())
    print(results)
''',
'''import json
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated)
        self.large = []  # min heap

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))

        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2


if __name__ == "__main__":
    operations = json.loads(input())
    values = json.loads(input())

    finder = MedianFinder()
    results = []
    for op, val in zip(operations, values):
        if op == "addNum":
            finder.addNum(val[0])
            results.append(None)
        elif op == "findMedian":
            results.append(finder.findMedian())
    print(results)
'''),

    # 21. K개 정렬 리스트 병합
    ("K개 정렬 리스트 병합",
'''import json
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_lists(arrays):
    lists = []
    for arr in arrays:
        if not arr:
            lists.append(None)
            continue
        head = ListNode(arr[0])
        current = head
        for v in arr[1:]:
            current.next = ListNode(v)
            current = current.next
        lists.append(head)
    return lists


def list_to_array(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def solution(lists: list) -> ListNode:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    arrays = json.loads(input())
    lists = build_lists(arrays)
    result = solution(lists)
    print(list_to_array(result))
''',
'''import json
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        return self.val < other.val


def build_lists(arrays):
    lists = []
    for arr in arrays:
        if not arr:
            lists.append(None)
            continue
        head = ListNode(arr[0])
        current = head
        for v in arr[1:]:
            current.next = ListNode(v)
            current = current.next
        lists.append(head)
    return lists


def list_to_array(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def solution(lists: list) -> ListNode:
    heap = []
    for lst in lists:
        if lst:
            heapq.heappush(heap, lst)

    dummy = ListNode()
    current = dummy

    while heap:
        node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, node.next)

    return dummy.next


if __name__ == "__main__":
    arrays = json.loads(input())
    lists = build_lists(arrays)
    result = solution(lists)
    print(list_to_array(result))
'''),

    # 22. 강의 순서 (위상 정렬)
    ("강의 순서 (위상 정렬)",
'''import json
from collections import deque

def solution(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    numCourses = int(input())
    prerequisites = json.loads(input())
    print(solution(numCourses, prerequisites))
''',
'''import json
from collections import deque

def solution(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = [[] for _ in range(numCourses)]
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []

    while queue:
        course = queue.popleft()
        result.append(course)
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return result if len(result) == numCourses else []


if __name__ == "__main__":
    numCourses = int(input())
    prerequisites = json.loads(input())
    print(solution(numCourses, prerequisites))
'''),

    # 23. 0/1 배낭 문제
    ("0/1 배낭 문제",
'''import json

def solution(weights: list[int], values: list[int], capacity: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    weights = json.loads(input())
    values = json.loads(input())
    capacity = int(input())
    print(solution(weights, values, capacity))
''',
'''import json

def solution(weights: list[int], values: list[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]


if __name__ == "__main__":
    weights = json.loads(input())
    values = json.loads(input())
    capacity = int(input())
    print(solution(weights, values, capacity))
'''),

    # 24. 피보나치 수
    ("피보나치 수",
'''def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
''',
'''def solution(n: int) -> int:
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
'''),

    # 25. 가장 긴 팰린드롬 부분수열
    ("가장 긴 팰린드롬 부분수열",
'''def solution(s: str) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
''',
'''def solution(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
'''),

    # 26. 최장 공통 부분수열 (LCS)
    ("최장 공통 부분수열 (LCS)",
'''def solution(text1: str, text2: str) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    text1 = input().strip()
    text2 = input().strip()
    print(solution(text1, text2))
''',
'''def solution(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


if __name__ == "__main__":
    text1 = input().strip()
    text2 = input().strip()
    print(solution(text1, text2))
'''),

    # 27. 단일 숫자 찾기
    ("단일 숫자 찾기",
'''import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 28. 구간 합 쿼리
    ("구간 합 쿼리",
'''import json

class NumArray:
    def __init__(self, nums: list[int]):
        # 여기에 코드를 작성하세요
        pass

    def sumRange(self, left: int, right: int) -> int:
        pass


if __name__ == "__main__":
    nums = json.loads(input())
    queries = json.loads(input())
    arr = NumArray(nums)
    results = [arr.sumRange(l, r) for l, r in queries]
    print(results)
''',
'''import json

class NumArray:
    def __init__(self, nums: list[int]):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix[right + 1] - self.prefix[left]


if __name__ == "__main__":
    nums = json.loads(input())
    queries = json.loads(input())
    arr = NumArray(nums)
    results = [arr.sumRange(l, r) for l, r in queries]
    print(results)
'''),

    # 29. 최단 경로 (다익스트라)
    ("최단 경로 (다익스트라)",
'''import json
import heapq

def solution(n: int, edges: list[list[int]], start: int) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    n = int(input())
    edges = json.loads(input())
    start = int(input())
    print(solution(n, edges, start))
''',
'''import json
import heapq

def solution(n: int, edges: list[list[int]], start: int) -> list[int]:
    graph = [[] for _ in range(n)]
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

    return [-1 if d == float('inf') else d for d in dist]


if __name__ == "__main__":
    n = int(input())
    edges = json.loads(input())
    start = int(input())
    print(solution(n, edges, start))
'''),

    # 30. 계단 오르기
    ("계단 오르기",
'''def solution(n: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
''',
'''def solution(n: int) -> int:
    if n <= 2:
        return n

    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr

    return curr


if __name__ == "__main__":
    n = int(input())
    print(solution(n))
'''),

    # 31. 정렬된 배열 합치기
    ("정렬된 배열 합치기",
'''import json

def solution(nums1: list[int], nums2: list[int]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums1 = json.loads(input())
    nums2 = json.loads(input())
    print(solution(nums1, nums2))
''',
'''import json

def solution(nums1: list[int], nums2: list[int]) -> list[int]:
    result = []
    i, j = 0, 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1

    result.extend(nums1[i:])
    result.extend(nums2[j:])
    return result


if __name__ == "__main__":
    nums1 = json.loads(input())
    nums2 = json.loads(input())
    print(solution(nums1, nums2))
'''),

    # 32. 부분배열의 최대 곱
    ("부분배열의 최대 곱",
'''import json

def solution(nums: list[int]) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> int:
    max_prod = nums[0]
    min_prod = nums[0]
    result = nums[0]

    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod

        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)

    return result


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 33. 회전된 배열에서 검색
    ("회전된 배열에서 검색",
'''import json

def solution(nums: list[int], target: int) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
''',
'''import json

def solution(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


if __name__ == "__main__":
    nums = json.loads(input())
    target = int(input())
    print(solution(nums, target))
'''),

    # 34. 이진 트리의 모든 경로
    ("이진 트리의 모든 경로",
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[str]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
''',
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[str]:
    if not root:
        return []

    result = []

    def dfs(node, path):
        if not node.left and not node.right:
            result.append(path + str(node.val))
            return
        if node.left:
            dfs(node.left, path + str(node.val) + "->")
        if node.right:
            dfs(node.right, path + str(node.val) + "->")

    dfs(root, "")
    return result


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
'''),

    # 35. 가장 긴 무중복 부분 문자열
    ("가장 긴 무중복 부분 문자열",
'''def solution(s: str) -> int:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
''',
'''def solution(s: str) -> int:
    char_index = {}
    max_length = 0
    start = 0

    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_length = max(max_length, i - start + 1)

    return max_length


if __name__ == "__main__":
    s = input().strip()
    print(solution(s))
'''),

    # 36. 구간 병합
    ("구간 병합",
'''import json

def solution(intervals: list[list[int]]) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    intervals = json.loads(input())
    print(solution(intervals))
''',
'''import json

def solution(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)

    return merged


if __name__ == "__main__":
    intervals = json.loads(input())
    print(solution(intervals))
'''),

    # 37. 조합 합
    ("조합 합",
'''import json

def solution(candidates: list[int], target: int) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    candidates = json.loads(input())
    target = int(input())
    print(solution(candidates, target))
''',
'''import json

def solution(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result


if __name__ == "__main__":
    candidates = json.loads(input())
    target = int(input())
    print(solution(candidates, target))
'''),

    # 38. 점프 게임
    ("점프 게임",
'''import json

def solution(nums: list[int]) -> bool:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> bool:
    max_reach = 0

    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)

    return True


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 39. 이진 트리 지그재그 레벨 순회
    ("이진 트리 지그재그 레벨 순회",
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
''',
'''import json
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def solution(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = deque()

        for _ in range(level_size):
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


if __name__ == "__main__":
    values = json.loads(input())
    root = build_tree(values)
    print(solution(root))
'''),

    # 40. 모든 중복 찾기
    ("모든 중복 찾기",
'''import json

def solution(nums: list[int]) -> list[int]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
''',
'''import json

def solution(nums: list[int]) -> list[int]:
    result = []

    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            result.append(abs(num))
        else:
            nums[idx] = -nums[idx]

    return result


if __name__ == "__main__":
    nums = json.loads(input())
    print(solution(nums))
'''),

    # 41. K쌍의 최소 합
    ("K쌍의 최소 합",
'''import json
import heapq

def solution(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    # 여기에 코드를 작성하세요
    pass


if __name__ == "__main__":
    nums1 = json.loads(input())
    nums2 = json.loads(input())
    k = int(input())
    print(solution(nums1, nums2, k))
''',
'''import json
import heapq

def solution(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    if not nums1 or not nums2:
        return []

    heap = [(nums1[0] + nums2[0], 0, 0)]
    visited = {(0, 0)}
    result = []

    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])

        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
            visited.add((i + 1, j))

        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))

    return result


if __name__ == "__main__":
    nums1 = json.loads(input())
    nums2 = json.loads(input())
    k = int(input())
    print(solution(nums1, nums2, k))
'''),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updated = 0
    for title, template, reference in UPDATES:
        cursor.execute(
            "UPDATE problems SET solution_template = ?, reference_solution = ? WHERE title = ?",
            (template, reference, title)
        )
        if cursor.rowcount > 0:
            print(f"[OK] Updated: {title}")
            updated += 1
        else:
            print(f"[SKIP] Not found: {title}")

    conn.commit()
    conn.close()
    print(f"\nTotal updated: {updated}/{len(UPDATES)}")


if __name__ == "__main__":
    main()
