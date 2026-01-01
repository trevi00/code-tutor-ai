"""Visualization domain value objects."""

from enum import Enum


class AlgorithmCategory(str, Enum):
    """Algorithm categories for visualization."""

    SORTING = "sorting"
    SEARCHING = "searching"
    DATA_STRUCTURE = "data_structure"
    GRAPH = "graph"
    DYNAMIC_PROGRAMMING = "dynamic_programming"


class AlgorithmType(str, Enum):
    """Specific algorithm types."""

    # Sorting
    BUBBLE_SORT = "bubble_sort"
    SELECTION_SORT = "selection_sort"
    INSERTION_SORT = "insertion_sort"
    MERGE_SORT = "merge_sort"
    QUICK_SORT = "quick_sort"
    HEAP_SORT = "heap_sort"

    # Searching
    LINEAR_SEARCH = "linear_search"
    BINARY_SEARCH = "binary_search"

    # Data Structures
    ARRAY_OPERATIONS = "array_operations"
    LINKED_LIST = "linked_list"
    STACK = "stack"
    QUEUE = "queue"
    BINARY_TREE = "binary_tree"
    BST = "bst"
    HEAP = "heap"

    # Graph
    BFS = "bfs"
    DFS = "dfs"
    DIJKSTRA = "dijkstra"

    # Dynamic Programming
    FIBONACCI = "fibonacci"
    KNAPSACK = "knapsack"
    LCS = "lcs"


class AnimationType(str, Enum):
    """Animation action types."""

    COMPARE = "compare"          # Comparing two elements
    SWAP = "swap"                # Swapping two elements
    SET = "set"                  # Setting a value
    HIGHLIGHT = "highlight"      # Highlighting elements
    UNHIGHLIGHT = "unhighlight"  # Removing highlight
    PIVOT = "pivot"              # Marking pivot
    SORTED = "sorted"            # Marking as sorted
    FOUND = "found"              # Element found
    NOT_FOUND = "not_found"      # Element not found
    PUSH = "push"                # Push to stack/queue
    POP = "pop"                  # Pop from stack/queue
    INSERT = "insert"            # Insert into structure
    DELETE = "delete"            # Delete from structure
    VISIT = "visit"              # Visit node (graph/tree)
    CURRENT = "current"          # Current position marker
    MERGE = "merge"              # Merging two arrays
    DIVIDE = "divide"            # Dividing array


class ElementState(str, Enum):
    """Visual state of an element."""

    DEFAULT = "default"
    COMPARING = "comparing"
    SWAPPING = "swapping"
    SORTED = "sorted"
    PIVOT = "pivot"
    CURRENT = "current"
    FOUND = "found"
    VISITED = "visited"
    ACTIVE = "active"


# Algorithm metadata
ALGORITHM_INFO = {
    AlgorithmType.BUBBLE_SORT: {
        "name": "버블 정렬",
        "name_en": "Bubble Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n²)",
        "space_complexity": "O(1)",
        "description": "인접한 두 원소를 비교하여 정렬하는 알고리즘",
        "stable": True,
    },
    AlgorithmType.SELECTION_SORT: {
        "name": "선택 정렬",
        "name_en": "Selection Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n²)",
        "space_complexity": "O(1)",
        "description": "최소값을 찾아 맨 앞으로 이동시키는 알고리즘",
        "stable": False,
    },
    AlgorithmType.INSERTION_SORT: {
        "name": "삽입 정렬",
        "name_en": "Insertion Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n²)",
        "space_complexity": "O(1)",
        "description": "정렬된 부분에 새 원소를 삽입하는 알고리즘",
        "stable": True,
    },
    AlgorithmType.MERGE_SORT: {
        "name": "병합 정렬",
        "name_en": "Merge Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n log n)",
        "space_complexity": "O(n)",
        "description": "분할 정복을 이용한 안정적인 정렬 알고리즘",
        "stable": True,
    },
    AlgorithmType.QUICK_SORT: {
        "name": "퀵 정렬",
        "name_en": "Quick Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n log n)",
        "space_complexity": "O(log n)",
        "description": "피벗을 기준으로 분할하는 빠른 정렬 알고리즘",
        "stable": False,
    },
    AlgorithmType.HEAP_SORT: {
        "name": "힙 정렬",
        "name_en": "Heap Sort",
        "category": AlgorithmCategory.SORTING,
        "time_complexity": "O(n log n)",
        "space_complexity": "O(1)",
        "description": "힙 자료구조를 이용한 정렬 알고리즘",
        "stable": False,
    },
    AlgorithmType.LINEAR_SEARCH: {
        "name": "선형 탐색",
        "name_en": "Linear Search",
        "category": AlgorithmCategory.SEARCHING,
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "description": "처음부터 끝까지 순차적으로 탐색",
        "stable": True,
    },
    AlgorithmType.BINARY_SEARCH: {
        "name": "이진 탐색",
        "name_en": "Binary Search",
        "category": AlgorithmCategory.SEARCHING,
        "time_complexity": "O(log n)",
        "space_complexity": "O(1)",
        "description": "정렬된 배열에서 반씩 나눠 탐색",
        "stable": True,
    },
    AlgorithmType.BFS: {
        "name": "너비 우선 탐색",
        "name_en": "Breadth-First Search",
        "category": AlgorithmCategory.GRAPH,
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "description": "가까운 노드부터 탐색하는 그래프 알고리즘",
        "stable": True,
    },
    AlgorithmType.DFS: {
        "name": "깊이 우선 탐색",
        "name_en": "Depth-First Search",
        "category": AlgorithmCategory.GRAPH,
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "description": "깊은 노드부터 탐색하는 그래프 알고리즘",
        "stable": True,
    },
    AlgorithmType.BINARY_TREE: {
        "name": "이진 트리",
        "name_en": "Binary Tree",
        "category": AlgorithmCategory.DATA_STRUCTURE,
        "time_complexity": "O(log n) ~ O(n)",
        "space_complexity": "O(n)",
        "description": "각 노드가 최대 두 개의 자식을 가지는 트리",
        "stable": True,
    },
    AlgorithmType.BST: {
        "name": "이진 탐색 트리",
        "name_en": "Binary Search Tree",
        "category": AlgorithmCategory.DATA_STRUCTURE,
        "time_complexity": "O(log n)",
        "space_complexity": "O(n)",
        "description": "왼쪽 < 루트 < 오른쪽 규칙을 따르는 트리",
        "stable": True,
    },
}
