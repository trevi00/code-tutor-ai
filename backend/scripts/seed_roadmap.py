"""Seed script for Learning Roadmap data."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select

from code_tutor.shared.infrastructure.database import init_db, get_session_context
# Import identity models first to ensure users table exists
import code_tutor.identity.infrastructure.models  # noqa: F401
from code_tutor.roadmap.infrastructure.models import (
    LearningPathModel,
    ModuleModel,
    LessonModel,
)
from code_tutor.roadmap.domain.value_objects import PathLevel, LessonType


# ============== Learning Path Data ==============

PATHS_DATA = [
    {
        "id": str(uuid4()),
        "level": PathLevel.BEGINNER.value,
        "title": "파이썬 입문",
        "description": "프로그래밍이 처음인 분을 위한 파이썬 기초 과정입니다. 변수, 조건문, 반복문, 자료구조의 기본을 배웁니다.",
        "icon": "🐍",
        "order": 1,
        "estimated_hours": 20,
        "modules": [
            {
                "title": "시작하기",
                "description": "파이썬과 프로그래밍의 첫 걸음",
                "lessons": [
                    {"title": "Hello World 출력하기", "type": LessonType.CONCEPT, "content": "# Hello World\n\n파이썬에서 가장 먼저 배우는 것은 화면에 텍스트를 출력하는 것입니다.\n\n```python\nprint(\"Hello, World!\")\n```\n\n`print()` 함수는 괄호 안의 내용을 화면에 출력합니다.", "xp": 10, "minutes": 5},
                    {"title": "변수 이해하기", "type": LessonType.CONCEPT, "content": "# 변수\n\n변수는 데이터를 저장하는 상자입니다.\n\n```python\nname = \"김철수\"\nage = 20\nheight = 175.5\n```\n\n변수에 값을 할당하려면 `=` 기호를 사용합니다.", "xp": 10, "minutes": 10},
                    {"title": "자료형 알아보기", "type": LessonType.CONCEPT, "content": "# 자료형\n\n파이썬의 기본 자료형:\n\n- `int`: 정수 (1, 2, -5)\n- `float`: 실수 (3.14, -0.5)\n- `str`: 문자열 (\"hello\")\n- `bool`: 불리언 (True, False)", "xp": 10, "minutes": 10},
                    {"title": "입력과 출력", "type": LessonType.CONCEPT, "content": "# 입력과 출력\n\n사용자로부터 입력받기:\n\n```python\nname = input(\"이름을 입력하세요: \")\nprint(f\"안녕하세요, {name}님!\")\n```", "xp": 10, "minutes": 10},
                    {"title": "기초 퀴즈", "type": LessonType.QUIZ, "content": "quiz_basics_1", "xp": 20, "minutes": 10},
                ],
            },
            {
                "title": "조건문",
                "description": "상황에 따라 다르게 동작하는 코드 작성하기",
                "lessons": [
                    {"title": "if 문", "type": LessonType.CONCEPT, "content": "# if 문\n\n조건에 따라 코드를 실행합니다.\n\n```python\nage = 20\nif age >= 18:\n    print(\"성인입니다\")\n```", "xp": 10, "minutes": 10},
                    {"title": "if-else 문", "type": LessonType.CONCEPT, "content": "# if-else 문\n\n```python\nage = 15\nif age >= 18:\n    print(\"성인입니다\")\nelse:\n    print(\"미성년자입니다\")\n```", "xp": 10, "minutes": 10},
                    {"title": "if-elif-else 문", "type": LessonType.CONCEPT, "content": "# if-elif-else 문\n\n```python\nscore = 85\nif score >= 90:\n    print(\"A\")\nelif score >= 80:\n    print(\"B\")\nelse:\n    print(\"C\")\n```", "xp": 10, "minutes": 10},
                    {"title": "중첩 조건문", "type": LessonType.CONCEPT, "content": "# 중첩 조건문\n\n조건문 안에 조건문을 넣을 수 있습니다.\n\n```python\nif age >= 18:\n    if has_license:\n        print(\"운전 가능\")\n```", "xp": 10, "minutes": 10},
                    {"title": "홀짝 판별하기", "type": LessonType.PROBLEM, "content": "problem_even_odd", "xp": 15, "minutes": 15},
                ],
            },
            {
                "title": "반복문",
                "description": "같은 작업을 여러 번 반복하기",
                "lessons": [
                    {"title": "for 문 기초", "type": LessonType.CONCEPT, "content": "# for 문\n\n```python\nfor i in range(5):\n    print(i)  # 0, 1, 2, 3, 4\n```", "xp": 10, "minutes": 10},
                    {"title": "while 문", "type": LessonType.CONCEPT, "content": "# while 문\n\n```python\ncount = 0\nwhile count < 5:\n    print(count)\n    count += 1\n```", "xp": 10, "minutes": 10},
                    {"title": "break와 continue", "type": LessonType.CONCEPT, "content": "# break와 continue\n\n```python\nfor i in range(10):\n    if i == 5:\n        break  # 반복 중단\n    if i == 3:\n        continue  # 다음 반복으로\n    print(i)\n```", "xp": 10, "minutes": 10},
                    {"title": "중첩 반복문", "type": LessonType.CONCEPT, "content": "# 중첩 반복문\n\n```python\nfor i in range(3):\n    for j in range(3):\n        print(f\"({i}, {j})\")\n```", "xp": 10, "minutes": 10},
                    {"title": "구구단 출력하기", "type": LessonType.PROBLEM, "content": "problem_multiplication_table", "xp": 15, "minutes": 15},
                ],
            },
            {
                "title": "리스트",
                "description": "여러 데이터를 하나로 묶어 관리하기",
                "lessons": [
                    {"title": "리스트 생성", "type": LessonType.CONCEPT, "content": "# 리스트\n\n```python\nfruits = [\"사과\", \"바나나\", \"오렌지\"]\nnumbers = [1, 2, 3, 4, 5]\n```", "xp": 10, "minutes": 10},
                    {"title": "인덱싱과 슬라이싱", "type": LessonType.CONCEPT, "content": "# 인덱싱과 슬라이싱\n\n```python\nfruits = [\"사과\", \"바나나\", \"오렌지\"]\nprint(fruits[0])      # 사과\nprint(fruits[-1])     # 오렌지\nprint(fruits[0:2])    # ['사과', '바나나']\n```", "xp": 10, "minutes": 10},
                    {"title": "리스트 메서드", "type": LessonType.CONCEPT, "content": "# 리스트 메서드\n\n```python\nfruits = [\"사과\"]\nfruits.append(\"바나나\")  # 추가\nfruits.remove(\"사과\")    # 삭제\nfruits.sort()            # 정렬\n```", "xp": 10, "minutes": 10},
                    {"title": "리스트 컴프리헨션", "type": LessonType.CONCEPT, "content": "# 리스트 컴프리헨션\n\n```python\nsquares = [x**2 for x in range(10)]\nevens = [x for x in range(10) if x % 2 == 0]\n```", "xp": 15, "minutes": 15},
                    {"title": "최댓값 찾기", "type": LessonType.PROBLEM, "content": "problem_find_max", "xp": 15, "minutes": 15},
                ],
            },
            {
                "title": "딕셔너리",
                "description": "키-값 쌍으로 데이터 관리하기",
                "lessons": [
                    {"title": "딕셔너리 생성", "type": LessonType.CONCEPT, "content": "# 딕셔너리\n\n```python\nperson = {\n    \"name\": \"김철수\",\n    \"age\": 25,\n    \"city\": \"서울\"\n}\n```", "xp": 10, "minutes": 10},
                    {"title": "딕셔너리 접근", "type": LessonType.CONCEPT, "content": "# 딕셔너리 접근\n\n```python\nprint(person[\"name\"])       # 김철수\nprint(person.get(\"email\"))  # None (없으면)\n```", "xp": 10, "minutes": 10},
                    {"title": "딕셔너리 메서드", "type": LessonType.CONCEPT, "content": "# 딕셔너리 메서드\n\n```python\nperson.keys()    # 키 목록\nperson.values()  # 값 목록\nperson.items()   # (키, 값) 쌍\n```", "xp": 10, "minutes": 10},
                    {"title": "딕셔너리 활용", "type": LessonType.CONCEPT, "content": "# 딕셔너리 활용\n\n```python\n# 빈도수 세기\nword = \"hello\"\nfreq = {}\nfor char in word:\n    freq[char] = freq.get(char, 0) + 1\n```", "xp": 15, "minutes": 15},
                    {"title": "입문 과정 최종 퀴즈", "type": LessonType.QUIZ, "content": "quiz_beginner_final", "xp": 30, "minutes": 20},
                ],
            },
        ],
    },
    {
        "id": str(uuid4()),
        "level": PathLevel.ELEMENTARY.value,
        "title": "기초 알고리즘",
        "description": "기본 문법을 마스터한 분을 위한 알고리즘 기초 과정입니다. 배열, 문자열, 정렬, 탐색, 재귀를 배웁니다.",
        "icon": "📚",
        "order": 2,
        "estimated_hours": 30,
        "modules": [
            {
                "title": "배열 다루기",
                "description": "배열/리스트 조작의 기본 패턴",
                "lessons": [
                    {"title": "배열 순회 패턴", "type": LessonType.PATTERN, "content": "pattern_array_traversal", "xp": 10, "minutes": 15},
                    {"title": "최대/최소값 찾기", "type": LessonType.PATTERN, "content": "pattern_min_max", "xp": 10, "minutes": 15},
                    {"title": "합계와 평균", "type": LessonType.PATTERN, "content": "pattern_sum_average", "xp": 10, "minutes": 10},
                    {"title": "Two Sum 문제", "type": LessonType.PROBLEM, "content": "problem_two_sum", "xp": 20, "minutes": 25},
                    {"title": "배열 회전", "type": LessonType.PROBLEM, "content": "problem_rotate_array", "xp": 15, "minutes": 20},
                ],
            },
            {
                "title": "문자열 처리",
                "description": "문자열 조작의 핵심 테크닉",
                "lessons": [
                    {"title": "문자열 뒤집기", "type": LessonType.PATTERN, "content": "pattern_reverse_string", "xp": 10, "minutes": 10},
                    {"title": "팰린드롬 확인", "type": LessonType.PATTERN, "content": "pattern_palindrome", "xp": 15, "minutes": 15},
                    {"title": "아나그램 판별", "type": LessonType.PATTERN, "content": "pattern_anagram", "xp": 15, "minutes": 15},
                    {"title": "문자열 압축", "type": LessonType.PROBLEM, "content": "problem_string_compression", "xp": 20, "minutes": 25},
                    {"title": "가장 긴 부분 문자열", "type": LessonType.PROBLEM, "content": "problem_longest_substring", "xp": 25, "minutes": 30},
                ],
            },
            {
                "title": "정렬",
                "description": "데이터를 순서대로 정렬하는 방법",
                "lessons": [
                    {"title": "버블 정렬", "type": LessonType.PATTERN, "content": "pattern_bubble_sort", "xp": 15, "minutes": 20},
                    {"title": "선택 정렬", "type": LessonType.PATTERN, "content": "pattern_selection_sort", "xp": 15, "minutes": 20},
                    {"title": "삽입 정렬", "type": LessonType.PATTERN, "content": "pattern_insertion_sort", "xp": 15, "minutes": 20},
                    {"title": "버블 정렬 타이핑", "type": LessonType.TYPING, "content": "typing_bubble_sort", "xp": 20, "minutes": 15},
                    {"title": "K번째 수 찾기", "type": LessonType.PROBLEM, "content": "problem_kth_number", "xp": 20, "minutes": 25},
                ],
            },
            {
                "title": "탐색",
                "description": "원하는 데이터를 효율적으로 찾기",
                "lessons": [
                    {"title": "선형 탐색", "type": LessonType.PATTERN, "content": "pattern_linear_search", "xp": 10, "minutes": 10},
                    {"title": "이진 탐색", "type": LessonType.PATTERN, "content": "pattern_binary_search", "xp": 20, "minutes": 25},
                    {"title": "이진 탐색 타이핑", "type": LessonType.TYPING, "content": "typing_binary_search", "xp": 20, "minutes": 15},
                    {"title": "Upper/Lower Bound", "type": LessonType.PATTERN, "content": "pattern_bound_search", "xp": 20, "minutes": 20},
                    {"title": "특정 수 찾기", "type": LessonType.PROBLEM, "content": "problem_search_number", "xp": 20, "minutes": 25},
                ],
            },
            {
                "title": "재귀",
                "description": "함수가 자기 자신을 호출하는 패턴",
                "lessons": [
                    {"title": "재귀 함수 이해", "type": LessonType.CONCEPT, "content": "# 재귀 함수\n\n재귀는 함수가 자기 자신을 호출하는 것입니다.\n\n```python\ndef countdown(n):\n    if n <= 0:  # 기저 조건\n        return\n    print(n)\n    countdown(n - 1)  # 재귀 호출\n```", "xp": 15, "minutes": 15},
                    {"title": "팩토리얼", "type": LessonType.PATTERN, "content": "pattern_factorial", "xp": 15, "minutes": 15},
                    {"title": "피보나치 수열", "type": LessonType.PATTERN, "content": "pattern_fibonacci", "xp": 20, "minutes": 20},
                    {"title": "하노이의 탑", "type": LessonType.PROBLEM, "content": "problem_hanoi", "xp": 25, "minutes": 30},
                    {"title": "재귀 퀴즈", "type": LessonType.QUIZ, "content": "quiz_recursion", "xp": 20, "minutes": 15},
                ],
            },
            {
                "title": "기초 자료구조",
                "description": "스택, 큐, 해시의 기본",
                "lessons": [
                    {"title": "스택 이해하기", "type": LessonType.PATTERN, "content": "pattern_stack", "xp": 15, "minutes": 15},
                    {"title": "큐 이해하기", "type": LessonType.PATTERN, "content": "pattern_queue", "xp": 15, "minutes": 15},
                    {"title": "해시맵 활용", "type": LessonType.PATTERN, "content": "pattern_hashmap", "xp": 15, "minutes": 15},
                    {"title": "괄호 검사", "type": LessonType.PROBLEM, "content": "problem_valid_parentheses", "xp": 20, "minutes": 20},
                    {"title": "기초 과정 최종 평가", "type": LessonType.QUIZ, "content": "quiz_elementary_final", "xp": 30, "minutes": 30},
                ],
            },
        ],
    },
    {
        "id": str(uuid4()),
        "level": PathLevel.INTERMEDIATE.value,
        "title": "중급 알고리즘",
        "description": "본격적인 알고리즘 학습 과정입니다. 스택/큐 심화, 그래프, 트리, DP, 이진탐색 응용을 다룹니다.",
        "icon": "🚀",
        "order": 3,
        "estimated_hours": 50,
        "modules": [
            {
                "title": "스택 심화",
                "description": "스택을 활용한 고급 문제 해결",
                "lessons": [
                    {"title": "괄호 매칭", "type": LessonType.PATTERN, "content": "pattern_parentheses_matching", "xp": 15, "minutes": 15},
                    {"title": "후위 표기식", "type": LessonType.PATTERN, "content": "pattern_postfix", "xp": 20, "minutes": 25},
                    {"title": "모노토닉 스택", "type": LessonType.PATTERN, "content": "pattern_monotonic_stack", "xp": 25, "minutes": 30},
                    {"title": "히스토그램", "type": LessonType.PROBLEM, "content": "problem_histogram", "xp": 30, "minutes": 35},
                    {"title": "탑 문제", "type": LessonType.PROBLEM, "content": "problem_tower", "xp": 25, "minutes": 30},
                ],
            },
            {
                "title": "큐 심화",
                "description": "슬라이딩 윈도우와 우선순위 큐",
                "lessons": [
                    {"title": "슬라이딩 윈도우", "type": LessonType.PATTERN, "content": "pattern_sliding_window", "xp": 20, "minutes": 25},
                    {"title": "슬라이딩 윈도우 타이핑", "type": LessonType.TYPING, "content": "typing_sliding_window", "xp": 20, "minutes": 15},
                    {"title": "우선순위 큐", "type": LessonType.PATTERN, "content": "pattern_priority_queue", "xp": 20, "minutes": 25},
                    {"title": "K번째 큰 수", "type": LessonType.PROBLEM, "content": "problem_kth_largest", "xp": 25, "minutes": 25},
                    {"title": "최솟값 찾기", "type": LessonType.PROBLEM, "content": "problem_min_window", "xp": 25, "minutes": 30},
                ],
            },
            {
                "title": "그래프 기초",
                "description": "그래프 표현과 탐색",
                "lessons": [
                    {"title": "그래프 표현법", "type": LessonType.CONCEPT, "content": "# 그래프 표현\n\n## 인접 리스트\n```python\ngraph = {\n    0: [1, 2],\n    1: [0, 3],\n    2: [0, 4],\n    3: [1],\n    4: [2]\n}\n```\n\n## 인접 행렬\n```python\nadj = [[0,1,1,0,0],\n       [1,0,0,1,0],\n       [1,0,0,0,1],\n       [0,1,0,0,0],\n       [0,0,1,0,0]]\n```", "xp": 15, "minutes": 20},
                    {"title": "DFS (깊이 우선 탐색)", "type": LessonType.PATTERN, "content": "pattern_dfs", "xp": 25, "minutes": 30},
                    {"title": "DFS 타이핑", "type": LessonType.TYPING, "content": "typing_dfs", "xp": 20, "minutes": 15},
                    {"title": "BFS (너비 우선 탐색)", "type": LessonType.PATTERN, "content": "pattern_bfs", "xp": 25, "minutes": 30},
                    {"title": "BFS 타이핑", "type": LessonType.TYPING, "content": "typing_bfs", "xp": 20, "minutes": 15},
                ],
            },
            {
                "title": "트리",
                "description": "트리 구조와 순회",
                "lessons": [
                    {"title": "트리 순회", "type": LessonType.PATTERN, "content": "pattern_tree_traversal", "xp": 20, "minutes": 25},
                    {"title": "이진 탐색 트리", "type": LessonType.PATTERN, "content": "pattern_bst", "xp": 25, "minutes": 30},
                    {"title": "힙 구조", "type": LessonType.PATTERN, "content": "pattern_heap", "xp": 25, "minutes": 30},
                    {"title": "트리의 높이", "type": LessonType.PROBLEM, "content": "problem_tree_height", "xp": 20, "minutes": 25},
                    {"title": "LCA (최소 공통 조상)", "type": LessonType.PROBLEM, "content": "problem_lca", "xp": 30, "minutes": 35},
                ],
            },
            {
                "title": "DP 입문",
                "description": "동적 프로그래밍의 기초",
                "lessons": [
                    {"title": "메모이제이션", "type": LessonType.CONCEPT, "content": "# 메모이제이션\n\n이미 계산한 값을 저장해두는 기법입니다.\n\n```python\nmemo = {}\ndef fib(n):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n-1) + fib(n-2)\n    return memo[n]\n```", "xp": 15, "minutes": 20},
                    {"title": "탑다운 vs 바텀업", "type": LessonType.CONCEPT, "content": "# DP 접근법\n\n## 탑다운 (재귀 + 메모이제이션)\n```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1) + fib(n-2)\n    return memo[n]\n```\n\n## 바텀업 (반복문)\n```python\ndef fib(n):\n    if n <= 1: return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]\n```", "xp": 20, "minutes": 25},
                    {"title": "계단 오르기", "type": LessonType.PROBLEM, "content": "problem_climb_stairs", "xp": 20, "minutes": 20},
                    {"title": "동전 교환", "type": LessonType.PROBLEM, "content": "problem_coin_change", "xp": 25, "minutes": 30},
                    {"title": "DP 퀴즈", "type": LessonType.QUIZ, "content": "quiz_dp_intro", "xp": 20, "minutes": 15},
                ],
            },
            {
                "title": "DP 패턴",
                "description": "주요 DP 패턴 학습",
                "lessons": [
                    {"title": "LCS (최장 공통 부분 수열)", "type": LessonType.PATTERN, "content": "pattern_lcs", "xp": 25, "minutes": 30},
                    {"title": "LIS (최장 증가 부분 수열)", "type": LessonType.PATTERN, "content": "pattern_lis", "xp": 25, "minutes": 30},
                    {"title": "배낭 문제", "type": LessonType.PATTERN, "content": "pattern_knapsack", "xp": 30, "minutes": 35},
                    {"title": "편집 거리", "type": LessonType.PROBLEM, "content": "problem_edit_distance", "xp": 30, "minutes": 35},
                    {"title": "팰린드롬 분할", "type": LessonType.PROBLEM, "content": "problem_palindrome_partition", "xp": 30, "minutes": 35},
                ],
            },
            {
                "title": "이진탐색 응용",
                "description": "이진탐색의 다양한 활용",
                "lessons": [
                    {"title": "파라메트릭 서치", "type": LessonType.PATTERN, "content": "pattern_parametric_search", "xp": 25, "minutes": 30},
                    {"title": "Lower Bound / Upper Bound", "type": LessonType.PATTERN, "content": "pattern_lower_upper_bound", "xp": 25, "minutes": 25},
                    {"title": "나무 자르기", "type": LessonType.PROBLEM, "content": "problem_wood_cutting", "xp": 25, "minutes": 30},
                    {"title": "공유기 설치", "type": LessonType.PROBLEM, "content": "problem_router_install", "xp": 30, "minutes": 35},
                    {"title": "이진탐색 퀴즈", "type": LessonType.QUIZ, "content": "quiz_binary_search", "xp": 20, "minutes": 15},
                ],
            },
            {
                "title": "분할정복 / 그리디",
                "description": "문제를 쪼개거나 최적해를 선택하기",
                "lessons": [
                    {"title": "분할정복 기초", "type": LessonType.PATTERN, "content": "pattern_divide_conquer", "xp": 20, "minutes": 25},
                    {"title": "머지 소트", "type": LessonType.PATTERN, "content": "pattern_merge_sort", "xp": 25, "minutes": 30},
                    {"title": "그리디 알고리즘", "type": LessonType.PATTERN, "content": "pattern_greedy", "xp": 20, "minutes": 25},
                    {"title": "활동 선택 문제", "type": LessonType.PROBLEM, "content": "problem_activity_selection", "xp": 25, "minutes": 25},
                    {"title": "중급 과정 최종 평가", "type": LessonType.QUIZ, "content": "quiz_intermediate_final", "xp": 40, "minutes": 40},
                ],
            },
        ],
    },
    {
        "id": str(uuid4()),
        "level": PathLevel.ADVANCED.value,
        "title": "고급 알고리즘",
        "description": "코딩테스트와 대회를 준비하는 분을 위한 고급 과정입니다. 최단경로, MST, 세그먼트 트리, 문자열 고급 알고리즘을 다룹니다.",
        "icon": "🏆",
        "order": 4,
        "estimated_hours": 60,
        "modules": [
            {
                "title": "고급 그래프",
                "description": "최단경로 알고리즘",
                "lessons": [
                    {"title": "다익스트라 알고리즘", "type": LessonType.PATTERN, "content": "pattern_dijkstra", "xp": 30, "minutes": 35},
                    {"title": "다익스트라 타이핑", "type": LessonType.TYPING, "content": "typing_dijkstra", "xp": 25, "minutes": 20},
                    {"title": "벨만-포드 알고리즘", "type": LessonType.PATTERN, "content": "pattern_bellman_ford", "xp": 30, "minutes": 35},
                    {"title": "플로이드-워셜 알고리즘", "type": LessonType.PATTERN, "content": "pattern_floyd_warshall", "xp": 30, "minutes": 35},
                    {"title": "최단경로 문제", "type": LessonType.PROBLEM, "content": "problem_shortest_path", "xp": 35, "minutes": 40},
                ],
            },
            {
                "title": "MST (최소 신장 트리)",
                "description": "그래프의 최소 비용 트리 찾기",
                "lessons": [
                    {"title": "크루스칼 알고리즘", "type": LessonType.PATTERN, "content": "pattern_kruskal", "xp": 30, "minutes": 35},
                    {"title": "프림 알고리즘", "type": LessonType.PATTERN, "content": "pattern_prim", "xp": 30, "minutes": 35},
                    {"title": "유니온-파인드", "type": LessonType.PATTERN, "content": "pattern_union_find", "xp": 30, "minutes": 30},
                    {"title": "유니온-파인드 타이핑", "type": LessonType.TYPING, "content": "typing_union_find", "xp": 25, "minutes": 20},
                    {"title": "네트워크 연결", "type": LessonType.PROBLEM, "content": "problem_network_connection", "xp": 35, "minutes": 40},
                ],
            },
            {
                "title": "세그먼트 트리",
                "description": "구간 쿼리를 효율적으로 처리",
                "lessons": [
                    {"title": "세그먼트 트리 기초", "type": LessonType.PATTERN, "content": "pattern_segment_tree", "xp": 35, "minutes": 40},
                    {"title": "세그먼트 트리 타이핑", "type": LessonType.TYPING, "content": "typing_segment_tree", "xp": 25, "minutes": 20},
                    {"title": "Lazy Propagation", "type": LessonType.PATTERN, "content": "pattern_lazy_propagation", "xp": 40, "minutes": 45},
                    {"title": "펜윅 트리", "type": LessonType.PATTERN, "content": "pattern_fenwick_tree", "xp": 35, "minutes": 35},
                    {"title": "구간 합 구하기", "type": LessonType.PROBLEM, "content": "problem_range_sum", "xp": 35, "minutes": 40},
                ],
            },
            {
                "title": "문자열 고급",
                "description": "효율적인 문자열 알고리즘",
                "lessons": [
                    {"title": "KMP 알고리즘", "type": LessonType.PATTERN, "content": "pattern_kmp", "xp": 35, "minutes": 40},
                    {"title": "KMP 타이핑", "type": LessonType.TYPING, "content": "typing_kmp", "xp": 25, "minutes": 20},
                    {"title": "트라이 (Trie)", "type": LessonType.PATTERN, "content": "pattern_trie", "xp": 35, "minutes": 35},
                    {"title": "라빈-카프 알고리즘", "type": LessonType.PATTERN, "content": "pattern_rabin_karp", "xp": 35, "minutes": 35},
                    {"title": "문자열 검색 문제", "type": LessonType.PROBLEM, "content": "problem_string_search", "xp": 35, "minutes": 40},
                ],
            },
            {
                "title": "고급 DP",
                "description": "비트마스크, 구간, 트리 DP",
                "lessons": [
                    {"title": "비트마스크 DP", "type": LessonType.PATTERN, "content": "pattern_bitmask_dp", "xp": 40, "minutes": 45},
                    {"title": "구간 DP", "type": LessonType.PATTERN, "content": "pattern_interval_dp", "xp": 40, "minutes": 45},
                    {"title": "트리 DP", "type": LessonType.PATTERN, "content": "pattern_tree_dp", "xp": 40, "minutes": 45},
                    {"title": "외판원 순회 (TSP)", "type": LessonType.PROBLEM, "content": "problem_tsp", "xp": 45, "minutes": 50},
                    {"title": "고급 DP 퀴즈", "type": LessonType.QUIZ, "content": "quiz_advanced_dp", "xp": 30, "minutes": 25},
                ],
            },
            {
                "title": "대회 문제",
                "description": "실전 대회 수준 문제 도전",
                "lessons": [
                    {"title": "코드포스 스타일 문제", "type": LessonType.PROBLEM, "content": "problem_codeforces_style", "xp": 50, "minutes": 60},
                    {"title": "삼성 SW 역량테스트 스타일", "type": LessonType.PROBLEM, "content": "problem_samsung_style", "xp": 50, "minutes": 60},
                    {"title": "카카오 코테 스타일", "type": LessonType.PROBLEM, "content": "problem_kakao_style", "xp": 50, "minutes": 60},
                    {"title": "ICPC 스타일 문제", "type": LessonType.PROBLEM, "content": "problem_icpc_style", "xp": 60, "minutes": 70},
                    {"title": "고급 과정 최종 평가", "type": LessonType.QUIZ, "content": "quiz_advanced_final", "xp": 50, "minutes": 50},
                ],
            },
        ],
    },
]


async def seed_roadmap():
    """Seed learning roadmap data."""
    print("Seeding Learning Roadmap data...")

    await init_db()

    async with get_session_context() as session:
        # Check if data already exists
        result = await session.execute(select(LearningPathModel).limit(1))
        if result.scalar_one_or_none():
            print("Roadmap data already exists. Skipping seed.")
            return

        total_lessons = 0

        for path_data in PATHS_DATA:
            # Create path
            path_model = LearningPathModel(
                id=path_data["id"],
                level=path_data["level"],
                title=path_data["title"],
                description=path_data["description"],
                icon=path_data["icon"],
                order=path_data["order"],
                estimated_hours=path_data["estimated_hours"],
                is_published=True,
            )
            session.add(path_model)
            await session.flush()

            module_order = 1
            for module_data in path_data["modules"]:
                # Create module
                module_model = ModuleModel(
                    id=str(uuid4()),
                    path_id=path_data["id"],
                    title=module_data["title"],
                    description=module_data["description"],
                    order=module_order,
                )
                session.add(module_model)
                await session.flush()

                lesson_order = 1
                for lesson_data in module_data["lessons"]:
                    # Create lesson
                    lesson_model = LessonModel(
                        id=str(uuid4()),
                        module_id=module_model.id,
                        title=lesson_data["title"],
                        description=lesson_data.get("description", ""),
                        lesson_type=lesson_data["type"].value,
                        content=lesson_data.get("content", ""),
                        order=lesson_order,
                        xp_reward=lesson_data.get("xp", 10),
                        estimated_minutes=lesson_data.get("minutes", 10),
                    )
                    session.add(lesson_model)
                    lesson_order += 1
                    total_lessons += 1

                module_order += 1

            print(f"  Created path: {path_data['title']} ({path_data['level']})")

        await session.commit()

    print(f"\nSeeding complete!")
    print(f"  - {len(PATHS_DATA)} learning paths")
    print(f"  - {total_lessons} total lessons")


if __name__ == "__main__":
    asyncio.run(seed_roadmap())
