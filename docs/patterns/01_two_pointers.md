# Pattern 01: Two Pointers (투 포인터)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Medium |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(n) |
| **공간복잡도** | O(1) |
| **선행 지식** | 배열, 정렬 |

## 정의

**Two Pointers**는 배열이나 리스트에서 두 개의 포인터를 사용하여 특정 조건을 만족하는 요소를 찾는 기법입니다. 일반적으로 정렬된 배열에서 사용하며, 브루트포스 O(n²)를 O(n)으로 최적화할 수 있습니다.

## 언제 사용하는가?

### ✅ 사용해야 하는 경우
- 정렬된 배열에서 두 수의 합/차를 찾을 때
- 팰린드롬(회문) 검사
- 중복 제거 (in-place)
- 배열의 양 끝에서 조건을 만족하는 쌍 찾기
- 정렬된 두 배열 병합
- 컨테이너 최대 물 문제

### ❌ 사용하지 말아야 하는 경우
- 배열이 정렬되어 있지 않고, 정렬 비용이 클 때
- 연속된 부분 배열을 찾을 때 (→ Sliding Window)
- 세 개 이상의 요소 조합이 필요할 때 (다른 기법 필요)

## 패턴 유형

### 유형 1: 양쪽 끝에서 시작 (Opposite Direction)

```
[1, 2, 3, 4, 5, 6, 7]
 ↑                 ↑
left             right

조건에 따라:
- 합이 작으면 → left++
- 합이 크면 → right--
```

### 유형 2: 같은 방향으로 이동 (Same Direction)

```
[1, 1, 2, 2, 3, 4, 4, 5]
 ↑  ↑
slow fast

fast가 앞서가며 탐색
slow는 결과 위치 추적
```

---

## 템플릿 코드

### 템플릿 1: 양쪽 끝에서 시작

```python
def two_pointers_opposite(arr: list, target: int) -> list:
    """
    정렬된 배열에서 합이 target인 두 수의 인덱스 찾기

    Args:
        arr: 정렬된 정수 배열
        target: 목표 합

    Returns:
        [left_index, right_index] 또는 [] (없으면)

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1   # 합을 키우기 위해 왼쪽 포인터 이동
        else:
            right -= 1  # 합을 줄이기 위해 오른쪽 포인터 이동

    return []  # 찾지 못함
```

### 템플릿 2: 같은 방향 (중복 제거)

```python
def two_pointers_same_direction(arr: list) -> int:
    """
    정렬된 배열에서 중복 제거 (in-place)

    Args:
        arr: 정렬된 정수 배열

    Returns:
        중복 제거 후 배열 길이

    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    slow = 0  # 고유한 요소의 마지막 위치

    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1
```

### 템플릿 3: 팰린드롬 검사

```python
def is_palindrome(s: str) -> bool:
    """
    문자열이 팰린드롬인지 검사

    Args:
        s: 검사할 문자열

    Returns:
        팰린드롬이면 True

    Time: O(n), Space: O(1)
    """
    # 알파벳과 숫자만 추출하여 소문자로 변환
    s = ''.join(c.lower() for c in s if c.isalnum())

    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True
```

### 템플릿 4: 세 수의 합 (3Sum)

```python
def three_sum(nums: list) -> list:
    """
    합이 0인 세 수의 조합 모두 찾기

    Args:
        nums: 정수 배열

    Returns:
        합이 0인 세 수의 조합 리스트

    Time: O(n²), Space: O(1) (결과 제외)
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # 중복 건너뛰기
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])

                # 중복 건너뛰기
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

---

## 예제 문제

### 문제 1: 두 수의 합 (Easy)

**문제 설명**
정렬된 정수 배열 `numbers`와 정수 `target`이 주어집니다. 합이 `target`인 두 수의 인덱스를 반환하세요. (1-indexed)

**입력/출력 예시**
```
입력: numbers = [2, 7, 11, 15], target = 9
출력: [1, 2]
설명: numbers[1] + numbers[2] = 2 + 7 = 9
```

**제약조건**
- 2 ≤ len(numbers) ≤ 3 × 10^4
- -1000 ≤ numbers[i] ≤ 1000
- numbers는 오름차순 정렬됨
- 정확히 하나의 정답이 존재

**풀이**
```python
def two_sum(numbers: list, target: int) -> list:
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []
```

**복잡도 분석**
- 시간: O(n) - 각 포인터가 최대 n번 이동
- 공간: O(1) - 추가 공간 없음

---

### 문제 2: 컨테이너에 담기는 최대 물의 양 (Medium)

**문제 설명**
높이 배열 `height`가 주어집니다. 두 막대 사이에 담을 수 있는 물의 최대량을 구하세요.

**입력/출력 예시**
```
입력: height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
출력: 49
설명: index 1과 8 사이, 높이 7, 너비 7 → 7 × 7 = 49
```

**시각화**
```
      |         |
      |         |   |
  |   |   |     |   |
  |   |   |   | |   |
  |   |   | | | |   |
  |   |   | | | | | |
  | | | | | | | | | |
  0 1 2 3 4 5 6 7 8
```

**풀이**
```python
def max_area(height: list) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # 면적 = 너비 × 높이 (낮은 막대 기준)
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        # 낮은 쪽을 이동해야 더 큰 면적 가능
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
```

**왜 낮은 쪽을 이동하나?**
- 높은 막대를 이동하면 너비는 줄고, 높이는 기존 낮은 막대로 고정
- 낮은 막대를 이동하면 너비는 줄지만, 높이가 커질 가능성 있음

**복잡도 분석**
- 시간: O(n)
- 공간: O(1)

---

### 문제 3: 중복 제거 (Easy)

**문제 설명**
정렬된 배열에서 중복을 제거하고, 고유한 요소의 개수를 반환하세요. (in-place)

**입력/출력 예시**
```
입력: nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
출력: 5
설명: nums = [0, 1, 2, 3, 4, ...] (앞 5개가 고유)
```

**풀이**
```python
def remove_duplicates(nums: list) -> int:
    if not nums:
        return 0

    slow = 0

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

**복잡도 분석**
- 시간: O(n)
- 공간: O(1)

---

### 문제 4: 세 수의 합 (Medium)

**문제 설명**
정수 배열에서 합이 0인 세 수의 조합을 모두 찾으세요. 중복 조합은 제외합니다.

**입력/출력 예시**
```
입력: nums = [-1, 0, 1, 2, -1, -4]
출력: [[-1, -1, 2], [-1, 0, 1]]
```

**풀이**
```python
def three_sum(nums: list) -> list:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

**복잡도 분석**
- 시간: O(n²) - 정렬 O(n log n) + 이중 루프 O(n²)
- 공간: O(1) (결과 제외)

---

### 문제 5: 유효한 팰린드롬 (Easy)

**문제 설명**
문자열이 팰린드롬인지 확인하세요. 알파벳과 숫자만 고려하고, 대소문자는 구분하지 않습니다.

**입력/출력 예시**
```
입력: s = "A man, a plan, a canal: Panama"
출력: true
설명: "amanaplanacanalpanama"는 팰린드롬
```

**풀이**
```python
def is_palindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        # 알파벳/숫자가 아니면 건너뛰기
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True
```

**복잡도 분석**
- 시간: O(n)
- 공간: O(1)

---

## Editorial (풀이 전략)

### Step 1: 문제 유형 파악
- 정렬된 배열에서 특정 조건 만족? → Two Pointers 가능성 높음
- "쌍", "두 개", "합/차" 키워드가 있는가?

### Step 2: 포인터 방향 결정
| 상황 | 포인터 방향 |
|------|------------|
| 합/차 찾기 | 양 끝에서 시작 (opposite) |
| 중복 제거 | 같은 방향 (same direction) |
| 팰린드롬 | 양 끝에서 시작 |
| 배열 병합 | 같은 방향 또는 양 끝 |

### Step 3: 이동 조건 설정
```python
while left < right:
    if condition_too_small:
        left += 1     # 값을 키워야 함
    elif condition_too_big:
        right -= 1    # 값을 줄여야 함
    else:
        # 찾음!
```

### Step 4: 경계 조건 확인
- 빈 배열 체크
- 단일 요소 배열
- 중복 요소 처리
- 포인터 교차 방지 (`left < right`)

---

## 자주 하는 실수

### 1. 정렬 안 함
```python
# ❌ 잘못된 코드
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    # nums가 정렬되어 있지 않으면 Two Pointers 작동 안 함!
```

### 2. 포인터 교차 조건 누락
```python
# ❌ 잘못된 코드
while left <= right:  # left < right 여야 함 (같으면 같은 요소)
```

### 3. 중복 건너뛰기 누락 (3Sum)
```python
# ❌ 중복 조합 발생
if total == 0:
    result.append([nums[i], nums[left], nums[right]])
    left += 1
    right -= 1
    # 중복 건너뛰기 없음 → 같은 조합 중복 추가됨
```

### 4. 무한 루프
```python
# ❌ 포인터 이동 안 함
while left < right:
    if nums[left] + nums[right] == target:
        return [left, right]
    # left++ 또는 right-- 없음 → 무한 루프
```

---

## 관련 패턴

| 패턴 | 관계 |
|------|------|
| **Sliding Window** | 연속 부분배열은 Sliding Window, 비연속 쌍은 Two Pointers |
| **Binary Search** | 정렬된 배열에서 단일 요소 찾기는 Binary Search |
| **Fast & Slow Pointers** | 링크드리스트 사이클 감지 등에 사용 |

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 | 비고 |
|---|-------|-------|------|
| 167 | Two Sum II | Easy | 기본 |
| 15 | 3Sum | Medium | 필수 |
| 11 | Container With Most Water | Medium | 필수 |
| 125 | Valid Palindrome | Easy | 기본 |
| 26 | Remove Duplicates | Easy | 기본 |
| 27 | Remove Element | Easy | 기본 |
| 42 | Trapping Rain Water | Hard | 심화 |
| 75 | Sort Colors | Medium | Dutch National Flag |
| 16 | 3Sum Closest | Medium | 응용 |
| 18 | 4Sum | Medium | 심화 |

### BOJ (백준)

| # | 문제명 | 난이도 | 비고 |
|---|-------|-------|------|
| 3273 | 두 수의 합 | Silver 3 | 기본 |
| 2470 | 두 용액 | Gold 5 | 필수 |
| 2467 | 용액 | Gold 5 | 필수 |
| 1806 | 부분합 | Gold 4 | 연속 부분합 |
| 1644 | 소수의 연속합 | Gold 3 | 소수 + 투포인터 |
| 2230 | 수 고르기 | Gold 5 | 정렬 + 투포인터 |
| 2003 | 수들의 합 2 | Silver 4 | 기본 |
| 1940 | 주몽 | Silver 4 | 기본 |
| 3649 | 로봇 프로젝트 | Gold 5 | 응용 |
| 7453 | 합이 0인 네 정수 | Gold 2 | 4Sum 변형 |

### 프로그래머스

| 문제명 | 난이도 | 비고 |
|-------|-------|------|
| 두 개 뽑아서 더하기 | Level 1 | 기본 |
| 구명보트 | Level 2 | 필수 (그리디 + 투포인터) |
| 숫자의 표현 | Level 2 | 연속 수열 합 |
| 다음 큰 숫자 | Level 2 | 응용 |

---

## 임베딩용 키워드

```
two pointers, 투 포인터, 양쪽 포인터, opposite direction, same direction,
정렬된 배열, sorted array, 두 수의 합, two sum, 세 수의 합, three sum,
팰린드롬, palindrome, 회문, 중복 제거, remove duplicates,
컨테이너 물, container water, in-place, 쌍 찾기, pair finding,
용액, 부분합, 구명보트, BOJ 3273, BOJ 2470, 프로그래머스
```
