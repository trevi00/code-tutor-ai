# Pattern 05: Cyclic Sort (사이클 정렬)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Medium |
| **빈출도** | ⭐⭐⭐ (중간) |
| **시간복잡도** | O(n) |
| **공간복잡도** | O(1) |
| **선행 지식** | 배열, In-place 알고리즘 |

## 정의

**Cyclic Sort**는 **1부터 N까지의 숫자**가 포함된 배열에서 각 숫자를 **올바른 인덱스에 배치**하는 기법입니다. 숫자 `i`는 인덱스 `i-1`에 위치해야 합니다.

## 핵심 아이디어

```
값 = 인덱스 + 1

숫자 1 → 인덱스 0
숫자 2 → 인덱스 1
숫자 3 → 인덱스 2
...
숫자 N → 인덱스 N-1
```

### 시각화

```
입력: [3, 1, 5, 4, 2]
       ↓
인덱스: 0  1  2  3  4

Step 1: arr[0]=3 → 3은 인덱스 2에 있어야 함 → swap(0, 2)
        [5, 1, 3, 4, 2]

Step 2: arr[0]=5 → 5는 인덱스 4에 있어야 함 → swap(0, 4)
        [2, 1, 3, 4, 5]

Step 3: arr[0]=2 → 2는 인덱스 1에 있어야 함 → swap(0, 1)
        [1, 2, 3, 4, 5]

Step 4: arr[0]=1 → 올바른 위치! → 다음 인덱스로

결과: [1, 2, 3, 4, 5] ✓
```

## 언제 사용하는가?

### ✅ 사용해야 하는 경우
- **1~N 범위 숫자**가 주어진 배열
- **누락된 숫자** 찾기
- **중복된 숫자** 찾기
- **제자리가 아닌 숫자** 찾기
- **O(1) 공간**으로 정렬/검색 필요

### ❌ 사용하지 말아야 하는 경우
- 숫자 범위가 1~N이 아닌 경우
- 음수가 포함된 경우
- 범위가 배열 크기보다 훨씬 큰 경우
- 일반적인 정렬 (→ Quick Sort, Merge Sort)

---

## 템플릿 코드

### 템플릿 1: 기본 Cyclic Sort

```python
def cyclic_sort(nums: list) -> list:
    """
    1~N 범위 숫자 배열을 O(n) 시간, O(1) 공간으로 정렬

    Args:
        nums: 1~N 범위의 숫자 배열

    Returns:
        정렬된 배열

    Time: O(n), Space: O(1)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1  # 숫자가 있어야 할 인덱스

        if nums[i] != nums[correct_idx]:
            # 올바른 위치로 swap
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            # 이미 올바른 위치면 다음으로
            i += 1

    return nums
```

### 템플릿 2: 누락된 숫자 찾기

```python
def find_missing_number(nums: list) -> int:
    """
    0~N 범위에서 누락된 숫자 찾기

    Args:
        nums: 0~N 범위의 숫자 배열 (하나 누락)

    Returns:
        누락된 숫자

    Time: O(n), Space: O(1)
    """
    i = 0
    n = len(nums)

    while i < n:
        # nums[i]가 올바른 인덱스에 있는지 확인
        # nums[i]가 n이면 범위 밖이므로 건너뜀
        if nums[i] < n and nums[i] != nums[nums[i]]:
            correct_idx = nums[i]
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # 잘못된 위치에 있는 인덱스 = 누락된 숫자
    for i in range(n):
        if nums[i] != i:
            return i

    return n  # 0~n-1 모두 있으면 n이 누락
```

### 템플릿 3: 모든 누락된 숫자 찾기

```python
def find_all_missing(nums: list) -> list:
    """
    1~N 범위에서 모든 누락된 숫자 찾기

    Args:
        nums: 1~N 범위의 숫자 배열 (중복 있음)

    Returns:
        누락된 숫자 리스트

    Time: O(n), Space: O(1) (결과 제외)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    missing = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            missing.append(i + 1)

    return missing
```

### 템플릿 4: 중복된 숫자 찾기

```python
def find_duplicate(nums: list) -> int:
    """
    1~N 범위에서 중복된 숫자 하나 찾기

    Args:
        nums: N+1개의 숫자 (1~N, 하나 중복)

    Returns:
        중복된 숫자

    Time: O(n), Space: O(1)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1

        if nums[i] != i + 1:
            if nums[i] != nums[correct_idx]:
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
            else:
                return nums[i]  # 중복 발견!
        else:
            i += 1

    return -1  # 중복 없음
```

### 템플릿 5: 모든 중복 숫자 찾기

```python
def find_all_duplicates(nums: list) -> list:
    """
    1~N 범위에서 모든 중복 숫자 찾기

    Args:
        nums: 1~N 범위의 숫자 배열 (일부 중복)

    Returns:
        중복된 숫자 리스트

    Time: O(n), Space: O(1) (결과 제외)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    duplicates = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            duplicates.append(nums[i])

    return duplicates
```

### 템플릿 6: 첫 번째 누락된 양수

```python
def first_missing_positive(nums: list) -> int:
    """
    정렬되지 않은 배열에서 첫 번째 누락된 양수 찾기

    Args:
        nums: 정수 배열 (음수, 0 포함 가능)

    Returns:
        첫 번째 누락된 양수

    Time: O(n), Space: O(1)
    """
    n = len(nums)
    i = 0

    while i < n:
        # 유효한 범위(1~n)이고 올바른 위치가 아니면 swap
        correct_idx = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # 첫 번째 잘못된 위치 찾기
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
```

---

## 예제 문제

### 문제 1: 누락된 숫자 (Easy)

**문제 설명**
0부터 N까지의 숫자 중 하나가 누락된 배열이 주어집니다. 누락된 숫자를 찾으세요.

**입력/출력 예시**
```
입력: nums = [3, 0, 1]
출력: 2

입력: nums = [9,6,4,2,3,5,7,0,1]
출력: 8
```

**풀이**
```python
def missing_number(nums: list) -> int:
    i = 0
    n = len(nums)

    while i < n:
        if nums[i] < n and nums[i] != nums[nums[i]]:
            nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
        else:
            i += 1

    for i in range(n):
        if nums[i] != i:
            return i

    return n
```

**다른 풀이 (XOR)**
```python
def missing_number(nums: list) -> int:
    xor = len(nums)
    for i, num in enumerate(nums):
        xor ^= i ^ num
    return xor
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 2: 사라진 모든 숫자 (Easy)

**문제 설명**
1부터 N까지의 숫자가 있어야 하는데, 일부가 중복되고 일부가 누락되었습니다. 누락된 모든 숫자를 찾으세요.

**입력/출력 예시**
```
입력: nums = [4,3,2,7,8,2,3,1]
출력: [5, 6]
```

**풀이**
```python
def find_disappeared_numbers(nums: list) -> list:
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    return [i + 1 for i in range(len(nums)) if nums[i] != i + 1]
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 3: 중복 숫자 찾기 (Medium)

**문제 설명**
N+1개의 정수가 있고, 각 정수는 1부터 N 사이입니다. 정확히 하나의 숫자가 중복됩니다. 중복된 숫자를 찾으세요.

**입력/출력 예시**
```
입력: nums = [1,3,4,2,2]
출력: 2

입력: nums = [3,1,3,4,2]
출력: 3
```

**풀이 (Cyclic Sort)**
```python
def find_the_duplicate(nums: list) -> int:
    i = 0
    while i < len(nums):
        if nums[i] != i + 1:
            correct_idx = nums[i] - 1
            if nums[i] != nums[correct_idx]:
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
            else:
                return nums[i]
        else:
            i += 1
    return -1
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 4: 중복된 모든 숫자 (Medium)

**문제 설명**
1부터 N 사이의 숫자로 구성된 배열에서 두 번 나타나는 모든 숫자를 찾으세요.

**입력/출력 예시**
```
입력: nums = [4,3,2,7,8,2,3,1]
출력: [2, 3]
```

**풀이**
```python
def find_duplicates(nums: list) -> list:
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    return [nums[i] for i in range(len(nums)) if nums[i] != i + 1]
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 5: 첫 번째 누락된 양수 (Hard)

**문제 설명**
정렬되지 않은 정수 배열에서 누락된 가장 작은 양수를 찾으세요.

**입력/출력 예시**
```
입력: nums = [3,4,-1,1]
출력: 2

입력: nums = [7,8,9,11,12]
출력: 1
```

**풀이**
```python
def first_missing_positive(nums: list) -> int:
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 6: K번째 누락된 양수 (Medium)

**문제 설명**
정렬된 배열에서 K번째 누락된 양수를 찾으세요.

**입력/출력 예시**
```
입력: arr = [2,3,4,7,11], k = 5
출력: 9
설명: 누락된 양수: [1,5,6,8,9,...], 5번째는 9
```

**풀이 (Binary Search)**
```python
def find_kth_positive(arr: list, k: int) -> int:
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        # arr[mid] 앞에 누락된 숫자 개수 = arr[mid] - (mid + 1)
        missing_count = arr[mid] - (mid + 1)

        if missing_count < k:
            left = mid + 1
        else:
            right = mid

    # k번째 누락된 숫자 = k + left
    return k + left
```

**복잡도**: 시간 O(log n), 공간 O(1)

---

## Editorial (풀이 전략)

### Step 1: 범위 확인
```
1~N 범위인가? → Cyclic Sort 적용 가능
0~N 범위인가? → 인덱스 조정 (nums[i] → nums[i])
음수/큰 수 포함? → 유효 범위만 처리
```

### Step 2: Swap 조건
```python
# 기본: 현재 값이 올바른 위치에 없으면 swap
if nums[i] != nums[correct_idx]:
    swap

# 0~N 범위: i번 인덱스에 i가 있어야 함
if nums[i] < n and nums[i] != nums[nums[i]]:
    swap

# 음수/큰 수 제외
if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
    swap
```

### Step 3: 결과 추출
```python
# 누락된 숫자: 잘못된 위치의 인덱스+1
if nums[i] != i + 1:
    missing.append(i + 1)

# 중복된 숫자: 잘못된 위치의 값
if nums[i] != i + 1:
    duplicates.append(nums[i])
```

---

## 자주 하는 실수

### 1. 인덱스 계산 오류
```python
# ❌ 1~N 범위인데 인덱스 그대로 사용
correct_idx = nums[i]  # 범위 초과!

# ✅ 올바른 계산
correct_idx = nums[i] - 1  # 숫자 1 → 인덱스 0
```

### 2. 무한 루프
```python
# ❌ swap 조건에서 같은 값 비교
if nums[i] != i + 1:  # 항상 swap하면 무한 루프

# ✅ 목표 위치의 값과 비교
if nums[i] != nums[correct_idx]:  # 다를 때만 swap
```

### 3. 범위 체크 누락
```python
# ❌ 음수/큰 수에서 인덱스 에러
correct_idx = nums[i] - 1  # nums[i] = -1이면?

# ✅ 범위 먼저 체크
if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
```

---

## 관련 패턴

| 패턴 | 관계 |
|------|------|
| **Fast & Slow Pointers** | 중복 찾기의 다른 방법 (Floyd's) |
| **Binary Search** | K번째 누락 숫자 등 |
| **Bit Manipulation** | XOR로 누락/중복 찾기 |

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 268 | Missing Number | Easy |
| 448 | Find All Numbers Disappeared | Easy |
| 287 | Find the Duplicate Number | Medium |
| 442 | Find All Duplicates in an Array | Medium |
| 41 | First Missing Positive | Hard |
| 645 | Set Mismatch | Easy |
| 1539 | Kth Missing Positive Number | Easy |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 10818 | 최소, 최대 | Bronze 3 | 기본 배열 |
| 1546 | 평균 | Bronze 1 | 배열 조작 |
| 3052 | 나머지 | Bronze 2 | 중복 제거 |
| 10807 | 개수 세기 | Bronze 5 | 카운팅 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 없는 숫자 더하기 | Level 1 | 누락 숫자 |
| 모의고사 | Level 1 | 사이클 패턴 |

---

## 임베딩용 키워드

```
cyclic sort, 사이클 정렬, 사이클릭 소트, missing number, 누락된 숫자,
duplicate number, 중복 숫자, 1 to N, 1부터 N까지,
in-place sort, 제자리 정렬, first missing positive, 첫 번째 양수,
swap in place, 스왑, 인덱스 매핑, BOJ, 프로그래머스
```
