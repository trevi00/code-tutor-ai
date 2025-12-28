# Pattern 09: Binary Search (이진 탐색)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(log n) |
| **공간복잡도** | O(1) |
| **선행 지식** | 정렬, 조건문 |

## 정의

**Binary Search (이진 탐색)**는 **정렬된** 배열에서 탐색 범위를 반씩 줄여가며 원하는 값을 찾는 알고리즘입니다.

## 핵심 아이디어

```
[1, 3, 5, 7, 9, 11, 13, 15, 17], target = 11

Step 1: left=0, right=8, mid=4 → arr[4]=9 < 11 → left=5
Step 2: left=5, right=8, mid=6 → arr[6]=13 > 11 → right=5
Step 3: left=5, right=5, mid=5 → arr[5]=11 = 11 ✓

탐색 횟수: 3회 (log₂9 ≈ 3.17)
```

---

## 템플릿 코드

### 템플릿 1: 기본 이진 탐색 (정확히 찾기)

```python
def binary_search(arr: list, target: int) -> int:
    """
    정확한 값의 인덱스 찾기

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # 없음
```

### 템플릿 2: Lower Bound (첫 번째 >= target)

```python
def lower_bound(arr: list, target: int) -> int:
    """
    target 이상인 첫 번째 인덱스

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 템플릿 3: Upper Bound (첫 번째 > target)

```python
def upper_bound(arr: list, target: int) -> int:
    """
    target 초과인 첫 번째 인덱스

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left
```

### 템플릿 4: 조건 만족하는 최솟값 찾기

```python
def find_minimum_satisfying(arr: list, condition) -> int:
    """
    condition(x)가 True인 최소 x 찾기
    (False, False, ..., True, True 형태)

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if condition(arr[mid]):
            result = mid
            right = mid - 1  # 더 작은 쪽 탐색
        else:
            left = mid + 1

    return result
```

### 템플릿 5: 회전 배열 탐색

```python
def search_rotated(nums: list, target: int) -> int:
    """
    회전된 정렬 배열에서 탐색

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # 왼쪽 정렬됨
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 오른쪽 정렬됨
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

### 템플릿 6: 제곱근/N번째 근 구하기

```python
def sqrt(x: int) -> int:
    """
    x의 제곱근 (정수 부분)

    Time: O(log x), Space: O(1)
    """
    if x < 2:
        return x

    left, right = 1, x // 2

    while left <= right:
        mid = left + (right - left) // 2

        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1

    return right
```

---

## 예제 문제

### 문제 1: 이진 탐색 (Easy)

**문제 설명**
정렬된 배열에서 target의 인덱스를 찾으세요.

**입력/출력 예시**
```
입력: nums = [-1,0,3,5,9,12], target = 9
출력: 4
```

**풀이**
```python
def search(nums: list, target: int) -> int:
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
```

---

### 문제 2: 첫 번째와 마지막 위치 (Medium)

**문제 설명**
정렬된 배열에서 target의 첫 번째와 마지막 인덱스를 찾으세요.

**입력/출력 예시**
```
입력: nums = [5,7,7,8,8,10], target = 8
출력: [3, 4]
```

**풀이**
```python
def search_range(nums: list, target: int) -> list:
    def find_first():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def find_last():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    first = find_first()
    last = find_last()

    if first <= last and 0 <= first < len(nums) and nums[first] == target:
        return [first, last]
    return [-1, -1]
```

---

### 문제 3: 회전 배열 탐색 (Medium)

**문제 설명**
한 지점에서 회전된 정렬 배열에서 target을 찾으세요.

**입력/출력 예시**
```
입력: nums = [4,5,6,7,0,1,2], target = 0
출력: 4
```

**풀이**
```python
def search(nums: list, target: int) -> int:
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
```

---

### 문제 4: 회전 배열 최솟값 (Medium)

**문제 설명**
회전된 정렬 배열의 최솟값을 찾으세요.

**입력/출력 예시**
```
입력: nums = [3,4,5,1,2]
출력: 1
```

**풀이**
```python
def find_min(nums: list) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]
```

---

### 문제 5: 배열에서 피크 원소 (Medium)

**문제 설명**
이웃한 요소보다 큰 피크 원소의 인덱스를 찾으세요.

**입력/출력 예시**
```
입력: nums = [1,2,3,1]
출력: 2 (nums[2] = 3이 피크)
```

**풀이**
```python
def find_peak_element(nums: list) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[mid + 1]:
            right = mid  # 피크는 왼쪽에
        else:
            left = mid + 1  # 피크는 오른쪽에

    return left
```

---

## Editorial (풀이 전략)

### Step 1: 탐색 조건 정의

| 상황 | left 이동 조건 | right 이동 조건 |
|------|---------------|----------------|
| 정확히 찾기 | arr[mid] < target | arr[mid] > target |
| Lower Bound | arr[mid] < target | arr[mid] >= target |
| Upper Bound | arr[mid] <= target | arr[mid] > target |

### Step 2: 종료 조건

```python
# left <= right: 정확한 값 찾기
while left <= right:

# left < right: 범위 좁히기 (답이 항상 존재)
while left < right:
```

### Step 3: mid 계산 (오버플로우 방지)

```python
# ✅ 안전한 방법
mid = left + (right - left) // 2

# ⚠️ 큰 수에서 오버플로우 가능 (Python은 괜찮음)
mid = (left + right) // 2
```

---

## 자주 하는 실수

### 1. 무한 루프
```python
# ❌ mid가 변하지 않음
left = mid  # right = mid - 1이어야 함

# ✅ 범위가 항상 줄어들도록
left = mid + 1
right = mid - 1
```

### 2. 경계 조건
```python
# ❌ 인덱스 초과
while left < right:
    if nums[mid + 1] ...  # right = mid일 때 mid+1 초과 가능

# ✅ 조건 체크
if mid + 1 < len(nums):
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 704 | Binary Search | Easy |
| 35 | Search Insert Position | Easy |
| 34 | First and Last Position | Medium |
| 33 | Search in Rotated Array | Medium |
| 153 | Find Minimum in Rotated Array | Medium |
| 162 | Find Peak Element | Medium |
| 69 | Sqrt(x) | Easy |
| 74 | Search a 2D Matrix | Medium |
| 875 | Koko Eating Bananas | Medium |
| 410 | Split Array Largest Sum | Hard |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1920 | 수 찾기 | Silver 4 | 기본 |
| 10816 | 숫자 카드 2 | Silver 4 | Lower/Upper Bound |
| 2805 | 나무 자르기 | Silver 2 | 파라메트릭 서치 필수 |
| 1654 | 랜선 자르기 | Silver 2 | 파라메트릭 서치 필수 |
| 2110 | 공유기 설치 | Gold 4 | 파라메트릭 서치 |
| 12015 | 가장 긴 증가하는 부분 수열 2 | Gold 2 | LIS + 이분탐색 |
| 1300 | K번째 수 | Gold 1 | 이분탐색 응용 |
| 2512 | 예산 | Silver 2 | 파라메트릭 서치 |
| 3020 | 개똥벌레 | Gold 5 | 누적합 + 이분탐색 |
| 2343 | 기타 레슨 | Silver 1 | 파라메트릭 서치 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 입국심사 | Level 3 | 파라메트릭 서치 |
| 징검다리 | Level 4 | 파라메트릭 서치 (카카오) |
| 순위 검색 | Level 2 | Lower Bound (카카오) |
| 징검다리 건너기 | Level 3 | 이분탐색 (카카오) |

---

## 임베딩용 키워드

```
binary search, 이진 탐색, 이분 탐색, lower bound, upper bound,
rotated array, 회전 배열, sorted array, 정렬 배열,
search insert position, peak element, 피크 원소,
O(log n), 로그 시간, bisect, 이분법,
파라메트릭 서치, parametric search, 나무 자르기, 랜선 자르기, 입국심사, BOJ, 프로그래머스
```
