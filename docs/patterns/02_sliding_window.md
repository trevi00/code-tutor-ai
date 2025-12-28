# Pattern 02: Sliding Window (슬라이딩 윈도우)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Easy ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (매우 높음) |
| **시간복잡도** | O(n) |
| **공간복잡도** | O(1) ~ O(k) |
| **선행 지식** | 배열, HashMap |

## 정의

**Sliding Window**는 배열이나 문자열에서 **연속된 부분 구간**을 효율적으로 탐색하는 기법입니다. 윈도우(창문)을 슬라이드하듯 이동하면서 구간의 상태를 업데이트합니다.

## 핵심 아이디어

브루트포스로 모든 부분 배열을 검사하면 O(n²)이지만, Sliding Window는 **이전 윈도우 정보를 재활용**하여 O(n)으로 최적화합니다.

```
브루트포스: 매번 k개 요소 합 계산 → O(n × k)

[1, 3, 2, 6, -1, 4, 1, 8, 2], k=5
 ─────────                    sum = 1+3+2+6+(-1) = 11
    ─────────                 sum = 3+2+6+(-1)+4 = 14 (처음부터 다시 계산)

Sliding Window: 이전 합에서 빠진 것 빼고, 새로 들어온 것 더함 → O(n)

[1, 3, 2, 6, -1, 4, 1, 8, 2], k=5
 ─────────                    sum = 11
    ─────────                 sum = 11 - 1 + 4 = 14 (1 빼고 4 더함)
```

## 언제 사용하는가?

### ✅ 사용해야 하는 경우
- 연속된 부분배열/부분문자열 문제
- "최대/최소 부분배열 합" 문제
- "K개의 요소를 가진 부분배열" 문제
- "조건을 만족하는 가장 긴/짧은 부분" 문제
- 문자열에서 아나그램/패턴 찾기

### ❌ 사용하지 말아야 하는 경우
- 연속되지 않은 요소 조합 (→ Two Pointers, DP)
- 전체 배열 정렬 필요 (→ 정렬 알고리즘)
- 부분집합 문제 (→ Backtracking, DP)

## 패턴 유형

### 유형 1: 고정 크기 윈도우 (Fixed Size)

윈도우 크기가 **K로 고정**된 경우

```
[1, 3, 2, 6, -1, 4, 1, 8, 2], k=3
 ─────                        sum = 6
    ─────                     sum = 11 (6 - 1 + 6 = 11)
       ─────                  sum = 7
```

### 유형 2: 가변 크기 윈도우 (Variable Size)

**조건을 만족**할 때까지 윈도우 확장/축소

```
목표: 합이 >= 7인 최소 길이 부분배열

[2, 3, 1, 2, 4, 3], target=7
 ──────           sum=6 (< 7, 확장)
 ─────────        sum=8 (>= 7, 길이 4 기록, 축소)
    ──────        sum=6 (< 7, 확장)
    ─────────     sum=10 (>= 7, 길이 4, 축소)
       ──────     sum=7 (>= 7, 길이 3, 축소)
          ────    sum=7 (>= 7, 길이 2 기록!) ← 최소
```

---

## 템플릿 코드

### 템플릿 1: 고정 크기 윈도우

```python
def fixed_sliding_window(arr: list, k: int) -> int:
    """
    크기 K인 윈도우의 최대 합

    Args:
        arr: 정수 배열
        k: 윈도우 크기

    Returns:
        최대 합

    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return 0

    # 첫 윈도우 계산
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # 윈도우 슬라이드
    for i in range(k, len(arr)):
        window_sum += arr[i]      # 새 요소 추가
        window_sum -= arr[i - k]  # 오래된 요소 제거
        max_sum = max(max_sum, window_sum)

    return max_sum
```

### 템플릿 2: 가변 크기 윈도우 (최소 길이)

```python
def variable_sliding_window_min(arr: list, target: int) -> int:
    """
    합이 target 이상인 최소 길이 부분배열

    Args:
        arr: 양의 정수 배열
        target: 목표 합

    Returns:
        최소 길이 (없으면 0)

    Time: O(n), Space: O(1)
    """
    left = 0
    window_sum = 0
    min_length = float('inf')

    for right in range(len(arr)):
        window_sum += arr[right]  # 윈도우 확장

        # 조건 만족하면 축소 시도
        while window_sum >= target:
            min_length = min(min_length, right - left + 1)
            window_sum -= arr[left]
            left += 1

    return min_length if min_length != float('inf') else 0
```

### 템플릿 3: 가변 크기 윈도우 (최대 길이)

```python
def variable_sliding_window_max(s: str, k: int) -> int:
    """
    최대 K개의 고유 문자를 가진 가장 긴 부분문자열

    Args:
        s: 문자열
        k: 허용되는 고유 문자 수

    Returns:
        최대 길이

    Time: O(n), Space: O(k)
    """
    from collections import defaultdict

    left = 0
    char_count = defaultdict(int)
    max_length = 0

    for right in range(len(s)):
        char_count[s[right]] += 1  # 윈도우 확장

        # 조건 위반 시 축소
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

### 템플릿 4: 문자열 패턴 매칭 (아나그램)

```python
def find_anagrams(s: str, p: str) -> list:
    """
    문자열 s에서 p의 아나그램 시작 인덱스 찾기

    Args:
        s: 검색 대상 문자열
        p: 패턴 문자열

    Returns:
        아나그램 시작 인덱스 리스트

    Time: O(n), Space: O(1) (알파벳 26개 고정)
    """
    from collections import Counter

    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window_count = Counter(s[:len(p)])
    result = []

    if window_count == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # 새 문자 추가
        window_count[s[i]] += 1

        # 오래된 문자 제거
        old_char = s[i - len(p)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        if window_count == p_count:
            result.append(i - len(p) + 1)

    return result
```

---

## 예제 문제

### 문제 1: 최대 부분배열 합 (Easy)

**문제 설명**
정수 배열과 정수 `k`가 주어집니다. 크기가 `k`인 연속 부분배열 중 합이 최대인 값을 반환하세요.

**입력/출력 예시**
```
입력: arr = [2, 1, 5, 1, 3, 2], k = 3
출력: 9
설명: [5, 1, 3]의 합이 9로 최대
```

**풀이**
```python
def max_subarray_sum(arr: list, k: int) -> int:
    if len(arr) < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum + arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 2: 최소 크기 부분배열 합 (Medium)

**문제 설명**
양의 정수 배열과 양의 정수 `target`이 주어집니다. 합이 `target` 이상인 최소 길이 부분배열을 찾으세요.

**입력/출력 예시**
```
입력: target = 7, nums = [2, 3, 1, 2, 4, 3]
출력: 2
설명: [4, 3]의 합이 7이고 길이가 2로 최소
```

**풀이**
```python
def min_subarray_len(target: int, nums: list) -> int:
    left = 0
    current_sum = 0
    min_length = float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]

        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_length if min_length != float('inf') else 0
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 3: 가장 긴 부분문자열 (K개 고유 문자) (Medium)

**문제 설명**
문자열 `s`와 정수 `k`가 주어집니다. 최대 `k`개의 고유 문자를 포함하는 가장 긴 부분문자열 길이를 반환하세요.

**입력/출력 예시**
```
입력: s = "araaci", k = 2
출력: 4
설명: "araa"가 2개의 고유 문자('a', 'r')로 길이 4
```

**풀이**
```python
def longest_substring_k_distinct(s: str, k: int) -> int:
    from collections import defaultdict

    left = 0
    char_count = defaultdict(int)
    max_length = 0

    for right in range(len(s)):
        char_count[s[right]] += 1

        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**복잡도**: 시간 O(n), 공간 O(k)

---

### 문제 4: 반복 문자 없는 가장 긴 부분문자열 (Medium)

**문제 설명**
문자열 `s`가 주어집니다. 반복 문자가 없는 가장 긴 부분문자열 길이를 반환하세요.

**입력/출력 예시**
```
입력: s = "abcabcbb"
출력: 3
설명: "abc"가 반복 없이 길이 3으로 최대
```

**풀이**
```python
def length_of_longest_substring(s: str) -> int:
    left = 0
    char_index = {}  # 각 문자의 마지막 인덱스
    max_length = 0

    for right in range(len(s)):
        # 중복 문자가 현재 윈도우 내에 있으면
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1

        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

**복잡도**: 시간 O(n), 공간 O(min(n, 알파벳 크기))

---

### 문제 5: 문자열 내 모든 아나그램 (Medium)

**문제 설명**
문자열 `s`와 패턴 `p`가 주어집니다. `s`에서 `p`의 아나그램 시작 인덱스를 모두 반환하세요.

**입력/출력 예시**
```
입력: s = "cbaebabacd", p = "abc"
출력: [0, 6]
설명: 인덱스 0의 "cba"와 인덱스 6의 "bac"가 "abc"의 아나그램
```

**풀이**
```python
def find_anagrams(s: str, p: str) -> list:
    from collections import Counter

    if len(p) > len(s):
        return []

    p_count = Counter(p)
    window_count = Counter()
    result = []
    k = len(p)

    for i in range(len(s)):
        # 윈도우에 새 문자 추가
        window_count[s[i]] += 1

        # 윈도우 크기 초과 시 왼쪽 문자 제거
        if i >= k:
            left_char = s[i - k]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

        # 아나그램 확인
        if window_count == p_count:
            result.append(i - k + 1)

    return result
```

**복잡도**: 시간 O(n), 공간 O(1)

---

### 문제 6: 최대 연속 1의 개수 III (Medium)

**문제 설명**
이진 배열 `nums`와 정수 `k`가 주어집니다. 최대 `k`개의 0을 1로 바꿀 수 있을 때, 연속된 1의 최대 길이를 반환하세요.

**입력/출력 예시**
```
입력: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
출력: 6
설명: [1,1,1,0,0,1,1,1,1,1,1] 마지막 6개 (0 두 개를 1로 변경)
```

**풀이**
```python
def longest_ones(nums: list, k: int) -> int:
    left = 0
    zeros_count = 0
    max_length = 0

    for right in range(len(nums)):
        if nums[right] == 0:
            zeros_count += 1

        while zeros_count > k:
            if nums[left] == 0:
                zeros_count -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**복잡도**: 시간 O(n), 공간 O(1)

---

## Editorial (풀이 전략)

### Step 1: 윈도우 유형 파악

| 키워드 | 윈도우 유형 |
|--------|------------|
| "크기 K인 부분배열" | 고정 크기 |
| "최대/최소 길이" | 가변 크기 |
| "조건을 만족하는" | 가변 크기 |
| "연속된" | Sliding Window 가능 |

### Step 2: 상태 변수 결정

```python
# 고정 크기: 합만 추적
window_sum = 0

# 가변 크기: 합 + 길이
window_sum = 0
min_length = float('inf')

# 문자열: 빈도수 맵
char_count = {}
```

### Step 3: 확장/축소 조건 설정

```python
# 고정 크기
for right in range(len(arr)):
    window_sum += arr[right]
    if right >= k - 1:  # 윈도우 크기 도달
        # 결과 갱신
        window_sum -= arr[right - k + 1]  # 왼쪽 제거

# 가변 크기
for right in range(len(arr)):
    window_sum += arr[right]  # 확장
    while condition_violated:  # 조건 위반 시
        window_sum -= arr[left]  # 축소
        left += 1
    # 결과 갱신
```

### Step 4: 결과 갱신 위치

- **최대 길이**: 축소 후 갱신
- **최소 길이**: 축소 전 (조건 만족할 때) 갱신
- **고정 크기**: 윈도우 완성 후 갱신

---

## 자주 하는 실수

### 1. 초기 윈도우 처리 누락
```python
# ❌ 잘못된 코드
for i in range(len(arr)):
    window_sum += arr[i] - arr[i - k]  # i < k일 때 인덱스 오류!

# ✅ 올바른 코드
window_sum = sum(arr[:k])  # 첫 윈도우 별도 계산
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i - k]
```

### 2. while vs if 혼동
```python
# ❌ 잘못된 코드 (if 사용)
if window_sum >= target:
    min_length = min(min_length, right - left + 1)
    window_sum -= arr[left]
    left += 1
# 한 번만 축소해서 최소 길이 못 찾음

# ✅ 올바른 코드 (while 사용)
while window_sum >= target:  # 조건 만족하는 동안 계속 축소
    min_length = min(min_length, right - left + 1)
    window_sum -= arr[left]
    left += 1
```

### 3. HashMap에서 0인 키 제거 안 함
```python
# ❌ 잘못된 코드
char_count[s[left]] -= 1
# len(char_count)가 줄지 않음!

# ✅ 올바른 코드
char_count[s[left]] -= 1
if char_count[s[left]] == 0:
    del char_count[s[left]]
```

### 4. 경계 조건 누락
```python
# ❌ 빈 배열/문자열 처리 안 함
def func(arr, k):
    window_sum = sum(arr[:k])  # arr이 비어있으면 에러

# ✅ 경계 조건 체크
def func(arr, k):
    if not arr or k > len(arr):
        return 0
    window_sum = sum(arr[:k])
```

---

## 관련 패턴

| 패턴 | 관계 |
|------|------|
| **Two Pointers** | 비연속 쌍은 Two Pointers, 연속 구간은 Sliding Window |
| **Prefix Sum** | 구간 합을 빠르게 계산할 때 활용 가능 |
| **HashMap** | 빈도수 추적에 자주 함께 사용 |

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 643 | Maximum Average Subarray I | Easy | 고정 |
| 209 | Minimum Size Subarray Sum | Medium | 가변 |
| 3 | Longest Substring Without Repeating | Medium | 가변 |
| 76 | Minimum Window Substring | Hard | 가변 |
| 438 | Find All Anagrams in a String | Medium | 고정 |
| 567 | Permutation in String | Medium | 고정 |
| 424 | Longest Repeating Character Replacement | Medium | 가변 |
| 1004 | Max Consecutive Ones III | Medium | 가변 |
| 239 | Sliding Window Maximum | Hard | 고정+Deque |
| 480 | Sliding Window Median | Hard | 심화 |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 2559 | 수열 | Silver 3 | 고정 크기 기본 |
| 2003 | 수들의 합 2 | Silver 4 | 가변 크기 |
| 1806 | 부분합 | Gold 4 | 가변 크기 필수 |
| 11003 | 최솟값 찾기 | Platinum 5 | 고정+Deque |
| 2096 | 내려가기 | Gold 4 | 슬라이딩 윈도우 DP |
| 10025 | 게으른 백곰 | Silver 3 | 고정 크기 |
| 15961 | 회전 초밥 | Gold 4 | 고정 크기 + 해시 |
| 2531 | 회전 초밥 | Silver 1 | 고정 크기 |
| 20437 | 문자열 게임 2 | Gold 5 | 가변 크기 |
| 12891 | DNA 비밀번호 | Silver 2 | 고정 크기 + 조건 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 연속 부분 수열 합의 개수 | Level 2 | 원형 배열 |
| 할인 행사 | Level 2 | 고정 크기 |
| 보석 쇼핑 | Level 3 | 가변 크기 (카카오) |
| 광고 삽입 | Level 3 | 누적합 + 슬라이딩 (카카오) |

---

## 임베딩용 키워드

```
sliding window, 슬라이딩 윈도우, 고정 크기, 가변 크기, fixed size, variable size,
연속 부분배열, contiguous subarray, 부분문자열, substring,
최대 합, maximum sum, 최소 길이, minimum length,
아나그램, anagram, 패턴 매칭, pattern matching,
윈도우 확장, 윈도우 축소, expand, shrink,
부분합, 회전 초밥, 보석 쇼핑, BOJ 1806, BOJ 2559, 프로그래머스
```
