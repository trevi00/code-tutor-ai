# Pattern 04: Merge Intervals (구간 병합)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium |
| **빈출도** | ⭐⭐⭐⭐ (높음) |
| **시간복잡도** | O(n log n) |
| **공간복잡도** | O(n) |
| **선행 지식** | 정렬, 배열 |

## 정의

**Merge Intervals**는 여러 개의 구간(interval)이 주어졌을 때, **겹치는 구간을 병합**하거나 **구간 관계를 분석**하는 기법입니다.

## 핵심 아이디어

1. **시작점 기준 정렬**
2. 현재 구간과 다음 구간 비교
3. **겹치면 병합**, 안 겹치면 새 구간 시작

```
정렬 전: [[1,4], [0,2], [3,5]]
정렬 후: [[0,2], [1,4], [3,5]]

병합 과정:
[0,2] ─┬─ [1,4] → [0,4] (겹침: 2 >= 1)
       └─ [3,5] → [0,5] (겹침: 4 >= 3)

결과: [[0,5]]
```

### 구간 관계 6가지

```
1. 분리 (Separate)
   ────     ────
   a  b     c  d   (b < c)

2. 인접 (Adjacent)
   ────────
   a      bc     d   (b == c)

3. 부분 겹침 (Overlapping)
   ──────────
   a    c  b    d   (a < c <= b < d)

4. 포함 (Containing)
   ──────────────
   a    c    d    b   (a <= c && d <= b)

5. 동일 (Equal)
   ──────────
   a,c      b,d   (a == c && b == d)

6. 피포함 (Contained)
   c    a    b    d   (c <= a && b <= d)
```

## 언제 사용하는가?

### ✅ 사용해야 하는 경우
- 겹치는 구간 병합
- 회의실 배정 문제
- 일정 충돌 확인
- 구간 교집합/합집합
- 달력 예약 시스템
- 스카이라인 문제

### ❌ 사용하지 말아야 하는 경우
- 단순 배열 검색 (→ Binary Search)
- 연속 부분배열 (→ Sliding Window)
- 정렬만 필요한 경우

---

## 템플릿 코드

### 템플릿 1: 구간 병합

```python
def merge_intervals(intervals: list) -> list:
    """
    겹치는 구간 병합

    Args:
        intervals: [[start, end], ...] 형태의 구간 리스트

    Returns:
        병합된 구간 리스트

    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []

    # 시작점 기준 정렬
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        # 겹치면 병합 (last의 끝 >= current의 시작)
        if last[1] >= current[0]:
            last[1] = max(last[1], current[1])
        else:
            # 안 겹치면 새 구간 추가
            merged.append(current)

    return merged
```

### 템플릿 2: 구간 삽입

```python
def insert_interval(intervals: list, new: list) -> list:
    """
    정렬된 구간 리스트에 새 구간 삽입 후 병합

    Args:
        intervals: 정렬된 구간 리스트
        new: 삽입할 구간 [start, end]

    Returns:
        병합된 구간 리스트

    Time: O(n), Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)

    # 1. new 앞에 있는 구간들 추가
    while i < n and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1

    # 2. 겹치는 구간들 병합
    while i < n and intervals[i][0] <= new[1]:
        new[0] = min(new[0], intervals[i][0])
        new[1] = max(new[1], intervals[i][1])
        i += 1
    result.append(new)

    # 3. new 뒤에 있는 구간들 추가
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

### 템플릿 3: 구간 교집합

```python
def interval_intersection(A: list, B: list) -> list:
    """
    두 구간 리스트의 교집합

    Args:
        A, B: 각각 정렬된 구간 리스트

    Returns:
        교집합 구간 리스트

    Time: O(n + m), Space: O(n + m)
    """
    result = []
    i = j = 0

    while i < len(A) and j < len(B):
        # 교집합 계산
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:  # 겹치면 추가
            result.append([start, end])

        # 끝이 더 작은 쪽 전진
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result
```

### 템플릿 4: 필요한 회의실 수 (최대 겹침)

```python
def min_meeting_rooms(intervals: list) -> int:
    """
    동시에 진행되는 최대 회의 수 = 필요한 회의실 수

    Args:
        intervals: [[start, end], ...] 회의 시간

    Returns:
        필요한 최소 회의실 수

    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return 0

    # 시작과 끝을 분리해서 정렬
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])

    rooms = 0
    max_rooms = 0
    s = e = 0

    while s < len(intervals):
        if starts[s] < ends[e]:
            rooms += 1  # 회의 시작
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1  # 회의 종료
            e += 1

    return max_rooms
```

### 템플릿 5: 최소 화살로 풍선 터뜨리기

```python
def find_min_arrows(points: list) -> int:
    """
    최소 화살로 모든 풍선 터뜨리기
    (겹치는 구간은 하나의 화살로)

    Args:
        points: [[start, end], ...] 풍선의 x 좌표 범위

    Returns:
        필요한 최소 화살 수

    Time: O(n log n), Space: O(1)
    """
    if not points:
        return 0

    # 끝점 기준 정렬 (그리디)
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]  # 첫 화살 위치

    for start, end in points[1:]:
        # 현재 화살로 터뜨릴 수 없으면
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows
```

---

## 예제 문제

### 문제 1: 구간 병합 (Medium)

**문제 설명**
겹치는 모든 구간을 병합하세요.

**입력/출력 예시**
```
입력: [[1,3],[2,6],[8,10],[15,18]]
출력: [[1,6],[8,10],[15,18]]
설명: [1,3]과 [2,6]이 겹쳐서 [1,6]으로 병합
```

**풀이**
```python
def merge(intervals: list) -> list:
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        if merged[-1][1] >= current[0]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)

    return merged
```

**복잡도**: 시간 O(n log n), 공간 O(n)

---

### 문제 2: 구간 삽입 (Medium)

**문제 설명**
정렬된 비겹침 구간 리스트에 새 구간을 삽입하고 병합하세요.

**입력/출력 예시**
```
입력: intervals = [[1,3],[6,9]], newInterval = [2,5]
출력: [[1,5],[6,9]]
```

**풀이**
```python
def insert(intervals: list, newInterval: list) -> list:
    result = []
    i = 0

    # newInterval 앞의 구간들
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # 겹치는 구간들 병합
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)

    # 나머지 구간들
    while i < len(intervals):
        result.append(intervals[i])
        i += 1

    return result
```

**복잡도**: 시간 O(n), 공간 O(n)

---

### 문제 3: 회의실 II (Medium)

**문제 설명**
회의 시간이 주어질 때, 필요한 최소 회의실 수를 구하세요.

**입력/출력 예시**
```
입력: [[0,30],[5,10],[15,20]]
출력: 2
설명: [0,30]과 [5,10]이 겹침 → 2개 필요
```

**풀이**
```python
def min_meeting_rooms(intervals: list) -> int:
    starts = sorted([i[0] for i in intervals])
    ends = sorted([i[1] for i in intervals])

    rooms = max_rooms = 0
    s = e = 0

    while s < len(intervals):
        if starts[s] < ends[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1

    return max_rooms
```

**복잡도**: 시간 O(n log n), 공간 O(n)

---

### 문제 4: 구간 리스트 교집합 (Medium)

**문제 설명**
두 개의 정렬된 구간 리스트의 교집합을 구하세요.

**입력/출력 예시**
```
입력: A = [[0,2],[5,10],[13,23],[24,25]]
       B = [[1,5],[8,12],[15,24],[25,26]]
출력: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

**풀이**
```python
def interval_intersection(A: list, B: list) -> list:
    result = []
    i = j = 0

    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:
            result.append([start, end])

        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result
```

**복잡도**: 시간 O(n + m), 공간 O(n + m)

---

### 문제 5: 풍선 터뜨리기 최소 화살 (Medium)

**문제 설명**
x축을 따라 풍선이 있습니다. 수직 화살을 쏘아 모든 풍선을 터뜨리는데 필요한 최소 화살 수를 구하세요.

**입력/출력 예시**
```
입력: [[10,16],[2,8],[1,6],[7,12]]
출력: 2
설명: x=6에서 [1,6],[2,8] 터뜨리고, x=11에서 [7,12],[10,16] 터뜨림
```

**풀이**
```python
def find_min_arrows(points: list) -> int:
    if not points:
        return 0

    points.sort(key=lambda x: x[1])  # 끝점 정렬
    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end

    return arrows
```

**복잡도**: 시간 O(n log n), 공간 O(1)

---

## Editorial (풀이 전략)

### Step 1: 정렬 기준 결정

| 문제 유형 | 정렬 기준 |
|-----------|----------|
| 구간 병합 | 시작점 오름차순 |
| 최소 화살 | 끝점 오름차순 |
| 회의실 수 | 시작점/끝점 분리 정렬 |

### Step 2: 겹침 조건 파악

```python
# 두 구간 [a, b], [c, d]가 겹치는 조건
# (a <= c <= b) or (c <= a <= d)
# 간단히: max(a, c) <= min(b, d)

def is_overlapping(interval1, interval2):
    return max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1])
```

### Step 3: 병합 로직

```python
# 겹치면: 새 구간 = [min(시작들), max(끝들)]
merged_start = min(a, c)
merged_end = max(b, d)
```

---

## 자주 하는 실수

### 1. 정렬 안 함
```python
# ❌ 정렬 없이 바로 처리
def merge(intervals):
    merged = [intervals[0]]
    for current in intervals[1:]:
        # 정렬 안 되어 있으면 잘못된 결과!
```

### 2. 병합 시 max 누락
```python
# ❌ current의 끝을 그대로 사용
if last[1] >= current[0]:
    last[1] = current[1]  # 틀림! [1,10]과 [2,5] → [1,5]가 됨

# ✅ max 사용
last[1] = max(last[1], current[1])
```

### 3. 경계 조건 (인접한 경우)
```python
# [1,2]와 [2,3]이 겹치는지?
# 문제에 따라 다름! 명확히 확인할 것

# 겹침으로 처리: >=
if last[1] >= current[0]

# 분리로 처리: >
if last[1] > current[0]
```

---

## 관련 패턴

| 패턴 | 관계 |
|------|------|
| **Greedy** | 최소 화살, 회의실 배정 등 |
| **Heap** | 회의실 II 문제의 다른 풀이 |
| **Sweep Line** | 스카이라인 등 고급 문제 |

---

## LeetCode / BOJ / 프로그래머스 추천 문제

### LeetCode

| # | 문제명 | 난이도 |
|---|-------|-------|
| 56 | Merge Intervals | Medium |
| 57 | Insert Interval | Medium |
| 252 | Meeting Rooms | Easy |
| 253 | Meeting Rooms II | Medium |
| 986 | Interval List Intersections | Medium |
| 452 | Minimum Arrows to Burst Balloons | Medium |
| 435 | Non-overlapping Intervals | Medium |
| 1288 | Remove Covered Intervals | Medium |
| 218 | The Skyline Problem | Hard |

### BOJ (백준)

| # | 문제명 | 난이도 | 유형 |
|---|-------|-------|------|
| 1931 | 회의실 배정 | Silver 1 | 구간 + 그리디 필수 |
| 2217 | 로프 | Silver 4 | 그리디 |
| 11399 | ATM | Silver 4 | 정렬 + 그리디 |
| 1374 | 강의실 | Gold 5 | 회의실 II 유사 |
| 2109 | 순회강연 | Gold 3 | 구간 + 우선순위큐 |
| 13334 | 철로 | Gold 2 | 스위프 라인 |
| 2170 | 선 긋기 | Gold 5 | 구간 병합 |

### 프로그래머스

| 문제명 | 난이도 | 유형 |
|-------|-------|------|
| 호텔 대실 | Level 2 | 회의실 배정 (카카오) |
| 단속카메라 | Level 3 | 구간 + 그리디 |
| 기지국 설치 | Level 3 | 구간 커버 |

---

## 임베딩용 키워드

```
merge intervals, 구간 병합, interval intersection, 구간 교집합,
meeting rooms, 회의실, overlapping intervals, 겹치는 구간,
insert interval, 구간 삽입, minimum arrows, 최소 화살,
sweep line, 스위프 라인, schedule, 스케줄, calendar, 달력,
회의실 배정, 강의실, 단속카메라, BOJ 1931, 프로그래머스
```
