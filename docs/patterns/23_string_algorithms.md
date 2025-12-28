# Pattern 23: String Algorithms (문자열 알고리즘)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐ (간헐적 출제) |
| **시간복잡도** | O(n) ~ O(n+m) |
| **공간복잡도** | O(n) |
| **선행 지식** | 문자열 기초, 해시 |

## 정의

**문자열 알고리즘**은 문자열에서 **패턴 검색**, **일치**, **변환** 등을 효율적으로 수행하는 알고리즘입니다.

## 주요 알고리즘

| 알고리즘 | 용도 | 시간복잡도 |
|---------|------|-----------|
| **KMP** | 패턴 매칭 | O(n+m) |
| **라빈-카프** | 패턴 매칭 (해시) | O(n+m) 평균 |
| **Z 알고리즘** | 접두사 일치 | O(n) |
| **Manacher** | 최장 팰린드롬 | O(n) |
| **문자열 해싱** | 부분 문자열 비교 | O(1) per query |

---

## 템플릿 코드

### 템플릿 1: KMP (Knuth-Morris-Pratt)

```python
def compute_lps(pattern: str) -> list:
    """
    LPS (Longest Proper Prefix which is also Suffix) 배열 계산

    Time: O(m)
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps


def kmp_search(text: str, pattern: str) -> list:
    """
    KMP 패턴 검색

    Time: O(n + m)
    Space: O(m)
    """
    n, m = len(text), len(pattern)

    if m == 0:
        return []

    lps = compute_lps(pattern)
    result = []

    i = 0  # text 인덱스
    j = 0  # pattern 인덱스

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

            if j == m:
                result.append(i - j)  # 매칭 시작 위치
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return result


def kmp_count(text: str, pattern: str) -> int:
    """패턴 등장 횟수"""
    return len(kmp_search(text, pattern))
```

### 템플릿 2: 라빈-카프 (Rabin-Karp)

```python
def rabin_karp(text: str, pattern: str, base: int = 31, mod: int = 10**9 + 9) -> list:
    """
    라빈-카프 해시 기반 패턴 검색

    Time: O(n + m) 평균, O(nm) 최악
    Space: O(1)
    """
    n, m = len(text), len(pattern)

    if m > n:
        return []

    # 패턴 해시 계산
    pattern_hash = 0
    text_hash = 0
    power = 1

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod
        if i < m - 1:
            power = (power * base) % mod

    result = []

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # 해시 충돌 확인
            if text[i:i + m] == pattern:
                result.append(i)

        # 슬라이딩 윈도우 해시 업데이트
        if i < n - m:
            text_hash = (text_hash - ord(text[i]) * power) % mod
            text_hash = (text_hash * base + ord(text[i + m])) % mod
            text_hash = (text_hash + mod) % mod  # 음수 방지

    return result
```

### 템플릿 3: 문자열 해싱 (부분 문자열 비교)

```python
class StringHash:
    """
    문자열 해싱 - O(1) 부분 문자열 비교

    Time: O(n) 전처리, O(1) 쿼리
    """
    def __init__(self, s: str, base: int = 31, mod: int = 10**9 + 9):
        self.n = len(s)
        self.base = base
        self.mod = mod

        # 해시 전처리
        self.hash = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash[i + 1] = (self.hash[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, left: int, right: int) -> int:
        """s[left:right+1]의 해시값"""
        h = (self.hash[right + 1] - self.hash[left] * self.power[right - left + 1]) % self.mod
        return (h + self.mod) % self.mod

    def compare(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """두 부분 문자열 비교"""
        if r1 - l1 != r2 - l2:
            return False
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)
```

### 템플릿 4: Z 알고리즘

```python
def z_function(s: str) -> list:
    """
    Z 배열 계산
    z[i] = s[i:]와 s의 최장 공통 접두사 길이

    Time: O(n)
    """
    n = len(s)
    z = [0] * n
    z[0] = n

    l, r = 0, 0

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z


def z_pattern_search(text: str, pattern: str) -> list:
    """Z 알고리즘을 이용한 패턴 검색"""
    concat = pattern + "$" + text
    z = z_function(concat)
    m = len(pattern)

    return [i - m - 1 for i in range(m + 1, len(concat)) if z[i] == m]
```

### 템플릿 5: Manacher 알고리즘 (최장 팰린드롬)

```python
def manacher(s: str) -> list:
    """
    Manacher 알고리즘 - 모든 위치에서 최장 팰린드롬 반지름

    Time: O(n)
    """
    # 문자 사이에 # 삽입 (짝수 길이 팰린드롬 처리)
    t = '#' + '#'.join(s) + '#'
    n = len(t)

    p = [0] * n  # 반지름
    c = r = 0    # 중심, 오른쪽 경계

    for i in range(n):
        if i < r:
            mirror = 2 * c - i
            p[i] = min(r - i, p[mirror])

        # 확장 시도
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        # 경계 업데이트
        if i + p[i] > r:
            c, r = i, i + p[i]

    return p


def longest_palindrome(s: str) -> str:
    """최장 팰린드롬 부분 문자열"""
    if not s:
        return ""

    t = '#' + '#'.join(s) + '#'
    p = manacher(s)

    max_len = max(p)
    center = p.index(max_len)

    # 원래 문자열에서 추출
    start = (center - max_len) // 2
    return s[start:start + max_len]
```

### 템플릿 6: 접미사 배열 (간단 버전)

```python
def suffix_array_simple(s: str) -> list:
    """
    접미사 배열 (간단 버전)

    Time: O(n log² n)
    """
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [idx for _, idx in suffixes]


def lcp_array(s: str, sa: list) -> list:
    """
    LCP 배열 (Longest Common Prefix)

    Time: O(n)
    """
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    for i, idx in enumerate(sa):
        rank[idx] = i

    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue

        j = sa[rank[i] - 1]

        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1

        lcp[rank[i]] = k

        if k > 0:
            k -= 1

    return lcp
```

---

## 예제 문제

### 문제 1: 찾기 (BOJ 1786) - Platinum 5

**문제 설명**
문자열 T에서 패턴 P가 등장하는 모든 위치를 찾으세요.

**풀이**
```python
def solve(T: str, P: str) -> tuple:
    positions = kmp_search(T, P)
    return len(positions), positions
```

---

### 문제 2: 접두사 (프로그래머스) - Level 2

**문제 설명**
전화번호 목록에서 접두사 관계가 있는지 확인.

**풀이 (정렬 + 비교)**
```python
def solution(phone_book: list) -> bool:
    phone_book.sort()

    for i in range(len(phone_book) - 1):
        if phone_book[i + 1].startswith(phone_book[i]):
            return False

    return True
```

---

### 문제 3: 가장 긴 팰린드롬 (프로그래머스 Level 3)

**문제 설명**
문자열에서 가장 긴 팰린드롬의 길이.

**풀이**
```python
def solution(s: str) -> int:
    t = '#' + '#'.join(s) + '#'
    n = len(t)

    p = [0] * n
    c = r = 0

    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])

        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        if i + p[i] > r:
            c, r = i, i + p[i]

    return max(p)
```

---

### 문제 4: 부분 문자열 뒤집기 (해싱)

**문제 설명**
문자열의 부분 문자열을 뒤집어서 팰린드롬을 만들 수 있는지.

**풀이**
```python
def can_make_palindrome(s: str) -> bool:
    n = len(s)

    # 정방향, 역방향 해시
    forward = StringHash(s)
    backward = StringHash(s[::-1])

    # 모든 부분 문자열에 대해 검사
    for i in range(n):
        for j in range(i, n):
            # s[i:j+1]을 뒤집었을 때 팰린드롬인지
            # 해시로 O(1) 비교
            ...

    return False
```

---

### 문제 5: 문자열 검색 (여러 패턴)

**문제 설명**
텍스트에서 여러 패턴의 등장 횟수.

**풀이 (각 패턴별 KMP)**
```python
def multi_pattern_search(text: str, patterns: list) -> dict:
    result = {}

    for pattern in patterns:
        result[pattern] = kmp_count(text, pattern)

    return result
```

---

## Editorial (풀이 전략)

### Step 1: 알고리즘 선택

| 상황 | 알고리즘 |
|------|---------|
| 단일 패턴 검색 | KMP |
| 여러 패턴 | Aho-Corasick / 각각 KMP |
| 부분 문자열 비교 | 문자열 해싱 |
| 최장 팰린드롬 | Manacher |
| 접두사 관련 | Z 알고리즘 |

### Step 2: KMP LPS 배열 이해

```
Pattern: "ABABAC"
LPS:     [0, 0, 1, 2, 3, 0]

의미: 각 위치에서 가장 긴 proper prefix = suffix 길이
- lps[4] = 3: "ABABA"에서 "ABA" = "ABA" (앞3, 뒤3)
```

### Step 3: 해시 충돌 방지

```python
# 두 개의 해시 사용
hash1 = get_hash(s, base=31, mod=10**9 + 7)
hash2 = get_hash(s, base=37, mod=10**9 + 9)

# 둘 다 일치해야 같은 문자열
```

---

## 자주 하는 실수

### 1. LPS 배열 계산 오류
```python
# ❌ i = 0부터 시작
for i in range(m):
    ...

# ✅ i = 1부터 시작 (lps[0] = 0 고정)
i = 1
while i < m:
    ...
```

### 2. 해시 음수 처리
```python
# ❌ 음수 해시 가능
hash = (hash - old * power) % mod

# ✅ 음수 방지
hash = ((hash - old * power) % mod + mod) % mod
```

### 3. 팰린드롬 인덱스 변환
```python
# # 삽입 후 인덱스와 원본 인덱스 변환
# t = '#a#b#a#'에서 인덱스 3 → 원본 인덱스 1
original_idx = (t_idx - 1) // 2
```

---

## LeetCode / BOJ 추천 문제

| 플랫폼 | # | 문제명 | 난이도 | 알고리즘 |
|--------|---|-------|-------|---------|
| LeetCode | 28 | Find the Index of First Occurrence | Easy | KMP |
| LeetCode | 5 | Longest Palindromic Substring | Medium | Manacher |
| LeetCode | 214 | Shortest Palindrome | Hard | KMP |
| LeetCode | 1392 | Longest Happy Prefix | Hard | Z/KMP |
| BOJ | 1786 | 찾기 | Platinum 5 | KMP |
| BOJ | 1305 | 광고 | Platinum 4 | KMP |
| BOJ | 10266 | 시계 사진들 | Platinum 4 | KMP |
| BOJ | 13713 | 문자열과 쿼리 | Platinum 2 | 해싱 |
| 프로그래머스 | - | 가장 긴 팰린드롬 | Level 3 | Manacher |

---

## 임베딩용 키워드

```
string algorithm, 문자열 알고리즘, KMP, Knuth-Morris-Pratt,
rabin karp, 라빈 카프, pattern matching, 패턴 매칭,
string hashing, 문자열 해싱, Z algorithm, Z 알고리즘,
manacher, 마나커, palindrome, 팰린드롬, LPS, failure function
```
