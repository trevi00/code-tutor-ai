# Pattern 19: Trie (트라이 / 접두사 트리)

## 개요

| 항목 | 내용 |
|------|------|
| **난이도** | Medium ~ Hard |
| **빈출도** | ⭐⭐⭐⭐⭐ (카카오 매년 출제) |
| **시간복잡도** | O(L) per operation |
| **공간복잡도** | O(N × L × 26) |
| **선행 지식** | 트리, 문자열 |

## 정의

**트라이(Trie)**는 문자열을 효율적으로 저장하고 검색하기 위한 트리 자료구조입니다. **접두사(Prefix)** 기반 검색에 최적화되어 있습니다.

## 핵심 아이디어

```
단어: ["apple", "app", "apt", "bat"]

        root
       /    \
      a      b
      |      |
      p      a
     /|\     |
    p t .    t
    |        |
    l        .
    |
    e
    |
    .

. = 단어 끝 표시

검색 "app": root → a → p → p (끝 표시 ✓) → 존재
접두사 "ap": root → a → p → 존재
```

## Trie vs 해시맵

| 특성 | Trie | 해시맵 |
|------|------|-------|
| 정확한 검색 | O(L) | O(L) |
| 접두사 검색 | O(L) ✅ | O(N×L) ❌ |
| 자동완성 | 쉬움 ✅ | 어려움 |
| 와일드카드 | 가능 | 어려움 |
| 공간 | 많음 | 적음 |

---

## 템플릿 코드

### 템플릿 1: 기본 Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    """
    기본 트라이 자료구조

    Time: O(L) per operation
    Space: O(N × L × alphabet_size)
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """단어 삽입"""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True

    def search(self, word: str) -> bool:
        """정확한 단어 검색"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """접두사 존재 여부"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """접두사에 해당하는 노드 찾기"""
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node
```

### 템플릿 2: 자동완성 (접두사로 시작하는 모든 단어)

```python
class AutocompleteTrie(Trie):
    """
    자동완성 기능 트라이
    """
    def autocomplete(self, prefix: str) -> list:
        """접두사로 시작하는 모든 단어 반환"""
        node = self._find_node(prefix)

        if node is None:
            return []

        results = []
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node: TrieNode, current: str, results: list):
        if node.is_end:
            results.append(current)

        for char, child in node.children.items():
            self._dfs(child, current + char, results)

    def autocomplete_top_k(self, prefix: str, k: int) -> list:
        """접두사로 시작하는 단어 중 상위 K개"""
        words = self.autocomplete(prefix)
        return words[:k]
```

### 템플릿 3: 단어 개수 추적

```python
class CountTrie:
    """
    각 접두사/단어의 개수를 추적하는 트라이
    """
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        node = self.root

        for char in word:
            if char not in node:
                node[char] = {'#count': 0, '#end': 0}
            node[char]['#count'] += 1  # 접두사 개수
            node = node[char]

        node['#end'] += 1  # 단어 개수

    def count_prefix(self, prefix: str) -> int:
        """접두사로 시작하는 단어 개수"""
        node = self.root

        for char in prefix:
            if char not in node:
                return 0
            node = node[char]

        return node['#count']

    def count_word(self, word: str) -> int:
        """정확히 일치하는 단어 개수"""
        node = self.root

        for char in word:
            if char not in node:
                return 0
            node = node[char]

        return node.get('#end', 0)
```

### 템플릿 4: 와일드카드 검색 (카카오 가사 검색)

```python
class WildcardTrie:
    """
    와일드카드(?) 검색 지원 트라이
    카카오 2020 가사 검색 문제
    """
    def __init__(self):
        self.root = {}
        self.reverse_root = {}  # 뒤집은 단어용

    def insert(self, word: str) -> None:
        # 정방향 삽입
        self._insert_to(self.root, word)
        # 역방향 삽입 (접미사 검색용)
        self._insert_to(self.reverse_root, word[::-1])

    def _insert_to(self, root: dict, word: str) -> None:
        node = root

        for char in word:
            if char not in node:
                node[char] = {'#count': 0}
            node[char]['#count'] += 1
            node = node[char]

    def count_match(self, pattern: str) -> int:
        """
        패턴 매칭 개수 (? = 임의의 한 문자)
        예: "fro??" → "frodo", "frost" 등 매칭
        """
        # ?가 앞에 있으면 역순 트라이 사용
        if pattern[0] == '?':
            return self._count(self.reverse_root, pattern[::-1])
        else:
            return self._count(self.root, pattern)

    def _count(self, root: dict, pattern: str) -> int:
        node = root

        for char in pattern:
            if char == '?':
                # ? 이후는 길이만 맞으면 됨
                return node.get('#count', 0)

            if char not in node:
                return 0

            node = node[char]

        return node.get('#count', 0)
```

### 템플릿 5: 길이별 트라이 (카카오 가사 검색 최적화)

```python
class LengthBasedTrie:
    """
    단어 길이별로 분리된 트라이
    가사 검색 문제 최적화
    """
    def __init__(self):
        self.tries = {}  # 길이 → 트라이
        self.reverse_tries = {}

    def insert(self, word: str) -> None:
        length = len(word)

        if length not in self.tries:
            self.tries[length] = {}
            self.reverse_tries[length] = {}

        self._insert_to(self.tries[length], word)
        self._insert_to(self.reverse_tries[length], word[::-1])

    def _insert_to(self, root: dict, word: str) -> None:
        node = root

        for char in word:
            if char not in node:
                node[char] = {'#count': 0}
            node[char]['#count'] += 1
            node = node[char]

    def count_match(self, pattern: str) -> int:
        length = len(pattern)

        if length not in self.tries:
            return 0

        if pattern[0] == '?':
            return self._count(self.reverse_tries[length], pattern[::-1])
        else:
            return self._count(self.tries[length], pattern)

    def _count(self, root: dict, pattern: str) -> int:
        node = root

        for char in pattern:
            if char == '?':
                return node.get('#count', 0)
            if char not in node:
                return 0
            node = node[char]

        return node.get('#count', 0)
```

### 템플릿 6: XOR 최대값 (비트 트라이)

```python
class XORTrie:
    """
    XOR 최대값을 위한 비트 트라이
    """
    def __init__(self, max_bits: int = 32):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num: int) -> None:
        node = self.root

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1

            if bit not in node:
                node[bit] = {}

            node = node[bit]

    def find_max_xor(self, num: int) -> int:
        """num과 XOR했을 때 최대값을 만드는 수와의 XOR 결과"""
        if not self.root:
            return 0

        node = self.root
        result = 0

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit  # 반대 비트

            if opposite in node:
                result |= (1 << i)
                node = node[opposite]
            elif bit in node:
                node = node[bit]
            else:
                break

        return result
```

---

## 예제 문제

### 문제 1: 트라이 구현 (LeetCode 208) - Medium

**문제 설명**
insert, search, startsWith 기능을 가진 트라이를 구현하세요.

**풀이**
```python
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '#' in node

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```

---

### 문제 2: 가사 검색 (카카오 2020) - Hard

**문제 설명**
가사 단어 리스트와 검색 키워드가 주어집니다. 키워드에서 ?는 임의의 한 문자와 매칭됩니다. 각 키워드에 매칭되는 가사 단어 개수를 구하세요.

**입력/출력 예시**
```
입력: words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
      queries = ["fro??", "????o", "fr???", "fro???", "pro?"]
출력: [3, 2, 4, 1, 0]
```

**풀이**
```python
def solution(words: list, queries: list) -> list:
    trie = LengthBasedTrie()

    for word in words:
        trie.insert(word)

    return [trie.count_match(query) for query in queries]
```

---

### 문제 3: 자동완성 (카카오 2018) - Hard

**문제 설명**
학습된 단어 목록에서 각 단어를 입력할 때 필요한 최소 타이핑 수를 구하세요.

**입력/출력 예시**
```
입력: words = ["go", "gone", "guild"]
출력: 7
설명: "go"(2) + "gone"(3) + "guild"(2) = 7
```

**풀이**
```python
def autocomplete_cost(words: list) -> int:
    # 접두사별 개수를 세는 트라이
    trie = CountTrie()

    for word in words:
        trie.insert(word)

    total = 0

    for word in words:
        node = trie.root
        cost = 0

        for char in word:
            cost += 1
            node = node[char]

            # 이 접두사를 가진 단어가 1개면 자동완성
            if node['#count'] == 1:
                break

        total += cost

    return total
```

---

### 문제 4: 접두사 검색 (프로그래머스) - Medium

**문제 설명**
문자열 배열에서 주어진 접두사로 시작하는 문자열 개수를 구하세요.

**풀이**
```python
def count_by_prefix(words: list, prefix: str) -> int:
    trie = CountTrie()

    for word in words:
        trie.insert(word)

    return trie.count_prefix(prefix)
```

---

### 문제 5: Maximum XOR (LeetCode 421) - Medium

**문제 설명**
배열에서 두 수를 골라 XOR한 최대값을 구하세요.

**입력/출력 예시**
```
입력: nums = [3, 10, 5, 25, 2, 8]
출력: 28
설명: 5 XOR 25 = 28
```

**풀이**
```python
def find_maximum_xor(nums: list) -> int:
    trie = XORTrie()

    for num in nums:
        trie.insert(num)

    max_xor = 0

    for num in nums:
        max_xor = max(max_xor, trie.find_max_xor(num))

    return max_xor
```

---

### 문제 6: 전화번호 목록 (프로그래머스 Level 2) - Medium

**문제 설명**
전화번호 목록에서 어떤 번호가 다른 번호의 접두사인 경우가 있는지 확인하세요.

**입력/출력 예시**
```
입력: ["119", "97674223", "1195524421"]
출력: false (119가 1195524421의 접두사)
```

**풀이**
```python
def solution(phone_book: list) -> bool:
    trie = Trie()

    for phone in phone_book:
        trie.insert(phone)

    for phone in phone_book:
        node = trie.root

        for i, char in enumerate(phone):
            node = node.children[char]

            # 중간에 끝나는 단어가 있으면 접두사 관계
            if node.is_end and i < len(phone) - 1:
                return False

    return True
```

---

## Editorial (풀이 전략)

### Step 1: 트라이 사용 시기

| 키워드 | 트라이 적용 |
|--------|-----------|
| 접두사 검색 | ✅ |
| 자동완성 | ✅ |
| 와일드카드 매칭 | ✅ |
| 문자열 집합 | ✅ |
| XOR 최대값 | ✅ (비트 트라이) |

### Step 2: 구현 선택

```
Q1: 접두사 개수도 필요한가?
    Yes → CountTrie

Q2: 와일드카드(?)가 있는가?
    Yes → WildcardTrie

Q3: 뒤에서부터 매칭이 필요한가?
    Yes → Reverse Trie 추가

Q4: 길이가 다양한가?
    Yes → 길이별 분리
```

### Step 3: 최적화 기법

```python
# 1. 딕셔너리 대신 배열 (알파벳만)
self.children = [None] * 26

# 2. 길이별 분리
self.tries = {}  # length → Trie

# 3. 정방향 + 역방향 트라이
self.forward = {}
self.backward = {}
```

---

## 자주 하는 실수

### 1. 단어 끝 표시 누락
```python
# ❌ 끝 표시 없음
def insert(self, word):
    for char in word:
        ...

# ✅ 끝 표시 필수
def insert(self, word):
    for char in word:
        ...
    node.is_end = True  # 또는 node['#'] = True
```

### 2. 접두사 vs 단어 검색 혼동
```python
# ❌ 접두사인데 단어 검색 사용
return node.is_end  # 틀림!

# ✅ 접두사는 노드 존재 여부만 확인
return node is not None
```

### 3. 개수 카운팅 위치
```python
# ❌ 노드 이동 전에 카운팅
node['#count'] += 1
node = node[char]

# ✅ 노드 이동 후에 카운팅
node = node[char]
node['#count'] += 1
```

---

## LeetCode / BOJ / 프로그래머스 추천 문제

| 플랫폼 | # | 문제명 | 난이도 |
|--------|---|-------|-------|
| LeetCode | 208 | Implement Trie | Medium |
| LeetCode | 211 | Design Add and Search Words | Medium |
| LeetCode | 212 | Word Search II | Hard |
| LeetCode | 421 | Maximum XOR | Medium |
| LeetCode | 648 | Replace Words | Medium |
| BOJ | 5052 | 전화번호 목록 | Gold 4 |
| BOJ | 14425 | 문자열 집합 | Silver 4 |
| 프로그래머스 | - | 가사 검색 (카카오 2020) | Level 4 |
| 프로그래머스 | - | 자동완성 (카카오 2018) | Level 4 |
| 프로그래머스 | - | 전화번호 목록 | Level 2 |

---

## 임베딩용 키워드

```
trie, 트라이, prefix tree, 접두사 트리, autocomplete, 자동완성,
wildcard, 와일드카드, 가사 검색, 카카오, prefix search, 접두사 검색,
XOR trie, 비트 트라이, dictionary, 사전, word search
```
