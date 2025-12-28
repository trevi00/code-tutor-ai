# Algorithm Patterns Index (알고리즘 패턴 인덱스)

한국 코딩 테스트(카카오, 삼성, 네이버, 프로그래머스) 완벽 대비를 위한 **25개 핵심 알고리즘 패턴** 문서입니다.

## 패턴 목록

### 기본 패턴 (1-15)

| # | 패턴명 | 난이도 | 빈출도 | 시간복잡도 |
|---|--------|--------|--------|-----------|
| 01 | [Two Pointers](./01_two_pointers.md) | Easy~Medium | ⭐⭐⭐⭐⭐ | O(n) |
| 02 | [Sliding Window](./02_sliding_window.md) | Medium | ⭐⭐⭐⭐⭐ | O(n) |
| 03 | [Fast & Slow Pointers](./03_fast_slow_pointers.md) | Easy~Medium | ⭐⭐⭐⭐ | O(n) |
| 04 | [Merge Intervals](./04_merge_intervals.md) | Medium | ⭐⭐⭐⭐ | O(n log n) |
| 05 | [Cyclic Sort](./05_cyclic_sort.md) | Easy~Medium | ⭐⭐⭐ | O(n) |
| 06 | [LinkedList Reversal](./06_linkedlist_reversal.md) | Easy~Medium | ⭐⭐⭐⭐ | O(n) |
| 07 | [BFS](./07_bfs.md) | Easy~Hard | ⭐⭐⭐⭐⭐ | O(V+E) |
| 08 | [DFS](./08_dfs.md) | Easy~Hard | ⭐⭐⭐⭐⭐ | O(V+E) |
| 09 | [Binary Search](./09_binary_search.md) | Easy~Hard | ⭐⭐⭐⭐⭐ | O(log n) |
| 10 | [Top K Elements](./10_top_k_elements.md) | Medium | ⭐⭐⭐⭐ | O(n log k) |
| 11 | [K-way Merge](./11_k_way_merge.md) | Medium~Hard | ⭐⭐⭐ | O(n log k) |
| 12 | [0/1 Knapsack DP](./12_dp_01_knapsack.md) | Medium~Hard | ⭐⭐⭐⭐⭐ | O(n×W) |
| 13 | [Unbounded Knapsack DP](./13_dp_unbounded_knapsack.md) | Medium~Hard | ⭐⭐⭐⭐ | O(n×W) |
| 14 | [Backtracking](./14_backtracking.md) | Medium~Hard | ⭐⭐⭐⭐⭐ | O(k^n) |
| 15 | [Greedy](./15_greedy.md) | Easy~Hard | ⭐⭐⭐⭐ | O(n log n) |

### 고급 패턴 (16-25) - 한국 코테 특화

| # | 패턴명 | 난이도 | 빈출도 | 시간복잡도 | 주요 출제 |
|---|--------|--------|--------|-----------|----------|
| 16 | [Union-Find](./16_union_find.md) | Medium | ⭐⭐⭐⭐ | O(α(n)) | 네트워크, 그룹화 |
| 17 | [Shortest Path](./17_shortest_path.md) | Medium~Hard | ⭐⭐⭐⭐⭐ | O(E log V) | 카카오 (합승 택시) |
| 18 | [Topological Sort](./18_topological_sort.md) | Medium | ⭐⭐⭐⭐ | O(V+E) | 순서 의존성 |
| 19 | [Trie](./19_trie.md) | Medium~Hard | ⭐⭐⭐⭐ | O(L) | 카카오 (가사 검색) |
| 20 | [Segment Tree](./20_segment_tree.md) | Hard | ⭐⭐⭐ | O(log n) | 구간 쿼리 |
| 21 | [Bitmask DP](./21_bitmask_dp.md) | Hard | ⭐⭐⭐⭐ | O(n²×2^n) | 삼성 (외판원) |
| 22 | [Simulation](./22_simulation.md) | Medium~Hard | ⭐⭐⭐⭐⭐ | O(varies) | 삼성 SW역량 |
| 23 | [String Algorithms](./23_string_algorithms.md) | Medium~Hard | ⭐⭐⭐ | O(n+m) | KMP, 문자열 |
| 24 | [MST](./24_mst.md) | Medium | ⭐⭐⭐⭐ | O(E log E) | 네트워크 연결 |
| 25 | [Math & Number Theory](./25_math_number_theory.md) | Easy~Hard | ⭐⭐⭐⭐ | O(√n) | GCD, 조합, 소수 |

---

## 기업별 필수 패턴

### 카카오

| 순위 | 패턴 | 대표 문제 |
|------|------|----------|
| 1 | Trie | 가사 검색, 자동완성 |
| 2 | Shortest Path | 합승 택시 요금 |
| 3 | Binary Search | 징검다리 |
| 4 | Simulation | 프렌즈4블록 |
| 5 | Greedy | 조이스틱, 큰 수 만들기 |

### 삼성 SW역량

| 순위 | 패턴 | 대표 문제 |
|------|------|----------|
| 1 | Simulation | 뱀, 로봇 청소기, 미세먼지 |
| 2 | BFS/DFS | 연구소, 아기 상어 |
| 3 | Bitmask DP | 외판원 순회, 발전소 |
| 4 | Backtracking | 스도쿠, N-Queen |
| 5 | Union-Find | 네트워크 연결 |

### 네이버/라인

| 순위 | 패턴 | 대표 문제 |
|------|------|----------|
| 1 | DP | 배낭, LIS, 구간 DP |
| 2 | Graph | 최단 경로, MST |
| 3 | String | KMP, 해싱 |
| 4 | Math | 조합, 모듈러 연산 |

---

## 학습 순서 추천

### Week 1-2: 기초

```
Two Pointers → Binary Search → Sliding Window → BFS/DFS
```

### Week 3-4: 중급

```
DP (Knapsack) → Backtracking → Greedy → Union-Find
```

### Week 5-6: 고급

```
Shortest Path → Topological Sort → Trie → Simulation
```

### Week 7-8: 심화

```
Segment Tree → Bitmask DP → String Algorithms → MST → Math
```

---

## 각 패턴 문서 구성

모든 패턴 문서는 동일한 구조로 작성되었습니다:

```
# Pattern N: 패턴명

## 개요
- 난이도, 빈출도, 시간/공간 복잡도

## 정의
- 패턴 설명 및 핵심 아이디어

## 템플릿 코드
- 5-10개의 다양한 변형별 템플릿

## 예제 문제
- 5-8개의 실제 문제와 풀이

## Editorial
- 풀이 전략 및 팁

## 자주 하는 실수
- 주의할 점

## LeetCode/BOJ 추천 문제
- 연습 문제 목록

## 임베딩용 키워드
- RAG 검색을 위한 키워드
```

---

## 패턴별 대표 문제

### 기본 패턴

| 패턴 | 대표 문제 | LeetCode # |
|------|----------|-----------|
| Two Pointers | Two Sum II | 167 |
| Sliding Window | Longest Substring Without Repeating | 3 |
| Fast & Slow | Linked List Cycle | 141 |
| Merge Intervals | Merge Intervals | 56 |
| Cyclic Sort | Find Missing Number | 268 |
| LinkedList Reversal | Reverse Linked List | 206 |
| BFS | Binary Tree Level Order | 102 |
| DFS | Number of Islands | 200 |
| Binary Search | Search in Rotated Array | 33 |
| Top K | Kth Largest Element | 215 |
| K-way Merge | Merge K Sorted Lists | 23 |
| 0/1 Knapsack | Partition Equal Subset Sum | 416 |
| Unbounded Knapsack | Coin Change | 322 |
| Backtracking | Permutations | 46 |
| Greedy | Jump Game | 55 |

### 고급 패턴

| 패턴 | 대표 문제 | 플랫폼 |
|------|----------|--------|
| Union-Find | 친구 네트워크 | BOJ 4195 |
| Shortest Path | 합승 택시 요금 | 카카오 2021 |
| Topological Sort | 줄 세우기 | BOJ 2252 |
| Trie | 가사 검색 | 카카오 2020 |
| Segment Tree | 구간 합 구하기 | BOJ 2042 |
| Bitmask DP | 외판원 순회 | BOJ 2098 |
| Simulation | 뱀 | BOJ 3190 |
| String (KMP) | 찾기 | BOJ 1786 |
| MST | 도시 분할 계획 | BOJ 1647 |
| Math | 이항 계수 | BOJ 11401 |

---

## RAG 시스템 연동

이 패턴 문서들은 RAG (Retrieval-Augmented Generation) 시스템에서 사용됩니다:

1. **임베딩**: DistilCodeBERT로 문서 임베딩
2. **인덱싱**: FAISS 벡터 DB에 저장
3. **검색**: 사용자 질문과 유사한 패턴 검색
4. **생성**: LLM이 패턴 기반 답변 생성

자세한 내용은 [RAG_ARCHITECTURE.md](../RAG_ARCHITECTURE.md) 참조.

---

## 통계

| 항목 | 수량 |
|------|------|
| 총 패턴 수 | 25개 |
| 기본 패턴 | 15개 |
| 고급 패턴 (한국 코테 특화) | 10개 |
| 총 템플릿 코드 | 200+ 개 |
| 총 예제 문제 | 150+ 개 |
| 추천 LeetCode 문제 | 200+ 개 |
| 추천 BOJ 문제 | 100+ 개 |
| 프로그래머스 문제 | 50+ 개 |
| 총 문서 크기 | ~500KB |

---

## 난이도별 분류

### Easy (입문)
- Two Pointers, Fast & Slow, Cyclic Sort
- Math (기초)

### Medium (중급)
- Sliding Window, Binary Search, BFS/DFS
- Merge Intervals, LinkedList, Top K
- Union-Find, Topological Sort, Greedy, MST

### Hard (고급)
- DP (Knapsack, Bitmask)
- Backtracking, Simulation
- Shortest Path, Trie, Segment Tree
- String Algorithms
