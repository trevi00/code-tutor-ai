# Code Tutor AI - 전체 기능 문서

이 문서는 Code Tutor AI의 모든 기능을 상세히 설명합니다.

---

## 목차

1. [AI 튜터](#ai-튜터)
2. [문제 풀이](#문제-풀이)
3. [패턴 학습](#패턴-학습)
4. [받아쓰기 연습](#받아쓰기-연습)
5. [게이미피케이션](#게이미피케이션)
6. [학습 대시보드](#학습-대시보드)
7. [알고리즘 시각화](#알고리즘-시각화)
8. [비주얼 디버거](#비주얼-디버거)
9. [코드 플레이그라운드](#코드-플레이그라운드)
10. [ML 기반 기능](#ml-기반-기능)

---

## AI 튜터

### 개요
한국어로 알고리즘을 학습할 수 있는 대화형 AI 튜터 시스템입니다.

### 주요 기능

#### 7단계 점진적 힌트 시스템
| 단계 | 내용 |
|------|------|
| 1 | 📋 문제 이해 확인 |
| 2 | 📎 관련 개념 연결 |
| 3 | 💡 핵심 아이디어 유도 |
| 4 | 📝 의사코드 작성 유도 |
| 5 | 🏗️ 구조 설계 가이드 |
| 6 | ✍️ 코드 작성 지원 |
| 7 | 📊 복잡도 분석 및 최적화 |

#### 코드 리뷰
- 시간/공간 복잡도 분석
- O(n²) → O(n) 최적화 제안
- 코드 스타일 개선 제안
- 잠재적 버그 탐지

#### 디버깅 도움
- 오류 원인 분석
- IndexError, RecursionError 등 해결 가이드
- 엣지 케이스 식별

#### 문제 추천
- 약점 기반 맞춤 추천
- 난이도별 단계적 추천
- 학습 진도 기반 추천

### 기술 스택
- **LLM**: Ollama (llama3)
- **RAG**: Sentence Transformers + 45개 패턴 지식베이스

---

## 문제 풀이

### 개요
150개의 알고리즘 문제를 Monaco Editor로 풀 수 있는 환경입니다.

### 문제 구성

#### 난이도 분포
| 난이도 | 개수 |
|--------|------|
| Easy | 44 |
| Medium | 60 |
| Hard | 46 |
| **합계** | **150** |

#### 17개 카테고리
1. **Array** - 배열 조작, 순회
2. **String** - 문자열 처리, 패턴 매칭
3. **Stack** - 스택 기반 문제
4. **Queue** - 큐, 덱 문제
5. **LinkedList** - 연결 리스트
6. **Tree** - 이진 트리, N진 트리
7. **Graph** - 그래프 탐색
8. **BST** - 이진 탐색 트리
9. **DP** - 동적 프로그래밍
10. **Binary Search** - 이진 탐색
11. **Sorting** - 정렬 알고리즘
12. **Greedy** - 탐욕 알고리즘
13. **Segment Tree** - 구간 쿼리
14. **Union-Find** - 분리 집합
15. **Shortest Path** - 최단 경로
16. **MST** - 최소 신장 트리
17. **Number Theory** - 정수론

### 대회 수준 문제 (25개)
- **코드포스 Div2 스타일** (8문제)
  - Lazy Propagation, 상태 압축 DP, 트리 DP
- **ICPC/IOI 스타일** (6문제)
  - Convex Hull Trick, FFT, Max Flow
- **삼성 SW 역량테스트** (6문제)
  - 복잡한 시뮬레이션, 완전탐색 최적화
- **카카오/네이버 코테** (5문제)
  - 문자열 파싱, 좌표 압축, 이벤트 기반 시뮬레이션

### 코드 실행
- **에디터**: Monaco Editor (VS Code 기반)
- **샌드박스**: Docker 컨테이너 (python:3.11-slim)
- **제한**: 타임아웃 5초, 메모리 256MB

---

## 패턴 학습

### 개요
45개의 알고리즘 패턴을 학습할 수 있습니다.

### 패턴 카테고리

#### 기본 패턴 (12개)
- Two Pointers
- Sliding Window
- Fast & Slow Pointers
- Binary Search
- Merge Intervals
- Cyclic Sort
- In-place Reversal
- BFS
- DFS
- Topological Sort
- Binary Tree BFS
- Binary Tree DFS

#### 고급 패턴 (15개)
- Subsets
- Modified Binary Search
- Bitwise XOR
- K-way Merge
- Knapsack (DP)
- Unbounded Knapsack
- Fibonacci Numbers
- Palindromic Subsequence
- Longest Common Substring
- Matrix Chain Multiplication
- 0/1 Knapsack
- Trie
- Union Find
- Segment Tree
- Monotonic Stack

#### Phase 4 신규 패턴 (18개)
- Sparse Table
- Sqrt Decomposition
- Persistent Segment Tree
- Treap
- Link-Cut Tree
- Bellman-Ford
- Floyd-Warshall
- Articulation Bridges
- 2-SAT
- LIS O(n log n)
- Bitmask DP
- Interval DP
- Digit DP
- Rabin-Karp
- Z-Algorithm
- Aho-Corasick
- Manacher
- Suffix Array

### 패턴 정보
각 패턴에 대해 다음 정보를 제공합니다:
- 개념 설명
- 적용 상황
- 템플릿 코드
- 시간/공간 복잡도
- 관련 문제

---

## 받아쓰기 연습

### 개요
알고리즘 템플릿을 타이핑하며 암기하는 연습 시스템입니다.

### 연습 목록

#### 기본 템플릿 (Easy)
| 이름 | 줄 수 | 글자 수 |
|------|-------|---------|
| 슬라이딩 윈도우 | 9 | 219 |
| 이진 탐색 | 14 | 296 |
| DFS 재귀 | 8 | 184 |
| BFS 큐 | 15 | 331 |
| Two Pointer | 11 | 217 |

#### 메서드 모음 (Easy)
| 이름 | 줄 수 | 글자 수 |
|------|-------|---------|
| 문자열 메서드 | 21 | 279 |
| 리스트 메서드 | 23 | 235 |
| 딕셔너리 메서드 | 21 | 278 |

#### 고급 알고리즘 (Hard)
| 이름 | 줄 수 | 글자 수 |
|------|-------|---------|
| 세그먼트 트리 구간합 | 30 | 584 |
| 펜윅 트리 (BIT) | 19 | 473 |

### 기능
- **실시간 WPM 측정**: 분당 타이핑 속도
- **정확률 분석**: 오타 비율 계산
- **마스터 시스템**: 5회 완료 시 마스터 달성
- **XP 보상**: 완료 시 경험치 획득
- **리더보드**: 최고 속도 랭킹

---

## 게이미피케이션

### 개요
학습 동기 부여를 위한 XP, 레벨, 뱃지 시스템입니다.

### XP 시스템

#### XP 획득 활동
| 활동 | XP |
|------|-----|
| 문제 풀이 (Easy) | 10 |
| 문제 풀이 (Medium) | 25 |
| 문제 풀이 (Hard) | 50 |
| 받아쓰기 완료 | 5 |
| 받아쓰기 고정확률 (95%+) | +5 |
| 받아쓰기 마스터 | 20 |
| 연속 학습 보너스 | 10/일 |

### 레벨 시스템
레벨업에 필요한 XP는 점진적으로 증가합니다.

### 뱃지 시스템

#### 뱃지 종류
| 뱃지 | 조건 |
|------|------|
| First Blood | 첫 문제 풀이 |
| Problem Solver | 10문제 풀이 |
| Algorithm Master | 50문제 풀이 |
| Speed Typist | WPM 60 달성 |
| Accuracy King | 정확률 99% 달성 |
| Week Streak | 7일 연속 학습 |
| Month Streak | 30일 연속 학습 |

### 리더보드
- **전체 랭킹**: 누적 XP 기준
- **주간 랭킹**: 이번 주 XP 기준
- **월간 랭킹**: 이번 달 XP 기준

---

## 학습 대시보드

### 개요
학습 진행 상황을 한눈에 확인할 수 있는 대시보드입니다.

### 위젯

#### 통계 카드
- 푼 문제 수
- 전체 성공률
- 연속 학습 일수 (스트릭)
- 총 획득 XP

#### 365일 히트맵
- GitHub 스타일 활동 히트맵
- 일별 문제 풀이 수 시각화

#### 카테고리별 진행률
- 17개 카테고리별 완료율
- 약점 카테고리 식별

#### XP/레벨 진행률
- 현재 레벨 및 다음 레벨까지 진행률
- 최근 XP 획득 내역

---

## 알고리즘 시각화

### 개요
알고리즘 동작을 애니메이션으로 시각화합니다.

### 지원 알고리즘

#### 정렬
- Bubble Sort
- Selection Sort
- Insertion Sort
- Merge Sort
- Quick Sort
- Heap Sort

#### 그래프 탐색
- BFS (너비 우선 탐색)
- DFS (깊이 우선 탐색)
- Dijkstra's Algorithm

#### 트리
- 전위 순회
- 중위 순회
- 후위 순회
- 레벨 순회

### 기능
- 속도 조절
- 단계별 실행
- 배열/그래프 커스텀 입력

---

## 비주얼 디버거

### 개요
코드 실행을 단계별로 추적할 수 있는 디버거입니다.

### 기능
- **단계별 실행**: Step Over, Step Into, Step Out
- **변수 추적**: 현재 스코프의 모든 변수 값 표시
- **콜 스택**: 함수 호출 스택 시각화
- **브레이크포인트**: 특정 라인에 중단점 설정
- **표현식 평가**: 실행 중 표현식 값 확인

---

## 코드 플레이그라운드

### 개요
자유롭게 코드를 실험할 수 있는 환경입니다.

### 기능
- **자유 코딩**: 문제 없이 자유롭게 코드 작성
- **실시간 실행**: 버튼 클릭으로 즉시 실행
- **입력 지원**: stdin 입력 지원
- **출력 확인**: stdout, stderr 분리 표시
- **스니펫 저장**: 자주 사용하는 코드 저장

---

## ML 기반 기능

### 추천 시스템 (NCF)
- **알고리즘**: Neural Collaborative Filtering
- **추천 전략**: Hybrid, Collaborative, Content-based
- **개인화**: 사용자별 풀이 기록 기반
- **캐싱**: Redis (1시간 TTL)

### 학습 분석 (LSTM)
- **성공률 예측**: 7일 후 성공률 예측
- **학습 속도 분석**: 성장 중 / 안정적 / 주의 필요
- **꾸준함 점수**: 0-100 스코어
- **성장률**: 지난 주 대비 변화

### 코드 품질 분석 (CodeBERT)
- **4차원 분석**:
  - 정확성 (Correctness)
  - 효율성 (Efficiency)
  - 가독성 (Readability)
  - 모범사례 (Best Practices)
- **코드 스멜 탐지**
- **등급 시스템**: A/B/C/D/F

---

*최종 업데이트: 2025-01-05*
