# Code Tutor AI

[![CI](https://github.com/trevi00/code-tutor-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/trevi00/code-tutor-ai/actions/workflows/ci.yml)
[![PR Check](https://github.com/trevi00/code-tutor-ai/actions/workflows/pr-check.yml/badge.svg)](https://github.com/trevi00/code-tutor-ai/actions/workflows/pr-check.yml)
[![E2E Tests](https://github.com/trevi00/code-tutor-ai/actions/workflows/e2e-full.yml/badge.svg)](https://github.com/trevi00/code-tutor-ai/actions/workflows/e2e-full.yml)

> **AI 기반 Python 알고리즘 & 자료구조 학습 플랫폼**

한국어 AI 튜터와 함께하는 맞춤형 코딩 교육 웹 애플리케이션

---

## 프로젝트 상태

**Phase 4 완료** - AI 튜터 고도화 및 대회 수준 문제 추가

| 기능 | 상태 |
|------|------|
| AI 튜터 채팅 (한국어) | ✅ 완료 |
| **AI 튜터 7단계 힌트 시스템** | ✅ 완료 |
| **코드 리뷰 (복잡도 분석 + 최적화 제안)** | ✅ 완료 |
| **디버깅 도움 시스템** | ✅ 완료 |
| 문제 풀이 (Monaco Editor) | ✅ 완료 |
| **대회 수준 문제 (150문제, 17카테고리)** | ✅ 완료 |
| 코드 실행 (Docker 샌드박스) | ✅ 완료 |
| 대시보드 (통계/스트릭/히트맵) | ✅ 완료 |
| **패턴 학습 (45개 알고리즘 패턴)** | ✅ 완료 |
| ML 추천 시스템 (NCF) | ✅ 완료 |
| AI 학습 분석 (LSTM) | ✅ 완료 |
| 코드 품질 분석 (CodeBERT) | ✅ 완료 |
| 품질 기반 추천 | ✅ 완료 |
| **받아쓰기 연습 (고급 템플릿)** | ✅ 완료 |
| **게이미피케이션 (XP/뱃지/레벨)** | ✅ 완료 |
| **알고리즘 시각화** | ✅ 완료 |
| **비주얼 디버거** | ✅ 완료 |
| **코드 플레이그라운드** | ✅ 완료 |
| 전체 UI 한글화 | ✅ 완료 |
| E2E 테스트 (41개) | ✅ 통과 |

---

## 주요 기능

### AI 튜터 (Phase 4 강화)
- 한국어 자연어로 알고리즘 개념 질문
- **7단계 점진적 힌트 시스템**
  1. 📋 문제 이해 확인
  2. 📎 관련 개념 연결
  3. 💡 핵심 아이디어 유도
  4. 📝 의사코드 작성 유도
  5. 🏗️ 구조 설계 가이드
  6. ✍️ 코드 작성 지원
  7. 📊 복잡도 분석 및 최적화
- **실시간 코드 리뷰** - 복잡도 분석, O(n²)→O(n) 최적화 제안
- **디버깅 도움** - 오류 원인 분석 및 해결 가이드
- **맞춤형 문제 추천** - 약점 기반 난이도별 추천

### 문제 풀이 (Phase 4 확장)
- **150개 알고리즘 문제** (Easy 44, Medium 60, Hard 46)
- **17개 카테고리**
  - 기본: Array, String, Stack, Queue, LinkedList
  - 트리/그래프: Tree, Graph, BST
  - 알고리즘: DP, Binary Search, Sorting, Greedy
  - 고급: Segment Tree, Union-Find, Shortest Path, MST, Number Theory
- **대회 수준 문제 (25개 신규)**
  - 코드포스 Div2 스타일 (Lazy Propagation, 상태 압축 DP)
  - ICPC/IOI 스타일 (Convex Hull Trick, FFT)
  - 삼성 SW 역량테스트 스타일 (복잡한 시뮬레이션)
  - 카카오/네이버 코테 스타일 (문자열 파싱, 좌표 압축)
- Monaco Editor (VS Code와 동일한 에디터)
- Docker 샌드박스에서 안전한 코드 실행

### 패턴 학습 (Phase 4 확장)
- **45개 알고리즘 패턴** (기존 24개 + 신규 18개)
- **신규 고급 패턴**
  - 자료구조: Sparse Table, Sqrt Decomposition, Persistent Segment Tree
  - 그래프: Bellman-Ford, Floyd-Warshall, Articulation Bridges, 2-SAT
  - DP: 0/1 Knapsack, LIS O(n log n), Bitmask DP, Interval DP, Digit DP
  - 문자열: Rabin-Karp, Z-Algorithm, Aho-Corasick, Manacher
- 패턴별 상세 설명 및 템플릿 코드

### 받아쓰기 연습 (Phase 4 신규)
- **10개 타이핑 연습** (기본 + 고급)
- 기본 템플릿: 슬라이딩 윈도우, 이진 탐색, DFS/BFS, Two Pointer
- **고급 템플릿**: 세그먼트 트리, 펜윅 트리
- 실시간 WPM 측정 및 정확률 분석
- 5회 완료 시 마스터 달성
- XP 보상 연동

### 게이미피케이션 (Phase 4 신규)
- **XP 시스템** - 활동별 경험치 획득
- **레벨 시스템** - 레벨업 진행률 표시
- **뱃지 시스템** - 업적 달성 시 뱃지 획득
- **리더보드** - 전체/주간/월간 랭킹

### 알고리즘 시각화
- 정렬 알고리즘 애니메이션
- 그래프 탐색 시각화 (BFS/DFS)
- 트리 순회 시각화

### 비주얼 디버거
- 단계별 코드 실행
- 변수 상태 추적
- 콜 스택 시각화

### 코드 플레이그라운드
- 자유로운 코드 실험 환경
- 실시간 실행 결과 확인

### 학습 대시보드
- 문제 풀이 통계 (푼 문제, 성공률)
- 연속 학습 스트릭
- 365일 히트맵
- 카테고리별 진행률
- XP/레벨 진행 현황

### ML 추천 시스템 (Phase 2)
- **NCF (Neural Collaborative Filtering)** 기반 개인화 추천
- 스킬 갭 분석 - 보완이 필요한 카테고리 식별
- 다음 도전 문제 추천 - 적정 난이도 문제 제안

### AI 학습 분석 (Phase 2)
- **LSTM 기반 성공률 예측** - 7일 후 성공률 예측
- 학습 속도 분석 (성장 중/안정적/주의 필요)
- 꾸준함 점수 (0-100)

### 코드 품질 분석 (Phase 3)
- **CodeBERT 기반 다차원 품질 분석**
  - 정확성 / 효율성 / 가독성 / 모범사례
- **코드 스멜 탐지** - 긴 함수, 깊은 중첩, 매직 넘버 등
- **등급 시스템** - A/B/C/D/F 등급 부여

---

## 기술 스택

### Backend
- **Framework**: FastAPI + Python 3.11
- **Database**: PostgreSQL 14 + SQLite (개발) + Redis
- **AI/LLM**: Ollama (llama3)
- **ML**: PyTorch (NCF, LSTM), NumPy, scikit-learn
- **NLP**: Sentence Transformers (RAG)
- **코드 실행**: Docker 샌드박스

### Frontend
- **Framework**: React 18 + TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Editor**: Monaco Editor
- **State**: Zustand + TanStack Query

### Testing
- **E2E**: Playwright (41개 테스트)
- **API**: pytest

---

## 빠른 시작

### 1. 저장소 클론
\`\`\`bash
git clone https://github.com/trevi00/code-tutor-ai.git
cd code-tutor-ai
\`\`\`

### 2. Backend 실행
\`\`\`bash
cd backend
uv sync
uv run uvicorn src.code_tutor.main:app --reload --port 8000
\`\`\`

### 3. Ollama 실행
\`\`\`bash
ollama run llama3
\`\`\`

### 4. Frontend 실행
\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`

### 5. 접속
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

---

## 개발 로드맵

| Phase | 상태 | 내용 |
|-------|------|------|
| **Phase 1** | ✅ 완료 | MVP (AI 채팅, 코드 리뷰, 문제 풀이) |
| **Phase 1.5** | ✅ 완료 | 패턴 학습, 한글화 |
| **Phase 2** | ✅ 완료 | 추천 시스템 (NCF), 학습 분석 (LSTM) |
| **Phase 3** | ✅ 완료 | 코드 품질 분석 (CodeBERT), 품질 기반 추천 |
| **Phase 4** | ✅ 완료 | AI 튜터 고도화, 대회 문제, 게이미피케이션 |
| **Phase 5** | 예정 | 멀티 언어 지원, 실시간 대결 모드 |

---

## 문서

| 문서 | 설명 |
|------|------|
| [PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md) | 프로젝트 개요 |
| [API_SPECIFICATION.md](./docs/API_SPECIFICATION.md) | API 명세 |
| [DDD_ARCHITECTURE.md](./docs/DDD_ARCHITECTURE.md) | 아키텍처 설계 |
| [SECURITY.md](./docs/SECURITY.md) | 보안 설계 |
| [AI_TUTOR_GUIDE.md](./docs/AI_TUTOR_GUIDE.md) | AI 튜터 사용 가이드 |
| [FEATURES.md](./docs/FEATURES.md) | 전체 기능 문서 |

---

## 라이선스

MIT License

---

**프로젝트가 마음에 드셨다면 Star를 눌러주세요!**
