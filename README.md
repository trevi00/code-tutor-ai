# Code Tutor AI

> **AI 기반 Python 알고리즘 & 자료구조 학습 플랫폼**

한국어 AI 튜터와 함께하는 맞춤형 코딩 교육 웹 애플리케이션

---

## 프로젝트 상태

**MVP 완료** - 모든 핵심 기능 구현 및 테스트 통과

| 기능 | 상태 |
|------|------|
| AI 튜터 채팅 (한국어) | ✅ 완료 |
| 코드 리뷰 (복잡도 분석) | ✅ 완료 |
| 문제 풀이 (Monaco Editor) | ✅ 완료 |
| 코드 실행 (Docker 샌드박스) | ✅ 완료 |
| 대시보드 (통계/스트릭/히트맵) | ✅ 완료 |
| E2E 테스트 (34개) | ✅ 통과 |

---

## 주요 기능

### AI 튜터
- 한국어 자연어로 알고리즘 개념 질문
- 실시간 코드 리뷰 & 복잡도 분석
- 대화형 문제 풀이 가이드
- 힌트 제공 (직접 답을 주지 않고 사고 유도)

### 문제 풀이
- 11개 알고리즘 문제 (Easy/Medium/Hard)
- 8개 카테고리 (Array, Stack, LinkedList, Tree, Graph, DP, Binary Search, String)
- Monaco Editor (VS Code와 동일한 에디터)
- Docker 샌드박스에서 안전한 코드 실행

### 학습 대시보드
- 문제 풀이 통계 (푼 문제, 성공률)
- 연속 학습 스트릭
- 365일 히트맵
- 카테고리별 진행률

---

## 기술 스택

### Backend
- **Framework**: FastAPI + Python 3.11
- **Database**: PostgreSQL 14 + Redis
- **AI/LLM**: Ollama (llama3)
- **코드 실행**: Docker 샌드박스 (python:3.11-slim)
- **아키텍처**: DDD + 헥사고날 아키텍처

### Frontend
- **Framework**: React 18 + TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Editor**: Monaco Editor
- **State**: Zustand + TanStack Query

### Testing
- **E2E**: Playwright (34개 테스트)
- **API**: pytest

---

## 빠른 시작

### 사전 요구사항
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- Ollama (llama3 모델)

### 1. 저장소 클론
```bash
git clone https://github.com/trevi00/code-tutor-ai.git
cd code-tutor-ai
```

### 2. 환경 변수 설정
```bash
# Backend
cp backend/.env.example backend/.env
# 필요한 값 수정 (DATABASE_URL, JWT_SECRET_KEY 등)
```

### 3. 인프라 실행
```bash
# PostgreSQL + Redis 실행
docker-compose up -d
```

### 4. Backend 실행
```bash
cd backend
uv sync  # 또는 pip install -r requirements.txt
uv run uvicorn src.code_tutor.main:app --reload --port 8000
```

### 5. Ollama 실행
```bash
ollama run llama3
```

### 6. Frontend 실행
```bash
cd frontend
npm install
npm run dev
```

### 7. 접속
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

---

## API 엔드포인트

### 인증
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/auth/register` | 회원가입 |
| POST | `/api/v1/auth/login` | 로그인 |
| POST | `/api/v1/auth/refresh` | 토큰 갱신 |
| GET | `/api/v1/auth/me` | 현재 사용자 |

### 문제
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/problems` | 문제 목록 |
| GET | `/api/v1/problems/{id}` | 문제 상세 |
| GET | `/api/v1/problems/{id}/hints` | 힌트 조회 |

### 제출
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/submissions` | 코드 제출 |
| GET | `/api/v1/submissions` | 제출 목록 |
| GET | `/api/v1/submissions/{id}` | 제출 상세 |

### AI 튜터
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/tutor/chat` | AI 채팅 |
| GET | `/api/v1/tutor/conversations` | 대화 기록 |
| POST | `/api/v1/tutor/review` | 코드 리뷰 |

### 코드 실행
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/execute/run` | 코드 실행 |

### 대시보드
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/dashboard` | 전체 대시보드 |

---

## 프로젝트 구조

```
code-tutor-ai/
├── backend/
│   └── src/code_tutor/
│       ├── auth/           # 인증 도메인
│       ├── problem/        # 문제 도메인
│       ├── submission/     # 제출 도메인
│       ├── tutor/          # AI 튜터 도메인
│       ├── execution/      # 코드 실행 도메인
│       ├── dashboard/      # 대시보드 도메인
│       └── shared/         # 공유 모듈
├── frontend/
│   └── src/
│       ├── pages/          # 페이지 컴포넌트
│       ├── shared/         # 공유 컴포넌트/훅/API
│       └── e2e/            # E2E 테스트
├── docs/                   # 프로젝트 문서
└── docker-compose.yml
```

---

## 테스트 실행

### E2E 테스트
```bash
cd frontend
npx playwright test
```

### 테스트 결과
```
Running 34 tests using 10 workers
34 passed (1.0m)
```

---

## 스크린샷

### 문제 풀이 페이지
- Monaco Editor로 코드 작성
- 실행 및 제출 버튼
- AI 도움 버튼으로 튜터 연결

### AI 튜터 채팅
- 한국어로 알고리즘 질문
- 코드 리뷰 요청
- 마크다운 형식 응답

### 대시보드
- 푼 문제 / 총 제출 / 성공률
- 연속 스트릭
- 카테고리별 진행률

---

## 환경 변수

### Backend (.env)
```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/codetutor

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Docker Sandbox
SANDBOX_TIMEOUT_SECONDS=5
SANDBOX_MEMORY_LIMIT_MB=256
```

---

## 문서

| 문서 | 설명 |
|------|------|
| [PROJECT_OVERVIEW.md](./docs/PROJECT_OVERVIEW.md) | 프로젝트 개요 |
| [API_SPECIFICATION.md](./docs/API_SPECIFICATION.md) | API 명세 |
| [DDD_ARCHITECTURE.md](./docs/DDD_ARCHITECTURE.md) | 아키텍처 설계 |
| [SECURITY.md](./docs/SECURITY.md) | 보안 설계 |

---

## 개발 로드맵

| Phase | 상태 | 내용 |
|-------|------|------|
| **Phase 1** | ✅ 완료 | MVP (AI 채팅, 코드 리뷰, 문제 풀이) |
| **Phase 2** | 예정 | 추천 시스템 (NCF), 학습 분석 |
| **Phase 3** | 예정 | 코드 분석 (CodeBERT), 성과 예측 (LSTM) |

---

## 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 라이선스

MIT License

---

## 저자

- **trevi00** - [GitHub](https://github.com/trevi00)

---

**프로젝트가 마음에 드셨다면 Star를 눌러주세요!**
