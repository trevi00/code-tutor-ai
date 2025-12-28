# Code Tutor AI - Documentation

## AI 기반 Python 알고리즘 학습 플랫폼

---

## 문서 구조

```
docs/
├── README.md                 # 이 파일 (문서 인덱스)
├── PROJECT_OVERVIEW.md       # 프로젝트 개요, 목표, 로드맵
├── DDD_ARCHITECTURE.md       # 도메인 설계, 헥사고날 아키텍처
├── API_SPECIFICATION.md      # REST API 명세
├── UI_DESIGN.md              # 화면 설계, 와이어프레임, 디자인 시스템
├── RAG_ARCHITECTURE.md       # RAG 시스템 설계
├── TESTING_STRATEGY.md       # 테스트 전략
├── SECURITY.md               # 보안 설계 (OWASP)
├── DEPLOYMENT.md             # 배포, CI/CD, 모니터링
└── patterns/                 # 25개 알고리즘 패턴
    ├── README.md
    ├── 01_two_pointers.md
    ├── ...
    └── 25_math_number_theory.md
```

---

## 문서 가이드

### 1. 프로젝트 이해하기

| 문서 | 대상 | 내용 |
|------|------|------|
| [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) | 전체 | 비전, 기능, 기술스택, 로드맵 |

### 2. 아키텍처 설계

| 문서 | 대상 | 내용 |
|------|------|------|
| [DDD_ARCHITECTURE.md](./DDD_ARCHITECTURE.md) | Backend | Bounded Context, Aggregate, Hexagonal |
| [API_SPECIFICATION.md](./API_SPECIFICATION.md) | Full-stack | REST API, WebSocket, 에러 코드 |
| [UI_DESIGN.md](./UI_DESIGN.md) | Frontend | 와이어프레임, 컴포넌트, 디자인 시스템 |
| [RAG_ARCHITECTURE.md](./RAG_ARCHITECTURE.md) | AI | LangChain, FAISS, 프롬프트 설계 |

### 3. 개발 가이드

| 문서 | 대상 | 내용 |
|------|------|------|
| [TESTING_STRATEGY.md](./TESTING_STRATEGY.md) | 개발자 | Unit, Integration, E2E 테스트 |
| [SECURITY.md](./SECURITY.md) | 개발자 | OWASP Top 10, 샌드박스 보안 |

### 4. 운영 가이드

| 문서 | 대상 | 내용 |
|------|------|------|
| [DEPLOYMENT.md](./DEPLOYMENT.md) | DevOps | Docker, CI/CD, 모니터링, 백업 |

### 5. 학습 콘텐츠

| 문서 | 대상 | 내용 |
|------|------|------|
| [patterns/README.md](./patterns/README.md) | 학습자 | 25개 알고리즘 패턴 개요 |

---

## 빠른 시작

### 개발 환경 설정

```bash
# 1. Repository Clone
git clone https://github.com/user/code-tutor-ai.git
cd code-tutor-ai

# 2. Backend 설정
cd backend
cp .env.example .env
uv sync

# 3. Frontend 설정
cd ../frontend
npm install

# 4. Docker 서비스 시작
docker compose up -d db redis

# 5. 개발 서버 실행
# Terminal 1: Backend
cd backend && uv run uvicorn src.code_tutor.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

### 프로덕션 배포

```bash
# 1. 환경 변수 설정
cp .env.production.example .env

# 2. Docker Compose 실행
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 3. 상태 확인
docker compose ps
curl http://localhost/health
```

---

## 기술 스택 요약

| 영역 | 기술 |
|------|------|
| **Backend** | FastAPI, Python 3.11, SQLAlchemy, Redis |
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS |
| **AI/ML** | EEVE-Korean-2.8B, PyTorch, LangChain, FAISS |
| **Database** | PostgreSQL 16, Redis 7 |
| **Infra** | Docker, Nginx, GitHub Actions |
| **Monitoring** | Prometheus, Grafana, Loki |

---

## 설계 원칙

### DDD (Domain-Driven Design)
- **4개 Bounded Context**: Identity, Learning, AI Tutor, Code Execution
- **높은 응집도**: Context별 도메인 모델 집중
- **낮은 결합도**: Port/Adapter 패턴, Domain Event

### 헥사고날 아키텍처
- **Domain Core**: 비즈니스 로직 (외부 의존성 없음)
- **Application Layer**: Use Case, DTO
- **Adapters**: REST, Database, LLM, Docker

### 보안
- **OWASP Top 10** 대응
- **샌드박스 격리**: Docker + seccomp + 리소스 제한
- **JWT 인증**: Access Token (15분) + Refresh Token (7일)

---

## 문서 버전

| 문서 | 버전 | 최종 수정 |
|------|------|-----------|
| PROJECT_OVERVIEW | 3.0 | 2025-12-26 |
| DDD_ARCHITECTURE | 1.0 | 2025-12-26 |
| API_SPECIFICATION | 1.0 | 2025-12-26 |
| UI_DESIGN | 1.0 | 2025-12-26 |
| RAG_ARCHITECTURE | 2.0 | 2025-12-26 |
| TESTING_STRATEGY | 1.0 | 2025-12-26 |
| SECURITY | 1.0 | 2025-12-26 |
| DEPLOYMENT | 1.0 | 2025-12-26 |
| patterns/ | 2.0 | 2025-12-26 |

---

## 기여 가이드

### 문서 수정 시
1. 해당 문서의 버전 번호 증가
2. 최종 수정일 업데이트
3. 변경 내용이 다른 문서에 영향을 주는지 확인

### 코드 작성 시
1. [TESTING_STRATEGY.md](./TESTING_STRATEGY.md) 테스트 가이드 준수
2. [SECURITY.md](./SECURITY.md) 보안 체크리스트 확인
3. [DDD_ARCHITECTURE.md](./DDD_ARCHITECTURE.md) 패키지 구조 준수
