# Code Tutor AI

> **AI 기반 Python 알고리즘 & 자료구조 학습 플랫폼**

한국어 LLM과 5개의 딥러닝 모델을 활용한 맞춤형 코딩 교육 웹 애플리케이션

---

## 주요 특징

### 🎓 AI 튜터 학습 (RAG 기반)
- **LeetCode 스타일 패턴 학습**: 15개 알고리즘 패턴 체계적 교육
- 한국어 자연어로 알고리즘 개념 질문
- 실시간 코드 리뷰 & 피드백
- 대화형 문제 풀이 가이드
- **환각 방지 RAG 시스템**: 지식 베이스 기반 정확한 답변

### 🤖 딥러닝 기반 기능 (5개 모델)
- **알고리즘 패턴 추천 (RAG)**: Two Pointers, Sliding Window 등 15개 패턴 학습
- **맞춤형 문제 추천**: 사용자 수준 분석 후 최적 문제 자동 선정
- **코드 유사도 검색**: 트랜스포머로 유사 풀이 찾기 & 표절 감지
- **학습 성과 예측**: LSTM으로 다음 주 성공률 및 취약점 분석
- **코드 품질 분석**: AI 기반 복잡도/가독성/버그 위험도 평가

### 💻 실전 기능
- 브라우저 코드 실행 (Docker 샌드박스)
- 30+ 알고리즘 문제 (난이도별)
- 학습 진도 대시보드

---

## 기술 스택

### AI/ML (5개 딥러닝 모델)
```
EEVE-Korean-2.8B    → AI 튜터 대화 (LLM)
DistilCodeBERT      → 코드 임베딩 (트랜스포머, 66M params)
NCF                 → 맞춤형 추천 (100K params)
LSTM                → 학습 예측 (500K params)
CodeBERT Classifier → 품질 분석 (50M params)
```

**메모리 최적화**: 4GB VRAM에서 동작 (동적 로딩 + 양자화)

### Backend
- FastAPI + Python 3.11
- PostgreSQL 14 + Redis
- **RAG 시스템**: LangChain + FAISS (벡터 검색)
- Docker (코드 샌드박스)

### Frontend
- React 18 + TypeScript
- Tailwind CSS + Monaco Editor
- Zustand + React Query

---

## 빠른 시작

```bash
# 1. 클론
git clone https://github.com/your-username/code-tutor-ai.git
cd code-tutor-ai

# 2. 환경 변수 설정
cp backend/.env.example backend/.env
# .env 파일 수정 (DATABASE_URL, JWT_SECRET_KEY 등)

# 3. Docker Compose 실행
docker-compose up -d

# 4. 접속
# http://localhost:3000 (Frontend)
# http://localhost:8000/docs (API Docs)
```

---

## 프로젝트 구조

```
code-tutor-ai/
├── backend/          # FastAPI + 5개 딥러닝 모델 + RAG 시스템
├── frontend/         # React + TypeScript
├── models/           # AI 모델 저장소
├── data/             # 알고리즘 패턴 지식 베이스 (15개 패턴)
├── docs/
│   ├── PRD.md                # 📋 통합 프로젝트 문서
│   └── RAG_ARCHITECTURE.md   # 🧠 RAG 시스템 아키텍처 (LeetCode 패턴)
└── docker-compose.yml
```

---

## 📋 문서

**[통합 PRD 문서](./docs/PRD.md)** - 모든 내용이 하나의 문서에!

- 프로젝트 개요 & 핵심 기능
- 5개 딥러닝 모델 상세 스펙
- 시스템 아키텍처 & API 명세
- 12주 개발 로드맵
- 요구사항 명세 & 성공 지표

**[RAG 시스템 아키텍처](./docs/RAG_ARCHITECTURE.md)** - LeetCode 스타일 패턴 학습!

- 15개 알고리즘 패턴 (Two Pointers, Sliding Window, DP 등)
- RAG (Retrieval-Augmented Generation) 구조
- LangChain + FAISS 통합
- 패턴 지식 베이스 설계
- 템플릿 코드 & Editorial

---

## 개발 로드맵

| Phase | 기간 | 주요 내용 |
|-------|------|-----------|
| **Phase 1** | Week 1-4 | MVP (AI 채팅, 코드 리뷰, 5개 문제) |
| **Phase 2** | Week 5-8 | 코드 실행, 30개 문제, 대시보드 |
| **Phase 3** | Week 9-12 | 딥러닝 통합 (추천, 임베딩, 품질 분석) |
| **Phase 4** | Week 13+ | LSTM 예측, 확장 기능 |

**현재 상태**: 문서화 완료 ✅ → MVP 개발 준비 중

---

## 딥러닝 모델 성능 목표

| 모델 | 성능 목표 |
|------|-----------|
| AI 튜터 | 응답 < 5초 |
| 코드 임베딩 | 임베딩 < 100ms, 검색 < 10ms |
| 문제 추천 | 추론 < 20ms, Precision@5 > 0.7 |
| 학습 예측 | 추론 < 5ms, MAE < 0.15 |
| 품질 분석 | 분석 < 200ms |

---

## 하드웨어 요구사항

### 개발 환경
- **GPU**: NVIDIA RTX 4050 (4GB VRAM)
- **CPU**: Intel i7-13700H (14코어)
- **RAM**: 16GB

### 제약사항
- ✅ 4GB VRAM에서 모든 모델 실행 가능
- ✅ 클라우드 GPU 불필요 (동적 로딩 + 양자화)
- ✅ CPU 오프로딩으로 경량 모델 병렬 처리

---

## 라이선스

- **프로젝트**: MIT License
- **EEVE-Korean-2.8B**: Apache 2.0
- **DistilCodeBERT**: Apache 2.0

---

## 기여

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

자세한 내용은 [통합 PRD 문서](./docs/PRD.md)를 참조하세요.

---

**⭐ 프로젝트가 마음에 드셨다면 Star를 눌러주세요!**
