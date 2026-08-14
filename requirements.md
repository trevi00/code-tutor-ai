---
origin: authored
authored_by: user + claude (R2 의도 복원 — D-014)
status: draft (human 게이트 미판정)
source_commit: 370caeaf1141
---

# code-tutor-ai 요구사항 (복원)

> 브라운필드 역설계 온보딩의 의도 복원 문서. 기능 요구는 **코드 실물이 근거**(evidence — 발명 아님),
> 목적·비기능 수위는 **사용자 확정**(2026-08-14 대화). 미확정 결정은 §4에 정직하게 남긴다.

## 1. 목적 (사용자 확정)

- REQ-001 **하네스 실전 검증 substrate**: 이 프로젝트는 하네스(설계→개발→검증 자율 루프)의 실전 검증 대상이다. 모든 변경은 하네스 파이프라인·게이트를 경유한다.
- REQ-002 **프로덕션 승격 가능성 검증**: 게이트 전량 PASS(빌드·테스트·e2e·보안·관측)를 프로덕션급 판정 기준으로 삼고, 실서비스 배포 가능한 상태까지 끌어올릴 수 있는지 검증한다.
- REQ-003 **운영 형태 = 셀프호스팅 데모** (사용자 확정 2026-08-14): docker-compose 단일 호스트 기동이 기준. 공개 서비스 수위의 보안·스케일 요구(WAF·멀티테넌시·오토스케일)는 Non-Goal.
- REQ-004 **AI 기능은 스텁 검증** (사용자 확정): tutor·ml 의 실 LLM 연동은 결정론 스텁/모킹으로 대체 — 게이트는 계약(요청/응답 형태)을 검증하고 모델 품질은 검증하지 않는다. 실 연동은 후속 결정.

## 2. 기능 요구 (코드 근거 — 12 도메인 슬라이스)

**검증 순서 (사용자 확정): P1 = 핵심 축(REQ-010·011·012 — 가입→문제→제출→채점 루프)부터 프로덕션급 관통, 나머지는 P2 확장.**

- REQ-010 [P1] 회원가입·로그인·토큰 인증·프로필 관리 (evidence: `backend/src/code_tutor/identity/`, EP-027~033)
- REQ-011 [P1] 알고리즘 문제 열람·코드 제출·채점·대시보드 (evidence: `learning/`, EP 29종 — 최대 도메인)
- REQ-012 [P1] 제출 코드의 격리 실행 (evidence: `execution/`, EP-014)
- REQ-013 AI 튜터 대화 (evidence: `tutor/`, conversations/messages 테이블)
- REQ-014 게이미피케이션 — 배지·스탯·챌린지 (evidence: `gamification/`, 테이블 5종)
- REQ-015 학습 로드맵 — 경로·모듈·레슨·진행도 (evidence: `roadmap/`, 테이블 6종)
- REQ-016 코드 플레이그라운드 — 템플릿·실행 이력 (evidence: `playground/`)
- REQ-017 실시간 협업 세션 — HTTP + WebSocket (evidence: `collaboration/`, EP-001~008)
- REQ-018 디버거 보조 (evidence: `debugger/`)
- REQ-019 타이핑 연습 (evidence: `typing_practice/`)
- REQ-020 코드 실행 시각화 (evidence: `visualization/`)
- REQ-021 성능 분석 (evidence: `performance/`)
- REQ-022 ML 보조 — 추천·RAG·품질 분석·예측 (evidence: `ml/` — 도메인 슬라이스 아님, 크로스컷)
- REQ-023 웹 프론트엔드 — React 19, 27페이지, Monaco 에디터 (evidence: `frontend/src/pages/`)

## 3. 비기능 요구 (프로덕션급 판정 기준 — REQ-002 의 구체화)

- REQ-030 **테스트 게이트**: 백엔드 단위/통합 테스트 전량 PASS + 프론트 e2e(Playwright) PASS 를 하네스 게이트로 강제 (현황: tests 36파일 — 통과 여부 미검증)
- REQ-031 **아키텍처 무결성**: 컴포넌트 순환 의존 0 (현황: **위반 — execution↔learning 순환 검출됨**, 역추출 실측)
- REQ-032 **스키마 정합**: alembic 마이그레이션과 models.py 동기화 (현황: 리비전 1개 vs 모델 30 — 드리프트 추정)
- REQ-033 **배포·관측**: docker-compose 기동 + /api/health·/metrics 정상 (evidence: `docker-compose.yml`, `monitoring/`)

## 4. 미확정 (사용자 결정 대기 — 발명 금지)

- ~~OPEN-1~~ → REQ-003 확정 (셀프호스팅 데모)
- ~~OPEN-2~~ → REQ-004 확정 (AI 스텁)
- ~~OPEN-3~~ → §2 P1/P2 확정 (핵심 축 우선)
- OPEN-4 KPI: 코드에 근거 없음 — 데모 성격상 미정의 유지 (Non-Goals 는 REQ-003 에 흡수)
