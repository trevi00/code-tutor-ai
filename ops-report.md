# ops-report — docker compose 전체 스택 기동 (AC-030)

- 일시: 2026-08-14
- 수행: `docker compose up -d --build` (backend·frontend 이미지 빌드 포함)

## RESULT: PASS — 전체 스택 기동, 헬스·메트릭 관측 확인

| 컨테이너 | 이미지 | 상태 | 포트 |
|---|---|---|---|
| codetutor-db | postgres:15-alpine | Up (healthy) | 5433→5432 |
| codetutor-redis | redis:7-alpine | Up (healthy) | 6380→6379 |
| codetutor-backend | code-tutor-ai-backend | Up | 8000→8000 |
| codetutor-frontend | code-tutor-ai-frontend | Up | 3000→80 |

## 관측

- `GET http://localhost:8000/api/health` → **200** (AC-030)
- `GET http://localhost:8000/metrics` → **200** (REQ-033)
- 마이그레이션: schema-sync 단계에서 `alembic upgrade head` 적용 완료(리비전 89089f1187c9), `alembic check` diff 0.

## 이슈·비고

- backend 컨테이너는 compose 내부 네트워크의 `db:5432`를 사용하고, 호스트 로컬 개발은 `backend/.env`의 `localhost:5433`을 사용한다 — 두 경로 모두 동작 확인.
- monitoring 프로필(prometheus·grafana)은 이번 기동 범위에 포함하지 않음(`--profile monitoring` 미지정). 메트릭 엔드포인트 자체는 노출 확인.
- 알려진 결함 0건 아님을 전제로 한 관측 결과이며, 상세 회귀는 파이프라인 게이트(테스트 스위트)가 판정한다.
