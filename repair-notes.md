# Repair Notes

## 2026-08-14 — 29 failures: UUID column bound with str (`'str' object has no attribute 'hex'`)

### Failing tests
- `tests/test_typing_repository.py::TestSQLAlchemyTypingAttemptRepository` (13 tests)
- `tests/test_roadmap_repository.py::TestSQLAlchemyUserProgressRepository` (14 tests)
- `tests/test_roadmap.py::TestRoadmapProgressAPI::test_get_progress_authenticated`, `test_get_next_lesson_authenticated` (2 tests, same cause via the service layer)

### Root cause (source bug — no tests were changed)
Commits `1714ad7` (typing_practice) and `3d8a0e8` (roadmap) deliberately changed the
`user_id` columns from `String(36)` to `UUID(as_uuid=True)` to fix a real foreign-key
type mismatch against `users.id` (which is a native `Uuid` column). However, the
repository implementations were never updated to match: they still converted every
`user_id` to `str(...)` before binding it (both in INSERT model constructors and in
`WHERE` comparisons). SQLAlchemy's `Uuid` bind processor on the SQLite test dialect
calls `value.hex`, which fails on a `str` with
`StatementError: (builtins.AttributeError) 'str' object has no attribute 'hex'`.

The rest of the codebase (playground, gamification, tutor, learning, collaboration)
binds real `uuid.UUID` objects against UUID columns; typing_practice and roadmap were
the only modules left binding strings for `user_id`. Fix mirrors that convention.

### Fix (repository layer only; models and tests untouched)
Bind `uuid.UUID` objects for `user_id` (the only UUID-typed columns in these two
modules); all other id columns in these modules remain `String(36)` and keep their
`str(...)` conversion.

- `backend/src/code_tutor/typing_practice/infrastructure/repository.py`
  (`SQLAlchemyTypingAttemptRepository`)
  - `save`: `user_id=attempt.user_id` (was `str(attempt.user_id)`)
  - `list_by_user_and_exercise`, `list_by_user`, `get_user_stats`,
    `get_mastered_exercise_ids`: `TypingAttemptModel.user_id == user_id`
    (was `== str(user_id)`)
  - `_to_entity` already handled both str and UUID; no change needed there.

- `backend/src/code_tutor/roadmap/infrastructure/repository.py`
  (`SQLAlchemyUserProgressRepository`)
  - `save_path_progress` / `save_lesson_progress`: bind `progress.user_id` directly
    in both the existence query and the model constructor (was `str(...)`)
  - `get_path_progress`, `get_all_path_progress`, `get_lesson_progress`,
    `get_module_lessons_progress`, `get_path_lessons_progress`,
    `get_completed_lesson_count`, `get_next_lesson`: compare `user_id` directly
    (was `== str(user_id)`)
  - `_path_progress_to_entity` / `_lesson_progress_to_entity`: `model.user_id` now
    comes back as a `UUID` object; replaced the unconditional `UUID(model.user_id)`
    with a str/UUID-tolerant conversion (same guard style as the typing repository).

### Result
- Previously failing files: `tests/test_typing_repository.py`,
  `tests/test_roadmap_repository.py`, `tests/test_roadmap.py` — 78 passed.
- Full suite: `uv run pytest tests -q --no-header` → **1659 passed, 0 failed**
  (was 1630 pass / 29 fail; no regressions).

## 2026-08-14 — Circular dependencies removed: execution <-> learning, learning <-> ml

### Structural moves (dependency-direction fix, no package merging)
1. **`SubmissionEvaluator` moved from execution to learning**
   - `backend/src/code_tutor/execution/application/services.py` → new module
     `backend/src/code_tutor/learning/application/submission_evaluator.py`.
   - The evaluator operates on learning domain entities (Problem, Submission,
     TestCase, SubmissionStatus, TestResult) and only *uses* execution's sandbox
     (`ExecutionRequest`, `DockerSandbox`/`MockSandbox`), so it belongs in
     learning. execution no longer imports learning at all; learning → execution
     remains (already-existing, allowed direction).
   - Updated importers: `learning/interface/routes.py`,
     `tests/test_execution_service.py`.
2. **ML-flavored endpoints moved out of the learning router into a new
   `backend/src/code_tutor/ml/interface/routes.py`** (new `ml.interface` package):
   `/problems/recommended`, `/problems/skill-gaps`, `/problems/next-challenge`,
   `/submit`, `/code/analyze`, `/code/classify`, `/patterns`,
   `/patterns/{pattern_id}`, `/patterns/search`, `/dashboard/insights`,
   `/submissions/{submission_id}/quality`, `/dashboard/quality`(+`/trends`,
   `/recent`, `/profile`, `/recommendations`, `/suggestions`) — i.e. every
   handler that imported `code_tutor.ml` (top-level or lazy), plus their ml DI
   providers and the `_get_recommendation_reason_kr` helper.
   - `main.py` mounts the new router with the same `/api/v1` prefix immediately
     after the learning router, so all URL paths and relative route-matching
     order are unchanged (verified by dumping the app's route table: 121 routes,
     identical paths).
   - The ml router imports learning DTOs/providers (ml → learning), which is the
     correct direction: ml consumes learning's data model; learning no longer
     knows ml exists.
   - Updated test import: `tests/test_learning_services.py`
     (`_get_recommendation_reason_kr`).

### Why the direction is now correct
- execution is a low-level sandboxing component: it should not know the learning
  domain. learning → execution only.
- ml builds analyses/recommendations *on top of* learning data:
  ml → learning only. New edges introduced: ml → gamification, ml → identity
  (from the moved `/submit` and auth deps) — both are sinks w.r.t. ml, no cycle.

### Result
- `grep -rn "code_tutor.learning" backend/src/code_tutor/execution/` → empty;
  `grep -rn "code_tutor\.ml" backend/src/code_tutor/learning/` → empty.
- Re-extracted `component-diagram.md` (reverse.py): `execution → learning` and
  `learning → ml` edges gone; 32 edges total; DFS cycle check → acyclic.
- Full suite: `uv run pytest tests -q --no-header` → **1659 passed, 0 failed**
  (baseline preserved, no tests weakened or skipped).

## 2026-08-14 — frontend e2e 전량 수리 (auth.setup 실패로 41개 전체 차단 → 41/41 통과)

### 근본 원인 (5건)
1. **백엔드: aware datetime → naive TIMESTAMP 컬럼 바인딩 실패 (INSERT 500)** — 도메인 계층은 `datetime.now(UTC)`(aware)를 쓰는데 모든 모델 컬럼은 `TIMESTAMP WITHOUT TIME ZONE`(naive). asyncpg가 aware 값을 거부해 회원가입(users INSERT)·제출(submissions INSERT)이 500. auth.setup의 가입 단계가 여기서 죽어 전체 스위트가 차단됨.
2. **시드 부재** — 새 compose DB에 문제 카탈로그 0건. `code_tutor.scripts.seed_problems`는 존재하지만 어디서도 실행되지 않았고, e2e 스펙들은 이전 DB에서 복사한 **하드코딩 문제 UUID**를 참조(시드는 uuid4 랜덤이라 재현 불가 구조).
3. **로그인 rate limit(5/min, burst 3, IP 키) vs 병렬 UI 로그인 14회** — 7개 스펙이 각자 UI 로그인을 수행해 대부분 429. storageState 공유용 setup 프로젝트가 이미 있는데 활용 안 됨.
4. **프론트 라우트 계약 위반** — 대시보드 최근 제출/품질 추천/제출 기록의 링크가 `/problems/{id}`로 향하는데 해당 라우트가 없음(`/problems/:id/solve`만 존재) → catch-all이 홈(`/`)으로 튕김.
5. **셀렉터 드리프트** — UI 리디자인 후 스펙이 옛 DOM을 참조: `h1("대시보드")`(현재는 인사말 h1), `table tbody tr`(현재는 카드/리스트 링크), 광역 `text=로그아웃`/`a[href*="/problems"]`(헤더 드롭다운의 CSS-hidden 중복 요소에 매칭), 41문제 페이지네이션으로 '두 수의 합'이 1페이지에 없음. 부수 발견: 문제 검색 UI가 보내는 `search` 파라미터를 백엔드가 무시(기능 자체가 no-op), ml data_aggregator의 `max(boolean)`은 PostgreSQL에 없는 함수(제출 흐름 500).

### 수정
- `backend/src/code_tutor/shared/infrastructure/database.py`: `NaiveUTCDateTime` TypeDecorator 추가(바인딩 시 aware→naive UTC 변환) + `Base.type_annotation_map` 등록. 9개 `infrastructure/models.py`의 `DateTime`을 전부 이 타입으로 교체(ORM·Core insert 모두 커버). `identity/infrastructure/repository.py`에도 경계 변환 헬퍼.
- `backend/.../ml/pipeline/data_aggregator.py`: `func.max(bool)` → `func.max(cast(..., Integer))`.
- `backend/.../learning`(routes/dto/services/repository 4파일): 문제 목록 `search`(title ILIKE) 파라미터 구현 — 프론트 검색창 계약 복원.
- `frontend/e2e/global-setup.ts`(신규) + `playwright.config.ts` globalSetup 등록: 고정 자격증명 테스트 유저 API 등록(멱등) + 문제 0건이면 컨테이너 내 시드 스크립트 실행(멱등) — 랜덤 값 없는 결정론적 시드.
- `frontend/e2e/helpers.ts`(신규): 제목→문제 ID API 조회. `problem-solve.spec.ts`·`problem-types.spec.ts`의 하드코딩 UUID 7건을 제목 조회로 교체.
- chat/dashboard/screenshots/problem-solve/problem-types 스펙: UI 로그인 제거, auth.setup의 storageState 재사용(로그인 UI 로그인은 auth.setup·manual-test에만 남김 → 회당 2회로 rate limit 내).
- `user.authenticated.spec.ts` 로그아웃: refresh 엔드포인트로 전용 토큰 발급 후 로그아웃(로그아웃이 access 토큰 jti를 블랙리스트하므로 공유 storageState 토큰 보호), 배너 스코프 셀렉터로 유저 메뉴 열고 실제 로그아웃 수행(기존 조건부 스킵보다 강화).
- `dashboard.authenticated.spec.ts`: `main` 스코프 문제 링크로 교체. `manual-test.spec.ts`: 검색창으로 '두 수의 합' 필터 후 확인, 드롭다운 내비는 hover로 열고 클릭(클릭 토글은 hover-open과 레이스). `dashboard.spec.ts`·`screenshots.spec.ts`: 현행 DOM 셀렉터로 갱신.
- `QualityRecommendations.tsx`·`DashboardPage.tsx`·`SubmissionsPage.tsx`: 문제 링크를 `/problems/{id}/solve`로 수정(존재하는 라우트로).

### 검증
`cd frontend && npm run test:e2e` 연속 2회 완주 — **41 passed, 0 failed, exit 0** (테스트 스킵/완화/타임아웃 증가 없음).

## 2026-08-14 — e2e-front 재실패 (iter=11, EX-28 연구-수리 tier): manual-test 로그인 429

### 재현
`cd frontend && npm run test:e2e` → **41 passed, 1 failed, exit 1**.
실패: `manual-test.spec.ts:29` waitForURL(/problems/) 타임아웃. error-context.md 스냅샷에
로그인 폼 위 배너 "Request failed with status code 429" — 로그인 rate limit 초과가 원인.

### 근본 원인
직전 수리 세션이 패턴 링크 hover 레이스를 진단하려고 만든 **`e2e/__diag.spec.ts`(untracked
디버깅 스펙)가 스위트에 방치**되어 있었다. 이 스펙은 자체 UI 로그인을 수행하므로 회당 UI 로그인이
auth.setup + code-review + manual-test = 3회(= 로그인 리미터 burst 정확히 한도, 검증된 기준선)에서
**4회로 증가**. 리미터는 5/min·burst 3·IP 키 인메모리 토큰버킷(`rate_limiter.py`)이고 네 로그인이
모두 chromium 페이즈 시작 ~15초 안에 몰리므로(리필 ≈1.25토큰) 마지막 로그인(manual-test)이
결정론적으로 429를 받는다. 실행 로그 타이밍과 일치: __diag 7.8s·code-review 11.4s 통과,
manual-test 12.0s 실패.

### 수정
`frontend/e2e/__diag.spec.ts` 삭제 — 제품 테스트가 아닌 진단 산물(25회 boundingBox 콘솔 덤프
루프, 의미 있는 단언 없음)이며, 그 진단의 결론은 이미 manual-test.spec.ts의 hover→click 주석으로
반영 완료. 검증된 41-테스트 기준선과 로그인 예산(3회)을 복원한다. 테스트 완화·스킵 아님 —
기준선 스위트는 그대로다.

### 잔여 리스크
UI 로그인 3회 = burst 3 정확히 한도. 여유 0이므로 로그인하는 스펙이 하나라도 추가되면 재발한다.
근본 여유 확보(code-review 스펙의 storageState 전환)는 별도 단위로 남김.
