---
origin: extracted
extractor: python_api v0
confidence: 0.9
source_commit: b68c615d9d24
evidence_spans: 120
trust: proposed
---
# API 명세 (역추출)

경로 합성: include prefix '/api/v1' + 라우터 prefix + 데코레이터 path.
response 는 response_model → 반환 어노테이션 폴백(미상=판정 불가 — 발명 금지).

## backend/src/code_tutor/collaboration/interface/http_routes.py

- EP-001 `POST /api/v1/collaboration/sessions` → SessionDetailResponse — 협업 세션 생성  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:31 -->
- EP-002 `GET /api/v1/collaboration/sessions` → SessionListResponse — 내 세션 목록 조회  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:54 -->
- EP-003 `GET /api/v1/collaboration/sessions/active` → SessionListResponse — 활성 공개 세션 목록  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:74 -->
- EP-004 `GET /api/v1/collaboration/sessions/{session_id}` → SessionDetailResponse — 세션 상세 조회  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:94 -->
- EP-005 `POST /api/v1/collaboration/sessions/{session_id}/join` → SessionDetailResponse — 세션 참여  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:120 -->
- EP-006 `POST /api/v1/collaboration/sessions/{session_id}/leave` → dict — 세션 나가기  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:157 -->
- EP-007 `POST /api/v1/collaboration/sessions/{session_id}/close` → dict — 세션 종료 (호스트 전용)  <!-- ev: backend/src/code_tutor/collaboration/interface/http_routes.py:182 -->
## backend/src/code_tutor/collaboration/interface/websocket_routes.py

- EP-008 `WEBSOCKET /api/v1/collaboration/ws/{session_id}` → 미상  <!-- ev: backend/src/code_tutor/collaboration/interface/websocket_routes.py:40 -->
## backend/src/code_tutor/debugger/interface/routes.py

- EP-009 `POST /api/v1/debugger` → dict — 코드 디버깅 실행  <!-- ev: backend/src/code_tutor/debugger/interface/routes.py:14 -->
- EP-010 `GET /api/v1/debugger/{session_id}` → dict — 디버그 세션 조회  <!-- ev: backend/src/code_tutor/debugger/interface/routes.py:37 -->
- EP-011 `GET /api/v1/debugger/{session_id}/step/{step_number}` → dict — 특정 스텝 조회  <!-- ev: backend/src/code_tutor/debugger/interface/routes.py:55 -->
- EP-012 `GET /api/v1/debugger/{session_id}/summary` → dict — 디버그 세션 요약  <!-- ev: backend/src/code_tutor/debugger/interface/routes.py:83 -->
- EP-013 `POST /api/v1/debugger/quick` → dict — 빠른 디버깅  <!-- ev: backend/src/code_tutor/debugger/interface/routes.py:101 -->
## backend/src/code_tutor/execution/interface/routes.py

- EP-014 `POST /api/v1/execute/run` → dict[str, Any] — 코드 실행  <!-- ev: backend/src/code_tutor/execution/interface/routes.py:25 -->
## backend/src/code_tutor/gamification/interface/routes.py

- EP-015 `GET /api/v1/gamification/overview` → 미상 — 게이미피케이션 전체 현황  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:77 -->
- EP-016 `GET /api/v1/gamification/badges` → 미상 — 전체 뱃지 목록  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:96 -->
- EP-017 `GET /api/v1/gamification/badges/me` → 미상 — 내 뱃지 목록  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:112 -->
- EP-018 `POST /api/v1/gamification/badges/check` → 미상 — 뱃지 획득 확인  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:130 -->
- EP-019 `GET /api/v1/gamification/stats` → 미상 — 내 게이미피케이션 통계  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:149 -->
- EP-020 `POST /api/v1/gamification/xp` → 미상 — XP 추가  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:167 -->
- EP-021 `POST /api/v1/gamification/activity/{action}` → 미상 — 활동 기록  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:188 -->
- EP-022 `GET /api/v1/gamification/leaderboard` → 미상 — 리더보드 조회  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:208 -->
- EP-023 `GET /api/v1/gamification/challenges` → 미상 — 내 챌린지 목록  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:232 -->
- EP-024 `POST /api/v1/gamification/challenges/{challenge_id}/join` → 미상 — 챌린지 참여  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:250 -->
- EP-025 `POST /api/v1/gamification/admin/seed-badges` → 미상 — 뱃지 시드 (관리자)  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:274 -->
- EP-026 `POST /api/v1/gamification/admin/challenges` → 미상 — 챌린지 생성 (관리자)  <!-- ev: backend/src/code_tutor/gamification/interface/routes.py:296 -->
## backend/src/code_tutor/identity/interface/routes.py

- EP-027 `POST /api/v1/auth/register` → UserResponse — 회원가입  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:53 -->
- EP-028 `POST /api/v1/auth/login` → LoginResponse — 로그인  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:83 -->
- EP-029 `POST /api/v1/auth/refresh` → TokenResponse — 토큰 갱신  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:109 -->
- EP-030 `POST /api/v1/auth/logout` → MessageResponse — 로그아웃  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:135 -->
- EP-031 `GET /api/v1/auth/me` → UserResponse — 내 프로필 조회  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:160 -->
- EP-032 `PUT /api/v1/auth/me` → UserResponse — 내 프로필 수정  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:177 -->
- EP-033 `PUT /api/v1/auth/me/password` → MessageResponse — 비밀번호 변경  <!-- ev: backend/src/code_tutor/identity/interface/routes.py:206 -->
## backend/src/code_tutor/learning/interface/routes.py

- EP-034 `GET /api/v1/problems` → ProblemListResponse — List problems  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:112 -->
- EP-035 `GET /api/v1/problems/{problem_id}` → ProblemResponse — Get problem details  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:136 -->
- EP-036 `POST /api/v1/problems` → ProblemResponse — Create a new problem (Admin only)  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:152 -->
- EP-037 `GET /api/v1/problems/{problem_id}/hints` → HintsResponse — Get hints for a problem  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:167 -->
- EP-038 `POST /api/v1/problems/{problem_id}/publish` → ProblemResponse — Publish a problem (Admin only)  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:187 -->
- EP-039 `POST /api/v1/submissions` → SubmissionResponse — Submit code solution  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:205 -->
- EP-040 `GET /api/v1/submissions/{submission_id}` → SubmissionResponse — Get submission details  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:223 -->
- EP-041 `POST /api/v1/submissions/{submission_id}/evaluate` → SubmissionResponse — Evaluate a submission  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:247 -->
- EP-042 `GET /api/v1/submissions` → list[SubmissionSummaryResponse] — List my submissions  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:283 -->
- EP-043 `GET /api/v1/problems/{problem_id}/submissions` → list[SubmissionSummaryResponse] — List my submissions for a problem  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:298 -->
- EP-044 `GET /api/v1/dashboard` → dict[str, Any] — Get user dashboard  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:322 -->
- EP-045 `GET /api/v1/dashboard/prediction` → dict[str, Any] — Get learning predictions  <!-- ev: backend/src/code_tutor/learning/interface/routes.py:343 -->
## backend/src/code_tutor/main.py

- EP-046 `GET /api/health` → dict  <!-- ev: backend/src/code_tutor/main.py:262 -->
- EP-047 `GET /` → dict  <!-- ev: backend/src/code_tutor/main.py:274 -->
- EP-048 `GET /metrics` → 미상  <!-- ev: backend/src/code_tutor/main.py:292 -->
## backend/src/code_tutor/ml/interface/routes.py

- EP-049 `GET /api/v1/problems/recommended` → list[RecommendedProblemResponse] — Get recommended problems for user  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:71 -->
- EP-050 `GET /api/v1/problems/skill-gaps` → list[dict] — Get skill gaps for user  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:128 -->
- EP-051 `GET /api/v1/problems/next-challenge` → dict | None — Get next challenge problem  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:143 -->
- EP-052 `POST /api/v1/submit` → SubmissionResponse — Submit and evaluate code  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:158 -->
- EP-053 `POST /api/v1/code/analyze` → dict[str, Any] — Analyze code with AI  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:255 -->
- EP-054 `POST /api/v1/code/classify` → dict[str, Any] — Classify code quality (Transformer)  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:292 -->
- EP-055 `GET /api/v1/patterns` → dict[str, Any] — List algorithm patterns  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:341 -->
- EP-056 `GET /api/v1/patterns/{pattern_id}` → dict[str, Any] — Get pattern details  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:375 -->
- EP-057 `POST /api/v1/patterns/search` → dict[str, Any] — Search patterns by query  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:409 -->
- EP-058 `GET /api/v1/dashboard/insights` → dict[str, Any] — Get learning insights  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:443 -->
- EP-059 `GET /api/v1/submissions/{submission_id}/quality` → dict[str, Any] — Get code quality analysis for submission  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:479 -->
- EP-060 `GET /api/v1/dashboard/quality` → dict[str, Any] — Get user quality statistics  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:532 -->
- EP-061 `GET /api/v1/dashboard/quality/trends` → dict[str, Any] — Get quality trends over time  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:552 -->
- EP-062 `GET /api/v1/dashboard/quality/recent` → dict[str, Any] — Get recent quality analyses  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:574 -->
- EP-063 `GET /api/v1/dashboard/quality/profile` → dict[str, Any] — Get user quality profile  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:611 -->
- EP-064 `GET /api/v1/dashboard/quality/recommendations` → dict[str, Any] — Get quality-based problem recommendations  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:634 -->
- EP-065 `GET /api/v1/dashboard/quality/suggestions` → dict[str, Any] — Get improvement suggestions  <!-- ev: backend/src/code_tutor/ml/interface/routes.py:661 -->
## backend/src/code_tutor/performance/interface/routes.py

- EP-066 `POST /api/v1/performance` → dict — 전체 성능 분석  <!-- ev: backend/src/code_tutor/performance/interface/routes.py:16 -->
- EP-067 `POST /api/v1/performance/quick` → dict — 빠른 복잡도 분석  <!-- ev: backend/src/code_tutor/performance/interface/routes.py:37 -->
- EP-068 `POST /api/v1/performance/complexity` → dict — 복잡도만 분석  <!-- ev: backend/src/code_tutor/performance/interface/routes.py:55 -->
## backend/src/code_tutor/playground/interface/routes.py

- EP-069 `POST /api/v1/playground` → PlaygroundDetailResponse — 플레이그라운드 생성  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:58 -->
- EP-070 `GET /api/v1/playground/mine` → PlaygroundListResponse — 내 플레이그라운드 목록  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:77 -->
- EP-071 `GET /api/v1/playground/public` → PlaygroundListResponse — 공개 플레이그라운드 목록  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:97 -->
- EP-072 `GET /api/v1/playground/popular` → PlaygroundListResponse — 인기 플레이그라운드 목록  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:116 -->
- EP-073 `GET /api/v1/playground/search` → PlaygroundListResponse — 플레이그라운드 검색  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:133 -->
- EP-074 `GET /api/v1/playground/languages` → LanguagesResponse — 지원 언어 목록  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:152 -->
- EP-075 `GET /api/v1/playground/default-code` → dict — 기본 코드 조회  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:168 -->
- EP-076 `GET /api/v1/playground/share/{share_code}` → PlaygroundDetailResponse — 공유 코드로 조회  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:185 -->
- EP-077 `GET /api/v1/playground/{playground_id}` → PlaygroundDetailResponse — 플레이그라운드 상세 조회  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:209 -->
- EP-078 `PUT /api/v1/playground/{playground_id}` → PlaygroundDetailResponse — 플레이그라운드 수정  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:241 -->
- EP-079 `DELETE /api/v1/playground/{playground_id}` → dict — 플레이그라운드 삭제  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:274 -->
- EP-080 `POST /api/v1/playground/{playground_id}/execute` → ExecutionResponse — 플레이그라운드 코드 실행  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:309 -->
- EP-081 `POST /api/v1/playground/{playground_id}/fork` → PlaygroundDetailResponse — 플레이그라운드 포크  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:342 -->
- EP-082 `POST /api/v1/playground/{playground_id}/regenerate-share-code` → dict — 공유 코드 재생성  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:375 -->
- EP-083 `GET /api/v1/playground/templates/list` → TemplateListResponse — 템플릿 목록 조회  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:410 -->
- EP-084 `GET /api/v1/playground/templates/popular` → TemplateListResponse — 인기 템플릿 목록  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:428 -->
- EP-085 `GET /api/v1/playground/templates/{template_id}` → TemplateResponse — 템플릿 상세 조회  <!-- ev: backend/src/code_tutor/playground/interface/routes.py:445 -->
## backend/src/code_tutor/roadmap/interface/routes.py

- EP-086 `GET /api/v1/roadmap/paths` → LearningPathListResponse — 학습 경로 목록 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:68 -->
- EP-087 `GET /api/v1/roadmap/paths/{path_id}` → LearningPathResponse — 학습 경로 상세 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:86 -->
- EP-088 `GET /api/v1/roadmap/paths/level/{level}` → LearningPathResponse — 레벨별 학습 경로 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:112 -->
- EP-089 `GET /api/v1/roadmap/paths/{path_id}/modules` → list[ModuleResponse] — 경로의 모듈 목록 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:138 -->
- EP-090 `GET /api/v1/roadmap/modules/{module_id}` → ModuleResponse — 모듈 상세 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:160 -->
- EP-091 `GET /api/v1/roadmap/modules/{module_id}/lessons` → list[LessonResponse] — 모듈의 레슨 목록 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:186 -->
- EP-092 `GET /api/v1/roadmap/lessons/{lesson_id}` → LessonResponse — 레슨 상세 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:208 -->
- EP-093 `GET /api/v1/roadmap/progress` → UserProgressResponse — 내 전체 진행 상황 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:237 -->
- EP-094 `GET /api/v1/roadmap/progress/paths/{path_id}` → PathProgressResponse — 특정 경로 진행 상황 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:255 -->
- EP-095 `POST /api/v1/roadmap/paths/{path_id}/start` → PathProgressResponse — 학습 경로 시작  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:281 -->
- EP-096 `POST /api/v1/roadmap/lessons/{lesson_id}/complete` → LessonProgressResponse — 레슨 완료 처리  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:310 -->
- EP-097 `GET /api/v1/roadmap/next-lesson` → LessonResponse | None — 다음 추천 레슨 조회  <!-- ev: backend/src/code_tutor/roadmap/interface/routes.py:340 -->
## backend/src/code_tutor/tutor/interface/routes.py

- EP-098 `POST /api/v1/tutor/chat` → ChatResponse — Send a chat message  <!-- ev: backend/src/code_tutor/tutor/interface/routes.py:43 -->
- EP-099 `GET /api/v1/tutor/conversations` → list[ConversationSummaryResponse] — List my conversations  <!-- ev: backend/src/code_tutor/tutor/interface/routes.py:66 -->
- EP-100 `GET /api/v1/tutor/conversations/{conversation_id}` → ConversationResponse — Get conversation details  <!-- ev: backend/src/code_tutor/tutor/interface/routes.py:81 -->
- EP-101 `POST /api/v1/tutor/conversations/{conversation_id}/close` → ConversationResponse — Close a conversation  <!-- ev: backend/src/code_tutor/tutor/interface/routes.py:98 -->
- EP-102 `POST /api/v1/tutor/review` → CodeReviewResponse — Get AI code review  <!-- ev: backend/src/code_tutor/tutor/interface/routes.py:115 -->
## backend/src/code_tutor/typing_practice/interface/routes.py

- EP-103 `GET /api/v1/typing-practice/exercises` → TypingExerciseListResponse — 타이핑 연습 목록 조회  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:59 -->
- EP-104 `GET /api/v1/typing-practice/exercises/{exercise_id}` → TypingExerciseResponse — 타이핑 연습 상세 조회  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:89 -->
- EP-105 `POST /api/v1/typing-practice/exercises` → TypingExerciseResponse — 타이핑 연습 생성 (관리자)  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:118 -->
- EP-106 `GET /api/v1/typing-practice/exercises/{exercise_id}/progress` → UserProgressResponse — 연습 진행 상황 조회  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:139 -->
- EP-107 `POST /api/v1/typing-practice/attempts` → TypingAttemptResponse — 타이핑 시도 시작  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:171 -->
- EP-108 `POST /api/v1/typing-practice/attempts/{attempt_id}/complete` → TypingAttemptResponse — 타이핑 시도 완료  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:194 -->
- EP-109 `GET /api/v1/typing-practice/stats` → UserTypingStatsResponse — 내 타이핑 통계 조회  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:253 -->
- EP-110 `GET /api/v1/typing-practice/mastered` → list[str] — 마스터한 연습 목록  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:271 -->
- EP-111 `GET /api/v1/typing-practice/leaderboard` → LeaderboardResponse — 리더보드 조회  <!-- ev: backend/src/code_tutor/typing_practice/interface/routes.py:291 -->
## backend/src/code_tutor/visualization/interface/routes.py

- EP-112 `GET /api/v1/visualization/algorithms` → 미상 — 알고리즘 목록 조회  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:20 -->
- EP-113 `GET /api/v1/visualization/algorithms/{algorithm_id}` → 미상 — 알고리즘 정보 조회  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:45 -->
- EP-114 `GET /api/v1/visualization/random-array` → 미상 — 랜덤 배열 생성  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:80 -->
- EP-115 `POST /api/v1/visualization/sorting` → 미상 — 정렬 알고리즘 시각화 생성  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:107 -->
- EP-116 `POST /api/v1/visualization/searching` → 미상 — 탐색 알고리즘 시각화 생성  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:129 -->
- EP-117 `POST /api/v1/visualization/graph` → 미상 — 그래프 알고리즘 시각화 생성  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:152 -->
- EP-118 `GET /api/v1/visualization/sorting/{algorithm_id}` → 미상 — 정렬 시각화 빠른 조회  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:173 -->
- EP-119 `GET /api/v1/visualization/searching/{algorithm_id}` → 미상 — 탐색 시각화 빠른 조회  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:202 -->
- EP-120 `GET /api/v1/visualization/graph/{algorithm_id}` → 미상 — 그래프 시각화 빠른 조회  <!-- ev: backend/src/code_tutor/visualization/interface/routes.py:233 -->

<!-- entries=120 spans=120 -->
