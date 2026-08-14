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
