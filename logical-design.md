---
origin: extracted
extractor: python_models v0
confidence: 0.9
source_commit: b68c615d9d24
evidence_spans: 30
trust: proposed
---
# 논리 설계 (역추출)

스키마 소스=models.py (alembic 리비전 1개 — 동기화 미보장, 실측).

## TBL-collaboration_sessions  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:22 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| problem_id | `UUID | None` | FK→problems.id, index, nullable |
| host_id | `UUID` | FK→users.id, index, not null |
| title | `str` | not null |
| status | `str` | index |
| code_content | `str` | — |
| language | `str` | — |
| version | `int` | — |
| max_participants | `int` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-session_participants  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:69 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| session_id | `UUID` | FK→collaboration_sessions.id, index, not null |
| user_id | `UUID` | FK→users.id, index, not null |
| username | `str` | not null |
| cursor_position | `dict | None` | nullable |
| selection_range | `dict | None` | nullable |
| is_active | `bool` | — |
| color | `str` | — |
| joined_at | `datetime` | — |

## TBL-code_changes  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:103 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| session_id | `UUID` | FK→collaboration_sessions.id, index, not null |
| user_id | `UUID` | FK→users.id, not null |
| operation | `dict` | not null |
| version | `int` | not null |
| timestamp | `datetime` | — |

## TBL-badges  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:30 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-user_badges  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:50 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-user_stats  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:64 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-challenges  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:95 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-user_challenges  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:115 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-users  <!-- ev: backend/src/code_tutor/identity/infrastructure/models.py:13 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| email | `str` | unique, index, not null |
| username | `str` | unique, index, not null |
| hashed_password | `str` | not null |
| role | `UserRole` | not null |
| is_active | `bool` | not null |
| is_verified | `bool` | not null |
| last_login_at | `datetime | None` | nullable |
| bio | `str | None` | nullable |
| created_at | `datetime` | not null |
| updated_at | `datetime` | not null |

## TBL-problems  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:27 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| title | `str` | not null |
| description | `str` | not null |
| difficulty | `Difficulty` | not null |
| category | `Category` | index, not null |
| constraints | `str` | — |
| hints | `list[str]` | — |
| solution_template | `str` | — |
| reference_solution | `str` | — |
| time_limit_ms | `int` | — |
| memory_limit_mb | `int` | — |
| is_published | `bool` | index |
| pattern_ids | `list[str]` | — |
| pattern_explanation | `str` | — |
| approach_hint | `str` | — |
| time_complexity_hint | `str` | — |
| space_complexity_hint | `str` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-test_cases  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:80 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| problem_id | `UUID` | FK→problems.id, index, not null |
| input_data | `str` | not null |
| expected_output | `str` | not null |
| is_sample | `bool` | — |
| order | `int` | — |
| created_at | `datetime` | — |

## TBL-submissions  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:107 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| user_id | `UUID` | FK→users.id, index, not null |
| problem_id | `UUID` | FK→problems.id, index, not null |
| code | `str` | not null |
| language | `str` | — |
| status | `SubmissionStatus` | index |
| test_results | `dict | None` | nullable |
| total_tests | `int` | — |
| passed_tests | `int` | — |
| execution_time_ms | `float` | — |
| memory_usage_mb | `float` | — |
| error_message | `str | None` | nullable |
| submitted_at | `datetime` | index |
| evaluated_at | `datetime | None` | nullable |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-daily_stats  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:26 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| user_id | `UUID` | FK→users.id, index, not null |
| stats_date | `date` | index, not null |
| problems_attempted | `int` | — |
| problems_solved | `int` | — |
| total_submissions | `int` | — |
| success_rate | `float` | — |
| avg_time_to_solve_ms | `float` | — |
| avg_memory_usage_mb | `float` | — |
| easy_solved | `int` | — |
| medium_solved | `int` | — |
| hard_solved | `int` | — |
| categories_attempted | `int` | — |
| category_breakdown | `dict` | — |
| streak_days | `int` | — |
| study_minutes | `int` | — |
| is_active_day | `bool` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-user_interactions  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:79 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| user_id | `UUID` | FK→users.id, index, not null |
| problem_id | `UUID` | FK→problems.id, index, not null |
| is_solved | `bool` | — |
| attempt_count | `int` | — |
| best_execution_time_ms | `float` | nullable |
| best_memory_usage_mb | `float` | nullable |
| first_attempt_at | `datetime` | nullable |
| solved_at | `datetime` | nullable |
| time_to_solve_seconds | `int` | nullable |
| interaction_score | `float` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-model_training_logs  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:129 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| model_type | `str` | index, not null |
| model_version | `str` | not null |
| model_path | `str` | nullable |
| training_started_at | `datetime` | not null |
| training_completed_at | `datetime` | nullable |
| training_samples | `int` | — |
| epochs_completed | `int` | — |
| metrics | `dict` | — |
| status | `str` | — |
| error_message | `str` | nullable |
| is_active | `bool` | — |
| created_at | `datetime` | — |

## TBL-code_quality_analyses  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:169 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| submission_id | `UUID` | FK→submissions.id, unique, index, not null |
| user_id | `UUID` | FK→users.id, index, not null |
| problem_id | `UUID` | FK→problems.id, index, not null |
| correctness_score | `int` | — |
| efficiency_score | `int` | — |
| readability_score | `int` | — |
| best_practices_score | `int` | — |
| overall_score | `int` | — |
| overall_grade | `str` | — |
| code_smells | `list` | — |
| code_smells_count | `int` | — |
| cyclomatic_complexity | `int` | — |
| cognitive_complexity | `int` | — |
| max_nesting_depth | `int` | — |
| lines_of_code | `int` | — |
| detected_patterns | `list` | — |
| suggestions | `list` | — |
| suggestions_count | `int` | — |
| language | `str` | — |
| analyzer_version | `str` | — |
| analyzed_at | `datetime` | — |

## TBL-quality_trends  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:239 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| user_id | `UUID` | FK→users.id, index, not null |
| trend_date | `date` | index, not null |
| avg_overall_score | `float` | — |
| avg_correctness | `float` | — |
| avg_efficiency | `float` | — |
| avg_readability | `float` | — |
| avg_best_practices | `float` | — |
| avg_cyclomatic | `float` | — |
| avg_cognitive | `float` | — |
| submissions_analyzed | `int` | — |
| total_smells | `int` | — |
| total_suggestions | `int` | — |
| improved_count | `int` | — |
| grade_distribution | `dict` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-playgrounds  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:26 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| owner_id | `UUID` | FK→users.id, index, not null |
| title | `str` | not null |
| description | `str` | — |
| code | `str` | not null |
| language | `str` | — |
| visibility | `str` | index |
| share_code | `str` | unique, index, not null |
| stdin | `str` | — |
| is_forked | `bool` | — |
| forked_from_id | `UUID | None` | FK→playgrounds.id, nullable |
| run_count | `int` | — |
| fork_count | `int` | — |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-code_templates  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:77 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| title | `str` | not null |
| description | `str` | — |
| code | `str` | not null |
| language | `str` | index |
| category | `str` | index |
| tags | `str` | — |
| usage_count | `int` | — |
| created_at | `datetime` | — |

## TBL-playground_executions  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:104 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| playground_id | `UUID` | FK→playgrounds.id, index, not null |
| user_id | `UUID | None` | FK→users.id, index, nullable |
| code | `str` | not null |
| stdin | `str` | — |
| stdout | `str` | — |
| stderr | `str` | — |
| exit_code | `int` | — |
| execution_time_ms | `float` | — |
| is_success | `bool` | — |
| executed_at | `datetime` | — |

## TBL-learning_paths  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:31 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-path_prerequisites  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:67 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-roadmap_modules  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:86 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-roadmap_lessons  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:109 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-user_path_progress  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:136 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-user_lesson_progress  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:156 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-conversations  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:14 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| user_id | `UUID` | FK→users.id, index, not null |
| problem_id | `UUID | None` | FK→problems.id, index, nullable |
| conversation_type | `ConversationType` | — |
| title | `str` | — |
| total_tokens | `int` | — |
| is_active | `bool` | index |
| created_at | `datetime` | — |
| updated_at | `datetime` | — |

## TBL-messages  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:56 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|
| id | `UUID` | PK |
| conversation_id | `UUID` | FK→conversations.id, index, not null |
| role | `MessageRole` | not null |
| content | `str` | not null |
| code_context | `dict | None` | nullable |
| tokens_used | `int` | — |
| created_at | `datetime` | — |

## TBL-typing_exercises  <!-- ev: backend/src/code_tutor/typing_practice/infrastructure/models.py:32 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

## TBL-typing_attempts  <!-- ev: backend/src/code_tutor/typing_practice/infrastructure/models.py:53 -->

| 컬럼 | 타입 | 제약 |
|---|---|---|

<!-- entries=30 spans=30 -->
