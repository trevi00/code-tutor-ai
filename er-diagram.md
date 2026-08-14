---
origin: extracted
extractor: python_models v0
confidence: 0.9
source_commit: b68c615d9d24
evidence_spans: 63
trust: proposed
---
# ERD (역추출)

## 엔티티

- ENT-collaboration_sessions (`CollaborationSessionModel`, 컬럼 11)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:22 -->
- ENT-session_participants (`SessionParticipantModel`, 컬럼 9)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:69 -->
- ENT-code_changes (`CodeChangeModel`, 컬럼 6)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:103 -->
- ENT-badges (`BadgeModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:30 -->
- ENT-user_badges (`UserBadgeModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:50 -->
- ENT-user_stats (`UserStatsModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:64 -->
- ENT-challenges (`ChallengeModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:95 -->
- ENT-user_challenges (`UserChallengeModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/gamification/infrastructure/models.py:115 -->
- ENT-users (`UserModel`, 컬럼 11)  <!-- ev: backend/src/code_tutor/identity/infrastructure/models.py:13 -->
- ENT-problems (`ProblemModel`, 컬럼 19)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:27 -->
- ENT-test_cases (`TestCaseModel`, 컬럼 7)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:80 -->
- ENT-submissions (`SubmissionModel`, 컬럼 16)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:107 -->
- ENT-daily_stats (`DailyStatsModel`, 컬럼 19)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:26 -->
- ENT-user_interactions (`UserInteractionModel`, 컬럼 13)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:79 -->
- ENT-model_training_logs (`ModelTrainingLogModel`, 컬럼 13)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:129 -->
- ENT-code_quality_analyses (`CodeQualityAnalysisModel`, 컬럼 22)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:169 -->
- ENT-quality_trends (`QualityTrendModel`, 컬럼 17)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:239 -->
- ENT-playgrounds (`PlaygroundModel`, 컬럼 15)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:26 -->
- ENT-code_templates (`CodeTemplateModel`, 컬럼 9)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:77 -->
- ENT-playground_executions (`ExecutionHistoryModel`, 컬럼 11)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:104 -->
- ENT-learning_paths (`LearningPathModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:31 -->
- ENT-path_prerequisites (`PathPrerequisiteModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:67 -->
- ENT-roadmap_modules (`ModuleModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:86 -->
- ENT-roadmap_lessons (`LessonModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:109 -->
- ENT-user_path_progress (`UserPathProgressModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:136 -->
- ENT-user_lesson_progress (`UserLessonProgressModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/roadmap/infrastructure/models.py:156 -->
- ENT-conversations (`ConversationModel`, 컬럼 9)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:14 -->
- ENT-messages (`MessageModel`, 컬럼 7)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:56 -->
- ENT-typing_exercises (`TypingExerciseModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/typing_practice/infrastructure/models.py:32 -->
- ENT-typing_attempts (`TypingAttemptModel`, 컬럼 0)  <!-- ev: backend/src/code_tutor/typing_practice/infrastructure/models.py:53 -->

## 관계

- ENT-collaboration_sessions → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:28 -->
- ENT-collaboration_sessions → ENT-users (FK `host_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:33 -->
- ENT-collaboration_sessions ↔ ENT-session_participants (1:N `participants`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:53 -->
- ENT-collaboration_sessions ↔ ENT-code_changes (1:N `code_changes`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:58 -->
- ENT-session_participants → ENT-collaboration_sessions (FK `session_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:75 -->
- ENT-session_participants → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:80 -->
- ENT-session_participants ↔ ENT-collaboration_sessions (1:1 `session`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:94 -->
- ENT-code_changes → ENT-collaboration_sessions (FK `session_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:109 -->
- ENT-code_changes → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:114 -->
- ENT-code_changes ↔ ENT-collaboration_sessions (1:1 `session`)  <!-- ev: backend/src/code_tutor/collaboration/infrastructure/models.py:124 -->
- ENT-problems ↔ ENT-test_cases (1:N `test_cases`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:68 -->
- ENT-problems ↔ ENT-submissions (1:N `submissions`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:74 -->
- ENT-test_cases → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:86 -->
- ENT-test_cases ↔ ENT-problems (1:1 `problem`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:101 -->
- ENT-submissions → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:113 -->
- ENT-submissions → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:118 -->
- ENT-submissions ↔ ENT-problems (1:1 `problem`)  <!-- ev: backend/src/code_tutor/learning/infrastructure/models.py:153 -->
- ENT-daily_stats → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:36 -->
- ENT-user_interactions → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:89 -->
- ENT-user_interactions → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:94 -->
- ENT-code_quality_analyses → ENT-submissions (FK `submission_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:180 -->
- ENT-code_quality_analyses → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:186 -->
- ENT-code_quality_analyses → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:191 -->
- ENT-quality_trends → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/ml/pipeline/models.py:249 -->
- ENT-playgrounds → ENT-users (FK `owner_id`)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:32 -->
- ENT-playgrounds → ENT-playgrounds (FK `forked_from_id`)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:57 -->
- ENT-playground_executions → ENT-playgrounds (FK `playground_id`)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:110 -->
- ENT-playground_executions → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/playground/infrastructure/models.py:115 -->
- ENT-conversations → ENT-users (FK `user_id`)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:20 -->
- ENT-conversations → ENT-problems (FK `problem_id`)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:25 -->
- ENT-conversations ↔ ENT-messages (1:N `messages`)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:48 -->
- ENT-messages → ENT-conversations (FK `conversation_id`)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:62 -->
- ENT-messages ↔ ENT-conversations (1:1 `conversation`)  <!-- ev: backend/src/code_tutor/tutor/infrastructure/models.py:80 -->

<!-- entries=30 spans=63 -->
