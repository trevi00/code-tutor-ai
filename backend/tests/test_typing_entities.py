"""Tests for Typing Practice Domain Entities and Value Objects."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from code_tutor.typing_practice.domain.entities import (
    TypingExercise,
    TypingAttempt,
    UserExerciseProgress,
    utc_now,
)
from code_tutor.typing_practice.domain.value_objects import (
    ExerciseCategory,
    AttemptStatus,
    Difficulty,
)


class TestExerciseCategory:
    """Tests for ExerciseCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert ExerciseCategory.TEMPLATE.value == "template"
        assert ExerciseCategory.METHOD.value == "method"
        assert ExerciseCategory.ALGORITHM.value == "algorithm"
        assert ExerciseCategory.PATTERN.value == "pattern"

    def test_category_from_string(self):
        """Test creating category from string."""
        assert ExerciseCategory("template") == ExerciseCategory.TEMPLATE
        assert ExerciseCategory("method") == ExerciseCategory.METHOD
        assert ExerciseCategory("algorithm") == ExerciseCategory.ALGORITHM
        assert ExerciseCategory("pattern") == ExerciseCategory.PATTERN


class TestAttemptStatus:
    """Tests for AttemptStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert AttemptStatus.IN_PROGRESS.value == "in_progress"
        assert AttemptStatus.COMPLETED.value == "completed"
        assert AttemptStatus.ABANDONED.value == "abandoned"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert AttemptStatus("in_progress") == AttemptStatus.IN_PROGRESS
        assert AttemptStatus("completed") == AttemptStatus.COMPLETED
        assert AttemptStatus("abandoned") == AttemptStatus.ABANDONED


class TestDifficulty:
    """Tests for Difficulty enum."""

    def test_difficulty_values(self):
        """Test all difficulty values exist."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_difficulty_from_string(self):
        """Test creating difficulty from string."""
        assert Difficulty("easy") == Difficulty.EASY
        assert Difficulty("medium") == Difficulty.MEDIUM
        assert Difficulty("hard") == Difficulty.HARD


class TestTypingExercise:
    """Tests for TypingExercise entity."""

    def test_exercise_creation(self):
        """Test creating an exercise directly."""
        exercise_id = uuid4()
        exercise = TypingExercise(
            id=exercise_id,
            title="Two Pointers Template",
            source_code="def two_pointers(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        pass",
            language="python",
            category=ExerciseCategory.TEMPLATE,
            difficulty=Difficulty.MEDIUM,
            description="Practice two pointers pattern",
            required_completions=5,
        )
        assert exercise.id == exercise_id
        assert exercise.title == "Two Pointers Template"
        assert exercise.language == "python"
        assert exercise.category == ExerciseCategory.TEMPLATE
        assert exercise.difficulty == Difficulty.MEDIUM
        assert exercise.required_completions == 5

    def test_exercise_create_factory(self):
        """Test exercise factory method."""
        exercise = TypingExercise.create(
            title="Binary Search",
            source_code="def binary_search(arr, target):\n    pass",
            language="python",
            category=ExerciseCategory.ALGORITHM,
            difficulty=Difficulty.EASY,
            description="Binary search template",
            required_completions=3,
        )
        assert exercise.id is not None
        assert exercise.title == "Binary Search"
        assert exercise.category == ExerciseCategory.ALGORITHM
        assert exercise.difficulty == Difficulty.EASY
        assert exercise.required_completions == 3
        assert exercise.is_published is True
        assert exercise.created_at is not None
        assert exercise.updated_at is not None

    def test_exercise_default_values(self):
        """Test exercise default values via factory."""
        exercise = TypingExercise.create(
            title="Simple Print",
            source_code="print('Hello')",
        )
        assert exercise.language == "python"
        assert exercise.category == ExerciseCategory.TEMPLATE
        assert exercise.difficulty == Difficulty.EASY
        assert exercise.description == ""
        assert exercise.required_completions == 5

    def test_exercise_update(self):
        """Test updating exercise properties."""
        exercise = TypingExercise.create(
            title="Original Title",
            source_code="print('original')",
        )
        original_updated_at = exercise.updated_at

        exercise.update(
            title="Updated Title",
            source_code="print('updated')",
            description="New description",
            difficulty=Difficulty.HARD,
        )

        assert exercise.title == "Updated Title"
        assert exercise.source_code == "print('updated')"
        assert exercise.description == "New description"
        assert exercise.difficulty == Difficulty.HARD
        # updated_at might be the same if executed quickly
        assert exercise.updated_at >= original_updated_at

    def test_exercise_update_partial(self):
        """Test partial update of exercise."""
        exercise = TypingExercise.create(
            title="Original",
            source_code="code",
            description="Original description",
        )

        exercise.update(title="New Title")

        assert exercise.title == "New Title"
        assert exercise.source_code == "code"
        assert exercise.description == "Original description"

    def test_exercise_char_count(self):
        """Test character count property."""
        exercise = TypingExercise.create(
            title="Test",
            source_code="print('hello')",  # 14 characters
        )
        assert exercise.char_count == 14

    def test_exercise_line_count(self):
        """Test line count property."""
        exercise = TypingExercise.create(
            title="Test",
            source_code="line1\nline2\nline3",  # 3 lines
        )
        assert exercise.line_count == 3

    def test_exercise_line_count_single_line(self):
        """Test line count for single line code."""
        exercise = TypingExercise.create(
            title="Test",
            source_code="single line",
        )
        assert exercise.line_count == 1

    def test_exercise_empty_source(self):
        """Test exercise with empty source code."""
        exercise = TypingExercise.create(
            title="Empty",
            source_code="",
        )
        assert exercise.char_count == 0
        assert exercise.line_count == 0


class TestTypingAttempt:
    """Tests for TypingAttempt entity."""

    def test_attempt_creation(self):
        """Test creating an attempt directly."""
        attempt_id = uuid4()
        user_id = uuid4()
        exercise_id = uuid4()
        attempt = TypingAttempt(
            id=attempt_id,
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=1,
        )
        assert attempt.id == attempt_id
        assert attempt.user_id == user_id
        assert attempt.exercise_id == exercise_id
        assert attempt.attempt_number == 1
        assert attempt.status == AttemptStatus.IN_PROGRESS
        assert attempt.accuracy == 0.0
        assert attempt.wpm == 0.0
        assert attempt.time_seconds == 0.0

    def test_attempt_create_factory(self):
        """Test attempt factory method."""
        user_id = uuid4()
        exercise_id = uuid4()
        attempt = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=2,
        )
        assert attempt.id is not None
        assert attempt.user_id == user_id
        assert attempt.exercise_id == exercise_id
        assert attempt.attempt_number == 2
        assert attempt.status == AttemptStatus.IN_PROGRESS
        assert attempt.started_at is not None
        assert attempt.completed_at is None

    def test_attempt_complete(self):
        """Test completing an attempt."""
        attempt = TypingAttempt.create(
            user_id=uuid4(),
            exercise_id=uuid4(),
            attempt_number=1,
        )

        attempt.complete(
            user_code="print('hello')",
            accuracy=95.5,
            wpm=60.0,
            time_seconds=120.0,
        )

        assert attempt.user_code == "print('hello')"
        assert attempt.accuracy == 95.5
        assert attempt.wpm == 60.0
        assert attempt.time_seconds == 120.0
        assert attempt.status == AttemptStatus.COMPLETED
        assert attempt.completed_at is not None

    def test_attempt_abandon(self):
        """Test abandoning an attempt."""
        attempt = TypingAttempt.create(
            user_id=uuid4(),
            exercise_id=uuid4(),
            attempt_number=1,
        )

        attempt.abandon()

        assert attempt.status == AttemptStatus.ABANDONED
        assert attempt.completed_at is not None

    def test_attempt_is_completed(self):
        """Test is_completed property."""
        attempt = TypingAttempt.create(
            user_id=uuid4(),
            exercise_id=uuid4(),
            attempt_number=1,
        )
        assert attempt.is_completed is False

        attempt.complete(
            user_code="code",
            accuracy=100.0,
            wpm=50.0,
            time_seconds=60.0,
        )
        assert attempt.is_completed is True

    def test_attempt_is_completed_abandoned(self):
        """Test is_completed is False for abandoned."""
        attempt = TypingAttempt.create(
            user_id=uuid4(),
            exercise_id=uuid4(),
            attempt_number=1,
        )
        attempt.abandon()
        assert attempt.is_completed is False


class TestUserExerciseProgress:
    """Tests for UserExerciseProgress entity."""

    def test_progress_creation(self):
        """Test creating progress directly."""
        user_id = uuid4()
        exercise_id = uuid4()
        progress = UserExerciseProgress(
            user_id=user_id,
            exercise_id=exercise_id,
            completed_attempts=3,
            best_accuracy=98.5,
            best_wpm=75.0,
            total_time_seconds=360.0,
            is_mastered=False,
        )
        assert progress.user_id == user_id
        assert progress.exercise_id == exercise_id
        assert progress.completed_attempts == 3
        assert progress.best_accuracy == 98.5
        assert progress.best_wpm == 75.0
        assert progress.total_time_seconds == 360.0
        assert progress.is_mastered is False

    def test_progress_from_attempts_empty(self):
        """Test creating progress from empty attempts list."""
        user_id = uuid4()
        exercise_id = uuid4()
        progress = UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=[],
            required_completions=5,
        )
        assert progress.completed_attempts == 0
        assert progress.best_accuracy == 0.0
        assert progress.best_wpm == 0.0
        assert progress.total_time_seconds == 0.0
        assert progress.is_mastered is False

    def test_progress_from_attempts_with_data(self):
        """Test creating progress from attempts list."""
        user_id = uuid4()
        exercise_id = uuid4()

        # Create completed attempts
        attempts = []
        for i in range(3):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise_id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=90.0 + i * 5,  # 90, 95, 100
                wpm=50.0 + i * 10,  # 50, 60, 70
                time_seconds=120.0,
            )
            attempts.append(attempt)

        progress = UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=attempts,
            required_completions=5,
        )

        assert progress.completed_attempts == 3
        assert progress.best_accuracy == 100.0
        assert progress.best_wpm == 70.0
        assert progress.total_time_seconds == 360.0
        assert progress.is_mastered is False  # 3 < 5

    def test_progress_from_attempts_mastered(self):
        """Test mastery when required completions reached."""
        user_id = uuid4()
        exercise_id = uuid4()

        attempts = []
        for i in range(5):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise_id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=95.0,
                wpm=60.0,
                time_seconds=100.0,
            )
            attempts.append(attempt)

        progress = UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=attempts,
            required_completions=5,
        )

        assert progress.completed_attempts == 5
        assert progress.is_mastered is True

    def test_progress_from_attempts_mixed_status(self):
        """Test progress calculation with mixed attempt statuses."""
        user_id = uuid4()
        exercise_id = uuid4()

        # One completed, one abandoned, one in progress
        attempt1 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=1,
        )
        attempt1.complete(
            user_code="code",
            accuracy=90.0,
            wpm=50.0,
            time_seconds=100.0,
        )

        attempt2 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=2,
        )
        attempt2.abandon()

        attempt3 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=3,
        )
        # Still in progress

        progress = UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=[attempt1, attempt2, attempt3],
            required_completions=5,
        )

        # Only completed attempts count
        assert progress.completed_attempts == 1
        assert progress.best_accuracy == 90.0
        assert progress.best_wpm == 50.0
        assert progress.total_time_seconds == 100.0
        assert progress.is_mastered is False


class TestUtcNow:
    """Tests for utc_now helper function."""

    def test_utc_now_returns_aware_datetime(self):
        """Test that utc_now returns timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test that utc_now returns current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after
