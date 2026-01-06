"""Tests for Typing Practice Application Services."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from code_tutor.typing_practice.domain.entities import (
    TypingExercise,
    TypingAttempt,
)
from code_tutor.typing_practice.domain.value_objects import (
    ExerciseCategory,
    AttemptStatus,
    Difficulty,
)
from code_tutor.typing_practice.application.services import TypingPracticeService
from code_tutor.typing_practice.application.dto import (
    CreateExerciseRequest,
    CompleteAttemptRequest,
)


@pytest.fixture
def mock_exercise_repo():
    """Create mock exercise repository."""
    return AsyncMock()


@pytest.fixture
def mock_attempt_repo():
    """Create mock attempt repository."""
    return AsyncMock()


@pytest.fixture
def service(mock_exercise_repo, mock_attempt_repo):
    """Create TypingPracticeService with mocks."""
    return TypingPracticeService(
        exercise_repo=mock_exercise_repo,
        attempt_repo=mock_attempt_repo,
    )


@pytest.fixture
def sample_exercise():
    """Create sample typing exercise."""
    return TypingExercise.create(
        title="Two Pointers Template",
        source_code="def two_pointers(arr):\n    left, right = 0, len(arr) - 1\n    return left, right",
        language="python",
        category=ExerciseCategory.TEMPLATE,
        difficulty=Difficulty.MEDIUM,
        description="Practice two pointers pattern",
        required_completions=5,
    )


@pytest.fixture
def sample_attempt(sample_exercise):
    """Create sample typing attempt."""
    return TypingAttempt.create(
        user_id=uuid4(),
        exercise_id=sample_exercise.id,
        attempt_number=1,
    )


class TestTypingPracticeServiceExerciseOperations:
    """Tests for exercise operations."""

    @pytest.mark.asyncio
    async def test_create_exercise(self, service, mock_exercise_repo):
        """Test creating a new exercise."""
        request = CreateExerciseRequest(
            title="Binary Search",
            source_code="def binary_search(arr, target):\n    pass",
            language="python",
            category=ExerciseCategory.ALGORITHM,
            difficulty=Difficulty.EASY,
            description="Binary search template",
            required_completions=3,
        )

        # Mock save to return the exercise
        async def save_exercise(exercise):
            return exercise

        mock_exercise_repo.save = AsyncMock(side_effect=save_exercise)

        result = await service.create_exercise(request)

        assert result.title == "Binary Search"
        assert result.category == ExerciseCategory.ALGORITHM
        assert result.difficulty == Difficulty.EASY
        assert result.required_completions == 3
        mock_exercise_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_exercise_found(self, service, mock_exercise_repo, sample_exercise):
        """Test getting an existing exercise."""
        mock_exercise_repo.get_by_id = AsyncMock(return_value=sample_exercise)

        result = await service.get_exercise(sample_exercise.id)

        assert result is not None
        assert result.id == sample_exercise.id
        assert result.title == sample_exercise.title
        mock_exercise_repo.get_by_id.assert_called_once_with(sample_exercise.id)

    @pytest.mark.asyncio
    async def test_get_exercise_not_found(self, service, mock_exercise_repo):
        """Test getting non-existent exercise."""
        mock_exercise_repo.get_by_id = AsyncMock(return_value=None)
        exercise_id = uuid4()

        result = await service.get_exercise(exercise_id)

        assert result is None
        mock_exercise_repo.get_by_id.assert_called_once_with(exercise_id)

    @pytest.mark.asyncio
    async def test_list_exercises(self, service, mock_exercise_repo, sample_exercise):
        """Test listing exercises."""
        exercises = [sample_exercise]
        mock_exercise_repo.list_all = AsyncMock(return_value=exercises)
        mock_exercise_repo.count = AsyncMock(return_value=1)

        result = await service.list_exercises(page=1, page_size=20)

        assert result.total == 1
        assert len(result.exercises) == 1
        assert result.page == 1
        assert result.page_size == 20
        mock_exercise_repo.list_all.assert_called_once_with(
            category=None,
            limit=20,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_list_exercises_with_category_filter(
        self, service, mock_exercise_repo, sample_exercise
    ):
        """Test listing exercises with category filter."""
        mock_exercise_repo.list_all = AsyncMock(return_value=[sample_exercise])
        mock_exercise_repo.count = AsyncMock(return_value=1)

        result = await service.list_exercises(
            category=ExerciseCategory.TEMPLATE,
            page=1,
            page_size=10,
        )

        assert result.total == 1
        mock_exercise_repo.list_all.assert_called_once_with(
            category=ExerciseCategory.TEMPLATE,
            limit=10,
            offset=0,
        )
        mock_exercise_repo.count.assert_called_once_with(
            category=ExerciseCategory.TEMPLATE
        )

    @pytest.mark.asyncio
    async def test_list_exercises_pagination(self, service, mock_exercise_repo):
        """Test listing exercises with pagination."""
        mock_exercise_repo.list_all = AsyncMock(return_value=[])
        mock_exercise_repo.count = AsyncMock(return_value=50)

        result = await service.list_exercises(page=3, page_size=10)

        assert result.page == 3
        assert result.page_size == 10
        mock_exercise_repo.list_all.assert_called_once_with(
            category=None,
            limit=10,
            offset=20,  # (3-1) * 10
        )


class TestTypingPracticeServiceAttemptOperations:
    """Tests for attempt operations."""

    @pytest.mark.asyncio
    async def test_start_attempt_first(
        self, service, mock_exercise_repo, mock_attempt_repo, sample_exercise
    ):
        """Test starting first attempt."""
        user_id = uuid4()
        mock_attempt_repo.list_by_user_and_exercise = AsyncMock(return_value=[])

        async def save_attempt(attempt):
            return attempt

        mock_attempt_repo.save = AsyncMock(side_effect=save_attempt)

        result = await service.start_attempt(
            user_id=user_id,
            exercise_id=sample_exercise.id,
        )

        assert result.user_id == user_id
        assert result.exercise_id == sample_exercise.id
        assert result.attempt_number == 1
        assert result.status == AttemptStatus.IN_PROGRESS
        mock_attempt_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_attempt_subsequent(
        self, service, mock_attempt_repo, sample_exercise
    ):
        """Test starting subsequent attempt after completed attempts."""
        user_id = uuid4()

        # Create existing completed attempts
        existing_attempts = []
        for i in range(2):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=sample_exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=90.0,
                wpm=50.0,
                time_seconds=100.0,
            )
            existing_attempts.append(attempt)

        mock_attempt_repo.list_by_user_and_exercise = AsyncMock(
            return_value=existing_attempts
        )

        async def save_attempt(attempt):
            return attempt

        mock_attempt_repo.save = AsyncMock(side_effect=save_attempt)

        result = await service.start_attempt(
            user_id=user_id,
            exercise_id=sample_exercise.id,
        )

        assert result.attempt_number == 3  # 2 completed + 1

    @pytest.mark.asyncio
    async def test_complete_attempt_success(
        self, service, mock_attempt_repo, sample_attempt
    ):
        """Test completing an attempt."""
        mock_attempt_repo.get_by_id = AsyncMock(return_value=sample_attempt)

        async def save_attempt(attempt):
            return attempt

        mock_attempt_repo.save = AsyncMock(side_effect=save_attempt)

        request = CompleteAttemptRequest(
            user_code="def two_pointers(arr):\n    left, right = 0, len(arr) - 1\n    return left, right",
            accuracy=98.5,
            wpm=65.0,
            time_seconds=45.0,
        )

        result = await service.complete_attempt(sample_attempt.id, request)

        assert result is not None
        assert result.status == AttemptStatus.COMPLETED
        assert result.accuracy == 98.5
        assert result.wpm == 65.0
        assert result.time_seconds == 45.0

    @pytest.mark.asyncio
    async def test_complete_attempt_not_found(self, service, mock_attempt_repo):
        """Test completing non-existent attempt."""
        mock_attempt_repo.get_by_id = AsyncMock(return_value=None)
        attempt_id = uuid4()

        request = CompleteAttemptRequest(
            user_code="code",
            accuracy=100.0,
            wpm=50.0,
            time_seconds=60.0,
        )

        result = await service.complete_attempt(attempt_id, request)

        assert result is None


class TestTypingPracticeServiceProgressOperations:
    """Tests for progress operations."""

    @pytest.mark.asyncio
    async def test_get_user_progress_success(
        self, service, mock_exercise_repo, mock_attempt_repo, sample_exercise
    ):
        """Test getting user progress on an exercise."""
        user_id = uuid4()
        mock_exercise_repo.get_by_id = AsyncMock(return_value=sample_exercise)

        # Create some completed attempts
        attempts = []
        for i in range(3):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=sample_exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=90.0 + i * 5,
                wpm=50.0 + i * 5,
                time_seconds=100.0,
            )
            attempts.append(attempt)

        mock_attempt_repo.list_by_user_and_exercise = AsyncMock(return_value=attempts)

        result = await service.get_user_progress(user_id, sample_exercise.id)

        assert result is not None
        assert result.user_id == user_id
        assert result.exercise_id == sample_exercise.id
        assert result.completed_attempts == 3
        assert result.required_completions == 5
        assert result.best_accuracy == 100.0
        assert result.best_wpm == 60.0
        assert result.is_mastered is False

    @pytest.mark.asyncio
    async def test_get_user_progress_exercise_not_found(
        self, service, mock_exercise_repo
    ):
        """Test getting progress for non-existent exercise."""
        mock_exercise_repo.get_by_id = AsyncMock(return_value=None)
        user_id = uuid4()
        exercise_id = uuid4()

        result = await service.get_user_progress(user_id, exercise_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_progress_mastered(
        self, service, mock_exercise_repo, mock_attempt_repo, sample_exercise
    ):
        """Test getting progress when exercise is mastered."""
        user_id = uuid4()
        mock_exercise_repo.get_by_id = AsyncMock(return_value=sample_exercise)

        # Create 5 completed attempts (mastery)
        attempts = []
        for i in range(5):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=sample_exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=95.0,
                wpm=60.0,
                time_seconds=100.0,
            )
            attempts.append(attempt)

        mock_attempt_repo.list_by_user_and_exercise = AsyncMock(return_value=attempts)

        result = await service.get_user_progress(user_id, sample_exercise.id)

        assert result.is_mastered is True
        assert result.completed_attempts == 5

    @pytest.mark.asyncio
    async def test_get_user_stats(self, service, mock_attempt_repo):
        """Test getting user's overall typing statistics."""
        user_id = uuid4()
        stats = {
            "total_exercises_attempted": 10,
            "total_exercises_mastered": 3,
            "total_attempts": 45,
            "average_accuracy": 92.5,
            "average_wpm": 55.0,
            "total_time_seconds": 5400.0,
            "best_wpm": 80.0,
        }
        mock_attempt_repo.get_user_stats = AsyncMock(return_value=stats)

        result = await service.get_user_stats(user_id)

        assert result.total_exercises_attempted == 10
        assert result.total_exercises_mastered == 3
        assert result.total_attempts == 45
        assert result.average_accuracy == 92.5
        assert result.average_wpm == 55.0
        assert result.total_time_seconds == 5400.0
        assert result.best_wpm == 80.0
        mock_attempt_repo.get_user_stats.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_get_user_stats_empty(self, service, mock_attempt_repo):
        """Test getting stats for user with no attempts."""
        user_id = uuid4()
        stats = {
            "total_exercises_attempted": 0,
            "total_exercises_mastered": 0,
            "total_attempts": 0,
            "average_accuracy": 0.0,
            "average_wpm": 0.0,
            "total_time_seconds": 0.0,
            "best_wpm": 0.0,
        }
        mock_attempt_repo.get_user_stats = AsyncMock(return_value=stats)

        result = await service.get_user_stats(user_id)

        assert result.total_exercises_attempted == 0
        assert result.total_exercises_mastered == 0
        assert result.total_attempts == 0


class TestTypingPracticeServiceHelperMethods:
    """Tests for helper methods."""

    @pytest.mark.asyncio
    async def test_to_exercise_response(self, service, sample_exercise):
        """Test exercise entity to response conversion."""
        response = service._to_exercise_response(sample_exercise)

        assert response.id == sample_exercise.id
        assert response.title == sample_exercise.title
        assert response.source_code == sample_exercise.source_code
        assert response.language == sample_exercise.language
        assert response.category == sample_exercise.category
        assert response.difficulty == sample_exercise.difficulty
        assert response.description == sample_exercise.description
        assert response.required_completions == sample_exercise.required_completions
        assert response.char_count == sample_exercise.char_count
        assert response.line_count == sample_exercise.line_count
        assert response.created_at == sample_exercise.created_at

    @pytest.mark.asyncio
    async def test_to_attempt_response(self, service, sample_attempt):
        """Test attempt entity to response conversion."""
        response = service._to_attempt_response(sample_attempt)

        assert response.id == sample_attempt.id
        assert response.user_id == sample_attempt.user_id
        assert response.exercise_id == sample_attempt.exercise_id
        assert response.attempt_number == sample_attempt.attempt_number
        assert response.accuracy == sample_attempt.accuracy
        assert response.wpm == sample_attempt.wpm
        assert response.time_seconds == sample_attempt.time_seconds
        assert response.status == sample_attempt.status
        assert response.started_at == sample_attempt.started_at
        assert response.completed_at == sample_attempt.completed_at

    @pytest.mark.asyncio
    async def test_to_attempt_response_completed(self, service, sample_attempt):
        """Test attempt response for completed attempt."""
        sample_attempt.complete(
            user_code="code",
            accuracy=95.0,
            wpm=60.0,
            time_seconds=120.0,
        )

        response = service._to_attempt_response(sample_attempt)

        assert response.status == AttemptStatus.COMPLETED
        assert response.accuracy == 95.0
        assert response.wpm == 60.0
        assert response.time_seconds == 120.0
        assert response.completed_at is not None
