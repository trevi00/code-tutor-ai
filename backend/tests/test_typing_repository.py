"""Tests for Typing Practice Repository Implementations."""

from uuid import uuid4, UUID as UUIDType

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from code_tutor.shared.infrastructure.database import Base
from code_tutor.identity.infrastructure.models import UserModel
from code_tutor.typing_practice.domain.entities import (
    TypingExercise,
    TypingAttempt,
)
from code_tutor.typing_practice.domain.value_objects import (
    ExerciseCategory,
    AttemptStatus,
    Difficulty,
)
from code_tutor.typing_practice.infrastructure.models import (
    TypingExerciseModel,
    TypingAttemptModel,
)
from code_tutor.typing_practice.infrastructure.repository import (
    SQLAlchemyTypingExerciseRepository,
    SQLAlchemyTypingAttemptRepository,
)


def _get_uuid(value) -> UUIDType:
    """Convert value to UUID if needed."""
    if isinstance(value, UUIDType):
        return value
    return UUIDType(value)


@pytest_asyncio.fixture
async def db_engine():
    """Create async engine for tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Create async session for tests."""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Create test user in database."""
    user = UserModel(
        id=uuid4(),
        email="typing_test@example.com",
        username="typing_tester",
        hashed_password="hashedpassword",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def exercise_repo(db_session: AsyncSession):
    """Create exercise repository."""
    return SQLAlchemyTypingExerciseRepository(db_session)


@pytest_asyncio.fixture
async def attempt_repo(db_session: AsyncSession):
    """Create attempt repository."""
    return SQLAlchemyTypingAttemptRepository(db_session)


@pytest_asyncio.fixture
async def sample_exercise(exercise_repo):
    """Create and save a sample exercise."""
    exercise = TypingExercise.create(
        title="Two Pointers Template",
        source_code="def two_pointers(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        pass",
        language="python",
        category=ExerciseCategory.TEMPLATE,
        difficulty=Difficulty.MEDIUM,
        description="Practice two pointers pattern",
        required_completions=5,
    )
    await exercise_repo.save(exercise)
    return exercise


class TestSQLAlchemyTypingExerciseRepository:
    """Tests for SQLAlchemy typing exercise repository."""

    @pytest.mark.asyncio
    async def test_save_new_exercise(self, exercise_repo):
        """Test saving a new exercise."""
        exercise = TypingExercise.create(
            title="Binary Search",
            source_code="def binary_search(arr, target):\n    pass",
            language="python",
            category=ExerciseCategory.ALGORITHM,
            difficulty=Difficulty.EASY,
        )

        saved = await exercise_repo.save(exercise)

        assert saved.id == exercise.id
        assert saved.title == "Binary Search"

    @pytest.mark.asyncio
    async def test_save_update_exercise(self, exercise_repo, sample_exercise):
        """Test updating an existing exercise."""
        sample_exercise.update(title="Updated Title")

        saved = await exercise_repo.save(sample_exercise)

        assert saved.title == "Updated Title"

        # Verify by fetching
        fetched = await exercise_repo.get_by_id(sample_exercise.id)
        assert fetched.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, exercise_repo, sample_exercise):
        """Test getting exercise by ID."""
        result = await exercise_repo.get_by_id(sample_exercise.id)

        assert result is not None
        assert result.id == sample_exercise.id
        assert result.title == sample_exercise.title
        assert result.source_code == sample_exercise.source_code
        assert result.category == ExerciseCategory.TEMPLATE
        assert result.difficulty == Difficulty.MEDIUM

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, exercise_repo):
        """Test getting non-existent exercise."""
        result = await exercise_repo.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, exercise_repo):
        """Test listing all exercises."""
        # Create multiple exercises
        for i in range(3):
            exercise = TypingExercise.create(
                title=f"Exercise {i}",
                source_code=f"code {i}",
            )
            await exercise_repo.save(exercise)

        result = await exercise_repo.list_all()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_all_with_category_filter(self, exercise_repo):
        """Test listing exercises with category filter."""
        # Create exercises with different categories
        template = TypingExercise.create(
            title="Template",
            source_code="template code",
            category=ExerciseCategory.TEMPLATE,
        )
        algorithm = TypingExercise.create(
            title="Algorithm",
            source_code="algorithm code",
            category=ExerciseCategory.ALGORITHM,
        )
        await exercise_repo.save(template)
        await exercise_repo.save(algorithm)

        result = await exercise_repo.list_all(category=ExerciseCategory.TEMPLATE)

        assert len(result) == 1
        assert result[0].category == ExerciseCategory.TEMPLATE

    @pytest.mark.asyncio
    async def test_list_all_with_pagination(self, exercise_repo):
        """Test listing exercises with pagination."""
        # Create 5 exercises
        for i in range(5):
            exercise = TypingExercise.create(
                title=f"Exercise {i}",
                source_code=f"code {i}",
            )
            await exercise_repo.save(exercise)

        result = await exercise_repo.list_all(limit=2, offset=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_only_published(self, exercise_repo):
        """Test that list_all only returns published exercises."""
        published = TypingExercise.create(
            title="Published",
            source_code="code",
        )
        unpublished = TypingExercise.create(
            title="Unpublished",
            source_code="code",
        )
        unpublished.is_published = False

        await exercise_repo.save(published)
        await exercise_repo.save(unpublished)

        result = await exercise_repo.list_all()

        assert len(result) == 1
        assert result[0].is_published is True

    @pytest.mark.asyncio
    async def test_delete_exercise(self, exercise_repo, sample_exercise):
        """Test deleting an exercise."""
        result = await exercise_repo.delete(sample_exercise.id)

        assert result is True

        # Verify deleted
        fetched = await exercise_repo.get_by_id(sample_exercise.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_non_existent(self, exercise_repo):
        """Test deleting non-existent exercise."""
        result = await exercise_repo.delete(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_count_all(self, exercise_repo):
        """Test counting all exercises."""
        for i in range(3):
            exercise = TypingExercise.create(
                title=f"Exercise {i}",
                source_code=f"code {i}",
            )
            await exercise_repo.save(exercise)

        count = await exercise_repo.count()

        assert count == 3

    @pytest.mark.asyncio
    async def test_count_with_category(self, exercise_repo):
        """Test counting exercises by category."""
        for i in range(2):
            template = TypingExercise.create(
                title=f"Template {i}",
                source_code=f"code {i}",
                category=ExerciseCategory.TEMPLATE,
            )
            await exercise_repo.save(template)

        algorithm = TypingExercise.create(
            title="Algorithm",
            source_code="code",
            category=ExerciseCategory.ALGORITHM,
        )
        await exercise_repo.save(algorithm)

        template_count = await exercise_repo.count(category=ExerciseCategory.TEMPLATE)
        algorithm_count = await exercise_repo.count(category=ExerciseCategory.ALGORITHM)

        assert template_count == 2
        assert algorithm_count == 1


class TestSQLAlchemyTypingAttemptRepository:
    """Tests for SQLAlchemy typing attempt repository."""

    @pytest.mark.asyncio
    async def test_save_new_attempt(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test saving a new attempt."""
        # Create exercise first
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)
        attempt = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise.id,
            attempt_number=1,
        )

        saved = await attempt_repo.save(attempt)

        assert saved.id == attempt.id
        assert saved.user_id == user_id
        assert saved.exercise_id == exercise.id

    @pytest.mark.asyncio
    async def test_save_update_attempt(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test updating an existing attempt."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)
        attempt = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise.id,
            attempt_number=1,
        )
        await attempt_repo.save(attempt)

        # Complete the attempt
        attempt.complete(
            user_code="completed code",
            accuracy=95.0,
            wpm=60.0,
            time_seconds=120.0,
        )
        await attempt_repo.save(attempt)

        # Verify
        fetched = await attempt_repo.get_by_id(attempt.id)
        assert fetched.status == AttemptStatus.COMPLETED
        assert fetched.accuracy == 95.0
        assert fetched.wpm == 60.0

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting attempt by ID."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)
        attempt = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise.id,
            attempt_number=1,
        )
        await attempt_repo.save(attempt)

        result = await attempt_repo.get_by_id(attempt.id)

        assert result is not None
        assert result.id == attempt.id
        assert result.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, attempt_repo):
        """Test getting non-existent attempt."""
        result = await attempt_repo.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_user_and_exercise(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test listing attempts by user and exercise."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)

        # Create multiple attempts
        for i in range(3):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=i + 1,
            )
            await attempt_repo.save(attempt)

        result = await attempt_repo.list_by_user_and_exercise(
            user_id=user_id,
            exercise_id=exercise.id,
        )

        assert len(result) == 3
        # Should be ordered by attempt_number
        assert result[0].attempt_number == 1
        assert result[1].attempt_number == 2
        assert result[2].attempt_number == 3

    @pytest.mark.asyncio
    async def test_list_by_user(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test listing all attempts by user."""
        # Create two exercises
        exercise1 = TypingExercise.create(
            title="Exercise 1",
            source_code="code1",
        )
        exercise2 = TypingExercise.create(
            title="Exercise 2",
            source_code="code2",
        )
        await exercise_repo.save(exercise1)
        await exercise_repo.save(exercise2)

        user_id = _get_uuid(test_user.id)

        # Create attempts for both exercises
        attempt1 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise1.id,
            attempt_number=1,
        )
        attempt2 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise2.id,
            attempt_number=1,
        )
        await attempt_repo.save(attempt1)
        await attempt_repo.save(attempt2)

        result = await attempt_repo.list_by_user(user_id)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_by_user_with_pagination(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test listing user attempts with pagination."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)

        # Create 5 attempts
        for i in range(5):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=i + 1,
            )
            await attempt_repo.save(attempt)

        result = await attempt_repo.list_by_user(user_id, limit=2, offset=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_user_progress(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting user progress on an exercise."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
            required_completions=5,
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)

        # Create completed attempts
        for i in range(3):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=90.0 + i * 5,
                wpm=50.0 + i * 10,
                time_seconds=100.0,
            )
            await attempt_repo.save(attempt)

        progress = await attempt_repo.get_user_progress(user_id, exercise.id)

        assert progress is not None
        assert progress.completed_attempts == 3
        assert progress.best_accuracy == 100.0
        assert progress.best_wpm == 70.0
        assert progress.is_mastered is False  # 3 < 5

    @pytest.mark.asyncio
    async def test_get_user_progress_no_attempts(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting progress when no attempts exist."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)
        progress = await attempt_repo.get_user_progress(user_id, exercise.id)

        assert progress is None

    @pytest.mark.asyncio
    async def test_get_user_stats(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting user's overall typing statistics."""
        # Create exercises
        exercise1 = TypingExercise.create(
            title="Exercise 1",
            source_code="code1",
        )
        exercise2 = TypingExercise.create(
            title="Exercise 2",
            source_code="code2",
        )
        await exercise_repo.save(exercise1)
        await exercise_repo.save(exercise2)

        user_id = _get_uuid(test_user.id)

        # Create completed attempts
        attempt1 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise1.id,
            attempt_number=1,
        )
        attempt1.complete(
            user_code="code",
            accuracy=90.0,
            wpm=50.0,
            time_seconds=100.0,
        )
        await attempt_repo.save(attempt1)

        attempt2 = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise2.id,
            attempt_number=1,
        )
        attempt2.complete(
            user_code="code",
            accuracy=100.0,
            wpm=70.0,
            time_seconds=80.0,
        )
        await attempt_repo.save(attempt2)

        stats = await attempt_repo.get_user_stats(user_id)

        assert stats["total_exercises_attempted"] == 2
        assert stats["total_attempts"] == 2
        assert stats["average_accuracy"] == 95.0  # (90 + 100) / 2
        assert stats["average_wpm"] == 60.0  # (50 + 70) / 2
        assert stats["total_time_seconds"] == 180.0  # 100 + 80
        assert stats["best_wpm"] == 70.0

    @pytest.mark.asyncio
    async def test_get_user_stats_no_attempts(self, attempt_repo, test_user):
        """Test getting stats for user with no attempts."""
        user_id = _get_uuid(test_user.id)
        stats = await attempt_repo.get_user_stats(user_id)

        assert stats["total_exercises_attempted"] == 0
        assert stats["total_exercises_mastered"] == 0
        assert stats["total_attempts"] == 0
        assert stats["average_accuracy"] == 0.0
        assert stats["average_wpm"] == 0.0

    @pytest.mark.asyncio
    async def test_get_mastered_exercise_ids(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting mastered exercise IDs."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
            required_completions=5,
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)

        # Create 5 completed attempts (mastery)
        for i in range(5):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=95.0,
                wpm=60.0,
                time_seconds=100.0,
            )
            await attempt_repo.save(attempt)

        mastered_ids = await attempt_repo.get_mastered_exercise_ids(user_id)

        assert len(mastered_ids) == 1
        assert str(exercise.id) in mastered_ids

    @pytest.mark.asyncio
    async def test_get_mastered_exercise_ids_not_mastered(
        self, attempt_repo, exercise_repo, test_user
    ):
        """Test getting mastered IDs when not enough completions."""
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
            required_completions=5,
        )
        await exercise_repo.save(exercise)

        user_id = _get_uuid(test_user.id)

        # Only 2 completions (not mastered)
        for i in range(2):
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=i + 1,
            )
            attempt.complete(
                user_code="code",
                accuracy=95.0,
                wpm=60.0,
                time_seconds=100.0,
            )
            await attempt_repo.save(attempt)

        mastered_ids = await attempt_repo.get_mastered_exercise_ids(user_id)

        assert len(mastered_ids) == 0

    @pytest.mark.asyncio
    async def test_get_leaderboard(
        self, attempt_repo, exercise_repo, db_session
    ):
        """Test getting leaderboard."""
        # Create exercise
        exercise = TypingExercise.create(
            title="Test Exercise",
            source_code="code",
        )
        await exercise_repo.save(exercise)

        # Create multiple users and attempts
        for i in range(3):
            user = UserModel(
                id=uuid4(),
                email=f"user{i}@example.com",
                username=f"user{i}",
                hashed_password="hash",
            )
            db_session.add(user)
            await db_session.commit()

            user_id = _get_uuid(user.id)
            attempt = TypingAttempt.create(
                user_id=user_id,
                exercise_id=exercise.id,
                attempt_number=1,
            )
            attempt.complete(
                user_code="code",
                accuracy=90.0,
                wpm=50.0 + i * 10,  # 50, 60, 70 WPM
                time_seconds=100.0,
            )
            await attempt_repo.save(attempt)

        leaderboard = await attempt_repo.get_leaderboard(limit=3)

        assert len(leaderboard) == 3
        # Should be sorted by WPM descending
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[0]["best_wpm"] == 70.0
        assert leaderboard[1]["rank"] == 2
        assert leaderboard[1]["best_wpm"] == 60.0
        assert leaderboard[2]["rank"] == 3
        assert leaderboard[2]["best_wpm"] == 50.0

    @pytest.mark.asyncio
    async def test_get_leaderboard_empty(self, attempt_repo):
        """Test getting leaderboard when no attempts."""
        leaderboard = await attempt_repo.get_leaderboard()
        assert len(leaderboard) == 0
