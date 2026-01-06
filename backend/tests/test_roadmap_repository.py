"""Tests for Roadmap Repository Implementations."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_tutor.roadmap.domain.entities import (
    LearningPath,
    Lesson,
    Module,
    UserLessonProgress,
    UserPathProgress,
)
from code_tutor.roadmap.domain.value_objects import (
    LessonType,
    PathLevel,
    ProgressStatus,
)
from code_tutor.roadmap.infrastructure.models import (
    LearningPathModel,
    LessonModel,
    ModuleModel,
    PathPrerequisiteModel,
    UserLessonProgressModel,
    UserPathProgressModel,
)
from code_tutor.roadmap.infrastructure.repository import (
    SQLAlchemyLearningPathRepository,
    SQLAlchemyLessonRepository,
    SQLAlchemyModuleRepository,
    SQLAlchemyUserProgressRepository,
)
from code_tutor.shared.infrastructure.database import Base
from code_tutor.identity.infrastructure.models import UserModel


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Create test database session with all tables."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    """Create a test user."""
    user = UserModel(
        id=uuid4(),
        email="test@example.com",
        username="testuser",
        hashed_password="hashedpassword",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def sample_path_model(db_session: AsyncSession):
    """Create a sample learning path in database."""
    path = LearningPathModel(
        id=str(uuid4()),
        level=PathLevel.BEGINNER.value,
        title="Python Basics",
        description="Learn Python from scratch",
        icon="snake",
        order=1,
        estimated_hours=20,
        is_published=True,
    )
    db_session.add(path)
    await db_session.commit()
    return path


@pytest_asyncio.fixture
async def sample_module_model(db_session: AsyncSession, sample_path_model):
    """Create a sample module in database."""
    module = ModuleModel(
        id=str(uuid4()),
        path_id=sample_path_model.id,
        title="Getting Started",
        description="Introduction to Python",
        order=1,
    )
    db_session.add(module)
    await db_session.commit()
    return module


@pytest_asyncio.fixture
async def sample_lesson_model(db_session: AsyncSession, sample_module_model):
    """Create a sample lesson in database."""
    lesson = LessonModel(
        id=str(uuid4()),
        module_id=sample_module_model.id,
        title="Hello World",
        description="Learn to print Hello World",
        lesson_type=LessonType.CONCEPT.value,
        content="# Hello World\nprint('Hello')",
        order=1,
        xp_reward=10,
        estimated_minutes=5,
    )
    db_session.add(lesson)
    await db_session.commit()
    return lesson


class TestSQLAlchemyLearningPathRepository:
    """Tests for LearningPathRepository implementation."""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session: AsyncSession):
        """Test getting non-existent path returns None."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.get_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, db_session: AsyncSession, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting existing path."""
        repo = SQLAlchemyLearningPathRepository(db_session)
        from uuid import UUID
        path_id = UUID(sample_path_model.id) if isinstance(sample_path_model.id, str) else sample_path_model.id

        result = await repo.get_by_id(path_id)

        assert result is not None
        assert result.title == "Python Basics"
        assert result.level == PathLevel.BEGINNER
        assert len(result.modules) == 1
        assert result.modules[0].title == "Getting Started"
        assert len(result.modules[0].lessons) == 1

    @pytest.mark.asyncio
    async def test_get_by_level_not_found(self, db_session: AsyncSession):
        """Test getting path by level when not found."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.get_by_level(PathLevel.ADVANCED)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_level_found(
        self, db_session: AsyncSession, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting path by level."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.get_by_level(PathLevel.BEGINNER)

        assert result is not None
        assert result.level == PathLevel.BEGINNER
        assert result.title == "Python Basics"

    @pytest.mark.asyncio
    async def test_list_all_empty(self, db_session: AsyncSession):
        """Test listing paths when empty."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.list_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_all_with_paths(
        self, db_session: AsyncSession, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test listing all paths."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.list_all()

        assert len(result) == 1
        assert result[0].title == "Python Basics"

    @pytest.mark.asyncio
    async def test_list_all_excludes_unpublished(self, db_session: AsyncSession):
        """Test listing paths excludes unpublished by default."""
        # Create unpublished path
        unpublished = LearningPathModel(
            id=str(uuid4()),
            level=PathLevel.ADVANCED.value,
            title="Advanced Path",
            is_published=False,
        )
        db_session.add(unpublished)
        await db_session.commit()

        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.list_all()
        assert len(result) == 0

        result_with_unpublished = await repo.list_all(include_unpublished=True)
        assert len(result_with_unpublished) == 1

    @pytest.mark.asyncio
    async def test_save_new_path(self, db_session: AsyncSession):
        """Test saving a new path."""
        repo = SQLAlchemyLearningPathRepository(db_session)
        path = LearningPath(
            level=PathLevel.INTERMEDIATE,
            title="Intermediate Python",
            description="Level up your Python skills",
            order=2,
        )

        result = await repo.save(path)

        assert result.id == path.id
        # Verify in database
        saved = await repo.get_by_id(path.id)
        assert saved is not None
        assert saved.title == "Intermediate Python"

    @pytest.mark.asyncio
    async def test_delete_path(
        self, db_session: AsyncSession, sample_path_model
    ):
        """Test deleting a path."""
        repo = SQLAlchemyLearningPathRepository(db_session)
        path_id_str = sample_path_model.id
        from uuid import UUID
        path_id = UUID(path_id_str)

        result = await repo.delete(path_id)

        assert result is True
        # Verify deleted
        deleted = await repo.get_by_id(path_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_path(self, db_session: AsyncSession):
        """Test deleting non-existent path returns False."""
        repo = SQLAlchemyLearningPathRepository(db_session)

        result = await repo.delete(uuid4())

        assert result is False


class TestSQLAlchemyModuleRepository:
    """Tests for ModuleRepository implementation."""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session: AsyncSession):
        """Test getting non-existent module returns None."""
        repo = SQLAlchemyModuleRepository(db_session)

        result = await repo.get_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, db_session: AsyncSession, sample_module_model, sample_lesson_model
    ):
        """Test getting existing module."""
        repo = SQLAlchemyModuleRepository(db_session)
        from uuid import UUID
        module_id = UUID(sample_module_model.id)

        result = await repo.get_by_id(module_id)

        assert result is not None
        assert result.title == "Getting Started"
        assert len(result.lessons) == 1
        assert result.lessons[0].title == "Hello World"

    @pytest.mark.asyncio
    async def test_get_by_path_id_empty(self, db_session: AsyncSession):
        """Test getting modules for non-existent path."""
        repo = SQLAlchemyModuleRepository(db_session)

        result = await repo.get_by_path_id(uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_path_id_with_modules(
        self, db_session: AsyncSession, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting modules for path."""
        repo = SQLAlchemyModuleRepository(db_session)
        from uuid import UUID
        path_id = UUID(sample_path_model.id)

        result = await repo.get_by_path_id(path_id)

        assert len(result) == 1
        assert result[0].title == "Getting Started"

    @pytest.mark.asyncio
    async def test_save_module(self, db_session: AsyncSession, sample_path_model):
        """Test saving a module."""
        repo = SQLAlchemyModuleRepository(db_session)
        from uuid import UUID
        module = Module(
            path_id=UUID(sample_path_model.id),
            title="New Module",
            description="A new module",
            order=2,
        )

        result = await repo.save(module)

        assert result.id == module.id
        # Verify in database
        saved = await repo.get_by_id(module.id)
        assert saved is not None
        assert saved.title == "New Module"


class TestSQLAlchemyLessonRepository:
    """Tests for LessonRepository implementation."""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session: AsyncSession):
        """Test getting non-existent lesson returns None."""
        repo = SQLAlchemyLessonRepository(db_session)

        result = await repo.get_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self, db_session: AsyncSession, sample_lesson_model
    ):
        """Test getting existing lesson."""
        repo = SQLAlchemyLessonRepository(db_session)
        from uuid import UUID
        lesson_id = UUID(sample_lesson_model.id)

        result = await repo.get_by_id(lesson_id)

        assert result is not None
        assert result.title == "Hello World"
        assert result.lesson_type == LessonType.CONCEPT
        assert result.xp_reward == 10

    @pytest.mark.asyncio
    async def test_get_by_module_id_empty(self, db_session: AsyncSession):
        """Test getting lessons for non-existent module."""
        repo = SQLAlchemyLessonRepository(db_session)

        result = await repo.get_by_module_id(uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_module_id_with_lessons(
        self, db_session: AsyncSession, sample_module_model, sample_lesson_model
    ):
        """Test getting lessons for module."""
        repo = SQLAlchemyLessonRepository(db_session)
        from uuid import UUID
        module_id = UUID(sample_module_model.id)

        result = await repo.get_by_module_id(module_id)

        assert len(result) == 1
        assert result[0].title == "Hello World"

    @pytest.mark.asyncio
    async def test_get_by_path_id(
        self, db_session: AsyncSession, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting lessons for path."""
        repo = SQLAlchemyLessonRepository(db_session)
        from uuid import UUID
        path_id = UUID(sample_path_model.id)

        result = await repo.get_by_path_id(path_id)

        assert len(result) == 1
        assert result[0].title == "Hello World"

    @pytest.mark.asyncio
    async def test_save_lesson(self, db_session: AsyncSession, sample_module_model):
        """Test saving a lesson."""
        repo = SQLAlchemyLessonRepository(db_session)
        from uuid import UUID
        lesson = Lesson(
            module_id=UUID(sample_module_model.id),
            title="New Lesson",
            description="A new lesson",
            lesson_type=LessonType.QUIZ,
            order=2,
            xp_reward=20,
        )

        result = await repo.save(lesson)

        assert result.id == lesson.id
        # Verify in database
        saved = await repo.get_by_id(lesson.id)
        assert saved is not None
        assert saved.title == "New Lesson"
        assert saved.lesson_type == LessonType.QUIZ


def _get_uuid(value):
    """Helper to convert string to UUID if needed."""
    from uuid import UUID as UUIDType
    if isinstance(value, UUIDType):
        return value
    return UUIDType(value)


class TestSQLAlchemyUserProgressRepository:
    """Tests for UserProgressRepository implementation."""

    @pytest.mark.asyncio
    async def test_get_path_progress_not_found(
        self, db_session: AsyncSession, test_user
    ):
        """Test getting non-existent path progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)

        result = await repo.get_path_progress(user_id, uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_path_progress(
        self, db_session: AsyncSession, test_user, sample_path_model
    ):
        """Test saving and retrieving path progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)

        progress = UserPathProgress(
            user_id=user_id,
            path_id=path_id,
            total_lessons=10,
        )
        progress.start()

        await repo.save_path_progress(progress)

        # Retrieve
        result = await repo.get_path_progress(user_id, path_id)

        assert result is not None
        assert result.user_id == user_id
        assert result.path_id == path_id
        assert result.status == ProgressStatus.IN_PROGRESS
        assert result.total_lessons == 10

    @pytest.mark.asyncio
    async def test_update_path_progress(
        self, db_session: AsyncSession, test_user, sample_path_model
    ):
        """Test updating existing path progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)

        # Create initial progress
        progress = UserPathProgress(
            user_id=user_id,
            path_id=path_id,
            total_lessons=10,
        )
        progress.start()
        await repo.save_path_progress(progress)

        # Update progress
        progress.update_progress(completed=5, total=10)
        await repo.save_path_progress(progress)

        # Verify update
        result = await repo.get_path_progress(user_id, path_id)
        assert result.completed_lessons == 5
        assert result.status == ProgressStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_get_all_path_progress(
        self, db_session: AsyncSession, test_user, sample_path_model
    ):
        """Test getting all path progress for user."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)

        # Create progress
        progress = UserPathProgress(
            user_id=user_id,
            path_id=path_id,
            total_lessons=10,
        )
        progress.start()
        await repo.save_path_progress(progress)

        result = await repo.get_all_path_progress(user_id)

        assert len(result) == 1
        assert result[0].path_id == path_id

    @pytest.mark.asyncio
    async def test_get_lesson_progress_not_found(
        self, db_session: AsyncSession, test_user
    ):
        """Test getting non-existent lesson progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)

        result = await repo.get_lesson_progress(user_id, uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_lesson_progress(
        self, db_session: AsyncSession, test_user, sample_lesson_model
    ):
        """Test saving and retrieving lesson progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        progress.complete(score=95)

        await repo.save_lesson_progress(progress)

        # Retrieve
        result = await repo.get_lesson_progress(user_id, lesson_id)

        assert result is not None
        assert result.user_id == user_id
        assert result.lesson_id == lesson_id
        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 95
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_update_lesson_progress(
        self, db_session: AsyncSession, test_user, sample_lesson_model
    ):
        """Test updating existing lesson progress."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        # Create initial progress
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        await repo.save_lesson_progress(progress)

        # Complete the lesson
        progress.complete(score=100)
        await repo.save_lesson_progress(progress)

        # Verify update
        result = await repo.get_lesson_progress(user_id, lesson_id)
        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 100

    @pytest.mark.asyncio
    async def test_get_module_lessons_progress(
        self, db_session: AsyncSession, test_user, sample_module_model, sample_lesson_model
    ):
        """Test getting lesson progress for a module."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        module_id = _get_uuid(sample_module_model.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        # Create progress
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        await repo.save_lesson_progress(progress)

        result = await repo.get_module_lessons_progress(user_id, module_id)

        assert len(result) == 1
        assert result[0].lesson_id == lesson_id

    @pytest.mark.asyncio
    async def test_get_path_lessons_progress(
        self, db_session: AsyncSession, test_user, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting lesson progress for a path."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        # Create progress
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        await repo.save_lesson_progress(progress)

        result = await repo.get_path_lessons_progress(user_id, path_id)

        assert len(result) == 1
        assert result[0].lesson_id == lesson_id

    @pytest.mark.asyncio
    async def test_get_completed_lesson_count(
        self, db_session: AsyncSession, test_user, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test counting completed lessons in a path."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        # Initially no completed lessons
        count = await repo.get_completed_lesson_count(user_id, path_id)
        assert count == 0

        # Complete a lesson
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        progress.complete()
        await repo.save_lesson_progress(progress)

        # Now count should be 1
        count = await repo.get_completed_lesson_count(user_id, path_id)
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_next_lesson_no_lessons(
        self, db_session: AsyncSession, test_user
    ):
        """Test getting next lesson when no lessons exist."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)

        result = await repo.get_next_lesson(user_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_lesson_with_lessons(
        self, db_session: AsyncSession, test_user, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting next incomplete lesson."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)

        result = await repo.get_next_lesson(user_id)

        assert result is not None
        assert result.title == "Hello World"

    @pytest.mark.asyncio
    async def test_get_next_lesson_skips_completed(
        self, db_session: AsyncSession, test_user, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test that completed lessons are skipped."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        lesson_id = _get_uuid(sample_lesson_model.id)

        # Complete the only lesson
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        progress.start()
        progress.complete()
        await repo.save_lesson_progress(progress)

        result = await repo.get_next_lesson(user_id)

        # No more lessons
        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_lesson_for_specific_path(
        self, db_session: AsyncSession, test_user, sample_path_model, sample_module_model, sample_lesson_model
    ):
        """Test getting next lesson for specific path."""
        repo = SQLAlchemyUserProgressRepository(db_session)
        user_id = _get_uuid(test_user.id)
        path_id = _get_uuid(sample_path_model.id)

        result = await repo.get_next_lesson(user_id, path_id)

        assert result is not None
        assert result.title == "Hello World"

        # Non-existent path should return None
        result_none = await repo.get_next_lesson(user_id, uuid4())
        assert result_none is None
