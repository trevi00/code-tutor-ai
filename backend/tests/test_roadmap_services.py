"""Tests for Roadmap Application Services."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from code_tutor.roadmap.application.dto import CompleteLessonRequest
from code_tutor.roadmap.application.services import RoadmapService
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


@pytest.fixture
def mock_path_repo():
    """Create mock LearningPathRepository."""
    return AsyncMock()


@pytest.fixture
def mock_module_repo():
    """Create mock ModuleRepository."""
    return AsyncMock()


@pytest.fixture
def mock_lesson_repo():
    """Create mock LessonRepository."""
    return AsyncMock()


@pytest.fixture
def mock_progress_repo():
    """Create mock UserProgressRepository."""
    return AsyncMock()


@pytest.fixture
def mock_xp_service():
    """Create mock XP service."""
    service = AsyncMock()
    service.add_xp = AsyncMock()
    service.set_path_level_completed = AsyncMock()
    return service


@pytest.fixture
def roadmap_service(
    mock_path_repo,
    mock_module_repo,
    mock_lesson_repo,
    mock_progress_repo,
    mock_xp_service,
):
    """Create RoadmapService with mocked dependencies."""
    return RoadmapService(
        path_repo=mock_path_repo,
        module_repo=mock_module_repo,
        lesson_repo=mock_lesson_repo,
        progress_repo=mock_progress_repo,
        xp_service=mock_xp_service,
    )


@pytest.fixture
def sample_path():
    """Create sample learning path with modules and lessons."""
    # Create path first
    path = LearningPath(
        level=PathLevel.BEGINNER,
        title="Python Basics",
        description="Learn Python from scratch",
        icon="snake",
        order=1,
        estimated_hours=20,
    )

    # Create module and add to path
    module = Module(
        title="Getting Started",
        description="Introduction to Python",
        order=1,
    )
    path.add_module(module)  # Sets module.path_id

    # Create lesson and add to module
    lesson = Lesson(
        title="Hello World",
        description="Learn to print Hello World",
        lesson_type=LessonType.CONCEPT,
        content="# Hello World\nprint('Hello')",
        order=1,
        xp_reward=10,
        estimated_minutes=5,
    )
    module.add_lesson(lesson)  # Sets lesson.module_id

    return path


@pytest.fixture
def sample_module(sample_path):
    """Get sample module from path."""
    return sample_path.modules[0]


@pytest.fixture
def sample_lesson(sample_module):
    """Get sample lesson from module."""
    return sample_module.lessons[0]


class TestRoadmapServicePathMethods:
    """Tests for path-related methods."""

    @pytest.mark.asyncio
    async def test_list_paths_empty(self, roadmap_service, mock_path_repo):
        """Test listing paths when no paths exist."""
        mock_path_repo.list_all.return_value = []

        result = await roadmap_service.list_paths()

        assert result.total == 0
        assert result.items == []
        mock_path_repo.list_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_paths_with_paths(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test listing paths with existing paths."""
        mock_path_repo.list_all.return_value = [sample_path]
        mock_progress_repo.get_path_progress.return_value = None

        result = await roadmap_service.list_paths()

        assert result.total == 1
        assert len(result.items) == 1
        assert result.items[0].title == "Python Basics"
        assert result.items[0].level == PathLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_list_paths_with_user_progress(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test listing paths includes user progress."""
        user_id = uuid4()
        progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=1,
            completed_lessons=1,
            status=ProgressStatus.COMPLETED,
        )
        mock_path_repo.list_all.return_value = [sample_path]
        mock_progress_repo.get_path_progress.return_value = progress

        result = await roadmap_service.list_paths(user_id=user_id)

        assert result.items[0].status == ProgressStatus.COMPLETED
        assert result.items[0].completed_lessons == 1

    @pytest.mark.asyncio
    async def test_get_path_not_found(self, roadmap_service, mock_path_repo):
        """Test getting non-existent path."""
        mock_path_repo.get_by_id.return_value = None

        result = await roadmap_service.get_path(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_path_found(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test getting existing path."""
        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_path(sample_path.id)

        assert result is not None
        assert result.title == "Python Basics"
        assert result.level == PathLevel.BEGINNER
        assert len(result.modules) == 1

    @pytest.mark.asyncio
    async def test_get_path_by_level_not_found(self, roadmap_service, mock_path_repo):
        """Test getting path by level when not found."""
        mock_path_repo.get_by_level.return_value = None

        result = await roadmap_service.get_path_by_level(PathLevel.ADVANCED)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_path_by_level_found(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test getting path by level when found."""
        mock_path_repo.get_by_level.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_path_by_level(PathLevel.BEGINNER)

        assert result is not None
        assert result.level == PathLevel.BEGINNER


class TestRoadmapServiceModuleMethods:
    """Tests for module-related methods."""

    @pytest.mark.asyncio
    async def test_get_module_not_found(self, roadmap_service, mock_module_repo):
        """Test getting non-existent module."""
        mock_module_repo.get_by_id.return_value = None

        result = await roadmap_service.get_module(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_module_found(
        self, roadmap_service, mock_module_repo, mock_progress_repo, sample_module
    ):
        """Test getting existing module."""
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_module(sample_module.id)

        assert result is not None
        assert result.title == "Getting Started"
        assert len(result.lessons) == 1

    @pytest.mark.asyncio
    async def test_get_path_modules_empty(self, roadmap_service, mock_module_repo):
        """Test getting modules for path with no modules."""
        mock_module_repo.get_by_path_id.return_value = []

        result = await roadmap_service.get_path_modules(uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_get_path_modules_with_modules(
        self, roadmap_service, mock_module_repo, mock_progress_repo, sample_module
    ):
        """Test getting modules for path with modules."""
        mock_module_repo.get_by_path_id.return_value = [sample_module]
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_path_modules(uuid4())

        assert len(result) == 1
        assert result[0].title == "Getting Started"


class TestRoadmapServiceLessonMethods:
    """Tests for lesson-related methods."""

    @pytest.mark.asyncio
    async def test_get_lesson_not_found(self, roadmap_service, mock_lesson_repo):
        """Test getting non-existent lesson."""
        mock_lesson_repo.get_by_id.return_value = None

        result = await roadmap_service.get_lesson(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_lesson_found(
        self, roadmap_service, mock_lesson_repo, mock_progress_repo, sample_lesson
    ):
        """Test getting existing lesson."""
        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_lesson(sample_lesson.id)

        assert result is not None
        assert result.title == "Hello World"
        assert result.lesson_type == LessonType.CONCEPT

    @pytest.mark.asyncio
    async def test_get_lesson_with_progress(
        self, roadmap_service, mock_lesson_repo, mock_progress_repo, sample_lesson
    ):
        """Test getting lesson includes user progress."""
        user_id = uuid4()
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=sample_lesson.id,
            status=ProgressStatus.COMPLETED,
            score=95,
        )
        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = progress

        result = await roadmap_service.get_lesson(sample_lesson.id, user_id=user_id)

        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 95

    @pytest.mark.asyncio
    async def test_get_module_lessons_empty(self, roadmap_service, mock_lesson_repo):
        """Test getting lessons for module with no lessons."""
        mock_lesson_repo.get_by_module_id.return_value = []

        result = await roadmap_service.get_module_lessons(uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_get_module_lessons_with_lessons(
        self, roadmap_service, mock_lesson_repo, mock_progress_repo, sample_lesson
    ):
        """Test getting lessons for module with lessons."""
        mock_lesson_repo.get_by_module_id.return_value = [sample_lesson]
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_module_lessons(uuid4())

        assert len(result) == 1
        assert result[0].title == "Hello World"


class TestRoadmapServiceProgressMethods:
    """Tests for progress-related methods."""

    @pytest.mark.asyncio
    async def test_get_user_progress_no_paths(
        self, roadmap_service, mock_path_repo, mock_progress_repo
    ):
        """Test getting user progress with no paths."""
        user_id = uuid4()
        mock_path_repo.list_all.return_value = []
        mock_progress_repo.get_next_lesson.return_value = None

        result = await roadmap_service.get_user_progress(user_id)

        assert result.total_paths == 0
        assert result.completed_paths == 0
        assert result.total_lessons == 0
        assert result.completed_lessons == 0

    @pytest.mark.asyncio
    async def test_get_user_progress_with_completed_path(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test getting user progress with completed path."""
        user_id = uuid4()
        progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=1,
            completed_lessons=1,
            status=ProgressStatus.COMPLETED,
        )
        mock_path_repo.list_all.return_value = [sample_path]
        mock_progress_repo.get_path_progress.return_value = progress
        mock_progress_repo.get_next_lesson.return_value = None

        result = await roadmap_service.get_user_progress(user_id)

        assert result.total_paths == 1
        assert result.completed_paths == 1
        assert result.completed_lessons == 1

    @pytest.mark.asyncio
    async def test_get_user_progress_with_in_progress_path(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test getting user progress with in-progress path."""
        user_id = uuid4()
        progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=2,
            completed_lessons=1,
            status=ProgressStatus.IN_PROGRESS,
        )
        mock_path_repo.list_all.return_value = [sample_path]
        mock_progress_repo.get_path_progress.return_value = progress
        mock_progress_repo.get_next_lesson.return_value = None

        result = await roadmap_service.get_user_progress(user_id)

        assert result.in_progress_paths == 1
        assert result.completed_lessons == 1
        assert result.current_path is not None

    @pytest.mark.asyncio
    async def test_get_path_progress_not_started(
        self, roadmap_service, mock_progress_repo, mock_path_repo, sample_path
    ):
        """Test getting path progress when not started."""
        user_id = uuid4()
        mock_progress_repo.get_path_progress.return_value = None
        mock_path_repo.get_by_id.return_value = sample_path

        result = await roadmap_service.get_path_progress(user_id, sample_path.id)

        assert result is not None
        assert result.status == ProgressStatus.NOT_STARTED
        assert result.completed_lessons == 0

    @pytest.mark.asyncio
    async def test_get_path_progress_not_found(
        self, roadmap_service, mock_progress_repo, mock_path_repo
    ):
        """Test getting path progress for non-existent path."""
        user_id = uuid4()
        path_id = uuid4()
        mock_progress_repo.get_path_progress.return_value = None
        mock_path_repo.get_by_id.return_value = None

        result = await roadmap_service.get_path_progress(user_id, path_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_path_progress_with_progress(
        self, roadmap_service, mock_progress_repo, sample_path
    ):
        """Test getting path progress with existing progress."""
        user_id = uuid4()
        progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=10,
            completed_lessons=5,
            status=ProgressStatus.IN_PROGRESS,
        )
        progress.start()
        mock_progress_repo.get_path_progress.return_value = progress

        result = await roadmap_service.get_path_progress(user_id, sample_path.id)

        assert result.status == ProgressStatus.IN_PROGRESS
        assert result.completed_lessons == 5
        assert result.completion_rate == 50.0

    @pytest.mark.asyncio
    async def test_start_path_not_found(
        self, roadmap_service, mock_path_repo
    ):
        """Test starting non-existent path raises error."""
        mock_path_repo.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Path not found"):
            await roadmap_service.start_path(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_start_path_already_started(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test starting already started path returns existing progress."""
        user_id = uuid4()
        existing_progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=1,
            status=ProgressStatus.IN_PROGRESS,
        )
        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = existing_progress

        result = await roadmap_service.start_path(user_id, sample_path.id)

        assert result.status == ProgressStatus.IN_PROGRESS
        mock_progress_repo.save_path_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_path_new(
        self,
        roadmap_service,
        mock_path_repo,
        mock_progress_repo,
        mock_xp_service,
        sample_path,
    ):
        """Test starting new path creates progress and awards XP."""
        user_id = uuid4()
        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None

        result = await roadmap_service.start_path(user_id, sample_path.id)

        assert result.status == ProgressStatus.IN_PROGRESS
        assert result.started_at is not None
        mock_progress_repo.save_path_progress.assert_called_once()
        mock_xp_service.add_xp.assert_called_once_with(user_id, "path_started")

    @pytest.mark.asyncio
    async def test_complete_lesson_not_found(self, roadmap_service, mock_lesson_repo):
        """Test completing non-existent lesson raises error."""
        mock_lesson_repo.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Lesson not found"):
            await roadmap_service.complete_lesson(
                uuid4(), uuid4(), CompleteLessonRequest()
            )

    @pytest.mark.asyncio
    async def test_complete_lesson_new(
        self,
        roadmap_service,
        mock_lesson_repo,
        mock_progress_repo,
        mock_module_repo,
        mock_xp_service,
        sample_lesson,
        sample_module,
    ):
        """Test completing lesson for first time."""
        user_id = uuid4()
        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_path_progress.return_value = None

        request = CompleteLessonRequest(score=100)
        result = await roadmap_service.complete_lesson(
            user_id, sample_lesson.id, request
        )

        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 100
        mock_progress_repo.save_lesson_progress.assert_called_once()
        mock_xp_service.add_xp.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_lesson_already_completed(
        self,
        roadmap_service,
        mock_lesson_repo,
        mock_progress_repo,
        mock_module_repo,
        mock_xp_service,
        sample_lesson,
        sample_module,
    ):
        """Test completing already completed lesson does not award XP again."""
        user_id = uuid4()
        existing_progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=sample_lesson.id,
            status=ProgressStatus.COMPLETED,
        )
        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = existing_progress
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_path_progress.return_value = None

        result = await roadmap_service.complete_lesson(
            user_id, sample_lesson.id, CompleteLessonRequest()
        )

        assert result.status == ProgressStatus.COMPLETED
        # XP should not be awarded again
        mock_xp_service.add_xp.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_lesson_completes_path(
        self,
        roadmap_service,
        mock_lesson_repo,
        mock_progress_repo,
        mock_module_repo,
        mock_path_repo,
        mock_xp_service,
        sample_lesson,
        sample_module,
        sample_path,
    ):
        """Test completing last lesson completes the path."""
        user_id = uuid4()
        path_progress = UserPathProgress(
            user_id=user_id,
            path_id=sample_path.id,
            total_lessons=1,
            completed_lessons=0,
            status=ProgressStatus.IN_PROGRESS,
        )

        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_path_progress.return_value = path_progress
        mock_progress_repo.get_completed_lesson_count.return_value = 1
        mock_path_repo.get_by_id.return_value = sample_path

        result = await roadmap_service.complete_lesson(
            user_id, sample_lesson.id, CompleteLessonRequest()
        )

        assert result.status == ProgressStatus.COMPLETED
        # Should award lesson XP and path completion XP
        assert mock_xp_service.add_xp.call_count == 2
        mock_xp_service.set_path_level_completed.assert_called_once_with(
            user_id, PathLevel.BEGINNER.value
        )

    @pytest.mark.asyncio
    async def test_get_next_lesson_none(
        self, roadmap_service, mock_progress_repo
    ):
        """Test getting next lesson when none available."""
        mock_progress_repo.get_next_lesson.return_value = None

        result = await roadmap_service.get_next_lesson(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_next_lesson_found(
        self, roadmap_service, mock_progress_repo, sample_lesson
    ):
        """Test getting next lesson when available."""
        user_id = uuid4()
        mock_progress_repo.get_next_lesson.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service.get_next_lesson(user_id)

        assert result is not None
        assert result.title == "Hello World"


class TestRoadmapServiceHelperMethods:
    """Tests for helper methods (response conversion)."""

    @pytest.mark.asyncio
    async def test_path_to_response_without_user(
        self, roadmap_service, mock_progress_repo, sample_path
    ):
        """Test converting path to response without user."""
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service._path_to_response(sample_path)

        assert result.id == sample_path.id
        assert result.title == "Python Basics"
        assert result.status == ProgressStatus.NOT_STARTED
        assert result.modules == []  # Not included by default

    @pytest.mark.asyncio
    async def test_path_to_response_with_modules(
        self, roadmap_service, mock_progress_repo, sample_path
    ):
        """Test converting path to response with modules included."""
        mock_progress_repo.get_path_progress.return_value = None
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await roadmap_service._path_to_response(
            sample_path, include_modules=True
        )

        assert len(result.modules) == 1
        assert result.modules[0].title == "Getting Started"

    @pytest.mark.asyncio
    async def test_module_to_response_calculates_completion(
        self, roadmap_service, mock_progress_repo, sample_module
    ):
        """Test module response calculates completion rate."""
        user_id = uuid4()
        completed_progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=sample_module.lessons[0].id,
            status=ProgressStatus.COMPLETED,
        )
        mock_progress_repo.get_lesson_progress.return_value = completed_progress

        result = await roadmap_service._module_to_response(
            sample_module, user_id, include_lessons=True
        )

        assert result.completed_lessons == 1
        assert result.completion_rate == 100.0

    @pytest.mark.asyncio
    async def test_lesson_to_response_without_user(
        self, roadmap_service, sample_lesson
    ):
        """Test converting lesson to response without user."""
        result = await roadmap_service._lesson_to_response(sample_lesson)

        assert result.id == sample_lesson.id
        assert result.title == "Hello World"
        assert result.status is None
        assert result.score is None

    @pytest.mark.asyncio
    async def test_lesson_to_response_with_user_progress(
        self, roadmap_service, mock_progress_repo, sample_lesson
    ):
        """Test converting lesson to response with user progress."""
        user_id = uuid4()
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=sample_lesson.id,
            status=ProgressStatus.COMPLETED,
            score=95,
        )
        progress.complete(score=95)
        mock_progress_repo.get_lesson_progress.return_value = progress

        result = await roadmap_service._lesson_to_response(sample_lesson, user_id)

        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 95


class TestRoadmapServiceWithoutXPService:
    """Tests for service behavior without XP service."""

    @pytest.fixture
    def service_without_xp(
        self,
        mock_path_repo,
        mock_module_repo,
        mock_lesson_repo,
        mock_progress_repo,
    ):
        """Create service without XP service."""
        return RoadmapService(
            path_repo=mock_path_repo,
            module_repo=mock_module_repo,
            lesson_repo=mock_lesson_repo,
            progress_repo=mock_progress_repo,
            xp_service=None,
        )

    @pytest.mark.asyncio
    async def test_start_path_without_xp_service(
        self,
        service_without_xp,
        mock_path_repo,
        mock_progress_repo,
        sample_path,
    ):
        """Test starting path works without XP service."""
        user_id = uuid4()
        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None

        result = await service_without_xp.start_path(user_id, sample_path.id)

        assert result.status == ProgressStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_complete_lesson_without_xp_service(
        self,
        service_without_xp,
        mock_lesson_repo,
        mock_progress_repo,
        mock_module_repo,
        sample_lesson,
        sample_module,
    ):
        """Test completing lesson works without XP service."""
        user_id = uuid4()
        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_path_progress.return_value = None

        result = await service_without_xp.complete_lesson(
            user_id, sample_lesson.id, CompleteLessonRequest(score=100)
        )

        assert result.status == ProgressStatus.COMPLETED
        assert result.score == 100


# ============== Route Tests ==============


class TestRoadmapRoutesUnit:
    """Unit tests for roadmap routes."""

    def test_router_has_expected_routes(self):
        """Test that router has all expected routes configured."""
        from code_tutor.roadmap.interface.routes import router

        route_paths = [r.path for r in router.routes]

        expected_paths = [
            "/roadmap/paths",
            "/roadmap/paths/{path_id}",
            "/roadmap/paths/level/{level}",
            "/roadmap/paths/{path_id}/modules",
            "/roadmap/modules/{module_id}",
            "/roadmap/modules/{module_id}/lessons",
            "/roadmap/lessons/{lesson_id}",
            "/roadmap/progress",
            "/roadmap/progress/paths/{path_id}",
            "/roadmap/paths/{path_id}/start",
            "/roadmap/lessons/{lesson_id}/complete",
            "/roadmap/next-lesson",
        ]

        for path in expected_paths:
            assert path in route_paths, f"Missing route: {path}"

    def test_router_prefix(self):
        """Test router has correct prefix."""
        from code_tutor.roadmap.interface.routes import router
        assert router.prefix == "/roadmap"

    def test_router_tags(self):
        """Test router has correct tags."""
        from code_tutor.roadmap.interface.routes import router
        assert "Roadmap" in router.tags


class TestGetRoadmapService:
    """Tests for get_roadmap_service dependency."""

    def test_get_roadmap_service_returns_service(self):
        """Test that get_roadmap_service returns a RoadmapService instance."""
        from code_tutor.roadmap.interface.routes import get_roadmap_service
        from code_tutor.roadmap.application.services import RoadmapService
        from code_tutor.gamification.application.services import XPService

        mock_db = MagicMock()
        mock_xp_service = MagicMock(spec=XPService)
        service = get_roadmap_service(mock_db, mock_xp_service)

        assert isinstance(service, RoadmapService)


class TestGetXpServiceRoute:
    """Tests for get_xp_service dependency."""

    def test_get_xp_service_returns_service(self):
        """Test that get_xp_service returns an XPService instance."""
        from code_tutor.roadmap.interface.routes import get_xp_service
        from code_tutor.gamification.application.services import XPService

        mock_db = MagicMock()
        service = get_xp_service(mock_db)

        assert isinstance(service, XPService)


class TestListPathsRoute:
    """Tests for list_paths route."""

    @pytest.mark.asyncio
    async def test_list_paths_success(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test list_paths route success."""
        from code_tutor.roadmap.interface.routes import list_paths
        from code_tutor.roadmap.application.dto import LearningPathListResponse

        mock_path_repo.list_all.return_value = [sample_path]
        mock_progress_repo.get_path_progress.return_value = None

        result = await list_paths(
            current_user=None,
            service=roadmap_service,
        )

        assert isinstance(result, LearningPathListResponse)
        assert result.total == 1


class TestGetPathRoute:
    """Tests for get_path route."""

    @pytest.mark.asyncio
    async def test_get_path_found(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test get_path route when found."""
        from code_tutor.roadmap.interface.routes import get_path
        from code_tutor.roadmap.application.dto import LearningPathResponse

        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_path(
            path_id=sample_path.id,
            current_user=None,
            service=roadmap_service,
        )

        assert isinstance(result, LearningPathResponse)
        assert result.title == "Python Basics"

    @pytest.mark.asyncio
    async def test_get_path_not_found(self, roadmap_service, mock_path_repo):
        """Test get_path route when not found."""
        from code_tutor.roadmap.interface.routes import get_path
        from fastapi import HTTPException

        mock_path_repo.get_by_id.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_path(
                path_id=uuid4(),
                current_user=None,
                service=roadmap_service,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Learning path not found"


class TestGetPathByLevelRoute:
    """Tests for get_path_by_level route."""

    @pytest.mark.asyncio
    async def test_get_path_by_level_found(
        self, roadmap_service, mock_path_repo, mock_progress_repo, sample_path
    ):
        """Test get_path_by_level route when found."""
        from code_tutor.roadmap.interface.routes import get_path_by_level
        from code_tutor.roadmap.application.dto import LearningPathResponse

        mock_path_repo.get_by_level.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_path_by_level(
            level=PathLevel.BEGINNER,
            current_user=None,
            service=roadmap_service,
        )

        assert isinstance(result, LearningPathResponse)
        assert result.level == PathLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_get_path_by_level_not_found(self, roadmap_service, mock_path_repo):
        """Test get_path_by_level route when not found."""
        from code_tutor.roadmap.interface.routes import get_path_by_level
        from fastapi import HTTPException

        mock_path_repo.get_by_level.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_path_by_level(
                level=PathLevel.ADVANCED,
                current_user=None,
                service=roadmap_service,
            )

        assert exc_info.value.status_code == 404
        assert "advanced" in exc_info.value.detail


class TestGetPathModulesRoute:
    """Tests for get_path_modules route."""

    @pytest.mark.asyncio
    async def test_get_path_modules_success(
        self, roadmap_service, mock_module_repo, mock_progress_repo, sample_module
    ):
        """Test get_path_modules route success."""
        from code_tutor.roadmap.interface.routes import get_path_modules

        mock_module_repo.get_by_path_id.return_value = [sample_module]
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_path_modules(
            path_id=uuid4(),
            current_user=None,
            service=roadmap_service,
        )

        assert len(result) == 1
        assert result[0].title == "Getting Started"


class TestGetModuleRoute:
    """Tests for get_module route."""

    @pytest.mark.asyncio
    async def test_get_module_found(
        self, roadmap_service, mock_module_repo, mock_progress_repo, sample_module
    ):
        """Test get_module route when found."""
        from code_tutor.roadmap.interface.routes import get_module
        from code_tutor.roadmap.application.dto import ModuleResponse

        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_module(
            module_id=sample_module.id,
            current_user=None,
            service=roadmap_service,
        )

        assert isinstance(result, ModuleResponse)
        assert result.title == "Getting Started"

    @pytest.mark.asyncio
    async def test_get_module_not_found(self, roadmap_service, mock_module_repo):
        """Test get_module route when not found."""
        from code_tutor.roadmap.interface.routes import get_module
        from fastapi import HTTPException

        mock_module_repo.get_by_id.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_module(
                module_id=uuid4(),
                current_user=None,
                service=roadmap_service,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Module not found"


class TestGetModuleLessonsRoute:
    """Tests for get_module_lessons route."""

    @pytest.mark.asyncio
    async def test_get_module_lessons_success(
        self, roadmap_service, mock_lesson_repo, mock_progress_repo, sample_lesson
    ):
        """Test get_module_lessons route success."""
        from code_tutor.roadmap.interface.routes import get_module_lessons

        mock_lesson_repo.get_by_module_id.return_value = [sample_lesson]
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_module_lessons(
            module_id=uuid4(),
            current_user=None,
            service=roadmap_service,
        )

        assert len(result) == 1
        assert result[0].title == "Hello World"


class TestGetLessonRoute:
    """Tests for get_lesson route."""

    @pytest.mark.asyncio
    async def test_get_lesson_found(
        self, roadmap_service, mock_lesson_repo, mock_progress_repo, sample_lesson
    ):
        """Test get_lesson route when found."""
        from code_tutor.roadmap.interface.routes import get_lesson
        from code_tutor.roadmap.application.dto import LessonResponse

        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None

        result = await get_lesson(
            lesson_id=sample_lesson.id,
            current_user=None,
            service=roadmap_service,
        )

        assert isinstance(result, LessonResponse)
        assert result.title == "Hello World"

    @pytest.mark.asyncio
    async def test_get_lesson_not_found(self, roadmap_service, mock_lesson_repo):
        """Test get_lesson route when not found."""
        from code_tutor.roadmap.interface.routes import get_lesson
        from fastapi import HTTPException

        mock_lesson_repo.get_by_id.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_lesson(
                lesson_id=uuid4(),
                current_user=None,
                service=roadmap_service,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Lesson not found"


class TestGetUserProgressRoute:
    """Tests for get_user_progress route."""

    @pytest.mark.asyncio
    async def test_get_user_progress_success(
        self, roadmap_service, mock_path_repo, mock_progress_repo
    ):
        """Test get_user_progress route success."""
        from code_tutor.roadmap.interface.routes import get_user_progress
        from code_tutor.roadmap.application.dto import UserProgressResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_path_repo.list_all.return_value = []
        mock_progress_repo.get_next_lesson.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        result = await get_user_progress(
            current_user=mock_user,
            service=roadmap_service,
        )

        assert isinstance(result, UserProgressResponse)
        assert result.total_paths == 0


class TestGetPathProgressRoute:
    """Tests for get_path_progress route."""

    @pytest.mark.asyncio
    async def test_get_path_progress_found(
        self, roadmap_service, mock_progress_repo, mock_path_repo, sample_path
    ):
        """Test get_path_progress route when found."""
        from code_tutor.roadmap.interface.routes import get_path_progress
        from code_tutor.roadmap.application.dto import PathProgressResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_progress_repo.get_path_progress.return_value = None
        mock_path_repo.get_by_id.return_value = sample_path

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        result = await get_path_progress(
            path_id=sample_path.id,
            current_user=mock_user,
            service=roadmap_service,
        )

        assert isinstance(result, PathProgressResponse)

    @pytest.mark.asyncio
    async def test_get_path_progress_not_found(
        self, roadmap_service, mock_progress_repo, mock_path_repo
    ):
        """Test get_path_progress route when path not found."""
        from code_tutor.roadmap.interface.routes import get_path_progress
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_progress_repo.get_path_progress.return_value = None
        mock_path_repo.get_by_id.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await get_path_progress(
                path_id=uuid4(),
                current_user=mock_user,
                service=roadmap_service,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Path not found"


class TestStartPathRoute:
    """Tests for start_path route."""

    @pytest.mark.asyncio
    async def test_start_path_success(
        self, roadmap_service, mock_path_repo, mock_progress_repo, mock_xp_service, sample_path
    ):
        """Test start_path route success."""
        from code_tutor.roadmap.interface.routes import start_path
        from code_tutor.roadmap.application.dto import PathProgressResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_path_repo.get_by_id.return_value = sample_path
        mock_progress_repo.get_path_progress.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        mock_db = AsyncMock()

        result = await start_path(
            path_id=sample_path.id,
            current_user=mock_user,
            service=roadmap_service,
            db=mock_db,
        )

        assert isinstance(result, PathProgressResponse)
        assert result.status == ProgressStatus.IN_PROGRESS
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_path_not_found(
        self, roadmap_service, mock_path_repo
    ):
        """Test start_path route when path not found."""
        from code_tutor.roadmap.interface.routes import start_path
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_path_repo.get_by_id.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        mock_db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await start_path(
                path_id=uuid4(),
                current_user=mock_user,
                service=roadmap_service,
                db=mock_db,
            )

        assert exc_info.value.status_code == 404


class TestCompleteLessonRoute:
    """Tests for complete_lesson route."""

    @pytest.mark.asyncio
    async def test_complete_lesson_route_success(
        self,
        roadmap_service,
        mock_lesson_repo,
        mock_progress_repo,
        mock_module_repo,
        mock_xp_service,
        sample_lesson,
        sample_module,
    ):
        """Test complete_lesson route success."""
        from code_tutor.roadmap.interface.routes import complete_lesson
        from code_tutor.roadmap.application.dto import LessonProgressResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_lesson_repo.get_by_id.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None
        mock_module_repo.get_by_id.return_value = sample_module
        mock_progress_repo.get_path_progress.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        mock_db = AsyncMock()
        request = CompleteLessonRequest(score=100)

        result = await complete_lesson(
            lesson_id=sample_lesson.id,
            request=request,
            current_user=mock_user,
            service=roadmap_service,
            db=mock_db,
        )

        assert isinstance(result, LessonProgressResponse)
        assert result.status == ProgressStatus.COMPLETED
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_lesson_route_not_found(
        self, roadmap_service, mock_lesson_repo
    ):
        """Test complete_lesson route when lesson not found."""
        from code_tutor.roadmap.interface.routes import complete_lesson
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_lesson_repo.get_by_id.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        mock_db = AsyncMock()
        request = CompleteLessonRequest()

        with pytest.raises(HTTPException) as exc_info:
            await complete_lesson(
                lesson_id=uuid4(),
                request=request,
                current_user=mock_user,
                service=roadmap_service,
                db=mock_db,
            )

        assert exc_info.value.status_code == 404


class TestGetNextLessonRoute:
    """Tests for get_next_lesson route."""

    @pytest.mark.asyncio
    async def test_get_next_lesson_route_found(
        self, roadmap_service, mock_progress_repo, sample_lesson
    ):
        """Test get_next_lesson route when found."""
        from code_tutor.roadmap.interface.routes import get_next_lesson
        from code_tutor.roadmap.application.dto import LessonResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_progress_repo.get_next_lesson.return_value = sample_lesson
        mock_progress_repo.get_lesson_progress.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        result = await get_next_lesson(
            path_id=None,
            current_user=mock_user,
            service=roadmap_service,
        )

        assert isinstance(result, LessonResponse)
        assert result.title == "Hello World"

    @pytest.mark.asyncio
    async def test_get_next_lesson_route_none(
        self, roadmap_service, mock_progress_repo
    ):
        """Test get_next_lesson route when none available."""
        from code_tutor.roadmap.interface.routes import get_next_lesson
        from code_tutor.identity.application.dto import UserResponse

        mock_progress_repo.get_next_lesson.return_value = None

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        result = await get_next_lesson(
            path_id=None,
            current_user=mock_user,
            service=roadmap_service,
        )

        assert result is None
