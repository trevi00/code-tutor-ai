"""Tests for Roadmap Domain Entities and Value Objects."""

from uuid import uuid4

import pytest

from code_tutor.roadmap.domain.entities import (
    Lesson,
    Module,
    LearningPath,
    UserPathProgress,
    UserLessonProgress,
)
from code_tutor.roadmap.domain.value_objects import (
    PathLevel,
    LessonType,
    ProgressStatus,
)


class TestPathLevel:
    """Tests for PathLevel enum."""

    def test_path_level_values(self):
        """Test all path level values exist."""
        assert PathLevel.BEGINNER.value == "beginner"
        assert PathLevel.ELEMENTARY.value == "elementary"
        assert PathLevel.INTERMEDIATE.value == "intermediate"
        assert PathLevel.ADVANCED.value == "advanced"

    def test_path_level_display_names(self):
        """Test Korean display names."""
        assert PathLevel.BEGINNER.display_name == "입문"
        assert PathLevel.ELEMENTARY.display_name == "초급"
        assert PathLevel.INTERMEDIATE.display_name == "중급"
        assert PathLevel.ADVANCED.display_name == "고급"

    def test_path_level_order(self):
        """Test ordering of path levels."""
        assert PathLevel.BEGINNER.order == 1
        assert PathLevel.ELEMENTARY.order == 2
        assert PathLevel.INTERMEDIATE.order == 3
        assert PathLevel.ADVANCED.order == 4


class TestLessonType:
    """Tests for LessonType enum."""

    def test_lesson_type_values(self):
        """Test all lesson type values exist."""
        assert LessonType.CONCEPT.value == "concept"
        assert LessonType.PROBLEM.value == "problem"
        assert LessonType.TYPING.value == "typing"
        assert LessonType.PATTERN.value == "pattern"
        assert LessonType.QUIZ.value == "quiz"

    def test_lesson_type_display_names(self):
        """Test Korean display names."""
        assert LessonType.CONCEPT.display_name == "개념"
        assert LessonType.PROBLEM.display_name == "문제"
        assert LessonType.TYPING.display_name == "타이핑"
        assert LessonType.PATTERN.display_name == "패턴"
        assert LessonType.QUIZ.display_name == "퀴즈"


class TestProgressStatus:
    """Tests for ProgressStatus enum."""

    def test_progress_status_values(self):
        """Test all progress status values exist."""
        assert ProgressStatus.NOT_STARTED.value == "not_started"
        assert ProgressStatus.IN_PROGRESS.value == "in_progress"
        assert ProgressStatus.COMPLETED.value == "completed"

    def test_progress_status_display_names(self):
        """Test Korean display names."""
        assert ProgressStatus.NOT_STARTED.display_name == "시작 전"
        assert ProgressStatus.IN_PROGRESS.display_name == "진행 중"
        assert ProgressStatus.COMPLETED.display_name == "완료"


class TestLesson:
    """Tests for Lesson entity."""

    def test_lesson_creation(self):
        """Test creating a lesson."""
        lesson = Lesson(
            title="Hello World",
            description="Learn to print Hello World",
            lesson_type=LessonType.CONCEPT,
            content="# Hello World\nprint('Hello')",
            order=1,
            xp_reward=10,
            estimated_minutes=5,
        )
        assert lesson.title == "Hello World"
        assert lesson.lesson_type == LessonType.CONCEPT
        assert lesson.xp_reward == 10
        assert lesson.estimated_minutes == 5

    def test_lesson_default_values(self):
        """Test lesson default values."""
        lesson = Lesson()
        assert lesson.title == ""
        assert lesson.lesson_type == LessonType.CONCEPT
        assert lesson.xp_reward == 10
        assert lesson.estimated_minutes == 10
        assert lesson.order == 0

    def test_lesson_repr(self):
        """Test lesson string representation."""
        lesson = Lesson(title="Test Lesson", lesson_type=LessonType.QUIZ)
        repr_str = repr(lesson)
        assert "Test Lesson" in repr_str
        assert "quiz" in repr_str.lower()


class TestModule:
    """Tests for Module entity."""

    def test_module_creation(self):
        """Test creating a module."""
        module = Module(
            title="Getting Started",
            description="Introduction to Python",
            order=1,
        )
        assert module.title == "Getting Started"
        assert module.order == 1
        assert module.lessons == []

    def test_module_lesson_count(self):
        """Test lesson count property."""
        module = Module(title="Test Module")
        assert module.lesson_count == 0

        lesson1 = Lesson(title="Lesson 1")
        lesson2 = Lesson(title="Lesson 2")
        module.lessons = [lesson1, lesson2]
        assert module.lesson_count == 2

    def test_module_total_xp(self):
        """Test total XP calculation."""
        module = Module(title="Test Module")
        lesson1 = Lesson(title="Lesson 1", xp_reward=10)
        lesson2 = Lesson(title="Lesson 2", xp_reward=20)
        module.lessons = [lesson1, lesson2]
        assert module.total_xp == 30

    def test_module_estimated_minutes(self):
        """Test estimated minutes calculation."""
        module = Module(title="Test Module")
        lesson1 = Lesson(title="Lesson 1", estimated_minutes=5)
        lesson2 = Lesson(title="Lesson 2", estimated_minutes=10)
        module.lessons = [lesson1, lesson2]
        assert module.estimated_minutes == 15

    def test_module_add_lesson(self):
        """Test adding a lesson to module."""
        module = Module(title="Test Module")
        module_id = module.id
        lesson = Lesson(title="New Lesson")

        module.add_lesson(lesson)

        assert lesson in module.lessons
        assert lesson.module_id == module_id

    def test_module_repr(self):
        """Test module string representation."""
        module = Module(title="Test Module")
        module.lessons = [Lesson(), Lesson()]
        repr_str = repr(module)
        assert "Test Module" in repr_str
        assert "2" in repr_str


class TestLearningPath:
    """Tests for LearningPath aggregate root."""

    def test_learning_path_creation(self):
        """Test creating a learning path."""
        path = LearningPath(
            level=PathLevel.BEGINNER,
            title="Python Basics",
            description="Learn Python from scratch",
            icon="🐍",
            order=1,
            estimated_hours=20,
        )
        assert path.title == "Python Basics"
        assert path.level == PathLevel.BEGINNER
        assert path.icon == "🐍"
        assert path.is_published is True

    def test_learning_path_module_count(self):
        """Test module count property."""
        path = LearningPath(title="Test Path")
        assert path.module_count == 0

        module1 = Module(title="Module 1")
        module2 = Module(title="Module 2")
        path.modules = [module1, module2]
        assert path.module_count == 2

    def test_learning_path_lesson_count(self):
        """Test lesson count across all modules."""
        path = LearningPath(title="Test Path")

        module1 = Module(title="Module 1")
        module1.lessons = [Lesson(), Lesson()]

        module2 = Module(title="Module 2")
        module2.lessons = [Lesson(), Lesson(), Lesson()]

        path.modules = [module1, module2]
        assert path.lesson_count == 5

    def test_learning_path_total_xp(self):
        """Test total XP across all modules."""
        path = LearningPath(title="Test Path")

        module1 = Module(title="Module 1")
        module1.lessons = [Lesson(xp_reward=10), Lesson(xp_reward=20)]

        module2 = Module(title="Module 2")
        module2.lessons = [Lesson(xp_reward=15)]

        path.modules = [module1, module2]
        assert path.total_xp == 45

    def test_learning_path_add_module(self):
        """Test adding a module to path."""
        path = LearningPath(title="Test Path")
        path_id = path.id
        module = Module(title="New Module")

        path.add_module(module)

        assert module in path.modules
        assert module.path_id == path_id

    def test_learning_path_prerequisites(self):
        """Test prerequisites list."""
        prereq_id = uuid4()
        path = LearningPath(
            title="Advanced Path",
            prerequisites=[prereq_id],
        )
        assert prereq_id in path.prerequisites

    def test_learning_path_repr(self):
        """Test learning path string representation."""
        path = LearningPath(title="Test Path", level=PathLevel.INTERMEDIATE)
        repr_str = repr(path)
        assert "Test Path" in repr_str
        assert "intermediate" in repr_str.lower()


class TestUserPathProgress:
    """Tests for UserPathProgress entity."""

    def test_user_path_progress_creation(self):
        """Test creating user path progress."""
        user_id = uuid4()
        path_id = uuid4()
        progress = UserPathProgress(
            user_id=user_id,
            path_id=path_id,
            total_lessons=25,
        )
        assert progress.user_id == user_id
        assert progress.path_id == path_id
        assert progress.status == ProgressStatus.NOT_STARTED
        assert progress.completed_lessons == 0
        assert progress.total_lessons == 25

    def test_user_path_progress_completion_rate(self):
        """Test completion rate calculation."""
        progress = UserPathProgress(
            completed_lessons=10,
            total_lessons=20,
        )
        assert progress.completion_rate == 50.0

    def test_user_path_progress_completion_rate_zero_total(self):
        """Test completion rate with zero total."""
        progress = UserPathProgress(total_lessons=0)
        assert progress.completion_rate == 0.0

    def test_user_path_progress_start(self):
        """Test starting a path."""
        progress = UserPathProgress()
        assert progress.status == ProgressStatus.NOT_STARTED
        assert progress.started_at is None

        progress.start()

        assert progress.status == ProgressStatus.IN_PROGRESS
        assert progress.started_at is not None

    def test_user_path_progress_start_idempotent(self):
        """Test that start is idempotent when already in progress."""
        progress = UserPathProgress()
        progress.start()
        first_start_time = progress.started_at

        # Try to start again
        progress.start()

        # Should not change
        assert progress.started_at == first_start_time

    def test_user_path_progress_complete(self):
        """Test completing a path."""
        progress = UserPathProgress()
        progress.start()

        progress.complete()

        assert progress.status == ProgressStatus.COMPLETED
        assert progress.completed_at is not None

    def test_user_path_progress_update_progress(self):
        """Test updating progress."""
        progress = UserPathProgress()
        progress.start()

        progress.update_progress(completed=5, total=10)

        assert progress.completed_lessons == 5
        assert progress.total_lessons == 10
        assert progress.status == ProgressStatus.IN_PROGRESS

    def test_user_path_progress_update_progress_auto_complete(self):
        """Test that update_progress auto-completes when all lessons done."""
        progress = UserPathProgress()
        progress.start()

        progress.update_progress(completed=10, total=10)

        assert progress.status == ProgressStatus.COMPLETED
        assert progress.completed_at is not None

    def test_user_path_progress_repr(self):
        """Test user path progress string representation."""
        user_id = uuid4()
        path_id = uuid4()
        progress = UserPathProgress(user_id=user_id, path_id=path_id)
        repr_str = repr(progress)
        assert str(user_id) in repr_str or "user=" in repr_str
        assert "not_started" in repr_str.lower()


class TestUserLessonProgress:
    """Tests for UserLessonProgress entity."""

    def test_user_lesson_progress_creation(self):
        """Test creating user lesson progress."""
        user_id = uuid4()
        lesson_id = uuid4()
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
        )
        assert progress.user_id == user_id
        assert progress.lesson_id == lesson_id
        assert progress.status == ProgressStatus.NOT_STARTED
        assert progress.attempts == 0
        assert progress.score is None

    def test_user_lesson_progress_start(self):
        """Test starting a lesson."""
        progress = UserLessonProgress()
        assert progress.status == ProgressStatus.NOT_STARTED
        assert progress.attempts == 0

        progress.start()

        assert progress.status == ProgressStatus.IN_PROGRESS
        assert progress.started_at is not None
        assert progress.attempts == 1

    def test_user_lesson_progress_multiple_attempts(self):
        """Test multiple attempts increment."""
        progress = UserLessonProgress()

        progress.start()
        assert progress.attempts == 1

        # Simulate retry (start again)
        progress.status = ProgressStatus.NOT_STARTED
        progress.start()
        assert progress.attempts == 2

    def test_user_lesson_progress_start_when_already_in_progress(self):
        """Test start() when already in progress (increments attempts only)."""
        progress = UserLessonProgress()
        progress.start()  # First start

        first_started_at = progress.started_at
        assert progress.status == ProgressStatus.IN_PROGRESS
        assert progress.attempts == 1

        # Call start() again when already IN_PROGRESS
        progress.start()

        # Status and started_at should not change
        assert progress.status == ProgressStatus.IN_PROGRESS
        assert progress.started_at == first_started_at
        # But attempts should still increment
        assert progress.attempts == 2

    def test_user_lesson_progress_complete(self):
        """Test completing a lesson."""
        progress = UserLessonProgress()
        progress.start()

        progress.complete()

        assert progress.status == ProgressStatus.COMPLETED
        assert progress.completed_at is not None

    def test_user_lesson_progress_complete_with_score(self):
        """Test completing a lesson with score."""
        progress = UserLessonProgress()
        progress.start()

        progress.complete(score=95)

        assert progress.status == ProgressStatus.COMPLETED
        assert progress.score == 95

    def test_user_lesson_progress_repr(self):
        """Test user lesson progress string representation."""
        user_id = uuid4()
        lesson_id = uuid4()
        progress = UserLessonProgress(user_id=user_id, lesson_id=lesson_id)
        repr_str = repr(progress)
        assert "not_started" in repr_str.lower()


class TestGenerateUuid:
    """Tests for generate_uuid helper function."""

    def test_generate_uuid_returns_string(self):
        """Test generate_uuid returns a valid UUID string."""
        from code_tutor.roadmap.infrastructure.models import generate_uuid

        result = generate_uuid()
        assert isinstance(result, str)
        assert len(result) == 36  # UUID string format
        # Validate it's a proper UUID format (8-4-4-4-12)
        parts = result.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_generate_uuid_is_unique(self):
        """Test generate_uuid returns unique values."""
        from code_tutor.roadmap.infrastructure.models import generate_uuid

        uuids = [generate_uuid() for _ in range(100)]
        assert len(uuids) == len(set(uuids))  # All unique
