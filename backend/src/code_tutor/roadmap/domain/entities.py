"""Domain entities for Learning Roadmap."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from code_tutor.shared.domain.base import AggregateRoot, Entity
from code_tutor.roadmap.domain.value_objects import (
    PathLevel,
    LessonType,
    ProgressStatus,
)


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)


class Lesson(Entity):
    """Individual learning unit within a module."""

    def __init__(
        self,
        id: UUID | None = None,
        module_id: UUID | None = None,
        title: str = "",
        description: str = "",
        lesson_type: LessonType = LessonType.CONCEPT,
        content: str = "",
        content_id: UUID | None = None,
        order: int = 0,
        xp_reward: int = 10,
        estimated_minutes: int = 10,
    ) -> None:
        super().__init__(id)
        self.module_id = module_id
        self.title = title
        self.description = description
        self.lesson_type = lesson_type
        self.content = content
        self.content_id = content_id
        self.order = order
        self.xp_reward = xp_reward
        self.estimated_minutes = estimated_minutes

    def __repr__(self) -> str:
        return f"Lesson(id={self.id}, title='{self.title}', type={self.lesson_type})"


class Module(Entity):
    """Chapter/section within a learning path."""

    def __init__(
        self,
        id: UUID | None = None,
        path_id: UUID | None = None,
        title: str = "",
        description: str = "",
        order: int = 0,
        lessons: list[Lesson] | None = None,
    ) -> None:
        super().__init__(id)
        self.path_id = path_id
        self.title = title
        self.description = description
        self.order = order
        self.lessons = lessons or []

    @property
    def lesson_count(self) -> int:
        return len(self.lessons)

    @property
    def total_xp(self) -> int:
        return sum(lesson.xp_reward for lesson in self.lessons)

    @property
    def estimated_minutes(self) -> int:
        return sum(lesson.estimated_minutes for lesson in self.lessons)

    def add_lesson(self, lesson: Lesson) -> None:
        lesson.module_id = self.id
        self.lessons.append(lesson)
        self._touch()

    def __repr__(self) -> str:
        return f"Module(id={self.id}, title='{self.title}', lessons={len(self.lessons)})"


class LearningPath(AggregateRoot):
    """Complete learning path/course."""

    def __init__(
        self,
        id: UUID | None = None,
        level: PathLevel = PathLevel.BEGINNER,
        title: str = "",
        description: str = "",
        icon: str = "",
        order: int = 0,
        estimated_hours: int = 0,
        prerequisites: list[UUID] | None = None,
        modules: list[Module] | None = None,
        is_published: bool = True,
    ) -> None:
        super().__init__(id)
        self.level = level
        self.title = title
        self.description = description
        self.icon = icon
        self.order = order
        self.estimated_hours = estimated_hours
        self.prerequisites = prerequisites or []
        self.modules = modules or []
        self.is_published = is_published

    @property
    def module_count(self) -> int:
        return len(self.modules)

    @property
    def lesson_count(self) -> int:
        return sum(module.lesson_count for module in self.modules)

    @property
    def total_xp(self) -> int:
        return sum(module.total_xp for module in self.modules)

    def add_module(self, module: Module) -> None:
        module.path_id = self.id
        self.modules.append(module)
        self._touch()

    def __repr__(self) -> str:
        return f"LearningPath(id={self.id}, title='{self.title}', level={self.level})"


class UserPathProgress(Entity):
    """User's progress on a learning path."""

    def __init__(
        self,
        id: UUID | None = None,
        user_id: UUID | None = None,
        path_id: UUID | None = None,
        status: ProgressStatus = ProgressStatus.NOT_STARTED,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        completed_lessons: int = 0,
        total_lessons: int = 0,
    ) -> None:
        super().__init__(id)
        self.user_id = user_id
        self.path_id = path_id
        self.status = status
        self.started_at = started_at
        self.completed_at = completed_at
        self.completed_lessons = completed_lessons
        self.total_lessons = total_lessons

    @property
    def completion_rate(self) -> float:
        if self.total_lessons == 0:
            return 0.0
        return (self.completed_lessons / self.total_lessons) * 100

    def start(self) -> None:
        if self.status == ProgressStatus.NOT_STARTED:
            self.status = ProgressStatus.IN_PROGRESS
            self.started_at = utc_now()
            self._touch()

    def complete(self) -> None:
        self.status = ProgressStatus.COMPLETED
        self.completed_at = utc_now()
        self._touch()

    def update_progress(self, completed: int, total: int) -> None:
        self.completed_lessons = completed
        self.total_lessons = total
        if completed >= total and total > 0:
            self.complete()
        self._touch()

    def __repr__(self) -> str:
        return f"UserPathProgress(user={self.user_id}, path={self.path_id}, status={self.status})"


class UserLessonProgress(Entity):
    """User's progress on a specific lesson."""

    def __init__(
        self,
        id: UUID | None = None,
        user_id: UUID | None = None,
        lesson_id: UUID | None = None,
        status: ProgressStatus = ProgressStatus.NOT_STARTED,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        score: int | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(id)
        self.user_id = user_id
        self.lesson_id = lesson_id
        self.status = status
        self.started_at = started_at
        self.completed_at = completed_at
        self.score = score
        self.attempts = attempts

    def start(self) -> None:
        if self.status == ProgressStatus.NOT_STARTED:
            self.status = ProgressStatus.IN_PROGRESS
            self.started_at = utc_now()
        self.attempts += 1
        self._touch()

    def complete(self, score: int | None = None) -> None:
        self.status = ProgressStatus.COMPLETED
        self.completed_at = utc_now()
        if score is not None:
            self.score = score
        self._touch()

    def __repr__(self) -> str:
        return f"UserLessonProgress(user={self.user_id}, lesson={self.lesson_id}, status={self.status})"
