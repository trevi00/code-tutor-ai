"""Repository interfaces for Learning Roadmap domain."""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.roadmap.domain.entities import (
    LearningPath,
    Lesson,
    Module,
    UserLessonProgress,
    UserPathProgress,
)
from code_tutor.roadmap.domain.value_objects import PathLevel


class LearningPathRepository(ABC):
    """Repository interface for LearningPath aggregate."""

    @abstractmethod
    async def get_by_id(self, path_id: UUID) -> LearningPath | None:
        """Get a learning path by ID."""
        pass

    @abstractmethod
    async def get_by_level(self, level: PathLevel) -> LearningPath | None:
        """Get a learning path by level."""
        pass

    @abstractmethod
    async def list_all(self, include_unpublished: bool = False) -> list[LearningPath]:
        """List all learning paths."""
        pass

    @abstractmethod
    async def save(self, path: LearningPath) -> LearningPath:
        """Save a learning path."""
        pass

    @abstractmethod
    async def delete(self, path_id: UUID) -> bool:
        """Delete a learning path."""
        pass


class ModuleRepository(ABC):
    """Repository interface for Module entity."""

    @abstractmethod
    async def get_by_id(self, module_id: UUID) -> Module | None:
        """Get a module by ID."""
        pass

    @abstractmethod
    async def get_by_path_id(self, path_id: UUID) -> list[Module]:
        """Get all modules for a path."""
        pass

    @abstractmethod
    async def save(self, module: Module) -> Module:
        """Save a module."""
        pass


class LessonRepository(ABC):
    """Repository interface for Lesson entity."""

    @abstractmethod
    async def get_by_id(self, lesson_id: UUID) -> Lesson | None:
        """Get a lesson by ID."""
        pass

    @abstractmethod
    async def get_by_module_id(self, module_id: UUID) -> list[Lesson]:
        """Get all lessons for a module."""
        pass

    @abstractmethod
    async def get_by_path_id(self, path_id: UUID) -> list[Lesson]:
        """Get all lessons for a path."""
        pass

    @abstractmethod
    async def save(self, lesson: Lesson) -> Lesson:
        """Save a lesson."""
        pass


class UserProgressRepository(ABC):
    """Repository interface for user progress tracking."""

    @abstractmethod
    async def get_path_progress(
        self, user_id: UUID, path_id: UUID
    ) -> UserPathProgress | None:
        """Get user's progress on a path."""
        pass

    @abstractmethod
    async def get_all_path_progress(self, user_id: UUID) -> list[UserPathProgress]:
        """Get user's progress on all paths."""
        pass

    @abstractmethod
    async def save_path_progress(
        self, progress: UserPathProgress
    ) -> UserPathProgress:
        """Save path progress."""
        pass

    @abstractmethod
    async def get_lesson_progress(
        self, user_id: UUID, lesson_id: UUID
    ) -> UserLessonProgress | None:
        """Get user's progress on a lesson."""
        pass

    @abstractmethod
    async def get_module_lessons_progress(
        self, user_id: UUID, module_id: UUID
    ) -> list[UserLessonProgress]:
        """Get user's progress on all lessons in a module."""
        pass

    @abstractmethod
    async def get_path_lessons_progress(
        self, user_id: UUID, path_id: UUID
    ) -> list[UserLessonProgress]:
        """Get user's progress on all lessons in a path."""
        pass

    @abstractmethod
    async def save_lesson_progress(
        self, progress: UserLessonProgress
    ) -> UserLessonProgress:
        """Save lesson progress."""
        pass

    @abstractmethod
    async def get_completed_lesson_count(
        self, user_id: UUID, path_id: UUID
    ) -> int:
        """Get count of completed lessons in a path."""
        pass

    @abstractmethod
    async def get_next_lesson(
        self, user_id: UUID, path_id: UUID | None = None
    ) -> Lesson | None:
        """Get the next incomplete lesson for user."""
        pass
