"""Repository interfaces for typing practice domain."""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.typing_practice.domain.entities import (
    TypingAttempt,
    TypingExercise,
    UserExerciseProgress,
)
from code_tutor.typing_practice.domain.value_objects import ExerciseCategory


class TypingExerciseRepository(ABC):
    """Repository interface for typing exercises."""

    @abstractmethod
    async def get_by_id(self, exercise_id: UUID) -> TypingExercise | None:
        """Get exercise by ID."""
        pass

    @abstractmethod
    async def list_all(
        self,
        category: ExerciseCategory | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TypingExercise]:
        """List all exercises with optional filtering."""
        pass

    @abstractmethod
    async def save(self, exercise: TypingExercise) -> TypingExercise:
        """Save or update an exercise."""
        pass

    @abstractmethod
    async def delete(self, exercise_id: UUID) -> bool:
        """Delete an exercise."""
        pass

    @abstractmethod
    async def count(self, category: ExerciseCategory | None = None) -> int:
        """Count exercises."""
        pass


class TypingAttemptRepository(ABC):
    """Repository interface for typing attempts."""

    @abstractmethod
    async def get_by_id(self, attempt_id: UUID) -> TypingAttempt | None:
        """Get attempt by ID."""
        pass

    @abstractmethod
    async def list_by_user_and_exercise(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> list[TypingAttempt]:
        """List attempts by user and exercise."""
        pass

    @abstractmethod
    async def list_by_user(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TypingAttempt]:
        """List all attempts by user."""
        pass

    @abstractmethod
    async def save(self, attempt: TypingAttempt) -> TypingAttempt:
        """Save or update an attempt."""
        pass

    @abstractmethod
    async def get_user_progress(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> UserExerciseProgress | None:
        """Get user's progress on an exercise."""
        pass

    @abstractmethod
    async def get_user_stats(self, user_id: UUID) -> dict:
        """Get user's overall typing statistics."""
        pass

    @abstractmethod
    async def get_leaderboard(self, limit: int = 10) -> list[dict]:
        """Get top performers by WPM."""
        pass
