"""Application services for typing practice."""

from typing import Optional
from uuid import UUID

from code_tutor.typing_practice.domain.entities import (
    TypingExercise,
    TypingAttempt,
    UserExerciseProgress,
)
from code_tutor.typing_practice.domain.repository import (
    TypingExerciseRepository,
    TypingAttemptRepository,
)
from code_tutor.typing_practice.domain.value_objects import (
    ExerciseCategory,
    Difficulty,
)
from code_tutor.typing_practice.application.dto import (
    CreateExerciseRequest,
    CompleteAttemptRequest,
    TypingExerciseResponse,
    TypingExerciseListResponse,
    TypingAttemptResponse,
    UserProgressResponse,
    UserTypingStatsResponse,
)


class TypingPracticeService:
    """Service for managing typing practice exercises and attempts."""

    def __init__(
        self,
        exercise_repo: TypingExerciseRepository,
        attempt_repo: TypingAttemptRepository,
    ):
        self.exercise_repo = exercise_repo
        self.attempt_repo = attempt_repo

    # ============== Exercise Operations ==============

    async def create_exercise(
        self,
        request: CreateExerciseRequest,
    ) -> TypingExerciseResponse:
        """Create a new typing exercise."""
        exercise = TypingExercise.create(
            title=request.title,
            source_code=request.source_code,
            language=request.language,
            category=request.category,
            difficulty=request.difficulty,
            description=request.description,
            required_completions=request.required_completions,
        )
        saved = await self.exercise_repo.save(exercise)
        return self._to_exercise_response(saved)

    async def get_exercise(
        self,
        exercise_id: UUID,
    ) -> Optional[TypingExerciseResponse]:
        """Get a typing exercise by ID."""
        exercise = await self.exercise_repo.get_by_id(exercise_id)
        if not exercise:
            return None
        return self._to_exercise_response(exercise)

    async def list_exercises(
        self,
        category: Optional[ExerciseCategory] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> TypingExerciseListResponse:
        """List typing exercises with pagination."""
        offset = (page - 1) * page_size
        exercises = await self.exercise_repo.list_all(
            category=category,
            limit=page_size,
            offset=offset,
        )
        total = await self.exercise_repo.count(category=category)

        return TypingExerciseListResponse(
            exercises=[self._to_exercise_response(e) for e in exercises],
            total=total,
            page=page,
            page_size=page_size,
        )

    # ============== Attempt Operations ==============

    async def start_attempt(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> TypingAttemptResponse:
        """Start a new typing attempt."""
        # Get existing attempts to determine attempt number
        existing = await self.attempt_repo.list_by_user_and_exercise(
            user_id=user_id,
            exercise_id=exercise_id,
        )
        completed_count = len([a for a in existing if a.is_completed])
        attempt_number = completed_count + 1

        attempt = TypingAttempt.create(
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=attempt_number,
        )
        saved = await self.attempt_repo.save(attempt)
        return self._to_attempt_response(saved)

    async def complete_attempt(
        self,
        attempt_id: UUID,
        request: CompleteAttemptRequest,
    ) -> Optional[TypingAttemptResponse]:
        """Complete a typing attempt."""
        attempt = await self.attempt_repo.get_by_id(attempt_id)
        if not attempt:
            return None

        attempt.complete(
            user_code=request.user_code,
            accuracy=request.accuracy,
            wpm=request.wpm,
            time_seconds=request.time_seconds,
        )
        saved = await self.attempt_repo.save(attempt)
        return self._to_attempt_response(saved)

    async def get_user_progress(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> Optional[UserProgressResponse]:
        """Get user's progress on an exercise."""
        exercise = await self.exercise_repo.get_by_id(exercise_id)
        if not exercise:
            return None

        attempts = await self.attempt_repo.list_by_user_and_exercise(
            user_id=user_id,
            exercise_id=exercise_id,
        )

        progress = UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=attempts,
            required_completions=exercise.required_completions,
        )

        return UserProgressResponse(
            user_id=progress.user_id,
            exercise_id=progress.exercise_id,
            completed_attempts=progress.completed_attempts,
            required_completions=exercise.required_completions,
            best_accuracy=progress.best_accuracy,
            best_wpm=progress.best_wpm,
            total_time_seconds=progress.total_time_seconds,
            is_mastered=progress.is_mastered,
            attempts=[self._to_attempt_response(a) for a in attempts],
        )

    async def get_user_stats(
        self,
        user_id: UUID,
    ) -> UserTypingStatsResponse:
        """Get user's overall typing statistics."""
        stats = await self.attempt_repo.get_user_stats(user_id)
        return UserTypingStatsResponse(**stats)

    # ============== Helper Methods ==============

    def _to_exercise_response(
        self,
        exercise: TypingExercise,
    ) -> TypingExerciseResponse:
        """Convert exercise entity to response."""
        return TypingExerciseResponse(
            id=exercise.id,
            title=exercise.title,
            source_code=exercise.source_code,
            language=exercise.language,
            category=exercise.category,
            difficulty=exercise.difficulty,
            description=exercise.description,
            required_completions=exercise.required_completions,
            char_count=exercise.char_count,
            line_count=exercise.line_count,
            created_at=exercise.created_at,
        )

    def _to_attempt_response(
        self,
        attempt: TypingAttempt,
    ) -> TypingAttemptResponse:
        """Convert attempt entity to response."""
        return TypingAttemptResponse(
            id=attempt.id,
            user_id=attempt.user_id,
            exercise_id=attempt.exercise_id,
            attempt_number=attempt.attempt_number,
            accuracy=attempt.accuracy,
            wpm=attempt.wpm,
            time_seconds=attempt.time_seconds,
            status=attempt.status,
            started_at=attempt.started_at,
            completed_at=attempt.completed_at,
        )
