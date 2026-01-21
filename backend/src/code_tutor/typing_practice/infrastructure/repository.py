"""Repository implementations for typing practice."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.shared.constants import TypingPractice as TypingConstants
from code_tutor.typing_practice.domain.entities import (
    TypingAttempt,
    TypingExercise,
    UserExerciseProgress,
)
from code_tutor.typing_practice.domain.repository import (
    TypingAttemptRepository,
    TypingExerciseRepository,
)
from code_tutor.typing_practice.domain.value_objects import (
    AttemptStatus,
    Difficulty,
    ExerciseCategory,
)
from code_tutor.typing_practice.infrastructure.models import (
    TypingAttemptModel,
    TypingExerciseModel,
)


class SQLAlchemyTypingExerciseRepository(TypingExerciseRepository):
    """SQLAlchemy implementation of typing exercise repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, exercise_id: UUID) -> TypingExercise | None:
        """Get exercise by ID."""
        stmt = select(TypingExerciseModel).where(
            TypingExerciseModel.id == str(exercise_id)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_all(
        self,
        category: ExerciseCategory | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TypingExercise]:
        """List all exercises with optional filtering."""
        stmt = select(TypingExerciseModel).where(
            TypingExerciseModel.is_published == True
        )

        if category:
            stmt = stmt.where(TypingExerciseModel.category == category.value)

        stmt = stmt.order_by(TypingExerciseModel.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def save(self, exercise: TypingExercise) -> TypingExercise:
        """Save or update an exercise."""
        stmt = select(TypingExerciseModel).where(
            TypingExerciseModel.id == str(exercise.id)
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update
            existing.title = exercise.title
            existing.source_code = exercise.source_code
            existing.language = exercise.language
            existing.category = exercise.category.value
            existing.difficulty = exercise.difficulty.value
            existing.description = exercise.description
            existing.required_completions = exercise.required_completions
            existing.is_published = exercise.is_published
            existing.updated_at = exercise.updated_at
        else:
            # Insert - convert UUID to string for SQLite
            model = TypingExerciseModel(
                id=str(exercise.id),
                title=exercise.title,
                source_code=exercise.source_code,
                language=exercise.language,
                category=exercise.category.value,
                difficulty=exercise.difficulty.value,
                description=exercise.description,
                required_completions=exercise.required_completions,
                is_published=exercise.is_published,
                created_at=exercise.created_at,
                updated_at=exercise.updated_at,
            )
            self.session.add(model)

        await self.session.commit()
        return exercise

    async def delete(self, exercise_id: UUID) -> bool:
        """Delete an exercise."""
        stmt = select(TypingExerciseModel).where(
            TypingExerciseModel.id == str(exercise_id)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            await self.session.delete(model)
            await self.session.commit()
            return True
        return False

    async def count(self, category: ExerciseCategory | None = None) -> int:
        """Count exercises."""
        stmt = select(func.count(TypingExerciseModel.id)).where(
            TypingExerciseModel.is_published == True
        )

        if category:
            stmt = stmt.where(TypingExerciseModel.category == category.value)

        result = await self.session.execute(stmt)
        return result.scalar() or 0

    def _to_entity(self, model: TypingExerciseModel) -> TypingExercise:
        """Convert model to entity."""
        return TypingExercise(
            id=UUID(model.id) if isinstance(model.id, str) else model.id,
            title=model.title,
            source_code=model.source_code,
            language=model.language,
            category=ExerciseCategory(model.category),
            difficulty=Difficulty(model.difficulty),
            description=model.description or "",
            required_completions=model.required_completions,
            is_published=model.is_published,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class SQLAlchemyTypingAttemptRepository(TypingAttemptRepository):
    """SQLAlchemy implementation of typing attempt repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, attempt_id: UUID) -> TypingAttempt | None:
        """Get attempt by ID."""
        stmt = select(TypingAttemptModel).where(
            TypingAttemptModel.id == str(attempt_id)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_by_user_and_exercise(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> list[TypingAttempt]:
        """List attempts by user and exercise."""
        stmt = (
            select(TypingAttemptModel)
            .where(
                TypingAttemptModel.user_id == str(user_id),
                TypingAttemptModel.exercise_id == str(exercise_id),
            )
            .order_by(TypingAttemptModel.attempt_number)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def list_by_user(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TypingAttempt]:
        """List all attempts by user."""
        stmt = (
            select(TypingAttemptModel)
            .where(TypingAttemptModel.user_id == str(user_id))
            .order_by(TypingAttemptModel.started_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def save(self, attempt: TypingAttempt) -> TypingAttempt:
        """Save or update an attempt."""
        stmt = select(TypingAttemptModel).where(
            TypingAttemptModel.id == str(attempt.id)
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update
            existing.user_code = attempt.user_code
            existing.accuracy = attempt.accuracy
            existing.wpm = attempt.wpm
            existing.time_seconds = attempt.time_seconds
            existing.status = attempt.status.value
            existing.completed_at = attempt.completed_at
        else:
            # Insert - convert UUIDs to strings for SQLite
            model = TypingAttemptModel(
                id=str(attempt.id),
                user_id=str(attempt.user_id),
                exercise_id=str(attempt.exercise_id),
                attempt_number=attempt.attempt_number,
                user_code=attempt.user_code,
                accuracy=attempt.accuracy,
                wpm=attempt.wpm,
                time_seconds=attempt.time_seconds,
                status=attempt.status.value,
                started_at=attempt.started_at,
                completed_at=attempt.completed_at,
            )
            self.session.add(model)

        await self.session.commit()
        return attempt

    async def get_user_progress(
        self,
        user_id: UUID,
        exercise_id: UUID,
    ) -> UserExerciseProgress | None:
        """Get user's progress on an exercise."""
        attempts = await self.list_by_user_and_exercise(user_id, exercise_id)
        if not attempts:
            return None

        # Get required completions from exercise
        from code_tutor.typing_practice.infrastructure.models import TypingExerciseModel

        stmt = select(TypingExerciseModel.required_completions).where(
            TypingExerciseModel.id == str(exercise_id)
        )
        result = await self.session.execute(stmt)
        required = result.scalar() or 5

        return UserExerciseProgress.from_attempts(
            user_id=user_id,
            exercise_id=exercise_id,
            attempts=attempts,
            required_completions=required,
        )

    async def get_user_stats(self, user_id: UUID) -> dict:
        """Get user's overall typing statistics."""
        # Get all completed attempts
        stmt = select(TypingAttemptModel).where(
            TypingAttemptModel.user_id == str(user_id),
            TypingAttemptModel.status == AttemptStatus.COMPLETED.value,
        )
        result = await self.session.execute(stmt)
        attempts = result.scalars().all()

        if not attempts:
            return {
                "total_exercises_attempted": 0,
                "total_exercises_mastered": 0,
                "total_attempts": 0,
                "average_accuracy": 0.0,
                "average_wpm": 0.0,
                "total_time_seconds": 0.0,
                "best_wpm": 0.0,
            }

        # Calculate stats
        unique_exercises = set(a.exercise_id for a in attempts)
        total_accuracy = sum(a.accuracy for a in attempts)
        total_wpm = sum(a.wpm for a in attempts)
        total_time = sum(a.time_seconds for a in attempts)

        # Count mastered exercises (MASTERY_THRESHOLD+ completions)
        exercise_counts = {}
        for a in attempts:
            exercise_counts[a.exercise_id] = exercise_counts.get(a.exercise_id, 0) + 1
        mastered = sum(
            1
            for count in exercise_counts.values()
            if count >= TypingConstants.MASTERY_THRESHOLD
        )

        return {
            "total_exercises_attempted": len(unique_exercises),
            "total_exercises_mastered": mastered,
            "total_attempts": len(attempts),
            "average_accuracy": total_accuracy / len(attempts) if attempts else 0.0,
            "average_wpm": total_wpm / len(attempts) if attempts else 0.0,
            "total_time_seconds": total_time,
            "best_wpm": max((a.wpm for a in attempts), default=0.0),
        }

    async def get_mastered_exercise_ids(self, user_id: UUID) -> list[str]:
        """
        Get list of mastered exercise IDs for a user.
        Optimized single query instead of N+1 queries.
        """
        # Single query with GROUP BY and HAVING to find mastered exercises
        stmt = (
            select(
                TypingAttemptModel.exercise_id,
                func.count(TypingAttemptModel.id).label("completion_count"),
            )
            .where(
                TypingAttemptModel.user_id == str(user_id),
                TypingAttemptModel.status == AttemptStatus.COMPLETED.value,
            )
            .group_by(TypingAttemptModel.exercise_id)
            .having(
                func.count(TypingAttemptModel.id) >= TypingConstants.MASTERY_THRESHOLD
            )
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        return [str(row.exercise_id) for row in rows]

    async def get_leaderboard(self, limit: int = 10) -> list[dict]:
        """Get top performers by WPM."""
        # This would need to join with users table for username
        # Simplified version without username
        stmt = (
            select(
                TypingAttemptModel.user_id,
                func.max(TypingAttemptModel.wpm).label("best_wpm"),
                func.avg(TypingAttemptModel.accuracy).label("avg_accuracy"),
                func.count(func.distinct(TypingAttemptModel.exercise_id)).label(
                    "exercises"
                ),
            )
            .where(TypingAttemptModel.status == AttemptStatus.COMPLETED.value)
            .group_by(TypingAttemptModel.user_id)
            .order_by(func.max(TypingAttemptModel.wpm).desc())
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        return [
            {
                "rank": i + 1,
                "user_id": str(row.user_id),
                "username": f"User_{str(row.user_id)[:8]}",  # Placeholder
                "best_wpm": row.best_wpm,
                "average_accuracy": row.avg_accuracy,
                "exercises_mastered": row.exercises,
            }
            for i, row in enumerate(rows)
        ]

    def _to_entity(self, model: TypingAttemptModel) -> TypingAttempt:
        """Convert model to entity."""
        return TypingAttempt(
            id=UUID(model.id) if isinstance(model.id, str) else model.id,
            user_id=UUID(model.user_id)
            if isinstance(model.user_id, str)
            else model.user_id,
            exercise_id=UUID(model.exercise_id)
            if isinstance(model.exercise_id, str)
            else model.exercise_id,
            attempt_number=model.attempt_number,
            user_code=model.user_code or "",
            accuracy=model.accuracy or 0.0,
            wpm=model.wpm or 0.0,
            time_seconds=model.time_seconds or 0.0,
            status=AttemptStatus(model.status),
            started_at=model.started_at,
            completed_at=model.completed_at,
        )
