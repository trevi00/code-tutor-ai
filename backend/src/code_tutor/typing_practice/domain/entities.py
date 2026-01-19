"""Domain entities for typing practice."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)

from code_tutor.typing_practice.domain.value_objects import (
    AttemptStatus,
    Difficulty,
    ExerciseCategory,
)


@dataclass
class TypingExercise:
    """Typing exercise entity - code to be memorized through repetition."""

    id: UUID
    title: str
    source_code: str
    language: str = "python"
    category: ExerciseCategory = ExerciseCategory.TEMPLATE
    difficulty: Difficulty = Difficulty.EASY
    description: str = ""
    required_completions: int = 5
    is_published: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        title: str,
        source_code: str,
        language: str = "python",
        category: ExerciseCategory = ExerciseCategory.TEMPLATE,
        difficulty: Difficulty = Difficulty.EASY,
        description: str = "",
        required_completions: int = 5,
    ) -> "TypingExercise":
        """Factory method to create a new exercise."""
        now = utc_now()
        return cls(
            id=uuid4(),
            title=title,
            source_code=source_code,
            language=language,
            category=category,
            difficulty=difficulty,
            description=description,
            required_completions=required_completions,
            created_at=now,
            updated_at=now,
        )

    def update(
        self,
        title: str | None = None,
        source_code: str | None = None,
        description: str | None = None,
        difficulty: Difficulty | None = None,
    ) -> None:
        """Update exercise properties."""
        if title is not None:
            self.title = title
        if source_code is not None:
            self.source_code = source_code
        if description is not None:
            self.description = description
        if difficulty is not None:
            self.difficulty = difficulty
        self.updated_at = utc_now()

    @property
    def char_count(self) -> int:
        """Get the character count of source code."""
        return len(self.source_code)

    @property
    def line_count(self) -> int:
        """Get the line count of source code."""
        return len(self.source_code.splitlines())


@dataclass
class TypingAttempt:
    """User's attempt at a typing exercise."""

    id: UUID
    user_id: UUID
    exercise_id: UUID
    attempt_number: int  # 1-5
    user_code: str = ""
    accuracy: float = 0.0  # 0-100
    wpm: float = 0.0  # Words per minute
    time_seconds: float = 0.0
    status: AttemptStatus = AttemptStatus.IN_PROGRESS
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    @classmethod
    def create(
        cls,
        user_id: UUID,
        exercise_id: UUID,
        attempt_number: int,
    ) -> "TypingAttempt":
        """Factory method to create a new attempt."""
        return cls(
            id=uuid4(),
            user_id=user_id,
            exercise_id=exercise_id,
            attempt_number=attempt_number,
            started_at=utc_now(),
        )

    def complete(
        self,
        user_code: str,
        accuracy: float,
        wpm: float,
        time_seconds: float,
    ) -> None:
        """Mark the attempt as completed."""
        self.user_code = user_code
        self.accuracy = accuracy
        self.wpm = wpm
        self.time_seconds = time_seconds
        self.status = AttemptStatus.COMPLETED
        self.completed_at = utc_now()

    def abandon(self) -> None:
        """Mark the attempt as abandoned."""
        self.status = AttemptStatus.ABANDONED
        self.completed_at = utc_now()

    @property
    def is_completed(self) -> bool:
        """Check if attempt is completed."""
        return self.status == AttemptStatus.COMPLETED


@dataclass
class UserExerciseProgress:
    """User's progress on a specific exercise."""

    user_id: UUID
    exercise_id: UUID
    completed_attempts: int
    best_accuracy: float
    best_wpm: float
    total_time_seconds: float
    is_mastered: bool  # True if completed required_completions times

    @classmethod
    def from_attempts(
        cls,
        user_id: UUID,
        exercise_id: UUID,
        attempts: list[TypingAttempt],
        required_completions: int = 5,
    ) -> "UserExerciseProgress":
        """Create progress from list of attempts."""
        completed = [a for a in attempts if a.is_completed]
        return cls(
            user_id=user_id,
            exercise_id=exercise_id,
            completed_attempts=len(completed),
            best_accuracy=max((a.accuracy for a in completed), default=0.0),
            best_wpm=max((a.wpm for a in completed), default=0.0),
            total_time_seconds=sum(a.time_seconds for a in completed),
            is_mastered=len(completed) >= required_completions,
        )
