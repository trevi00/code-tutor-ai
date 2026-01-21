"""DTOs for typing practice application layer."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from code_tutor.typing_practice.domain.value_objects import (
    AttemptStatus,
    Difficulty,
    ExerciseCategory,
)

# ============== Request DTOs ==============


class CreateExerciseRequest(BaseModel):
    """Request to create a new typing exercise."""

    title: str = Field(..., min_length=1, max_length=255)
    source_code: str = Field(..., min_length=1)
    language: str = "python"
    category: ExerciseCategory = ExerciseCategory.TEMPLATE
    difficulty: Difficulty = Difficulty.EASY
    description: str = ""
    required_completions: int = Field(default=5, ge=1, le=10)


class StartAttemptRequest(BaseModel):
    """Request to start a new typing attempt."""

    exercise_id: UUID


class CompleteAttemptRequest(BaseModel):
    """Request to complete a typing attempt."""

    user_code: str
    accuracy: float = Field(..., ge=0, le=100)
    wpm: float = Field(..., ge=0)
    time_seconds: float = Field(..., ge=0)


# ============== Response DTOs ==============


class TypingExerciseResponse(BaseModel):
    """Response for a typing exercise."""

    id: UUID
    title: str
    source_code: str
    language: str
    category: ExerciseCategory
    difficulty: Difficulty
    description: str
    required_completions: int
    char_count: int
    line_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class TypingExerciseListResponse(BaseModel):
    """Response for listing typing exercises."""

    exercises: list[TypingExerciseResponse]
    total: int
    page: int
    page_size: int


class TypingAttemptResponse(BaseModel):
    """Response for a typing attempt."""

    id: UUID
    user_id: UUID
    exercise_id: UUID
    attempt_number: int
    accuracy: float
    wpm: float
    time_seconds: float
    status: AttemptStatus
    started_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class UserProgressResponse(BaseModel):
    """Response for user's progress on an exercise."""

    user_id: UUID
    exercise_id: UUID
    completed_attempts: int
    required_completions: int
    best_accuracy: float
    best_wpm: float
    total_time_seconds: float
    is_mastered: bool
    attempts: list[TypingAttemptResponse]


class UserTypingStatsResponse(BaseModel):
    """Response for user's overall typing statistics."""

    total_exercises_attempted: int
    total_exercises_mastered: int
    total_attempts: int
    average_accuracy: float
    average_wpm: float
    total_time_seconds: float
    best_wpm: float


class LeaderboardEntryResponse(BaseModel):
    """Response for a leaderboard entry."""

    rank: int
    user_id: UUID
    username: str
    best_wpm: float
    average_accuracy: float
    exercises_mastered: int


class LeaderboardResponse(BaseModel):
    """Response for leaderboard."""

    entries: list[LeaderboardEntryResponse]
