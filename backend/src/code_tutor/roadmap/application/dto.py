"""DTOs for Learning Roadmap application layer."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from code_tutor.roadmap.domain.value_objects import (
    LessonType,
    PathLevel,
    ProgressStatus,
)

# ============== Lesson DTOs ==============


class LessonResponse(BaseModel):
    """Response DTO for a lesson."""

    id: UUID
    module_id: UUID
    title: str
    description: str
    lesson_type: LessonType
    content: str
    content_id: UUID | None = None
    order: int
    xp_reward: int
    estimated_minutes: int

    # Progress info (when user is logged in)
    status: ProgressStatus | None = None
    completed_at: datetime | None = None
    score: int | None = None

    class Config:
        from_attributes = True


class LessonListResponse(BaseModel):
    """Response DTO for list of lessons."""

    items: list[LessonResponse]
    total: int


# ============== Module DTOs ==============


class ModuleResponse(BaseModel):
    """Response DTO for a module."""

    id: UUID
    path_id: UUID
    title: str
    description: str
    order: int
    lesson_count: int
    total_xp: int
    estimated_minutes: int
    lessons: list[LessonResponse] = Field(default_factory=list)

    # Progress info
    completed_lessons: int = 0
    completion_rate: float = 0.0

    class Config:
        from_attributes = True


class ModuleListResponse(BaseModel):
    """Response DTO for list of modules."""

    items: list[ModuleResponse]
    total: int


# ============== Learning Path DTOs ==============


class LearningPathResponse(BaseModel):
    """Response DTO for a learning path."""

    id: UUID
    level: PathLevel
    level_display: str
    title: str
    description: str
    icon: str
    order: int
    estimated_hours: int
    module_count: int
    lesson_count: int
    total_xp: int
    prerequisites: list[UUID] = Field(default_factory=list)
    modules: list[ModuleResponse] = Field(default_factory=list)

    # Progress info
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    completed_lessons: int = 0
    completion_rate: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class LearningPathListResponse(BaseModel):
    """Response DTO for list of learning paths."""

    items: list[LearningPathResponse]
    total: int


# ============== Progress DTOs ==============


class UserProgressResponse(BaseModel):
    """Response DTO for user progress overview."""

    total_paths: int
    completed_paths: int
    in_progress_paths: int
    total_lessons: int
    completed_lessons: int
    total_xp_earned: int
    current_path: LearningPathResponse | None = None
    next_lesson: LessonResponse | None = None
    paths: list[LearningPathResponse] = Field(default_factory=list)


class PathProgressResponse(BaseModel):
    """Response DTO for path progress."""

    path_id: UUID
    status: ProgressStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    completed_lessons: int
    total_lessons: int
    completion_rate: float


class LessonProgressResponse(BaseModel):
    """Response DTO for lesson progress."""

    lesson_id: UUID
    status: ProgressStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    score: int | None = None
    attempts: int


# ============== Request DTOs ==============


class CompleteLessonRequest(BaseModel):
    """Request DTO for completing a lesson."""

    score: int | None = Field(None, ge=0, le=100)


class StartPathRequest(BaseModel):
    """Request DTO for starting a learning path."""

    pass  # No additional data needed


# ============== Summary DTOs ==============


class RoadmapSummaryResponse(BaseModel):
    """Response DTO for roadmap summary."""

    paths: list[LearningPathResponse]
    user_progress: UserProgressResponse | None = None
