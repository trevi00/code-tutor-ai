"""FastAPI routes for Learning Roadmap."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.shared.infrastructure.database import get_async_session as get_db
from code_tutor.identity.interface.dependencies import get_current_user, get_optional_user
from code_tutor.identity.application.dto import UserResponse
from code_tutor.roadmap.domain.value_objects import PathLevel
from code_tutor.roadmap.application.dto import (
    LearningPathResponse,
    LearningPathListResponse,
    ModuleResponse,
    LessonResponse,
    UserProgressResponse,
    PathProgressResponse,
    LessonProgressResponse,
    CompleteLessonRequest,
)
from code_tutor.roadmap.application.services import RoadmapService
from code_tutor.roadmap.infrastructure.repository import (
    SQLAlchemyLearningPathRepository,
    SQLAlchemyModuleRepository,
    SQLAlchemyLessonRepository,
    SQLAlchemyUserProgressRepository,
)
from code_tutor.gamification.application.services import BadgeService, XPService
from code_tutor.gamification.infrastructure.repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
)

router = APIRouter(prefix="/roadmap", tags=["Learning Roadmap"])


def get_xp_service(db: AsyncSession = Depends(get_db)) -> XPService:
    """Dependency to get XP service for gamification integration."""
    badge_repo = SQLAlchemyBadgeRepository(db)
    user_badge_repo = SQLAlchemyUserBadgeRepository(db)
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    badge_service = BadgeService(badge_repo, user_badge_repo, user_stats_repo)
    return XPService(user_stats_repo, badge_service)


def get_roadmap_service(
    db: AsyncSession = Depends(get_db),
    xp_service: XPService = Depends(get_xp_service),
) -> RoadmapService:
    """Dependency to get roadmap service."""
    path_repo = SQLAlchemyLearningPathRepository(db)
    module_repo = SQLAlchemyModuleRepository(db)
    lesson_repo = SQLAlchemyLessonRepository(db)
    progress_repo = SQLAlchemyUserProgressRepository(db)
    return RoadmapService(path_repo, module_repo, lesson_repo, progress_repo, xp_service)


# ============== Path Endpoints ==============


@router.get("/paths", response_model=LearningPathListResponse)
async def list_paths(
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """List all learning paths."""
    user_id = current_user.id if current_user else None
    return await service.list_paths(user_id)


@router.get("/paths/{path_id}", response_model=LearningPathResponse)
async def get_path(
    path_id: UUID,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get a specific learning path with modules and lessons."""
    user_id = current_user.id if current_user else None
    path = await service.get_path(path_id, user_id)
    if not path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning path not found",
        )
    return path


@router.get("/paths/level/{level}", response_model=LearningPathResponse)
async def get_path_by_level(
    level: PathLevel,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get a learning path by level."""
    user_id = current_user.id if current_user else None
    path = await service.get_path_by_level(level, user_id)
    if not path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Learning path for level '{level.value}' not found",
        )
    return path


@router.get("/paths/{path_id}/modules", response_model=list[ModuleResponse])
async def get_path_modules(
    path_id: UUID,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get all modules for a learning path."""
    user_id = current_user.id if current_user else None
    return await service.get_path_modules(path_id, user_id)


# ============== Module Endpoints ==============


@router.get("/modules/{module_id}", response_model=ModuleResponse)
async def get_module(
    module_id: UUID,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get a specific module with lessons."""
    user_id = current_user.id if current_user else None
    module = await service.get_module(module_id, user_id)
    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found",
        )
    return module


@router.get("/modules/{module_id}/lessons", response_model=list[LessonResponse])
async def get_module_lessons(
    module_id: UUID,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get all lessons for a module."""
    user_id = current_user.id if current_user else None
    return await service.get_module_lessons(module_id, user_id)


# ============== Lesson Endpoints ==============


@router.get("/lessons/{lesson_id}", response_model=LessonResponse)
async def get_lesson(
    lesson_id: UUID,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get a specific lesson."""
    user_id = current_user.id if current_user else None
    lesson = await service.get_lesson(lesson_id, user_id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found",
        )
    return lesson


# ============== Progress Endpoints ==============


@router.get("/progress", response_model=UserProgressResponse)
async def get_user_progress(
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get current user's overall progress."""
    return await service.get_user_progress(current_user.id)


@router.get("/progress/paths/{path_id}", response_model=PathProgressResponse)
async def get_path_progress(
    path_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get current user's progress on a specific path."""
    progress = await service.get_path_progress(current_user.id, path_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Path not found",
        )
    return progress


@router.post("/paths/{path_id}/start", response_model=PathProgressResponse)
async def start_path(
    path_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
    db: AsyncSession = Depends(get_db),
):
    """Start a learning path."""
    try:
        progress = await service.start_path(current_user.id, path_id)
        await db.commit()
        return progress
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/lessons/{lesson_id}/complete", response_model=LessonProgressResponse)
async def complete_lesson(
    lesson_id: UUID,
    request: CompleteLessonRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
    db: AsyncSession = Depends(get_db),
):
    """Complete a lesson."""
    try:
        progress = await service.complete_lesson(
            current_user.id, lesson_id, request
        )
        await db.commit()
        return progress
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/next-lesson", response_model=Optional[LessonResponse])
async def get_next_lesson(
    path_id: Optional[UUID] = None,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get the next lesson for the current user."""
    return await service.get_next_lesson(current_user.id, path_id)
