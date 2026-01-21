"""FastAPI routes for Learning Roadmap."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.gamification.application.services import BadgeService, XPService
from code_tutor.gamification.infrastructure.repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
)
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import (
    get_current_user,
    get_optional_user,
)
from code_tutor.roadmap.application.dto import (
    CompleteLessonRequest,
    LearningPathListResponse,
    LearningPathResponse,
    LessonProgressResponse,
    LessonResponse,
    ModuleResponse,
    PathProgressResponse,
    UserProgressResponse,
)
from code_tutor.roadmap.application.services import RoadmapService
from code_tutor.roadmap.domain.value_objects import PathLevel
from code_tutor.roadmap.infrastructure.repository import (
    SQLAlchemyLearningPathRepository,
    SQLAlchemyLessonRepository,
    SQLAlchemyModuleRepository,
    SQLAlchemyUserProgressRepository,
)
from code_tutor.shared.infrastructure.database import get_async_session as get_db

router = APIRouter(prefix="/roadmap", tags=["Roadmap"])


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
    return RoadmapService(
        path_repo, module_repo, lesson_repo, progress_repo, xp_service
    )


# ============== Path Endpoints ==============


@router.get(
    "/paths",
    response_model=LearningPathListResponse,
    summary="학습 경로 목록 조회",
    description="모든 학습 경로(입문, 초급, 중급, 고급)를 조회합니다. 로그인한 경우 진행 상황이 포함됩니다.",
    responses={
        200: {"description": "학습 경로 목록 반환"},
    },
)
async def list_paths(
    current_user: UserResponse | None = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """List all learning paths."""
    user_id = current_user.id if current_user else None
    return await service.list_paths(user_id)


@router.get(
    "/paths/{path_id}",
    response_model=LearningPathResponse,
    summary="학습 경로 상세 조회",
    description="특정 학습 경로의 상세 정보(모듈, 레슨 포함)를 조회합니다.",
    responses={
        200: {"description": "학습 경로 상세 정보"},
        404: {"description": "학습 경로를 찾을 수 없음"},
    },
)
async def get_path(
    path_id: UUID,
    current_user: UserResponse | None = Depends(get_optional_user),
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


@router.get(
    "/paths/level/{level}",
    response_model=LearningPathResponse,
    summary="레벨별 학습 경로 조회",
    description="레벨(beginner, elementary, intermediate, advanced)로 학습 경로를 조회합니다.",
    responses={
        200: {"description": "학습 경로 상세 정보"},
        404: {"description": "해당 레벨의 학습 경로를 찾을 수 없음"},
    },
)
async def get_path_by_level(
    level: PathLevel,
    current_user: UserResponse | None = Depends(get_optional_user),
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


@router.get(
    "/paths/{path_id}/modules",
    response_model=list[ModuleResponse],
    summary="경로의 모듈 목록 조회",
    description="특정 학습 경로에 포함된 모든 모듈을 조회합니다.",
    responses={
        200: {"description": "모듈 목록 반환"},
    },
)
async def get_path_modules(
    path_id: UUID,
    current_user: UserResponse | None = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get all modules for a learning path."""
    user_id = current_user.id if current_user else None
    return await service.get_path_modules(path_id, user_id)


# ============== Module Endpoints ==============


@router.get(
    "/modules/{module_id}",
    response_model=ModuleResponse,
    summary="모듈 상세 조회",
    description="특정 모듈의 상세 정보(레슨 포함)를 조회합니다.",
    responses={
        200: {"description": "모듈 상세 정보"},
        404: {"description": "모듈을 찾을 수 없음"},
    },
)
async def get_module(
    module_id: UUID,
    current_user: UserResponse | None = Depends(get_optional_user),
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


@router.get(
    "/modules/{module_id}/lessons",
    response_model=list[LessonResponse],
    summary="모듈의 레슨 목록 조회",
    description="특정 모듈에 포함된 모든 레슨(강의, 문제, 퀴즈 등)을 조회합니다.",
    responses={
        200: {"description": "레슨 목록 반환"},
    },
)
async def get_module_lessons(
    module_id: UUID,
    current_user: UserResponse | None = Depends(get_optional_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get all lessons for a module."""
    user_id = current_user.id if current_user else None
    return await service.get_module_lessons(module_id, user_id)


# ============== Lesson Endpoints ==============


@router.get(
    "/lessons/{lesson_id}",
    response_model=LessonResponse,
    summary="레슨 상세 조회",
    description="특정 레슨의 상세 정보(타입, 콘텐츠, XP 보상 등)를 조회합니다.",
    responses={
        200: {"description": "레슨 상세 정보"},
        404: {"description": "레슨을 찾을 수 없음"},
    },
)
async def get_lesson(
    lesson_id: UUID,
    current_user: UserResponse | None = Depends(get_optional_user),
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


@router.get(
    "/progress",
    response_model=UserProgressResponse,
    summary="내 전체 진행 상황 조회",
    description="현재 로그인한 사용자의 모든 학습 경로에 대한 전체 진행 상황을 조회합니다.",
    responses={
        200: {"description": "전체 진행 상황 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_user_progress(
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get current user's overall progress."""
    return await service.get_user_progress(current_user.id)


@router.get(
    "/progress/paths/{path_id}",
    response_model=PathProgressResponse,
    summary="특정 경로 진행 상황 조회",
    description="특정 학습 경로에 대한 현재 사용자의 진행 상황(완료율, 완료 레슨 수 등)을 조회합니다.",
    responses={
        200: {"description": "경로 진행 상황 반환"},
        401: {"description": "인증 필요"},
        404: {"description": "학습 경로를 찾을 수 없음"},
    },
)
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


@router.post(
    "/paths/{path_id}/start",
    response_model=PathProgressResponse,
    summary="학습 경로 시작",
    description="특정 학습 경로를 시작합니다. 시작 시간이 기록되고 진행 상황 추적이 시작됩니다.",
    responses={
        200: {"description": "경로 시작 성공, 진행 상황 반환"},
        401: {"description": "인증 필요"},
        404: {"description": "학습 경로를 찾을 수 없음"},
    },
)
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


@router.post(
    "/lessons/{lesson_id}/complete",
    response_model=LessonProgressResponse,
    summary="레슨 완료 처리",
    description="특정 레슨을 완료 처리합니다. 퀴즈의 경우 점수를 함께 전송하며, XP가 자동으로 부여됩니다.",
    responses={
        200: {"description": "레슨 완료 성공, XP 부여"},
        401: {"description": "인증 필요"},
        404: {"description": "레슨을 찾을 수 없음"},
    },
)
async def complete_lesson(
    lesson_id: UUID,
    request: CompleteLessonRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
    db: AsyncSession = Depends(get_db),
):
    """Complete a lesson."""
    try:
        progress = await service.complete_lesson(current_user.id, lesson_id, request)
        await db.commit()
        return progress
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/next-lesson",
    response_model=LessonResponse | None,
    summary="다음 추천 레슨 조회",
    description="현재 사용자에게 추천하는 다음 레슨을 조회합니다. path_id를 지정하면 해당 경로 내에서 다음 레슨을 반환합니다.",
    responses={
        200: {"description": "다음 레슨 반환 (없으면 null)"},
        401: {"description": "인증 필요"},
    },
)
async def get_next_lesson(
    path_id: UUID | None = None,
    current_user: UserResponse = Depends(get_current_user),
    service: RoadmapService = Depends(get_roadmap_service),
):
    """Get the next lesson for the current user."""
    return await service.get_next_lesson(current_user.id, path_id)
