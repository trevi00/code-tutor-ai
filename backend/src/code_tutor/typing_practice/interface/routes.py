"""FastAPI routes for typing practice."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.gamification.application.services import BadgeService, XPService
from code_tutor.gamification.infrastructure.repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
)
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_admin_user, get_current_user
from code_tutor.shared.constants import Pagination
from code_tutor.shared.constants import TypingPractice as TypingConstants
from code_tutor.shared.infrastructure.database import get_async_session as get_db
from code_tutor.typing_practice.application.dto import (
    CompleteAttemptRequest,
    CreateExerciseRequest,
    LeaderboardResponse,
    StartAttemptRequest,
    TypingAttemptResponse,
    TypingExerciseListResponse,
    TypingExerciseResponse,
    UserProgressResponse,
    UserTypingStatsResponse,
)
from code_tutor.typing_practice.application.services import TypingPracticeService
from code_tutor.typing_practice.domain.value_objects import ExerciseCategory
from code_tutor.typing_practice.infrastructure.repository import (
    SQLAlchemyTypingAttemptRepository,
    SQLAlchemyTypingExerciseRepository,
)

router = APIRouter(prefix="/typing-practice", tags=["Typing Practice"])


def get_typing_service(db: AsyncSession = Depends(get_db)) -> TypingPracticeService:
    """Dependency to get typing practice service."""
    exercise_repo = SQLAlchemyTypingExerciseRepository(db)
    attempt_repo = SQLAlchemyTypingAttemptRepository(db)
    return TypingPracticeService(exercise_repo, attempt_repo)


def get_xp_service(db: AsyncSession = Depends(get_db)) -> XPService:
    """Dependency to get XP service for gamification integration."""
    badge_repo = SQLAlchemyBadgeRepository(db)
    user_badge_repo = SQLAlchemyUserBadgeRepository(db)
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    badge_service = BadgeService(badge_repo, user_badge_repo, user_stats_repo)
    return XPService(user_stats_repo, badge_service)


# ============== Exercise Endpoints ==============


@router.get(
    "/exercises",
    response_model=TypingExerciseListResponse,
    summary="타이핑 연습 목록 조회",
    description="카테고리별로 타이핑 연습 목록을 조회합니다. 페이지네이션을 지원합니다.",
    responses={
        200: {"description": "타이핑 연습 목록 반환"},
    },
)
async def list_exercises(
    category: ExerciseCategory | None = Query(
        None, description="카테고리 필터 (algorithm, pattern, syntax, typing)"
    ),
    page: int = Query(Pagination.DEFAULT_PAGE, ge=1, description="페이지 번호"),
    page_size: int = Query(
        Pagination.DEFAULT_PAGE_SIZE,
        ge=1,
        le=Pagination.MAX_PAGE_SIZE,
        description="페이지당 항목 수",
    ),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """List all typing exercises."""
    return await service.list_exercises(
        category=category,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/exercises/{exercise_id}",
    response_model=TypingExerciseResponse,
    summary="타이핑 연습 상세 조회",
    description="특정 타이핑 연습의 상세 정보를 조회합니다.",
    responses={
        200: {"description": "타이핑 연습 상세 정보"},
        404: {"description": "연습을 찾을 수 없음"},
    },
)
async def get_exercise(
    exercise_id: UUID,
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Get a specific typing exercise."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Looking for exercise: {exercise_id} (type: {type(exercise_id)})")
    exercise = await service.get_exercise(exercise_id)
    logger.info(f"Exercise found: {exercise is not None}")
    if not exercise:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exercise not found",
        )
    return exercise


@router.post(
    "/exercises",
    response_model=TypingExerciseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="타이핑 연습 생성 (관리자)",
    description="새로운 타이핑 연습을 생성합니다. 관리자 권한이 필요합니다.",
    responses={
        201: {"description": "타이핑 연습 생성 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "관리자 권한 필요"},
    },
)
async def create_exercise(
    request: CreateExerciseRequest,
    current_user: UserResponse = Depends(get_admin_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Create a new typing exercise (admin only)."""
    return await service.create_exercise(request)


@router.get(
    "/exercises/{exercise_id}/progress",
    response_model=UserProgressResponse,
    summary="연습 진행 상황 조회",
    description="특정 연습에 대한 현재 사용자의 진행 상황(완료 횟수, 마스터리 여부 등)을 조회합니다.",
    responses={
        200: {"description": "진행 상황 반환"},
        401: {"description": "인증 필요"},
        404: {"description": "연습을 찾을 수 없음"},
    },
)
async def get_exercise_progress(
    exercise_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Get current user's progress on an exercise."""
    progress = await service.get_user_progress(
        user_id=current_user.id,
        exercise_id=exercise_id,
    )
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exercise not found",
        )
    return progress


# ============== Attempt Endpoints ==============


@router.post(
    "/attempts",
    response_model=TypingAttemptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="타이핑 시도 시작",
    description="새로운 타이핑 시도를 시작합니다. 시작 시간이 기록됩니다.",
    responses={
        201: {"description": "시도 시작 성공"},
        401: {"description": "인증 필요"},
    },
)
async def start_attempt(
    request: StartAttemptRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Start a new typing attempt."""
    return await service.start_attempt(
        user_id=current_user.id,
        exercise_id=request.exercise_id,
    )


@router.post(
    "/attempts/{attempt_id}/complete",
    response_model=TypingAttemptResponse,
    summary="타이핑 시도 완료",
    description="타이핑 시도를 완료하고 결과(정확도, WPM, 소요시간)를 기록합니다. XP가 자동으로 부여됩니다.",
    responses={
        200: {"description": "시도 완료 성공, XP 부여"},
        401: {"description": "인증 필요"},
        404: {"description": "시도를 찾을 수 없음"},
    },
)
async def complete_attempt(
    attempt_id: UUID,
    request: CompleteAttemptRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: TypingPracticeService = Depends(get_typing_service),
    xp_service: XPService = Depends(get_xp_service),
    db: AsyncSession = Depends(get_db),
):
    """Complete a typing attempt with results."""
    attempt = await service.complete_attempt(attempt_id, request)
    if not attempt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attempt not found",
        )

    # Award XP for completing attempt
    try:
        # Base XP for attempt completion
        await xp_service.add_xp(current_user.id, "typing_attempt_completed")

        # Bonus XP for high accuracy
        if request.accuracy >= TypingConstants.HIGH_ACCURACY_THRESHOLD:
            await xp_service.add_xp(current_user.id, "typing_high_accuracy")

        # Check if exercise is now mastered
        progress = await service.get_user_progress(current_user.id, attempt.exercise_id)
        if (
            progress
            and progress.is_mastered
            and progress.completed_attempts == TypingConstants.MASTERY_THRESHOLD
        ):
            # Just reached mastery (5th completion)
            await xp_service.add_xp(current_user.id, "typing_exercise_mastered")

        await db.commit()
    except Exception as e:
        # Log error but don't fail the request
        import logging

        logging.getLogger(__name__).warning(f"Failed to award XP: {e}")

    return attempt


# ============== Stats Endpoints ==============


@router.get(
    "/stats",
    response_model=UserTypingStatsResponse,
    summary="내 타이핑 통계 조회",
    description="현재 사용자의 타이핑 통계(총 시도, 평균 WPM, 평균 정확도, 마스터한 연습 수 등)를 조회합니다.",
    responses={
        200: {"description": "통계 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_user_stats(
    current_user: UserResponse = Depends(get_current_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Get current user's typing statistics."""
    return await service.get_user_stats(current_user.id)


@router.get(
    "/mastered",
    response_model=list[str],
    summary="마스터한 연습 목록",
    description="현재 사용자가 마스터한(5회 이상 완료한) 연습의 ID 목록을 조회합니다.",
    responses={
        200: {"description": "마스터한 연습 ID 목록"},
        401: {"description": "인증 필요"},
    },
)
async def get_mastered_exercises(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get list of mastered exercise IDs for current user."""
    attempt_repo = SQLAlchemyTypingAttemptRepository(db)
    # Optimized: single GROUP BY query instead of N+1 queries
    return await attempt_repo.get_mastered_exercise_ids(current_user.id)


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    summary="리더보드 조회",
    description="타이핑 연습 리더보드를 조회합니다. 최고 WPM 기준으로 정렬됩니다.",
    responses={
        200: {"description": "리더보드 반환"},
    },
)
async def get_leaderboard(
    limit: int = Query(
        Pagination.LEADERBOARD_DEFAULT_LIMIT,
        ge=1,
        le=Pagination.LEADERBOARD_MAX_LIMIT,
        description="반환할 항목 수",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Get typing practice leaderboard."""
    attempt_repo = SQLAlchemyTypingAttemptRepository(db)
    entries = await attempt_repo.get_leaderboard(limit)

    from code_tutor.typing_practice.application.dto import LeaderboardEntryResponse

    return LeaderboardResponse(
        entries=[
            LeaderboardEntryResponse(
                rank=e["rank"],
                user_id=UUID(e["user_id"]),
                username=e["username"],
                best_wpm=e["best_wpm"],
                average_accuracy=e["average_accuracy"],
                exercises_mastered=e["exercises_mastered"],
            )
            for e in entries
        ]
    )
