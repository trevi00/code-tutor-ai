"""FastAPI routes for typing practice."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.shared.infrastructure.database import get_async_session as get_db
from code_tutor.identity.interface.dependencies import get_current_user, get_admin_user
from code_tutor.identity.application.dto import UserResponse
from code_tutor.typing_practice.domain.value_objects import ExerciseCategory
from code_tutor.typing_practice.application.dto import (
    CreateExerciseRequest,
    StartAttemptRequest,
    CompleteAttemptRequest,
    TypingExerciseResponse,
    TypingExerciseListResponse,
    TypingAttemptResponse,
    UserProgressResponse,
    UserTypingStatsResponse,
    LeaderboardResponse,
)
from code_tutor.typing_practice.application.services import TypingPracticeService
from code_tutor.typing_practice.infrastructure.repository import (
    SQLAlchemyTypingExerciseRepository,
    SQLAlchemyTypingAttemptRepository,
)
from code_tutor.gamification.application.services import XPService, BadgeService
from code_tutor.gamification.infrastructure.repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
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

@router.get("/exercises", response_model=TypingExerciseListResponse)
async def list_exercises(
    category: Optional[ExerciseCategory] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """List all typing exercises."""
    return await service.list_exercises(
        category=category,
        page=page,
        page_size=page_size,
    )


@router.get("/exercises/{exercise_id}", response_model=TypingExerciseResponse)
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


@router.post("/exercises", response_model=TypingExerciseResponse, status_code=status.HTTP_201_CREATED)
async def create_exercise(
    request: CreateExerciseRequest,
    current_user: UserResponse = Depends(get_admin_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Create a new typing exercise (admin only)."""
    return await service.create_exercise(request)


@router.get("/exercises/{exercise_id}/progress", response_model=UserProgressResponse)
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

@router.post("/attempts", response_model=TypingAttemptResponse, status_code=status.HTTP_201_CREATED)
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


@router.post("/attempts/{attempt_id}/complete", response_model=TypingAttemptResponse)
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

        # Bonus XP for high accuracy (95%+)
        if request.accuracy >= 95:
            await xp_service.add_xp(current_user.id, "typing_high_accuracy")

        # Check if exercise is now mastered
        progress = await service.get_user_progress(current_user.id, attempt.exercise_id)
        if progress and progress.is_mastered and progress.completed_attempts == 5:
            # Just reached mastery (5th completion)
            await xp_service.add_xp(current_user.id, "typing_exercise_mastered")

        await db.commit()
    except Exception as e:
        # Log error but don't fail the request
        import logging
        logging.getLogger(__name__).warning(f"Failed to award XP: {e}")

    return attempt


# ============== Stats Endpoints ==============

@router.get("/stats", response_model=UserTypingStatsResponse)
async def get_user_stats(
    current_user: UserResponse = Depends(get_current_user),
    service: TypingPracticeService = Depends(get_typing_service),
):
    """Get current user's typing statistics."""
    return await service.get_user_stats(current_user.id)


@router.get("/mastered", response_model=list[str])
async def get_mastered_exercises(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get list of mastered exercise IDs for current user."""
    attempt_repo = SQLAlchemyTypingAttemptRepository(db)
    # Optimized: single GROUP BY query instead of N+1 queries
    return await attempt_repo.get_mastered_exercise_ids(current_user.id)


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    limit: int = Query(10, ge=1, le=100, description="Number of entries"),
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
