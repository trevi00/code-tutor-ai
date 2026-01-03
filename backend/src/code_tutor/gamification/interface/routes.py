"""Gamification API routes."""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query

from code_tutor.shared.api_response import success_response
from code_tutor.shared.infrastructure.database import get_async_session as get_db
from code_tutor.identity.interface.dependencies import get_current_user
from code_tutor.identity.application.dto import UserResponse
from code_tutor.gamification.application.dto import (
    BadgeResponse,
    UserBadgesResponse,
    UserStatsResponse,
    AddXPRequest,
    XPAddedResponse,
    LeaderboardResponse,
    ChallengeResponse,
    UserChallengeResponse,
    ChallengesResponse,
    GamificationOverviewResponse,
)
from code_tutor.gamification.application.services import (
    BadgeService,
    XPService,
    LeaderboardService,
    ChallengeService,
    GamificationService,
)
from code_tutor.gamification.infrastructure.repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
    SQLAlchemyChallengeRepository,
    SQLAlchemyUserChallengeRepository,
)
from code_tutor.gamification.domain.value_objects import ChallengeType


router = APIRouter(prefix="/gamification", tags=["gamification"])


def get_badge_service(db=Depends(get_db)) -> BadgeService:
    """Get badge service."""
    badge_repo = SQLAlchemyBadgeRepository(db)
    user_badge_repo = SQLAlchemyUserBadgeRepository(db)
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    return BadgeService(badge_repo, user_badge_repo, user_stats_repo)


def get_xp_service(
    db=Depends(get_db),
    badge_service: BadgeService = Depends(get_badge_service),
) -> XPService:
    """Get XP service."""
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    return XPService(user_stats_repo, badge_service)


def get_leaderboard_service(db=Depends(get_db)) -> LeaderboardService:
    """Get leaderboard service."""
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    return LeaderboardService(user_stats_repo)


def get_challenge_service(db=Depends(get_db)) -> ChallengeService:
    """Get challenge service."""
    challenge_repo = SQLAlchemyChallengeRepository(db)
    user_challenge_repo = SQLAlchemyUserChallengeRepository(db)
    user_stats_repo = SQLAlchemyUserStatsRepository(db)
    return ChallengeService(challenge_repo, user_challenge_repo, user_stats_repo)


def get_gamification_service(
    badge_service: BadgeService = Depends(get_badge_service),
    xp_service: XPService = Depends(get_xp_service),
    leaderboard_service: LeaderboardService = Depends(get_leaderboard_service),
    challenge_service: ChallengeService = Depends(get_challenge_service),
) -> GamificationService:
    """Get gamification service."""
    return GamificationService(
        badge_service, xp_service, leaderboard_service, challenge_service
    )


# Overview
@router.get("/overview")
async def get_gamification_overview(
    current_user: UserResponse = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service),
):
    """Get complete gamification overview for current user."""
    data = await service.get_overview(current_user.id)
    return success_response(data)


# Badges
@router.get("/badges")
async def get_all_badges(
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Get all available badges."""
    data = await badge_service.get_all_badges()
    return success_response(data)


@router.get("/badges/me")
async def get_my_badges(
    current_user: UserResponse = Depends(get_current_user),
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Get current user's badges."""
    data = await badge_service.get_user_badges(current_user.id)
    return success_response(data)


@router.post("/badges/check")
async def check_badges(
    current_user: UserResponse = Depends(get_current_user),
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Check and award any new badges for current user."""
    data = await badge_service.check_and_award_badges(current_user.id)
    return success_response(data)


# Stats & XP
@router.get("/stats")
async def get_my_stats(
    current_user: UserResponse = Depends(get_current_user),
    xp_service: XPService = Depends(get_xp_service),
):
    """Get current user's gamification stats."""
    data = await xp_service.get_user_stats(current_user.id)
    return success_response(data)


@router.post("/xp")
async def add_xp(
    request: AddXPRequest,
    current_user: UserResponse = Depends(get_current_user),
    xp_service: XPService = Depends(get_xp_service),
):
    """Add XP for an action (internal use or admin)."""
    data = await xp_service.add_xp(
        current_user.id, request.action, request.custom_amount
    )
    return success_response(data)


@router.post("/activity/{action}")
async def record_activity(
    action: str,
    current_user: UserResponse = Depends(get_current_user),
    xp_service: XPService = Depends(get_xp_service),
):
    """Record an activity and award XP."""
    data = await xp_service.record_activity(current_user.id, action)
    return success_response(data)


# Leaderboard
@router.get("/leaderboard")
async def get_leaderboard(
    period: str = Query("all", pattern="^(all|weekly|monthly)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserResponse] = Depends(get_current_user),
    leaderboard_service: LeaderboardService = Depends(get_leaderboard_service),
):
    """Get leaderboard."""
    user_id = current_user.id if current_user else None
    data = await leaderboard_service.get_leaderboard(
        period=period, limit=limit, offset=offset, user_id=user_id
    )
    return success_response(data)


# Challenges
@router.get("/challenges")
async def get_my_challenges(
    current_user: UserResponse = Depends(get_current_user),
    challenge_service: ChallengeService = Depends(get_challenge_service),
):
    """Get current user's challenges."""
    data = await challenge_service.get_user_challenges(current_user.id)
    return success_response(data)


@router.post("/challenges/{challenge_id}/join")
async def join_challenge(
    challenge_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    challenge_service: ChallengeService = Depends(get_challenge_service),
):
    """Join a challenge."""
    try:
        data = await challenge_service.join_challenge(current_user.id, challenge_id)
        return success_response(data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Admin: Seed badges
@router.post("/admin/seed-badges")
async def seed_badges(
    badge_service: BadgeService = Depends(get_badge_service),
    db=Depends(get_db),
):
    """Seed predefined badges (admin only)."""
    count = await badge_service.seed_badges()
    await db.commit()
    return success_response({"message": f"Seeded {count} new badges"})


# Admin: Create challenge
@router.post("/admin/challenges")
async def create_challenge(
    name: str,
    description: str,
    challenge_type: ChallengeType,
    target_action: str,
    target_value: int,
    xp_reward: int,
    duration_days: int = 7,
    challenge_service: ChallengeService = Depends(get_challenge_service),
    db=Depends(get_db),
):
    """Create a new challenge (admin only)."""
    data = await challenge_service.create_challenge(
        name=name,
        description=description,
        challenge_type=challenge_type,
        target_action=target_action,
        target_value=target_value,
        xp_reward=xp_reward,
        duration_days=duration_days,
    )
    await db.commit()
    return success_response(data)
