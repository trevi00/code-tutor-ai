"""Gamification API routes."""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query

from code_tutor.shared.api_response import success_response
from code_tutor.shared.infrastructure.database import get_async_session as get_db
from code_tutor.identity.interface.dependencies import get_current_user, get_admin_user
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


router = APIRouter(prefix="/gamification", tags=["Gamification"])


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
@router.get(
    "/overview",
    summary="게이미피케이션 전체 현황",
    description="현재 사용자의 XP, 레벨, 뱃지, 챌린지 등 게이미피케이션 전체 현황을 조회합니다.",
    responses={
        200: {"description": "게이미피케이션 현황 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_gamification_overview(
    current_user: UserResponse = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service),
):
    """Get complete gamification overview for current user."""
    data = await service.get_overview(current_user.id)
    return success_response(data)


# Badges
@router.get(
    "/badges",
    summary="전체 뱃지 목록",
    description="획득 가능한 모든 뱃지 목록과 획득 조건을 조회합니다.",
    responses={
        200: {"description": "뱃지 목록 반환"},
    },
)
async def get_all_badges(
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Get all available badges."""
    data = await badge_service.get_all_badges()
    return success_response(data)


@router.get(
    "/badges/me",
    summary="내 뱃지 목록",
    description="현재 사용자가 획득한 뱃지 목록을 조회합니다.",
    responses={
        200: {"description": "획득한 뱃지 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_my_badges(
    current_user: UserResponse = Depends(get_current_user),
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Get current user's badges."""
    data = await badge_service.get_user_badges(current_user.id)
    return success_response(data)


@router.post(
    "/badges/check",
    summary="뱃지 획득 확인",
    description="현재 사용자의 활동을 기반으로 새로 획득 가능한 뱃지를 확인하고 부여합니다.",
    responses={
        200: {"description": "새로 획득한 뱃지 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def check_badges(
    current_user: UserResponse = Depends(get_current_user),
    badge_service: BadgeService = Depends(get_badge_service),
):
    """Check and award any new badges for current user."""
    data = await badge_service.check_and_award_badges(current_user.id)
    return success_response(data)


# Stats & XP
@router.get(
    "/stats",
    summary="내 게이미피케이션 통계",
    description="현재 사용자의 XP, 레벨, 총 문제 풀이 수 등 통계를 조회합니다.",
    responses={
        200: {"description": "통계 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_my_stats(
    current_user: UserResponse = Depends(get_current_user),
    xp_service: XPService = Depends(get_xp_service),
):
    """Get current user's gamification stats."""
    data = await xp_service.get_user_stats(current_user.id)
    return success_response(data)


@router.post(
    "/xp",
    summary="XP 추가",
    description="특정 활동에 대해 XP를 추가합니다. 내부 또는 관리자용입니다.",
    responses={
        200: {"description": "XP 추가 결과 반환"},
        401: {"description": "인증 필요"},
    },
)
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


@router.post(
    "/activity/{action}",
    summary="활동 기록",
    description="활동(problem_solved, lesson_completed 등)을 기록하고 XP를 부여합니다.",
    responses={
        200: {"description": "활동 기록 및 XP 부여 결과"},
        401: {"description": "인증 필요"},
    },
)
async def record_activity(
    action: str,
    current_user: UserResponse = Depends(get_current_user),
    xp_service: XPService = Depends(get_xp_service),
):
    """Record an activity and award XP."""
    data = await xp_service.record_activity(current_user.id, action)
    return success_response(data)


# Leaderboard
@router.get(
    "/leaderboard",
    summary="리더보드 조회",
    description="XP 기준 리더보드를 조회합니다. 전체, 주간, 월간 기간별 조회가 가능합니다.",
    responses={
        200: {"description": "리더보드 반환"},
    },
)
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
@router.get(
    "/challenges",
    summary="내 챌린지 목록",
    description="현재 사용자가 참여 중인 챌린지 목록과 진행 상황을 조회합니다.",
    responses={
        200: {"description": "챌린지 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def get_my_challenges(
    current_user: UserResponse = Depends(get_current_user),
    challenge_service: ChallengeService = Depends(get_challenge_service),
):
    """Get current user's challenges."""
    data = await challenge_service.get_user_challenges(current_user.id)
    return success_response(data)


@router.post(
    "/challenges/{challenge_id}/join",
    summary="챌린지 참여",
    description="특정 챌린지에 참여합니다. 챌린지 완료 시 보상 XP를 획득합니다.",
    responses={
        200: {"description": "챌린지 참여 성공"},
        401: {"description": "인증 필요"},
        404: {"description": "챌린지를 찾을 수 없음"},
    },
)
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
@router.post(
    "/admin/seed-badges",
    summary="뱃지 시드 (관리자)",
    description="미리 정의된 뱃지를 데이터베이스에 시딩합니다. 관리자 전용입니다.",
    responses={
        200: {"description": "시딩된 뱃지 수 반환"},
        401: {"description": "인증 필요"},
        403: {"description": "관리자 권한 필요"},
    },
)
async def seed_badges(
    current_user: UserResponse = Depends(get_admin_user),
    badge_service: BadgeService = Depends(get_badge_service),
    db=Depends(get_db),
):
    """Seed predefined badges (admin only)."""
    count = await badge_service.seed_badges()
    await db.commit()
    return success_response({"message": f"Seeded {count} new badges"})


# Admin: Create challenge
@router.post(
    "/admin/challenges",
    summary="챌린지 생성 (관리자)",
    description="새로운 챌린지를 생성합니다. 관리자 전용입니다.",
    responses={
        200: {"description": "챌린지 생성 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "관리자 권한 필요"},
    },
)
async def create_challenge(
    name: str,
    description: str,
    challenge_type: ChallengeType,
    target_action: str,
    target_value: int,
    xp_reward: int,
    duration_days: int = 7,
    current_user: UserResponse = Depends(get_admin_user),
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
