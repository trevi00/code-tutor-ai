"""Gamification DTOs."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from code_tutor.gamification.domain.value_objects import (
    BadgeRarity,
    BadgeCategory,
    ChallengeType,
    ChallengeStatus,
)


# Badge DTOs
class BadgeResponse(BaseModel):
    """Badge response."""

    id: UUID
    name: str
    description: str
    icon: str
    rarity: BadgeRarity
    category: BadgeCategory
    requirement: str
    requirement_value: int
    xp_reward: int

    class Config:
        from_attributes = True


class UserBadgeResponse(BaseModel):
    """User badge response."""

    id: UUID
    badge: BadgeResponse
    earned_at: datetime

    class Config:
        from_attributes = True


class BadgesListResponse(BaseModel):
    """Badges list response."""

    badges: list[BadgeResponse]
    total: int


class UserBadgesResponse(BaseModel):
    """User badges response."""

    earned: list[UserBadgeResponse]
    available: list[BadgeResponse]
    total_earned: int
    total_available: int


# Stats DTOs
class UserStatsResponse(BaseModel):
    """User stats response."""

    total_xp: int
    level: int
    level_title: str
    xp_progress: int = Field(description="XP progress in current level")
    xp_for_next_level: int = Field(description="XP needed for next level")
    xp_percentage: float
    current_streak: int
    longest_streak: int
    problems_solved: int
    problems_solved_first_try: int
    patterns_mastered: int
    collaborations_count: int
    playgrounds_created: int
    playgrounds_shared: int
    # Roadmap progress
    lessons_completed: int = 0
    paths_completed: int = 0

    class Config:
        from_attributes = True


class AddXPRequest(BaseModel):
    """Request to add XP."""

    action: str = Field(description="Action that earned XP")
    custom_amount: Optional[int] = Field(
        None, description="Custom XP amount (overrides action default)"
    )


class XPAddedResponse(BaseModel):
    """XP added response."""

    xp_added: int
    total_xp: int
    level: int
    level_title: str
    leveled_up: bool
    new_badges: list[BadgeResponse] = []


# Leaderboard DTOs
class LeaderboardEntryResponse(BaseModel):
    """Leaderboard entry response."""

    rank: int
    user_id: UUID
    username: str
    total_xp: int
    level: int
    level_title: str
    problems_solved: int
    current_streak: int


class LeaderboardResponse(BaseModel):
    """Leaderboard response."""

    entries: list[LeaderboardEntryResponse]
    period: str
    total_users: int
    user_rank: Optional[int] = None


# Challenge DTOs
class ChallengeResponse(BaseModel):
    """Challenge response."""

    id: UUID
    name: str
    description: str
    challenge_type: ChallengeType
    target_action: str
    target_value: int
    xp_reward: int
    start_date: datetime
    end_date: datetime
    time_remaining: Optional[str] = None

    class Config:
        from_attributes = True


class UserChallengeResponse(BaseModel):
    """User challenge response."""

    id: UUID
    challenge: ChallengeResponse
    current_progress: int
    status: ChallengeStatus
    progress_percentage: float
    started_at: datetime
    completed_at: Optional[datetime] = None


class ChallengesResponse(BaseModel):
    """Challenges list response."""

    active: list[UserChallengeResponse]
    completed: list[UserChallengeResponse]
    available: list[ChallengeResponse]


# Gamification Overview DTO
class GamificationOverviewResponse(BaseModel):
    """Complete gamification overview."""

    stats: UserStatsResponse
    recent_badges: list[UserBadgeResponse]
    active_challenges: list[UserChallengeResponse]
    leaderboard_rank: int
    next_badge_progress: Optional[dict] = None
