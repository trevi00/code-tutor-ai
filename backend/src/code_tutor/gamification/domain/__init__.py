"""Gamification domain layer."""

from .entities import (
    PREDEFINED_BADGES,
    Badge,
    Challenge,
    LeaderboardEntry,
    UserBadge,
    UserChallenge,
    UserStats,
)
from .repository import (
    BadgeRepository,
    ChallengeRepository,
    UserBadgeRepository,
    UserChallengeRepository,
    UserStatsRepository,
)
from .value_objects import (
    LEVEL_THRESHOLDS,
    LEVEL_TITLES,
    XP_REWARDS,
    BadgeCategory,
    BadgeRarity,
    ChallengeStatus,
    ChallengeType,
    calculate_level,
    get_level_title,
    xp_for_next_level,
)

__all__ = [
    # Value Objects
    "BadgeRarity",
    "BadgeCategory",
    "ChallengeType",
    "ChallengeStatus",
    "XP_REWARDS",
    "LEVEL_THRESHOLDS",
    "LEVEL_TITLES",
    "calculate_level",
    "xp_for_next_level",
    "get_level_title",
    # Entities
    "Badge",
    "UserBadge",
    "UserStats",
    "Challenge",
    "UserChallenge",
    "LeaderboardEntry",
    "PREDEFINED_BADGES",
    # Repositories
    "BadgeRepository",
    "UserBadgeRepository",
    "UserStatsRepository",
    "ChallengeRepository",
    "UserChallengeRepository",
]
