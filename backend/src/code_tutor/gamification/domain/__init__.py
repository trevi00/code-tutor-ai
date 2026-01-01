"""Gamification domain layer."""

from .value_objects import (
    BadgeRarity,
    BadgeCategory,
    ChallengeType,
    ChallengeStatus,
    XP_REWARDS,
    LEVEL_THRESHOLDS,
    LEVEL_TITLES,
    calculate_level,
    xp_for_next_level,
    get_level_title,
)
from .entities import (
    Badge,
    UserBadge,
    UserStats,
    Challenge,
    UserChallenge,
    LeaderboardEntry,
    PREDEFINED_BADGES,
)
from .repository import (
    BadgeRepository,
    UserBadgeRepository,
    UserStatsRepository,
    ChallengeRepository,
    UserChallengeRepository,
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
