"""Gamification infrastructure layer."""

from .models import (
    BadgeModel,
    ChallengeModel,
    UserBadgeModel,
    UserChallengeModel,
    UserStatsModel,
)
from .repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyChallengeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserChallengeRepository,
    SQLAlchemyUserStatsRepository,
)

__all__ = [
    # Models
    "BadgeModel",
    "UserBadgeModel",
    "UserStatsModel",
    "ChallengeModel",
    "UserChallengeModel",
    # Repositories
    "SQLAlchemyBadgeRepository",
    "SQLAlchemyUserBadgeRepository",
    "SQLAlchemyUserStatsRepository",
    "SQLAlchemyChallengeRepository",
    "SQLAlchemyUserChallengeRepository",
]
