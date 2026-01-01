"""Gamification infrastructure layer."""

from .models import (
    BadgeModel,
    UserBadgeModel,
    UserStatsModel,
    ChallengeModel,
    UserChallengeModel,
)
from .repository import (
    SQLAlchemyBadgeRepository,
    SQLAlchemyUserBadgeRepository,
    SQLAlchemyUserStatsRepository,
    SQLAlchemyChallengeRepository,
    SQLAlchemyUserChallengeRepository,
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
