"""Gamification repository interfaces."""

from abc import ABC, abstractmethod
from uuid import UUID

from .entities import (
    Badge,
    Challenge,
    LeaderboardEntry,
    UserBadge,
    UserChallenge,
    UserStats,
)
from .value_objects import BadgeCategory, ChallengeType


class BadgeRepository(ABC):
    """Badge repository interface."""

    @abstractmethod
    async def get_by_id(self, badge_id: UUID) -> Badge | None:
        """Get badge by ID."""
        pass

    @abstractmethod
    async def get_all(self) -> list[Badge]:
        """Get all badges."""
        pass

    @abstractmethod
    async def get_by_category(self, category: BadgeCategory) -> list[Badge]:
        """Get badges by category."""
        pass

    @abstractmethod
    async def create(self, badge: Badge) -> Badge:
        """Create a badge."""
        pass

    @abstractmethod
    async def exists_by_name(self, name: str) -> bool:
        """Check if badge exists by name."""
        pass


class UserBadgeRepository(ABC):
    """User badge repository interface."""

    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> list[UserBadge]:
        """Get all badges earned by user."""
        pass

    @abstractmethod
    async def has_badge(self, user_id: UUID, badge_id: UUID) -> bool:
        """Check if user has a specific badge."""
        pass

    @abstractmethod
    async def award_badge(self, user_badge: UserBadge) -> UserBadge:
        """Award a badge to user."""
        pass

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[UserBadge]:
        """Get recently awarded badges."""
        pass


class UserStatsRepository(ABC):
    """User stats repository interface."""

    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> UserStats | None:
        """Get user stats."""
        pass

    @abstractmethod
    async def create(self, stats: UserStats) -> UserStats:
        """Create user stats."""
        pass

    @abstractmethod
    async def update(self, stats: UserStats) -> UserStats:
        """Update user stats."""
        pass

    @abstractmethod
    async def get_or_create(self, user_id: UUID) -> UserStats:
        """Get or create user stats."""
        pass

    @abstractmethod
    async def get_leaderboard(
        self,
        limit: int = 100,
        offset: int = 0,
        period: str | None = None,  # "weekly", "monthly", "all"
    ) -> list[LeaderboardEntry]:
        """Get leaderboard entries."""
        pass

    @abstractmethod
    async def get_user_rank(self, user_id: UUID) -> int:
        """Get user's rank on leaderboard."""
        pass


class ChallengeRepository(ABC):
    """Challenge repository interface."""

    @abstractmethod
    async def get_by_id(self, challenge_id: UUID) -> Challenge | None:
        """Get challenge by ID."""
        pass

    @abstractmethod
    async def get_active(self, challenge_type: ChallengeType | None = None) -> list[Challenge]:
        """Get active challenges."""
        pass

    @abstractmethod
    async def create(self, challenge: Challenge) -> Challenge:
        """Create a challenge."""
        pass


class UserChallengeRepository(ABC):
    """User challenge repository interface."""

    @abstractmethod
    async def get_by_user(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> list[UserChallenge]:
        """Get user's challenges."""
        pass

    @abstractmethod
    async def get_by_user_and_challenge(
        self,
        user_id: UUID,
        challenge_id: UUID,
    ) -> UserChallenge | None:
        """Get specific user challenge."""
        pass

    @abstractmethod
    async def create(self, user_challenge: UserChallenge) -> UserChallenge:
        """Create user challenge progress."""
        pass

    @abstractmethod
    async def update(self, user_challenge: UserChallenge) -> UserChallenge:
        """Update user challenge progress."""
        pass
