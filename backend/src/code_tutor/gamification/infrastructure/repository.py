"""Gamification repository implementations."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from code_tutor.gamification.domain.entities import (
    Badge,
    UserBadge,
    UserStats,
    Challenge,
    UserChallenge,
    LeaderboardEntry,
)
from code_tutor.gamification.domain.repository import (
    BadgeRepository,
    UserBadgeRepository,
    UserStatsRepository,
    ChallengeRepository,
    UserChallengeRepository,
)
from code_tutor.gamification.domain.value_objects import (
    BadgeCategory,
    BadgeRarity,
    ChallengeType,
    ChallengeStatus,
    calculate_level,
    get_level_title,
)
from .models import (
    BadgeModel,
    UserBadgeModel,
    UserStatsModel,
    ChallengeModel,
    UserChallengeModel,
)


def _model_to_badge(model: BadgeModel) -> Badge:
    """Convert model to entity."""
    return Badge(
        id=model.id,
        name=model.name,
        description=model.description,
        icon=model.icon,
        rarity=model.rarity,
        category=model.category,
        requirement=model.requirement,
        requirement_value=model.requirement_value,
        xp_reward=model.xp_reward,
        created_at=model.created_at,
    )


def _model_to_user_badge(model: UserBadgeModel) -> UserBadge:
    """Convert model to entity."""
    user_badge = UserBadge(
        id=model.id,
        user_id=model.user_id,
        badge_id=model.badge_id,
        earned_at=model.earned_at,
    )
    if model.badge:
        user_badge.badge = _model_to_badge(model.badge)
    return user_badge


def _model_to_user_stats(model: UserStatsModel) -> UserStats:
    """Convert model to entity."""
    return UserStats(
        id=model.id,
        user_id=model.user_id,
        total_xp=model.total_xp,
        current_streak=model.current_streak,
        longest_streak=model.longest_streak,
        problems_solved=model.problems_solved,
        problems_solved_first_try=model.problems_solved_first_try,
        patterns_mastered=model.patterns_mastered,
        collaborations_count=model.collaborations_count,
        playgrounds_created=model.playgrounds_created,
        playgrounds_shared=model.playgrounds_shared,
        last_activity_date=model.last_activity_date,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _model_to_challenge(model: ChallengeModel) -> Challenge:
    """Convert model to entity."""
    return Challenge(
        id=model.id,
        name=model.name,
        description=model.description,
        challenge_type=model.challenge_type,
        target_action=model.target_action,
        target_value=model.target_value,
        xp_reward=model.xp_reward,
        start_date=model.start_date,
        end_date=model.end_date,
        created_at=model.created_at,
    )


def _model_to_user_challenge(model: UserChallengeModel) -> UserChallenge:
    """Convert model to entity."""
    user_challenge = UserChallenge(
        id=model.id,
        user_id=model.user_id,
        challenge_id=model.challenge_id,
        current_progress=model.current_progress,
        status=model.status,
        started_at=model.started_at,
        completed_at=model.completed_at,
    )
    if model.challenge:
        user_challenge.challenge = _model_to_challenge(model.challenge)
    return user_challenge


class SQLAlchemyBadgeRepository(BadgeRepository):
    """SQLAlchemy badge repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, badge_id: UUID) -> Optional[Badge]:
        result = await self.session.execute(
            select(BadgeModel).where(BadgeModel.id == badge_id)
        )
        model = result.scalar_one_or_none()
        return _model_to_badge(model) if model else None

    async def get_all(self) -> list[Badge]:
        result = await self.session.execute(
            select(BadgeModel).order_by(BadgeModel.rarity, BadgeModel.name)
        )
        return [_model_to_badge(m) for m in result.scalars().all()]

    async def get_by_category(self, category: BadgeCategory) -> list[Badge]:
        result = await self.session.execute(
            select(BadgeModel)
            .where(BadgeModel.category == category)
            .order_by(BadgeModel.rarity)
        )
        return [_model_to_badge(m) for m in result.scalars().all()]

    async def create(self, badge: Badge) -> Badge:
        model = BadgeModel(
            id=badge.id,
            name=badge.name,
            description=badge.description,
            icon=badge.icon,
            rarity=badge.rarity,
            category=badge.category,
            requirement=badge.requirement,
            requirement_value=badge.requirement_value,
            xp_reward=badge.xp_reward,
            created_at=badge.created_at,
        )
        self.session.add(model)
        await self.session.flush()
        return badge

    async def exists_by_name(self, name: str) -> bool:
        result = await self.session.execute(
            select(BadgeModel.id).where(BadgeModel.name == name)
        )
        return result.scalar_one_or_none() is not None


class SQLAlchemyUserBadgeRepository(UserBadgeRepository):
    """SQLAlchemy user badge repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user(self, user_id: UUID) -> list[UserBadge]:
        result = await self.session.execute(
            select(UserBadgeModel)
            .options(selectinload(UserBadgeModel.badge))
            .where(UserBadgeModel.user_id == user_id)
            .order_by(desc(UserBadgeModel.earned_at))
        )
        return [_model_to_user_badge(m) for m in result.scalars().all()]

    async def has_badge(self, user_id: UUID, badge_id: UUID) -> bool:
        result = await self.session.execute(
            select(UserBadgeModel.id).where(
                UserBadgeModel.user_id == user_id,
                UserBadgeModel.badge_id == badge_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def award_badge(self, user_badge: UserBadge) -> UserBadge:
        model = UserBadgeModel(
            id=user_badge.id,
            user_id=user_badge.user_id,
            badge_id=user_badge.badge_id,
            earned_at=user_badge.earned_at,
        )
        self.session.add(model)
        await self.session.flush()
        return user_badge

    async def get_recent(self, limit: int = 10) -> list[UserBadge]:
        result = await self.session.execute(
            select(UserBadgeModel)
            .options(selectinload(UserBadgeModel.badge))
            .order_by(desc(UserBadgeModel.earned_at))
            .limit(limit)
        )
        return [_model_to_user_badge(m) for m in result.scalars().all()]


class SQLAlchemyUserStatsRepository(UserStatsRepository):
    """SQLAlchemy user stats repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user(self, user_id: UUID) -> Optional[UserStats]:
        result = await self.session.execute(
            select(UserStatsModel).where(UserStatsModel.user_id == user_id)
        )
        model = result.scalar_one_or_none()
        return _model_to_user_stats(model) if model else None

    async def create(self, stats: UserStats) -> UserStats:
        model = UserStatsModel(
            id=stats.id,
            user_id=stats.user_id,
            total_xp=stats.total_xp,
            current_streak=stats.current_streak,
            longest_streak=stats.longest_streak,
            problems_solved=stats.problems_solved,
            problems_solved_first_try=stats.problems_solved_first_try,
            patterns_mastered=stats.patterns_mastered,
            collaborations_count=stats.collaborations_count,
            playgrounds_created=stats.playgrounds_created,
            playgrounds_shared=stats.playgrounds_shared,
            last_activity_date=stats.last_activity_date,
            created_at=stats.created_at,
            updated_at=stats.updated_at,
        )
        self.session.add(model)
        await self.session.flush()
        return stats

    async def update(self, stats: UserStats) -> UserStats:
        result = await self.session.execute(
            select(UserStatsModel).where(UserStatsModel.user_id == stats.user_id)
        )
        model = result.scalar_one_or_none()
        if model:
            model.total_xp = stats.total_xp
            model.current_streak = stats.current_streak
            model.longest_streak = stats.longest_streak
            model.problems_solved = stats.problems_solved
            model.problems_solved_first_try = stats.problems_solved_first_try
            model.patterns_mastered = stats.patterns_mastered
            model.collaborations_count = stats.collaborations_count
            model.playgrounds_created = stats.playgrounds_created
            model.playgrounds_shared = stats.playgrounds_shared
            model.last_activity_date = stats.last_activity_date
            model.updated_at = datetime.utcnow()
            await self.session.flush()
        return stats

    async def get_or_create(self, user_id: UUID) -> UserStats:
        stats = await self.get_by_user(user_id)
        if stats:
            return stats
        new_stats = UserStats.create(user_id)
        return await self.create(new_stats)

    async def get_leaderboard(
        self,
        limit: int = 100,
        offset: int = 0,
        period: Optional[str] = None,
    ) -> list[LeaderboardEntry]:
        # Import here to avoid circular import
        from code_tutor.auth.infrastructure.models import UserModel

        query = (
            select(UserStatsModel, UserModel.username)
            .join(UserModel, UserStatsModel.user_id == UserModel.id)
            .order_by(desc(UserStatsModel.total_xp))
        )

        # Filter by period if specified
        if period == "weekly":
            week_ago = datetime.utcnow() - timedelta(days=7)
            query = query.where(UserStatsModel.last_activity_date >= week_ago)
        elif period == "monthly":
            month_ago = datetime.utcnow() - timedelta(days=30)
            query = query.where(UserStatsModel.last_activity_date >= month_ago)

        query = query.offset(offset).limit(limit)
        result = await self.session.execute(query)

        entries = []
        for idx, (stats, username) in enumerate(result.all(), start=offset + 1):
            level = calculate_level(stats.total_xp)
            entries.append(
                LeaderboardEntry(
                    user_id=stats.user_id,
                    username=username,
                    total_xp=stats.total_xp,
                    level=level,
                    level_title=get_level_title(level),
                    rank=idx,
                    problems_solved=stats.problems_solved,
                    current_streak=stats.current_streak,
                )
            )
        return entries

    async def get_user_rank(self, user_id: UUID) -> int:
        # Get user's XP
        result = await self.session.execute(
            select(UserStatsModel.total_xp).where(UserStatsModel.user_id == user_id)
        )
        user_xp = result.scalar_one_or_none()
        if user_xp is None:
            return 0

        # Count users with higher XP
        result = await self.session.execute(
            select(func.count(UserStatsModel.id)).where(
                UserStatsModel.total_xp > user_xp
            )
        )
        higher_count = result.scalar() or 0
        return higher_count + 1


class SQLAlchemyChallengeRepository(ChallengeRepository):
    """SQLAlchemy challenge repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, challenge_id: UUID) -> Optional[Challenge]:
        result = await self.session.execute(
            select(ChallengeModel).where(ChallengeModel.id == challenge_id)
        )
        model = result.scalar_one_or_none()
        return _model_to_challenge(model) if model else None

    async def get_active(
        self,
        challenge_type: Optional[ChallengeType] = None,
    ) -> list[Challenge]:
        now = datetime.utcnow()
        query = select(ChallengeModel).where(
            ChallengeModel.start_date <= now,
            ChallengeModel.end_date >= now,
        )
        if challenge_type:
            query = query.where(ChallengeModel.challenge_type == challenge_type)
        query = query.order_by(ChallengeModel.end_date)

        result = await self.session.execute(query)
        return [_model_to_challenge(m) for m in result.scalars().all()]

    async def create(self, challenge: Challenge) -> Challenge:
        model = ChallengeModel(
            id=challenge.id,
            name=challenge.name,
            description=challenge.description,
            challenge_type=challenge.challenge_type,
            target_action=challenge.target_action,
            target_value=challenge.target_value,
            xp_reward=challenge.xp_reward,
            start_date=challenge.start_date,
            end_date=challenge.end_date,
            created_at=challenge.created_at,
        )
        self.session.add(model)
        await self.session.flush()
        return challenge


class SQLAlchemyUserChallengeRepository(UserChallengeRepository):
    """SQLAlchemy user challenge repository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> list[UserChallenge]:
        query = (
            select(UserChallengeModel)
            .options(selectinload(UserChallengeModel.challenge))
            .where(UserChallengeModel.user_id == user_id)
        )
        if active_only:
            query = query.where(UserChallengeModel.status == ChallengeStatus.ACTIVE)
        query = query.order_by(UserChallengeModel.started_at)

        result = await self.session.execute(query)
        return [_model_to_user_challenge(m) for m in result.scalars().all()]

    async def get_by_user_and_challenge(
        self,
        user_id: UUID,
        challenge_id: UUID,
    ) -> Optional[UserChallenge]:
        result = await self.session.execute(
            select(UserChallengeModel)
            .options(selectinload(UserChallengeModel.challenge))
            .where(
                UserChallengeModel.user_id == user_id,
                UserChallengeModel.challenge_id == challenge_id,
            )
        )
        model = result.scalar_one_or_none()
        return _model_to_user_challenge(model) if model else None

    async def create(self, user_challenge: UserChallenge) -> UserChallenge:
        model = UserChallengeModel(
            id=user_challenge.id,
            user_id=user_challenge.user_id,
            challenge_id=user_challenge.challenge_id,
            current_progress=user_challenge.current_progress,
            status=user_challenge.status,
            started_at=user_challenge.started_at,
            completed_at=user_challenge.completed_at,
        )
        self.session.add(model)
        await self.session.flush()
        return user_challenge

    async def update(self, user_challenge: UserChallenge) -> UserChallenge:
        result = await self.session.execute(
            select(UserChallengeModel).where(
                UserChallengeModel.id == user_challenge.id
            )
        )
        model = result.scalar_one_or_none()
        if model:
            model.current_progress = user_challenge.current_progress
            model.status = user_challenge.status
            model.completed_at = user_challenge.completed_at
            await self.session.flush()
        return user_challenge
