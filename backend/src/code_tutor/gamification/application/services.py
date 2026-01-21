"""Gamification services."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from code_tutor.gamification.domain.entities import (
    PREDEFINED_BADGES,
    Badge,
    Challenge,
    LeaderboardEntry,
    UserBadge,
    UserChallenge,
    UserStats,
)
from code_tutor.gamification.domain.repository import (
    BadgeRepository,
    ChallengeRepository,
    UserBadgeRepository,
    UserChallengeRepository,
    UserStatsRepository,
)
from code_tutor.gamification.domain.value_objects import (
    XP_REWARDS,
    ChallengeStatus,
    ChallengeType,
)

from .dto import (
    BadgeResponse,
    ChallengeResponse,
    ChallengesResponse,
    GamificationOverviewResponse,
    LeaderboardEntryResponse,
    LeaderboardResponse,
    UserBadgeResponse,
    UserBadgesResponse,
    UserChallengeResponse,
    UserStatsResponse,
    XPAddedResponse,
)


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)


class BadgeService:
    """Badge management service."""

    def __init__(
        self,
        badge_repo: BadgeRepository,
        user_badge_repo: UserBadgeRepository,
        user_stats_repo: UserStatsRepository,
    ):
        self.badge_repo = badge_repo
        self.user_badge_repo = user_badge_repo
        self.user_stats_repo = user_stats_repo

    async def seed_badges(self) -> int:
        """Seed predefined badges. Returns count of new badges."""
        count = 0
        for badge_data in PREDEFINED_BADGES:
            if not await self.badge_repo.exists_by_name(badge_data["name"]):
                badge = Badge.create(**badge_data)
                await self.badge_repo.create(badge)
                count += 1
        return count

    async def get_all_badges(self) -> list[BadgeResponse]:
        """Get all available badges."""
        badges = await self.badge_repo.get_all()
        return [self._to_badge_response(b) for b in badges]

    async def get_user_badges(self, user_id: UUID) -> UserBadgesResponse:
        """Get user's earned and available badges."""
        all_badges = await self.badge_repo.get_all()
        user_badges = await self.user_badge_repo.get_by_user(user_id)

        earned_ids = {ub.badge_id for ub in user_badges}
        available = [b for b in all_badges if b.id not in earned_ids]

        return UserBadgesResponse(
            earned=[self._to_user_badge_response(ub) for ub in user_badges],
            available=[self._to_badge_response(b) for b in available],
            total_earned=len(user_badges),
            total_available=len(available),
        )

    async def check_and_award_badges(self, user_id: UUID) -> list[BadgeResponse]:
        """Check if user qualifies for any new badges and award them."""
        stats = await self.user_stats_repo.get_or_create(user_id)
        all_badges = await self.badge_repo.get_all()
        new_badges = []

        for badge in all_badges:
            # Check if user already has badge
            if await self.user_badge_repo.has_badge(user_id, badge.id):
                continue

            # Check if user qualifies
            if self._check_badge_requirement(stats, badge):
                user_badge = UserBadge.create(user_id, badge.id)
                await self.user_badge_repo.award_badge(user_badge)

                # Award XP for badge
                if badge.xp_reward > 0:
                    stats.add_xp(badge.xp_reward)
                    await self.user_stats_repo.update(stats)

                new_badges.append(self._to_badge_response(badge))

        return new_badges

    def _check_badge_requirement(self, stats: UserStats, badge: Badge) -> bool:
        """Check if user stats meet badge requirement."""
        # Boolean fields (path level completions)
        bool_stat_mapping = {
            "beginner_path_completed": stats.beginner_path_completed,
            "elementary_path_completed": stats.elementary_path_completed,
            "intermediate_path_completed": stats.intermediate_path_completed,
            "advanced_path_completed": stats.advanced_path_completed,
        }

        # Check boolean requirements first
        if badge.requirement in bool_stat_mapping:
            return bool_stat_mapping[badge.requirement]

        # Numeric requirements
        stat_mapping = {
            "problems_solved": stats.problems_solved,
            "problems_solved_first_try": stats.problems_solved_first_try,
            "current_streak": stats.current_streak,
            "longest_streak": stats.longest_streak,
            "patterns_mastered": stats.patterns_mastered,
            "collaborations_count": stats.collaborations_count,
            "playgrounds_created": stats.playgrounds_created,
            "playgrounds_shared": stats.playgrounds_shared,
            "lessons_completed": stats.lessons_completed,
            "paths_completed": stats.paths_completed,
        }
        current_value = stat_mapping.get(badge.requirement, 0)
        return current_value >= badge.requirement_value

    def _to_badge_response(self, badge: Badge) -> BadgeResponse:
        return BadgeResponse(
            id=badge.id,
            name=badge.name,
            description=badge.description,
            icon=badge.icon,
            rarity=badge.rarity,
            category=badge.category,
            requirement=badge.requirement,
            requirement_value=badge.requirement_value,
            xp_reward=badge.xp_reward,
        )

    def _to_user_badge_response(self, user_badge: UserBadge) -> UserBadgeResponse:
        return UserBadgeResponse(
            id=user_badge.id,
            badge=self._to_badge_response(user_badge.badge),
            earned_at=user_badge.earned_at,
        )


class XPService:
    """XP and stats management service."""

    def __init__(
        self,
        user_stats_repo: UserStatsRepository,
        badge_service: BadgeService,
    ):
        self.user_stats_repo = user_stats_repo
        self.badge_service = badge_service

    async def get_user_stats(self, user_id: UUID) -> UserStatsResponse:
        """Get user's gamification stats."""
        stats = await self.user_stats_repo.get_or_create(user_id)
        return self._to_stats_response(stats)

    async def add_xp(
        self,
        user_id: UUID,
        action: str,
        custom_amount: int | None = None,
    ) -> XPAddedResponse:
        """Add XP for an action."""
        stats = await self.user_stats_repo.get_or_create(user_id)
        old_level = stats.level

        # Determine XP amount
        xp_amount = custom_amount if custom_amount else XP_REWARDS.get(action, 0)
        stats.add_xp(xp_amount)

        # Update streak
        stats.update_streak(utc_now())

        # Update specific counters based on action
        if action == "problem_solved":
            stats.increment_problems_solved(first_try=False)
        elif action == "problem_solved_first_try":
            stats.increment_problems_solved(first_try=True)
        elif action == "collaboration_session":
            stats.collaborations_count += 1
        elif action == "playground_created":
            stats.playgrounds_created += 1
        elif action == "playground_shared":
            stats.playgrounds_shared += 1
        elif action == "lesson_completed":
            stats.increment_lessons_completed()
        elif action == "path_completed":
            stats.increment_paths_completed()

        await self.user_stats_repo.update(stats)

        # Check for new badges
        new_badges = await self.badge_service.check_and_award_badges(user_id)

        leveled_up = stats.level > old_level

        return XPAddedResponse(
            xp_added=xp_amount,
            total_xp=stats.total_xp,
            level=stats.level,
            level_title=stats.level_title,
            leveled_up=leveled_up,
            new_badges=new_badges,
        )

    async def record_activity(self, user_id: UUID, action: str) -> XPAddedResponse:
        """Record an activity and award XP."""
        return await self.add_xp(user_id, action)

    async def set_path_level_completed(self, user_id: UUID, level: str) -> None:
        """Set a specific path level as completed for badge tracking."""
        stats = await self.user_stats_repo.get_or_create(user_id)
        stats.set_path_level_completed(level)
        await self.user_stats_repo.update(stats)

        # Check for new badges after setting the path level
        await self.badge_service.check_and_award_badges(user_id)

    def _to_stats_response(self, stats: UserStats) -> UserStatsResponse:
        current_xp, current_threshold, next_threshold = stats.xp_progress
        return UserStatsResponse(
            total_xp=stats.total_xp,
            level=stats.level,
            level_title=stats.level_title,
            xp_progress=current_xp - current_threshold,
            xp_for_next_level=next_threshold - current_threshold,
            xp_percentage=stats.xp_percentage,
            current_streak=stats.current_streak,
            longest_streak=stats.longest_streak,
            problems_solved=stats.problems_solved,
            problems_solved_first_try=stats.problems_solved_first_try,
            patterns_mastered=stats.patterns_mastered,
            collaborations_count=stats.collaborations_count,
            playgrounds_created=stats.playgrounds_created,
            playgrounds_shared=stats.playgrounds_shared,
            lessons_completed=stats.lessons_completed,
            paths_completed=stats.paths_completed,
        )


class LeaderboardService:
    """Leaderboard service."""

    def __init__(self, user_stats_repo: UserStatsRepository):
        self.user_stats_repo = user_stats_repo

    async def get_leaderboard(
        self,
        period: str = "all",
        limit: int = 100,
        offset: int = 0,
        user_id: UUID | None = None,
    ) -> LeaderboardResponse:
        """Get leaderboard."""
        entries = await self.user_stats_repo.get_leaderboard(
            limit=limit,
            offset=offset,
            period=period if period != "all" else None,
        )

        user_rank = None
        if user_id:
            user_rank = await self.user_stats_repo.get_user_rank(user_id)

        return LeaderboardResponse(
            entries=[self._to_entry_response(e) for e in entries],
            period=period,
            total_users=len(entries),  # Could be improved with a count query
            user_rank=user_rank,
        )

    def _to_entry_response(self, entry: LeaderboardEntry) -> LeaderboardEntryResponse:
        return LeaderboardEntryResponse(
            rank=entry.rank,
            user_id=entry.user_id,
            username=entry.username,
            total_xp=entry.total_xp,
            level=entry.level,
            level_title=entry.level_title,
            problems_solved=entry.problems_solved,
            current_streak=entry.current_streak,
        )


class ChallengeService:
    """Challenge management service."""

    def __init__(
        self,
        challenge_repo: ChallengeRepository,
        user_challenge_repo: UserChallengeRepository,
        user_stats_repo: UserStatsRepository,
    ):
        self.challenge_repo = challenge_repo
        self.user_challenge_repo = user_challenge_repo
        self.user_stats_repo = user_stats_repo

    async def create_challenge(
        self,
        name: str,
        description: str,
        challenge_type: ChallengeType,
        target_action: str,
        target_value: int,
        xp_reward: int,
        duration_days: int,
    ) -> ChallengeResponse:
        """Create a new challenge."""
        now = utc_now()
        challenge = Challenge.create(
            name=name,
            description=description,
            challenge_type=challenge_type,
            target_action=target_action,
            target_value=target_value,
            xp_reward=xp_reward,
            start_date=now,
            end_date=now + timedelta(days=duration_days),
        )
        await self.challenge_repo.create(challenge)
        return self._to_challenge_response(challenge)

    async def get_user_challenges(self, user_id: UUID) -> ChallengesResponse:
        """Get user's challenges."""
        active_challenges = await self.challenge_repo.get_active()
        user_challenges = await self.user_challenge_repo.get_by_user(
            user_id, active_only=False
        )

        user_challenge_ids = {uc.challenge_id for uc in user_challenges}
        available = [c for c in active_challenges if c.id not in user_challenge_ids]

        active = [uc for uc in user_challenges if uc.status == ChallengeStatus.ACTIVE]
        completed = [
            uc for uc in user_challenges if uc.status == ChallengeStatus.COMPLETED
        ]

        return ChallengesResponse(
            active=[self._to_user_challenge_response(uc) for uc in active],
            completed=[self._to_user_challenge_response(uc) for uc in completed],
            available=[self._to_challenge_response(c) for c in available],
        )

    async def join_challenge(
        self, user_id: UUID, challenge_id: UUID
    ) -> UserChallengeResponse:
        """Join a challenge."""
        # Check if already joined
        existing = await self.user_challenge_repo.get_by_user_and_challenge(
            user_id, challenge_id
        )
        if existing:
            return self._to_user_challenge_response(existing)

        # Get challenge
        challenge = await self.challenge_repo.get_by_id(challenge_id)
        if not challenge:
            raise ValueError("Challenge not found")

        user_challenge = UserChallenge.create(user_id, challenge_id)
        user_challenge.challenge = challenge
        await self.user_challenge_repo.create(user_challenge)

        return self._to_user_challenge_response(user_challenge)

    async def update_challenge_progress(
        self, user_id: UUID, action: str
    ) -> list[UserChallengeResponse]:
        """Update progress for all active challenges matching the action."""
        stats = await self.user_stats_repo.get_or_create(user_id)
        user_challenges = await self.user_challenge_repo.get_by_user(
            user_id, active_only=True
        )

        updated = []
        for uc in user_challenges:
            if uc.challenge and uc.challenge.target_action == action:
                # Get current value for this action
                progress = self._get_progress_for_action(stats, action)
                completed = uc.update_progress(progress)
                await self.user_challenge_repo.update(uc)

                if completed:
                    # Award XP for completion
                    stats.add_xp(uc.challenge.xp_reward)
                    await self.user_stats_repo.update(stats)

                updated.append(self._to_user_challenge_response(uc))

        return updated

    def _get_progress_for_action(self, stats: UserStats, action: str) -> int:
        """Get current progress value for an action."""
        mapping = {
            "solve_problems": stats.problems_solved,
            "maintain_streak": stats.current_streak,
            "complete_patterns": stats.patterns_mastered,
            "collaborate": stats.collaborations_count,
        }
        return mapping.get(action, 0)

    def _to_challenge_response(self, challenge: Challenge) -> ChallengeResponse:
        time_remaining = None
        if challenge.is_active:
            now = utc_now()
            # Ensure timezone-aware comparison
            end_date = (
                challenge.end_date.replace(tzinfo=UTC)
                if challenge.end_date.tzinfo is None
                else challenge.end_date
            )
            delta = end_date - now
            if delta.days > 0:
                time_remaining = f"{delta.days}일 남음"
            elif delta.seconds > 3600:
                time_remaining = f"{delta.seconds // 3600}시간 남음"
            else:
                time_remaining = f"{delta.seconds // 60}분 남음"

        return ChallengeResponse(
            id=challenge.id,
            name=challenge.name,
            description=challenge.description,
            challenge_type=challenge.challenge_type,
            target_action=challenge.target_action,
            target_value=challenge.target_value,
            xp_reward=challenge.xp_reward,
            start_date=challenge.start_date,
            end_date=challenge.end_date,
            time_remaining=time_remaining,
        )

    def _to_user_challenge_response(
        self, user_challenge: UserChallenge
    ) -> UserChallengeResponse:
        challenge_response = None
        if user_challenge.challenge:
            challenge_response = self._to_challenge_response(user_challenge.challenge)
        return UserChallengeResponse(
            id=user_challenge.id,
            challenge=challenge_response,
            current_progress=user_challenge.current_progress,
            status=user_challenge.status,
            progress_percentage=user_challenge.progress_percentage,
            started_at=user_challenge.started_at,
            completed_at=user_challenge.completed_at,
        )


class GamificationService:
    """Main gamification service for overview."""

    def __init__(
        self,
        badge_service: BadgeService,
        xp_service: XPService,
        leaderboard_service: LeaderboardService,
        challenge_service: ChallengeService,
    ):
        self.badge_service = badge_service
        self.xp_service = xp_service
        self.leaderboard_service = leaderboard_service
        self.challenge_service = challenge_service

    async def get_overview(self, user_id: UUID) -> GamificationOverviewResponse:
        """Get complete gamification overview for user."""
        stats = await self.xp_service.get_user_stats(user_id)
        user_badges = await self.badge_service.get_user_badges(user_id)
        challenges = await self.challenge_service.get_user_challenges(user_id)
        leaderboard = await self.leaderboard_service.get_leaderboard(
            period="all", limit=1, user_id=user_id
        )

        # Get next badge progress
        next_badge_progress = None
        if user_badges.available:
            # Find closest badge to achieve
            closest = min(user_badges.available, key=lambda b: b.requirement_value)
            current_value = self._get_stat_value(stats, closest.requirement)
            next_badge_progress = {
                "badge": closest.name,
                "current": current_value,
                "required": closest.requirement_value,
                "percentage": min(
                    100, (current_value / closest.requirement_value) * 100
                ),
            }

        return GamificationOverviewResponse(
            stats=stats,
            recent_badges=user_badges.earned[:5],
            active_challenges=challenges.active,
            leaderboard_rank=leaderboard.user_rank or 0,
            next_badge_progress=next_badge_progress,
        )

    def _get_stat_value(self, stats: UserStatsResponse, requirement: str) -> int:
        """Get stat value from stats response."""
        mapping = {
            "problems_solved": stats.problems_solved,
            "problems_solved_first_try": stats.problems_solved_first_try,
            "current_streak": stats.current_streak,
            "longest_streak": stats.longest_streak,
            "patterns_mastered": stats.patterns_mastered,
            "collaborations_count": stats.collaborations_count,
            "playgrounds_created": stats.playgrounds_created,
            "playgrounds_shared": stats.playgrounds_shared,
            "lessons_completed": stats.lessons_completed,
            "paths_completed": stats.paths_completed,
        }
        return mapping.get(requirement, 0)
