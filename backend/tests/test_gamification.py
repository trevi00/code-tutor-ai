"""Tests for Gamification Module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from code_tutor.gamification.domain.value_objects import (
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
from code_tutor.gamification.domain.entities import (
    Badge,
    UserBadge,
    UserStats,
    Challenge,
    UserChallenge,
    LeaderboardEntry,
    PREDEFINED_BADGES,
    utc_now,
)


# ============= Value Objects Tests =============

class TestBadgeRarity:
    """Tests for BadgeRarity enum."""

    def test_rarity_values(self):
        """Test all rarity values exist."""
        assert BadgeRarity.COMMON.value == "common"
        assert BadgeRarity.UNCOMMON.value == "uncommon"
        assert BadgeRarity.RARE.value == "rare"
        assert BadgeRarity.EPIC.value == "epic"
        assert BadgeRarity.LEGENDARY.value == "legendary"

    def test_rarity_from_string(self):
        """Test creating rarity from string."""
        assert BadgeRarity("common") == BadgeRarity.COMMON
        assert BadgeRarity("legendary") == BadgeRarity.LEGENDARY


class TestBadgeCategory:
    """Tests for BadgeCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert BadgeCategory.PROBLEM_SOLVING.value == "problem_solving"
        assert BadgeCategory.STREAK.value == "streak"
        assert BadgeCategory.MASTERY.value == "mastery"
        assert BadgeCategory.SOCIAL.value == "social"
        assert BadgeCategory.SPECIAL.value == "special"


class TestChallengeType:
    """Tests for ChallengeType enum."""

    def test_challenge_type_values(self):
        """Test all challenge type values exist."""
        assert ChallengeType.DAILY.value == "daily"
        assert ChallengeType.WEEKLY.value == "weekly"
        assert ChallengeType.MONTHLY.value == "monthly"
        assert ChallengeType.SPECIAL.value == "special"


class TestChallengeStatus:
    """Tests for ChallengeStatus enum."""

    def test_challenge_status_values(self):
        """Test all challenge status values exist."""
        assert ChallengeStatus.ACTIVE.value == "active"
        assert ChallengeStatus.COMPLETED.value == "completed"
        assert ChallengeStatus.EXPIRED.value == "expired"


class TestXPRewards:
    """Tests for XP_REWARDS constants."""

    def test_xp_rewards_contains_expected_actions(self):
        """Test XP rewards has expected actions."""
        expected_actions = [
            "problem_solved",
            "problem_solved_first_try",
            "daily_login",
            "streak_bonus",
            "collaboration_session",
            "playground_created",
            "playground_shared",
            "typing_attempt_completed",
            "lesson_completed",
            "path_completed",
        ]
        for action in expected_actions:
            assert action in XP_REWARDS
            assert XP_REWARDS[action] > 0

    def test_first_try_more_than_regular(self):
        """Test first try gives more XP than regular solve."""
        assert XP_REWARDS["problem_solved_first_try"] > XP_REWARDS["problem_solved"]


class TestLevelThresholds:
    """Tests for level thresholds."""

    def test_level_thresholds_ascending(self):
        """Test level thresholds are ascending."""
        for i in range(1, len(LEVEL_THRESHOLDS)):
            assert LEVEL_THRESHOLDS[i] > LEVEL_THRESHOLDS[i - 1]

    def test_level_thresholds_start_at_zero(self):
        """Test first threshold is 0."""
        assert LEVEL_THRESHOLDS[0] == 0


class TestCalculateLevel:
    """Tests for calculate_level function."""

    def test_level_1_at_zero_xp(self):
        """Test level 1 at 0 XP."""
        assert calculate_level(0) == 1

    def test_level_1_below_threshold(self):
        """Test level 1 below second threshold."""
        assert calculate_level(50) == 1
        assert calculate_level(99) == 1

    def test_level_2_at_threshold(self):
        """Test level 2 at threshold."""
        assert calculate_level(100) == 2

    def test_level_progression(self):
        """Test level progression."""
        # Level 1: 0-99
        assert calculate_level(0) == 1
        # Level 2: 100-249
        assert calculate_level(100) == 2
        assert calculate_level(249) == 2
        # Level 3: 250-499
        assert calculate_level(250) == 3
        assert calculate_level(499) == 3
        # Level 4: 500-849
        assert calculate_level(500) == 4

    def test_max_level(self):
        """Test max level for very high XP."""
        # XP above all thresholds
        assert calculate_level(100000) == len(LEVEL_THRESHOLDS)


class TestXpForNextLevel:
    """Tests for xp_for_next_level function."""

    def test_xp_for_next_level_at_level_1(self):
        """Test XP thresholds at level 1."""
        current, next_thresh = xp_for_next_level(50)
        assert current == 0  # Level 1 starts at 0
        assert next_thresh == 100  # Level 2 starts at 100

    def test_xp_for_next_level_at_level_2(self):
        """Test XP thresholds at level 2."""
        current, next_thresh = xp_for_next_level(150)
        assert current == 100  # Level 2 starts at 100
        assert next_thresh == 250  # Level 3 starts at 250

    def test_xp_for_next_level_at_max(self):
        """Test XP thresholds at max level."""
        current, next_thresh = xp_for_next_level(100000)
        assert current == next_thresh  # At max level


class TestGetLevelTitle:
    """Tests for get_level_title function."""

    def test_level_titles_exist(self):
        """Test level titles exist for levels 1-20."""
        for level in range(1, 21):
            title = get_level_title(level)
            assert title is not None
            assert len(title) > 0

    def test_specific_titles(self):
        """Test specific level titles."""
        assert get_level_title(1) == "입문자"
        assert get_level_title(5) == "도전자"
        assert get_level_title(10) == "그랜드마스터"
        assert get_level_title(20) == "무한"

    def test_beyond_max_level(self):
        """Test title for level beyond defined."""
        # Should return max level title
        title = get_level_title(100)
        assert title == LEVEL_TITLES[max(LEVEL_TITLES.keys())]


# ============= Entities Tests =============

class TestBadge:
    """Tests for Badge entity."""

    def test_badge_creation(self):
        """Test creating a badge."""
        badge_id = uuid4()
        badge = Badge(
            id=badge_id,
            name="First Steps",
            description="Solve your first problem",
            icon="🎯",
            rarity=BadgeRarity.COMMON,
            category=BadgeCategory.PROBLEM_SOLVING,
            requirement="problems_solved",
            requirement_value=1,
            xp_reward=50,
        )
        assert badge.id == badge_id
        assert badge.name == "First Steps"
        assert badge.rarity == BadgeRarity.COMMON
        assert badge.category == BadgeCategory.PROBLEM_SOLVING
        assert badge.requirement_value == 1
        assert badge.xp_reward == 50

    def test_badge_create_factory(self):
        """Test badge factory method."""
        badge = Badge.create(
            name="Test Badge",
            description="Test description",
            icon="⭐",
            rarity=BadgeRarity.RARE,
            category=BadgeCategory.MASTERY,
            requirement="patterns_mastered",
            requirement_value=5,
            xp_reward=100,
        )
        assert badge.id is not None
        assert badge.name == "Test Badge"
        assert badge.rarity == BadgeRarity.RARE


class TestUserBadge:
    """Tests for UserBadge entity."""

    def test_user_badge_creation(self):
        """Test creating a user badge."""
        user_badge_id = uuid4()
        user_id = uuid4()
        badge_id = uuid4()
        user_badge = UserBadge(
            id=user_badge_id,
            user_id=user_id,
            badge_id=badge_id,
        )
        assert user_badge.id == user_badge_id
        assert user_badge.user_id == user_id
        assert user_badge.badge_id == badge_id
        assert user_badge.badge is None

    def test_user_badge_create_factory(self):
        """Test user badge factory method."""
        user_id = uuid4()
        badge_id = uuid4()
        user_badge = UserBadge.create(user_id, badge_id)
        assert user_badge.id is not None
        assert user_badge.user_id == user_id
        assert user_badge.badge_id == badge_id


class TestUserStats:
    """Tests for UserStats entity."""

    def test_user_stats_creation(self):
        """Test creating user stats."""
        stats_id = uuid4()
        user_id = uuid4()
        stats = UserStats(
            id=stats_id,
            user_id=user_id,
            total_xp=500,
            current_streak=7,
            problems_solved=20,
        )
        assert stats.id == stats_id
        assert stats.user_id == user_id
        assert stats.total_xp == 500
        assert stats.current_streak == 7
        assert stats.problems_solved == 20

    def test_user_stats_create_factory(self):
        """Test user stats factory method."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        assert stats.id is not None
        assert stats.user_id == user_id
        assert stats.total_xp == 0
        assert stats.current_streak == 0
        assert stats.problems_solved == 0

    def test_user_stats_defaults(self):
        """Test user stats default values."""
        stats = UserStats(id=uuid4(), user_id=uuid4())
        assert stats.total_xp == 0
        assert stats.current_streak == 0
        assert stats.longest_streak == 0
        assert stats.problems_solved == 0
        assert stats.problems_solved_first_try == 0
        assert stats.patterns_mastered == 0
        assert stats.collaborations_count == 0
        assert stats.playgrounds_created == 0
        assert stats.playgrounds_shared == 0
        assert stats.lessons_completed == 0
        assert stats.paths_completed == 0
        assert stats.beginner_path_completed is False
        assert stats.elementary_path_completed is False
        assert stats.intermediate_path_completed is False
        assert stats.advanced_path_completed is False

    def test_user_stats_level_property(self):
        """Test level property calculation."""
        stats = UserStats(id=uuid4(), user_id=uuid4(), total_xp=0)
        assert stats.level == 1

        stats.total_xp = 500
        assert stats.level == 4

        stats.total_xp = 10000
        assert stats.level == 13

    def test_user_stats_level_title_property(self):
        """Test level title property."""
        stats = UserStats(id=uuid4(), user_id=uuid4(), total_xp=0)
        assert stats.level_title == "입문자"

        stats.total_xp = 500
        assert stats.level == 4
        assert stats.level_title == "탐구자"

    def test_user_stats_xp_progress_property(self):
        """Test XP progress property."""
        stats = UserStats(id=uuid4(), user_id=uuid4(), total_xp=150)
        current_xp, current_threshold, next_threshold = stats.xp_progress
        assert current_xp == 150
        assert current_threshold == 100
        assert next_threshold == 250

    def test_user_stats_xp_percentage_property(self):
        """Test XP percentage property."""
        # At 150 XP, level 2 (100-250 range), 50/150 = 33.33%
        stats = UserStats(id=uuid4(), user_id=uuid4(), total_xp=150)
        percentage = stats.xp_percentage
        assert 33 <= percentage <= 34

        # At exactly 100 XP, should be 0%
        stats.total_xp = 100
        assert stats.xp_percentage == 0.0

        # At 175 XP, 75/150 = 50%
        stats.total_xp = 175
        assert stats.xp_percentage == 50.0

    def test_user_stats_add_xp(self):
        """Test adding XP."""
        stats = UserStats(id=uuid4(), user_id=uuid4(), total_xp=100)
        new_total = stats.add_xp(50)
        assert new_total == 150
        assert stats.total_xp == 150

    def test_user_stats_update_streak_first_activity(self):
        """Test updating streak on first activity."""
        stats = UserStats(id=uuid4(), user_id=uuid4())
        assert stats.last_activity_date is None
        result = stats.update_streak(datetime.now(timezone.utc))
        assert result is True
        assert stats.current_streak == 1
        assert stats.last_activity_date is not None

    def test_user_stats_update_streak_same_day(self):
        """Test updating streak on same day (no change)."""
        now = datetime.now(timezone.utc)
        stats = UserStats(
            id=uuid4(),
            user_id=uuid4(),
            current_streak=5,
            last_activity_date=now,
        )
        result = stats.update_streak(now)
        assert result is False  # No change
        assert stats.current_streak == 5

    def test_user_stats_update_streak_consecutive_day(self):
        """Test updating streak on consecutive day."""
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        stats = UserStats(
            id=uuid4(),
            user_id=uuid4(),
            current_streak=5,
            longest_streak=5,
            last_activity_date=yesterday,
        )
        result = stats.update_streak(datetime.now(timezone.utc))
        assert result is True
        assert stats.current_streak == 6
        assert stats.longest_streak == 6

    def test_user_stats_update_streak_broken(self):
        """Test streak broken after missing days."""
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        stats = UserStats(
            id=uuid4(),
            user_id=uuid4(),
            current_streak=10,
            longest_streak=10,
            last_activity_date=three_days_ago,
        )
        result = stats.update_streak(datetime.now(timezone.utc))
        assert result is False  # Streak broken
        assert stats.current_streak == 1
        assert stats.longest_streak == 10  # Longest preserved

    def test_user_stats_increment_problems_solved(self):
        """Test incrementing problems solved."""
        stats = UserStats(id=uuid4(), user_id=uuid4())
        stats.increment_problems_solved(first_try=False)
        assert stats.problems_solved == 1
        assert stats.problems_solved_first_try == 0

        stats.increment_problems_solved(first_try=True)
        assert stats.problems_solved == 2
        assert stats.problems_solved_first_try == 1

    def test_user_stats_increment_lessons_completed(self):
        """Test incrementing lessons completed."""
        stats = UserStats(id=uuid4(), user_id=uuid4())
        stats.increment_lessons_completed()
        assert stats.lessons_completed == 1
        stats.increment_lessons_completed()
        assert stats.lessons_completed == 2

    def test_user_stats_increment_paths_completed(self):
        """Test incrementing paths completed."""
        stats = UserStats(id=uuid4(), user_id=uuid4())
        stats.increment_paths_completed()
        assert stats.paths_completed == 1

    def test_user_stats_set_path_level_completed(self):
        """Test setting path level completed."""
        stats = UserStats(id=uuid4(), user_id=uuid4())

        stats.set_path_level_completed("beginner")
        assert stats.beginner_path_completed is True
        assert stats.elementary_path_completed is False

        stats.set_path_level_completed("ELEMENTARY")  # Case insensitive
        assert stats.elementary_path_completed is True

        stats.set_path_level_completed("intermediate")
        assert stats.intermediate_path_completed is True

        stats.set_path_level_completed("advanced")
        assert stats.advanced_path_completed is True


class TestChallenge:
    """Tests for Challenge entity."""

    def test_challenge_creation(self):
        """Test creating a challenge."""
        challenge_id = uuid4()
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=7)
        challenge = Challenge(
            id=challenge_id,
            name="Weekly Warrior",
            description="Solve 10 problems this week",
            challenge_type=ChallengeType.WEEKLY,
            target_action="solve_problems",
            target_value=10,
            xp_reward=500,
            start_date=start,
            end_date=end,
        )
        assert challenge.id == challenge_id
        assert challenge.name == "Weekly Warrior"
        assert challenge.challenge_type == ChallengeType.WEEKLY
        assert challenge.target_value == 10
        assert challenge.xp_reward == 500

    def test_challenge_create_factory(self):
        """Test challenge factory method."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=1)
        challenge = Challenge.create(
            name="Daily Challenge",
            description="Solve 3 problems today",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=3,
            xp_reward=100,
            start_date=start,
            end_date=end,
        )
        assert challenge.id is not None
        assert challenge.name == "Daily Challenge"

    def test_challenge_is_active_true(self):
        """Test is_active when challenge is active."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=1,
            xp_reward=10,
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )
        assert challenge.is_active is True

    def test_challenge_is_active_false_ended(self):
        """Test is_active when challenge has ended."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=1,
            xp_reward=10,
            start_date=now - timedelta(days=2),
            end_date=now - timedelta(days=1),
        )
        assert challenge.is_active is False

    def test_challenge_is_active_false_not_started(self):
        """Test is_active when challenge hasn't started."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=1,
            xp_reward=10,
            start_date=now + timedelta(days=1),
            end_date=now + timedelta(days=2),
        )
        assert challenge.is_active is False


class TestUserChallenge:
    """Tests for UserChallenge entity."""

    def test_user_challenge_creation(self):
        """Test creating user challenge."""
        user_challenge_id = uuid4()
        user_id = uuid4()
        challenge_id = uuid4()
        user_challenge = UserChallenge(
            id=user_challenge_id,
            user_id=user_id,
            challenge_id=challenge_id,
        )
        assert user_challenge.id == user_challenge_id
        assert user_challenge.user_id == user_id
        assert user_challenge.challenge_id == challenge_id
        assert user_challenge.current_progress == 0
        assert user_challenge.status == ChallengeStatus.ACTIVE

    def test_user_challenge_create_factory(self):
        """Test user challenge factory method."""
        user_id = uuid4()
        challenge_id = uuid4()
        user_challenge = UserChallenge.create(user_id, challenge_id)
        assert user_challenge.id is not None
        assert user_challenge.user_id == user_id
        assert user_challenge.challenge_id == challenge_id

    def test_user_challenge_update_progress_not_completed(self):
        """Test updating progress without completing."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=10,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=challenge.id,
            challenge=challenge,
        )
        result = user_challenge.update_progress(5)
        assert result is False  # Not completed
        assert user_challenge.current_progress == 5
        assert user_challenge.status == ChallengeStatus.ACTIVE

    def test_user_challenge_update_progress_completed(self):
        """Test updating progress and completing."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=10,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=challenge.id,
            challenge=challenge,
        )
        result = user_challenge.update_progress(10)
        assert result is True  # Completed
        assert user_challenge.status == ChallengeStatus.COMPLETED
        assert user_challenge.completed_at is not None

    def test_user_challenge_update_progress_already_completed(self):
        """Test updating progress when already completed."""
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=uuid4(),
            status=ChallengeStatus.COMPLETED,
        )
        result = user_challenge.update_progress(100)
        assert result is False  # No change

    def test_user_challenge_progress_percentage(self):
        """Test progress percentage calculation."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=10,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=challenge.id,
            current_progress=5,
            challenge=challenge,
        )
        assert user_challenge.progress_percentage == 50.0

    def test_user_challenge_progress_percentage_no_challenge(self):
        """Test progress percentage with no challenge attached."""
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=uuid4(),
            current_progress=5,
        )
        assert user_challenge.progress_percentage == 0.0

    def test_user_challenge_progress_percentage_capped(self):
        """Test progress percentage capped at 100."""
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            id=uuid4(),
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="test",
            target_value=10,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge(
            id=uuid4(),
            user_id=uuid4(),
            challenge_id=challenge.id,
            current_progress=15,  # Over target
            challenge=challenge,
        )
        assert user_challenge.progress_percentage == 100.0


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry entity."""

    def test_leaderboard_entry_creation(self):
        """Test creating leaderboard entry."""
        user_id = uuid4()
        entry = LeaderboardEntry(
            user_id=user_id,
            username="testuser",
            total_xp=5000,
            level=10,
            level_title="그랜드마스터",
            rank=1,
            problems_solved=50,
            current_streak=30,
        )
        assert entry.user_id == user_id
        assert entry.username == "testuser"
        assert entry.total_xp == 5000
        assert entry.level == 10
        assert entry.rank == 1
        assert entry.problems_solved == 50


class TestPredefinedBadges:
    """Tests for predefined badges."""

    def test_predefined_badges_exist(self):
        """Test predefined badges list is not empty."""
        assert len(PREDEFINED_BADGES) > 0

    def test_predefined_badges_have_required_fields(self):
        """Test all predefined badges have required fields."""
        required_fields = [
            "name", "description", "icon", "rarity",
            "category", "requirement", "requirement_value", "xp_reward"
        ]
        for badge_data in PREDEFINED_BADGES:
            for field in required_fields:
                assert field in badge_data, f"Badge missing {field}"

    def test_predefined_badges_unique_names(self):
        """Test all predefined badges have unique names."""
        names = [b["name"] for b in PREDEFINED_BADGES]
        assert len(names) == len(set(names))


# ============= Helper Function Tests =============

class TestUtcNow:
    """Tests for utc_now helper function."""

    def test_utc_now_is_aware(self):
        """Test that utc_now returns timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test that utc_now returns current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after


# ============= Integration Tests =============

class TestGamificationIntegration:
    """Integration tests for gamification module."""

    def test_level_progression_simulation(self):
        """Test simulating level progression."""
        stats = UserStats.create(uuid4())

        # Simulate solving problems
        for i in range(20):
            stats.add_xp(XP_REWARDS["problem_solved"])
            stats.increment_problems_solved(first_try=(i % 3 == 0))

        # Should have earned XP and increased level
        assert stats.total_xp == 20 * 50  # 1000 XP
        assert stats.level > 1
        assert stats.problems_solved == 20
        assert stats.problems_solved_first_try == 7  # 0, 3, 6, 9, 12, 15, 18

    def test_streak_simulation(self):
        """Test simulating streak over multiple days."""
        stats = UserStats.create(uuid4())
        today = datetime.now(timezone.utc)

        # Simulate 7 consecutive days
        for i in range(7):
            day = today - timedelta(days=6-i)
            stats.update_streak(day)

        assert stats.current_streak == 7
        assert stats.longest_streak == 7

        # Miss 2 days
        day_after_break = today + timedelta(days=2)
        stats.update_streak(day_after_break)
        assert stats.current_streak == 1  # Reset
        assert stats.longest_streak == 7  # Preserved

    def test_path_completion_workflow(self):
        """Test path completion workflow."""
        stats = UserStats.create(uuid4())

        # Complete beginner path
        stats.set_path_level_completed("beginner")
        stats.increment_paths_completed()
        assert stats.beginner_path_completed is True
        assert stats.paths_completed == 1

        # Complete elementary path
        stats.set_path_level_completed("elementary")
        stats.increment_paths_completed()
        assert stats.elementary_path_completed is True
        assert stats.paths_completed == 2

        # Complete all paths
        stats.set_path_level_completed("intermediate")
        stats.increment_paths_completed()
        stats.set_path_level_completed("advanced")
        stats.increment_paths_completed()

        assert stats.intermediate_path_completed is True
        assert stats.advanced_path_completed is True
        assert stats.paths_completed == 4

    def test_challenge_workflow(self):
        """Test challenge workflow."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)

        # Create challenge
        challenge = Challenge.create(
            name="Solve 5 problems",
            description="Solve 5 problems to complete",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=5,
            xp_reward=200,
            start_date=now,
            end_date=now + timedelta(days=1),
        )

        # User joins challenge
        user_challenge = UserChallenge.create(user_id, challenge.id)
        user_challenge.challenge = challenge
        assert user_challenge.status == ChallengeStatus.ACTIVE
        assert user_challenge.progress_percentage == 0.0

        # User makes progress
        user_challenge.update_progress(3)
        assert user_challenge.progress_percentage == 60.0
        assert user_challenge.status == ChallengeStatus.ACTIVE

        # User completes challenge
        completed = user_challenge.update_progress(5)
        assert completed is True
        assert user_challenge.status == ChallengeStatus.COMPLETED
        assert user_challenge.progress_percentage == 100.0
