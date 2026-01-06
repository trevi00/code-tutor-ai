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


# ============= Service Tests =============

class TestBadgeService:
    """Tests for BadgeService."""

    @pytest.fixture
    def mock_badge_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user_badge_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user_stats_repo(self):
        return AsyncMock()

    @pytest.fixture
    def badge_service(self, mock_badge_repo, mock_user_badge_repo, mock_user_stats_repo):
        from code_tutor.gamification.application.services import BadgeService
        return BadgeService(mock_badge_repo, mock_user_badge_repo, mock_user_stats_repo)

    @pytest.mark.asyncio
    async def test_seed_badges_creates_new_badges(self, badge_service, mock_badge_repo):
        """Test seeding badges when none exist."""
        mock_badge_repo.exists_by_name.return_value = False
        mock_badge_repo.create.return_value = None

        count = await badge_service.seed_badges()

        assert count == len(PREDEFINED_BADGES)
        assert mock_badge_repo.exists_by_name.call_count == len(PREDEFINED_BADGES)
        assert mock_badge_repo.create.call_count == len(PREDEFINED_BADGES)

    @pytest.mark.asyncio
    async def test_seed_badges_skips_existing(self, badge_service, mock_badge_repo):
        """Test seeding badges skips existing ones."""
        mock_badge_repo.exists_by_name.return_value = True

        count = await badge_service.seed_badges()

        assert count == 0
        assert mock_badge_repo.create.call_count == 0

    @pytest.mark.asyncio
    async def test_get_all_badges(self, badge_service, mock_badge_repo):
        """Test getting all badges."""
        badges = [
            Badge.create(
                name="Test Badge",
                description="Test",
                icon="⭐",
                rarity=BadgeRarity.COMMON,
                category=BadgeCategory.PROBLEM_SOLVING,
                requirement="problems_solved",
                requirement_value=1,
                xp_reward=50,
            )
        ]
        mock_badge_repo.get_all.return_value = badges

        result = await badge_service.get_all_badges()

        assert len(result) == 1
        assert result[0].name == "Test Badge"

    @pytest.mark.asyncio
    async def test_get_user_badges(self, badge_service, mock_badge_repo, mock_user_badge_repo):
        """Test getting user's badges."""
        user_id = uuid4()
        badge = Badge.create(
            name="First Badge",
            description="Test",
            icon="🎯",
            rarity=BadgeRarity.COMMON,
            category=BadgeCategory.PROBLEM_SOLVING,
            requirement="problems_solved",
            requirement_value=1,
            xp_reward=50,
        )
        user_badge = UserBadge.create(user_id, badge.id)
        user_badge.badge = badge

        mock_badge_repo.get_all.return_value = [badge]
        mock_user_badge_repo.get_by_user.return_value = [user_badge]

        result = await badge_service.get_user_badges(user_id)

        assert result.total_earned == 1
        assert result.total_available == 0

    @pytest.mark.asyncio
    async def test_check_and_award_badges_awards_qualified(
        self, badge_service, mock_badge_repo, mock_user_badge_repo, mock_user_stats_repo
    ):
        """Test awarding badges when user qualifies."""
        user_id = uuid4()
        badge = Badge.create(
            name="Solver",
            description="Solve 1 problem",
            icon="✅",
            rarity=BadgeRarity.COMMON,
            category=BadgeCategory.PROBLEM_SOLVING,
            requirement="problems_solved",
            requirement_value=1,
            xp_reward=50,
        )
        stats = UserStats.create(user_id)
        stats.problems_solved = 5  # Exceeds requirement

        mock_user_stats_repo.get_or_create.return_value = stats
        mock_badge_repo.get_all.return_value = [badge]
        mock_user_badge_repo.has_badge.return_value = False
        mock_user_badge_repo.award_badge.return_value = None
        mock_user_stats_repo.update.return_value = None

        result = await badge_service.check_and_award_badges(user_id)

        assert len(result) == 1
        assert result[0].name == "Solver"
        mock_user_badge_repo.award_badge.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_award_badges_skips_already_earned(
        self, badge_service, mock_badge_repo, mock_user_badge_repo, mock_user_stats_repo
    ):
        """Test skipping badges user already has."""
        user_id = uuid4()
        badge = Badge.create(
            name="Solver",
            description="Solve 1 problem",
            icon="✅",
            rarity=BadgeRarity.COMMON,
            category=BadgeCategory.PROBLEM_SOLVING,
            requirement="problems_solved",
            requirement_value=1,
            xp_reward=50,
        )
        stats = UserStats.create(user_id)
        stats.problems_solved = 5

        mock_user_stats_repo.get_or_create.return_value = stats
        mock_badge_repo.get_all.return_value = [badge]
        mock_user_badge_repo.has_badge.return_value = True  # Already has

        result = await badge_service.check_and_award_badges(user_id)

        assert len(result) == 0
        mock_user_badge_repo.award_badge.assert_not_called()

    def test_check_badge_requirement_problems_solved(self, badge_service):
        """Test checking problems_solved requirement."""
        stats = UserStats.create(uuid4())
        stats.problems_solved = 10
        badge = Badge.create(
            name="Test",
            description="Test",
            icon="⭐",
            rarity=BadgeRarity.COMMON,
            category=BadgeCategory.PROBLEM_SOLVING,
            requirement="problems_solved",
            requirement_value=5,
            xp_reward=50,
        )

        assert badge_service._check_badge_requirement(stats, badge) is True

        badge.requirement_value = 20
        assert badge_service._check_badge_requirement(stats, badge) is False

    def test_check_badge_requirement_path_completed(self, badge_service):
        """Test checking path completion requirement."""
        stats = UserStats.create(uuid4())
        stats.beginner_path_completed = True
        badge = Badge.create(
            name="Test",
            description="Test",
            icon="⭐",
            rarity=BadgeRarity.RARE,
            category=BadgeCategory.MASTERY,
            requirement="beginner_path_completed",
            requirement_value=1,
            xp_reward=100,
        )

        assert badge_service._check_badge_requirement(stats, badge) is True

        stats.beginner_path_completed = False
        assert badge_service._check_badge_requirement(stats, badge) is False


class TestXPService:
    """Tests for XPService."""

    @pytest.fixture
    def mock_user_stats_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_badge_service(self):
        return AsyncMock()

    @pytest.fixture
    def xp_service(self, mock_user_stats_repo, mock_badge_service):
        from code_tutor.gamification.application.services import XPService
        return XPService(mock_user_stats_repo, mock_badge_service)

    @pytest.mark.asyncio
    async def test_get_user_stats(self, xp_service, mock_user_stats_repo):
        """Test getting user stats."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        stats.total_xp = 500
        stats.problems_solved = 10
        mock_user_stats_repo.get_or_create.return_value = stats

        result = await xp_service.get_user_stats(user_id)

        assert result.total_xp == 500
        assert result.problems_solved == 10
        assert result.level > 1

    @pytest.mark.asyncio
    async def test_add_xp_problem_solved(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test adding XP for problem solved."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        result = await xp_service.add_xp(user_id, "problem_solved")

        assert result.xp_added == XP_REWARDS["problem_solved"]
        assert result.total_xp == XP_REWARDS["problem_solved"]
        assert stats.problems_solved == 1
        mock_user_stats_repo.update.assert_called()

    @pytest.mark.asyncio
    async def test_add_xp_problem_solved_first_try(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test adding XP for problem solved first try."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        result = await xp_service.add_xp(user_id, "problem_solved_first_try")

        assert result.xp_added == XP_REWARDS["problem_solved_first_try"]
        assert stats.problems_solved == 1
        assert stats.problems_solved_first_try == 1

    @pytest.mark.asyncio
    async def test_add_xp_custom_amount(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test adding custom XP amount."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        result = await xp_service.add_xp(user_id, "custom_action", custom_amount=200)

        assert result.xp_added == 200
        assert result.total_xp == 200

    @pytest.mark.asyncio
    async def test_add_xp_level_up(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test level up when adding XP."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        stats.total_xp = 95  # Close to level 2 (100)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        result = await xp_service.add_xp(user_id, "problem_solved")  # +50 XP

        assert result.leveled_up is True
        assert result.level == 2

    @pytest.mark.asyncio
    async def test_add_xp_various_actions(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test XP for various actions updates correct counters."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        # collaboration_session
        await xp_service.add_xp(user_id, "collaboration_session")
        assert stats.collaborations_count == 1

        # playground_created
        await xp_service.add_xp(user_id, "playground_created")
        assert stats.playgrounds_created == 1

        # playground_shared
        await xp_service.add_xp(user_id, "playground_shared")
        assert stats.playgrounds_shared == 1

        # lesson_completed
        await xp_service.add_xp(user_id, "lesson_completed")
        assert stats.lessons_completed == 1

        # path_completed
        await xp_service.add_xp(user_id, "path_completed")
        assert stats.paths_completed == 1

    @pytest.mark.asyncio
    async def test_record_activity(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test recording activity."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        result = await xp_service.record_activity(user_id, "daily_login")

        assert result.xp_added == XP_REWARDS["daily_login"]

    @pytest.mark.asyncio
    async def test_set_path_level_completed(self, xp_service, mock_user_stats_repo, mock_badge_service):
        """Test setting path level completed."""
        user_id = uuid4()
        stats = UserStats.create(user_id)
        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_stats_repo.update.return_value = None
        mock_badge_service.check_and_award_badges.return_value = []

        await xp_service.set_path_level_completed(user_id, "beginner")

        assert stats.beginner_path_completed is True
        mock_user_stats_repo.update.assert_called()
        mock_badge_service.check_and_award_badges.assert_called_with(user_id)


class TestLeaderboardService:
    """Tests for LeaderboardService."""

    @pytest.fixture
    def mock_user_stats_repo(self):
        return AsyncMock()

    @pytest.fixture
    def leaderboard_service(self, mock_user_stats_repo):
        from code_tutor.gamification.application.services import LeaderboardService
        return LeaderboardService(mock_user_stats_repo)

    @pytest.mark.asyncio
    async def test_get_leaderboard(self, leaderboard_service, mock_user_stats_repo):
        """Test getting leaderboard."""
        entries = [
            LeaderboardEntry(
                user_id=uuid4(),
                username="user1",
                total_xp=1000,
                level=5,
                level_title="도전자",
                rank=1,
                problems_solved=20,
                current_streak=10,
            ),
            LeaderboardEntry(
                user_id=uuid4(),
                username="user2",
                total_xp=500,
                level=3,
                level_title="탐험가",
                rank=2,
                problems_solved=10,
                current_streak=5,
            ),
        ]
        mock_user_stats_repo.get_leaderboard.return_value = entries
        mock_user_stats_repo.get_user_rank.return_value = 1

        result = await leaderboard_service.get_leaderboard(
            period="all", limit=100, offset=0, user_id=entries[0].user_id
        )

        assert result.period == "all"
        assert result.total_users == 2
        assert len(result.entries) == 2
        assert result.entries[0].rank == 1
        assert result.user_rank == 1

    @pytest.mark.asyncio
    async def test_get_leaderboard_weekly(self, leaderboard_service, mock_user_stats_repo):
        """Test getting weekly leaderboard."""
        mock_user_stats_repo.get_leaderboard.return_value = []

        result = await leaderboard_service.get_leaderboard(period="weekly")

        assert result.period == "weekly"
        mock_user_stats_repo.get_leaderboard.assert_called_with(
            limit=100, offset=0, period="weekly"
        )


class TestChallengeService:
    """Tests for ChallengeService."""

    @pytest.fixture
    def mock_challenge_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user_challenge_repo(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user_stats_repo(self):
        return AsyncMock()

    @pytest.fixture
    def challenge_service(self, mock_challenge_repo, mock_user_challenge_repo, mock_user_stats_repo):
        from code_tutor.gamification.application.services import ChallengeService
        return ChallengeService(mock_challenge_repo, mock_user_challenge_repo, mock_user_stats_repo)

    @pytest.mark.asyncio
    async def test_create_challenge(self, challenge_service, mock_challenge_repo):
        """Test creating a challenge."""
        mock_challenge_repo.create.return_value = None

        result = await challenge_service.create_challenge(
            name="Weekly Warrior",
            description="Solve 10 problems",
            challenge_type=ChallengeType.WEEKLY,
            target_action="solve_problems",
            target_value=10,
            xp_reward=500,
            duration_days=7,
        )

        assert result.name == "Weekly Warrior"
        assert result.challenge_type == ChallengeType.WEEKLY
        assert result.target_value == 10
        assert result.xp_reward == 500
        mock_challenge_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_challenges(self, challenge_service, mock_challenge_repo, mock_user_challenge_repo):
        """Test getting user's challenges."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        challenge = Challenge.create(
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=5,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge.create(user_id, challenge.id)
        user_challenge.challenge = challenge

        mock_challenge_repo.get_active.return_value = [challenge]
        mock_user_challenge_repo.get_by_user.return_value = [user_challenge]

        result = await challenge_service.get_user_challenges(user_id)

        assert len(result.active) == 1
        assert len(result.completed) == 0
        assert len(result.available) == 0  # Already joined

    @pytest.mark.asyncio
    async def test_join_challenge(self, challenge_service, mock_challenge_repo, mock_user_challenge_repo):
        """Test joining a challenge."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        challenge = Challenge.create(
            name="Test Challenge",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=5,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )

        mock_user_challenge_repo.get_by_user_and_challenge.return_value = None
        mock_challenge_repo.get_by_id.return_value = challenge
        mock_user_challenge_repo.create.return_value = None

        result = await challenge_service.join_challenge(user_id, challenge.id)

        assert result.challenge.name == "Test Challenge"
        assert result.current_progress == 0
        mock_user_challenge_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_join_challenge_already_joined(self, challenge_service, mock_user_challenge_repo):
        """Test joining a challenge user already joined."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        challenge = Challenge.create(
            name="Test",
            description="Test",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=5,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        existing = UserChallenge.create(user_id, challenge.id)
        existing.challenge = challenge
        existing.current_progress = 3

        mock_user_challenge_repo.get_by_user_and_challenge.return_value = existing

        result = await challenge_service.join_challenge(user_id, challenge.id)

        assert result.current_progress == 3  # Returns existing
        mock_user_challenge_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_join_challenge_not_found(self, challenge_service, mock_challenge_repo, mock_user_challenge_repo):
        """Test joining non-existent challenge."""
        mock_user_challenge_repo.get_by_user_and_challenge.return_value = None
        mock_challenge_repo.get_by_id.return_value = None

        with pytest.raises(ValueError, match="Challenge not found"):
            await challenge_service.join_challenge(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_update_challenge_progress(
        self, challenge_service, mock_user_challenge_repo, mock_user_stats_repo
    ):
        """Test updating challenge progress."""
        user_id = uuid4()
        now = datetime.now(timezone.utc)
        challenge = Challenge.create(
            name="Solver",
            description="Solve 5 problems",
            challenge_type=ChallengeType.DAILY,
            target_action="solve_problems",
            target_value=5,
            xp_reward=100,
            start_date=now,
            end_date=now + timedelta(days=1),
        )
        user_challenge = UserChallenge.create(user_id, challenge.id)
        user_challenge.challenge = challenge

        stats = UserStats.create(user_id)
        stats.problems_solved = 5

        mock_user_stats_repo.get_or_create.return_value = stats
        mock_user_challenge_repo.get_by_user.return_value = [user_challenge]
        mock_user_challenge_repo.update.return_value = None
        mock_user_stats_repo.update.return_value = None

        result = await challenge_service.update_challenge_progress(user_id, "solve_problems")

        assert len(result) == 1
        assert result[0].status == ChallengeStatus.COMPLETED

    def test_get_progress_for_action(self, challenge_service):
        """Test getting progress for various actions."""
        stats = UserStats.create(uuid4())
        stats.problems_solved = 10
        stats.current_streak = 5
        stats.patterns_mastered = 3
        stats.collaborations_count = 2

        assert challenge_service._get_progress_for_action(stats, "solve_problems") == 10
        assert challenge_service._get_progress_for_action(stats, "maintain_streak") == 5
        assert challenge_service._get_progress_for_action(stats, "complete_patterns") == 3
        assert challenge_service._get_progress_for_action(stats, "collaborate") == 2
        assert challenge_service._get_progress_for_action(stats, "unknown") == 0


class TestGamificationService:
    """Tests for GamificationService."""

    @pytest.fixture
    def mock_badge_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_xp_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_leaderboard_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_challenge_service(self):
        return AsyncMock()

    @pytest.fixture
    def gamification_service(
        self, mock_badge_service, mock_xp_service, mock_leaderboard_service, mock_challenge_service
    ):
        from code_tutor.gamification.application.services import GamificationService
        return GamificationService(
            mock_badge_service, mock_xp_service, mock_leaderboard_service, mock_challenge_service
        )

    @pytest.mark.asyncio
    async def test_get_overview(
        self,
        gamification_service,
        mock_badge_service,
        mock_xp_service,
        mock_leaderboard_service,
        mock_challenge_service,
    ):
        """Test getting gamification overview."""
        user_id = uuid4()
        from code_tutor.gamification.application.dto import (
            UserStatsResponse,
            UserBadgesResponse,
            ChallengesResponse,
            LeaderboardResponse,
            BadgeResponse,
        )

        stats = UserStatsResponse(
            total_xp=500,
            level=4,
            level_title="탐구자",
            xp_progress=150,
            xp_for_next_level=350,
            xp_percentage=42.8,
            current_streak=5,
            longest_streak=10,
            problems_solved=20,
            problems_solved_first_try=10,
            patterns_mastered=3,
            collaborations_count=2,
            playgrounds_created=1,
            playgrounds_shared=0,
            lessons_completed=5,
            paths_completed=0,
        )

        available_badge = BadgeResponse(
            id=uuid4(),
            name="Streak Master",
            description="Maintain 30 day streak",
            icon="🔥",
            rarity=BadgeRarity.EPIC,
            category=BadgeCategory.STREAK,
            requirement="current_streak",
            requirement_value=30,
            xp_reward=500,
        )

        badges = UserBadgesResponse(
            earned=[],
            available=[available_badge],
            total_earned=0,
            total_available=1,
        )

        challenges = ChallengesResponse(active=[], completed=[], available=[])

        leaderboard = LeaderboardResponse(
            entries=[],
            period="all",
            total_users=0,
            user_rank=5,
        )

        mock_xp_service.get_user_stats.return_value = stats
        mock_badge_service.get_user_badges.return_value = badges
        mock_challenge_service.get_user_challenges.return_value = challenges
        mock_leaderboard_service.get_leaderboard.return_value = leaderboard

        result = await gamification_service.get_overview(user_id)

        assert result.stats.total_xp == 500
        assert result.leaderboard_rank == 5
        assert result.next_badge_progress is not None
        assert result.next_badge_progress["badge"] == "Streak Master"
        assert result.next_badge_progress["current"] == 5  # current_streak
        assert result.next_badge_progress["required"] == 30


# ============= API Route Tests =============

@pytest.fixture
def mock_current_user():
    """Create mock current user."""
    from code_tutor.identity.application.dto import UserResponse
    return UserResponse(
        id=uuid4(),
        email="test@example.com",
        username="testuser",
        role="student",
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_admin_user():
    """Create mock admin user."""
    from code_tutor.identity.application.dto import UserResponse
    return UserResponse(
        id=uuid4(),
        email="admin@example.com",
        username="admin",
        role="admin",
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
    )


class TestGamificationRoutesUnit:
    """Unit tests for gamification API routes with mocked services."""

    def test_router_prefix(self):
        """Test router has correct prefix."""
        from code_tutor.gamification.interface.routes import router
        assert router.prefix == "/gamification"

    def test_router_tags(self):
        """Test router has correct tags."""
        from code_tutor.gamification.interface.routes import router
        assert "Gamification" in router.tags

    def test_router_has_expected_routes(self):
        """Test router has all expected routes."""
        from code_tutor.gamification.interface.routes import router

        route_paths = [r.path for r in router.routes]

        expected_paths = [
            "/gamification/overview",
            "/gamification/badges",
            "/gamification/badges/me",
            "/gamification/badges/check",
            "/gamification/stats",
            "/gamification/xp",
            "/gamification/activity/{action}",
            "/gamification/leaderboard",
            "/gamification/challenges",
            "/gamification/challenges/{challenge_id}/join",
            "/gamification/admin/seed-badges",
            "/gamification/admin/challenges",
        ]

        for path in expected_paths:
            assert path in route_paths, f"Missing route: {path}"

    def test_service_factory_functions_exist(self):
        """Test service factory functions exist."""
        from code_tutor.gamification.interface.routes import (
            get_badge_service,
            get_xp_service,
            get_leaderboard_service,
            get_challenge_service,
            get_gamification_service,
        )

        assert callable(get_badge_service)
        assert callable(get_xp_service)
        assert callable(get_leaderboard_service)
        assert callable(get_challenge_service)
        assert callable(get_gamification_service)
