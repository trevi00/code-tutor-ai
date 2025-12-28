"""Unit tests for Dashboard Service and DTOs"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from code_tutor.learning.application.dashboard_dto import (
    DashboardResponse,
    UserStats,
    CategoryProgress,
    RecentSubmission,
    StreakInfo,
)
from code_tutor.learning.domain.value_objects import Category, Difficulty


class TestUserStats:
    """Tests for UserStats DTO"""

    def test_user_stats_creation(self):
        """Test creating user stats"""
        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=5,
            total_submissions=20,
            overall_success_rate=50.0,
            easy_solved=3,
            medium_solved=2,
            hard_solved=0,
            streak=StreakInfo(),
        )
        assert stats.total_problems_attempted == 10
        assert stats.total_problems_solved == 5
        assert stats.overall_success_rate == 50.0

    def test_user_stats_empty(self):
        """Test empty user stats"""
        stats = UserStats(
            total_problems_attempted=0,
            total_problems_solved=0,
            total_submissions=0,
            overall_success_rate=0,
            easy_solved=0,
            medium_solved=0,
            hard_solved=0,
            streak=StreakInfo(),
        )
        assert stats.total_submissions == 0
        assert stats.streak.current_streak == 0


class TestStreakInfo:
    """Tests for StreakInfo DTO"""

    def test_streak_info_defaults(self):
        """Test default streak info"""
        streak = StreakInfo()
        assert streak.current_streak == 0
        assert streak.longest_streak == 0
        assert streak.last_activity_date is None

    def test_streak_info_with_data(self):
        """Test streak info with data"""
        now = datetime.now(timezone.utc)
        streak = StreakInfo(
            current_streak=5,
            longest_streak=10,
            last_activity_date=now,
        )
        assert streak.current_streak == 5
        assert streak.longest_streak == 10
        assert streak.last_activity_date == now


class TestCategoryProgress:
    """Tests for CategoryProgress DTO"""

    def test_category_progress_creation(self):
        """Test creating category progress"""
        progress = CategoryProgress(
            category="array",
            total_problems=10,
            solved_problems=5,
            success_rate=50.0,
        )
        assert progress.category == "array"
        assert progress.total_problems == 10
        assert progress.solved_problems == 5
        assert progress.success_rate == 50.0

    def test_category_progress_zero(self):
        """Test category progress with zero problems"""
        progress = CategoryProgress(
            category="tree",
            total_problems=0,
            solved_problems=0,
            success_rate=0.0,
        )
        assert progress.total_problems == 0


class TestRecentSubmission:
    """Tests for RecentSubmission DTO"""

    def test_recent_submission_creation(self):
        """Test creating recent submission"""
        now = datetime.now(timezone.utc)
        submission = RecentSubmission(
            id=uuid4(),
            problem_id=uuid4(),
            problem_title="Two Sum",
            status="accepted",
            submitted_at=now,
        )
        assert submission.problem_title == "Two Sum"
        assert submission.status == "accepted"


class TestDashboardResponse:
    """Tests for DashboardResponse DTO"""

    def test_dashboard_response_creation(self):
        """Test creating dashboard response"""
        response = DashboardResponse(
            stats=UserStats(
                total_problems_attempted=5,
                total_problems_solved=3,
                total_submissions=10,
                overall_success_rate=60.0,
                easy_solved=2,
                medium_solved=1,
                hard_solved=0,
                streak=StreakInfo(current_streak=2, longest_streak=5),
            ),
            category_progress=[
                CategoryProgress(
                    category="array",
                    total_problems=10,
                    solved_problems=3,
                    success_rate=30.0,
                )
            ],
            recent_submissions=[],
        )
        assert response.stats.total_problems_solved == 3
        assert len(response.category_progress) == 1
        assert len(response.recent_submissions) == 0

    def test_dashboard_response_empty(self):
        """Test empty dashboard response"""
        response = DashboardResponse(
            stats=UserStats(
                total_problems_attempted=0,
                total_problems_solved=0,
                total_submissions=0,
                overall_success_rate=0,
                easy_solved=0,
                medium_solved=0,
                hard_solved=0,
                streak=StreakInfo(),
            ),
            category_progress=[],
            recent_submissions=[],
        )
        assert response.stats.total_submissions == 0
        assert response.category_progress == []


class TestDifficultyEnum:
    """Tests for Difficulty enum"""

    def test_difficulty_values(self):
        """Test difficulty enum values"""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"


class TestCategoryEnum:
    """Tests for Category enum"""

    def test_category_values(self):
        """Test category enum values"""
        assert Category.ARRAY.value == "array"
        assert Category.STRING.value == "string"
        assert Category.LINKED_LIST.value == "linked_list"
        assert Category.DYNAMIC_PROGRAMMING.value == "dp"
        assert Category.TREE.value == "tree"
        assert Category.GRAPH.value == "graph"

    def test_all_categories_exist(self):
        """Test that all expected categories exist"""
        # Match actual Category enum values
        expected_categories = [
            "array", "string", "linked_list", "stack", "queue",
            "hash_table", "tree", "graph", "sorting", "searching",
            "dp", "greedy", "backtracking", "recursion",
            "two_pointers", "sliding_window", "binary_search",
            "bfs", "dfs", "divide_and_conquer", "design",
        ]
        actual_values = [c.value for c in Category]
        for expected in expected_categories:
            assert expected in actual_values, f"{expected} not found in Category enum"
