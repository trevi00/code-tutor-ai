"""Gamification domain entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)

from .value_objects import (
    BadgeRarity,
    BadgeCategory,
    ChallengeType,
    ChallengeStatus,
    calculate_level,
    get_level_title,
    xp_for_next_level,
)


@dataclass
class Badge:
    """Badge definition."""

    id: UUID
    name: str
    description: str
    icon: str  # Icon identifier or emoji
    rarity: BadgeRarity
    category: BadgeCategory
    requirement: str  # Human-readable requirement
    requirement_value: int  # Numeric threshold
    xp_reward: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        icon: str,
        rarity: BadgeRarity,
        category: BadgeCategory,
        requirement: str,
        requirement_value: int,
        xp_reward: int = 0,
    ) -> "Badge":
        """Create a new badge."""
        return cls(
            id=uuid4(),
            name=name,
            description=description,
            icon=icon,
            rarity=rarity,
            category=category,
            requirement=requirement,
            requirement_value=requirement_value,
            xp_reward=xp_reward,
        )


@dataclass
class UserBadge:
    """User's earned badge."""

    id: UUID
    user_id: UUID
    badge_id: UUID
    earned_at: datetime = field(default_factory=datetime.utcnow)
    badge: Optional[Badge] = None

    @classmethod
    def create(cls, user_id: UUID, badge_id: UUID) -> "UserBadge":
        """Create a new user badge."""
        return cls(
            id=uuid4(),
            user_id=user_id,
            badge_id=badge_id,
        )


@dataclass
class UserStats:
    """User's gamification stats."""

    id: UUID
    user_id: UUID
    total_xp: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    problems_solved: int = 0
    problems_solved_first_try: int = 0
    patterns_mastered: int = 0
    collaborations_count: int = 0
    playgrounds_created: int = 0
    playgrounds_shared: int = 0
    # Roadmap progress
    lessons_completed: int = 0
    paths_completed: int = 0
    # Path level completion flags
    beginner_path_completed: bool = False
    elementary_path_completed: bool = False
    intermediate_path_completed: bool = False
    advanced_path_completed: bool = False
    last_activity_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, user_id: UUID) -> "UserStats":
        """Create new user stats."""
        return cls(
            id=uuid4(),
            user_id=user_id,
        )

    @property
    def level(self) -> int:
        """Calculate user's level from XP."""
        return calculate_level(self.total_xp)

    @property
    def level_title(self) -> str:
        """Get user's level title."""
        return get_level_title(self.level)

    @property
    def xp_progress(self) -> tuple[int, int, int]:
        """Get XP progress (current, threshold, next_threshold)."""
        current_threshold, next_threshold = xp_for_next_level(self.total_xp)
        return self.total_xp, current_threshold, next_threshold

    @property
    def xp_percentage(self) -> float:
        """Get XP progress percentage."""
        _, current_threshold, next_threshold = self.xp_progress
        if next_threshold == current_threshold:
            return 100.0
        return ((self.total_xp - current_threshold) / (next_threshold - current_threshold)) * 100

    def add_xp(self, amount: int) -> int:
        """Add XP and return new total."""
        self.total_xp += amount
        self.updated_at = utc_now()
        return self.total_xp

    def update_streak(self, today: datetime) -> bool:
        """Update streak based on activity date.

        Returns True if streak was continued/increased.
        """
        if self.last_activity_date is None:
            self.current_streak = 1
            self.last_activity_date = today
            self.updated_at = utc_now()
            return True

        days_diff = (today.date() - self.last_activity_date.date()).days

        if days_diff == 0:
            # Same day, no change
            return False
        elif days_diff == 1:
            # Consecutive day
            self.current_streak += 1
            self.longest_streak = max(self.longest_streak, self.current_streak)
            self.last_activity_date = today
            self.updated_at = utc_now()
            return True
        else:
            # Streak broken
            self.current_streak = 1
            self.last_activity_date = today
            self.updated_at = utc_now()
            return False

    def increment_problems_solved(self, first_try: bool = False) -> None:
        """Increment problems solved counter."""
        self.problems_solved += 1
        if first_try:
            self.problems_solved_first_try += 1
        self.updated_at = utc_now()

    def increment_lessons_completed(self) -> None:
        """Increment lessons completed counter."""
        self.lessons_completed += 1
        self.updated_at = utc_now()

    def increment_paths_completed(self) -> None:
        """Increment paths completed counter."""
        self.paths_completed += 1
        self.updated_at = utc_now()

    def set_path_level_completed(self, level: str) -> None:
        """Set a specific path level as completed."""
        level_lower = level.lower()
        if level_lower == "beginner":
            self.beginner_path_completed = True
        elif level_lower == "elementary":
            self.elementary_path_completed = True
        elif level_lower == "intermediate":
            self.intermediate_path_completed = True
        elif level_lower == "advanced":
            self.advanced_path_completed = True
        self.updated_at = utc_now()


@dataclass
class Challenge:
    """Challenge/Quest definition."""

    id: UUID
    name: str
    description: str
    challenge_type: ChallengeType
    target_action: str  # e.g., "solve_problems", "maintain_streak"
    target_value: int
    xp_reward: int
    start_date: datetime
    end_date: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        challenge_type: ChallengeType,
        target_action: str,
        target_value: int,
        xp_reward: int,
        start_date: datetime,
        end_date: datetime,
    ) -> "Challenge":
        """Create a new challenge."""
        return cls(
            id=uuid4(),
            name=name,
            description=description,
            challenge_type=challenge_type,
            target_action=target_action,
            target_value=target_value,
            xp_reward=xp_reward,
            start_date=start_date,
            end_date=end_date,
        )

    @property
    def is_active(self) -> bool:
        """Check if challenge is currently active."""
        now = utc_now()
        return self.start_date <= now <= self.end_date


@dataclass
class UserChallenge:
    """User's challenge progress."""

    id: UUID
    user_id: UUID
    challenge_id: UUID
    current_progress: int = 0
    status: ChallengeStatus = ChallengeStatus.ACTIVE
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    challenge: Optional[Challenge] = None

    @classmethod
    def create(cls, user_id: UUID, challenge_id: UUID) -> "UserChallenge":
        """Create new user challenge progress."""
        return cls(
            id=uuid4(),
            user_id=user_id,
            challenge_id=challenge_id,
        )

    def update_progress(self, value: int) -> bool:
        """Update progress and check completion.

        Returns True if challenge was just completed.
        """
        if self.status != ChallengeStatus.ACTIVE:
            return False

        self.current_progress = value

        if self.challenge and self.current_progress >= self.challenge.target_value:
            self.status = ChallengeStatus.COMPLETED
            self.completed_at = utc_now()
            return True

        return False

    @property
    def progress_percentage(self) -> float:
        """Get progress percentage."""
        if not self.challenge:
            return 0.0
        return min(100.0, (self.current_progress / self.challenge.target_value) * 100)


@dataclass
class LeaderboardEntry:
    """Leaderboard entry."""

    user_id: UUID
    username: str
    total_xp: int
    level: int
    level_title: str
    rank: int
    problems_solved: int = 0
    current_streak: int = 0


# Predefined badges
PREDEFINED_BADGES = [
    # Problem Solving Badges
    {
        "name": "첫 발자국",
        "description": "첫 번째 문제를 해결했습니다",
        "icon": "🎯",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "문제 해결사",
        "description": "10개의 문제를 해결했습니다",
        "icon": "⭐",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 10,
        "xp_reward": 100,
    },
    {
        "name": "알고리즘 마스터",
        "description": "50개의 문제를 해결했습니다",
        "icon": "🏆",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 50,
        "xp_reward": 250,
    },
    {
        "name": "코딩 천재",
        "description": "100개의 문제를 해결했습니다",
        "icon": "👑",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 100,
        "xp_reward": 500,
    },
    {
        "name": "완벽주의자",
        "description": "10개의 문제를 첫 시도에 해결했습니다",
        "icon": "💎",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved_first_try",
        "requirement_value": 10,
        "xp_reward": 200,
    },
    # Streak Badges
    {
        "name": "시작이 반",
        "description": "3일 연속 학습했습니다",
        "icon": "🔥",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 3,
        "xp_reward": 30,
    },
    {
        "name": "습관 형성",
        "description": "7일 연속 학습했습니다",
        "icon": "🔥",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 7,
        "xp_reward": 70,
    },
    {
        "name": "꾸준함의 힘",
        "description": "30일 연속 학습했습니다",
        "icon": "🌟",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 30,
        "xp_reward": 300,
    },
    {
        "name": "철인",
        "description": "100일 연속 학습했습니다",
        "icon": "🏅",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.STREAK,
        "requirement": "longest_streak",
        "requirement_value": 100,
        "xp_reward": 1000,
    },
    # Mastery Badges
    {
        "name": "패턴 학습자",
        "description": "1개의 패턴을 마스터했습니다",
        "icon": "📚",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "패턴 전문가",
        "description": "5개의 패턴을 마스터했습니다",
        "icon": "🎓",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 5,
        "xp_reward": 250,
    },
    # Social Badges
    {
        "name": "협력자",
        "description": "첫 협업 세션을 완료했습니다",
        "icon": "🤝",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "collaborations_count",
        "requirement_value": 1,
        "xp_reward": 30,
    },
    {
        "name": "팀 플레이어",
        "description": "10번의 협업 세션을 완료했습니다",
        "icon": "👥",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "collaborations_count",
        "requirement_value": 10,
        "xp_reward": 100,
    },
    {
        "name": "창작자",
        "description": "플레이그라운드를 5개 생성했습니다",
        "icon": "✨",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_created",
        "requirement_value": 5,
        "xp_reward": 75,
    },
    {
        "name": "공유자",
        "description": "플레이그라운드를 5번 공유했습니다",
        "icon": "🔗",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_shared",
        "requirement_value": 5,
        "xp_reward": 100,
    },
    # Roadmap Learning Badges
    {
        "name": "학습 시작",
        "description": "첫 번째 레슨을 완료했습니다",
        "icon": "📖",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 1,
        "xp_reward": 20,
    },
    {
        "name": "꾸준한 학습자",
        "description": "10개의 레슨을 완료했습니다",
        "icon": "📚",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 10,
        "xp_reward": 100,
    },
    {
        "name": "열정적 학습자",
        "description": "50개의 레슨을 완료했습니다",
        "icon": "🎯",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 50,
        "xp_reward": 300,
    },
    {
        "name": "학습 마니아",
        "description": "100개의 레슨을 완료했습니다",
        "icon": "🏅",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 100,
        "xp_reward": 500,
    },
    {
        "name": "첫 경로 완주",
        "description": "첫 번째 학습 경로를 완료했습니다",
        "icon": "🛤️",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "paths_completed",
        "requirement_value": 1,
        "xp_reward": 200,
    },
    {
        "name": "로드맵 정복자",
        "description": "모든 학습 경로를 완료했습니다 (4개)",
        "icon": "👑",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.MASTERY,
        "requirement": "paths_completed",
        "requirement_value": 4,
        "xp_reward": 1000,
    },
    # Path Level Completion Badges
    {
        "name": "파이썬 입문자",
        "description": "파이썬 입문 경로를 완료했습니다",
        "icon": "🐍",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "beginner_path_completed",
        "requirement_value": 1,
        "xp_reward": 150,
    },
    {
        "name": "기초 마스터",
        "description": "기초 알고리즘 경로를 완료했습니다",
        "icon": "📚",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "elementary_path_completed",
        "requirement_value": 1,
        "xp_reward": 250,
    },
    {
        "name": "알고리즘 중수",
        "description": "중급 알고리즘 경로를 완료했습니다",
        "icon": "🚀",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.MASTERY,
        "requirement": "intermediate_path_completed",
        "requirement_value": 1,
        "xp_reward": 400,
    },
    {
        "name": "알고리즘 고수",
        "description": "고급 알고리즘 경로를 완료했습니다",
        "icon": "🏆",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.MASTERY,
        "requirement": "advanced_path_completed",
        "requirement_value": 1,
        "xp_reward": 600,
    },
]
