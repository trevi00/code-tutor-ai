"""Gamification domain value objects."""

from enum import Enum


class BadgeRarity(str, Enum):
    """Badge rarity levels."""

    COMMON = "common"        # Easy to get
    UNCOMMON = "uncommon"    # Some effort needed
    RARE = "rare"            # Significant achievement
    EPIC = "epic"            # Major milestone
    LEGENDARY = "legendary"  # Exceptional achievement


class BadgeCategory(str, Enum):
    """Badge categories."""

    PROBLEM_SOLVING = "problem_solving"  # Solving problems
    STREAK = "streak"                    # Consecutive days
    MASTERY = "mastery"                  # Pattern mastery
    SOCIAL = "social"                    # Collaboration, sharing
    SPECIAL = "special"                  # Special events


class ChallengeType(str, Enum):
    """Challenge types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SPECIAL = "special"


class ChallengeStatus(str, Enum):
    """Challenge status."""

    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"


# XP rewards for different actions
XP_REWARDS = {
    "problem_solved": 50,
    "problem_solved_first_try": 100,
    "daily_login": 10,
    "streak_bonus": 5,  # Per day in streak
    "code_review_received": 20,
    "collaboration_session": 30,
    "playground_created": 15,
    "playground_shared": 25,
    # Typing practice rewards
    "typing_attempt_completed": 20,  # Each attempt completion
    "typing_exercise_mastered": 50,  # Bonus for mastering (5 completions)
    "typing_high_accuracy": 10,  # Bonus for 95%+ accuracy
}

# Level thresholds
LEVEL_THRESHOLDS = [
    0,      # Level 1
    100,    # Level 2
    250,    # Level 3
    500,    # Level 4
    850,    # Level 5
    1300,   # Level 6
    1900,   # Level 7
    2650,   # Level 8
    3550,   # Level 9
    4600,   # Level 10
    5800,   # Level 11
    7150,   # Level 12
    8650,   # Level 13
    10300,  # Level 14
    12100,  # Level 15
    14050,  # Level 16
    16150,  # Level 17
    18400,  # Level 18
    20800,  # Level 19
    23350,  # Level 20
]

# Level titles
LEVEL_TITLES = {
    1: "입문자",
    2: "초보자",
    3: "학습자",
    4: "탐구자",
    5: "도전자",
    6: "해결사",
    7: "숙련자",
    8: "전문가",
    9: "마스터",
    10: "그랜드마스터",
    11: "레전드",
    12: "신화",
    13: "영웅",
    14: "전설",
    15: "불멸",
    16: "초월자",
    17: "신",
    18: "창조자",
    19: "우주",
    20: "무한",
}


def calculate_level(xp: int) -> int:
    """Calculate level from XP."""
    for level, threshold in enumerate(LEVEL_THRESHOLDS, start=1):
        if xp < threshold:
            return level - 1
    return len(LEVEL_THRESHOLDS)


def xp_for_next_level(current_xp: int) -> tuple[int, int]:
    """Get XP needed for next level and current progress.

    Returns:
        (current_level_xp, next_level_xp)
    """
    level = calculate_level(current_xp)
    if level >= len(LEVEL_THRESHOLDS):
        return LEVEL_THRESHOLDS[-1], LEVEL_THRESHOLDS[-1]

    current_threshold = LEVEL_THRESHOLDS[level - 1] if level > 0 else 0
    next_threshold = LEVEL_THRESHOLDS[level] if level < len(LEVEL_THRESHOLDS) else LEVEL_THRESHOLDS[-1]

    return current_threshold, next_threshold


def get_level_title(level: int) -> str:
    """Get title for a level."""
    if level in LEVEL_TITLES:
        return LEVEL_TITLES[level]
    return LEVEL_TITLES[max(LEVEL_TITLES.keys())]
