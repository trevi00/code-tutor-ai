"""Seed script for gamification badges and challenges."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select

from code_tutor.shared.infrastructure.database import init_db, get_session_context
from code_tutor.gamification.infrastructure.models import BadgeModel, ChallengeModel
from code_tutor.gamification.domain.value_objects import (
    BadgeRarity,
    BadgeCategory,
    ChallengeType,
)


# ============== Badge Data ==============

BADGES_DATA = [
    # Problem Solving - Common
    {
        "name": "First Blood",
        "description": "첫 번째 문제를 해결했습니다!",
        "icon": "trophy",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "Problem Solver",
        "description": "10개의 문제를 해결했습니다.",
        "icon": "check-circle",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 10,
        "xp_reward": 100,
    },
    {
        "name": "Dedicated Learner",
        "description": "50개의 문제를 해결했습니다.",
        "icon": "star",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 50,
        "xp_reward": 250,
    },
    {
        "name": "Algorithm Expert",
        "description": "100개의 문제를 해결했습니다.",
        "icon": "award",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 100,
        "xp_reward": 500,
    },
    {
        "name": "Coding Master",
        "description": "500개의 문제를 해결했습니다.",
        "icon": "crown",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 500,
        "xp_reward": 1000,
    },
    {
        "name": "Legendary Coder",
        "description": "1000개의 문제를 해결했습니다!",
        "icon": "gem",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved",
        "requirement_value": 1000,
        "xp_reward": 2500,
    },

    # Perfect Solutions
    {
        "name": "Perfect Start",
        "description": "첫 시도에 문제를 해결했습니다!",
        "icon": "zap",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved_first_try",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "Sharp Mind",
        "description": "10개의 문제를 첫 시도에 해결했습니다.",
        "icon": "brain",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved_first_try",
        "requirement_value": 10,
        "xp_reward": 200,
    },
    {
        "name": "Genius",
        "description": "50개의 문제를 첫 시도에 해결했습니다.",
        "icon": "lightbulb",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.PROBLEM_SOLVING,
        "requirement": "problems_solved_first_try",
        "requirement_value": 50,
        "xp_reward": 750,
    },

    # Streak Badges
    {
        "name": "Getting Started",
        "description": "3일 연속 학습했습니다.",
        "icon": "flame",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 3,
        "xp_reward": 50,
    },
    {
        "name": "Week Warrior",
        "description": "7일 연속 학습했습니다.",
        "icon": "fire",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 7,
        "xp_reward": 150,
    },
    {
        "name": "Consistent Learner",
        "description": "14일 연속 학습했습니다.",
        "icon": "calendar",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 14,
        "xp_reward": 300,
    },
    {
        "name": "Monthly Champion",
        "description": "30일 연속 학습했습니다.",
        "icon": "calendar-check",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 30,
        "xp_reward": 500,
    },
    {
        "name": "Unstoppable",
        "description": "100일 연속 학습했습니다!",
        "icon": "infinity",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.STREAK,
        "requirement": "current_streak",
        "requirement_value": 100,
        "xp_reward": 2000,
    },

    # Pattern Mastery
    {
        "name": "Pattern Learner",
        "description": "첫 번째 패턴을 마스터했습니다.",
        "icon": "puzzle",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 1,
        "xp_reward": 75,
    },
    {
        "name": "Pattern Hunter",
        "description": "5개의 패턴을 마스터했습니다.",
        "icon": "target",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 5,
        "xp_reward": 200,
    },
    {
        "name": "Pattern Master",
        "description": "15개의 패턴을 마스터했습니다.",
        "icon": "layers",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 15,
        "xp_reward": 500,
    },
    {
        "name": "Pattern Guru",
        "description": "모든 핵심 패턴을 마스터했습니다.",
        "icon": "book-open",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.MASTERY,
        "requirement": "patterns_mastered",
        "requirement_value": 25,
        "xp_reward": 1000,
    },

    # Roadmap/Learning Path
    {
        "name": "Journey Begins",
        "description": "첫 번째 레슨을 완료했습니다.",
        "icon": "play",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 1,
        "xp_reward": 25,
    },
    {
        "name": "Active Learner",
        "description": "10개의 레슨을 완료했습니다.",
        "icon": "book",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 10,
        "xp_reward": 100,
    },
    {
        "name": "Knowledge Seeker",
        "description": "50개의 레슨을 완료했습니다.",
        "icon": "graduation-cap",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.MASTERY,
        "requirement": "lessons_completed",
        "requirement_value": 50,
        "xp_reward": 300,
    },
    {
        "name": "Path Finder",
        "description": "첫 번째 학습 경로를 완료했습니다.",
        "icon": "map",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.MASTERY,
        "requirement": "paths_completed",
        "requirement_value": 1,
        "xp_reward": 500,
    },
    {
        "name": "Full Stack Learner",
        "description": "모든 학습 경로를 완료했습니다.",
        "icon": "mountain",
        "rarity": BadgeRarity.LEGENDARY,
        "category": BadgeCategory.MASTERY,
        "requirement": "paths_completed",
        "requirement_value": 4,
        "xp_reward": 3000,
    },

    # Social/Collaboration
    {
        "name": "Team Player",
        "description": "첫 번째 협업 세션에 참여했습니다.",
        "icon": "users",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "collaborations_count",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "Collaborator",
        "description": "10회 협업 세션에 참여했습니다.",
        "icon": "handshake",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "collaborations_count",
        "requirement_value": 10,
        "xp_reward": 200,
    },
    {
        "name": "Community Leader",
        "description": "50회 협업 세션에 참여했습니다.",
        "icon": "globe",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.SOCIAL,
        "requirement": "collaborations_count",
        "requirement_value": 50,
        "xp_reward": 500,
    },

    # Playground
    {
        "name": "Creator",
        "description": "첫 번째 플레이그라운드를 만들었습니다.",
        "icon": "code",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_created",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "Prolific Creator",
        "description": "10개의 플레이그라운드를 만들었습니다.",
        "icon": "folder-code",
        "rarity": BadgeRarity.UNCOMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_created",
        "requirement_value": 10,
        "xp_reward": 200,
    },
    {
        "name": "Sharer",
        "description": "플레이그라운드를 공유했습니다.",
        "icon": "share-2",
        "rarity": BadgeRarity.COMMON,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_shared",
        "requirement_value": 1,
        "xp_reward": 50,
    },
    {
        "name": "Open Source Spirit",
        "description": "10개의 플레이그라운드를 공유했습니다.",
        "icon": "heart",
        "rarity": BadgeRarity.RARE,
        "category": BadgeCategory.SOCIAL,
        "requirement": "playgrounds_shared",
        "requirement_value": 10,
        "xp_reward": 300,
    },

    # Special
    {
        "name": "Early Adopter",
        "description": "Code Tutor AI의 초기 사용자입니다.",
        "icon": "rocket",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.SPECIAL,
        "requirement": "special_early_adopter",
        "requirement_value": 1,
        "xp_reward": 500,
    },
    {
        "name": "Bug Hunter",
        "description": "버그를 발견하고 리포트했습니다.",
        "icon": "bug",
        "rarity": BadgeRarity.EPIC,
        "category": BadgeCategory.SPECIAL,
        "requirement": "special_bug_report",
        "requirement_value": 1,
        "xp_reward": 250,
    },
]


# ============== Challenge Data ==============

def get_challenges_data():
    """Generate challenges with dynamic dates."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    return [
        # Daily Challenges
        {
            "name": "Daily Grind",
            "description": "오늘 문제 3개 해결하기",
            "challenge_type": ChallengeType.DAILY,
            "target_action": "problems_solved",
            "target_value": 3,
            "xp_reward": 100,
            "start_date": today_start,
            "end_date": today_start + timedelta(days=1),
        },
        {
            "name": "Code Practice",
            "description": "오늘 타이핑 연습 2회 완료하기",
            "challenge_type": ChallengeType.DAILY,
            "target_action": "typing_completed",
            "target_value": 2,
            "xp_reward": 50,
            "start_date": today_start,
            "end_date": today_start + timedelta(days=1),
        },

        # Weekly Challenges
        {
            "name": "Weekly Warrior",
            "description": "이번 주 문제 15개 해결하기",
            "challenge_type": ChallengeType.WEEKLY,
            "target_action": "problems_solved",
            "target_value": 15,
            "xp_reward": 500,
            "start_date": week_start,
            "end_date": week_start + timedelta(days=7),
        },
        {
            "name": "Streak Builder",
            "description": "이번 주 5일 연속 학습하기",
            "challenge_type": ChallengeType.WEEKLY,
            "target_action": "streak_days",
            "target_value": 5,
            "xp_reward": 300,
            "start_date": week_start,
            "end_date": week_start + timedelta(days=7),
        },
        {
            "name": "Pattern Practice",
            "description": "이번 주 새로운 패턴 3개 학습하기",
            "challenge_type": ChallengeType.WEEKLY,
            "target_action": "patterns_learned",
            "target_value": 3,
            "xp_reward": 400,
            "start_date": week_start,
            "end_date": week_start + timedelta(days=7),
        },

        # Monthly Challenges
        {
            "name": "Monthly Master",
            "description": "이번 달 문제 50개 해결하기",
            "challenge_type": ChallengeType.MONTHLY,
            "target_action": "problems_solved",
            "target_value": 50,
            "xp_reward": 1500,
            "start_date": month_start,
            "end_date": (month_start + timedelta(days=32)).replace(day=1),
        },
        {
            "name": "Dedicated Learner",
            "description": "이번 달 20일 이상 학습하기",
            "challenge_type": ChallengeType.MONTHLY,
            "target_action": "active_days",
            "target_value": 20,
            "xp_reward": 1000,
            "start_date": month_start,
            "end_date": (month_start + timedelta(days=32)).replace(day=1),
        },
        {
            "name": "Path Progress",
            "description": "이번 달 학습 경로 하나 완료하기",
            "challenge_type": ChallengeType.MONTHLY,
            "target_action": "paths_completed",
            "target_value": 1,
            "xp_reward": 2000,
            "start_date": month_start,
            "end_date": (month_start + timedelta(days=32)).replace(day=1),
        },
    ]


async def seed_badges():
    """Seed badges."""
    print("Seeding Badges...")

    await init_db()

    async with get_session_context() as session:
        created = 0
        skipped = 0

        for badge_data in BADGES_DATA:
            # Check if badge already exists
            result = await session.execute(
                select(BadgeModel).where(BadgeModel.name == badge_data["name"])
            )
            if result.scalar_one_or_none():
                print(f"  [SKIP] {badge_data['name']} - already exists")
                skipped += 1
                continue

            # Create badge
            badge = BadgeModel(
                id=uuid4(),
                name=badge_data["name"],
                description=badge_data["description"],
                icon=badge_data["icon"],
                rarity=badge_data["rarity"],
                category=badge_data["category"],
                requirement=badge_data["requirement"],
                requirement_value=badge_data["requirement_value"],
                xp_reward=badge_data["xp_reward"],
            )
            session.add(badge)
            print(f"  [ADD] {badge_data['name']} ({badge_data['rarity'].value})")
            created += 1

        await session.commit()

    print(f"\nBadges seeding complete: {created} created, {skipped} skipped")
    return created, skipped


async def seed_challenges():
    """Seed challenges."""
    print("\nSeeding Challenges...")

    await init_db()

    async with get_session_context() as session:
        # Delete old challenges first
        result = await session.execute(select(ChallengeModel))
        existing = result.scalars().all()
        for challenge in existing:
            await session.delete(challenge)

        challenges_data = get_challenges_data()
        created = 0

        for challenge_data in challenges_data:
            challenge = ChallengeModel(
                id=uuid4(),
                name=challenge_data["name"],
                description=challenge_data["description"],
                challenge_type=challenge_data["challenge_type"],
                target_action=challenge_data["target_action"],
                target_value=challenge_data["target_value"],
                xp_reward=challenge_data["xp_reward"],
                start_date=challenge_data["start_date"],
                end_date=challenge_data["end_date"],
            )
            session.add(challenge)
            print(f"  [ADD] {challenge_data['name']} ({challenge_data['challenge_type'].value})")
            created += 1

        await session.commit()

    print(f"\nChallenges seeding complete: {created} created")
    return created


async def seed_gamification():
    """Seed all gamification data."""
    await seed_badges()
    await seed_challenges()


if __name__ == "__main__":
    asyncio.run(seed_gamification())
