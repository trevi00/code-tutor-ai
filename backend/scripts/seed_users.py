"""Seed script for initial users (admin and test users)."""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import select

from code_tutor.shared.infrastructure.database import init_db, get_session_context
from code_tutor.identity.infrastructure.models import UserModel
from code_tutor.identity.domain.value_objects import UserRole
from code_tutor.shared.security import hash_password


# ============== User Data ==============

USERS_DATA = [
    {
        "email": "admin@codetutor.ai",
        "username": "admin",
        "password": "Admin123!@#",
        "role": UserRole.ADMIN,
        "bio": "Code Tutor AI Administrator",
    },
    {
        "email": "test@example.com",
        "username": "testuser",
        "password": "Test123!@#",
        "role": UserRole.STUDENT,
        "bio": "Test user for development",
    },
    {
        "email": "demo@codetutor.ai",
        "username": "demo",
        "password": "Demo123!@#",
        "role": UserRole.STUDENT,
        "bio": "Demo account for showcasing features",
    },
]


async def seed_users():
    """Seed initial users."""
    print("Seeding Users...")

    await init_db()

    async with get_session_context() as session:
        created = 0
        skipped = 0

        for user_data in USERS_DATA:
            # Check if user already exists
            result = await session.execute(
                select(UserModel).where(UserModel.email == user_data["email"])
            )
            if result.scalar_one_or_none():
                print(f"  [SKIP] {user_data['username']} - already exists")
                skipped += 1
                continue

            # Create user
            user = UserModel(
                id=uuid4(),
                email=user_data["email"],
                username=user_data["username"],
                hashed_password=hash_password(user_data["password"]),
                role=user_data["role"],
                is_active=True,
                is_verified=True,
                bio=user_data["bio"],
            )
            session.add(user)
            print(f"  [ADD] {user_data['username']} ({user_data['role'].value})")
            created += 1

        await session.commit()

    print(f"\nUsers seeding complete: {created} created, {skipped} skipped")
    return created, skipped


if __name__ == "__main__":
    asyncio.run(seed_users())
