"""Pytest configuration and fixtures"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_tutor.main import app
from code_tutor.shared.infrastructure.database import Base, get_async_session

# Test database URL (use SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client"""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_user_data() -> dict[str, Any]:
    """Sample user registration data"""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "TestPass123",
    }


@pytest.fixture
def sample_problem_data() -> dict[str, Any]:
    """Sample problem creation data"""
    return {
        "title": "Two Sum",
        "description": "Given an array of integers, return indices of two numbers that add up to a target.",
        "difficulty": "easy",
        "category": "array",
        "constraints": "2 <= nums.length <= 10^4",
        "hints": ["Try using a hash map"],
        "solution_template": "def two_sum(nums, target):\n    pass",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "test_cases": [
            {
                "input_data": "[2,7,11,15]\n9",
                "expected_output": "[0, 1]",
                "is_sample": True,
            },
        ],
    }
