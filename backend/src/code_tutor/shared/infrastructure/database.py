"""Database configuration and session management"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from code_tutor.shared.config import get_settings

# Naming convention for constraints
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class NaiveUTCDateTime(TypeDecorator[datetime]):
    """DateTime type that stores timezone-aware datetimes as naive UTC.

    The domain layer uses timezone-aware UTC datetimes (``datetime.now(UTC)``)
    while the persistence schema uses ``TIMESTAMP WITHOUT TIME ZONE`` columns.
    asyncpg rejects aware datetimes for naive columns, so values are normalized
    to naive UTC at the bind boundary. Naive values are assumed to already be
    UTC and are passed through unchanged.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: Any | None, dialect: Dialect
    ) -> datetime | None:
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # Ensure Mapped[datetime] annotations (without an explicit column type)
    # also normalize aware datetimes to naive UTC on write.
    type_annotation_map = {datetime: NaiveUTCDateTime()}


# Global engine and session factory
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the database engine"""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
    return _engine


def async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory"""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions"""
    factory = async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions (for non-FastAPI use)"""
    factory = async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database (create tables)"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections"""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
