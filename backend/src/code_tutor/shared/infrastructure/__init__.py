"""Shared infrastructure components"""

from code_tutor.shared.infrastructure.database import (
    Base,
    async_session_factory,
    get_async_session,
    init_db,
    close_db,
)
from code_tutor.shared.infrastructure.redis import RedisClient, get_redis_client, close_redis
from code_tutor.shared.infrastructure.logging import configure_logging, get_logger, LoggerMixin
from code_tutor.shared.infrastructure.repository import Repository, SQLAlchemyRepository
from code_tutor.shared.infrastructure.uow import UnitOfWork, SQLAlchemyUnitOfWork

__all__ = [
    # Database
    "Base",
    "async_session_factory",
    "get_async_session",
    "init_db",
    "close_db",
    # Redis
    "RedisClient",
    "get_redis_client",
    "close_redis",
    # Logging
    "configure_logging",
    "get_logger",
    "LoggerMixin",
    # Repository & UoW
    "Repository",
    "SQLAlchemyRepository",
    "UnitOfWork",
    "SQLAlchemyUnitOfWork",
]
