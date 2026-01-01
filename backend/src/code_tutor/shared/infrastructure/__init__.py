"""Shared infrastructure components"""

from code_tutor.shared.infrastructure.database import (
    Base,
    async_session_factory,
    close_db,
    get_async_session,
    init_db,
)
from code_tutor.shared.infrastructure.logging import (
    LoggerMixin,
    configure_logging,
    get_logger,
)
from code_tutor.shared.infrastructure.redis import (
    RedisClient,
    close_redis,
    get_redis_client,
)
from code_tutor.shared.infrastructure.repository import Repository, SQLAlchemyRepository
from code_tutor.shared.infrastructure.uow import SQLAlchemyUnitOfWork, UnitOfWork

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
