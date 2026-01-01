"""Dependency Injection Container using dependency-injector"""

from dependency_injector import containers, providers

from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.database import async_session_factory
from code_tutor.shared.infrastructure.redis import RedisClient


class Container(containers.DeclarativeContainer):
    """Main DI container for the application"""

    wiring_config = containers.WiringConfiguration(
        packages=["code_tutor"],
    )

    # Configuration
    config = providers.Singleton(get_settings)

    # Database session factory
    db_session_factory = providers.Singleton(async_session_factory)

    # Redis client (async factory)
    redis_client = providers.Factory(RedisClient.create)
