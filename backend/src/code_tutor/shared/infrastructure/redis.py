"""Redis client configuration"""

import json
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis

from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)

_redis_client: Redis | None = None
_redis_available: bool | None = None


async def get_redis_client() -> Redis | None:
    """Get or create Redis client. Returns None if Redis is not available."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is None:
        settings = get_settings()
        try:
            _redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            await _redis_client.ping()
            _redis_available = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Token features will be limited.")
            _redis_available = False
            _redis_client = None
            return None

    return _redis_client


async def close_redis() -> None:
    """Close Redis connection"""
    global _redis_client, _redis_available
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        _redis_available = None


class RedisClient:
    """High-level Redis client wrapper. Handles missing Redis gracefully."""

    def __init__(self, client: Redis | None) -> None:
        self._client = client

    @property
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self._client is not None

    @classmethod
    async def create(cls) -> "RedisClient":
        """Factory method to create RedisClient"""
        client = await get_redis_client()
        return cls(client)

    async def get(self, key: str) -> str | None:
        """Get a string value"""
        if not self._client:
            return None
        return await self._client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        expire_seconds: int | None = None,
    ) -> bool:
        """Set a string value with optional expiration"""
        if not self._client:
            return True  # Pretend success when Redis is not available
        return await self._client.set(key, value, ex=expire_seconds)

    async def delete(self, key: str) -> int:
        """Delete a key"""
        if not self._client:
            return 0
        return await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self._client:
            return False
        return await self._client.exists(key) > 0

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Get and parse JSON value"""
        value = await self.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        expire_seconds: int | None = None,
    ) -> bool:
        """Set JSON value"""
        return await self.set(key, json.dumps(value), expire_seconds)

    async def incr(self, key: str) -> int:
        """Increment integer value"""
        if not self._client:
            return 0
        return await self._client.incr(key)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key"""
        if not self._client:
            return True
        return await self._client.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        """Get TTL of key"""
        if not self._client:
            return -1
        return await self._client.ttl(key)

    # Session/Token management helpers
    async def store_refresh_token(
        self,
        user_id: str,
        token: str,
        expire_days: int = 7,
    ) -> bool:
        """Store refresh token for user"""
        key = f"refresh_token:{user_id}"
        return await self.set(key, token, expire_seconds=expire_days * 86400)

    async def get_refresh_token(self, user_id: str) -> str | None:
        """Get stored refresh token for user"""
        key = f"refresh_token:{user_id}"
        return await self.get(key)

    async def invalidate_refresh_token(self, user_id: str) -> int:
        """Invalidate refresh token for user"""
        key = f"refresh_token:{user_id}"
        return await self.delete(key)

    async def blacklist_token(self, token_jti: str, expire_seconds: int) -> bool:
        """Add token to blacklist"""
        key = f"token_blacklist:{token_jti}"
        return await self.set(key, "1", expire_seconds=expire_seconds)

    async def is_token_blacklisted(self, token_jti: str) -> bool:
        """Check if token is blacklisted"""
        key = f"token_blacklist:{token_jti}"
        return await self.exists(key)
