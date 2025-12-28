"""Identity API dependencies"""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.domain.repository import UserRepository
from code_tutor.identity.infrastructure.repository import SQLAlchemyUserRepository
from code_tutor.shared.exceptions import UnauthorizedError
from code_tutor.shared.infrastructure.database import get_async_session
from code_tutor.shared.infrastructure.redis import RedisClient, get_redis_client
from code_tutor.shared.security import decode_token, TokenPayload

# Security scheme
security = HTTPBearer()


async def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> UserRepository:
    """Get user repository instance"""
    return SQLAlchemyUserRepository(session)


async def get_redis(
) -> RedisClient | None:
    """Get Redis client (optional)"""
    try:
        client = await get_redis_client()
        return RedisClient(client)
    except Exception:
        return None


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    redis: Annotated[RedisClient | None, Depends(get_redis)],
) -> UserResponse:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials

    try:
        payload = decode_token(token)
        token_payload = TokenPayload(payload)

        # Validate token type
        if not token_payload.is_access_token:
            raise UnauthorizedError("Invalid token type")

        # Check if token is blacklisted
        if redis:
            if await redis.is_token_blacklisted(token_payload.jti):
                raise UnauthorizedError("Token has been revoked")

        # Get user
        user = await user_repo.get_by_id(UUID(token_payload.user_id))
        if user is None:
            raise UnauthorizedError("User not found")

        return UserResponse(
            id=user.id,
            email=str(user.email) if user.email else "",
            username=str(user.username) if user.username else "",
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )

    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
) -> UserResponse:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return current_user
