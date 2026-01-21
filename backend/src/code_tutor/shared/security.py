"""Security utilities - password hashing and JWT handling"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import bcrypt
import jwt
from jwt.exceptions import InvalidTokenError

from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import UnauthorizedError


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token"""
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = utc_now() + expires_delta
    else:
        expire = utc_now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update(
        {
            "exp": expire,
            "iat": utc_now(),
            "jti": str(uuid4()),
            "type": "access",
        }
    )

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT refresh token"""
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = utc_now() + expires_delta
    else:
        expire = utc_now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update(
        {
            "exp": expire,
            "iat": utc_now(),
            "jti": str(uuid4()),
            "type": "refresh",
        }
    )

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT token"""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except InvalidTokenError as e:
        raise UnauthorizedError(f"Invalid token: {str(e)}")


def get_token_jti(token: str) -> str:
    """Extract JTI from token without full verification"""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        return payload.get("jti", "")
    except InvalidTokenError:
        return ""


class TokenPayload:
    """Token payload data class"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.sub: str = payload.get("sub", "")
        # Use fromtimestamp with UTC timezone for timezone-aware datetimes
        self.exp: datetime = datetime.fromtimestamp(payload.get("exp", 0), tz=UTC)
        self.iat: datetime = datetime.fromtimestamp(payload.get("iat", 0), tz=UTC)
        self.jti: str = payload.get("jti", "")
        self.type: str = payload.get("type", "")

    @property
    def user_id(self) -> str:
        return self.sub

    @property
    def is_expired(self) -> bool:
        return utc_now() > self.exp

    @property
    def is_access_token(self) -> bool:
        return self.type == "access"

    @property
    def is_refresh_token(self) -> bool:
        return self.type == "refresh"
