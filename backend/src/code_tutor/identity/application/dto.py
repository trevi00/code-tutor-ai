"""Identity DTOs (Data Transfer Objects)"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# Request DTOs
class RegisterRequest(BaseModel):
    """User registration request"""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """User login request"""

    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    """Change password request"""

    old_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""

    refresh_token: str


class UpdateProfileRequest(BaseModel):
    """Update user profile request"""

    username: str | None = Field(None, min_length=3, max_length=30)
    bio: str | None = Field(None, max_length=200)


# Response DTOs
class UserResponse(BaseModel):
    """User response DTO"""

    id: UUID
    email: str
    username: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None
    bio: str | None = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response DTO"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class LoginResponse(BaseModel):
    """Login response DTO"""

    user: UserResponse
    tokens: TokenResponse


class MessageResponse(BaseModel):
    """Generic message response"""

    message: str
