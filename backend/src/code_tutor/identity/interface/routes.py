"""Identity API routes"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from code_tutor.identity.application.dto import (
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from code_tutor.identity.application.services import AuthService, UserService
from code_tutor.identity.domain.repository import UserRepository
from code_tutor.identity.interface.dependencies import (
    get_current_active_user,
    get_redis,
    get_user_repository,
)
from code_tutor.shared.exceptions import AppException
from code_tutor.shared.infrastructure.redis import RedisClient

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    redis: Annotated[RedisClient | None, Depends(get_redis)],
) -> AuthService:
    """Get auth service instance"""
    return AuthService(user_repo, redis)


def get_user_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> UserService:
    """Get user service instance"""
    return UserService(user_repo)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    request: RegisterRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    """Register a new user account"""
    try:
        return await auth_service.register(request)
    except AppException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST
            if e.code == "VALIDATION_ERROR"
            else status.HTTP_409_CONFLICT,
            detail=e.message,
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email and password",
)
async def login(
    request: LoginRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> LoginResponse:
    """Authenticate user and return tokens"""
    try:
        return await auth_service.login(request)
    except AppException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Refresh access token using refresh token"""
    try:
        return await auth_service.refresh_tokens(request.refresh_token)
    except AppException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
        )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout current user",
)
async def logout(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Logout user and invalidate tokens"""
    await auth_service.logout(credentials.credentials)
    return MessageResponse(message="Successfully logged out")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_me(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> UserResponse:
    """Get current authenticated user's profile"""
    return current_user


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """Update current user's profile (username, bio)"""
    try:
        return await user_service.update_profile(current_user.id, request)
    except AppException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST
            if e.code == "VALIDATION_ERROR"
            else status.HTTP_409_CONFLICT,
            detail=e.message,
        )


@router.put(
    "/me/password",
    response_model=MessageResponse,
    summary="Change current user password",
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> MessageResponse:
    """Change current user's password"""
    try:
        await user_service.change_password(current_user.id, request)
        return MessageResponse(message="Password changed successfully")
    except AppException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
