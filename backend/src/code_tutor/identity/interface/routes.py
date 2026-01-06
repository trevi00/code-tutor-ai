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
from code_tutor.shared.middleware.rate_limiter import (
    login_rate_limit,
    register_rate_limit,
    refresh_rate_limit,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


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
    summary="회원가입",
    description="새로운 사용자 계정을 생성합니다. 이메일과 사용자명은 고유해야 합니다.",
    responses={
        201: {"description": "회원가입 성공"},
        400: {"description": "유효하지 않은 입력 데이터"},
        409: {"description": "이메일 또는 사용자명 중복"},
        429: {"description": "요청 횟수 초과 (분당 3회 제한)"},
    },
)
async def register(
    request: RegisterRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    _: Annotated[None, Depends(register_rate_limit)],
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
    summary="로그인",
    description="이메일과 비밀번호로 로그인하여 JWT 토큰을 발급받습니다.",
    responses={
        200: {"description": "로그인 성공, 액세스 토큰과 리프레시 토큰 반환"},
        401: {"description": "이메일 또는 비밀번호 불일치"},
        429: {"description": "요청 횟수 초과 (분당 5회 제한)"},
    },
)
async def login(
    request: LoginRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    _: Annotated[None, Depends(login_rate_limit)],
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
    summary="토큰 갱신",
    description="리프레시 토큰을 사용하여 새로운 액세스 토큰을 발급받습니다.",
    responses={
        200: {"description": "토큰 갱신 성공"},
        401: {"description": "유효하지 않거나 만료된 리프레시 토큰"},
        429: {"description": "요청 횟수 초과 (분당 10회 제한)"},
    },
)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    _: Annotated[None, Depends(refresh_rate_limit)],
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
    summary="로그아웃",
    description="현재 세션을 종료하고 토큰을 무효화합니다.",
    responses={
        200: {"description": "로그아웃 성공"},
        401: {"description": "인증 필요"},
    },
)
async def logout(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Logout user and invalidate tokens"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    await auth_service.logout(credentials.credentials)
    return MessageResponse(message="Successfully logged out")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="내 프로필 조회",
    description="현재 로그인한 사용자의 프로필 정보를 조회합니다.",
    responses={
        200: {"description": "프로필 조회 성공"},
        401: {"description": "인증 필요"},
    },
)
async def get_me(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> UserResponse:
    """Get current authenticated user's profile"""
    return current_user


@router.put(
    "/me",
    response_model=UserResponse,
    summary="내 프로필 수정",
    description="현재 로그인한 사용자의 프로필(사용자명, 소개글)을 수정합니다.",
    responses={
        200: {"description": "프로필 수정 성공"},
        400: {"description": "유효하지 않은 입력 데이터"},
        401: {"description": "인증 필요"},
        409: {"description": "사용자명 중복"},
    },
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
    summary="비밀번호 변경",
    description="현재 비밀번호를 확인한 후 새 비밀번호로 변경합니다.",
    responses={
        200: {"description": "비밀번호 변경 성공"},
        400: {"description": "현재 비밀번호 불일치 또는 유효하지 않은 새 비밀번호"},
        401: {"description": "인증 필요"},
    },
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
