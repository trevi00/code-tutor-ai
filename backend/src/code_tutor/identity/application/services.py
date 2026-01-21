"""Identity application services (use cases)"""

from uuid import UUID

from code_tutor.identity.application.dto import (
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from code_tutor.identity.domain.entities import User
from code_tutor.identity.domain.repository import UserRepository
from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import ConflictError, NotFoundError, UnauthorizedError
from code_tutor.shared.infrastructure.logging import get_logger
from code_tutor.shared.infrastructure.redis import RedisClient
from code_tutor.shared.security import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_jti,
    utc_now,
)

logger = get_logger(__name__)


class UserService:
    """User management service"""

    def __init__(self, user_repository: UserRepository) -> None:
        self._user_repo = user_repository

    async def get_user_by_id(self, user_id: UUID) -> UserResponse:
        """Get user by ID"""
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))
        return self._to_response(user)

    async def get_user_by_email(self, email: str) -> UserResponse:
        """Get user by email"""
        user = await self._user_repo.get_by_email(email)
        if user is None:
            raise NotFoundError("User", email)
        return self._to_response(user)

    async def update_profile(
        self,
        user_id: UUID,
        request: UpdateProfileRequest,
    ) -> UserResponse:
        """Update user profile"""
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))

        # Check if new username is taken by another user
        if request.username is not None:
            existing = await self._user_repo.get_by_username(request.username)
            if existing and existing.id != user_id:
                raise ConflictError(
                    "Username is already taken",
                    {"username": request.username},
                )

        # Update profile
        user.update_profile(
            username=request.username,
            bio=request.bio,
        )
        updated_user = await self._user_repo.update(user)

        logger.info("User profile updated", user_id=str(user_id))
        return self._to_response(updated_user)

    async def change_password(
        self,
        user_id: UUID,
        request: ChangePasswordRequest,
    ) -> None:
        """Change user password"""
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            raise NotFoundError("User", str(user_id))

        # change_password validates old password and updates
        user.change_password(request.old_password, request.new_password)
        await self._user_repo.update(user)

        logger.info("User password changed", user_id=str(user_id))

    def _to_response(self, user: User) -> UserResponse:
        """Convert User entity to UserResponse DTO"""
        return UserResponse(
            id=user.id,
            email=str(user.email) if user.email else "",
            username=str(user.username) if user.username else "",
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            bio=user.bio,
        )


class AuthService:
    """Authentication service"""

    def __init__(
        self,
        user_repository: UserRepository,
        redis_client: RedisClient | None = None,
    ) -> None:
        self._user_repo = user_repository
        self._redis = redis_client
        self._settings = get_settings()

    async def register(self, request: RegisterRequest) -> UserResponse:
        """Register a new user"""
        # Check for existing email
        if await self._user_repo.exists_by_email(request.email):
            raise ConflictError(
                "User with this email already exists",
                {"email": request.email},
            )

        # Check for existing username
        if await self._user_repo.exists_by_username(request.username):
            raise ConflictError(
                "User with this username already exists",
                {"username": request.username},
            )

        # Create user
        user = User.create(
            email=request.email,
            username=request.username,
            password=request.password,
        )

        # Save to repository
        saved_user = await self._user_repo.add(user)

        logger.info(
            "User registered",
            user_id=str(saved_user.id),
            email=request.email,
        )

        return self._to_response(saved_user)

    async def login(self, request: LoginRequest) -> LoginResponse:
        """Authenticate user and return tokens"""
        # Find user by email
        user = await self._user_repo.get_by_email(request.email)
        if user is None:
            raise UnauthorizedError("Invalid email or password")

        # Verify password
        if not user.verify_password(request.password):
            raise UnauthorizedError("Invalid email or password")

        # Check if user is active
        if not user.is_active:
            raise UnauthorizedError("Account is deactivated")

        # Record login
        user.record_login()
        await self._user_repo.update(user)

        # Create tokens
        tokens = await self._create_tokens(user)

        logger.info("User logged in", user_id=str(user.id))

        return LoginResponse(
            user=self._to_response(user),
            tokens=tokens,
        )

    async def refresh_tokens(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        # Decode refresh token
        payload = decode_token(refresh_token)
        token_payload = TokenPayload(payload)

        # Validate token type
        if not token_payload.is_refresh_token:
            raise UnauthorizedError("Invalid token type")

        # Check if token is blacklisted
        if self._redis:
            if await self._redis.is_token_blacklisted(token_payload.jti):
                raise UnauthorizedError("Token has been revoked")

        # Get user
        user = await self._user_repo.get_by_id(UUID(token_payload.user_id))
        if user is None:
            raise UnauthorizedError("User not found")

        if not user.is_active:
            raise UnauthorizedError("Account is deactivated")

        # Blacklist old refresh token
        if self._redis:
            # Calculate actual remaining time until expiration
            remaining_time = max(
                0, int((token_payload.exp - utc_now()).total_seconds())
            )
            if remaining_time > 0:
                await self._redis.blacklist_token(token_payload.jti, remaining_time)

        # Create new tokens
        return await self._create_tokens(user)

    async def logout(self, access_token: str) -> None:
        """Logout user by blacklisting tokens"""
        if not self._redis:
            return

        jti = get_token_jti(access_token)
        if jti:
            # Blacklist for remaining token lifetime
            await self._redis.blacklist_token(
                jti,
                self._settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )

        logger.info("User logged out")

    async def _create_tokens(self, user: User) -> TokenResponse:
        """Create access and refresh tokens for user"""
        token_data = {
            "sub": str(user.id),
            "email": str(user.email) if user.email else "",
            "role": user.role.value,
        }

        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        # Store refresh token in Redis if available
        if self._redis:
            await self._redis.store_refresh_token(
                str(user.id),
                refresh_token,
                self._settings.REFRESH_TOKEN_EXPIRE_DAYS,
            )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self._settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    def _to_response(self, user: User) -> UserResponse:
        """Convert User entity to UserResponse DTO"""
        return UserResponse(
            id=user.id,
            email=str(user.email) if user.email else "",
            username=str(user.username) if user.username else "",
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            bio=user.bio,
        )
