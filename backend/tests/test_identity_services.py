"""Unit tests for Identity Domain"""

import pytest
from uuid import uuid4

from code_tutor.identity.domain.value_objects import (
    Email,
    Password,
    Username,
    UserId,
    UserRole,
    HashedPassword,
)
from code_tutor.shared.exceptions import ValidationError


class TestEmail:
    """Tests for Email value object"""

    def test_valid_email(self):
        """Test creating valid email"""
        email = Email("test@example.com")
        assert str(email) == "test@example.com"

    def test_email_local_part(self):
        """Test email local part extraction"""
        email = Email("user@example.com")
        assert email.local_part == "user"

    def test_email_domain(self):
        """Test email domain extraction"""
        email = Email("user@example.com")
        assert email.domain == "example.com"

    def test_invalid_email_format(self):
        """Test invalid email format"""
        with pytest.raises(ValidationError):
            Email("invalid-email")

    def test_empty_email(self):
        """Test empty email"""
        with pytest.raises(ValidationError):
            Email("")

    def test_email_equality(self):
        """Test email equality"""
        email1 = Email("test@example.com")
        email2 = Email("test@example.com")
        assert email1 == email2

    def test_email_with_subdomain(self):
        """Test email with subdomain"""
        email = Email("user@mail.example.com")
        assert str(email) == "user@mail.example.com"
        assert email.domain == "mail.example.com"


class TestPassword:
    """Tests for Password value object"""

    def test_valid_password(self):
        """Test valid password creation"""
        password = Password("SecurePass123")
        assert password.value == "SecurePass123"

    def test_short_password(self):
        """Test password minimum length"""
        with pytest.raises(ValidationError):
            Password("short")

    def test_empty_password(self):
        """Test empty password"""
        with pytest.raises(ValidationError):
            Password("")

    def test_password_strength_valid(self):
        """Test valid strong password"""
        password = Password("SecurePass123!")  # 특수문자 포함
        password.validate_strength()  # Should not raise

    def test_password_strength_no_uppercase(self):
        """Test password without uppercase"""
        password = Password("securepass123")
        with pytest.raises(ValidationError):
            password.validate_strength()

    def test_password_strength_no_lowercase(self):
        """Test password without lowercase"""
        password = Password("SECUREPASS123")
        with pytest.raises(ValidationError):
            password.validate_strength()

    def test_password_strength_no_digit(self):
        """Test password without digit"""
        password = Password("SecurePassword")
        with pytest.raises(ValidationError):
            password.validate_strength()


class TestUsername:
    """Tests for Username value object"""

    def test_valid_username(self):
        """Test valid username creation"""
        username = Username("testuser")
        assert str(username) == "testuser"

    def test_username_with_numbers(self):
        """Test username with numbers"""
        username = Username("user123")
        assert str(username) == "user123"

    def test_username_with_underscore(self):
        """Test username with underscore"""
        username = Username("test_user")
        assert str(username) == "test_user"

    def test_username_with_hyphen(self):
        """Test username with hyphen"""
        username = Username("test-user")
        assert str(username) == "test-user"

    def test_short_username(self):
        """Test username minimum length"""
        with pytest.raises(ValidationError):
            Username("ab")

    def test_empty_username(self):
        """Test empty username"""
        with pytest.raises(ValidationError):
            Username("")

    def test_username_with_special_chars(self):
        """Test username with invalid special characters"""
        with pytest.raises(ValidationError):
            Username("user@name")

    def test_long_username(self):
        """Test username maximum length"""
        with pytest.raises(ValidationError):
            Username("a" * 31)


class TestUserId:
    """Tests for UserId value object"""

    def test_generate_user_id(self):
        """Test generating new user ID"""
        user_id = UserId.generate()
        assert user_id.value is not None

    def test_user_id_from_string(self):
        """Test creating user ID from string"""
        uuid_str = str(uuid4())
        user_id = UserId.from_string(uuid_str)
        assert str(user_id) == uuid_str

    def test_invalid_user_id_string(self):
        """Test creating user ID from invalid string"""
        with pytest.raises(ValidationError):
            UserId.from_string("invalid-uuid")

    def test_user_id_equality(self):
        """Test user ID equality"""
        uuid_val = uuid4()
        user_id1 = UserId(value=uuid_val)
        user_id2 = UserId(value=uuid_val)
        assert user_id1 == user_id2


class TestUserRole:
    """Tests for UserRole enum"""

    def test_user_role_values(self):
        """Test user role enum values"""
        assert UserRole.STUDENT.value == "student"
        assert UserRole.ADMIN.value == "admin"

    def test_user_role_from_string(self):
        """Test creating role from string value"""
        role = UserRole("student")
        assert role == UserRole.STUDENT


class TestHashedPassword:
    """Tests for HashedPassword value object"""

    def test_hashed_password_never_exposed(self):
        """Test that hashed password never exposes hash"""
        hashed = HashedPassword(value="$2b$12$somehash")
        assert str(hashed) == "***"

    def test_hashed_password_value(self):
        """Test accessing hashed password value"""
        hash_value = "$2b$12$somehash"
        hashed = HashedPassword(value=hash_value)
        assert hashed.value == hash_value


# ==================== User Entity Tests ====================


class TestUserEntity:
    """Tests for User aggregate root"""

    def test_user_create(self):
        """Test creating a new user via factory method"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        assert user.id is not None
        assert str(user.email) == "test@example.com"
        assert str(user.username) == "testuser"
        assert user.hashed_password is not None
        assert user.role == UserRole.STUDENT
        assert user.is_active is True
        assert user.is_verified is False
        assert len(user.domain_events) == 1  # UserCreated event

    def test_user_create_with_admin_role(self):
        """Test creating a user with admin role"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="admin@example.com",
            username="adminuser",
            password="SecurePass123!",
            role=UserRole.ADMIN,
        )

        assert user.role == UserRole.ADMIN

    def test_user_verify_password_correct(self):
        """Test verifying correct password"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        assert user.verify_password("SecurePass123!") is True

    def test_user_verify_password_incorrect(self):
        """Test verifying incorrect password"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        assert user.verify_password("WrongPassword123") is False

    def test_user_verify_password_no_password(self):
        """Test verify password when no password is set"""
        from code_tutor.identity.domain.entities import User

        user = User(
            id=uuid4(),
            email=Email("test@example.com"),
            username=Username("testuser"),
            hashed_password=None,
        )

        assert user.verify_password("anypassword") is False

    def test_user_change_password(self):
        """Test changing password"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )
        user.clear_domain_events()

        user.change_password("SecurePass123!", "NewSecurePass456!")

        assert user.verify_password("NewSecurePass456!") is True
        assert user.verify_password("SecurePass123!") is False
        assert len(user.domain_events) == 1  # PasswordChanged event

    def test_user_change_password_wrong_old_password(self):
        """Test changing password with wrong old password"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        with pytest.raises(ValidationError, match="Current password is incorrect"):
            user.change_password("WrongOldPassword", "NewSecurePass456!")

    def test_user_record_login(self):
        """Test recording login"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )
        user.clear_domain_events()

        assert user.last_login_at is None
        user.record_login()

        assert user.last_login_at is not None
        assert len(user.domain_events) == 1  # UserLoggedIn event

    def test_user_activate(self):
        """Test activating user"""
        from code_tutor.identity.domain.entities import User

        user = User(
            id=uuid4(),
            email=Email("test@example.com"),
            username=Username("testuser"),
            is_active=False,
        )

        user.activate()
        assert user.is_active is True

    def test_user_deactivate(self):
        """Test deactivating user"""
        from code_tutor.identity.domain.entities import User

        user = User(
            id=uuid4(),
            email=Email("test@example.com"),
            username=Username("testuser"),
            is_active=True,
        )

        user.deactivate()
        assert user.is_active is False

    def test_user_verify_email(self):
        """Test verifying email"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        assert user.is_verified is False
        user.verify_email()
        assert user.is_verified is True

    def test_user_promote_to_admin(self):
        """Test promoting user to admin"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        assert user.role == UserRole.STUDENT
        user.promote_to_admin()
        assert user.role == UserRole.ADMIN

    def test_user_update_profile(self):
        """Test updating user profile"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        user.update_profile(username="newusername", bio="My new bio")

        assert str(user.username) == "newusername"
        assert user.bio == "My new bio"

    def test_user_update_profile_username_only(self):
        """Test updating only username"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        user.update_profile(username="newusername")

        assert str(user.username) == "newusername"
        assert user.bio is None

    def test_user_update_profile_bio_only(self):
        """Test updating only bio"""
        from code_tutor.identity.domain.entities import User

        user = User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

        user.update_profile(bio="New bio text")

        assert str(user.username) == "testuser"
        assert user.bio == "New bio text"


# ==================== Service Tests with Mocks ====================


from unittest.mock import AsyncMock, MagicMock, patch


class TestUserService:
    """Tests for UserService"""

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository"""
        return AsyncMock()

    @pytest.fixture
    def user_service(self, mock_user_repo):
        """Create UserService instance"""
        from code_tutor.identity.application.services import UserService

        return UserService(mock_user_repo)

    @pytest.fixture
    def sample_user(self):
        """Create sample user entity"""
        from code_tutor.identity.domain.entities import User

        return User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_service, mock_user_repo, sample_user):
        """Test getting user by ID"""
        mock_user_repo.get_by_id.return_value = sample_user

        result = await user_service.get_user_by_id(sample_user.id)

        assert result.id == sample_user.id
        assert result.email == "test@example.com"
        assert result.username == "testuser"
        mock_user_repo.get_by_id.assert_called_once_with(sample_user.id)

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, user_service, mock_user_repo):
        """Test getting non-existent user by ID"""
        from code_tutor.shared.exceptions import NotFoundError

        mock_user_repo.get_by_id.return_value = None
        user_id = uuid4()

        with pytest.raises(NotFoundError):
            await user_service.get_user_by_id(user_id)

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, user_service, mock_user_repo, sample_user):
        """Test getting user by email"""
        mock_user_repo.get_by_email.return_value = sample_user

        result = await user_service.get_user_by_email("test@example.com")

        assert result.email == "test@example.com"
        mock_user_repo.get_by_email.assert_called_once_with("test@example.com")

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, user_service, mock_user_repo):
        """Test getting non-existent user by email"""
        from code_tutor.shared.exceptions import NotFoundError

        mock_user_repo.get_by_email.return_value = None

        with pytest.raises(NotFoundError):
            await user_service.get_user_by_email("nonexistent@example.com")

    @pytest.mark.asyncio
    async def test_update_profile(self, user_service, mock_user_repo, sample_user):
        """Test updating user profile"""
        from code_tutor.identity.application.dto import UpdateProfileRequest

        mock_user_repo.get_by_id.return_value = sample_user
        mock_user_repo.get_by_username.return_value = None
        mock_user_repo.update.return_value = sample_user

        request = UpdateProfileRequest(username="newusername", bio="New bio")
        result = await user_service.update_profile(sample_user.id, request)

        assert result.id == sample_user.id
        mock_user_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_profile_username_taken(
        self, user_service, mock_user_repo, sample_user
    ):
        """Test updating profile with taken username"""
        from code_tutor.identity.application.dto import UpdateProfileRequest
        from code_tutor.identity.domain.entities import User
        from code_tutor.shared.exceptions import ConflictError

        other_user = User.create(
            email="other@example.com",
            username="otheruser",
            password="SecurePass123!",
        )

        mock_user_repo.get_by_id.return_value = sample_user
        mock_user_repo.get_by_username.return_value = other_user

        request = UpdateProfileRequest(username="otheruser")

        with pytest.raises(ConflictError, match="Username is already taken"):
            await user_service.update_profile(sample_user.id, request)

    @pytest.mark.asyncio
    async def test_update_profile_user_not_found(self, user_service, mock_user_repo):
        """Test updating profile for non-existent user"""
        from code_tutor.identity.application.dto import UpdateProfileRequest
        from code_tutor.shared.exceptions import NotFoundError

        mock_user_repo.get_by_id.return_value = None

        request = UpdateProfileRequest(username="newname")

        with pytest.raises(NotFoundError):
            await user_service.update_profile(uuid4(), request)

    @pytest.mark.asyncio
    async def test_change_password(self, user_service, mock_user_repo, sample_user):
        """Test changing password"""
        from code_tutor.identity.application.dto import ChangePasswordRequest

        mock_user_repo.get_by_id.return_value = sample_user
        mock_user_repo.update.return_value = sample_user

        request = ChangePasswordRequest(
            old_password="SecurePass123!", new_password="NewSecurePass456!"
        )
        await user_service.change_password(sample_user.id, request)

        mock_user_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_change_password_user_not_found(self, user_service, mock_user_repo):
        """Test changing password for non-existent user"""
        from code_tutor.identity.application.dto import ChangePasswordRequest
        from code_tutor.shared.exceptions import NotFoundError

        mock_user_repo.get_by_id.return_value = None

        request = ChangePasswordRequest(
            old_password="OldPass123!", new_password="NewPass456!"
        )

        with pytest.raises(NotFoundError):
            await user_service.change_password(uuid4(), request)


class TestAuthService:
    """Tests for AuthService"""

    @pytest.fixture
    def mock_user_repo(self):
        """Create mock user repository"""
        return AsyncMock()

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock = AsyncMock()
        mock.is_token_blacklisted.return_value = False
        mock.blacklist_token.return_value = None
        mock.store_refresh_token.return_value = None
        return mock

    @pytest.fixture
    def auth_service(self, mock_user_repo, mock_redis):
        """Create AuthService instance"""
        from code_tutor.identity.application.services import AuthService

        return AuthService(mock_user_repo, mock_redis)

    @pytest.fixture
    def auth_service_no_redis(self, mock_user_repo):
        """Create AuthService without Redis"""
        from code_tutor.identity.application.services import AuthService

        return AuthService(mock_user_repo, None)

    @pytest.fixture
    def sample_user(self):
        """Create sample user entity"""
        from code_tutor.identity.domain.entities import User

        return User.create(
            email="test@example.com",
            username="testuser",
            password="SecurePass123!",
        )

    @pytest.mark.asyncio
    async def test_register_success(self, auth_service, mock_user_repo):
        """Test successful user registration"""
        from code_tutor.identity.application.dto import RegisterRequest

        mock_user_repo.exists_by_email.return_value = False
        mock_user_repo.exists_by_username.return_value = False
        mock_user_repo.add.side_effect = lambda user: user

        request = RegisterRequest(
            email="new@example.com",
            username="newuser",
            password="SecurePass123!",
        )

        result = await auth_service.register(request)

        assert result.email == "new@example.com"
        assert result.username == "newuser"
        mock_user_repo.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_email_exists(self, auth_service, mock_user_repo):
        """Test registration with existing email"""
        from code_tutor.identity.application.dto import RegisterRequest
        from code_tutor.shared.exceptions import ConflictError

        mock_user_repo.exists_by_email.return_value = True

        request = RegisterRequest(
            email="existing@example.com",
            username="newuser",
            password="SecurePass123!",
        )

        with pytest.raises(ConflictError, match="email already exists"):
            await auth_service.register(request)

    @pytest.mark.asyncio
    async def test_register_username_exists(self, auth_service, mock_user_repo):
        """Test registration with existing username"""
        from code_tutor.identity.application.dto import RegisterRequest
        from code_tutor.shared.exceptions import ConflictError

        mock_user_repo.exists_by_email.return_value = False
        mock_user_repo.exists_by_username.return_value = True

        request = RegisterRequest(
            email="new@example.com",
            username="existinguser",
            password="SecurePass123!",
        )

        with pytest.raises(ConflictError, match="username already exists"):
            await auth_service.register(request)

    @pytest.mark.asyncio
    async def test_login_success(self, auth_service, mock_user_repo, sample_user):
        """Test successful login"""
        from code_tutor.identity.application.dto import LoginRequest

        mock_user_repo.get_by_email.return_value = sample_user
        mock_user_repo.update.return_value = sample_user

        request = LoginRequest(email="test@example.com", password="SecurePass123!")

        result = await auth_service.login(request)

        assert result.user.email == "test@example.com"
        assert result.tokens.access_token is not None
        assert result.tokens.refresh_token is not None

    @pytest.mark.asyncio
    async def test_login_user_not_found(self, auth_service, mock_user_repo):
        """Test login with non-existent email"""
        from code_tutor.identity.application.dto import LoginRequest
        from code_tutor.shared.exceptions import UnauthorizedError

        mock_user_repo.get_by_email.return_value = None

        request = LoginRequest(email="nonexistent@example.com", password="Password123!")

        with pytest.raises(UnauthorizedError, match="Invalid email or password"):
            await auth_service.login(request)

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_service, mock_user_repo, sample_user):
        """Test login with wrong password"""
        from code_tutor.identity.application.dto import LoginRequest
        from code_tutor.shared.exceptions import UnauthorizedError

        mock_user_repo.get_by_email.return_value = sample_user

        request = LoginRequest(email="test@example.com", password="WrongPassword123!")

        with pytest.raises(UnauthorizedError, match="Invalid email or password"):
            await auth_service.login(request)

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, auth_service, mock_user_repo, sample_user):
        """Test login with inactive user"""
        from code_tutor.identity.application.dto import LoginRequest
        from code_tutor.shared.exceptions import UnauthorizedError

        sample_user.deactivate()
        mock_user_repo.get_by_email.return_value = sample_user

        request = LoginRequest(email="test@example.com", password="SecurePass123!")

        with pytest.raises(UnauthorizedError, match="Account is deactivated"):
            await auth_service.login(request)

    @pytest.mark.asyncio
    async def test_logout_with_redis(self, auth_service, mock_redis):
        """Test logout with Redis"""
        from code_tutor.shared.security import create_access_token

        token = create_access_token({"sub": str(uuid4()), "email": "test@example.com", "role": "student"})

        await auth_service.logout(token)

        mock_redis.blacklist_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_logout_without_redis(self, auth_service_no_redis):
        """Test logout without Redis (no-op)"""
        from code_tutor.shared.security import create_access_token

        token = create_access_token({"sub": str(uuid4()), "email": "test@example.com", "role": "student"})

        # Should not raise
        await auth_service_no_redis.logout(token)

    @pytest.mark.asyncio
    async def test_refresh_tokens_success(
        self, auth_service, mock_user_repo, mock_redis, sample_user
    ):
        """Test successful token refresh"""
        from code_tutor.shared.security import create_refresh_token

        mock_user_repo.get_by_id.return_value = sample_user
        mock_redis.is_token_blacklisted.return_value = False

        refresh_token = create_refresh_token(
            {"sub": str(sample_user.id), "email": "test@example.com", "role": "student"}
        )

        result = await auth_service.refresh_tokens(refresh_token)

        assert result.access_token is not None
        assert result.refresh_token is not None

    @pytest.mark.asyncio
    async def test_refresh_tokens_with_access_token(self, auth_service, mock_user_repo):
        """Test refresh with access token instead of refresh token"""
        from code_tutor.shared.exceptions import UnauthorizedError
        from code_tutor.shared.security import create_access_token

        access_token = create_access_token(
            {"sub": str(uuid4()), "email": "test@example.com", "role": "student"}
        )

        with pytest.raises(UnauthorizedError, match="Invalid token type"):
            await auth_service.refresh_tokens(access_token)

    @pytest.mark.asyncio
    async def test_refresh_tokens_blacklisted(
        self, auth_service, mock_user_repo, mock_redis, sample_user
    ):
        """Test refresh with blacklisted token"""
        from code_tutor.shared.exceptions import UnauthorizedError
        from code_tutor.shared.security import create_refresh_token

        mock_redis.is_token_blacklisted.return_value = True

        refresh_token = create_refresh_token(
            {"sub": str(sample_user.id), "email": "test@example.com", "role": "student"}
        )

        with pytest.raises(UnauthorizedError, match="Token has been revoked"):
            await auth_service.refresh_tokens(refresh_token)

    @pytest.mark.asyncio
    async def test_refresh_tokens_user_not_found(
        self, auth_service, mock_user_repo, mock_redis
    ):
        """Test refresh when user no longer exists"""
        from code_tutor.shared.exceptions import UnauthorizedError
        from code_tutor.shared.security import create_refresh_token

        mock_user_repo.get_by_id.return_value = None
        mock_redis.is_token_blacklisted.return_value = False

        refresh_token = create_refresh_token(
            {"sub": str(uuid4()), "email": "test@example.com", "role": "student"}
        )

        with pytest.raises(UnauthorizedError, match="User not found"):
            await auth_service.refresh_tokens(refresh_token)

    @pytest.mark.asyncio
    async def test_refresh_tokens_inactive_user(
        self, auth_service, mock_user_repo, mock_redis, sample_user
    ):
        """Test refresh with inactive user"""
        from code_tutor.shared.exceptions import UnauthorizedError
        from code_tutor.shared.security import create_refresh_token

        sample_user.deactivate()
        mock_user_repo.get_by_id.return_value = sample_user
        mock_redis.is_token_blacklisted.return_value = False

        refresh_token = create_refresh_token(
            {"sub": str(sample_user.id), "email": "test@example.com", "role": "student"}
        )

        with pytest.raises(UnauthorizedError, match="Account is deactivated"):
            await auth_service.refresh_tokens(refresh_token)


# ==================== Domain Events Tests ====================


class TestDomainEvents:
    """Tests for domain events"""

    def test_user_created_event(self):
        """Test UserCreated event"""
        from code_tutor.identity.domain.events import UserCreated

        user_id = uuid4()
        event = UserCreated(user_id=user_id, email="test@example.com", username="testuser")

        assert event.user_id == user_id
        assert event.email == "test@example.com"
        assert event.username == "testuser"
        assert event.event_type == "UserCreated"

    def test_user_logged_in_event(self):
        """Test UserLoggedIn event"""
        from code_tutor.identity.domain.events import UserLoggedIn

        user_id = uuid4()
        event = UserLoggedIn(user_id=user_id)

        assert event.user_id == user_id
        assert event.event_type == "UserLoggedIn"

    def test_password_changed_event(self):
        """Test PasswordChanged event"""
        from code_tutor.identity.domain.events import PasswordChanged

        user_id = uuid4()
        event = PasswordChanged(user_id=user_id)

        assert event.user_id == user_id
        assert event.event_type == "PasswordChanged"


# ==================== Routes Unit Tests ====================


class TestIdentityRoutesUnit:
    """Unit tests for identity routes configuration"""

    def test_auth_router_prefix(self):
        """Test auth router has correct prefix"""
        from code_tutor.identity.interface.routes import router

        assert router.prefix == "/auth"

    def test_auth_router_tags(self):
        """Test auth router has correct tags"""
        from code_tutor.identity.interface.routes import router

        assert "Authentication" in router.tags

    def test_auth_router_has_expected_routes(self):
        """Test auth router has expected routes"""
        from code_tutor.identity.interface.routes import router

        route_paths = [r.path for r in router.routes]

        expected_paths = [
            "/auth/register",
            "/auth/login",
            "/auth/refresh",
            "/auth/logout",
            "/auth/me",
            "/auth/me/password",
        ]

        for path in expected_paths:
            assert path in route_paths, f"Missing route: {path}"


# ==================== Dependencies Unit Tests ====================


class TestDependenciesUnit:
    """Unit tests for dependencies"""

    def test_security_scheme_exists(self):
        """Test security scheme is configured"""
        from code_tutor.identity.interface.dependencies import security

        assert security is not None

    def test_dependency_functions_exist(self):
        """Test dependency functions exist"""
        from code_tutor.identity.interface.dependencies import (
            get_user_repository,
            get_redis,
            get_current_user,
            get_current_active_user,
            get_admin_user,
            get_optional_user,
        )

        assert callable(get_user_repository)
        assert callable(get_redis)
        assert callable(get_current_user)
        assert callable(get_current_active_user)
        assert callable(get_admin_user)
        assert callable(get_optional_user)
