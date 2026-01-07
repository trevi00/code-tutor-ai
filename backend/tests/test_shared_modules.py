"""Unit tests for Shared Modules"""

import pytest
from uuid import uuid4

from code_tutor.shared.api_response import (
    success_response,
    error_response,
    paginated_response,
    ErrorCodes,
)
from code_tutor.shared.exceptions import (
    AppException,
    NotFoundError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DomainError,
    ForbiddenError,
    UnauthorizedError,
    SandboxError,
    LLMError,
)
from code_tutor.shared.config import Settings


class TestApiResponse:
    """Tests for API response utilities"""

    def test_success_response(self):
        """Test success response format"""
        data = {"id": 1, "name": "test"}
        response = success_response(data)

        assert response["success"] is True
        assert response["data"] == data
        assert "meta" in response
        assert "request_id" in response["meta"]

    def test_success_response_structure(self):
        """Test success response structure"""
        response = success_response({"result": "ok"})

        assert response["success"] is True
        assert "meta" in response
        assert "timestamp" in response["meta"]

    def test_error_response(self):
        """Test error response format"""
        response = error_response(
            code="NOT_FOUND",
            message="Resource not found",
        )

        assert response["success"] is False
        assert response["error"]["code"] == "NOT_FOUND"
        assert response["error"]["message"] == "Resource not found"

    def test_error_response_with_details(self):
        """Test error response with details"""
        response = error_response(
            code="VALIDATION_ERROR",
            message="Invalid input",
            details={"field": "email", "reason": "invalid format"},
        )

        assert response["success"] is False
        assert response["error"]["details"] is not None
        assert response["error"]["details"]["field"] == "email"

    def test_paginated_response(self):
        """Test paginated response"""
        items = [{"id": 1}, {"id": 2}]
        response = paginated_response(items, page=1, limit=10, total_count=50)

        assert response["success"] is True
        assert response["data"] == items
        assert response["pagination"]["current_page"] == 1
        assert response["pagination"]["total_pages"] == 5
        assert response["pagination"]["has_next"] is True
        assert response["pagination"]["has_prev"] is False


class TestErrorCodes:
    """Tests for error codes"""

    def test_client_error_codes(self):
        """Test 4xx error codes exist"""
        assert ErrorCodes.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCodes.INVALID_CREDENTIALS == "INVALID_CREDENTIALS"
        assert ErrorCodes.TOKEN_EXPIRED == "TOKEN_EXPIRED"
        assert ErrorCodes.FORBIDDEN == "FORBIDDEN"

    def test_server_error_codes(self):
        """Test 5xx error codes exist"""
        assert ErrorCodes.INTERNAL_ERROR == "INTERNAL_ERROR"
        assert ErrorCodes.LLM_ERROR == "LLM_ERROR"
        assert ErrorCodes.SANDBOX_ERROR == "SANDBOX_ERROR"


class TestExceptions:
    """Tests for custom exceptions"""

    def test_app_exception_base(self):
        """Test AppException base class"""
        error = AppException("Something went wrong", code="TEST_ERROR")
        assert str(error) == "Something went wrong"
        assert error.code == "TEST_ERROR"

    def test_not_found_error(self):
        """Test NotFoundError"""
        error = NotFoundError("User", "123")
        assert "User" in str(error)
        assert error.code == "NOT_FOUND"
        assert error.entity == "User"

    def test_not_found_error_without_id(self):
        """Test NotFoundError without identifier"""
        error = NotFoundError("User")
        assert "User not found" == str(error)

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Invalid email format")
        assert "email" in str(error).lower()
        assert error.code == "VALIDATION_ERROR"

    def test_authentication_error(self):
        """Test AuthenticationError"""
        error = AuthenticationError("Invalid credentials")
        assert error.code == "AUTHENTICATION_ERROR"

    def test_authentication_error_default(self):
        """Test AuthenticationError with default message"""
        error = AuthenticationError()
        assert "Authentication failed" in str(error)

    def test_authorization_error(self):
        """Test AuthorizationError"""
        error = AuthorizationError("Access denied")
        assert error.code == "AUTHORIZATION_ERROR"

    def test_conflict_error(self):
        """Test ConflictError"""
        error = ConflictError("Email already exists")
        assert error.code == "CONFLICT"

    def test_domain_error(self):
        """Test DomainError"""
        error = DomainError("Invalid state transition")
        assert error.code == "DOMAIN_ERROR"

    def test_forbidden_error(self):
        """Test ForbiddenError"""
        error = ForbiddenError("Permission denied")
        assert error.code == "FORBIDDEN"

    def test_unauthorized_error(self):
        """Test UnauthorizedError"""
        error = UnauthorizedError("Please login")
        assert error.code == "UNAUTHORIZED"

    def test_sandbox_error(self):
        """Test SandboxError"""
        error = SandboxError("Execution timeout")
        assert error.code == "SANDBOX_ERROR"

    def test_llm_error(self):
        """Test LLMError"""
        error = LLMError("API rate limit exceeded")
        assert error.code == "LLM_ERROR"

    def test_exception_inheritance(self):
        """Test all errors inherit from AppException"""
        assert issubclass(NotFoundError, AppException)
        assert issubclass(ValidationError, AppException)
        assert issubclass(AuthenticationError, AppException)
        assert issubclass(AuthorizationError, AppException)
        assert issubclass(ConflictError, AppException)
        assert issubclass(DomainError, AppException)
        assert issubclass(ForbiddenError, AppException)


class TestSettings:
    """Tests for Settings configuration"""

    def test_settings_defaults(self):
        """Test settings have default values"""
        settings = Settings()

        assert settings.APP_NAME is not None
        assert settings.DEBUG is not None

    def test_settings_database_url(self):
        """Test database URL setting"""
        settings = Settings()
        assert settings.DATABASE_URL is not None

    def test_settings_jwt(self):
        """Test JWT settings"""
        settings = Settings()
        assert settings.JWT_SECRET_KEY is not None
        assert settings.JWT_ALGORITHM == "HS256"
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0

    def test_settings_sandbox(self):
        """Test sandbox settings"""
        settings = Settings()
        assert settings.SANDBOX_TIMEOUT_SECONDS > 0
        assert settings.SANDBOX_MEMORY_LIMIT_MB > 0


# ============== Security Module Tests ==============


class TestSecurityUtcNow:
    """Tests for utc_now function."""

    def test_utc_now_returns_datetime(self):
        """Test utc_now returns datetime."""
        from code_tutor.shared.security import utc_now
        from datetime import datetime, timezone

        result = utc_now()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test utc_now returns recent time."""
        from code_tutor.shared.security import utc_now
        from datetime import datetime, timezone, timedelta

        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)

        assert before <= result <= after


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password(self):
        """Test password hashing."""
        from code_tutor.shared.security import hash_password

        password = "SecurePassword123!"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_hash_password_unique(self):
        """Test each hash is unique due to salt."""
        from code_tutor.shared.security import hash_password

        password = "SamePassword"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2  # Different salts

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        from code_tutor.shared.security import hash_password, verify_password

        password = "MyPassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        from code_tutor.shared.security import hash_password, verify_password

        password = "CorrectPassword"
        hashed = hash_password(password)

        assert verify_password("WrongPassword", hashed) is False


class TestJWTTokens:
    """Tests for JWT token functions."""

    def test_create_access_token(self):
        """Test creating access token."""
        from code_tutor.shared.security import create_access_token, decode_token

        data = {"sub": "user123", "email": "test@example.com", "role": "student"}
        token = create_access_token(data)

        assert token is not None
        assert len(token) > 0

        # Decode and verify
        payload = decode_token(token)
        assert payload["sub"] == "user123"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_access_token_with_expires_delta(self):
        """Test creating access token with custom expiry."""
        from code_tutor.shared.security import create_access_token, decode_token
        from datetime import timedelta

        data = {"sub": "user123"}
        token = create_access_token(data, expires_delta=timedelta(hours=2))

        payload = decode_token(token)
        assert payload["sub"] == "user123"

    def test_create_refresh_token(self):
        """Test creating refresh token."""
        from code_tutor.shared.security import create_refresh_token, decode_token

        data = {"sub": "user123", "email": "test@example.com", "role": "student"}
        token = create_refresh_token(data)

        assert token is not None

        payload = decode_token(token)
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"

    def test_create_refresh_token_with_expires_delta(self):
        """Test creating refresh token with custom expiry."""
        from code_tutor.shared.security import create_refresh_token, decode_token
        from datetime import timedelta

        data = {"sub": "user123"}
        token = create_refresh_token(data, expires_delta=timedelta(days=14))

        payload = decode_token(token)
        assert payload["type"] == "refresh"

    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        from code_tutor.shared.security import decode_token
        from code_tutor.shared.exceptions import UnauthorizedError

        with pytest.raises(UnauthorizedError) as exc_info:
            decode_token("invalid.token.here")

        assert "Invalid token" in str(exc_info.value)

    def test_get_token_jti(self):
        """Test getting JTI from token."""
        from code_tutor.shared.security import create_access_token, get_token_jti

        data = {"sub": "user123"}
        token = create_access_token(data)

        jti = get_token_jti(token)
        assert jti is not None
        assert len(jti) > 0

    def test_get_token_jti_invalid_token(self):
        """Test getting JTI from invalid token returns empty."""
        from code_tutor.shared.security import get_token_jti

        jti = get_token_jti("invalid.token.here")
        assert jti == ""


class TestTokenPayload:
    """Tests for TokenPayload class."""

    def test_token_payload_init(self):
        """Test TokenPayload initialization."""
        from code_tutor.shared.security import TokenPayload
        from datetime import datetime, timezone
        import time

        now = int(time.time())
        payload = {
            "sub": "user123",
            "exp": now + 3600,
            "iat": now,
            "jti": "unique-jti-123",
            "type": "access",
        }

        token_payload = TokenPayload(payload)

        assert token_payload.sub == "user123"
        assert token_payload.jti == "unique-jti-123"
        assert token_payload.type == "access"
        assert token_payload.user_id == "user123"

    def test_token_payload_is_access_token(self):
        """Test is_access_token property."""
        from code_tutor.shared.security import TokenPayload
        import time

        now = int(time.time())
        payload = {"sub": "user1", "exp": now + 3600, "iat": now, "jti": "jti", "type": "access"}
        token_payload = TokenPayload(payload)

        assert token_payload.is_access_token is True
        assert token_payload.is_refresh_token is False

    def test_token_payload_is_refresh_token(self):
        """Test is_refresh_token property."""
        from code_tutor.shared.security import TokenPayload
        import time

        now = int(time.time())
        payload = {"sub": "user1", "exp": now + 3600, "iat": now, "jti": "jti", "type": "refresh"}
        token_payload = TokenPayload(payload)

        assert token_payload.is_refresh_token is True
        assert token_payload.is_access_token is False

    def test_token_payload_is_expired(self):
        """Test is_expired property."""
        from code_tutor.shared.security import TokenPayload
        import time

        now = int(time.time())
        # Expired token
        expired_payload = {"sub": "user1", "exp": now - 3600, "iat": now - 7200, "jti": "jti", "type": "access"}
        expired_token = TokenPayload(expired_payload)
        assert expired_token.is_expired is True

        # Valid token
        valid_payload = {"sub": "user1", "exp": now + 3600, "iat": now, "jti": "jti", "type": "access"}
        valid_token = TokenPayload(valid_payload)
        assert valid_token.is_expired is False


# ============== Domain Base Tests ==============


class TestDomainUtcNow:
    """Tests for domain utc_now function."""

    def test_utc_now_returns_datetime(self):
        """Test utc_now returns datetime."""
        from code_tutor.shared.domain.base import utc_now
        from datetime import datetime, timezone

        result = utc_now()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc


class TestValueObject:
    """Tests for ValueObject base class."""

    def test_value_object_equality(self):
        """Test value objects are equal by attributes."""
        from code_tutor.shared.domain.base import ValueObject

        class Money(ValueObject):
            def __init__(self, amount: int, currency: str):
                self.amount = amount
                self.currency = currency

        m1 = Money(100, "USD")
        m2 = Money(100, "USD")
        m3 = Money(200, "USD")

        assert m1 == m2
        assert m1 != m3

    def test_value_object_inequality_different_type(self):
        """Test value objects not equal to different types."""
        from code_tutor.shared.domain.base import ValueObject

        class Money(ValueObject):
            def __init__(self, amount: int):
                self.amount = amount

        m = Money(100)
        assert m != "not a money"
        assert m != 100

    def test_value_object_hash(self):
        """Test value objects can be hashed."""
        from code_tutor.shared.domain.base import ValueObject

        class Money(ValueObject):
            def __init__(self, amount: int, currency: str):
                self.amount = amount
                self.currency = currency

        m1 = Money(100, "USD")
        m2 = Money(100, "USD")

        assert hash(m1) == hash(m2)

        # Can be used in sets
        money_set = {m1, m2}
        assert len(money_set) == 1

    def test_value_object_repr(self):
        """Test value object repr."""
        from code_tutor.shared.domain.base import ValueObject

        class Money(ValueObject):
            def __init__(self, amount: int):
                self.amount = amount

        m = Money(100)
        repr_str = repr(m)
        assert "Money" in repr_str
        assert "amount=100" in repr_str


class TestEntity:
    """Tests for Entity base class."""

    def test_entity_creation_with_id(self):
        """Test entity creation with provided ID."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            def __init__(self, id=None, name=""):
                super().__init__(id)
                self.name = name

        user_id = uuid4()
        user = User(id=user_id, name="John")

        assert user.id == user_id
        assert user.name == "John"

    def test_entity_creation_without_id(self):
        """Test entity creation generates ID."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user = User()
        assert user.id is not None

    def test_entity_timestamps(self):
        """Test entity has timestamps."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user = User()
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_entity_touch(self):
        """Test entity _touch updates timestamp."""
        from code_tutor.shared.domain.base import Entity
        import time

        class User(Entity):
            def update_name(self, name):
                self._touch()

        user = User()
        original_updated_at = user.updated_at

        time.sleep(0.01)  # Small delay
        user.update_name("New Name")

        assert user.updated_at >= original_updated_at

    def test_entity_equality(self):
        """Test entities are equal by ID."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user_id = uuid4()
        user1 = User(id=user_id)
        user2 = User(id=user_id)
        user3 = User()

        assert user1 == user2
        assert user1 != user3

    def test_entity_inequality_different_type(self):
        """Test entity not equal to different types."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user = User()
        assert user != "not a user"
        assert user != 123

    def test_entity_hash(self):
        """Test entity can be hashed."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user_id = uuid4()
        user1 = User(id=user_id)
        user2 = User(id=user_id)

        assert hash(user1) == hash(user2)

        # Can be used in sets
        users = {user1, user2}
        assert len(users) == 1

    def test_entity_repr(self):
        """Test entity repr."""
        from code_tutor.shared.domain.base import Entity

        class User(Entity):
            pass

        user = User()
        repr_str = repr(user)
        assert "User" in repr_str
        assert "id=" in repr_str


class TestAggregateRoot:
    """Tests for AggregateRoot base class."""

    def test_aggregate_root_creation(self):
        """Test aggregate root creation."""
        from code_tutor.shared.domain.base import AggregateRoot

        class Order(AggregateRoot):
            pass

        order = Order()
        assert order.id is not None
        assert order.version == 0
        assert order.domain_events == []

    def test_aggregate_root_add_domain_event(self):
        """Test adding domain events."""
        from code_tutor.shared.domain.base import AggregateRoot

        class Order(AggregateRoot):
            pass

        class OrderCreatedEvent:
            pass

        order = Order()
        event = OrderCreatedEvent()
        order.add_domain_event(event)

        assert len(order.domain_events) == 1
        assert order.domain_events[0] == event

    def test_aggregate_root_clear_domain_events(self):
        """Test clearing domain events."""
        from code_tutor.shared.domain.base import AggregateRoot

        class Order(AggregateRoot):
            pass

        order = Order()
        order.add_domain_event("event1")
        order.add_domain_event("event2")

        events = order.clear_domain_events()

        assert len(events) == 2
        assert order.domain_events == []

    def test_aggregate_root_increment_version(self):
        """Test version increment."""
        from code_tutor.shared.domain.base import AggregateRoot

        class Order(AggregateRoot):
            pass

        order = Order()
        assert order.version == 0

        order.increment_version()
        assert order.version == 1

        order.increment_version()
        assert order.version == 2

    def test_aggregate_root_domain_events_returns_copy(self):
        """Test domain_events property returns a copy."""
        from code_tutor.shared.domain.base import AggregateRoot

        class Order(AggregateRoot):
            pass

        order = Order()
        order.add_domain_event("event1")

        events = order.domain_events
        events.append("should not affect original")

        assert len(order.domain_events) == 1


# ============== Exception Handlers Tests ==============


class TestExceptionHandlers:
    """Tests for exception handlers."""

    @pytest.fixture
    def test_app(self):
        """Create a test FastAPI app with exception handlers."""
        from fastapi import FastAPI
        from code_tutor.shared.exception_handlers import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test-not-found")
        async def raise_not_found():
            raise NotFoundError("User", "123")

        @app.get("/test-validation")
        async def raise_validation():
            raise ValidationError("Invalid email")

        @app.get("/test-auth")
        async def raise_auth():
            raise AuthenticationError("Bad credentials")

        @app.get("/test-authz")
        async def raise_authz():
            raise AuthorizationError("Access denied")

        @app.get("/test-conflict")
        async def raise_conflict():
            raise ConflictError("Already exists")

        @app.get("/test-app-exception")
        async def raise_app_exception():
            raise AppException("Generic error", code="CUSTOM_ERROR")

        @app.get("/test-generic")
        async def raise_generic():
            raise RuntimeError("Unexpected error")

        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        from fastapi.testclient import TestClient
        return TestClient(test_app, raise_server_exceptions=False)

    def test_not_found_handler(self, client):
        """Test NotFoundError handler."""
        response = client.get("/test-not-found")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "USER_NOT_FOUND" in data["error"]["code"]

    def test_validation_error_handler(self, client):
        """Test ValidationError handler."""
        response = client.get("/test-validation")
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_authentication_error_handler(self, client):
        """Test AuthenticationError handler."""
        response = client.get("/test-auth")
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False

    def test_authorization_error_handler(self, client):
        """Test AuthorizationError handler."""
        response = client.get("/test-authz")
        assert response.status_code == 403
        data = response.json()
        assert data["success"] is False

    def test_conflict_error_handler(self, client):
        """Test ConflictError handler."""
        response = client.get("/test-conflict")
        assert response.status_code == 409
        data = response.json()
        assert data["success"] is False

    def test_app_exception_handler(self, client):
        """Test generic AppException handler."""
        response = client.get("/test-app-exception")
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False

    def test_generic_exception_handler(self, client):
        """Test generic exception handler."""
        response = client.get("/test-generic")
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "INTERNAL_ERROR"

    def test_http_exception_handler(self, test_app, client):
        """Test HTTP exception handler."""
        from fastapi import HTTPException

        @test_app.get("/test-http-exception")
        async def raise_http():
            raise HTTPException(status_code=429, detail="Too many requests")

        response = client.get("/test-http-exception")
        assert response.status_code == 429
        data = response.json()
        assert data["success"] is False

    def test_validation_exception_handler(self, test_app, client):
        """Test RequestValidationError handler."""
        from pydantic import BaseModel

        class TestInput(BaseModel):
            email: str
            age: int

        @test_app.post("/test-pydantic-validation")
        async def validate_input(data: TestInput):
            return {"ok": True}

        response = client.post("/test-pydantic-validation", json={"email": 123})
        assert response.status_code == 422 or response.status_code == 400


# ============== Domain Events Tests ==============


class TestDomainEvent:
    """Tests for DomainEvent base class."""

    def test_domain_event_event_type(self):
        """Test event_type property returns class name."""
        from dataclasses import dataclass
        from code_tutor.shared.domain.events import DomainEvent

        @dataclass(frozen=True)
        class UserCreatedEvent(DomainEvent):
            user_id: str = ""

        event = UserCreatedEvent(user_id="user123")
        assert event.event_type == "UserCreatedEvent"

    def test_domain_event_has_id_and_timestamp(self):
        """Test DomainEvent has event_id and occurred_at."""
        from dataclasses import dataclass
        from code_tutor.shared.domain.events import DomainEvent

        @dataclass(frozen=True)
        class TestEvent(DomainEvent):
            pass

        event = TestEvent()
        assert event.event_id is not None
        assert event.occurred_at is not None


class TestIntegrationEvent:
    """Tests for IntegrationEvent base class."""

    def test_integration_event_event_type(self):
        """Test event_type property returns class name."""
        from dataclasses import dataclass
        from code_tutor.shared.domain.events import IntegrationEvent

        @dataclass(frozen=True)
        class UserSyncEvent(IntegrationEvent):
            user_id: str = ""

        event = UserSyncEvent(user_id="user123", source_context="auth")
        assert event.event_type == "UserSyncEvent"

    def test_integration_event_has_source_context(self):
        """Test IntegrationEvent has source_context."""
        from dataclasses import dataclass
        from code_tutor.shared.domain.events import IntegrationEvent

        @dataclass(frozen=True)
        class TestEvent(IntegrationEvent):
            pass

        event = TestEvent(source_context="payment")
        assert event.source_context == "payment"
        assert event.event_id is not None


# ============== Config Production Validation Tests ==============


class TestSettingsProductionValidation:
    """Tests for Settings production validation."""

    def test_production_requires_secure_jwt_secret(self):
        """Test production env rejects default JWT secret."""
        from code_tutor.shared.config import Settings, ConfigurationError
        import os

        with pytest.raises(ConfigurationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="change-this-secret-key-in-production",
                DATABASE_URL="postgresql+asyncpg://prod:pass@prod-db:5432/app",
            )

        assert "JWT_SECRET_KEY" in str(exc_info.value)

    def test_production_requires_long_jwt_secret(self):
        """Test production env rejects short JWT secret."""
        from code_tutor.shared.config import Settings, ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="short",  # Less than 32 chars
                DATABASE_URL="postgresql+asyncpg://prod:pass@prod-db:5432/app",
            )

        assert "32 characters" in str(exc_info.value)

    def test_production_rejects_wildcard_cors(self):
        """Test production env rejects wildcard CORS."""
        from code_tutor.shared.config import Settings, ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="a" * 64,  # Long enough
                CORS_ORIGINS=["*"],
                DATABASE_URL="postgresql+asyncpg://prod:pass@prod-db:5432/app",
            )

        assert "Wildcard" in str(exc_info.value)

    def test_production_rejects_localhost_database(self):
        """Test production env rejects localhost database."""
        from code_tutor.shared.config import Settings, ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            Settings(
                ENVIRONMENT="production",
                JWT_SECRET_KEY="a" * 64,
                CORS_ORIGINS=["https://myapp.com"],
                DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/db",
            )

        assert "localhost" in str(exc_info.value)

    def test_development_allows_defaults(self):
        """Test development env allows default values."""
        from code_tutor.shared.config import Settings

        # Should not raise any errors
        settings = Settings(ENVIRONMENT="development")
        assert settings.ENVIRONMENT == "development"


# ============== Container Tests ==============


class TestContainer:
    """Tests for dependency injection container."""

    def test_container_exists(self):
        """Test Container class can be imported."""
        from code_tutor.shared.container import Container

        assert Container is not None
        assert hasattr(Container, "config")
        assert hasattr(Container, "db_session_factory")
        assert hasattr(Container, "redis_client")

    def test_container_config_provider(self):
        """Test config provider returns Settings."""
        from code_tutor.shared.container import Container
        from code_tutor.shared.config import Settings

        container = Container()
        config = container.config()

        assert isinstance(config, Settings)


# ============== Redis Client Tests ==============


class TestRedisClientWithoutConnection:
    """Tests for RedisClient when Redis is not available."""

    def test_redis_client_not_available(self):
        """Test RedisClient handles missing connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        assert client.is_available is False

    @pytest.mark.asyncio
    async def test_redis_get_returns_none_when_unavailable(self):
        """Test get returns None when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.get("any_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_set_returns_true_when_unavailable(self):
        """Test set returns True when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.set("key", "value")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_delete_returns_zero_when_unavailable(self):
        """Test delete returns 0 when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.delete("key")
        assert result == 0

    @pytest.mark.asyncio
    async def test_redis_exists_returns_false_when_unavailable(self):
        """Test exists returns False when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.exists("key")
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_incr_returns_zero_when_unavailable(self):
        """Test incr returns 0 when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.incr("counter")
        assert result == 0

    @pytest.mark.asyncio
    async def test_redis_expire_returns_true_when_unavailable(self):
        """Test expire returns True when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.expire("key", 60)
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_ttl_returns_negative_when_unavailable(self):
        """Test ttl returns -1 when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.ttl("key")
        assert result == -1

    @pytest.mark.asyncio
    async def test_redis_get_json_returns_none_when_unavailable(self):
        """Test get_json returns None when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.get_json("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_set_json_returns_true_when_unavailable(self):
        """Test set_json returns True when Redis unavailable."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(None)
        result = await client.set_json("key", {"data": "value"})
        assert result is True


class TestRedisClientWithMock:
    """Tests for RedisClient with mocked Redis connection."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        from unittest.mock import AsyncMock, MagicMock

        mock = MagicMock()
        mock.get = AsyncMock(return_value="test_value")
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=1)
        mock.incr = AsyncMock(return_value=5)
        mock.expire = AsyncMock(return_value=True)
        mock.ttl = AsyncMock(return_value=3600)
        return mock

    def test_redis_client_is_available(self, mock_redis):
        """Test is_available returns True with connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        assert client.is_available is True

    @pytest.mark.asyncio
    async def test_redis_get_with_connection(self, mock_redis):
        """Test get with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.get("my_key")

        assert result == "test_value"
        mock_redis.get.assert_called_once_with("my_key")

    @pytest.mark.asyncio
    async def test_redis_set_with_connection(self, mock_redis):
        """Test set with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.set("key", "value", expire_seconds=300)

        assert result is True
        mock_redis.set.assert_called_once_with("key", "value", ex=300)

    @pytest.mark.asyncio
    async def test_redis_delete_with_connection(self, mock_redis):
        """Test delete with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.delete("key")

        assert result == 1
        mock_redis.delete.assert_called_once_with("key")

    @pytest.mark.asyncio
    async def test_redis_exists_with_connection(self, mock_redis):
        """Test exists with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.exists("key")

        assert result is True

    @pytest.mark.asyncio
    async def test_redis_incr_with_connection(self, mock_redis):
        """Test incr with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.incr("counter")

        assert result == 5
        mock_redis.incr.assert_called_once_with("counter")

    @pytest.mark.asyncio
    async def test_redis_expire_with_connection(self, mock_redis):
        """Test expire with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.expire("key", 60)

        assert result is True
        mock_redis.expire.assert_called_once_with("key", 60)

    @pytest.mark.asyncio
    async def test_redis_ttl_with_connection(self, mock_redis):
        """Test ttl with active connection."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.ttl("key")

        assert result == 3600
        mock_redis.ttl.assert_called_once_with("key")

    @pytest.mark.asyncio
    async def test_redis_get_json_with_connection(self, mock_redis):
        """Test get_json parses JSON correctly."""
        from unittest.mock import AsyncMock
        from code_tutor.shared.infrastructure.redis import RedisClient

        mock_redis.get = AsyncMock(return_value='{"name": "test", "count": 42}')
        client = RedisClient(mock_redis)
        result = await client.get_json("json_key")

        assert result == {"name": "test", "count": 42}

    @pytest.mark.asyncio
    async def test_redis_set_json_with_connection(self, mock_redis):
        """Test set_json serializes JSON correctly."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.set_json("key", {"data": "value"}, expire_seconds=60)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert '"data": "value"' in call_args[0][1] or '"data":"value"' in call_args[0][1]


class TestRedisTokenHelpers:
    """Tests for Redis token management helpers."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        from unittest.mock import AsyncMock, MagicMock

        mock = MagicMock()
        mock.get = AsyncMock(return_value="refresh_token_value")
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=1)
        return mock

    @pytest.mark.asyncio
    async def test_store_refresh_token(self, mock_redis):
        """Test storing refresh token."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.store_refresh_token("user123", "token_abc", expire_days=7)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "refresh_token:user123" in call_args[0]
        assert call_args[1]["ex"] == 7 * 86400

    @pytest.mark.asyncio
    async def test_get_refresh_token(self, mock_redis):
        """Test getting refresh token."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.get_refresh_token("user123")

        assert result == "refresh_token_value"

    @pytest.mark.asyncio
    async def test_invalidate_refresh_token(self, mock_redis):
        """Test invalidating refresh token."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.invalidate_refresh_token("user123")

        assert result == 1

    @pytest.mark.asyncio
    async def test_blacklist_token(self, mock_redis):
        """Test blacklisting token."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.blacklist_token("jti123", expire_seconds=3600)

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_token_blacklisted(self, mock_redis):
        """Test checking if token is blacklisted."""
        from code_tutor.shared.infrastructure.redis import RedisClient

        client = RedisClient(mock_redis)
        result = await client.is_token_blacklisted("jti123")

        assert result is True


# ============== UnitOfWork Tests ==============


class TestUnitOfWorkAbstract:
    """Tests for UnitOfWork abstract class."""

    def test_unit_of_work_is_abstract(self):
        """Test UnitOfWork cannot be instantiated."""
        from code_tutor.shared.infrastructure.uow import UnitOfWork

        with pytest.raises(TypeError):
            UnitOfWork()


class TestSQLAlchemyUnitOfWork:
    """Tests for SQLAlchemyUnitOfWork."""

    def test_session_property_raises_when_not_started(self):
        """Test session property raises when UoW not started."""
        from code_tutor.shared.infrastructure.uow import SQLAlchemyUnitOfWork

        uow = SQLAlchemyUnitOfWork()
        with pytest.raises(RuntimeError) as exc_info:
            _ = uow.session

        assert "not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_commit_does_nothing_when_no_session(self):
        """Test commit is safe when session is None."""
        from code_tutor.shared.infrastructure.uow import SQLAlchemyUnitOfWork

        uow = SQLAlchemyUnitOfWork()
        # Should not raise
        await uow.commit()

    @pytest.mark.asyncio
    async def test_rollback_does_nothing_when_no_session(self):
        """Test rollback is safe when session is None."""
        from code_tutor.shared.infrastructure.uow import SQLAlchemyUnitOfWork

        uow = SQLAlchemyUnitOfWork()
        # Should not raise
        await uow.rollback()
