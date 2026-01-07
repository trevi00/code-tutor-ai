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
