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
