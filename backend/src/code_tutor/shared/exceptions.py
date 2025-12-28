"""Application-wide exceptions"""

from typing import Any


class AppException(Exception):
    """Base application exception"""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ValidationError(AppException):
    """Validation error"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "VALIDATION_ERROR", details)


class NotFoundError(AppException):
    """Resource not found error"""

    def __init__(
        self,
        resource: str,
        identifier: str | None = None,
    ) -> None:
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"
        super().__init__(message, "NOT_FOUND", {"resource": resource, "id": identifier})
        self.entity = resource


class ConflictError(AppException):
    """Conflict error (e.g., duplicate resource)"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "CONFLICT", details)


class UnauthorizedError(AppException):
    """Authentication required error"""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message, "UNAUTHORIZED")


class AuthenticationError(AppException):
    """Authentication error"""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, "AUTHENTICATION_ERROR")


class AuthorizationError(AppException):
    """Authorization error"""

    def __init__(self, message: str = "Access denied") -> None:
        super().__init__(message, "AUTHORIZATION_ERROR")


class ForbiddenError(AppException):
    """Permission denied error"""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(message, "FORBIDDEN")


class DomainError(AppException):
    """Domain logic error"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "DOMAIN_ERROR", details)


class ExternalServiceError(AppException):
    """External service error"""

    def __init__(
        self,
        service: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"External service error ({service}): {message}",
            "EXTERNAL_SERVICE_ERROR",
            {"service": service, **(details or {})},
        )


class SandboxError(AppException):
    """Code execution sandbox error"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "SANDBOX_ERROR", details)


class LLMError(AppException):
    """LLM service error"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "LLM_ERROR", details)
