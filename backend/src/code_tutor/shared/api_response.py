"""Standardized API Response format following PRD specification"""

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    """Response metadata"""

    request_id: str = Field(default_factory=lambda: f"req-{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorDetail(BaseModel):
    """Error detail structure"""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""

    success: bool
    data: T | None = None
    error: ErrorDetail | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


class PaginationMeta(BaseModel):
    """Pagination metadata"""

    current_page: int
    total_pages: int
    total_count: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response"""

    success: bool = True
    data: T
    pagination: PaginationMeta
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


def success_response(data: Any, meta: ResponseMeta | None = None) -> dict[str, Any]:
    """Create a success response"""
    return {
        "success": True,
        "data": data,
        "meta": (meta or ResponseMeta()).model_dump(mode="json"),
    }


def error_response(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    meta: ResponseMeta | None = None,
) -> dict[str, Any]:
    """Create an error response"""
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "meta": (meta or ResponseMeta()).model_dump(mode="json"),
    }


def paginated_response(
    items: list[Any],
    page: int,
    limit: int,
    total_count: int,
) -> dict[str, Any]:
    """Create a paginated response"""
    total_pages = (total_count + limit - 1) // limit if limit > 0 else 0

    return {
        "success": True,
        "data": items,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_count": total_count,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "meta": ResponseMeta().model_dump(mode="json"),
    }


# Error codes following PRD specification
class ErrorCodes:
    """Standard error codes"""

    # 4xx Client Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    FORBIDDEN = "FORBIDDEN"
    PROBLEM_NOT_FOUND = "PROBLEM_NOT_FOUND"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    SUBMISSION_NOT_FOUND = "SUBMISSION_NOT_FOUND"
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    EMAIL_ALREADY_EXISTS = "EMAIL_ALREADY_EXISTS"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # 5xx Server Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    SANDBOX_ERROR = "SANDBOX_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
