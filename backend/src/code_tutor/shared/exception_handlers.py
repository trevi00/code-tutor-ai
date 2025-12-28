"""Global exception handlers for standardized API responses"""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from code_tutor.shared.api_response import ErrorCodes, error_response
from code_tutor.shared.exceptions import (
    AppException,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers"""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors"""
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        field = ".".join(str(loc) for loc in first_error.get("loc", [])[1:])
        message = first_error.get("msg", "Validation error")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response(
                code=ErrorCodes.VALIDATION_ERROR,
                message=f"입력값 검증 실패: {message}",
                details={
                    "field": field,
                    "errors": [
                        {
                            "field": ".".join(str(loc) for loc in e.get("loc", [])[1:]),
                            "message": e.get("msg"),
                            "type": e.get("type"),
                        }
                        for e in errors
                    ],
                },
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions"""
        code_map = {
            400: ErrorCodes.VALIDATION_ERROR,
            401: ErrorCodes.INVALID_CREDENTIALS,
            403: ErrorCodes.FORBIDDEN,
            404: ErrorCodes.PROBLEM_NOT_FOUND,
            409: ErrorCodes.EMAIL_ALREADY_EXISTS,
            429: ErrorCodes.RATE_LIMIT_EXCEEDED,
            500: ErrorCodes.INTERNAL_ERROR,
            503: ErrorCodes.SERVICE_UNAVAILABLE,
        }

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(
                code=code_map.get(exc.status_code, ErrorCodes.INTERNAL_ERROR),
                message=str(exc.detail),
            ),
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(
        request: Request, exc: NotFoundError
    ) -> JSONResponse:
        """Handle not found errors"""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_response(
                code=f"{exc.entity.upper()}_NOT_FOUND",
                message=exc.message,
            ),
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle validation errors"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response(
                code=ErrorCodes.VALIDATION_ERROR,
                message=exc.message,
            ),
        )

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        """Handle authentication errors"""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error_response(
                code=ErrorCodes.INVALID_CREDENTIALS,
                message=exc.message,
            ),
        )

    @app.exception_handler(AuthorizationError)
    async def authz_error_handler(
        request: Request, exc: AuthorizationError
    ) -> JSONResponse:
        """Handle authorization errors"""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=error_response(
                code=ErrorCodes.FORBIDDEN,
                message=exc.message,
            ),
        )

    @app.exception_handler(ConflictError)
    async def conflict_error_handler(
        request: Request, exc: ConflictError
    ) -> JSONResponse:
        """Handle conflict errors"""
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=error_response(
                code=ErrorCodes.EMAIL_ALREADY_EXISTS,
                message=exc.message,
            ),
        )

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        """Handle generic app exceptions"""
        status_code_map = {
            "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
            "AUTHENTICATION_ERROR": status.HTTP_401_UNAUTHORIZED,
            "AUTHORIZATION_ERROR": status.HTTP_403_FORBIDDEN,
            "NOT_FOUND": status.HTTP_404_NOT_FOUND,
            "CONFLICT": status.HTTP_409_CONFLICT,
        }

        return JSONResponse(
            status_code=status_code_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR),
            content=error_response(
                code=exc.code,
                message=exc.message,
            ),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions"""
        logger.exception(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response(
                code=ErrorCodes.INTERNAL_ERROR,
                message="서버 내부 오류가 발생했습니다.",
            ),
        )
