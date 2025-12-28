"""Rate limiting middleware"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )


class InMemoryRateLimiter:
    """Simple in-memory rate limiter using token bucket algorithm"""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens: dict[str, float] = defaultdict(lambda: float(burst_size))
        self.last_update: dict[str, float] = defaultdict(time.time)
        self.token_rate = requests_per_minute / 60.0  # tokens per second

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        Returns (allowed, retry_after_seconds)
        """
        current_time = time.time()
        time_passed = current_time - self.last_update[key]
        self.last_update[key] = current_time

        # Add tokens based on time passed
        self.tokens[key] = min(
            self.burst_size,
            self.tokens[key] + time_passed * self.token_rate,
        )

        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True, 0

        # Calculate retry after
        tokens_needed = 1 - self.tokens[key]
        retry_after = int(tokens_needed / self.token_rate) + 1
        return False, retry_after

    def get_remaining(self, key: str) -> int:
        """Get remaining requests"""
        return max(0, int(self.tokens.get(key, self.burst_size)))


# Global rate limiter instances for different endpoints
_rate_limiters: dict[str, InMemoryRateLimiter] = {}


def get_rate_limiter(name: str, requests_per_minute: int = 60, burst_size: int = 10) -> InMemoryRateLimiter:
    """Get or create a rate limiter"""
    if name not in _rate_limiters:
        _rate_limiters[name] = InMemoryRateLimiter(requests_per_minute, burst_size)
    return _rate_limiters[name]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(requests_per_minute, burst_size)
        self.exclude_paths = exclude_paths or ["/api/health", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get client identifier (IP or user ID)
        client_id = self._get_client_id(request)

        # Check rate limit
        allowed, retry_after = self.limiter.is_allowed(client_id)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request.url.path,
                retry_after=retry_after,
            )
            raise RateLimitExceeded(retry_after)

        # Add rate limit headers
        response = await call_next(request)
        remaining = self.limiter.get_remaining(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from auth header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Use a hash of the token for privacy
            token = auth_header[7:]
            return f"user:{hash(token) % 1000000}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"


# Endpoint-specific rate limiters
def rate_limit(
    requests_per_minute: int = 60,
    burst_size: int = 10,
):
    """
    Decorator for endpoint-specific rate limiting.

    Usage:
        @router.post("/submit")
        @rate_limit(requests_per_minute=10)
        async def submit(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        limiter = InMemoryRateLimiter(requests_per_minute, burst_size)
        func._rate_limiter = limiter

        async def wrapper(request: Request, *args, **kwargs):
            client_id = _get_client_id_from_request(request)
            allowed, retry_after = limiter.is_allowed(client_id)

            if not allowed:
                raise RateLimitExceeded(retry_after)

            return await func(request, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def _get_client_id_from_request(request: Request) -> str:
    """Get client identifier for rate limiting"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return f"user:{hash(token) % 1000000}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"
