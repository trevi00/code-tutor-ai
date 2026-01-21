"""Rate limiting middleware"""

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

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

    # Cleanup settings
    CLEANUP_INTERVAL = 100  # Run cleanup every N requests
    ENTRY_TTL = 600  # Remove entries not accessed for 10 minutes

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens: dict[str, float] = defaultdict(lambda: float(burst_size))
        self.last_update: dict[str, float] = defaultdict(time.time)
        self.token_rate = requests_per_minute / 60.0  # tokens per second
        self._request_count = 0

    def _cleanup_stale_entries(self) -> None:
        """Remove entries that haven't been accessed recently"""
        current_time = time.time()
        cutoff = current_time - self.ENTRY_TTL

        # Find stale keys
        stale_keys = [
            key for key, last_time in self.last_update.items() if last_time < cutoff
        ]

        # Remove stale entries
        for key in stale_keys:
            self.tokens.pop(key, None)
            self.last_update.pop(key, None)

        if stale_keys:
            logger.debug(
                "Rate limiter cleanup completed",
                removed_entries=len(stale_keys),
                remaining_entries=len(self.tokens),
            )

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.
        Returns (allowed, retry_after_seconds)
        """
        # Periodic cleanup
        self._request_count += 1
        if self._request_count >= self.CLEANUP_INTERVAL:
            self._cleanup_stale_entries()
            self._request_count = 0

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

    def get_stats(self) -> dict:
        """Get limiter statistics for monitoring"""
        return {
            "active_entries": len(self.tokens),
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
        }


# Global rate limiter instances for different endpoints
_rate_limiters: dict[str, InMemoryRateLimiter] = {}


def get_rate_limiter(
    name: str, requests_per_minute: int = 60, burst_size: int = 10
) -> InMemoryRateLimiter:
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
        self.exclude_paths = exclude_paths or [
            "/api/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

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
            # Use SHA256 hash of the token for consistent identification
            token = auth_header[7:]
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            return f"user:{token_hash}"

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
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        return f"user:{token_hash}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


# Authentication-specific rate limiters (stricter limits for security)
_auth_login_limiter = InMemoryRateLimiter(
    requests_per_minute=5,  # 5 login attempts per minute
    burst_size=3,  # Max 3 rapid attempts
)
_auth_register_limiter = InMemoryRateLimiter(
    requests_per_minute=3,  # 3 registrations per minute
    burst_size=2,  # Max 2 rapid attempts
)
_auth_refresh_limiter = InMemoryRateLimiter(
    requests_per_minute=10,  # 10 token refreshes per minute
    burst_size=5,
)


def check_auth_rate_limit(limiter: InMemoryRateLimiter, request: Request) -> None:
    """Check rate limit and raise exception if exceeded"""
    client_id = _get_client_id_from_request(request)
    allowed, retry_after = limiter.is_allowed(client_id)

    if not allowed:
        logger.warning(
            "Auth rate limit exceeded",
            client_id=client_id,
            path=request.url.path,
            retry_after=retry_after,
        )
        raise RateLimitExceeded(retry_after)


def login_rate_limit(request: Request) -> None:
    """Dependency for login rate limiting (5/min)"""
    check_auth_rate_limit(_auth_login_limiter, request)


def register_rate_limit(request: Request) -> None:
    """Dependency for registration rate limiting (3/min)"""
    check_auth_rate_limit(_auth_register_limiter, request)


def refresh_rate_limit(request: Request) -> None:
    """Dependency for token refresh rate limiting (10/min)"""
    check_auth_rate_limit(_auth_refresh_limiter, request)


def reset_auth_rate_limiters() -> None:
    """Reset all auth rate limiters (for testing)"""
    global _auth_login_limiter, _auth_register_limiter, _auth_refresh_limiter
    _auth_login_limiter = InMemoryRateLimiter(requests_per_minute=5, burst_size=3)
    _auth_register_limiter = InMemoryRateLimiter(requests_per_minute=3, burst_size=2)
    _auth_refresh_limiter = InMemoryRateLimiter(requests_per_minute=10, burst_size=5)


# =============================================================================
# Code Execution Rate Limiters (expensive operations)
# =============================================================================
_code_execution_limiter = InMemoryRateLimiter(
    requests_per_minute=20,  # 20 executions per minute
    burst_size=5,  # Max 5 rapid executions
)
_playground_execution_limiter = InMemoryRateLimiter(
    requests_per_minute=30,  # 30 playground runs per minute
    burst_size=10,  # Max 10 rapid runs
)


def code_execution_rate_limit(request: Request) -> None:
    """Dependency for code execution rate limiting (20/min)"""
    check_auth_rate_limit(_code_execution_limiter, request)


def playground_rate_limit(request: Request) -> None:
    """Dependency for playground execution rate limiting (30/min)"""
    check_auth_rate_limit(_playground_execution_limiter, request)


# =============================================================================
# AI Tutor Rate Limiters (API cost control)
# =============================================================================
_ai_chat_limiter = InMemoryRateLimiter(
    requests_per_minute=10,  # 10 AI messages per minute
    burst_size=3,  # Max 3 rapid messages
)
_ai_hint_limiter = InMemoryRateLimiter(
    requests_per_minute=20,  # 20 hints per minute
    burst_size=5,  # Max 5 rapid hints
)
_ai_review_limiter = InMemoryRateLimiter(
    requests_per_minute=5,  # 5 code reviews per minute (expensive)
    burst_size=2,  # Max 2 rapid reviews
)


def ai_chat_rate_limit(request: Request) -> None:
    """Dependency for AI chat rate limiting (10/min)"""
    check_auth_rate_limit(_ai_chat_limiter, request)


def ai_hint_rate_limit(request: Request) -> None:
    """Dependency for AI hint rate limiting (20/min)"""
    check_auth_rate_limit(_ai_hint_limiter, request)


def ai_review_rate_limit(request: Request) -> None:
    """Dependency for AI code review rate limiting (5/min)"""
    check_auth_rate_limit(_ai_review_limiter, request)


# =============================================================================
# Reset functions for testing
# =============================================================================
def reset_execution_rate_limiters() -> None:
    """Reset execution rate limiters (for testing)"""
    global _code_execution_limiter, _playground_execution_limiter
    _code_execution_limiter = InMemoryRateLimiter(requests_per_minute=20, burst_size=5)
    _playground_execution_limiter = InMemoryRateLimiter(
        requests_per_minute=30, burst_size=10
    )


def reset_ai_rate_limiters() -> None:
    """Reset AI rate limiters (for testing)"""
    global _ai_chat_limiter, _ai_hint_limiter, _ai_review_limiter
    _ai_chat_limiter = InMemoryRateLimiter(requests_per_minute=10, burst_size=3)
    _ai_hint_limiter = InMemoryRateLimiter(requests_per_minute=20, burst_size=5)
    _ai_review_limiter = InMemoryRateLimiter(requests_per_minute=5, burst_size=2)
