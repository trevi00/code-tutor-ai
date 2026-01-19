"""Middleware package"""

from code_tutor.shared.middleware.rate_limiter import (
    RateLimitExceeded,
    RateLimitMiddleware,
    # AI rate limiters
    ai_chat_rate_limit,
    ai_hint_rate_limit,
    ai_review_rate_limit,
    # Execution rate limiters
    code_execution_rate_limit,
    # Auth rate limiters
    login_rate_limit,
    playground_rate_limit,
    rate_limit,
    refresh_rate_limit,
    register_rate_limit,
)

__all__ = [
    "RateLimitExceeded",
    "RateLimitMiddleware",
    "rate_limit",
    # Auth
    "login_rate_limit",
    "register_rate_limit",
    "refresh_rate_limit",
    # Execution
    "code_execution_rate_limit",
    "playground_rate_limit",
    # AI
    "ai_chat_rate_limit",
    "ai_hint_rate_limit",
    "ai_review_rate_limit",
]
