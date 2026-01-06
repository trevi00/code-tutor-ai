"""Shared constants for the application."""


# =============================================================================
# Pagination
# =============================================================================
class Pagination:
    """Pagination-related constants."""

    DEFAULT_PAGE: int = 1
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # Leaderboard
    LEADERBOARD_DEFAULT_LIMIT: int = 10
    LEADERBOARD_MAX_LIMIT: int = 100

    # Gamification leaderboard
    GAMIFICATION_LEADERBOARD_DEFAULT: int = 100
    GAMIFICATION_LEADERBOARD_MAX: int = 500

    # Recommendations
    RECOMMENDATION_DEFAULT_LIMIT: int = 5
    RECOMMENDATION_MAX_LIMIT: int = 20

    # Statistics
    STATS_DEFAULT_TOP_K: int = 3
    STATS_MAX_TOP_K: int = 10
    STATS_DEFAULT_DAYS: int = 30
    STATS_MIN_DAYS: int = 7
    STATS_MAX_DAYS: int = 90


# =============================================================================
# Typing Practice
# =============================================================================
class TypingPractice:
    """Typing practice-related constants."""

    # Mastery
    MASTERY_THRESHOLD: int = 5  # Number of completions required for mastery
    MIN_REQUIRED_COMPLETIONS: int = 1
    MAX_REQUIRED_COMPLETIONS: int = 10

    # Accuracy
    HIGH_ACCURACY_THRESHOLD: float = 95.0  # Percentage for bonus XP


# =============================================================================
# Code Execution
# =============================================================================
class Execution:
    """Code execution-related constants."""

    DEFAULT_TIMEOUT_SECONDS: int = 5
    MIN_TIMEOUT_SECONDS: int = 1
    MAX_TIMEOUT_SECONDS: int = 30

    PLAYGROUND_DEFAULT_TIMEOUT: int = 10
    DEBUGGER_MAX_EXECUTION_TIME: int = 10

    # Problem constraints
    DEFAULT_TIME_LIMIT_MS: int = 1000
    MIN_TIME_LIMIT_MS: int = 100
    MAX_TIME_LIMIT_MS: int = 10000

    DEFAULT_MEMORY_LIMIT_MB: int = 256
    MIN_MEMORY_LIMIT_MB: int = 32
    MAX_MEMORY_LIMIT_MB: int = 512


# =============================================================================
# Collaboration
# =============================================================================
class Collaboration:
    """Collaboration-related constants."""

    DEFAULT_MAX_PARTICIPANTS: int = 5
    MIN_PARTICIPANTS: int = 2
    MAX_PARTICIPANTS: int = 10


# =============================================================================
# Quality Analysis
# =============================================================================
class QualityAnalysis:
    """Code quality analysis constants."""

    WEAK_DIMENSION_THRESHOLD: int = 60
    STRONG_DIMENSION_THRESHOLD: int = 80

    # Complexity expectations by difficulty
    COMPLEXITY_BY_DIFFICULTY = {
        "easy": {"max_cyclomatic": 5, "max_cognitive": 8},
        "medium": {"max_cyclomatic": 10, "max_cognitive": 15},
        "hard": {"max_cyclomatic": 15, "max_cognitive": 25},
    }


# =============================================================================
# Cache TTL (seconds)
# =============================================================================
class CacheTTL:
    """Cache Time-To-Live constants in seconds."""

    RECOMMENDATIONS: int = 3600  # 1 hour
    PREDICTIONS: int = 21600  # 6 hours
    USER_STATS: int = 1800  # 30 minutes
    INTERACTION_MATRIX: int = 86400  # 24 hours


# =============================================================================
# Rate Limiting
# =============================================================================
class RateLimiting:
    """Rate limiting constants."""

    DEFAULT_REQUESTS_PER_MINUTE: int = 60
    DEFAULT_BURST_SIZE: int = 20
