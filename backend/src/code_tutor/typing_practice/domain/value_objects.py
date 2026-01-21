"""Value objects for typing practice domain."""

from enum import Enum


class ExerciseCategory(str, Enum):
    """Category of typing exercise."""

    TEMPLATE = "template"  # Algorithm templates (two-pointers, BFS, etc.)
    METHOD = "method"  # String/list method practice
    ALGORITHM = "algorithm"  # Full algorithm implementations
    PATTERN = "pattern"  # Design patterns


class AttemptStatus(str, Enum):
    """Status of a typing attempt."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class Difficulty(str, Enum):
    """Difficulty level of exercise."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
