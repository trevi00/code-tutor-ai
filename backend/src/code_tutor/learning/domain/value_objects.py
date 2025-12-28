"""Learning domain value objects"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

from code_tutor.shared.domain.base import ValueObject
from code_tutor.shared.exceptions import ValidationError


class Difficulty(str, Enum):
    """Problem difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SubmissionStatus(str, Enum):
    """Submission evaluation status"""
    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    WRONG_ANSWER = "wrong_answer"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    RUNTIME_ERROR = "runtime_error"
    COMPILATION_ERROR = "compilation_error"


class Category(str, Enum):
    """Problem categories (algorithm patterns)"""
    ARRAY = "array"
    STRING = "string"
    LINKED_LIST = "linked_list"
    STACK = "stack"
    QUEUE = "queue"
    HASH_TABLE = "hash_table"
    TREE = "tree"
    GRAPH = "graph"
    SORTING = "sorting"
    SEARCHING = "searching"
    DYNAMIC_PROGRAMMING = "dp"
    GREEDY = "greedy"
    BACKTRACKING = "backtracking"
    RECURSION = "recursion"
    TWO_POINTERS = "two_pointers"
    SLIDING_WINDOW = "sliding_window"
    BINARY_SEARCH = "binary_search"
    BFS = "bfs"
    DFS = "dfs"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    DESIGN = "design"


@dataclass(frozen=True)
class ProblemId(ValueObject):
    """Problem ID value object"""

    value: UUID

    @classmethod
    def generate(cls) -> "ProblemId":
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> "ProblemId":
        try:
            return cls(value=UUID(value))
        except ValueError:
            raise ValidationError(f"Invalid problem ID format: {value}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SubmissionId(ValueObject):
    """Submission ID value object"""

    value: UUID

    @classmethod
    def generate(cls) -> "SubmissionId":
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> "SubmissionId":
        try:
            return cls(value=UUID(value))
        except ValueError:
            raise ValidationError(f"Invalid submission ID format: {value}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ExecutionResult(ValueObject):
    """Result of code execution"""

    output: str
    error: str | None
    execution_time_ms: float
    memory_usage_mb: float
    is_correct: bool


@dataclass(frozen=True)
class TestResult(ValueObject):
    """Result of a single test case"""

    test_case_id: UUID
    input_data: str
    expected_output: str
    actual_output: str
    is_passed: bool
    execution_time_ms: float
    error_message: str | None = None
