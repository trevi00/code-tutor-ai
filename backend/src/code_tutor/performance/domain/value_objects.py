"""Performance analysis value objects."""

from enum import Enum


class ComplexityClass(str, Enum):
    """Big-O complexity classes."""

    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    UNKNOWN = "Unknown"


class AnalysisStatus(str, Enum):
    """Status of analysis."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class PerformanceIssueType(str, Enum):
    """Types of performance issues."""

    NESTED_LOOP = "nested_loop"
    INEFFICIENT_ALGORITHM = "inefficient_algorithm"
    MEMORY_LEAK = "memory_leak"
    EXCESSIVE_RECURSION = "excessive_recursion"
    UNNECESSARY_COMPUTATION = "unnecessary_computation"
    LARGE_DATA_STRUCTURE = "large_data_structure"
    STRING_CONCATENATION = "string_concatenation"
    GLOBAL_VARIABLE = "global_variable"


class IssueSeverity(str, Enum):
    """Severity levels for performance issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Complexity ranking for comparison
COMPLEXITY_RANK = {
    ComplexityClass.CONSTANT: 1,
    ComplexityClass.LOGARITHMIC: 2,
    ComplexityClass.LINEAR: 3,
    ComplexityClass.LINEARITHMIC: 4,
    ComplexityClass.QUADRATIC: 5,
    ComplexityClass.CUBIC: 6,
    ComplexityClass.EXPONENTIAL: 7,
    ComplexityClass.FACTORIAL: 8,
    ComplexityClass.UNKNOWN: 9,
}


# Descriptions for complexity classes
COMPLEXITY_DESCRIPTIONS = {
    ComplexityClass.CONSTANT: "입력 크기에 관계없이 일정한 시간",
    ComplexityClass.LOGARITHMIC: "입력이 커져도 천천히 증가 (이진 탐색)",
    ComplexityClass.LINEAR: "입력에 비례하여 증가",
    ComplexityClass.LINEARITHMIC: "효율적인 정렬 알고리즘 (병합 정렬, 퀵 정렬)",
    ComplexityClass.QUADRATIC: "중첩 루프, 입력의 제곱에 비례",
    ComplexityClass.CUBIC: "삼중 루프, 입력의 세제곱에 비례",
    ComplexityClass.EXPONENTIAL: "매우 빠르게 증가, 큰 입력에 비실용적",
    ComplexityClass.FACTORIAL: "극도로 빠르게 증가, 작은 입력만 가능",
    ComplexityClass.UNKNOWN: "복잡도를 분석할 수 없음",
}


def compare_complexity(a: ComplexityClass, b: ComplexityClass) -> int:
    """Compare two complexity classes.

    Returns:
        -1 if a < b (a is more efficient)
        0 if a == b
        1 if a > b (a is less efficient)
    """
    rank_a = COMPLEXITY_RANK.get(a, 9)
    rank_b = COMPLEXITY_RANK.get(b, 9)
    if rank_a < rank_b:
        return -1
    elif rank_a > rank_b:
        return 1
    return 0
