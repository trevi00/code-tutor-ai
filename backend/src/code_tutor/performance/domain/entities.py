"""Performance analysis entities."""

from dataclasses import dataclass, field
from typing import Optional

from .value_objects import (
    AnalysisStatus,
    ComplexityClass,
    IssueSeverity,
    PerformanceIssueType,
)


@dataclass
class LoopInfo:
    """Information about a loop in the code."""

    line_number: int
    loop_type: str  # "for", "while"
    nesting_level: int
    iteration_variable: Optional[str] = None
    iterable: Optional[str] = None
    estimated_iterations: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    line_number: int
    parameters: list[str] = field(default_factory=list)
    is_recursive: bool = False
    calls_count: int = 0
    complexity: ComplexityClass = ComplexityClass.UNKNOWN


@dataclass
class PerformanceIssue:
    """A detected performance issue."""

    issue_type: PerformanceIssueType
    severity: IssueSeverity
    line_number: int
    message: str
    suggestion: str
    code_snippet: Optional[str] = None


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    peak_memory_mb: float
    average_memory_mb: float
    allocations_count: int
    deallocations_count: int
    largest_object_mb: float
    largest_object_type: Optional[str] = None


@dataclass
class RuntimeMetrics:
    """Runtime performance metrics."""

    execution_time_ms: float
    cpu_time_ms: float
    function_calls: int
    line_executions: int
    peak_call_depth: int


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis."""

    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    time_explanation: str
    space_explanation: str
    loops: list[LoopInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    max_nesting_depth: int = 0
    recursive_functions: list[str] = field(default_factory=list)


@dataclass
class PerformanceProfile:
    """Complete performance profile of code."""

    status: AnalysisStatus
    complexity: Optional[ComplexityAnalysis] = None
    runtime: Optional[RuntimeMetrics] = None
    memory: Optional[MemoryMetrics] = None
    issues: list[PerformanceIssue] = field(default_factory=list)
    optimization_score: int = 100  # 0-100
    error: Optional[str] = None


@dataclass
class FunctionProfile:
    """Profile for a single function."""

    name: str
    calls: int
    total_time_ms: float
    own_time_ms: float  # excluding subcalls
    avg_time_ms: float
    percentage: float  # of total execution time


@dataclass
class LineProfile:
    """Profile for a single line."""

    line_number: int
    code: str
    hits: int
    time_ms: float
    time_per_hit_ms: float
    percentage: float


@dataclass
class HotspotAnalysis:
    """Analysis of performance hotspots."""

    hotspot_functions: list[FunctionProfile] = field(default_factory=list)
    hotspot_lines: list[LineProfile] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    bottleneck_function: Optional[str] = None
    bottleneck_line: Optional[int] = None
