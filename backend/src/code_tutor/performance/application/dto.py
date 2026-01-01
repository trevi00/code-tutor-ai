"""Performance analysis DTOs."""

from typing import Optional

from pydantic import BaseModel, Field

from ..domain import (
    AnalysisStatus,
    ComplexityClass,
    IssueSeverity,
    PerformanceIssueType,
)


# Request DTOs
class AnalyzeRequest(BaseModel):
    """Request for code analysis."""

    code: str = Field(..., min_length=1, description="Python code to analyze")
    input_data: Optional[str] = Field(None, description="Optional input data")
    include_runtime: bool = Field(True, description="Include runtime profiling")
    include_memory: bool = Field(True, description="Include memory profiling")


class QuickAnalyzeRequest(BaseModel):
    """Request for quick complexity-only analysis."""

    code: str = Field(..., min_length=1, description="Python code to analyze")


# Response DTOs
class LoopInfoResponse(BaseModel):
    """Loop information response."""

    line_number: int
    loop_type: str
    nesting_level: int
    iteration_variable: Optional[str] = None
    iterable: Optional[str] = None
    estimated_iterations: Optional[str] = None


class FunctionInfoResponse(BaseModel):
    """Function information response."""

    name: str
    line_number: int
    parameters: list[str]
    is_recursive: bool
    calls_count: int
    complexity: ComplexityClass


class PerformanceIssueResponse(BaseModel):
    """Performance issue response."""

    issue_type: PerformanceIssueType
    severity: IssueSeverity
    line_number: int
    message: str
    suggestion: str
    code_snippet: Optional[str] = None


class ComplexityResponse(BaseModel):
    """Complexity analysis response."""

    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    time_explanation: str
    space_explanation: str
    max_nesting_depth: int
    loops: list[LoopInfoResponse]
    functions: list[FunctionInfoResponse]
    recursive_functions: list[str]


class RuntimeMetricsResponse(BaseModel):
    """Runtime metrics response."""

    execution_time_ms: float
    cpu_time_ms: float
    function_calls: int
    line_executions: int
    peak_call_depth: int


class MemoryMetricsResponse(BaseModel):
    """Memory metrics response."""

    peak_memory_mb: float
    average_memory_mb: float
    allocations_count: int
    deallocations_count: int
    largest_object_mb: float
    largest_object_type: Optional[str] = None


class FunctionProfileResponse(BaseModel):
    """Function profile response."""

    name: str
    calls: int
    total_time_ms: float
    own_time_ms: float
    avg_time_ms: float
    percentage: float


class HotspotResponse(BaseModel):
    """Hotspot analysis response."""

    hotspot_functions: list[FunctionProfileResponse]
    total_execution_time_ms: float
    bottleneck_function: Optional[str] = None
    bottleneck_line: Optional[int] = None


class PerformanceResponse(BaseModel):
    """Complete performance analysis response."""

    status: AnalysisStatus
    complexity: Optional[ComplexityResponse] = None
    runtime: Optional[RuntimeMetricsResponse] = None
    memory: Optional[MemoryMetricsResponse] = None
    hotspots: Optional[HotspotResponse] = None
    issues: list[PerformanceIssueResponse] = []
    optimization_score: int = 100
    error: Optional[str] = None


class QuickAnalyzeResponse(BaseModel):
    """Quick analysis response (complexity only)."""

    status: AnalysisStatus
    time_complexity: ComplexityClass
    space_complexity: ComplexityClass
    time_explanation: str
    space_explanation: str
    max_nesting_depth: int
    issues_count: int
    error: Optional[str] = None
