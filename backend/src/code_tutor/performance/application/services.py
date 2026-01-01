"""Performance analysis service."""

from typing import Optional

from ..domain import (
    AnalysisStatus,
    COMPLEXITY_RANK,
    IssueSeverity,
    PerformanceIssue,
    PerformanceIssueType,
)
from .complexity_analyzer import analyze_complexity
from .dto import (
    AnalyzeRequest,
    ComplexityResponse,
    FunctionInfoResponse,
    FunctionProfileResponse,
    HotspotResponse,
    LoopInfoResponse,
    MemoryMetricsResponse,
    PerformanceIssueResponse,
    PerformanceResponse,
    QuickAnalyzeRequest,
    QuickAnalyzeResponse,
    RuntimeMetricsResponse,
)
from .profiler import profile_code


class PerformanceService:
    """Service for code performance analysis."""

    async def analyze(self, request: AnalyzeRequest) -> PerformanceResponse:
        """Perform full performance analysis."""
        try:
            # Static complexity analysis
            complexity_result, static_issues = analyze_complexity(request.code)

            # Runtime profiling
            runtime_metrics = None
            memory_metrics = None
            hotspots = None
            runtime_issues: list[PerformanceIssue] = []

            if request.include_runtime or request.include_memory:
                runtime, memory, hotspot_result = profile_code(
                    request.code, request.input_data
                )

                if request.include_runtime:
                    runtime_metrics = RuntimeMetricsResponse(
                        execution_time_ms=runtime.execution_time_ms,
                        cpu_time_ms=runtime.cpu_time_ms,
                        function_calls=runtime.function_calls,
                        line_executions=runtime.line_executions,
                        peak_call_depth=runtime.peak_call_depth,
                    )

                    hotspots = HotspotResponse(
                        hotspot_functions=[
                            FunctionProfileResponse(
                                name=f.name,
                                calls=f.calls,
                                total_time_ms=f.total_time_ms,
                                own_time_ms=f.own_time_ms,
                                avg_time_ms=f.avg_time_ms,
                                percentage=f.percentage,
                            )
                            for f in hotspot_result.hotspot_functions
                        ],
                        total_execution_time_ms=hotspot_result.total_execution_time_ms,
                        bottleneck_function=hotspot_result.bottleneck_function,
                        bottleneck_line=hotspot_result.bottleneck_line,
                    )

                    # Check for slow execution
                    if runtime.execution_time_ms > 1000:
                        runtime_issues.append(
                            PerformanceIssue(
                                issue_type=PerformanceIssueType.INEFFICIENT_ALGORITHM,
                                severity=IssueSeverity.WARNING,
                                line_number=1,
                                message=f"실행 시간이 {runtime.execution_time_ms:.0f}ms로 느립니다",
                                suggestion="알고리즘 최적화를 고려하세요",
                            )
                        )

                if request.include_memory:
                    memory_metrics = MemoryMetricsResponse(
                        peak_memory_mb=memory.peak_memory_mb,
                        average_memory_mb=memory.average_memory_mb,
                        allocations_count=memory.allocations_count,
                        deallocations_count=memory.deallocations_count,
                        largest_object_mb=memory.largest_object_mb,
                        largest_object_type=memory.largest_object_type,
                    )

                    # Check for high memory usage
                    if memory.peak_memory_mb > 100:
                        runtime_issues.append(
                            PerformanceIssue(
                                issue_type=PerformanceIssueType.LARGE_DATA_STRUCTURE,
                                severity=IssueSeverity.WARNING,
                                line_number=1,
                                message=f"메모리 사용량이 {memory.peak_memory_mb:.1f}MB로 높습니다",
                                suggestion="제너레이터나 스트리밍 처리를 고려하세요",
                            )
                        )

            # Build complexity response
            complexity_response = ComplexityResponse(
                time_complexity=complexity_result.time_complexity,
                space_complexity=complexity_result.space_complexity,
                time_explanation=complexity_result.time_explanation,
                space_explanation=complexity_result.space_explanation,
                max_nesting_depth=complexity_result.max_nesting_depth,
                loops=[
                    LoopInfoResponse(
                        line_number=l.line_number,
                        loop_type=l.loop_type,
                        nesting_level=l.nesting_level,
                        iteration_variable=l.iteration_variable,
                        iterable=l.iterable,
                        estimated_iterations=l.estimated_iterations,
                    )
                    for l in complexity_result.loops
                ],
                functions=[
                    FunctionInfoResponse(
                        name=f.name,
                        line_number=f.line_number,
                        parameters=f.parameters,
                        is_recursive=f.is_recursive,
                        calls_count=f.calls_count,
                        complexity=f.complexity,
                    )
                    for f in complexity_result.functions
                ],
                recursive_functions=complexity_result.recursive_functions,
            )

            # Combine all issues
            all_issues = static_issues + runtime_issues
            issues_response = [
                PerformanceIssueResponse(
                    issue_type=i.issue_type,
                    severity=i.severity,
                    line_number=i.line_number,
                    message=i.message,
                    suggestion=i.suggestion,
                    code_snippet=i.code_snippet,
                )
                for i in all_issues
            ]

            # Calculate optimization score
            score = self._calculate_score(complexity_result, all_issues)

            return PerformanceResponse(
                status=AnalysisStatus.COMPLETED,
                complexity=complexity_response,
                runtime=runtime_metrics,
                memory=memory_metrics,
                hotspots=hotspots,
                issues=issues_response,
                optimization_score=score,
            )

        except Exception as e:
            return PerformanceResponse(
                status=AnalysisStatus.ERROR,
                error=str(e),
                optimization_score=0,
            )

    async def quick_analyze(self, request: QuickAnalyzeRequest) -> QuickAnalyzeResponse:
        """Perform quick complexity-only analysis."""
        try:
            complexity_result, issues = analyze_complexity(request.code)

            return QuickAnalyzeResponse(
                status=AnalysisStatus.COMPLETED,
                time_complexity=complexity_result.time_complexity,
                space_complexity=complexity_result.space_complexity,
                time_explanation=complexity_result.time_explanation,
                space_explanation=complexity_result.space_explanation,
                max_nesting_depth=complexity_result.max_nesting_depth,
                issues_count=len(issues),
            )

        except Exception as e:
            return QuickAnalyzeResponse(
                status=AnalysisStatus.ERROR,
                time_complexity=complexity_result.time_complexity,
                space_complexity=complexity_result.space_complexity,
                time_explanation="",
                space_explanation="",
                max_nesting_depth=0,
                issues_count=0,
                error=str(e),
            )

    def _calculate_score(
        self, complexity, issues: list[PerformanceIssue]
    ) -> int:
        """Calculate optimization score (0-100)."""
        score = 100

        # Deduct for complexity
        time_rank = COMPLEXITY_RANK.get(complexity.time_complexity, 5)
        if time_rank > 3:  # Worse than linear
            score -= (time_rank - 3) * 10

        space_rank = COMPLEXITY_RANK.get(complexity.space_complexity, 5)
        if space_rank > 3:
            score -= (space_rank - 3) * 5

        # Deduct for issues
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score -= 20
            elif issue.severity == IssueSeverity.ERROR:
                score -= 10
            elif issue.severity == IssueSeverity.WARNING:
                score -= 5
            elif issue.severity == IssueSeverity.INFO:
                score -= 2

        # Deduct for deep nesting
        if complexity.max_nesting_depth > 3:
            score -= (complexity.max_nesting_depth - 3) * 5

        return max(0, min(100, score))


# Singleton instance
performance_service = PerformanceService()
