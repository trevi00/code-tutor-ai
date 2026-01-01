"""Runtime profiler for code execution."""

import cProfile
import io
import pstats
import sys
import time
import tracemalloc
from contextlib import redirect_stdout
from typing import Optional

from ..domain import (
    FunctionProfile,
    HotspotAnalysis,
    LineProfile,
    MemoryMetrics,
    RuntimeMetrics,
)


class RuntimeProfiler:
    """Profiles code execution for performance metrics."""

    def __init__(self, code: str, input_data: Optional[str] = None):
        self.code = code
        self.input_data = input_data
        self.lines = code.split("\n")

    def profile(self) -> tuple[RuntimeMetrics, HotspotAnalysis]:
        """Profile code execution."""
        # Wrap code in a function for profiling
        wrapped_code = self._wrap_code()

        # Setup namespace
        namespace: dict = {"__name__": "__main__", "__builtins__": __builtins__}

        # Setup input
        if self.input_data:
            input_lines = self.input_data.strip().split("\n")
            input_iter = iter(input_lines)
            namespace["input"] = lambda: next(input_iter, "")

        # Compile code
        try:
            compiled = compile(wrapped_code, "<profiled>", "exec")
        except SyntaxError as e:
            return self._error_metrics(str(e)), HotspotAnalysis()

        # Profile execution
        profiler = cProfile.Profile()
        start_time = time.perf_counter()

        try:
            with redirect_stdout(io.StringIO()):
                profiler.enable()
                exec(compiled, namespace)
                profiler.disable()
        except Exception as e:
            profiler.disable()
            return self._error_metrics(str(e)), HotspotAnalysis()

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Analyze profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")

        # Extract metrics
        runtime_metrics = self._extract_runtime_metrics(stats, execution_time_ms)
        hotspot_analysis = self._extract_hotspots(stats, execution_time_ms)

        return runtime_metrics, hotspot_analysis

    def _wrap_code(self) -> str:
        """Wrap code in a function for better profiling."""
        # Indent code
        indented = "\n".join("    " + line for line in self.lines)
        return f"def __profiled_main__():\n{indented}\n\n__profiled_main__()"

    def _extract_runtime_metrics(
        self, stats: pstats.Stats, execution_time_ms: float
    ) -> RuntimeMetrics:
        """Extract runtime metrics from profile stats."""
        total_calls = 0
        total_time = 0.0
        max_depth = 0

        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            total_calls += nc
            total_time += tt

            # Estimate call depth from caller chain
            depth = len(callers) if callers else 0
            max_depth = max(max_depth, depth)

        return RuntimeMetrics(
            execution_time_ms=execution_time_ms,
            cpu_time_ms=total_time * 1000,
            function_calls=total_calls,
            line_executions=0,  # Would need line profiler
            peak_call_depth=max_depth,
        )

    def _extract_hotspots(
        self, stats: pstats.Stats, total_time_ms: float
    ) -> HotspotAnalysis:
        """Extract hotspot functions from profile."""
        hotspot_functions: list[FunctionProfile] = []

        for func, (cc, nc, tt, ct, _) in stats.stats.items():
            filename, line, name = func

            # Skip built-in and system functions
            if filename.startswith("<") and name not in ["__profiled_main__"]:
                if name in ["<module>", "<listcomp>", "<dictcomp>", "<genexpr>"]:
                    pass  # Include these
                else:
                    continue

            # Calculate metrics
            total_time_func = ct * 1000  # cumulative time
            own_time_func = tt * 1000  # own time
            avg_time = total_time_func / nc if nc > 0 else 0
            percentage = (total_time_func / total_time_ms * 100) if total_time_ms > 0 else 0

            # Use clean name
            clean_name = name
            if name == "__profiled_main__":
                clean_name = "<main>"

            profile = FunctionProfile(
                name=clean_name,
                calls=nc,
                total_time_ms=total_time_func,
                own_time_ms=own_time_func,
                avg_time_ms=avg_time,
                percentage=percentage,
            )
            hotspot_functions.append(profile)

        # Sort by total time
        hotspot_functions.sort(key=lambda x: x.total_time_ms, reverse=True)

        # Find bottleneck
        bottleneck_function = None
        if hotspot_functions:
            # Exclude main wrapper
            for func in hotspot_functions:
                if func.name != "<main>":
                    bottleneck_function = func.name
                    break

        return HotspotAnalysis(
            hotspot_functions=hotspot_functions[:10],  # Top 10
            hotspot_lines=[],  # Would need line profiler
            total_execution_time_ms=total_time_ms,
            bottleneck_function=bottleneck_function,
        )

    def _error_metrics(self, error: str) -> RuntimeMetrics:
        """Return error metrics."""
        return RuntimeMetrics(
            execution_time_ms=0,
            cpu_time_ms=0,
            function_calls=0,
            line_executions=0,
            peak_call_depth=0,
        )


class MemoryProfiler:
    """Profiles memory usage during code execution."""

    def __init__(self, code: str, input_data: Optional[str] = None):
        self.code = code
        self.input_data = input_data

    def profile(self) -> MemoryMetrics:
        """Profile memory usage."""
        # Setup namespace
        namespace: dict = {"__name__": "__main__", "__builtins__": __builtins__}

        # Setup input
        if self.input_data:
            input_lines = self.input_data.strip().split("\n")
            input_iter = iter(input_lines)
            namespace["input"] = lambda: next(input_iter, "")

        # Compile code
        try:
            compiled = compile(self.code, "<memory_profiled>", "exec")
        except SyntaxError:
            return self._empty_metrics()

        # Start memory tracking
        tracemalloc.start()

        try:
            with redirect_stdout(io.StringIO()):
                exec(compiled, namespace)
        except Exception:
            pass
        finally:
            # Get memory snapshot
            snapshot = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # Analyze snapshot
        stats = snapshot.statistics("lineno")

        total_size = sum(stat.size for stat in stats)
        largest = stats[0] if stats else None

        allocations = len(stats)

        return MemoryMetrics(
            peak_memory_mb=peak / (1024 * 1024),
            average_memory_mb=current / (1024 * 1024),
            allocations_count=allocations,
            deallocations_count=0,  # Not tracked
            largest_object_mb=largest.size / (1024 * 1024) if largest else 0,
            largest_object_type=str(largest.traceback) if largest else None,
        )

    def _empty_metrics(self) -> MemoryMetrics:
        """Return empty metrics on error."""
        return MemoryMetrics(
            peak_memory_mb=0,
            average_memory_mb=0,
            allocations_count=0,
            deallocations_count=0,
            largest_object_mb=0,
        )


def profile_code(
    code: str, input_data: Optional[str] = None
) -> tuple[RuntimeMetrics, MemoryMetrics, HotspotAnalysis]:
    """Profile code and return all metrics."""
    runtime_profiler = RuntimeProfiler(code, input_data)
    memory_profiler = MemoryProfiler(code, input_data)

    runtime_metrics, hotspots = runtime_profiler.profile()
    memory_metrics = memory_profiler.profile()

    return runtime_metrics, memory_metrics, hotspots
