"""Debugger application services."""


from code_tutor.debugger.domain import (
    DebugResult,
    ExecutionStep,
    StackFrame,
    Variable,
)

from .dto import (
    DebugRequest,
    DebugResponse,
    DebugSummaryResponse,
    ExecutionStepResponse,
    StackFrameResponse,
    StepInfoResponse,
    VariableResponse,
)
from .tracer import trace_code


class DebugService:
    """Service for debugging code."""

    def __init__(self):
        self._sessions: dict[str, DebugResult] = {}

    async def debug_code(self, request: DebugRequest) -> DebugResponse:
        """Execute code with step-by-step tracing."""
        result = trace_code(
            code=request.code,
            input_data=request.input_data,
            breakpoints=request.breakpoints,
        )

        # Store session for later retrieval
        session_id = str(result.session_id)
        self._sessions[session_id] = result

        return self._to_debug_response(result)

    async def get_session(self, session_id: str) -> DebugResponse | None:
        """Get a debug session by ID."""
        result = self._sessions.get(session_id)
        if result:
            return self._to_debug_response(result)
        return None

    async def get_step(
        self,
        session_id: str,
        step_number: int,
        breakpoints: list[int] = None,
    ) -> StepInfoResponse | None:
        """Get information about a specific step."""
        result = self._sessions.get(session_id)
        if not result or step_number < 1 or step_number > len(result.steps):
            return None

        step = result.steps[step_number - 1]
        is_breakpoint = step.line_number in (breakpoints or [])

        return StepInfoResponse(
            step=self._to_step_response(step),
            has_previous=step_number > 1,
            has_next=step_number < len(result.steps),
            is_breakpoint=is_breakpoint,
        )

    async def get_summary(self, session_id: str) -> DebugSummaryResponse | None:
        """Get a summary of the debug session."""
        result = self._sessions.get(session_id)
        if not result:
            return None

        # Collect unique functions and variables
        functions = set()
        variables = set()
        lines_executed = set()
        error_line = None

        for step in result.steps:
            functions.add(step.function_name)
            lines_executed.add(step.line_number)
            for var in step.variables:
                variables.add(var.name)
            if step.exception:
                error_line = step.line_number

        return DebugSummaryResponse(
            session_id=session_id,
            status=result.status,
            total_steps=result.total_steps,
            total_lines=len(lines_executed),
            functions_called=sorted(list(functions)),
            variables_used=sorted(list(variables)),
            has_error=result.error is not None,
            error_line=error_line,
            execution_time_ms=result.execution_time_ms,
        )

    def _to_debug_response(self, result: DebugResult) -> DebugResponse:
        """Convert DebugResult to DebugResponse."""
        return DebugResponse(
            session_id=str(result.session_id),
            status=result.status,
            total_steps=result.total_steps,
            output=result.output,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            steps=[self._to_step_response(step) for step in result.steps],
        )

    def _to_step_response(self, step: ExecutionStep) -> ExecutionStepResponse:
        """Convert ExecutionStep to response."""
        return ExecutionStepResponse(
            step_number=step.step_number,
            step_type=step.step_type,
            line_number=step.line_number,
            line_content=step.line_content,
            function_name=step.function_name,
            variables=[self._to_variable_response(v) for v in step.variables],
            call_stack=[self._to_stack_frame_response(f) for f in step.call_stack],
            output=step.output,
            return_value=step.return_value,
            exception=step.exception,
        )

    def _to_variable_response(self, var: Variable) -> VariableResponse:
        """Convert Variable to response."""
        return VariableResponse(
            name=var.name,
            value=var.value,
            type=var.type,
        )

    def _to_stack_frame_response(self, frame: StackFrame) -> StackFrameResponse:
        """Convert StackFrame to response."""
        return StackFrameResponse(
            function_name=frame.function_name,
            filename=frame.filename,
            line_number=frame.line_number,
            local_variables=[
                self._to_variable_response(v) for v in frame.local_variables
            ],
        )


# Singleton instance
debug_service = DebugService()
