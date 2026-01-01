"""Python code tracer for step-by-step debugging."""

import sys
import io
import copy
import time
import traceback
from typing import Any, Optional
from contextlib import redirect_stdout, redirect_stderr

from code_tutor.debugger.domain import (
    StepType,
    Variable,
    StackFrame,
    ExecutionStep,
    DebugSession,
    DebugResult,
    DebugStatus,
    MAX_STEPS,
    MAX_EXECUTION_TIME,
)


class CodeTracer:
    """Traces Python code execution step by step."""

    def __init__(
        self,
        code: str,
        input_data: str = "",
        breakpoints: list[int] = None,
        max_steps: int = MAX_STEPS,
        max_time: float = MAX_EXECUTION_TIME,
    ):
        self.code = code
        self.input_data = input_data
        self.breakpoints = set(breakpoints or [])
        self.max_steps = max_steps
        self.max_time = max_time

        self.steps: list[ExecutionStep] = []
        self.step_count = 0
        self.start_time: float = 0
        self.output_buffer = io.StringIO()
        self.error: Optional[str] = None
        self.lines: list[str] = []
        self.call_stack: list[StackFrame] = []
        self.user_module_name = "<user_code>"

        # Skip internal frames
        self._skip_frames = {"<string>", "<module>"}

    def trace(self) -> DebugResult:
        """Execute code with tracing and return result."""
        session = DebugSession.create(
            code=self.code,
            input_data=self.input_data,
            breakpoints=list(self.breakpoints),
        )
        session.status = DebugStatus.RUNNING

        self.lines = self.code.split("\n")
        self.start_time = time.time()

        # Prepare execution environment
        input_lines = self.input_data.split("\n") if self.input_data else []
        input_iter = iter(input_lines)

        def mock_input(prompt=""):
            """Mock input function."""
            self.output_buffer.write(prompt)
            try:
                value = next(input_iter)
                self.output_buffer.write(value + "\n")
                return value
            except StopIteration:
                return ""

        # Create execution namespace
        namespace = {
            "__name__": self.user_module_name,
            "__builtins__": __builtins__,
            "input": mock_input,
            "print": self._traced_print,
        }

        try:
            # Compile code
            compiled = compile(self.code, self.user_module_name, "exec")

            # Execute with tracing
            with redirect_stdout(self.output_buffer), redirect_stderr(self.output_buffer):
                sys.settrace(self._trace_callback)
                try:
                    exec(compiled, namespace)
                finally:
                    sys.settrace(None)

            session.complete(self.output_buffer.getvalue())

        except TimeoutError:
            session.status = DebugStatus.TIMEOUT
            session.error = "Execution timed out"
            self.error = "Execution timed out"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            session.fail(f"{error_msg}\n{tb}")
            self.error = error_msg

            # Add exception step
            if self.steps:
                last_step = self.steps[-1]
                last_step.exception = error_msg

        session.steps = self.steps
        session.total_steps = len(self.steps)

        execution_time = (time.time() - self.start_time) * 1000

        return DebugResult(
            session_id=session.id,
            status=session.status,
            steps=self.steps,
            total_steps=len(self.steps),
            output=self.output_buffer.getvalue(),
            error=self.error,
            execution_time_ms=execution_time,
        )

    def _traced_print(self, *args, **kwargs):
        """Traced print function."""
        output = io.StringIO()
        print(*args, file=output, **kwargs)
        result = output.getvalue()
        self.output_buffer.write(result)

        # Update last step with output
        if self.steps:
            self.steps[-1].output += result

    def _trace_callback(self, frame, event: str, arg):
        """Trace callback for sys.settrace."""
        # Check limits
        if self.step_count >= self.max_steps:
            raise TimeoutError("Maximum steps exceeded")

        elapsed = time.time() - self.start_time
        if elapsed > self.max_time:
            raise TimeoutError("Execution timed out")

        # Only trace user code
        filename = frame.f_code.co_filename
        if filename != self.user_module_name:
            return self._trace_callback

        # Get event type
        if event == "call":
            step_type = StepType.CALL
        elif event == "line":
            step_type = StepType.LINE
        elif event == "return":
            step_type = StepType.RETURN
        elif event == "exception":
            step_type = StepType.EXCEPTION
        else:
            return self._trace_callback

        # Get line info
        line_no = frame.f_lineno
        line_content = self.lines[line_no - 1] if 0 < line_no <= len(self.lines) else ""
        func_name = frame.f_code.co_name

        # Skip module-level for certain events
        if func_name == "<module>" and event == "call":
            return self._trace_callback

        # Extract variables
        variables = self._extract_variables(frame)

        # Update call stack
        self._update_call_stack(frame, event, variables)

        # Create step
        self.step_count += 1
        step = ExecutionStep(
            step_number=self.step_count,
            step_type=step_type,
            line_number=line_no,
            line_content=line_content.strip(),
            function_name=func_name if func_name != "<module>" else "main",
            variables=variables,
            call_stack=[copy.copy(f) for f in self.call_stack],
            output="",
            return_value=repr(arg) if event == "return" and arg is not None else None,
        )

        self.steps.append(step)

        return self._trace_callback

    def _extract_variables(self, frame) -> list[Variable]:
        """Extract local variables from frame."""
        variables = []

        # Get local variables
        for name, value in frame.f_locals.items():
            # Skip internal variables
            if name.startswith("_") or name in ("__builtins__", "__name__", "__doc__"):
                continue

            # Skip functions and modules
            if callable(value) and not isinstance(value, type):
                continue

            try:
                var = Variable.from_python_value(name, value)
                variables.append(var)
            except Exception:
                # Skip variables that can't be represented
                pass

        return variables

    def _update_call_stack(self, frame, event: str, variables: list[Variable]) -> None:
        """Update the call stack based on event."""
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno

        if event == "call":
            # Push new frame
            stack_frame = StackFrame(
                function_name=func_name if func_name != "<module>" else "main",
                filename=self.user_module_name,
                line_number=line_no,
                local_variables=variables,
            )
            self.call_stack.append(stack_frame)

        elif event == "return":
            # Pop frame
            if self.call_stack:
                self.call_stack.pop()

        elif event == "line" and self.call_stack:
            # Update current frame
            self.call_stack[-1].line_number = line_no
            self.call_stack[-1].local_variables = variables


def trace_code(
    code: str,
    input_data: str = "",
    breakpoints: list[int] = None,
) -> DebugResult:
    """Convenience function to trace code execution."""
    tracer = CodeTracer(code, input_data, breakpoints)
    return tracer.trace()
