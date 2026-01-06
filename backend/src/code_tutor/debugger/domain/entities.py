"""Debugger domain entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)

from .value_objects import (
    StepType,
    VariableType,
    DebugStatus,
    get_variable_type,
    format_variable_value,
)


@dataclass
class Variable:
    """A variable at a specific point in execution."""

    name: str
    value: str  # Formatted string representation
    type: VariableType
    raw_value: Any = field(repr=False, default=None)

    @classmethod
    def from_python_value(cls, name: str, value: Any) -> "Variable":
        """Create Variable from Python value."""
        return cls(
            name=name,
            value=format_variable_value(value),
            type=get_variable_type(value),
            raw_value=value,
        )


@dataclass
class StackFrame:
    """A stack frame in the call stack."""

    function_name: str
    filename: str
    line_number: int
    local_variables: list[Variable] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "filename": self.filename,
            "line_number": self.line_number,
            "local_variables": [
                {"name": v.name, "value": v.value, "type": v.type.value}
                for v in self.local_variables
            ],
        }


@dataclass
class ExecutionStep:
    """A single step in code execution."""

    step_number: int
    step_type: StepType
    line_number: int
    line_content: str
    function_name: str
    variables: list[Variable] = field(default_factory=list)
    call_stack: list[StackFrame] = field(default_factory=list)
    output: str = ""
    return_value: Optional[str] = None
    exception: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "function_name": self.function_name,
            "variables": [
                {"name": v.name, "value": v.value, "type": v.type.value}
                for v in self.variables
            ],
            "call_stack": [frame.to_dict() for frame in self.call_stack],
            "output": self.output,
            "return_value": self.return_value,
            "exception": self.exception,
        }


@dataclass
class DebugSession:
    """A debugging session."""

    id: UUID
    code: str
    input_data: str
    status: DebugStatus
    steps: list[ExecutionStep] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    output: str = ""
    error: Optional[str] = None
    breakpoints: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        code: str,
        input_data: str = "",
        breakpoints: list[int] = None,
    ) -> "DebugSession":
        """Create a new debug session."""
        return cls(
            id=uuid4(),
            code=code,
            input_data=input_data,
            status=DebugStatus.PENDING,
            breakpoints=breakpoints or [],
        )

    def add_step(self, step: ExecutionStep) -> None:
        """Add an execution step."""
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def complete(self, output: str = "") -> None:
        """Mark session as completed."""
        self.status = DebugStatus.COMPLETED
        self.output = output
        self.completed_at = utc_now()

    def fail(self, error: str) -> None:
        """Mark session as failed."""
        self.status = DebugStatus.ERROR
        self.error = error
        self.completed_at = utc_now()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "output": self.output,
            "error": self.error,
            "breakpoints": self.breakpoints,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class DebugResult:
    """Result of a debug execution."""

    session_id: UUID
    status: DebugStatus
    steps: list[ExecutionStep]
    total_steps: int
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": str(self.session_id),
            "status": self.status.value,
            "total_steps": self.total_steps,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "steps": [step.to_dict() for step in self.steps],
        }
