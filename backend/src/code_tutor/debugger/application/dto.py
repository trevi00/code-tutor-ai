"""Debugger DTOs."""

from typing import Optional
from pydantic import BaseModel, Field

from code_tutor.debugger.domain import StepType, VariableType, DebugStatus


class DebugRequest(BaseModel):
    """Request to debug code."""

    code: str = Field(description="Python code to debug")
    input_data: str = Field(default="", description="Input data for the code")
    breakpoints: list[int] = Field(
        default_factory=list, description="Line numbers to set breakpoints"
    )


class VariableResponse(BaseModel):
    """Variable information."""

    name: str
    value: str
    type: VariableType


class StackFrameResponse(BaseModel):
    """Stack frame information."""

    function_name: str
    filename: str
    line_number: int
    local_variables: list[VariableResponse]


class ExecutionStepResponse(BaseModel):
    """A single execution step."""

    step_number: int
    step_type: StepType
    line_number: int
    line_content: str
    function_name: str
    variables: list[VariableResponse]
    call_stack: list[StackFrameResponse]
    output: str
    return_value: Optional[str] = None
    exception: Optional[str] = None


class DebugResponse(BaseModel):
    """Debug execution result."""

    session_id: str
    status: DebugStatus
    total_steps: int
    output: str
    error: Optional[str] = None
    execution_time_ms: float
    steps: list[ExecutionStepResponse]


class StepInfoResponse(BaseModel):
    """Information about a specific step."""

    step: ExecutionStepResponse
    has_previous: bool
    has_next: bool
    is_breakpoint: bool


class DebugSummaryResponse(BaseModel):
    """Summary of debug execution."""

    session_id: str
    status: DebugStatus
    total_steps: int
    total_lines: int
    functions_called: list[str]
    variables_used: list[str]
    has_error: bool
    error_line: Optional[int] = None
    execution_time_ms: float
