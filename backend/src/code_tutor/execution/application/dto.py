"""Code Execution DTOs"""

from uuid import UUID

from pydantic import BaseModel, Field


class ExecuteCodeRequest(BaseModel):
    """Request to execute code"""

    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(default="python", pattern="^(python)$")
    stdin: str = Field(default="", max_length=10000)
    timeout_seconds: int = Field(default=5, ge=1, le=30)


class ExecuteCodeResponse(BaseModel):
    """Response from code execution"""

    execution_id: UUID
    status: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    memory_usage_mb: float
    error_message: str | None = None
    is_success: bool

    class Config:
        from_attributes = True
