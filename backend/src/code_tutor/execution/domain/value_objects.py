"""Code Execution domain value objects"""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from code_tutor.shared.domain.base import ValueObject


class ExecutionStatus(str, Enum):
    """Code execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    RUNTIME_ERROR = "runtime_error"
    COMPILATION_ERROR = "compilation_error"


@dataclass(frozen=True)
class ExecutionRequest(ValueObject):
    """Request to execute code"""

    code: str
    language: str = "python"
    stdin: str = ""
    timeout_seconds: int = 5
    memory_limit_mb: int = 256
    cpu_limit: float = 0.5
    execution_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ExecutionResult(ValueObject):
    """Result of code execution"""

    execution_id: UUID
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    @property
    def output(self) -> str:
        """Get normalized output (stdout with trailing whitespace stripped)"""
        return self.stdout.rstrip()
