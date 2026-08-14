"""Code Execution application services"""

from code_tutor.execution.application.dto import ExecuteCodeRequest, ExecuteCodeResponse
from code_tutor.execution.domain.value_objects import ExecutionRequest, ExecutionResult
from code_tutor.execution.infrastructure.sandbox import DockerSandbox, MockSandbox
from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ExecutionService:
    """Code execution service"""

    def __init__(self, use_docker: bool = True) -> None:
        self._settings = get_settings()
        self._sandbox = DockerSandbox() if use_docker else MockSandbox()

    async def execute_code(self, request: ExecuteCodeRequest) -> ExecuteCodeResponse:
        """Execute code and return result"""
        execution_request = ExecutionRequest(
            code=request.code,
            language=request.language,
            stdin=request.stdin,
            timeout_seconds=request.timeout_seconds,
            memory_limit_mb=self._settings.SANDBOX_MEMORY_LIMIT_MB,
            cpu_limit=self._settings.SANDBOX_CPU_LIMIT,
        )

        result = await self._sandbox.execute(execution_request)

        logger.info(
            "Code executed",
            execution_id=str(result.execution_id),
            status=result.status.value,
            time_ms=result.execution_time_ms,
        )

        return self._to_response(result)

    def _to_response(self, result: ExecutionResult) -> ExecuteCodeResponse:
        """Convert ExecutionResult to ExecuteCodeResponse"""
        return ExecuteCodeResponse(
            execution_id=result.execution_id,
            status=result.status.value,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            memory_usage_mb=result.memory_usage_mb,
            error_message=result.error_message,
            is_success=result.is_success,
        )
