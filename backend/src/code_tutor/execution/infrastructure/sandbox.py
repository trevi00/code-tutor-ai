"""Docker sandbox for code execution"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID

from code_tutor.execution.domain.value_objects import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from code_tutor.shared.config import get_settings
from code_tutor.shared.constants import Truncation
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DockerSandbox:
    """
    Docker-based sandbox for secure code execution.
    Runs user code in isolated containers with resource limits.
    """

    # Pin to specific version for reproducibility
    # Update periodically after testing (check: https://hub.docker.com/_/python)
    DOCKER_IMAGE = "python:3.11.9-slim"

    def __init__(self) -> None:
        self._settings = get_settings()

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code in a sandboxed Docker container"""
        start_time = time.perf_counter()

        try:
            # Create temporary directory for code
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = Path(temp_dir) / "solution.py"
                code_file.write_text(request.code)

                # Build docker command
                docker_cmd = self._build_docker_command(
                    code_path=str(code_file),
                    stdin=request.stdin,
                    timeout=request.timeout_seconds,
                    memory_mb=request.memory_limit_mb,
                    cpu_limit=request.cpu_limit,
                )

                # Execute with timeout
                try:
                    process = await asyncio.create_subprocess_shell(
                        docker_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=request.stdin.encode()),
                        timeout=request.timeout_seconds + 2,  # Extra buffer
                    )

                    execution_time = (time.perf_counter() - start_time) * 1000

                    if process.returncode == 0:
                        return ExecutionResult(
                            execution_id=request.execution_id,
                            status=ExecutionStatus.SUCCESS,
                            stdout=stdout.decode("utf-8", errors="replace"),
                            stderr=stderr.decode("utf-8", errors="replace"),
                            exit_code=0,
                            execution_time_ms=execution_time,
                        )
                    else:
                        # Check for specific error types
                        stderr_text = stderr.decode("utf-8", errors="replace")

                        if (
                            "MemoryError" in stderr_text
                            or "killed" in stderr_text.lower()
                        ):
                            status = ExecutionStatus.MEMORY_EXCEEDED
                        else:
                            status = ExecutionStatus.RUNTIME_ERROR

                        return ExecutionResult(
                            execution_id=request.execution_id,
                            status=status,
                            stdout=stdout.decode("utf-8", errors="replace"),
                            stderr=stderr_text,
                            exit_code=process.returncode or 1,
                            execution_time_ms=execution_time,
                            error_message=stderr_text[:Truncation.ERROR_MESSAGE_MAX] if stderr_text else None,
                        )

                except TimeoutError:
                    execution_time = (time.perf_counter() - start_time) * 1000

                    # Kill the container if still running
                    await self._cleanup_container(request.execution_id)

                    return ExecutionResult(
                        execution_id=request.execution_id,
                        status=ExecutionStatus.TIMEOUT,
                        execution_time_ms=execution_time,
                        error_message=f"Execution exceeded time limit of {request.timeout_seconds}s",
                    )

        except Exception as e:
            logger.error("Sandbox execution error", error=str(e))
            return ExecutionResult(
                execution_id=request.execution_id,
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=str(e),
            )

    def _build_docker_command(
        self,
        code_path: str,
        stdin: str,
        timeout: int,
        memory_mb: int,
        cpu_limit: float,
    ) -> str:
        """Build Docker run command with security constraints"""
        return (
            f"docker run --rm "
            f"--network none "  # No network access
            f"--memory {memory_mb}m "
            f"--memory-swap {memory_mb}m "  # No swap
            f"--cpus {cpu_limit} "
            f"--pids-limit 50 "  # Limit processes
            f"--read-only "  # Read-only filesystem
            f"--tmpfs /tmp:rw,noexec,nosuid,size=64m "
            f"-v {code_path}:/code/solution.py:ro "
            f"-w /code "
            f"{self.DOCKER_IMAGE} "
            f"python -u solution.py"
        )

    async def _cleanup_container(self, execution_id: UUID) -> None:
        """Cleanup any orphaned containers"""
        try:
            # Find and kill containers with the execution ID label
            process = await asyncio.create_subprocess_shell(
                f"docker ps -q --filter label=execution_id={execution_id}",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            container_ids = stdout.decode().strip().split("\n")
            for container_id in container_ids:
                if container_id:
                    await asyncio.create_subprocess_shell(
                        f"docker kill {container_id}",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
        except Exception as e:
            logger.warning("Container cleanup failed", error=str(e))


class MockSandbox:
    """
    Mock sandbox for testing without Docker.
    Uses subprocess with limited security.
    """

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code using subprocess (for development only)"""
        import subprocess

        start_time = time.perf_counter()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = Path(temp_dir) / "solution.py"
                code_file.write_text(request.code, encoding="utf-8")

                try:
                    result = subprocess.run(
                        [sys.executable, str(code_file)],
                        input=request.stdin,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_seconds,
                        encoding="utf-8",
                        errors="replace",
                    )

                    execution_time = (time.perf_counter() - start_time) * 1000

                    if result.returncode == 0:
                        return ExecutionResult(
                            execution_id=request.execution_id,
                            status=ExecutionStatus.SUCCESS,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            exit_code=0,
                            execution_time_ms=execution_time,
                        )
                    else:
                        return ExecutionResult(
                            execution_id=request.execution_id,
                            status=ExecutionStatus.RUNTIME_ERROR,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            exit_code=result.returncode,
                            execution_time_ms=execution_time,
                            error_message=result.stderr[:Truncation.ERROR_MESSAGE_MAX]
                            if result.stderr
                            else None,
                        )

                except subprocess.TimeoutExpired:
                    return ExecutionResult(
                        execution_id=request.execution_id,
                        status=ExecutionStatus.TIMEOUT,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        error_message=f"Execution exceeded time limit of {request.timeout_seconds}s",
                    )

        except Exception as e:
            import traceback

            error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"MockSandbox execution error: {error_detail}")
            return ExecutionResult(
                execution_id=request.execution_id,
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=error_detail[:Truncation.ERROR_MESSAGE_MAX],
            )
