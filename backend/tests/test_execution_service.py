"""Unit tests for Execution Service"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from code_tutor.execution.application.services import ExecutionService
from code_tutor.execution.application.dto import ExecuteCodeRequest, ExecuteCodeResponse
from code_tutor.execution.domain.value_objects import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)


@pytest.fixture
def mock_sandbox():
    """Create mock sandbox executor"""
    sandbox = AsyncMock()
    sandbox.execute.return_value = ExecutionResult(
        execution_id=uuid4(),
        status=ExecutionStatus.SUCCESS,
        stdout="Hello, World!\n",
        stderr="",
        exit_code=0,
        execution_time_ms=50.0,
        memory_usage_mb=10.0,
    )
    return sandbox


@pytest.fixture
def execution_service(mock_sandbox):
    """Create execution service with mock sandbox"""
    service = ExecutionService(use_docker=False)
    service._sandbox = mock_sandbox
    return service


class TestExecutionService:
    """Tests for ExecutionService"""

    @pytest.mark.asyncio
    async def test_execute_code_success(self, execution_service, mock_sandbox):
        """Test successful code execution"""
        request = ExecuteCodeRequest(
            code="print('Hello, World!')",
            language="python",
        )

        result = await execution_service.execute_code(request)

        assert result is not None
        assert result.status == ExecutionStatus.SUCCESS.value
        assert "Hello" in result.stdout
        mock_sandbox.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_code_with_stdin(self, execution_service, mock_sandbox):
        """Test code execution with stdin"""
        mock_sandbox.execute.return_value = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.SUCCESS,
            stdout="Hello, Claude!\n",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
        )

        request = ExecuteCodeRequest(
            code="name = input()\\nprint(f'Hello, {name}!')",
            language="python",
            stdin="Claude",
        )

        result = await execution_service.execute_code(request)

        assert result is not None
        assert result.is_success is True

    @pytest.mark.asyncio
    async def test_execute_code_timeout(self, execution_service, mock_sandbox):
        """Test code execution timeout"""
        mock_sandbox.execute.return_value = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.TIMEOUT,
            stdout="",
            stderr="Execution timed out",
            exit_code=-1,
            execution_time_ms=5000.0,
            memory_usage_mb=0.0,
            error_message="Execution timed out",
        )

        request = ExecuteCodeRequest(
            code="while True: pass",
            language="python",
            timeout_seconds=5,
        )

        result = await execution_service.execute_code(request)

        assert result.status == ExecutionStatus.TIMEOUT.value
        assert result.is_success is False

    @pytest.mark.asyncio
    async def test_execute_code_runtime_error(self, execution_service, mock_sandbox):
        """Test code execution with runtime error"""
        mock_sandbox.execute.return_value = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.RUNTIME_ERROR,
            stdout="",
            stderr="ZeroDivisionError: division by zero",
            exit_code=1,
            execution_time_ms=10.0,
            memory_usage_mb=5.0,
            error_message="ZeroDivisionError",
        )

        request = ExecuteCodeRequest(
            code="x = 1 / 0",
            language="python",
        )

        result = await execution_service.execute_code(request)

        assert result.status == ExecutionStatus.RUNTIME_ERROR.value
        assert "ZeroDivisionError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_code_memory_exceeded(self, execution_service, mock_sandbox):
        """Test code execution memory limit exceeded"""
        mock_sandbox.execute.return_value = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.MEMORY_EXCEEDED,
            stdout="",
            stderr="Memory limit exceeded",
            exit_code=-1,
            execution_time_ms=100.0,
            memory_usage_mb=512.0,
            error_message="Memory limit exceeded",
        )

        request = ExecuteCodeRequest(
            code="x = [0] * 10**9",  # Large memory allocation
            language="python",
        )

        result = await execution_service.execute_code(request)

        assert result.status == ExecutionStatus.MEMORY_EXCEEDED.value
        assert result.is_success is False


class TestExecutionStatus:
    """Tests for ExecutionStatus enum"""

    def test_status_values(self):
        """Test execution status values"""
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.RUNTIME_ERROR.value == "runtime_error"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.MEMORY_EXCEEDED.value == "memory_exceeded"
        assert ExecutionStatus.COMPILATION_ERROR.value == "compilation_error"
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"


class TestExecutionResult:
    """Tests for ExecutionResult value object"""

    def test_success_result(self):
        """Test successful execution result"""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.SUCCESS,
            stdout="output",
            stderr="",
            exit_code=0,
            execution_time_ms=100.0,
            memory_usage_mb=50.0,
        )
        assert result.is_success is True
        assert result.output == "output"

    def test_failure_result(self):
        """Test failed execution result"""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.RUNTIME_ERROR,
            stdout="",
            stderr="error",
            exit_code=1,
            execution_time_ms=100.0,
            memory_usage_mb=50.0,
        )
        assert result.is_success is False

    def test_timeout_result(self):
        """Test timeout execution result"""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.TIMEOUT,
            stdout="partial",
            stderr="timeout",
            exit_code=-1,
            execution_time_ms=10000.0,
            memory_usage_mb=0.0,
        )
        assert result.is_success is False
        assert result.status == ExecutionStatus.TIMEOUT

    def test_output_stripping(self):
        """Test output trailing whitespace stripping"""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.SUCCESS,
            stdout="Hello\n\n  ",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
        )
        assert result.output == "Hello"


class TestExecutionRequest:
    """Tests for ExecutionRequest value object"""

    def test_default_values(self):
        """Test execution request with defaults"""
        request = ExecutionRequest(code="print('hello')")
        assert request.language == "python"
        assert request.stdin == ""
        assert request.timeout_seconds == 5
        assert request.memory_limit_mb == 256

    def test_custom_values(self):
        """Test execution request with custom values"""
        request = ExecutionRequest(
            code="print('hello')",
            language="python",
            stdin="input data",
            timeout_seconds=10,
            memory_limit_mb=512,
            cpu_limit=1.0,
        )
        assert request.stdin == "input data"
        assert request.timeout_seconds == 10
        assert request.memory_limit_mb == 512
        assert request.cpu_limit == 1.0
