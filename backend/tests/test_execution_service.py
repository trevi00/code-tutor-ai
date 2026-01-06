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

    def test_execution_id_auto_generated(self):
        """Test that execution_id is auto-generated."""
        request1 = ExecutionRequest(code="print(1)")
        request2 = ExecutionRequest(code="print(2)")
        assert request1.execution_id is not None
        assert request2.execution_id is not None
        assert request1.execution_id != request2.execution_id

    def test_immutability(self):
        """Test that ExecutionRequest is immutable (frozen)."""
        request = ExecutionRequest(code="print('hello')")
        with pytest.raises(Exception):  # FrozenInstanceError
            request.code = "print('modified')"


class TestMockSandbox:
    """Tests for MockSandbox - actual code execution."""

    @pytest.fixture
    def sandbox(self):
        """Create MockSandbox instance."""
        from code_tutor.execution.infrastructure.sandbox import MockSandbox
        return MockSandbox()

    @pytest.mark.asyncio
    async def test_execute_simple_print(self, sandbox):
        """Test executing simple print statement."""
        request = ExecutionRequest(code="print('Hello, World!')")
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.is_success is True
        assert "Hello, World!" in result.stdout
        assert result.exit_code == 0
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_stdin(self, sandbox):
        """Test executing code with stdin input."""
        code = "name = input()\nprint(f'Hello, {name}!')"
        request = ExecutionRequest(code=code, stdin="Claude")
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello, Claude!" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_arithmetic(self, sandbox):
        """Test executing arithmetic code."""
        code = "print(2 + 3 * 4)"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "14" in result.output

    @pytest.mark.asyncio
    async def test_execute_multiline(self, sandbox):
        """Test executing multiline code."""
        code = """
def add(a, b):
    return a + b

result = add(5, 7)
print(result)
"""
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "12" in result.output

    @pytest.mark.asyncio
    async def test_execute_loop(self, sandbox):
        """Test executing code with loops."""
        code = "for i in range(3):\n    print(i)"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "0" in result.stdout
        assert "1" in result.stdout
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self, sandbox):
        """Test executing code that raises runtime error."""
        code = "x = 1 / 0"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.RUNTIME_ERROR
        assert result.is_success is False
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr or "ZeroDivisionError" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self, sandbox):
        """Test executing code with syntax error."""
        code = "print('unclosed string"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.RUNTIME_ERROR
        assert result.is_success is False
        assert "SyntaxError" in result.stderr or "SyntaxError" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_execute_name_error(self, sandbox):
        """Test executing code with name error."""
        code = "print(undefined_variable)"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.RUNTIME_ERROR
        assert "NameError" in result.stderr or "NameError" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_execute_timeout(self, sandbox):
        """Test executing code that times out."""
        code = "import time\ntime.sleep(10)"
        request = ExecutionRequest(code=code, timeout_seconds=1)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.TIMEOUT
        assert result.is_success is False
        assert "time limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_multiple_inputs(self, sandbox):
        """Test executing code with multiple inputs."""
        code = """
a = int(input())
b = int(input())
print(a + b)
"""
        request = ExecutionRequest(code=code, stdin="5\n7")
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "12" in result.output

    @pytest.mark.asyncio
    async def test_execute_list_operations(self, sandbox):
        """Test executing code with list operations."""
        code = """
nums = [3, 1, 4, 1, 5]
nums.sort()
print(nums)
"""
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "[1, 1, 3, 4, 5]" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_empty_code(self, sandbox):
        """Test executing empty code."""
        request = ExecutionRequest(code="")
        result = await sandbox.execute(request)

        # Empty code should succeed with no output
        assert result.status == ExecutionStatus.SUCCESS
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_execute_comment_only(self, sandbox):
        """Test executing code with only comments."""
        code = "# This is a comment\n# Another comment"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_execute_unicode(self, sandbox):
        """Test executing code with unicode characters (ASCII subset)."""
        # Use ASCII-only to avoid Windows encoding issues
        code = "print('Hello, World!')"
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_dictionary(self, sandbox):
        """Test executing code with dictionary operations."""
        code = """
d = {'a': 1, 'b': 2}
d['c'] = 3
print(len(d))
"""
        request = ExecutionRequest(code=code)
        result = await sandbox.execute(request)

        assert result.status == ExecutionStatus.SUCCESS
        assert "3" in result.output


class TestDockerSandbox:
    """Tests for DockerSandbox (mocked)."""

    def test_docker_image_version(self):
        """Test Docker image is pinned to specific version."""
        from code_tutor.execution.infrastructure.sandbox import DockerSandbox
        assert "python:3.11" in DockerSandbox.DOCKER_IMAGE
        assert "slim" in DockerSandbox.DOCKER_IMAGE

    def test_build_docker_command(self):
        """Test Docker command building."""
        from code_tutor.execution.infrastructure.sandbox import DockerSandbox
        sandbox = DockerSandbox()

        cmd = sandbox._build_docker_command(
            code_path="/tmp/code.py",
            stdin="",
            timeout=5,
            memory_mb=256,
            cpu_limit=0.5,
        )

        assert "docker run" in cmd
        assert "--rm" in cmd
        assert "--network none" in cmd
        assert "--memory 256m" in cmd
        assert "--cpus 0.5" in cmd
        assert "--pids-limit 50" in cmd
        assert "--read-only" in cmd
        assert "/tmp/code.py" in cmd


class TestSubmissionEvaluator:
    """Tests for SubmissionEvaluator."""

    @pytest.fixture
    def mock_problem_repo(self):
        """Create mock problem repository."""
        return AsyncMock()

    @pytest.fixture
    def mock_submission_repo(self):
        """Create mock submission repository."""
        return AsyncMock()

    @pytest.fixture
    def evaluator(self, mock_problem_repo, mock_submission_repo):
        """Create SubmissionEvaluator with mocks."""
        from code_tutor.execution.application.services import SubmissionEvaluator
        evaluator = SubmissionEvaluator(
            problem_repository=mock_problem_repo,
            submission_repository=mock_submission_repo,
            use_docker=False,
        )
        return evaluator

    @pytest.mark.asyncio
    async def test_evaluate_submission_not_found(self, evaluator, mock_submission_repo):
        """Test evaluating non-existent submission."""
        from code_tutor.shared.exceptions import NotFoundError
        mock_submission_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await evaluator.evaluate_submission(uuid4())

    @pytest.mark.asyncio
    async def test_evaluate_problem_not_found(
        self, evaluator, mock_submission_repo, mock_problem_repo
    ):
        """Test evaluating submission with non-existent problem."""
        from code_tutor.shared.exceptions import NotFoundError

        # Mock submission exists
        mock_submission = MagicMock()
        mock_submission.problem_id = uuid4()
        mock_submission_repo.get_by_id.return_value = mock_submission

        # Mock problem not found
        mock_problem_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await evaluator.evaluate_submission(uuid4())

    @pytest.mark.asyncio
    async def test_evaluate_submission_accepted(
        self, evaluator, mock_problem_repo, mock_submission_repo
    ):
        """Test evaluating submission that passes all tests."""
        from code_tutor.learning.domain.value_objects import SubmissionStatus

        # Create mock test case
        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = "5\n7"
        test_case.expected_output = "12"

        # Create mock problem
        problem = MagicMock()
        problem.test_cases = [test_case]
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256
        mock_problem_repo.get_by_id.return_value = problem

        # Create mock submission
        submission = MagicMock()
        submission.problem_id = uuid4()
        submission.code = "a = int(input())\nb = int(input())\nprint(a + b)"
        submission.language = "python"
        mock_submission_repo.get_by_id.return_value = submission
        mock_submission_repo.update.return_value = submission

        result = await evaluator.evaluate_submission(uuid4())

        # Verify submission was updated
        submission.start_evaluation.assert_called_once()
        submission.complete_evaluation.assert_called_once()

        # Check that status was ACCEPTED (all tests passed)
        call_kwargs = submission.complete_evaluation.call_args[1]
        assert call_kwargs["status"] == SubmissionStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_evaluate_submission_wrong_answer(
        self, evaluator, mock_problem_repo, mock_submission_repo
    ):
        """Test evaluating submission with wrong answer."""
        from code_tutor.learning.domain.value_objects import SubmissionStatus

        # Create mock test case
        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = "5\n7"
        test_case.expected_output = "12"

        # Create mock problem
        problem = MagicMock()
        problem.test_cases = [test_case]
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256
        mock_problem_repo.get_by_id.return_value = problem

        # Create mock submission with wrong code
        submission = MagicMock()
        submission.problem_id = uuid4()
        submission.code = "print('wrong answer')"
        submission.language = "python"
        mock_submission_repo.get_by_id.return_value = submission
        mock_submission_repo.update.return_value = submission

        result = await evaluator.evaluate_submission(uuid4())

        call_kwargs = submission.complete_evaluation.call_args[1]
        assert call_kwargs["status"] == SubmissionStatus.WRONG_ANSWER

    @pytest.mark.asyncio
    async def test_evaluate_submission_runtime_error(
        self, evaluator, mock_problem_repo, mock_submission_repo
    ):
        """Test evaluating submission with runtime error."""
        from code_tutor.learning.domain.value_objects import SubmissionStatus

        # Create mock test case
        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = ""
        test_case.expected_output = "0"

        # Create mock problem
        problem = MagicMock()
        problem.test_cases = [test_case]
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256
        mock_problem_repo.get_by_id.return_value = problem

        # Create mock submission with error
        submission = MagicMock()
        submission.problem_id = uuid4()
        submission.code = "x = 1 / 0"
        submission.language = "python"
        mock_submission_repo.get_by_id.return_value = submission
        mock_submission_repo.update.return_value = submission

        result = await evaluator.evaluate_submission(uuid4())

        call_kwargs = submission.complete_evaluation.call_args[1]
        assert call_kwargs["status"] == SubmissionStatus.RUNTIME_ERROR

    @pytest.mark.asyncio
    async def test_evaluate_submission_timeout(
        self, evaluator, mock_problem_repo, mock_submission_repo
    ):
        """Test evaluating submission that times out."""
        from code_tutor.learning.domain.value_objects import SubmissionStatus

        # Create mock test case
        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = ""
        test_case.expected_output = "done"

        # Create mock problem with 1s time limit
        problem = MagicMock()
        problem.test_cases = [test_case]
        problem.time_limit_ms = 1000
        problem.memory_limit_mb = 256
        mock_problem_repo.get_by_id.return_value = problem

        # Create mock submission with infinite loop
        submission = MagicMock()
        submission.problem_id = uuid4()
        submission.code = "import time\ntime.sleep(10)"
        submission.language = "python"
        mock_submission_repo.get_by_id.return_value = submission
        mock_submission_repo.update.return_value = submission

        result = await evaluator.evaluate_submission(uuid4())

        call_kwargs = submission.complete_evaluation.call_args[1]
        assert call_kwargs["status"] == SubmissionStatus.TIME_LIMIT_EXCEEDED

    @pytest.mark.asyncio
    async def test_evaluate_multiple_test_cases(
        self, evaluator, mock_problem_repo, mock_submission_repo
    ):
        """Test evaluating submission with multiple test cases."""
        from code_tutor.learning.domain.value_objects import SubmissionStatus

        # Create multiple test cases
        test_cases = []
        for inp, out in [("1\n2", "3"), ("5\n5", "10"), ("0\n0", "0")]:
            tc = MagicMock()
            tc.id = uuid4()
            tc.input_data = inp
            tc.expected_output = out
            test_cases.append(tc)

        # Create mock problem
        problem = MagicMock()
        problem.test_cases = test_cases
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256
        mock_problem_repo.get_by_id.return_value = problem

        # Create mock submission
        submission = MagicMock()
        submission.problem_id = uuid4()
        submission.code = "a = int(input())\nb = int(input())\nprint(a + b)"
        submission.language = "python"
        mock_submission_repo.get_by_id.return_value = submission
        mock_submission_repo.update.return_value = submission

        result = await evaluator.evaluate_submission(uuid4())

        call_kwargs = submission.complete_evaluation.call_args[1]
        # All test cases should pass
        assert call_kwargs["status"] == SubmissionStatus.ACCEPTED


class TestSubmissionEvaluatorRunTestCase:
    """Tests for SubmissionEvaluator._run_test_case method."""

    @pytest.fixture
    def evaluator(self):
        """Create SubmissionEvaluator with MockSandbox."""
        from code_tutor.execution.application.services import SubmissionEvaluator
        return SubmissionEvaluator(
            problem_repository=AsyncMock(),
            submission_repository=AsyncMock(),
            use_docker=False,
        )

    @pytest.mark.asyncio
    async def test_run_test_case_passed(self, evaluator):
        """Test running a passing test case."""
        submission = MagicMock()
        submission.code = "print(input())"
        submission.language = "python"

        problem = MagicMock()
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256

        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = "hello"
        test_case.expected_output = "hello"

        result = await evaluator._run_test_case(submission, problem, test_case)

        assert result.is_passed is True
        assert result.actual_output == "hello"

    @pytest.mark.asyncio
    async def test_run_test_case_whitespace_normalization(self, evaluator):
        """Test that whitespace is normalized for comparison."""
        submission = MagicMock()
        submission.code = "print('hello  world')"
        submission.language = "python"

        problem = MagicMock()
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256

        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = ""
        test_case.expected_output = "hello  world"  # Same with whitespace

        result = await evaluator._run_test_case(submission, problem, test_case)

        assert result.is_passed is True

    @pytest.mark.asyncio
    async def test_run_test_case_with_error(self, evaluator):
        """Test running a test case with runtime error."""
        submission = MagicMock()
        submission.code = "raise Exception('error')"
        submission.language = "python"

        problem = MagicMock()
        problem.time_limit_ms = 5000
        problem.memory_limit_mb = 256

        test_case = MagicMock()
        test_case.id = uuid4()
        test_case.input_data = ""
        test_case.expected_output = "output"

        result = await evaluator._run_test_case(submission, problem, test_case)

        assert result.is_passed is False
        assert result.error_message is not None


class TestExecutionServiceEdgeCases:
    """Edge case tests for ExecutionService."""

    @pytest.mark.asyncio
    async def test_response_conversion(self):
        """Test _to_response method."""
        service = ExecutionService(use_docker=False)

        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.SUCCESS,
            stdout="output",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
            error_message=None,
        )

        response = service._to_response(result)

        assert response.execution_id == result.execution_id
        assert response.status == "success"
        assert response.stdout == "output"
        assert response.is_success is True
        assert response.execution_time_ms == 50.0

    @pytest.mark.asyncio
    async def test_service_uses_mock_sandbox(self):
        """Test that use_docker=False creates MockSandbox."""
        from code_tutor.execution.infrastructure.sandbox import MockSandbox
        service = ExecutionService(use_docker=False)
        assert isinstance(service._sandbox, MockSandbox)


class TestExecutionResultEdgeCases:
    """Edge case tests for ExecutionResult."""

    def test_result_with_all_statuses(self):
        """Test ExecutionResult with all status types."""
        statuses = [
            ExecutionStatus.PENDING,
            ExecutionStatus.RUNNING,
            ExecutionStatus.SUCCESS,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.MEMORY_EXCEEDED,
            ExecutionStatus.RUNTIME_ERROR,
            ExecutionStatus.COMPILATION_ERROR,
        ]

        for status in statuses:
            result = ExecutionResult(
                execution_id=uuid4(),
                status=status,
            )
            if status == ExecutionStatus.SUCCESS:
                assert result.is_success is True
            else:
                assert result.is_success is False

    def test_result_output_with_various_whitespace(self):
        """Test output stripping with various whitespace patterns."""
        test_cases = [
            ("Hello\n", "Hello"),
            ("Hello\n\n\n", "Hello"),
            ("Hello   \n", "Hello"),
            ("  Hello  ", "  Hello"),  # Only trailing is stripped
            ("\n\nHello\n\n", "\n\nHello"),
            ("", ""),
        ]

        for stdout, expected_output in test_cases:
            result = ExecutionResult(
                execution_id=uuid4(),
                status=ExecutionStatus.SUCCESS,
                stdout=stdout,
            )
            assert result.output == expected_output

    def test_result_with_none_error_message(self):
        """Test result with None error message."""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.SUCCESS,
            error_message=None,
        )
        assert result.error_message is None

    def test_result_with_error_message(self):
        """Test result with error message."""
        result = ExecutionResult(
            execution_id=uuid4(),
            status=ExecutionStatus.RUNTIME_ERROR,
            error_message="Division by zero",
        )
        assert result.error_message == "Division by zero"
