"""Code Execution application services"""

from uuid import UUID

from code_tutor.execution.application.dto import ExecuteCodeRequest, ExecuteCodeResponse
from code_tutor.execution.domain.value_objects import ExecutionRequest, ExecutionResult
from code_tutor.execution.infrastructure.sandbox import DockerSandbox, MockSandbox
from code_tutor.learning.domain.entities import Problem, Submission, TestCase
from code_tutor.learning.domain.repository import (
    ProblemRepository,
    SubmissionRepository,
)
from code_tutor.learning.domain.value_objects import SubmissionStatus, TestResult
from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import NotFoundError
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


class SubmissionEvaluator:
    """Service to evaluate submissions against test cases"""

    def __init__(
        self,
        problem_repository: ProblemRepository,
        submission_repository: SubmissionRepository,
        use_docker: bool = True,
    ) -> None:
        self._problem_repo = problem_repository
        self._submission_repo = submission_repository
        self._settings = get_settings()
        self._sandbox = DockerSandbox() if use_docker else MockSandbox()

    async def evaluate_submission(self, submission_id: UUID) -> Submission:
        """Evaluate a submission against all test cases"""
        # Get submission
        submission = await self._submission_repo.get_by_id(submission_id)
        if submission is None:
            raise NotFoundError("Submission", str(submission_id))

        # Get problem with test cases
        problem = await self._problem_repo.get_by_id(submission.problem_id)
        if problem is None:
            raise NotFoundError("Problem", str(submission.problem_id))

        # Mark as running
        submission.start_evaluation()
        await self._submission_repo.update(submission)

        # Run against each test case
        test_results: list[TestResult] = []
        total_time = 0.0
        max_memory = 0.0

        for test_case in problem.test_cases:
            result = await self._run_test_case(submission, problem, test_case)
            test_results.append(result)
            total_time += result.execution_time_ms
            # Note: memory tracking would need actual Docker stats

        # Determine final status
        all_passed = all(r.is_passed for r in test_results)
        has_timeout = any(
            r.error_message
            and (
                "timeout" in r.error_message.lower()
                or "time limit" in r.error_message.lower()
            )
            for r in test_results
        )
        has_memory_error = any(
            r.error_message and "memory" in r.error_message.lower()
            for r in test_results
        )

        if all_passed:
            status = SubmissionStatus.ACCEPTED
        elif has_timeout:
            status = SubmissionStatus.TIME_LIMIT_EXCEEDED
        elif has_memory_error:
            status = SubmissionStatus.MEMORY_LIMIT_EXCEEDED
        else:
            # Check if it's a runtime error or wrong answer
            has_runtime_error = any(
                r.error_message for r in test_results if not r.is_passed
            )
            status = (
                SubmissionStatus.RUNTIME_ERROR
                if has_runtime_error
                else SubmissionStatus.WRONG_ANSWER
            )

        # Complete evaluation
        submission.complete_evaluation(
            status=status,
            test_results=test_results,
            execution_time_ms=total_time,
            memory_usage_mb=max_memory,
            error_message=test_results[0].error_message
            if not all_passed and test_results
            else None,
        )

        saved = await self._submission_repo.update(submission)

        logger.info(
            "Submission evaluated",
            submission_id=str(submission_id),
            status=status.value,
            passed=sum(1 for r in test_results if r.is_passed),
            total=len(test_results),
        )

        return saved

    async def _run_test_case(
        self,
        submission: Submission,
        problem: Problem,
        test_case: TestCase,
    ) -> TestResult:
        """Run code against a single test case"""
        request = ExecutionRequest(
            code=submission.code,
            language=submission.language,
            stdin=test_case.input_data,
            timeout_seconds=problem.time_limit_ms // 1000 or 5,
            memory_limit_mb=problem.memory_limit_mb,
            cpu_limit=self._settings.SANDBOX_CPU_LIMIT,
        )

        result = await self._sandbox.execute(request)

        # Compare output (normalize whitespace for comparison)
        actual_output = result.output.strip()
        expected_output = test_case.expected_output.strip()

        # Normalize: remove all whitespace for comparison
        actual_normalized = "".join(actual_output.split())
        expected_normalized = "".join(expected_output.split())
        is_passed = result.is_success and actual_normalized == expected_normalized

        return TestResult(
            test_case_id=test_case.id,
            input_data=test_case.input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            is_passed=is_passed,
            execution_time_ms=result.execution_time_ms,
            error_message=result.error_message if not result.is_success else None,
        )
