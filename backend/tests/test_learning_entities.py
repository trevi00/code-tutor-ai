"""Unit tests for Learning Domain Entities"""

import pytest
from uuid import uuid4
from datetime import datetime

from code_tutor.learning.domain.entities import Problem, Submission, TestCase
from code_tutor.learning.domain.value_objects import (
    Category,
    Difficulty,
    SubmissionStatus,
    TestResult,
)
from code_tutor.shared.exceptions import DomainError


class TestProblem:
    """Tests for Problem entity"""

    def test_create_problem(self):
        """Test creating a problem"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers that add up to target",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        assert problem.title == "Two Sum"
        assert problem.difficulty == Difficulty.EASY
        assert problem.category == Category.ARRAY
        assert problem.is_published is False

    def test_problem_publish(self):
        """Test publishing a problem"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        problem.add_test_case(
            input_data="[1,2,3]",
            expected_output="6",
            is_sample=True,
        )
        problem.publish()
        assert problem.is_published is True

    def test_problem_publish_without_test_cases(self):
        """Test publishing fails without test cases"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        with pytest.raises(DomainError):
            problem.publish()

    def test_problem_add_test_case(self):
        """Test adding test case to problem"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        problem.add_test_case(
            input_data="[1,2,3]",
            expected_output="6",
            is_sample=True,
        )
        assert len(problem.test_cases) == 1
        assert problem.test_cases[0].input_data == "[1,2,3]"

    def test_problem_sample_test_cases(self):
        """Test filtering sample test cases"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        problem.add_test_case(input_data="1", expected_output="1", is_sample=True)
        problem.add_test_case(input_data="2", expected_output="2", is_sample=False)
        problem.add_test_case(input_data="3", expected_output="3", is_sample=True)

        assert len(problem.test_cases) == 3
        assert len(problem.sample_test_cases) == 2

    def test_problem_update_content(self):
        """Test updating problem content"""
        problem = Problem.create(
            title="Original Title",
            description="Original description",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        problem.update_content(
            title="Updated Title",
            description="Updated description",
        )
        assert problem.title == "Updated Title"
        assert problem.description == "Updated description"

    def test_problem_unpublish(self):
        """Test unpublishing a problem"""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
        )
        problem.add_test_case(input_data="1", expected_output="1", is_sample=True)
        problem.publish()
        assert problem.is_published is True

        problem.unpublish()
        assert problem.is_published is False

    def test_problem_with_hints(self):
        """Test problem with hints"""
        hints = ["Use a hash map", "Consider O(n) solution"]
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            hints=hints,
        )
        assert len(problem.hints) == 2
        assert "hash map" in problem.hints[0]


class TestTestCase:
    """Tests for TestCase entity"""

    def test_create_test_case(self):
        """Test creating a test case"""
        problem_id = uuid4()
        test_case = TestCase(
            problem_id=problem_id,
            input_data="[1, 2, 3]",
            expected_output="6",
            is_sample=True,
            order=0,
        )
        assert test_case.input_data == "[1, 2, 3]"
        assert test_case.expected_output == "6"
        assert test_case.is_sample is True

    def test_test_case_order(self):
        """Test test case ordering"""
        test_case1 = TestCase(input_data="1", expected_output="1", order=0)
        test_case2 = TestCase(input_data="2", expected_output="2", order=1)
        assert test_case1.order < test_case2.order


class TestSubmission:
    """Tests for Submission entity"""

    def test_create_submission(self):
        """Test creating a submission"""
        problem_id = uuid4()
        user_id = uuid4()

        submission = Submission.create(
            problem_id=problem_id,
            user_id=user_id,
            code="def solution(nums): pass",
            language="python",
        )

        assert submission.problem_id == problem_id
        assert submission.user_id == user_id
        assert submission.status == SubmissionStatus.PENDING
        assert submission.code == "def solution(nums): pass"

    def test_submission_start_evaluation(self):
        """Test starting submission evaluation"""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solution(): pass",
            language="python",
        )
        submission.start_evaluation()
        assert submission.status == SubmissionStatus.RUNNING

    def test_submission_complete_accepted(self):
        """Test completing submission as accepted"""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solution(): pass",
            language="python",
        )
        submission.start_evaluation()

        test_results = [
            TestResult(
                test_case_id=uuid4(),
                input_data="1",
                expected_output="1",
                actual_output="1",
                is_passed=True,
                execution_time_ms=50.0,
            )
        ]

        submission.complete_evaluation(
            status=SubmissionStatus.ACCEPTED,
            test_results=test_results,
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
        )

        assert submission.status == SubmissionStatus.ACCEPTED
        assert submission.passed_tests == 1
        assert submission.total_tests == 1
        assert submission.is_accepted is True

    def test_submission_complete_wrong_answer(self):
        """Test completing submission with wrong answer"""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solution(): return 0",
            language="python",
        )
        submission.start_evaluation()

        test_results = [
            TestResult(
                test_case_id=uuid4(),
                input_data="1",
                expected_output="1",
                actual_output="0",
                is_passed=False,
                execution_time_ms=50.0,
            )
        ]

        submission.complete_evaluation(
            status=SubmissionStatus.WRONG_ANSWER,
            test_results=test_results,
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
        )

        assert submission.status == SubmissionStatus.WRONG_ANSWER
        assert submission.passed_tests == 0
        assert submission.is_accepted is False


class TestTestResult:
    """Tests for TestResult value object"""

    def test_passed_test_result(self):
        """Test passed test result"""
        result = TestResult(
            test_case_id=uuid4(),
            input_data="[1, 2, 3]",
            expected_output="6",
            actual_output="6",
            is_passed=True,
            execution_time_ms=25.0,
        )
        assert result.is_passed is True
        assert result.expected_output == result.actual_output

    def test_failed_test_result(self):
        """Test failed test result"""
        result = TestResult(
            test_case_id=uuid4(),
            input_data="[1, 2, 3]",
            expected_output="6",
            actual_output="5",
            is_passed=False,
            execution_time_ms=25.0,
            error_message=None,
        )
        assert result.is_passed is False
        assert result.expected_output != result.actual_output

    def test_error_test_result(self):
        """Test result with error"""
        result = TestResult(
            test_case_id=uuid4(),
            input_data="[1, 2, 3]",
            expected_output="6",
            actual_output="",
            is_passed=False,
            execution_time_ms=0.0,
            error_message="ZeroDivisionError",
        )
        assert result.is_passed is False
        assert result.error_message is not None


class TestSubmissionStatus:
    """Tests for SubmissionStatus enum"""

    def test_all_status_values(self):
        """Test all submission status values exist"""
        assert SubmissionStatus.PENDING.value == "pending"
        assert SubmissionStatus.RUNNING.value == "running"
        assert SubmissionStatus.ACCEPTED.value == "accepted"
        assert SubmissionStatus.WRONG_ANSWER.value == "wrong_answer"
        assert SubmissionStatus.TIME_LIMIT_EXCEEDED.value == "time_limit_exceeded"
        assert SubmissionStatus.MEMORY_LIMIT_EXCEEDED.value == "memory_limit_exceeded"
        assert SubmissionStatus.RUNTIME_ERROR.value == "runtime_error"
        assert SubmissionStatus.COMPILATION_ERROR.value == "compilation_error"


# ============== Additional Coverage Tests ==============


class TestTestCaseProperties:
    """Additional tests for TestCase properties."""

    def test_problem_id_property(self):
        """Test TestCase.problem_id property access."""
        problem_id = uuid4()
        test_case = TestCase(
            problem_id=problem_id,
            input_data="test",
            expected_output="result",
        )
        assert test_case.problem_id == problem_id

    def test_problem_id_none(self):
        """Test TestCase.problem_id when None."""
        test_case = TestCase(
            input_data="test",
            expected_output="result",
        )
        assert test_case.problem_id is None


class TestProblemUpdateContentBranches:
    """Tests for Problem.update_content method branches."""

    def test_update_content_constraints_only(self):
        """Test updating only constraints."""
        problem = Problem.create(
            title="Original",
            description="Original desc",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            constraints="Old constraints",
        )
        original_title = problem.title

        problem.update_content(constraints="New constraints")

        assert problem.title == original_title  # Unchanged
        assert problem.constraints == "New constraints"

    def test_update_content_hints_only(self):
        """Test updating only hints."""
        problem = Problem.create(
            title="Original",
            description="Original desc",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            hints=["Old hint"],
        )
        original_title = problem.title

        problem.update_content(hints=["New hint 1", "New hint 2"])

        assert problem.title == original_title  # Unchanged
        assert problem.hints == ["New hint 1", "New hint 2"]

    def test_update_content_none_values(self):
        """Test update_content with all None values."""
        problem = Problem.create(
            title="Original",
            description="Original desc",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            constraints="Original constraints",
            hints=["Original hint"],
        )

        problem.update_content(
            title=None,
            description=None,
            constraints=None,
            hints=None,
        )

        # All should remain unchanged
        assert problem.title == "Original"
        assert problem.description == "Original desc"
        assert problem.constraints == "Original constraints"
        assert problem.hints == ["Original hint"]


class TestSubmissionProperties:
    """Tests for Submission properties coverage."""

    def test_submission_language_property(self):
        """Test Submission.language property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="print('hello')",
            language="python",
        )
        assert submission.language == "python"

    def test_submission_test_results_property(self):
        """Test Submission.test_results property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): pass",
        )
        submission.start_evaluation()

        test_results = [
            TestResult(
                test_case_id=uuid4(),
                input_data="1",
                expected_output="1",
                actual_output="1",
                is_passed=True,
                execution_time_ms=10.0,
            ),
            TestResult(
                test_case_id=uuid4(),
                input_data="2",
                expected_output="2",
                actual_output="2",
                is_passed=True,
                execution_time_ms=15.0,
            ),
        ]

        submission.complete_evaluation(
            status=SubmissionStatus.ACCEPTED,
            test_results=test_results,
            execution_time_ms=25.0,
            memory_usage_mb=8.0,
        )

        assert len(submission.test_results) == 2
        assert submission.test_results[0].is_passed is True

    def test_submission_execution_time_property(self):
        """Test Submission.execution_time_ms property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): pass",
        )
        submission.start_evaluation()
        submission.complete_evaluation(
            status=SubmissionStatus.ACCEPTED,
            test_results=[],
            execution_time_ms=123.45,
            memory_usage_mb=10.0,
        )

        assert submission.execution_time_ms == 123.45

    def test_submission_memory_usage_property(self):
        """Test Submission.memory_usage_mb property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): pass",
        )
        submission.start_evaluation()
        submission.complete_evaluation(
            status=SubmissionStatus.ACCEPTED,
            test_results=[],
            execution_time_ms=50.0,
            memory_usage_mb=32.5,
        )

        assert submission.memory_usage_mb == 32.5

    def test_submission_error_message_property(self):
        """Test Submission.error_message property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): 1/0",
        )
        submission.start_evaluation()
        submission.complete_evaluation(
            status=SubmissionStatus.RUNTIME_ERROR,
            test_results=[],
            execution_time_ms=10.0,
            memory_usage_mb=5.0,
            error_message="ZeroDivisionError: division by zero",
        )

        assert submission.error_message == "ZeroDivisionError: division by zero"

    def test_submission_error_message_none(self):
        """Test Submission.error_message when None."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): return 1",
        )
        assert submission.error_message is None

    def test_submission_submitted_at_property(self):
        """Test Submission.submitted_at property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): pass",
        )

        assert submission.submitted_at is not None
        assert isinstance(submission.submitted_at, datetime)

    def test_submission_evaluated_at_property(self):
        """Test Submission.evaluated_at property."""
        submission = Submission.create(
            problem_id=uuid4(),
            user_id=uuid4(),
            code="def solve(): pass",
        )

        # Before evaluation
        assert submission.evaluated_at is None

        # After evaluation
        submission.start_evaluation()
        submission.complete_evaluation(
            status=SubmissionStatus.ACCEPTED,
            test_results=[],
            execution_time_ms=50.0,
            memory_usage_mb=10.0,
        )

        assert submission.evaluated_at is not None
        assert isinstance(submission.evaluated_at, datetime)


class TestProblemPatternProperties:
    """Tests for Problem pattern-related properties."""

    def test_problem_pattern_ids(self):
        """Test Problem.pattern_ids property."""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            pattern_ids=["two-pointers", "hash-map"],
        )
        assert problem.pattern_ids == ["two-pointers", "hash-map"]

    def test_problem_pattern_explanation(self):
        """Test Problem.pattern_explanation property."""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            pattern_explanation="Use a hash map for O(n) solution",
        )
        assert problem.pattern_explanation == "Use a hash map for O(n) solution"

    def test_problem_approach_hint(self):
        """Test Problem.approach_hint property."""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            approach_hint="Consider using complement",
        )
        assert problem.approach_hint == "Consider using complement"

    def test_problem_complexity_hints(self):
        """Test Problem complexity hint properties."""
        problem = Problem.create(
            title="Two Sum",
            description="Find two numbers",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            time_complexity_hint="O(n)",
            space_complexity_hint="O(n)",
        )
        assert problem.time_complexity_hint == "O(n)"
        assert problem.space_complexity_hint == "O(n)"


# ============== Value Objects Coverage Tests ==============


class TestProblemIdValueObject:
    """Tests for ProblemId value object."""

    def test_generate_problem_id(self):
        """Test generating a new ProblemId."""
        from code_tutor.learning.domain.value_objects import ProblemId

        problem_id = ProblemId.generate()
        assert problem_id.value is not None

    def test_problem_id_from_string(self):
        """Test creating ProblemId from string."""
        from code_tutor.learning.domain.value_objects import ProblemId

        uuid_str = "12345678-1234-5678-1234-567812345678"
        problem_id = ProblemId.from_string(uuid_str)
        assert str(problem_id) == uuid_str

    def test_problem_id_from_string_invalid(self):
        """Test creating ProblemId from invalid string."""
        from code_tutor.learning.domain.value_objects import ProblemId
        from code_tutor.shared.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ProblemId.from_string("not-a-valid-uuid")

        assert "Invalid problem ID format" in str(exc_info.value)

    def test_problem_id_str(self):
        """Test ProblemId string representation."""
        from code_tutor.learning.domain.value_objects import ProblemId

        problem_id = ProblemId.generate()
        str_repr = str(problem_id)
        assert len(str_repr) == 36  # UUID format


class TestSubmissionIdValueObject:
    """Tests for SubmissionId value object."""

    def test_generate_submission_id(self):
        """Test generating a new SubmissionId."""
        from code_tutor.learning.domain.value_objects import SubmissionId

        submission_id = SubmissionId.generate()
        assert submission_id.value is not None

    def test_submission_id_from_string(self):
        """Test creating SubmissionId from string."""
        from code_tutor.learning.domain.value_objects import SubmissionId

        uuid_str = "abcdef12-1234-5678-1234-567812345678"
        submission_id = SubmissionId.from_string(uuid_str)
        assert str(submission_id) == uuid_str

    def test_submission_id_from_string_invalid(self):
        """Test creating SubmissionId from invalid string."""
        from code_tutor.learning.domain.value_objects import SubmissionId
        from code_tutor.shared.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SubmissionId.from_string("invalid-uuid")

        assert "Invalid submission ID format" in str(exc_info.value)

    def test_submission_id_str(self):
        """Test SubmissionId string representation."""
        from code_tutor.learning.domain.value_objects import SubmissionId

        submission_id = SubmissionId.generate()
        str_repr = str(submission_id)
        assert len(str_repr) == 36  # UUID format


class TestExecutionResult:
    """Tests for ExecutionResult value object."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        from code_tutor.learning.domain.value_objects import ExecutionResult

        result = ExecutionResult(
            output="6",
            error=None,
            execution_time_ms=25.0,
            memory_usage_mb=10.0,
            is_correct=True,
        )

        assert result.output == "6"
        assert result.error is None
        assert result.is_correct is True

    def test_execution_result_error(self):
        """Test execution result with error."""
        from code_tutor.learning.domain.value_objects import ExecutionResult

        result = ExecutionResult(
            output="",
            error="RuntimeError: index out of range",
            execution_time_ms=5.0,
            memory_usage_mb=8.0,
            is_correct=False,
        )

        assert result.error is not None
        assert result.is_correct is False
