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
