"""Learning domain entities"""

from datetime import datetime, timezone
from uuid import UUID

from code_tutor.learning.domain.events import (
    ProblemCreated,
    SubmissionCreated,
    SubmissionEvaluated,
)
from code_tutor.learning.domain.value_objects import (
    Category,
    Difficulty,
    SubmissionStatus,
    TestResult,
)
from code_tutor.shared.domain.base import AggregateRoot, Entity


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)


class TestCase(Entity):
    """Test case entity"""

    def __init__(
        self,
        id: UUID | None = None,
        problem_id: UUID | None = None,
        input_data: str = "",
        expected_output: str = "",
        is_sample: bool = False,
        order: int = 0,
    ) -> None:
        super().__init__(id)
        self._problem_id = problem_id
        self._input_data = input_data
        self._expected_output = expected_output
        self._is_sample = is_sample
        self._order = order

    @property
    def problem_id(self) -> UUID | None:
        return self._problem_id

    @property
    def input_data(self) -> str:
        return self._input_data

    @property
    def expected_output(self) -> str:
        return self._expected_output

    @property
    def is_sample(self) -> bool:
        return self._is_sample

    @property
    def order(self) -> int:
        return self._order


class Problem(AggregateRoot):
    """Problem aggregate root"""

    def __init__(
        self,
        id: UUID | None = None,
        title: str = "",
        description: str = "",
        difficulty: Difficulty = Difficulty.EASY,
        category: Category = Category.ARRAY,
        constraints: str = "",
        hints: list[str] | None = None,
        solution_template: str = "",
        reference_solution: str = "",
        time_limit_ms: int = 1000,
        memory_limit_mb: int = 256,
        is_published: bool = False,
        test_cases: list[TestCase] | None = None,
        # Pattern-related fields
        pattern_ids: list[str] | None = None,
        pattern_explanation: str = "",
        approach_hint: str = "",
        time_complexity_hint: str = "",
        space_complexity_hint: str = "",
    ) -> None:
        super().__init__(id)
        self._title = title
        self._description = description
        self._difficulty = difficulty
        self._category = category
        self._constraints = constraints
        self._hints = hints or []
        self._solution_template = solution_template
        self._reference_solution = reference_solution
        self._time_limit_ms = time_limit_ms
        self._memory_limit_mb = memory_limit_mb
        self._is_published = is_published
        self._test_cases = test_cases or []
        # Pattern-related fields
        self._pattern_ids = pattern_ids or []
        self._pattern_explanation = pattern_explanation
        self._approach_hint = approach_hint
        self._time_complexity_hint = time_complexity_hint
        self._space_complexity_hint = space_complexity_hint

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        difficulty: Difficulty,
        category: Category,
        constraints: str = "",
        hints: list[str] | None = None,
        solution_template: str = "",
        reference_solution: str = "",
        time_limit_ms: int = 1000,
        memory_limit_mb: int = 256,
        pattern_ids: list[str] | None = None,
        pattern_explanation: str = "",
        approach_hint: str = "",
        time_complexity_hint: str = "",
        space_complexity_hint: str = "",
    ) -> "Problem":
        """Factory method to create a new problem"""
        problem = cls(
            title=title,
            description=description,
            difficulty=difficulty,
            category=category,
            constraints=constraints,
            hints=hints,
            solution_template=solution_template,
            reference_solution=reference_solution,
            time_limit_ms=time_limit_ms,
            memory_limit_mb=memory_limit_mb,
            pattern_ids=pattern_ids,
            pattern_explanation=pattern_explanation,
            approach_hint=approach_hint,
            time_complexity_hint=time_complexity_hint,
            space_complexity_hint=space_complexity_hint,
        )

        problem.add_domain_event(
            ProblemCreated(
                problem_id=problem.id,
                title=title,
                category=category.value,
            )
        )

        return problem

    # Properties
    @property
    def title(self) -> str:
        return self._title

    @property
    def description(self) -> str:
        return self._description

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    @property
    def category(self) -> Category:
        return self._category

    @property
    def constraints(self) -> str:
        return self._constraints

    @property
    def hints(self) -> list[str]:
        return self._hints.copy()

    @property
    def solution_template(self) -> str:
        return self._solution_template

    @property
    def reference_solution(self) -> str:
        return self._reference_solution

    @property
    def time_limit_ms(self) -> int:
        return self._time_limit_ms

    @property
    def memory_limit_mb(self) -> int:
        return self._memory_limit_mb

    @property
    def is_published(self) -> bool:
        return self._is_published

    @property
    def test_cases(self) -> list[TestCase]:
        return self._test_cases.copy()

    @property
    def sample_test_cases(self) -> list[TestCase]:
        return [tc for tc in self._test_cases if tc.is_sample]

    @property
    def pattern_ids(self) -> list[str]:
        return self._pattern_ids.copy()

    @property
    def pattern_explanation(self) -> str:
        return self._pattern_explanation

    @property
    def approach_hint(self) -> str:
        return self._approach_hint

    @property
    def time_complexity_hint(self) -> str:
        return self._time_complexity_hint

    @property
    def space_complexity_hint(self) -> str:
        return self._space_complexity_hint

    # Behavior methods
    def add_test_case(
        self,
        input_data: str,
        expected_output: str,
        is_sample: bool = False,
    ) -> TestCase:
        """Add a test case to the problem"""
        order = len(self._test_cases)
        test_case = TestCase(
            problem_id=self.id,
            input_data=input_data,
            expected_output=expected_output,
            is_sample=is_sample,
            order=order,
        )
        self._test_cases.append(test_case)
        self._touch()
        return test_case

    def publish(self) -> None:
        """Publish the problem"""
        if not self._test_cases:
            from code_tutor.shared.exceptions import DomainError

            raise DomainError("Problem must have at least one test case")
        self._is_published = True
        self._touch()

    def unpublish(self) -> None:
        """Unpublish the problem"""
        self._is_published = False
        self._touch()

    def update_content(
        self,
        title: str | None = None,
        description: str | None = None,
        constraints: str | None = None,
        hints: list[str] | None = None,
    ) -> None:
        """Update problem content"""
        if title is not None:
            self._title = title
        if description is not None:
            self._description = description
        if constraints is not None:
            self._constraints = constraints
        if hints is not None:
            self._hints = hints
        self._touch()


class Submission(AggregateRoot):
    """Submission aggregate root"""

    def __init__(
        self,
        id: UUID | None = None,
        user_id: UUID | None = None,
        problem_id: UUID | None = None,
        code: str = "",
        language: str = "python",
        status: SubmissionStatus = SubmissionStatus.PENDING,
        test_results: list[TestResult] | None = None,
        total_tests: int = 0,
        passed_tests: int = 0,
        execution_time_ms: float = 0.0,
        memory_usage_mb: float = 0.0,
        error_message: str | None = None,
        submitted_at: datetime | None = None,
        evaluated_at: datetime | None = None,
    ) -> None:
        super().__init__(id)
        self._user_id = user_id
        self._problem_id = problem_id
        self._code = code
        self._language = language
        self._status = status
        self._test_results = test_results or []
        self._total_tests = total_tests
        self._passed_tests = passed_tests
        self._execution_time_ms = execution_time_ms
        self._memory_usage_mb = memory_usage_mb
        self._error_message = error_message
        self._submitted_at = submitted_at or utc_now()
        self._evaluated_at = evaluated_at

    @classmethod
    def create(
        cls,
        user_id: UUID,
        problem_id: UUID,
        code: str,
        language: str = "python",
    ) -> "Submission":
        """Factory method to create a new submission"""
        submission = cls(
            user_id=user_id,
            problem_id=problem_id,
            code=code,
            language=language,
            status=SubmissionStatus.PENDING,
        )

        submission.add_domain_event(
            SubmissionCreated(
                submission_id=submission.id,
                user_id=user_id,
                problem_id=problem_id,
            )
        )

        return submission

    # Properties
    @property
    def user_id(self) -> UUID | None:
        return self._user_id

    @property
    def problem_id(self) -> UUID | None:
        return self._problem_id

    @property
    def code(self) -> str:
        return self._code

    @property
    def language(self) -> str:
        return self._language

    @property
    def status(self) -> SubmissionStatus:
        return self._status

    @property
    def test_results(self) -> list[TestResult]:
        return self._test_results.copy()

    @property
    def total_tests(self) -> int:
        return self._total_tests

    @property
    def passed_tests(self) -> int:
        return self._passed_tests

    @property
    def execution_time_ms(self) -> float:
        return self._execution_time_ms

    @property
    def memory_usage_mb(self) -> float:
        return self._memory_usage_mb

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def submitted_at(self) -> datetime:
        return self._submitted_at

    @property
    def evaluated_at(self) -> datetime | None:
        return self._evaluated_at

    @property
    def is_accepted(self) -> bool:
        return self._status == SubmissionStatus.ACCEPTED

    # Behavior methods
    def start_evaluation(self) -> None:
        """Mark submission as running"""
        self._status = SubmissionStatus.RUNNING
        self._touch()

    def complete_evaluation(
        self,
        status: SubmissionStatus,
        test_results: list[TestResult],
        execution_time_ms: float,
        memory_usage_mb: float,
        error_message: str | None = None,
    ) -> None:
        """Complete the evaluation with results"""
        self._status = status
        self._test_results = test_results
        self._total_tests = len(test_results)
        self._passed_tests = sum(1 for r in test_results if r.is_passed)
        self._execution_time_ms = execution_time_ms
        self._memory_usage_mb = memory_usage_mb
        self._error_message = error_message
        self._evaluated_at = utc_now()
        self._touch()

        self.add_domain_event(
            SubmissionEvaluated(
                submission_id=self.id,
                status=status.value,
                passed_tests=self._passed_tests,
                total_tests=self._total_tests,
            )
        )
