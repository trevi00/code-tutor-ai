"""Learning application services (use cases)"""

from uuid import UUID

from code_tutor.learning.application.dto import (
    CreateProblemRequest,
    CreateSubmissionRequest,
    HintsResponse,
    ProblemFilterParams,
    ProblemListResponse,
    ProblemResponse,
    ProblemSummaryResponse,
    RecommendedProblemResponse,
    SubmissionResponse,
    SubmissionSummaryResponse,
    TestCaseResponse,
    TestResultResponse,
)
from code_tutor.learning.domain.entities import Problem, Submission
from code_tutor.learning.domain.repository import ProblemRepository, SubmissionRepository
from code_tutor.shared.exceptions import NotFoundError
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ProblemService:
    """Problem management service"""

    def __init__(self, problem_repository: ProblemRepository) -> None:
        self._problem_repo = problem_repository

    async def create_problem(self, request: CreateProblemRequest) -> ProblemResponse:
        """Create a new problem"""
        problem = Problem.create(
            title=request.title,
            description=request.description,
            difficulty=request.difficulty,
            category=request.category,
            constraints=request.constraints,
            hints=request.hints,
            solution_template=request.solution_template,
            reference_solution=request.reference_solution,
            time_limit_ms=request.time_limit_ms,
            memory_limit_mb=request.memory_limit_mb,
            pattern_ids=request.pattern_ids,
            pattern_explanation=request.pattern_explanation,
            approach_hint=request.approach_hint,
            time_complexity_hint=request.time_complexity_hint,
            space_complexity_hint=request.space_complexity_hint,
        )

        # Add test cases
        for tc_request in request.test_cases:
            problem.add_test_case(
                input_data=tc_request.input_data,
                expected_output=tc_request.expected_output,
                is_sample=tc_request.is_sample,
            )

        saved_problem = await self._problem_repo.add(problem)

        logger.info("Problem created", problem_id=str(saved_problem.id), title=request.title)

        return self._to_response(saved_problem)

    async def get_problem(self, problem_id: UUID) -> ProblemResponse:
        """Get problem by ID"""
        problem = await self._problem_repo.get_by_id(problem_id)
        if problem is None:
            raise NotFoundError("Problem", str(problem_id))
        return self._to_response(problem)

    async def list_problems(self, params: ProblemFilterParams) -> ProblemListResponse:
        """List problems with filters and pagination"""
        offset = (params.page - 1) * params.size

        problems = await self._problem_repo.get_published(
            category=params.category,
            difficulty=params.difficulty,
            pattern_id=params.pattern_id,
            limit=params.size,
            offset=offset,
        )

        total = await self._problem_repo.count_published(
            category=params.category,
            difficulty=params.difficulty,
            pattern_id=params.pattern_id,
        )

        pages = (total + params.size - 1) // params.size

        return ProblemListResponse(
            items=[self._to_summary(p) for p in problems],
            total=total,
            page=params.page,
            size=params.size,
            pages=pages,
        )

    async def publish_problem(self, problem_id: UUID) -> ProblemResponse:
        """Publish a problem"""
        problem = await self._problem_repo.get_by_id(problem_id)
        if problem is None:
            raise NotFoundError("Problem", str(problem_id))

        problem.publish()
        updated = await self._problem_repo.update(problem)

        logger.info("Problem published", problem_id=str(problem_id))

        return self._to_response(updated)

    async def delete_problem(self, problem_id: UUID) -> bool:
        """Delete a problem"""
        deleted = await self._problem_repo.delete(problem_id)
        if deleted:
            logger.info("Problem deleted", problem_id=str(problem_id))
        return deleted

    async def get_hints(self, problem_id: UUID, hint_index: int | None = None) -> HintsResponse:
        """Get hints for a problem"""
        problem = await self._problem_repo.get_by_id(problem_id)
        if problem is None:
            raise NotFoundError("Problem", str(problem_id))

        hints = problem.hints
        if hint_index is not None:
            # Return hints up to the specified index (progressive reveal)
            hints = hints[:min(hint_index + 1, len(hints))]

        return HintsResponse(
            problem_id=problem_id,
            hints=hints,
            total_hints=len(problem.hints),
        )

    async def get_recommended_problems(
        self,
        user_id: UUID,
        limit: int = 5,
    ) -> list[RecommendedProblemResponse]:
        """Get recommended problems for a user based on their history"""
        # Get all published problems
        all_problems = await self._problem_repo.get_published(limit=100, offset=0)

        recommendations = []
        for problem in all_problems[:limit]:
            # Simple recommendation logic - can be enhanced with ML later
            reason = self._get_recommendation_reason(problem)
            recommendations.append(
                RecommendedProblemResponse(
                    id=problem.id,
                    title=problem.title,
                    difficulty=problem.difficulty.value,
                    category=problem.category.value,
                    reason=reason,
                )
            )

        return recommendations

    def _get_recommendation_reason(self, problem: Problem) -> str:
        """Generate recommendation reason for a problem"""
        reasons = {
            "easy": "Good for practice and building confidence",
            "medium": "Great for skill development",
            "hard": "Challenge yourself with this problem",
        }
        return reasons.get(problem.difficulty.value, "Recommended for you")

    def _to_response(self, problem: Problem) -> ProblemResponse:
        """Convert Problem entity to ProblemResponse"""
        return ProblemResponse(
            id=problem.id,
            title=problem.title,
            description=problem.description,
            difficulty=problem.difficulty.value,
            category=problem.category.value,
            constraints=problem.constraints,
            hints=problem.hints,
            solution_template=problem.solution_template,
            time_limit_ms=problem.time_limit_ms,
            memory_limit_mb=problem.memory_limit_mb,
            is_published=problem.is_published,
            test_cases=[
                TestCaseResponse(
                    id=tc.id,
                    input_data=tc.input_data,
                    expected_output=tc.expected_output,
                    is_sample=tc.is_sample,
                    order=tc.order,
                )
                for tc in problem.sample_test_cases  # Only return sample test cases
            ],
            pattern_ids=problem.pattern_ids,
            pattern_explanation=problem.pattern_explanation,
            approach_hint=problem.approach_hint,
            time_complexity_hint=problem.time_complexity_hint,
            space_complexity_hint=problem.space_complexity_hint,
            created_at=problem.created_at,
            updated_at=problem.updated_at,
        )

    def _to_summary(self, problem: Problem) -> ProblemSummaryResponse:
        """Convert Problem entity to ProblemSummaryResponse"""
        return ProblemSummaryResponse(
            id=problem.id,
            title=problem.title,
            difficulty=problem.difficulty.value,
            category=problem.category.value,
            is_published=problem.is_published,
            pattern_ids=problem.pattern_ids,
            created_at=problem.created_at,
        )


class SubmissionService:
    """Submission management service"""

    def __init__(
        self,
        submission_repository: SubmissionRepository,
        problem_repository: ProblemRepository,
    ) -> None:
        self._submission_repo = submission_repository
        self._problem_repo = problem_repository

    async def create_submission(
        self,
        user_id: UUID,
        request: CreateSubmissionRequest,
    ) -> SubmissionResponse:
        """Create a new submission"""
        # Verify problem exists
        problem = await self._problem_repo.get_by_id(request.problem_id)
        if problem is None:
            raise NotFoundError("Problem", str(request.problem_id))

        # Create submission
        submission = Submission.create(
            user_id=user_id,
            problem_id=request.problem_id,
            code=request.code,
            language=request.language,
        )

        saved = await self._submission_repo.add(submission)

        logger.info(
            "Submission created",
            submission_id=str(saved.id),
            user_id=str(user_id),
            problem_id=str(request.problem_id),
        )

        return self._to_response(saved)

    async def get_submission(self, submission_id: UUID) -> SubmissionResponse:
        """Get submission by ID"""
        submission = await self._submission_repo.get_by_id(submission_id)
        if submission is None:
            raise NotFoundError("Submission", str(submission_id))
        return self._to_response(submission)

    async def get_user_submissions(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SubmissionSummaryResponse]:
        """Get user's submissions"""
        submissions = await self._submission_repo.get_by_user(user_id, limit, offset)
        return [self._to_summary(s) for s in submissions]

    async def get_user_problem_submissions(
        self,
        user_id: UUID,
        problem_id: UUID,
        limit: int = 10,
    ) -> list[SubmissionSummaryResponse]:
        """Get user's submissions for a specific problem"""
        submissions = await self._submission_repo.get_user_problem_submissions(
            user_id, problem_id, limit
        )
        return [self._to_summary(s) for s in submissions]

    async def has_user_solved(self, user_id: UUID, problem_id: UUID) -> bool:
        """Check if user has solved a problem"""
        return await self._submission_repo.has_user_solved(user_id, problem_id)

    def _to_response(self, submission: Submission) -> SubmissionResponse:
        """Convert Submission entity to SubmissionResponse"""
        return SubmissionResponse(
            id=submission.id,
            user_id=submission.user_id,
            problem_id=submission.problem_id,
            code=submission.code,
            language=submission.language,
            status=submission.status.value,
            test_results=[
                TestResultResponse(
                    test_case_id=tr.test_case_id,
                    input_data=tr.input_data,
                    expected_output=tr.expected_output,
                    actual_output=tr.actual_output,
                    is_passed=tr.is_passed,
                    execution_time_ms=tr.execution_time_ms,
                    error_message=tr.error_message,
                )
                for tr in submission.test_results
            ],
            total_tests=submission.total_tests,
            passed_tests=submission.passed_tests,
            execution_time_ms=submission.execution_time_ms,
            memory_usage_mb=submission.memory_usage_mb,
            error_message=submission.error_message,
            submitted_at=submission.submitted_at,
            evaluated_at=submission.evaluated_at,
        )

    def _to_summary(self, submission: Submission) -> SubmissionSummaryResponse:
        """Convert Submission entity to SubmissionSummaryResponse"""
        return SubmissionSummaryResponse(
            id=submission.id,
            problem_id=submission.problem_id,
            status=submission.status.value,
            passed_tests=submission.passed_tests,
            total_tests=submission.total_tests,
            submitted_at=submission.submitted_at,
        )
