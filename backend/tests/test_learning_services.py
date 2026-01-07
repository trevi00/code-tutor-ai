"""Unit tests for Learning Services"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from code_tutor.learning.application.services import ProblemService, SubmissionService
from code_tutor.learning.application.dto import (
    CreateProblemRequest,
    CreateSubmissionRequest,
    CreateTestCaseRequest,
    ProblemFilterParams,
)
from code_tutor.learning.domain.entities import Problem, TestCase
from code_tutor.learning.domain.value_objects import Category, Difficulty
from code_tutor.shared.exceptions import NotFoundError


@pytest.fixture
def mock_problem_repository():
    """Create mock problem repository"""
    return AsyncMock()


@pytest.fixture
def mock_submission_repository():
    """Create mock submission repository"""
    return AsyncMock()


@pytest.fixture
def problem_service(mock_problem_repository):
    """Create problem service with mock repository"""
    return ProblemService(mock_problem_repository)


@pytest.fixture
def submission_service(mock_submission_repository, mock_problem_repository):
    """Create submission service with mock repositories"""
    return SubmissionService(mock_submission_repository, mock_problem_repository)


@pytest.fixture
def sample_problem():
    """Create sample problem entity"""
    problem = Problem.create(
        title="Test Problem",
        description="Test description",
        difficulty=Difficulty.EASY,
        category=Category.ARRAY,
    )
    problem.add_test_case(
        input_data="[1,2,3]",
        expected_output="6",
        is_sample=True,
    )
    return problem


class TestProblemService:
    """Tests for ProblemService"""

    @pytest.mark.asyncio
    async def test_get_problem_success(self, problem_service, mock_problem_repository, sample_problem):
        """Test getting an existing problem"""
        mock_problem_repository.get_by_id.return_value = sample_problem

        result = await problem_service.get_problem(sample_problem.id)

        assert result.id == sample_problem.id
        assert result.title == sample_problem.title
        mock_problem_repository.get_by_id.assert_called_once_with(sample_problem.id)

    @pytest.mark.asyncio
    async def test_get_problem_not_found(self, problem_service, mock_problem_repository):
        """Test getting a non-existent problem"""
        mock_problem_repository.get_by_id.return_value = None
        problem_id = uuid4()

        with pytest.raises(NotFoundError):
            await problem_service.get_problem(problem_id)

    @pytest.mark.asyncio
    async def test_list_problems_empty(self, problem_service, mock_problem_repository):
        """Test listing problems when none exist"""
        mock_problem_repository.get_published.return_value = []
        mock_problem_repository.count_published.return_value = 0
        params = ProblemFilterParams(page=1, size=10)

        result = await problem_service.list_problems(params)

        assert result.items == []
        assert result.total == 0
        mock_problem_repository.get_published.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_problems_with_filters(self, problem_service, mock_problem_repository, sample_problem):
        """Test listing problems with category and difficulty filters"""
        mock_problem_repository.get_published.return_value = [sample_problem]
        mock_problem_repository.count_published.return_value = 1
        params = ProblemFilterParams(
            category=Category.ARRAY,
            difficulty=Difficulty.EASY,
            page=1,
            size=10,
        )

        result = await problem_service.list_problems(params)

        assert len(result.items) == 1
        assert result.total == 1
        mock_problem_repository.get_published.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_problem(self, problem_service, mock_problem_repository, sample_problem):
        """Test creating a new problem"""
        mock_problem_repository.add.return_value = sample_problem

        request = CreateProblemRequest(
            title="Test Problem",
            description="Test description",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            constraints="",
            hints=["hint1"],
            solution_template="def solve():\n    pass",
            time_limit_ms=1000,
            memory_limit_mb=256,
            test_cases=[
                CreateTestCaseRequest(
                    input_data="[1,2,3]",
                    expected_output="6",
                    is_sample=True,
                )
            ],
        )

        result = await problem_service.create_problem(request)

        assert result.title == sample_problem.title
        mock_problem_repository.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_problem_success(self, problem_service, mock_problem_repository, sample_problem):
        """Test publishing a problem"""
        mock_problem_repository.get_by_id.return_value = sample_problem
        mock_problem_repository.update.return_value = sample_problem

        result = await problem_service.publish_problem(sample_problem.id)

        assert result is not None
        mock_problem_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_problem_not_found(self, problem_service, mock_problem_repository):
        """Test publishing a non-existent problem"""
        mock_problem_repository.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await problem_service.publish_problem(uuid4())


class TestSubmissionService:
    """Tests for SubmissionService"""

    @pytest.mark.asyncio
    async def test_create_submission(self, submission_service, mock_submission_repository, mock_problem_repository, sample_problem):
        """Test creating a new submission"""
        mock_problem_repository.get_by_id.return_value = sample_problem

        # Create mock submission with empty list for test_results
        mock_submission = MagicMock()
        mock_submission.id = uuid4()
        mock_submission.user_id = uuid4()
        mock_submission.problem_id = sample_problem.id
        mock_submission.code = "def solve(): pass"
        mock_submission.language = "python"
        mock_submission.status = MagicMock(value="pending")
        mock_submission.test_results = []  # Empty list instead of None
        mock_submission.total_tests = 0
        mock_submission.passed_tests = 0
        mock_submission.execution_time_ms = 0
        mock_submission.memory_usage_mb = 0
        mock_submission.error_message = None
        mock_submission.submitted_at = MagicMock()
        mock_submission.evaluated_at = None

        mock_submission_repository.add.return_value = mock_submission

        user_id = uuid4()
        request = CreateSubmissionRequest(
            problem_id=sample_problem.id,
            code="def solve(): pass",
            language="python",
        )

        result = await submission_service.create_submission(user_id, request)

        assert result is not None
        mock_submission_repository.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_submission_problem_not_found(self, submission_service, mock_problem_repository):
        """Test creating submission for non-existent problem"""
        mock_problem_repository.get_by_id.return_value = None

        request = CreateSubmissionRequest(
            problem_id=uuid4(),
            code="def solve(): pass",
            language="python",
        )

        with pytest.raises(NotFoundError):
            await submission_service.create_submission(uuid4(), request)

    @pytest.mark.asyncio
    async def test_get_submission_success(self, submission_service, mock_submission_repository):
        """Test getting an existing submission"""
        mock_submission = MagicMock()
        mock_submission.id = uuid4()
        mock_submission.user_id = uuid4()
        mock_submission.problem_id = uuid4()
        mock_submission.code = "def solve(): pass"
        mock_submission.language = "python"
        mock_submission.status = MagicMock(value="accepted")
        mock_submission.test_results = []
        mock_submission.total_tests = 1
        mock_submission.passed_tests = 1
        mock_submission.execution_time_ms = 100
        mock_submission.memory_usage_mb = 10
        mock_submission.error_message = None
        mock_submission.submitted_at = MagicMock()
        mock_submission.evaluated_at = MagicMock()

        mock_submission_repository.get_by_id.return_value = mock_submission

        result = await submission_service.get_submission(mock_submission.id)

        assert result.id == mock_submission.id
        mock_submission_repository.get_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_submission_not_found(self, submission_service, mock_submission_repository):
        """Test getting a non-existent submission"""
        mock_submission_repository.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await submission_service.get_submission(uuid4())

    @pytest.mark.asyncio
    async def test_get_user_submissions(self, submission_service, mock_submission_repository):
        """Test getting user's submissions"""
        mock_submission_repository.get_by_user.return_value = []

        result = await submission_service.get_user_submissions(uuid4(), limit=10, offset=0)

        assert result == []
        mock_submission_repository.get_by_user.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_problem_submissions(self, submission_service, mock_submission_repository):
        """Test getting user's submissions for a specific problem"""
        mock_submission_repository.get_user_problem_submissions.return_value = []
        user_id = uuid4()
        problem_id = uuid4()

        result = await submission_service.get_user_problem_submissions(user_id, problem_id, limit=10)

        assert result == []
        mock_submission_repository.get_user_problem_submissions.assert_called_once_with(
            user_id, problem_id, 10
        )

    @pytest.mark.asyncio
    async def test_has_user_solved_true(self, submission_service, mock_submission_repository):
        """Test checking if user has solved a problem - true case"""
        mock_submission_repository.has_user_solved.return_value = True
        user_id = uuid4()
        problem_id = uuid4()

        result = await submission_service.has_user_solved(user_id, problem_id)

        assert result is True
        mock_submission_repository.has_user_solved.assert_called_once_with(user_id, problem_id)

    @pytest.mark.asyncio
    async def test_has_user_solved_false(self, submission_service, mock_submission_repository):
        """Test checking if user has solved a problem - false case"""
        mock_submission_repository.has_user_solved.return_value = False

        result = await submission_service.has_user_solved(uuid4(), uuid4())

        assert result is False


class TestProblemServiceAdditional:
    """Additional tests for ProblemService"""

    @pytest.fixture
    def mock_problem_repository(self):
        return AsyncMock()

    @pytest.fixture
    def problem_service(self, mock_problem_repository):
        return ProblemService(mock_problem_repository)

    @pytest.fixture
    def sample_problem(self):
        problem = Problem.create(
            title="Test Problem",
            description="Test description",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            hints=["Hint 1", "Hint 2", "Hint 3"],
        )
        problem.add_test_case(
            input_data="[1,2,3]",
            expected_output="6",
            is_sample=True,
        )
        return problem

    @pytest.mark.asyncio
    async def test_get_hints_all(self, problem_service, mock_problem_repository, sample_problem):
        """Test getting all hints for a problem"""
        mock_problem_repository.get_by_id.return_value = sample_problem

        result = await problem_service.get_hints(sample_problem.id, hint_index=None)

        assert result.problem_id == sample_problem.id
        assert len(result.hints) == 3
        assert result.total_hints == 3

    @pytest.mark.asyncio
    async def test_get_hints_progressive(self, problem_service, mock_problem_repository, sample_problem):
        """Test getting hints progressively"""
        mock_problem_repository.get_by_id.return_value = sample_problem

        # Request only first hint
        result = await problem_service.get_hints(sample_problem.id, hint_index=0)
        assert len(result.hints) == 1
        assert result.hints[0] == "Hint 1"

        # Request first two hints
        result = await problem_service.get_hints(sample_problem.id, hint_index=1)
        assert len(result.hints) == 2

    @pytest.mark.asyncio
    async def test_get_hints_not_found(self, problem_service, mock_problem_repository):
        """Test getting hints for non-existent problem"""
        mock_problem_repository.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await problem_service.get_hints(uuid4())

    @pytest.mark.asyncio
    async def test_delete_problem_success(self, problem_service, mock_problem_repository):
        """Test deleting a problem"""
        mock_problem_repository.delete.return_value = True
        problem_id = uuid4()

        result = await problem_service.delete_problem(problem_id)

        assert result is True
        mock_problem_repository.delete.assert_called_once_with(problem_id)

    @pytest.mark.asyncio
    async def test_delete_problem_not_found(self, problem_service, mock_problem_repository):
        """Test deleting a non-existent problem"""
        mock_problem_repository.delete.return_value = False

        result = await problem_service.delete_problem(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_get_recommended_problems(self, problem_service, mock_problem_repository, sample_problem):
        """Test getting recommended problems"""
        mock_problem_repository.get_published.return_value = [sample_problem]

        result = await problem_service.get_recommended_problems(uuid4(), limit=5)

        assert len(result) == 1
        assert result[0].title == sample_problem.title
        assert result[0].difficulty == "easy"

    @pytest.mark.asyncio
    async def test_get_recommended_problems_empty(self, problem_service, mock_problem_repository):
        """Test getting recommendations when no problems exist"""
        mock_problem_repository.get_published.return_value = []

        result = await problem_service.get_recommended_problems(uuid4())

        assert result == []

    def test_get_recommendation_reason_easy(self, problem_service):
        """Test recommendation reason for easy problem"""
        problem = Problem.create(
            title="Easy", description="", difficulty=Difficulty.EASY, category=Category.ARRAY
        )
        reason = problem_service._get_recommendation_reason(problem)
        assert "practice" in reason.lower() or "confidence" in reason.lower()

    def test_get_recommendation_reason_medium(self, problem_service):
        """Test recommendation reason for medium problem"""
        problem = Problem.create(
            title="Medium", description="", difficulty=Difficulty.MEDIUM, category=Category.ARRAY
        )
        reason = problem_service._get_recommendation_reason(problem)
        assert "skill" in reason.lower() or "development" in reason.lower()

    def test_get_recommendation_reason_hard(self, problem_service):
        """Test recommendation reason for hard problem"""
        problem = Problem.create(
            title="Hard", description="", difficulty=Difficulty.HARD, category=Category.ARRAY
        )
        reason = problem_service._get_recommendation_reason(problem)
        assert "challenge" in reason.lower()


# ============== Learning Routes Unit Tests ==============


class TestLearningRoutesUnit:
    """Unit tests for learning routes."""

    @pytest.fixture
    def mock_problem_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_submission_service(self):
        return AsyncMock()

    @pytest.fixture
    def mock_user(self):
        from code_tutor.identity.application.dto import UserResponse
        mock = MagicMock(spec=UserResponse)
        mock.id = uuid4()
        mock.email = "test@example.com"
        mock.username = "testuser"
        mock.role = "student"
        return mock

    @pytest.fixture
    def mock_admin_user(self):
        from code_tutor.identity.application.dto import UserResponse
        mock = MagicMock(spec=UserResponse)
        mock.id = uuid4()
        mock.email = "admin@example.com"
        mock.username = "admin"
        mock.role = "admin"
        return mock

    @pytest.mark.asyncio
    async def test_get_problem_repository_dependency(self):
        """Test get_problem_repository dependency."""
        from code_tutor.learning.interface.routes import get_problem_repository
        from code_tutor.learning.infrastructure.repository import SQLAlchemyProblemRepository

        mock_session = AsyncMock()
        repo = await get_problem_repository(mock_session)

        assert isinstance(repo, SQLAlchemyProblemRepository)

    @pytest.mark.asyncio
    async def test_get_submission_repository_dependency(self):
        """Test get_submission_repository dependency."""
        from code_tutor.learning.interface.routes import get_submission_repository
        from code_tutor.learning.infrastructure.repository import SQLAlchemySubmissionRepository

        mock_session = AsyncMock()
        repo = await get_submission_repository(mock_session)

        assert isinstance(repo, SQLAlchemySubmissionRepository)

    @pytest.mark.asyncio
    async def test_get_problem_service_dependency(self):
        """Test get_problem_service dependency."""
        from code_tutor.learning.interface.routes import get_problem_service
        from code_tutor.learning.application.services import ProblemService

        mock_repo = AsyncMock()
        service = await get_problem_service(mock_repo)

        assert isinstance(service, ProblemService)

    @pytest.mark.asyncio
    async def test_get_submission_service_dependency(self):
        """Test get_submission_service dependency."""
        from code_tutor.learning.interface.routes import get_submission_service
        from code_tutor.learning.application.services import SubmissionService

        mock_submission_repo = AsyncMock()
        mock_problem_repo = AsyncMock()
        service = await get_submission_service(mock_submission_repo, mock_problem_repo)

        assert isinstance(service, SubmissionService)

    @pytest.mark.asyncio
    async def test_list_problems_route(self, mock_problem_service):
        """Test list_problems route."""
        from code_tutor.learning.interface.routes import list_problems
        from code_tutor.learning.application.dto import ProblemListResponse

        mock_response = ProblemListResponse(
            items=[], total=0, page=1, size=20, pages=0
        )
        mock_problem_service.list_problems = AsyncMock(return_value=mock_response)

        result = await list_problems(
            service=mock_problem_service,
            category=None,
            difficulty=None,
            pattern=None,
            page=1,
            size=20,
        )

        assert result.total == 0
        assert result.items == []
        mock_problem_service.list_problems.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_problem_route_success(self, mock_problem_service):
        """Test get_problem route success."""
        from code_tutor.learning.interface.routes import get_problem
        from code_tutor.learning.application.dto import ProblemResponse
        from datetime import datetime

        problem_id = uuid4()
        mock_response = ProblemResponse(
            id=problem_id,
            title="Test",
            description="Test desc",
            difficulty="easy",
            category="array",
            constraints="",
            hints=[],
            solution_template="",
            reference_solution="",
            time_limit_ms=1000,
            memory_limit_mb=256,
            is_published=True,
            test_cases=[],
            pattern_ids=[],
            pattern_explanation="",
            approach_hint="",
            time_complexity_hint="",
            space_complexity_hint="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_problem_service.get_problem = AsyncMock(return_value=mock_response)

        result = await get_problem(problem_id, mock_problem_service)

        assert result.id == problem_id
        mock_problem_service.get_problem.assert_called_once_with(problem_id)

    @pytest.mark.asyncio
    async def test_get_problem_route_not_found(self, mock_problem_service):
        """Test get_problem route when not found."""
        from code_tutor.learning.interface.routes import get_problem
        from code_tutor.shared.exceptions import AppException
        from fastapi import HTTPException

        problem_id = uuid4()
        mock_problem_service.get_problem = AsyncMock(
            side_effect=AppException("Problem not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_problem(problem_id, mock_problem_service)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_problem_route(self, mock_problem_service, mock_admin_user):
        """Test create_problem route."""
        from code_tutor.learning.interface.routes import create_problem
        from code_tutor.learning.application.dto import (
            CreateProblemRequest,
            CreateTestCaseRequest,
            ProblemResponse,
        )
        from datetime import datetime

        problem_id = uuid4()
        request = CreateProblemRequest(
            title="Test Problem",
            description="Test description",
            difficulty=Difficulty.EASY,
            category=Category.ARRAY,
            constraints="",
            hints=[],
            solution_template="",
            time_limit_ms=1000,
            memory_limit_mb=256,
            test_cases=[
                CreateTestCaseRequest(
                    input_data="[1,2]",
                    expected_output="3",
                    is_sample=True,
                )
            ],
        )
        mock_response = ProblemResponse(
            id=problem_id,
            title="Test Problem",
            description="Test description",
            difficulty="easy",
            category="array",
            constraints="",
            hints=[],
            solution_template="",
            reference_solution="",
            time_limit_ms=1000,
            memory_limit_mb=256,
            is_published=False,
            test_cases=[],
            pattern_ids=[],
            pattern_explanation="",
            approach_hint="",
            time_complexity_hint="",
            space_complexity_hint="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_problem_service.create_problem = AsyncMock(return_value=mock_response)

        result = await create_problem(request, mock_problem_service, mock_admin_user)

        assert result.title == "Test Problem"
        mock_problem_service.create_problem.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_get_problem_hints_route_success(self, mock_problem_service, mock_user):
        """Test get_problem_hints route success."""
        from code_tutor.learning.interface.routes import get_problem_hints
        from code_tutor.learning.application.dto import HintsResponse

        problem_id = uuid4()
        mock_response = HintsResponse(
            problem_id=problem_id,
            hints=["Hint 1", "Hint 2"],
            total_hints=3,
        )
        mock_problem_service.get_hints = AsyncMock(return_value=mock_response)

        result = await get_problem_hints(
            problem_id, mock_problem_service, mock_user, hint_index=1
        )

        assert result.problem_id == problem_id
        assert len(result.hints) == 2
        mock_problem_service.get_hints.assert_called_once_with(problem_id, 1)

    @pytest.mark.asyncio
    async def test_get_problem_hints_route_not_found(self, mock_problem_service, mock_user):
        """Test get_problem_hints route when not found."""
        from code_tutor.learning.interface.routes import get_problem_hints
        from code_tutor.shared.exceptions import AppException
        from fastapi import HTTPException

        mock_problem_service.get_hints = AsyncMock(
            side_effect=AppException("Problem not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_problem_hints(uuid4(), mock_problem_service, mock_user)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_publish_problem_route_success(self, mock_problem_service, mock_admin_user):
        """Test publish_problem route success."""
        from code_tutor.learning.interface.routes import publish_problem
        from code_tutor.learning.application.dto import ProblemResponse
        from datetime import datetime

        problem_id = uuid4()
        mock_response = ProblemResponse(
            id=problem_id,
            title="Test",
            description="",
            difficulty="easy",
            category="array",
            constraints="",
            hints=[],
            solution_template="",
            reference_solution="",
            time_limit_ms=1000,
            memory_limit_mb=256,
            is_published=True,
            test_cases=[],
            pattern_ids=[],
            pattern_explanation="",
            approach_hint="",
            time_complexity_hint="",
            space_complexity_hint="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_problem_service.publish_problem = AsyncMock(return_value=mock_response)

        result = await publish_problem(problem_id, mock_problem_service, mock_admin_user)

        assert result.is_published is True
        mock_problem_service.publish_problem.assert_called_once_with(problem_id)

    @pytest.mark.asyncio
    async def test_publish_problem_route_error(self, mock_problem_service, mock_admin_user):
        """Test publish_problem route error."""
        from code_tutor.learning.interface.routes import publish_problem
        from code_tutor.shared.exceptions import AppException
        from fastapi import HTTPException

        mock_problem_service.publish_problem = AsyncMock(
            side_effect=AppException("Cannot publish")
        )

        with pytest.raises(HTTPException) as exc_info:
            await publish_problem(uuid4(), mock_problem_service, mock_admin_user)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_create_submission_route_success(self, mock_submission_service, mock_user):
        """Test create_submission route success."""
        from code_tutor.learning.interface.routes import create_submission
        from code_tutor.learning.application.dto import CreateSubmissionRequest, SubmissionResponse
        from datetime import datetime

        submission_id = uuid4()
        problem_id = uuid4()
        request = CreateSubmissionRequest(
            problem_id=problem_id,
            code="def solve(): pass",
            language="python",
        )
        mock_response = SubmissionResponse(
            id=submission_id,
            user_id=mock_user.id,
            problem_id=problem_id,
            code="def solve(): pass",
            language="python",
            status="pending",
            test_results=[],
            total_tests=0,
            passed_tests=0,
            execution_time_ms=0,
            memory_usage_mb=0,
            error_message=None,
            submitted_at=datetime.now(),
            evaluated_at=None,
        )
        mock_submission_service.create_submission = AsyncMock(return_value=mock_response)

        result = await create_submission(request, mock_submission_service, mock_user)

        assert result.id == submission_id
        mock_submission_service.create_submission.assert_called_once_with(mock_user.id, request)

    @pytest.mark.asyncio
    async def test_create_submission_route_error(self, mock_submission_service, mock_user):
        """Test create_submission route error."""
        from code_tutor.learning.interface.routes import create_submission
        from code_tutor.learning.application.dto import CreateSubmissionRequest
        from code_tutor.shared.exceptions import AppException
        from fastapi import HTTPException

        request = CreateSubmissionRequest(
            problem_id=uuid4(),
            code="def solve(): pass",
            language="python",
        )
        mock_submission_service.create_submission = AsyncMock(
            side_effect=AppException("Problem not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_submission(request, mock_submission_service, mock_user)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_submission_route_success(self, mock_submission_service, mock_user):
        """Test get_submission route success."""
        from code_tutor.learning.interface.routes import get_submission
        from code_tutor.learning.application.dto import SubmissionResponse
        from datetime import datetime

        submission_id = uuid4()
        mock_response = SubmissionResponse(
            id=submission_id,
            user_id=mock_user.id,
            problem_id=uuid4(),
            code="def solve(): pass",
            language="python",
            status="accepted",
            test_results=[],
            total_tests=1,
            passed_tests=1,
            execution_time_ms=50,
            memory_usage_mb=10,
            error_message=None,
            submitted_at=datetime.now(),
            evaluated_at=datetime.now(),
        )
        mock_submission_service.get_submission = AsyncMock(return_value=mock_response)

        result = await get_submission(submission_id, mock_submission_service, mock_user)

        assert result.id == submission_id
        mock_submission_service.get_submission.assert_called_once_with(submission_id)

    @pytest.mark.asyncio
    async def test_get_submission_route_forbidden(self, mock_submission_service, mock_user):
        """Test get_submission route forbidden for other user's submission."""
        from code_tutor.learning.interface.routes import get_submission
        from code_tutor.learning.application.dto import SubmissionResponse
        from fastapi import HTTPException
        from datetime import datetime

        submission_id = uuid4()
        other_user_id = uuid4()
        mock_response = SubmissionResponse(
            id=submission_id,
            user_id=other_user_id,  # Different user
            problem_id=uuid4(),
            code="def solve(): pass",
            language="python",
            status="accepted",
            test_results=[],
            total_tests=1,
            passed_tests=1,
            execution_time_ms=50,
            memory_usage_mb=10,
            error_message=None,
            submitted_at=datetime.now(),
            evaluated_at=datetime.now(),
        )
        mock_submission_service.get_submission = AsyncMock(return_value=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            await get_submission(submission_id, mock_submission_service, mock_user)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_submission_route_not_found(self, mock_submission_service, mock_user):
        """Test get_submission route when not found."""
        from code_tutor.learning.interface.routes import get_submission
        from code_tutor.shared.exceptions import AppException
        from fastapi import HTTPException

        mock_submission_service.get_submission = AsyncMock(
            side_effect=AppException("Submission not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_submission(uuid4(), mock_submission_service, mock_user)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_my_submissions_route(self, mock_submission_service, mock_user):
        """Test list_my_submissions route."""
        from code_tutor.learning.interface.routes import list_my_submissions

        mock_submission_service.get_user_submissions = AsyncMock(return_value=[])

        result = await list_my_submissions(mock_submission_service, mock_user, limit=20, offset=0)

        assert result == []
        mock_submission_service.get_user_submissions.assert_called_once_with(
            mock_user.id, 20, 0
        )

    @pytest.mark.asyncio
    async def test_list_problem_submissions_route(self, mock_submission_service, mock_user):
        """Test list_problem_submissions route."""
        from code_tutor.learning.interface.routes import list_problem_submissions

        problem_id = uuid4()
        mock_submission_service.get_user_problem_submissions = AsyncMock(return_value=[])

        result = await list_problem_submissions(
            problem_id, mock_submission_service, mock_user, limit=10
        )

        assert result == []
        mock_submission_service.get_user_problem_submissions.assert_called_once_with(
            mock_user.id, problem_id, 10
        )

    def test_get_recommendation_reason_kr(self):
        """Test Korean recommendation reason translation."""
        from code_tutor.learning.interface.routes import _get_recommendation_reason_kr

        assert "비슷한" in _get_recommendation_reason_kr("similar_users")
        assert "학습 패턴" in _get_recommendation_reason_kr("content_match")
        assert "AI" in _get_recommendation_reason_kr("hybrid")
        assert "인기" in _get_recommendation_reason_kr("popular")
        assert "추천" in _get_recommendation_reason_kr("recommended")
        assert "추천" in _get_recommendation_reason_kr("unknown_reason")


# ============== Submission Service _to_summary Tests ==============


class TestSubmissionServiceToSummary:
    """Tests for SubmissionService _to_summary method."""

    @pytest.fixture
    def submission_service(self):
        from code_tutor.learning.application.services import SubmissionService
        return SubmissionService(AsyncMock(), AsyncMock())

    def test_to_summary_converts_submission(self, submission_service):
        """Test _to_summary converts submission to summary response."""
        from code_tutor.learning.domain.entities import Submission
        from code_tutor.learning.domain.value_objects import SubmissionStatus
        from datetime import datetime, timezone

        submission = Submission.create(
            user_id=uuid4(),
            problem_id=uuid4(),
            code="def solve(): pass",
            language="python",
        )
        # Manually set some values for testing
        submission._status = SubmissionStatus.ACCEPTED
        submission._passed_tests = 5
        submission._total_tests = 5

        result = submission_service._to_summary(submission)

        assert result.id == submission.id
        assert result.problem_id == submission.problem_id
        assert result.status == "accepted"
        assert result.passed_tests == 5
        assert result.total_tests == 5
        assert result.submitted_at is not None


# ============== Dashboard Service Tests ==============


class TestDashboardService:
    """Tests for DashboardService."""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def dashboard_service(self, mock_session):
        """Create dashboard service with mock session."""
        from code_tutor.learning.application.dashboard_service import DashboardService
        return DashboardService(mock_session)

    def test_calculate_activity_level_zero(self, dashboard_service):
        """Test activity level 0 for zero count."""
        assert dashboard_service._calculate_activity_level(0) == 0

    def test_calculate_activity_level_one(self, dashboard_service):
        """Test activity level 1 for 1-2 submissions."""
        assert dashboard_service._calculate_activity_level(1) == 1
        assert dashboard_service._calculate_activity_level(2) == 1

    def test_calculate_activity_level_two(self, dashboard_service):
        """Test activity level 2 for 3-5 submissions."""
        assert dashboard_service._calculate_activity_level(3) == 2
        assert dashboard_service._calculate_activity_level(5) == 2

    def test_calculate_activity_level_three(self, dashboard_service):
        """Test activity level 3 for 6-10 submissions."""
        assert dashboard_service._calculate_activity_level(6) == 3
        assert dashboard_service._calculate_activity_level(10) == 3

    def test_calculate_activity_level_four(self, dashboard_service):
        """Test activity level 4 for more than 10 submissions."""
        assert dashboard_service._calculate_activity_level(11) == 4
        assert dashboard_service._calculate_activity_level(100) == 4

    def test_calculate_recent_trend_insufficient_data(self, dashboard_service):
        """Test trend calculation with insufficient data."""
        from code_tutor.learning.application.dashboard_dto import RecentSubmission
        from datetime import datetime, timezone

        # Less than 3 submissions
        submissions = [
            RecentSubmission(
                id=uuid4(),
                problem_id=uuid4(),
                problem_title="Test",
                status="accepted",
                submitted_at=datetime.now(timezone.utc),
            )
        ]
        assert dashboard_service._calculate_recent_trend(submissions) == 0.0

        # Empty list
        assert dashboard_service._calculate_recent_trend([]) == 0.0

    def test_calculate_recent_trend_positive(self, dashboard_service):
        """Test positive trend calculation."""
        from code_tutor.learning.application.dashboard_dto import RecentSubmission
        from datetime import datetime, timezone

        # Recent submissions are more successful
        submissions = [
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T1", status="accepted", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T2", status="accepted", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T3", status="wrong_answer", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T4", status="wrong_answer", submitted_at=datetime.now(timezone.utc)),
        ]
        trend = dashboard_service._calculate_recent_trend(submissions)
        assert trend > 0  # Recent half has better success rate

    def test_calculate_recent_trend_negative(self, dashboard_service):
        """Test negative trend calculation."""
        from code_tutor.learning.application.dashboard_dto import RecentSubmission
        from datetime import datetime, timezone

        # Older submissions were more successful
        submissions = [
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T1", status="wrong_answer", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T2", status="wrong_answer", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T3", status="accepted", submitted_at=datetime.now(timezone.utc)),
            RecentSubmission(id=uuid4(), problem_id=uuid4(), problem_title="T4", status="accepted", submitted_at=datetime.now(timezone.utc)),
        ]
        trend = dashboard_service._calculate_recent_trend(submissions)
        assert trend < 0  # Recent half has worse success rate

    def test_generate_skill_predictions_empty(self, dashboard_service):
        """Test skill predictions with empty progress."""
        predictions = dashboard_service._generate_skill_predictions([])
        assert predictions == []

    def test_generate_skill_predictions_with_data(self, dashboard_service):
        """Test skill predictions with category progress."""
        from code_tutor.learning.application.dashboard_dto import CategoryProgress

        progress = [
            CategoryProgress(category="array", total_problems=10, solved_problems=5, success_rate=80.0),
            CategoryProgress(category="string", total_problems=5, solved_problems=1, success_rate=30.0),
        ]

        predictions = dashboard_service._generate_skill_predictions(progress)

        assert len(predictions) == 2
        # Check that predictions have expected fields
        for pred in predictions:
            assert hasattr(pred, 'category')
            assert hasattr(pred, 'current_level')
            assert hasattr(pred, 'predicted_level')
            assert hasattr(pred, 'confidence')
            assert hasattr(pred, 'recommended_focus')

    def test_generate_skill_predictions_zero_problems(self, dashboard_service):
        """Test skill predictions skips categories with zero problems."""
        from code_tutor.learning.application.dashboard_dto import CategoryProgress

        progress = [
            CategoryProgress(category="array", total_problems=0, solved_problems=0, success_rate=0.0),
        ]

        predictions = dashboard_service._generate_skill_predictions(progress)
        assert predictions == []

    def test_generate_skill_predictions_recommended_focus(self, dashboard_service):
        """Test skill predictions marks recommended focus for low completion."""
        from code_tutor.learning.application.dashboard_dto import CategoryProgress

        # Low completion rate (1/10 = 10%) with enough problems
        progress = [
            CategoryProgress(category="array", total_problems=10, solved_problems=1, success_rate=50.0),
        ]

        predictions = dashboard_service._generate_skill_predictions(progress)

        assert len(predictions) == 1
        assert predictions[0].recommended_focus is True

    def test_generate_insights_positive_trend(self, dashboard_service):
        """Test insights generation with positive trend."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo

        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=8,
            total_submissions=20,
            overall_success_rate=80.0,
            easy_solved=3,
            medium_solved=3,
            hard_solved=2,
            streak=StreakInfo(current_streak=2, longest_streak=5),
        )
        category_progress = []
        trend = 0.2  # Positive trend

        insights = dashboard_service._generate_insights(stats, category_progress, trend)

        # Should have trend insight
        trend_insights = [i for i in insights if i.type == "trend"]
        assert len(trend_insights) >= 1
        assert "상승" in trend_insights[0].message

    def test_generate_insights_negative_trend(self, dashboard_service):
        """Test insights generation with negative trend."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo

        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=5,
            total_submissions=20,
            overall_success_rate=50.0,
            easy_solved=3,
            medium_solved=2,
            hard_solved=0,
            streak=StreakInfo(current_streak=1, longest_streak=3),
        )
        category_progress = []
        trend = -0.2  # Negative trend

        insights = dashboard_service._generate_insights(stats, category_progress, trend)

        # Should have trend insight about decline
        trend_insights = [i for i in insights if i.type == "trend"]
        assert len(trend_insights) >= 1
        assert "하락" in trend_insights[0].message

    def test_generate_insights_long_streak(self, dashboard_service):
        """Test insights generation with long streak."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo

        stats = UserStats(
            total_problems_attempted=20,
            total_problems_solved=15,
            total_submissions=50,
            overall_success_rate=75.0,
            easy_solved=5,
            medium_solved=7,
            hard_solved=3,
            streak=StreakInfo(current_streak=10, longest_streak=10),
        )

        insights = dashboard_service._generate_insights(stats, [], 0.0)

        # Should have achievement insight for 7+ day streak
        achievement_insights = [i for i in insights if i.type == "achievement"]
        assert len(achievement_insights) >= 1
        assert "연속" in achievement_insights[0].message

    def test_generate_insights_medium_streak(self, dashboard_service):
        """Test insights generation with medium streak (3-6 days)."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo

        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=8,
            total_submissions=20,
            overall_success_rate=70.0,
            easy_solved=4,
            medium_solved=3,
            hard_solved=1,
            streak=StreakInfo(current_streak=5, longest_streak=5),
        )

        insights = dashboard_service._generate_insights(stats, [], 0.0)

        # Should have achievement insight for 3-6 day streak
        achievement_insights = [i for i in insights if i.type == "achievement"]
        assert len(achievement_insights) >= 1
        assert "7일" in achievement_insights[0].message  # Encouraging to reach 7

    def test_generate_insights_weak_category(self, dashboard_service):
        """Test insights generation with weak category."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo, CategoryProgress

        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=5,
            total_submissions=20,
            overall_success_rate=50.0,
            easy_solved=3,
            medium_solved=2,
            hard_solved=0,
            streak=StreakInfo(current_streak=1, longest_streak=3),
        )
        category_progress = [
            CategoryProgress(category="dynamic_programming", total_problems=5, solved_problems=1, success_rate=30.0),
        ]

        insights = dashboard_service._generate_insights(stats, category_progress, 0.0)

        # Should have recommendation insight for weak category
        rec_insights = [i for i in insights if i.type == "recommendation"]
        assert len(rec_insights) >= 1


class TestDashboardServiceAsync:
    """Async tests for DashboardService."""

    @pytest.fixture
    def mock_session(self):
        """Create mock async session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def dashboard_service(self, mock_session):
        """Create dashboard service with mock session."""
        from code_tutor.learning.application.dashboard_service import DashboardService
        return DashboardService(mock_session)

    @pytest.mark.asyncio
    async def test_get_streak_info_no_submissions(self, dashboard_service, mock_session):
        """Test streak info with no submissions."""
        # Mock empty result
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        streak = await dashboard_service._get_streak_info(uuid4())

        assert streak.current_streak == 0
        assert streak.longest_streak == 0

    @pytest.mark.asyncio
    async def test_get_streak_info_with_string_dates(self, dashboard_service, mock_session):
        """Test streak info handles string dates from SQLite."""
        from datetime import datetime, timedelta, timezone

        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # Mock result with string dates (SQLite format)
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (today.isoformat(),),
            (yesterday.isoformat(),),
        ]
        mock_session.execute.return_value = mock_result

        streak = await dashboard_service._get_streak_info(uuid4())

        assert streak.current_streak >= 1
        assert streak.longest_streak >= 1

    @pytest.mark.asyncio
    async def test_get_user_stats_empty(self, dashboard_service, mock_session):
        """Test user stats with no data."""
        # Create mock results for all queries
        mock_result1 = MagicMock()
        mock_result1.first.return_value = MagicMock(total_submissions=0, accepted_submissions=0)

        mock_result2 = MagicMock()
        mock_result2.first.return_value = MagicMock(attempted=0, solved=0)

        mock_result3 = MagicMock()
        mock_result3.all.return_value = []

        # Streak info query
        mock_result4 = MagicMock()
        mock_result4.all.return_value = []

        mock_session.execute.side_effect = [mock_result1, mock_result2, mock_result3, mock_result4]

        stats = await dashboard_service._get_user_stats(uuid4())

        assert stats.total_submissions == 0
        assert stats.total_problems_solved == 0
        assert stats.overall_success_rate == 0

    @pytest.mark.asyncio
    async def test_get_category_progress_empty(self, dashboard_service, mock_session):
        """Test category progress with no data."""
        mock_result1 = MagicMock()
        mock_result1.all.return_value = []

        mock_result2 = MagicMock()
        mock_result2.all.return_value = []

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        progress = await dashboard_service._get_category_progress(uuid4())

        assert progress == []

    @pytest.mark.asyncio
    async def test_get_recent_submissions_empty(self, dashboard_service, mock_session):
        """Test recent submissions with no data."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        submissions = await dashboard_service._get_recent_submissions(uuid4())

        assert submissions == []

    @pytest.mark.asyncio
    async def test_get_heatmap_data_empty(self, dashboard_service, mock_session):
        """Test heatmap data generation with no submissions."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        heatmap = await dashboard_service._get_heatmap_data(uuid4(), days=7)

        # Should still return 8 days of data (7 + today)
        assert len(heatmap) == 8
        # All should have level 0
        assert all(h.level == 0 for h in heatmap)

    @pytest.mark.asyncio
    async def test_generate_recommendations_with_failed(self, dashboard_service, mock_session):
        """Test recommendations include retry for failed submissions."""
        from code_tutor.learning.application.dashboard_dto import (
            UserStats, StreakInfo, CategoryProgress, RecentSubmission
        )
        from datetime import datetime, timezone

        stats = UserStats(
            total_problems_attempted=5,
            total_problems_solved=3,
            total_submissions=10,
            overall_success_rate=60.0,
            easy_solved=2,
            medium_solved=1,
            hard_solved=0,
            streak=StreakInfo(current_streak=1, longest_streak=2),
        )
        category_progress = []
        recent_submissions = [
            RecentSubmission(
                id=uuid4(),
                problem_id=uuid4(),
                problem_title="Failed Problem",
                status="wrong_answer",
                submitted_at=datetime.now(timezone.utc),
            ),
        ]

        recommendations = await dashboard_service._generate_recommendations(
            uuid4(), stats, category_progress, recent_submissions
        )

        # Should have review recommendation
        review_recs = [r for r in recommendations if r.type == "review"]
        assert len(review_recs) >= 1

    @pytest.mark.asyncio
    async def test_generate_recommendations_weak_category(self, dashboard_service, mock_session):
        """Test recommendations for weak category."""
        from code_tutor.learning.application.dashboard_dto import (
            UserStats, StreakInfo, CategoryProgress
        )

        stats = UserStats(
            total_problems_attempted=10,
            total_problems_solved=5,
            total_submissions=20,
            overall_success_rate=50.0,
            easy_solved=3,
            medium_solved=2,
            hard_solved=0,
            streak=StreakInfo(current_streak=1, longest_streak=3),
        )
        category_progress = [
            CategoryProgress(category="graph", total_problems=5, solved_problems=1, success_rate=30.0),
        ]

        recommendations = await dashboard_service._generate_recommendations(
            uuid4(), stats, category_progress, []
        )

        # Should have practice recommendation
        practice_recs = [r for r in recommendations if r.type == "practice"]
        assert len(practice_recs) >= 1

    @pytest.mark.asyncio
    async def test_generate_recommendations_challenge(self, dashboard_service, mock_session):
        """Test challenge recommendation for good performers."""
        from code_tutor.learning.application.dashboard_dto import UserStats, StreakInfo

        stats = UserStats(
            total_problems_attempted=20,
            total_problems_solved=15,
            total_submissions=50,
            overall_success_rate=80.0,
            easy_solved=5,
            medium_solved=7,
            hard_solved=3,
            streak=StreakInfo(current_streak=5, longest_streak=10),
        )

        recommendations = await dashboard_service._generate_recommendations(
            uuid4(), stats, [], []
        )

        # Should have challenge recommendation
        challenge_recs = [r for r in recommendations if r.type == "challenge"]
        assert len(challenge_recs) >= 1

    @pytest.mark.asyncio
    async def test_get_prediction(self, dashboard_service, mock_session):
        """Test get_prediction method."""
        # Mock all required queries
        # Stats query 1
        mock_result1 = MagicMock()
        mock_result1.first.return_value = MagicMock(total_submissions=10, accepted_submissions=7)

        # Stats query 2
        mock_result2 = MagicMock()
        mock_result2.first.return_value = MagicMock(attempted=5, solved=4)

        # Stats query 3 (difficulty)
        mock_result3 = MagicMock()
        mock_result3.all.return_value = []

        # Streak query
        mock_result4 = MagicMock()
        mock_result4.all.return_value = []

        # Category progress query 1
        mock_result5 = MagicMock()
        mock_result5.all.return_value = []

        # Category progress query 2
        mock_result6 = MagicMock()
        mock_result6.all.return_value = []

        # Recent submissions query
        mock_result7 = MagicMock()
        mock_result7.all.return_value = []

        mock_session.execute.side_effect = [
            mock_result1, mock_result2, mock_result3, mock_result4,
            mock_result5, mock_result6, mock_result7
        ]

        prediction = await dashboard_service.get_prediction(uuid4())

        assert prediction.current_success_rate >= 0
        assert prediction.predicted_success_rate >= 0
        assert prediction.prediction_period == "next_week"
        assert 0 <= prediction.confidence <= 1
