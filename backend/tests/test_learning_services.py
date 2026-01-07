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
