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
