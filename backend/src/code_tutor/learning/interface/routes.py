"""Learning API routes"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.learning.application.dashboard_dto import DashboardResponse
from code_tutor.learning.application.dashboard_service import DashboardService
from code_tutor.learning.application.dto import (
    CreateProblemRequest,
    CreateSubmissionRequest,
    HintsResponse,
    ProblemFilterParams,
    ProblemListResponse,
    ProblemResponse,
    RecommendedProblemResponse,
    SubmissionResponse,
    SubmissionSummaryResponse,
)
from code_tutor.learning.application.services import ProblemService, SubmissionService
from code_tutor.learning.domain.repository import ProblemRepository, SubmissionRepository
from code_tutor.learning.domain.value_objects import Category, Difficulty
from code_tutor.learning.infrastructure.repository import (
    SQLAlchemyProblemRepository,
    SQLAlchemySubmissionRepository,
)
from code_tutor.shared.api_response import success_response
from code_tutor.shared.exceptions import AppException
from code_tutor.shared.infrastructure.database import get_async_session

router = APIRouter(tags=["Learning"])


# Dependencies
async def get_problem_repository(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ProblemRepository:
    return SQLAlchemyProblemRepository(session)


async def get_submission_repository(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> SubmissionRepository:
    return SQLAlchemySubmissionRepository(session)


async def get_problem_service(
    repo: Annotated[ProblemRepository, Depends(get_problem_repository)],
) -> ProblemService:
    return ProblemService(repo)


async def get_submission_service(
    submission_repo: Annotated[SubmissionRepository, Depends(get_submission_repository)],
    problem_repo: Annotated[ProblemRepository, Depends(get_problem_repository)],
) -> SubmissionService:
    return SubmissionService(submission_repo, problem_repo)


# Problem endpoints
@router.get(
    "/problems",
    response_model=ProblemListResponse,
    summary="List problems",
)
async def list_problems(
    service: Annotated[ProblemService, Depends(get_problem_service)],
    category: Category | None = None,
    difficulty: Difficulty | None = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> ProblemListResponse:
    """List published problems with optional filters"""
    params = ProblemFilterParams(
        category=category,
        difficulty=difficulty,
        page=page,
        size=size,
    )
    return await service.list_problems(params)


@router.get(
    "/problems/{problem_id}",
    response_model=ProblemResponse,
    summary="Get problem details",
)
async def get_problem(
    problem_id: UUID,
    service: Annotated[ProblemService, Depends(get_problem_service)],
) -> ProblemResponse:
    """Get problem by ID with sample test cases"""
    try:
        return await service.get_problem(problem_id)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post(
    "/problems",
    response_model=ProblemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new problem (Admin only)",
)
async def create_problem(
    request: CreateProblemRequest,
    service: Annotated[ProblemService, Depends(get_problem_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> ProblemResponse:
    """Create a new problem (requires admin role)"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return await service.create_problem(request)


@router.get(
    "/problems/recommended",
    response_model=list[RecommendedProblemResponse],
    summary="Get recommended problems for user",
)
async def get_recommended_problems(
    service: Annotated[ProblemService, Depends(get_problem_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=5, ge=1, le=20),
) -> list[RecommendedProblemResponse]:
    """Get personalized problem recommendations based on user's history"""
    return await service.get_recommended_problems(current_user.id, limit)


@router.get(
    "/problems/{problem_id}/hints",
    response_model=HintsResponse,
    summary="Get hints for a problem",
)
async def get_problem_hints(
    problem_id: UUID,
    service: Annotated[ProblemService, Depends(get_problem_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    hint_index: int | None = Query(default=None, ge=0, description="Get hints up to this index"),
) -> HintsResponse:
    """Get hints for a problem (progressive reveal)"""
    try:
        return await service.get_hints(problem_id, hint_index)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post(
    "/problems/{problem_id}/publish",
    response_model=ProblemResponse,
    summary="Publish a problem (Admin only)",
)
async def publish_problem(
    problem_id: UUID,
    service: Annotated[ProblemService, Depends(get_problem_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> ProblemResponse:
    """Publish a problem"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    try:
        return await service.publish_problem(problem_id)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


# Submission endpoints
@router.post(
    "/submissions",
    response_model=SubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit code solution",
)
async def create_submission(
    request: CreateSubmissionRequest,
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> SubmissionResponse:
    """Submit a code solution for evaluation"""
    try:
        return await service.create_submission(current_user.id, request)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


@router.get(
    "/submissions/{submission_id}",
    response_model=SubmissionResponse,
    summary="Get submission details",
)
async def get_submission(
    submission_id: UUID,
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> SubmissionResponse:
    """Get submission by ID"""
    try:
        submission = await service.get_submission(submission_id)
        # Only allow owner or admin to view
        if submission.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
        return submission
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get(
    "/submissions",
    response_model=list[SubmissionSummaryResponse],
    summary="List my submissions",
)
async def list_my_submissions(
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[SubmissionSummaryResponse]:
    """List current user's submissions"""
    return await service.get_user_submissions(current_user.id, limit, offset)


@router.get(
    "/problems/{problem_id}/submissions",
    response_model=list[SubmissionSummaryResponse],
    summary="List my submissions for a problem",
)
async def list_problem_submissions(
    problem_id: UUID,
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=10, ge=1, le=50),
) -> list[SubmissionSummaryResponse]:
    """List current user's submissions for a specific problem"""
    return await service.get_user_problem_submissions(current_user.id, problem_id, limit)


# Dashboard endpoint
async def get_dashboard_service(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> DashboardService:
    return DashboardService(session)


@router.get(
    "/dashboard",
    summary="Get user dashboard",
)
async def get_dashboard(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get user dashboard with statistics, progress, and recent activity.

    Returns:
    - Overall statistics (problems solved, success rate, streaks)
    - Progress by category
    - Recent submissions
    """
    service = DashboardService(session)
    dashboard = await service.get_dashboard(current_user.id)
    return success_response(dashboard.model_dump(mode="json"))
