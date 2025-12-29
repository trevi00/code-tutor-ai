"""Learning API routes"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.learning.application.dashboard_dto import DashboardResponse, PredictionResponse
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
from code_tutor.execution.application.services import SubmissionEvaluator
from code_tutor.learning.application.services import ProblemService, SubmissionService
from code_tutor.shared.config import get_settings
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


async def get_submission_evaluator(
    problem_repo: Annotated[ProblemRepository, Depends(get_problem_repository)],
    submission_repo: Annotated[SubmissionRepository, Depends(get_submission_repository)],
) -> SubmissionEvaluator:
    settings = get_settings()
    use_docker = settings.ENVIRONMENT != "development"
    return SubmissionEvaluator(problem_repo, submission_repo, use_docker=use_docker)


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


@router.post(
    "/submissions/{submission_id}/evaluate",
    response_model=SubmissionResponse,
    summary="Evaluate a submission",
)
async def evaluate_submission(
    submission_id: UUID,
    evaluator: Annotated[SubmissionEvaluator, Depends(get_submission_evaluator)],
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> SubmissionResponse:
    """
    Evaluate a pending submission against all test cases.

    - Runs code in sandboxed environment
    - Compares output with expected results
    - Updates submission status (ACCEPTED, WRONG_ANSWER, etc.)
    """
    try:
        # Check if user owns the submission
        submission = await service.get_submission(submission_id)
        if submission.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        # Run evaluation
        await evaluator.evaluate_submission(submission_id)

        # Return updated submission
        return await service.get_submission(submission_id)
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


@router.get(
    "/dashboard/prediction",
    summary="Get learning predictions",
)
async def get_prediction(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get AI-powered learning predictions and recommendations.

    Returns:
    - Current and predicted success rates
    - Personalized insights based on learning patterns
    - Recommendations for improvement
    """
    service = DashboardService(session)
    prediction = await service.get_prediction(current_user.id)
    return success_response(prediction.model_dump(mode="json"))


# ML-powered endpoints
@router.post(
    "/code/analyze",
    summary="Analyze code with AI",
)
async def analyze_code(
    code: str,
    language: str = Query(default="python"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
) -> dict[str, Any]:
    """
    Analyze code using CodeBERT and pattern detection.

    Returns:
    - Detected algorithm patterns
    - Code quality score
    - Complexity metrics
    - Improvement suggestions
    """
    try:
        from code_tutor.ml import get_code_analyzer
        analyzer = get_code_analyzer()
        result = analyzer.analyze(code, language)
        return success_response(result)
    except Exception as e:
        # Fallback to basic analysis
        return success_response({
            "patterns": [],
            "quality": {"score": 70, "grade": "C"},
            "complexity": {"cyclomatic": 1},
            "suggestions": ["코드 분석 기능을 사용할 수 없습니다."],
            "error": str(e)
        })


@router.post(
    "/code/classify",
    summary="Classify code quality (Transformer)",
)
async def classify_code(
    code: str,
    language: str = Query(default="python"),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
) -> dict[str, Any]:
    """
    Classify code quality using Transformer (CodeBERT Classifier).

    Evaluates code across 4 dimensions:
    - Correctness: 정확성
    - Efficiency: 효율성
    - Readability: 가독성
    - Best Practices: 베스트 프랙티스

    Returns:
    - Overall score and grade
    - Dimension-wise scores
    - Improvement suggestions
    """
    try:
        from code_tutor.ml import get_code_classifier
        classifier = get_code_classifier()
        result = classifier.classify(code, language)
        suggestions = classifier.get_improvement_suggestions(result)
        result["suggestions"] = suggestions
        return success_response(result)
    except Exception as e:
        # Fallback response
        return success_response({
            "overall_score": 70,
            "overall_grade": "C",
            "dimensions": {
                "correctness": {"label": "fair", "score": 70},
                "efficiency": {"label": "fair", "score": 70},
                "readability": {"label": "fair", "score": 70},
                "best_practices": {"label": "fair", "score": 70}
            },
            "suggestions": [],
            "error": str(e)
        })


@router.get(
    "/patterns",
    summary="List algorithm patterns",
)
async def list_patterns(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
) -> dict[str, Any]:
    """
    List all available algorithm patterns in the knowledge base.

    Returns:
    - List of 25 algorithm patterns with descriptions
    """
    try:
        from code_tutor.ml.rag import PatternKnowledgeBase
        kb = PatternKnowledgeBase()
        patterns = [
            {
                "id": p["id"],
                "name": p["name"],
                "name_ko": p["name_ko"],
                "description": p["description"],
                "description_ko": p["description_ko"],
                "time_complexity": p["time_complexity"],
                "space_complexity": p["space_complexity"],
                "use_cases": p["use_cases"],
                "keywords": p["keywords"]
            }
            for p in kb.patterns
        ]
        return success_response({"patterns": patterns, "total": len(patterns)})
    except Exception as e:
        return success_response({"patterns": [], "total": 0, "error": str(e)})


@router.get(
    "/patterns/{pattern_id}",
    summary="Get pattern details",
)
async def get_pattern(
    pattern_id: str,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
) -> dict[str, Any]:
    """
    Get detailed information about a specific algorithm pattern.

    Returns:
    - Full pattern description
    - Example code
    - Use cases
    """
    try:
        from code_tutor.ml.rag import PatternKnowledgeBase
        kb = PatternKnowledgeBase()
        pattern = kb.get_pattern(pattern_id)
        if pattern:
            return success_response(pattern)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pattern not found: {pattern_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/patterns/search",
    summary="Search patterns by query",
)
async def search_patterns(
    query: str,
    top_k: int = Query(default=3, ge=1, le=10),
    current_user: Annotated[UserResponse, Depends(get_current_active_user)] = None,
) -> dict[str, Any]:
    """
    Search for algorithm patterns using semantic search.

    Args:
    - query: Natural language query (e.g., "배열에서 두 수의 합 찾기")
    - top_k: Number of results to return

    Returns:
    - Matching patterns with similarity scores
    """
    try:
        from code_tutor.ml import get_rag_engine
        rag = get_rag_engine()
        rag.initialize()

        patterns = rag.retrieve(query, top_k=top_k)
        return success_response({
            "query": query,
            "patterns": patterns,
            "total": len(patterns)
        })
    except Exception as e:
        return success_response({
            "query": query,
            "patterns": [],
            "total": 0,
            "error": str(e)
        })


@router.get(
    "/dashboard/insights",
    summary="Get learning insights",
)
async def get_insights(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get AI-generated learning insights.

    Returns:
    - Learning velocity analysis
    - Skill gaps identification
    - Study schedule recommendations
    """
    try:
        from code_tutor.ml import get_learning_predictor
        predictor = get_learning_predictor()

        # Get user's daily stats (simplified - would come from DB in production)
        # For now, return mock insights
        insights = {
            "velocity": {
                "velocity": "steady",
                "problems_per_day": 2.5,
                "improvement_rate": 5.0,
                "consistency_score": 75
            },
            "skill_gaps": [],
            "study_recommendations": [
                {
                    "type": "consistency",
                    "message": "매일 꾸준히 학습하면 더 빠른 성장이 가능합니다.",
                }
            ],
            "insights": [
                {
                    "type": "trend",
                    "message": "지속적인 학습으로 실력이 향상되고 있습니다!",
                    "sentiment": "positive"
                }
            ]
        }

        return success_response(insights)
    except Exception as e:
        return success_response({
            "velocity": None,
            "skill_gaps": [],
            "study_recommendations": [],
            "insights": [],
            "error": str(e)
        })
