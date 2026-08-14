"""ML-powered API routes.

These endpoints expose the ml component's services (recommendation, insights,
code quality analysis, pattern knowledge base). They were extracted from the
learning interface so the dependency stays one-directional: ml -> learning.

IMPORTANT: URL paths must remain identical to their original locations under
the learning router; this router is mounted with the same "/api/v1" prefix
and defines the same paths.
"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.gamification.application.services import XPService
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.learning.application.dto import (
    CreateSubmissionRequest,
    RecommendedProblemResponse,
    SubmissionResponse,
)
from code_tutor.learning.application.services import SubmissionService
from code_tutor.learning.application.submission_evaluator import SubmissionEvaluator
from code_tutor.learning.interface.routes import (
    get_submission_evaluator,
    get_submission_service,
    get_xp_service,
)
from code_tutor.ml.analysis import CodeQualityService, QualityRecommender
from code_tutor.ml.prediction import InsightsService
from code_tutor.ml.recommendation import RecommenderService
from code_tutor.shared.api_response import success_response
from code_tutor.shared.exceptions import AppException
from code_tutor.shared.infrastructure.database import get_async_session
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ML"])


# Dependencies
async def get_recommender_service(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> RecommenderService:
    return RecommenderService(session)


async def get_insights_service(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> InsightsService:
    return InsightsService(session)


async def get_quality_service(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> CodeQualityService:
    return CodeQualityService(session)


async def get_quality_recommender(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> QualityRecommender:
    return QualityRecommender(session)


@router.get(
    "/problems/recommended",
    response_model=list[RecommendedProblemResponse],
    summary="Get recommended problems for user",
)
async def get_recommended_problems(
    recommender: Annotated[RecommenderService, Depends(get_recommender_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=5, ge=1, le=20),
    strategy: str = Query(default="hybrid", regex="^(hybrid|collaborative|content)$"),
    difficulty: str | None = Query(default=None),
    category: str | None = Query(default=None),
) -> list[RecommendedProblemResponse]:
    """Get personalized problem recommendations based on user's history.

    Uses ML-based recommendation system with NCF collaborative filtering.

    Args:
        limit: Number of recommendations (1-20)
        strategy: Recommendation strategy (hybrid, collaborative, content)
        difficulty: Filter by difficulty (easy, medium, hard)
        category: Filter by category
    """
    recommendations = await recommender.get_recommendations(
        user_id=current_user.id,
        limit=limit,
        strategy=strategy,
        difficulty_filter=difficulty,
        category_filter=category,
    )

    return [
        RecommendedProblemResponse(
            id=UUID(rec["id"]),
            title=rec["title"],
            difficulty=rec["difficulty"],
            category=rec["category"],
            reason=_get_recommendation_reason_kr(rec.get("reason", "recommended")),
            score=rec.get("score", 0.5),
            pattern_ids=rec.get("pattern_ids", []),
        )
        for rec in recommendations
    ]


def _get_recommendation_reason_kr(reason: str) -> str:
    """Convert recommendation reason to Korean."""
    reasons = {
        "similar_users": "비슷한 사용자들이 풀었어요",
        "content_match": "당신의 학습 패턴에 맞는 문제예요",
        "hybrid": "AI가 추천하는 문제예요",
        "popular": "인기 있는 문제예요",
        "recommended": "추천 문제예요",
    }
    return reasons.get(reason, "추천 문제예요")


@router.get(
    "/problems/skill-gaps",
    summary="Get skill gaps for user",
)
async def get_skill_gaps(
    recommender: Annotated[RecommenderService, Depends(get_recommender_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> list[dict]:
    """Get categories/patterns where user has skill gaps.

    Returns categories with less than 30% completion.
    """
    return await recommender.get_skill_gaps(current_user.id)


@router.get(
    "/problems/next-challenge",
    summary="Get next challenge problem",
)
async def get_next_challenge(
    recommender: Annotated[RecommenderService, Depends(get_recommender_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict | None:
    """Get the next appropriate challenge problem for user.

    Returns a problem at the right difficulty level based on user's progress.
    """
    return await recommender.get_next_challenge(current_user.id)


@router.post(
    "/submit",
    response_model=SubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit and evaluate code",
)
async def submit_and_evaluate(
    request: CreateSubmissionRequest,
    service: Annotated[SubmissionService, Depends(get_submission_service)],
    evaluator: Annotated[SubmissionEvaluator, Depends(get_submission_evaluator)],
    recommender: Annotated[RecommenderService, Depends(get_recommender_service)],
    quality_service: Annotated[CodeQualityService, Depends(get_quality_service)],
    xp_service: Annotated[XPService, Depends(get_xp_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> SubmissionResponse:
    """
    Submit code and immediately evaluate against all test cases.

    This is a convenience endpoint that combines:
    1. POST /submissions (create submission)
    2. POST /submissions/{id}/evaluate (run evaluation)
    3. Analyze code quality (async, non-blocking)
    4. Update gamification stats (XP, badges) on success

    Returns the evaluated submission with status and test results.
    """
    try:
        # Check if user has already solved this problem (for gamification)
        already_solved = await service.has_user_solved(
            current_user.id, request.problem_id
        )

        # Check if this is the user's first submission for this problem
        previous_submissions = await service.get_user_problem_submissions(
            current_user.id, request.problem_id, limit=1
        )
        is_first_attempt = len(previous_submissions) == 0

        # Create submission
        submission = await service.create_submission(current_user.id, request)

        # Evaluate immediately
        await evaluator.evaluate_submission(submission.id)

        # Get updated submission with results
        result = await service.get_submission(submission.id)

        # Update recommender with new interaction
        is_solved = result.status == "accepted"
        await recommender.update_user_interaction(
            user_id=current_user.id,
            problem_id=request.problem_id,
            is_solved=is_solved,
        )

        # Update gamification stats if this is a NEW successful solve
        logger.info(
            "Gamification check",
            is_solved=is_solved,
            already_solved=already_solved,
            is_first_attempt=is_first_attempt,
        )
        if is_solved and not already_solved:
            try:
                # Award XP based on whether this was first attempt
                action = (
                    "problem_solved_first_try" if is_first_attempt else "problem_solved"
                )
                logger.info(
                    "Recording activity", action=action, user_id=str(current_user.id)
                )
                await xp_service.record_activity(current_user.id, action)
                logger.info("Activity recorded successfully")
            except Exception as e:
                logger.error(
                    "Gamification failed", error=str(e), error_type=type(e).__name__
                )
                pass  # Gamification failure shouldn't affect submission

        # Analyze code quality (best effort, don't fail on error)
        try:
            await quality_service.analyze_submission(
                submission_id=submission.id,
                user_id=current_user.id,
                problem_id=request.problem_id,
                code=request.code,
                language=request.language,
            )
        except Exception:
            pass  # Quality analysis failure shouldn't affect submission

        return result
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


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
        return success_response(
            {
                "patterns": [],
                "quality": {"score": 70, "grade": "C"},
                "complexity": {"cyclomatic": 1},
                "suggestions": ["코드 분석 기능을 사용할 수 없습니다."],
                "error": str(e),
            }
        )


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
        return success_response(
            {
                "overall_score": 70,
                "overall_grade": "C",
                "dimensions": {
                    "correctness": {"label": "fair", "score": 70},
                    "efficiency": {"label": "fair", "score": 70},
                    "readability": {"label": "fair", "score": 70},
                    "best_practices": {"label": "fair", "score": 70},
                },
                "suggestions": [],
                "error": str(e),
            }
        )


@router.get(
    "/patterns",
    summary="List algorithm patterns",
)
async def list_patterns() -> dict[str, Any]:
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
                "keywords": p["keywords"],
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
            detail=f"Pattern not found: {pattern_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post(
    "/patterns/search",
    summary="Search patterns by query",
)
async def search_patterns(
    query: str,
    top_k: int = Query(default=3, ge=1, le=10),
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
        return success_response(
            {"query": query, "patterns": patterns, "total": len(patterns)}
        )
    except Exception as e:
        return success_response(
            {"query": query, "patterns": [], "total": 0, "error": str(e)}
        )


@router.get(
    "/dashboard/insights",
    summary="Get learning insights",
)
async def get_insights(
    insights_service: Annotated[InsightsService, Depends(get_insights_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get AI-generated learning insights powered by LSTM predictions.

    Returns:
    - Learning velocity analysis (learning speed and trends)
    - Success rate predictions (7-day forecast)
    - Skill gaps identification
    - Personalized study schedule recommendations
    """
    try:
        insights = await insights_service.get_full_insights(current_user.id)
        return success_response(insights)
    except Exception as e:
        # Fallback to empty insights on error
        return success_response(
            {
                "velocity": None,
                "prediction": None,
                "schedule": None,
                "skill_gaps": [],
                "insights": [],
                "study_recommendations": [],
                "error": str(e),
            }
        )


# Code Quality Analysis endpoints
@router.get(
    "/submissions/{submission_id}/quality",
    summary="Get code quality analysis for submission",
)
async def get_submission_quality(
    submission_id: UUID,
    quality_service: Annotated[CodeQualityService, Depends(get_quality_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get code quality analysis for a specific submission.

    Returns:
    - Multi-dimensional quality scores (correctness, efficiency, readability, best practices)
    - Code smells detected
    - Complexity metrics
    - Improvement suggestions
    """
    analysis = await quality_service.get_submission_quality(submission_id)

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quality analysis not found for this submission",
        )

    return success_response(
        {
            "submission_id": str(analysis.submission_id),
            "overall_score": analysis.overall_score,
            "overall_grade": analysis.overall_grade,
            "dimensions": {
                "correctness": analysis.correctness_score,
                "efficiency": analysis.efficiency_score,
                "readability": analysis.readability_score,
                "best_practices": analysis.best_practices_score,
            },
            "complexity": {
                "cyclomatic": analysis.cyclomatic_complexity,
                "cognitive": analysis.cognitive_complexity,
                "max_nesting": analysis.max_nesting_depth,
                "lines_of_code": analysis.lines_of_code,
            },
            "code_smells": analysis.code_smells,
            "code_smells_count": analysis.code_smells_count,
            "detected_patterns": analysis.detected_patterns,
            "suggestions": analysis.suggestions,
            "suggestions_count": analysis.suggestions_count,
            "analyzed_at": analysis.analyzed_at.isoformat(),
        }
    )


@router.get(
    "/dashboard/quality",
    summary="Get user quality statistics",
)
async def get_quality_stats(
    quality_service: Annotated[CodeQualityService, Depends(get_quality_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get aggregated code quality statistics for current user.

    Returns:
    - Average scores across all dimensions
    - Grade distribution
    - Total analyses count
    """
    stats = await quality_service.get_user_quality_stats(current_user.id)
    return success_response(stats)


@router.get(
    "/dashboard/quality/trends",
    summary="Get quality trends over time",
)
async def get_quality_trends(
    quality_service: Annotated[CodeQualityService, Depends(get_quality_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    days: int = Query(default=30, ge=7, le=90),
) -> dict[str, Any]:
    """
    Get code quality score trends over time.

    Args:
        days: Number of days to look back (7-90)

    Returns:
    - Daily quality metrics for charting
    """
    trends = await quality_service.get_quality_trends(current_user.id, days)
    return success_response({"trends": trends, "days": days})


@router.get(
    "/dashboard/quality/recent",
    summary="Get recent quality analyses",
)
async def get_recent_quality(
    quality_service: Annotated[CodeQualityService, Depends(get_quality_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """
    Get recent code quality analyses for current user.

    Returns:
    - List of recent analyses with scores
    """
    analyses = await quality_service.get_recent_analyses(current_user.id, limit)

    return success_response(
        {
            "analyses": [
                {
                    "submission_id": str(a.submission_id),
                    "problem_id": str(a.problem_id),
                    "overall_score": a.overall_score,
                    "overall_grade": a.overall_grade,
                    "code_smells_count": a.code_smells_count,
                    "suggestions_count": a.suggestions_count,
                    "analyzed_at": a.analyzed_at.isoformat(),
                }
                for a in analyses
            ],
            "total": len(analyses),
        }
    )


# Quality-based recommendations
@router.get(
    "/dashboard/quality/profile",
    summary="Get user quality profile",
)
async def get_quality_profile(
    quality_recommender: Annotated[
        QualityRecommender, Depends(get_quality_recommender)
    ],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get user's code quality profile based on submission history.

    Returns:
    - Average dimension scores
    - Weak and strong areas
    - Common code smells
    - Improvement trend
    """
    profile = await quality_recommender.get_quality_profile(current_user.id)
    return success_response(profile)


@router.get(
    "/dashboard/quality/recommendations",
    summary="Get quality-based problem recommendations",
)
async def get_quality_recommendations(
    quality_recommender: Annotated[
        QualityRecommender, Depends(get_quality_recommender)
    ],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    """
    Get personalized problem recommendations based on code quality analysis.

    Recommends problems that will help improve weak quality dimensions.

    Returns:
    - List of recommended problems with quality focus reasons
    """
    recommendations = await quality_recommender.get_quality_recommendations(
        current_user.id, limit
    )
    return success_response(
        {"recommendations": recommendations, "total": len(recommendations)}
    )


@router.get(
    "/dashboard/quality/suggestions",
    summary="Get improvement suggestions",
)
async def get_improvement_suggestions(
    quality_recommender: Annotated[
        QualityRecommender, Depends(get_quality_recommender)
    ],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Get personalized code quality improvement suggestions.

    Returns:
    - Actionable suggestions for improving code quality
    - Tips based on common code smells
    - Dimension-specific advice
    """
    suggestions = await quality_recommender.get_improvement_suggestions(current_user.id)
    return success_response({"suggestions": suggestions, "total": len(suggestions)})
