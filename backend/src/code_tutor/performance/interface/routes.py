"""Performance analysis API routes."""

from fastapi import APIRouter

from code_tutor.shared.api_response import success_response

from ..application import (
    AnalyzeRequest,
    QuickAnalyzeRequest,
    performance_service,
)

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.post("")
async def analyze_performance(request: AnalyzeRequest) -> dict:
    """Perform full performance analysis on code.

    Includes:
    - Static complexity analysis (time and space)
    - Runtime profiling (execution time, function calls)
    - Memory profiling (peak usage, allocations)
    - Performance issues and suggestions
    """
    result = await performance_service.analyze(request)
    return success_response(result.model_dump())


@router.post("/quick")
async def quick_analyze(request: QuickAnalyzeRequest) -> dict:
    """Perform quick complexity-only analysis.

    Returns time and space complexity without runtime profiling.
    Faster than full analysis, suitable for real-time feedback.
    """
    result = await performance_service.quick_analyze(request)
    return success_response(result.model_dump())


@router.post("/complexity")
async def analyze_complexity_only(request: QuickAnalyzeRequest) -> dict:
    """Analyze only time and space complexity.

    Same as /quick but with more explicit naming.
    """
    result = await performance_service.quick_analyze(request)
    return success_response(result.model_dump())
