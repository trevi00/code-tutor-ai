"""Performance analysis API routes."""

from fastapi import APIRouter

from code_tutor.shared.api_response import success_response

from ..application import (
    AnalyzeRequest,
    QuickAnalyzeRequest,
    performance_service,
)

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.post(
    "",
    summary="전체 성능 분석",
    description="코드의 전체 성능을 분석합니다. 시간/공간 복잡도, 런타임 프로파일링, 메모리 사용량, 최적화 제안을 포함합니다.",
    responses={
        200: {"description": "성능 분석 결과 반환"},
    },
)
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


@router.post(
    "/quick",
    summary="빠른 복잡도 분석",
    description="런타임 프로파일링 없이 시간/공간 복잡도만 빠르게 분석합니다. 실시간 피드백에 적합합니다.",
    responses={
        200: {"description": "복잡도 분석 결과 반환"},
    },
)
async def quick_analyze(request: QuickAnalyzeRequest) -> dict:
    """Perform quick complexity-only analysis.

    Returns time and space complexity without runtime profiling.
    Faster than full analysis, suitable for real-time feedback.
    """
    result = await performance_service.quick_analyze(request)
    return success_response(result.model_dump())


@router.post(
    "/complexity",
    summary="복잡도만 분석",
    description="시간 및 공간 복잡도만 분석합니다. /quick과 동일하지만 명시적인 이름입니다.",
    responses={
        200: {"description": "복잡도 분석 결과 반환"},
    },
)
async def analyze_complexity_only(request: QuickAnalyzeRequest) -> dict:
    """Analyze only time and space complexity.

    Same as /quick but with more explicit naming.
    """
    result = await performance_service.quick_analyze(request)
    return success_response(result.model_dump())
