"""Debugger API routes."""


from fastapi import APIRouter, HTTPException, Query

from code_tutor.debugger.application import (
    DebugRequest,
    debug_service,
)
from code_tutor.shared.api_response import success_response

router = APIRouter(prefix="/debugger", tags=["Debugger"])


@router.post(
    "",
    response_model=dict,
    summary="코드 디버깅 실행",
    description="코드를 단계별로 실행하며 각 스텝의 변수 값, 콜스택, 출력을 기록합니다.",
    responses={
        200: {"description": "디버깅 결과 반환 (모든 스텝 정보 포함)"},
    },
)
async def debug_code(request: DebugRequest):
    """
    Execute code with step-by-step debugging.

    Returns all execution steps including:
    - Line numbers executed
    - Variable values at each step
    - Call stack at each step
    - Output produced
    """
    result = await debug_service.debug_code(request)
    return success_response(result.model_dump())


@router.get(
    "/{session_id}",
    response_model=dict,
    summary="디버그 세션 조회",
    description="세션 ID로 저장된 디버그 세션 정보를 조회합니다.",
    responses={
        200: {"description": "디버그 세션 정보 반환"},
        404: {"description": "세션을 찾을 수 없음"},
    },
)
async def get_debug_session(session_id: str):
    """Get a debug session by ID."""
    result = await debug_service.get_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Debug session not found")
    return success_response(result.model_dump())


@router.get(
    "/{session_id}/step/{step_number}",
    response_model=dict,
    summary="특정 스텝 조회",
    description="디버그 세션의 특정 스텝 정보(변수, 콜스택, 실행 라인)를 조회합니다.",
    responses={
        200: {"description": "스텝 정보 반환"},
        404: {"description": "스텝을 찾을 수 없음"},
    },
)
async def get_step(
    session_id: str,
    step_number: int,
    breakpoints: str | None = Query(None, description="Comma-separated line numbers"),
):
    """Get information about a specific step."""
    bp_list = []
    if breakpoints:
        bp_list = [int(x.strip()) for x in breakpoints.split(",") if x.strip().isdigit()]

    result = await debug_service.get_step(session_id, step_number, bp_list)
    if not result:
        raise HTTPException(status_code=404, detail="Step not found")
    return success_response(result.model_dump())


@router.get(
    "/{session_id}/summary",
    response_model=dict,
    summary="디버그 세션 요약",
    description="디버그 세션의 요약 정보(총 스텝 수, 실행 시간, 최종 출력 등)를 조회합니다.",
    responses={
        200: {"description": "세션 요약 반환"},
        404: {"description": "세션을 찾을 수 없음"},
    },
)
async def get_debug_summary(session_id: str):
    """Get a summary of the debug session."""
    result = await debug_service.get_summary(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Debug session not found")
    return success_response(result.model_dump())


@router.post(
    "/quick",
    response_model=dict,
    summary="빠른 디버깅",
    description="세션 저장 없이 간단하게 디버깅합니다. 실시간 피드백에 적합합니다.",
    responses={
        200: {"description": "간소화된 디버깅 결과 반환"},
    },
)
async def quick_debug(request: DebugRequest):
    """
    Quick debug that returns a compact response.

    Good for simple debugging without storing session.
    """
    result = await debug_service.debug_code(request)

    # Return compact format
    compact = {
        "status": result.status.value,
        "total_steps": result.total_steps,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
        "steps": [
            {
                "step": s.step_number,
                "line": s.line_number,
                "code": s.line_content,
                "type": s.step_type.value,
                "vars": {v.name: v.value for v in s.variables},
            }
            for s in result.steps
        ],
    }

    return success_response(compact)
