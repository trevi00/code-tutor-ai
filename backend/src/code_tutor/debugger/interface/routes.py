"""Debugger API routes."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from code_tutor.shared.api_response import success_response
from code_tutor.debugger.application import (
    DebugRequest,
    DebugResponse,
    StepInfoResponse,
    DebugSummaryResponse,
    debug_service,
)

router = APIRouter(prefix="/debugger", tags=["Debugger"])


@router.post("", response_model=dict)
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


@router.get("/{session_id}", response_model=dict)
async def get_debug_session(session_id: str):
    """Get a debug session by ID."""
    result = await debug_service.get_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Debug session not found")
    return success_response(result.model_dump())


@router.get("/{session_id}/step/{step_number}", response_model=dict)
async def get_step(
    session_id: str,
    step_number: int,
    breakpoints: Optional[str] = Query(None, description="Comma-separated line numbers"),
):
    """Get information about a specific step."""
    bp_list = []
    if breakpoints:
        bp_list = [int(x.strip()) for x in breakpoints.split(",") if x.strip().isdigit()]

    result = await debug_service.get_step(session_id, step_number, bp_list)
    if not result:
        raise HTTPException(status_code=404, detail="Step not found")
    return success_response(result.model_dump())


@router.get("/{session_id}/summary", response_model=dict)
async def get_debug_summary(session_id: str):
    """Get a summary of the debug session."""
    result = await debug_service.get_summary(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Debug session not found")
    return success_response(result.model_dump())


@router.post("/quick", response_model=dict)
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
