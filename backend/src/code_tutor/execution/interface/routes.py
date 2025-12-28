"""Code Execution API routes"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from code_tutor.execution.application.dto import ExecuteCodeRequest, ExecuteCodeResponse
from code_tutor.execution.application.services import ExecutionService
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.shared.api_response import success_response
from code_tutor.shared.config import get_settings

router = APIRouter(prefix="/execute", tags=["Code Execution"])


def get_execution_service() -> ExecutionService:
    """Get execution service instance"""
    settings = get_settings()
    # Use Docker in production, mock in development
    use_docker = settings.ENVIRONMENT != "development"
    return ExecutionService(use_docker=use_docker)


@router.post(
    "/run",
    summary="Execute code in sandbox",
)
async def execute_code(
    request: ExecuteCodeRequest,
    service: Annotated[ExecutionService, Depends(get_execution_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> dict[str, Any]:
    """
    Execute code in a sandboxed environment.

    - Code is run in an isolated Docker container
    - Network access is disabled
    - Resource limits are enforced
    - Maximum execution time is 30 seconds
    """
    result = await service.execute_code(request)
    return success_response(result.model_dump())
