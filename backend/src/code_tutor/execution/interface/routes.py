"""Code Execution API routes"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from code_tutor.execution.application.dto import ExecuteCodeRequest
from code_tutor.execution.application.services import ExecutionService
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.shared.config import get_settings
from code_tutor.shared.middleware import code_execution_rate_limit

router = APIRouter(prefix="/execute", tags=["Code Execution"])


def get_execution_service() -> ExecutionService:
    """Get execution service instance"""
    settings = get_settings()
    # Use Docker in production, mock in development
    use_docker = settings.ENVIRONMENT != "development"
    return ExecutionService(use_docker=use_docker)


@router.post(
    "/run",
    summary="코드 실행",
    description="샌드박스 환경에서 코드를 안전하게 실행합니다. Docker 컨테이너에서 격리 실행되며, 네트워크 접근이 차단되고 리소스 제한이 적용됩니다.",
    responses={
        200: {"description": "실행 결과 반환 (stdout, stderr, 실행시간)"},
        401: {"description": "인증 필요"},
        400: {"description": "잘못된 코드 또는 언어"},
        429: {"description": "요청 한도 초과 (20/min)"},
    },
)
async def execute_code(
    request: ExecuteCodeRequest,
    service: Annotated[ExecutionService, Depends(get_execution_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _: Annotated[None, Depends(code_execution_rate_limit)],
) -> dict[str, Any]:
    """
    Execute code in a sandboxed environment.

    - Code is run in an isolated Docker container
    - Network access is disabled
    - Resource limits are enforced
    - Maximum execution time is 30 seconds
    - Rate limited to 20 requests per minute
    """
    result = await service.execute_code(request)
    return result.model_dump()
