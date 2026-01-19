"""HTTP routes for collaboration session management."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from code_tutor.collaboration.application.dto import (
    CreateSessionRequest,
    SessionDetailResponse,
    SessionListResponse,
)
from code_tutor.collaboration.application.services import CollaborationService
from code_tutor.collaboration.infrastructure.repository import (
    SQLAlchemyCollaborationRepository,
)
from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_user
from code_tutor.shared.infrastructure.database import get_async_session

router = APIRouter(prefix="/collaboration", tags=["Collaboration"])


async def get_collaboration_service(db=Depends(get_async_session)) -> CollaborationService:
    """Get collaboration service with repository."""
    repository = SQLAlchemyCollaborationRepository(db)
    return CollaborationService(repository)


@router.post(
    "/sessions",
    response_model=SessionDetailResponse,
    summary="협업 세션 생성",
    description="새로운 실시간 협업 세션을 생성합니다. 생성자가 호스트가 됩니다.",
    responses={
        200: {"description": "세션 생성 성공"},
        401: {"description": "인증 필요"},
    },
)
async def create_session(
    request: CreateSessionRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionDetailResponse:
    """Create a new collaboration session."""
    return await service.create_session(
        request,
        current_user.id,
        current_user.username,
    )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="내 세션 목록 조회",
    description="현재 사용자가 참여한 협업 세션 목록을 조회합니다.",
    responses={
        200: {"description": "세션 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def list_sessions(
    active_only: bool = True,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionListResponse:
    """List user's collaboration sessions."""
    sessions = await service.get_user_sessions(current_user.id, active_only)
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/active",
    response_model=SessionListResponse,
    summary="활성 공개 세션 목록",
    description="현재 활성화된 공개 협업 세션 목록을 조회합니다. 참여 가능한 세션을 찾을 때 사용합니다.",
    responses={
        200: {"description": "활성 세션 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def list_active_sessions(
    limit: int = 10,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionListResponse:
    """List active public sessions."""
    sessions = await service.get_active_sessions(limit)
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="세션 상세 조회",
    description="특정 협업 세션의 상세 정보(참여자, 코드 상태 등)를 조회합니다.",
    responses={
        200: {"description": "세션 상세 정보"},
        401: {"description": "인증 필요"},
        404: {"description": "세션을 찾을 수 없음"},
    },
)
async def get_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionDetailResponse:
    """Get session details."""
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


@router.post(
    "/sessions/{session_id}/join",
    response_model=SessionDetailResponse,
    summary="세션 참여",
    description="기존 협업 세션에 참여합니다. 참여 후 WebSocket으로 실시간 협업이 가능합니다.",
    responses={
        200: {"description": "세션 참여 성공"},
        400: {"description": "세션이 가득 찼거나 참여 불가"},
        401: {"description": "인증 필요"},
        404: {"description": "세션을 찾을 수 없음"},
    },
)
async def join_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionDetailResponse:
    """Join a collaboration session."""
    try:
        session = await service.join_session(
            session_id,
            current_user.id,
            current_user.username,
        )
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return session
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/sessions/{session_id}/leave",
    summary="세션 나가기",
    description="현재 참여 중인 협업 세션에서 나갑니다.",
    responses={
        200: {"description": "세션 나가기 성공"},
        401: {"description": "인증 필요"},
        404: {"description": "세션을 찾을 수 없거나 참여 중이 아님"},
    },
)
async def leave_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> dict:
    """Leave a collaboration session."""
    success = await service.leave_session(session_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or user not in session",
        )
    return {"status": "left"}


@router.post(
    "/sessions/{session_id}/close",
    summary="세션 종료 (호스트 전용)",
    description="협업 세션을 종료합니다. 세션 호스트만 종료할 수 있습니다.",
    responses={
        200: {"description": "세션 종료 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "호스트만 종료 가능"},
        404: {"description": "세션을 찾을 수 없음"},
    },
)
async def close_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> dict:
    """Close a collaboration session (host only)."""
    try:
        success = await service.close_session(session_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        return {"status": "closed"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
