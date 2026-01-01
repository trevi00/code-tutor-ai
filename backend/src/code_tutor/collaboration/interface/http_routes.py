"""HTTP routes for collaboration session management."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from code_tutor.identity.interface.dependencies import get_current_user
from code_tutor.identity.application.dto import UserResponse
from code_tutor.collaboration.application.dto import (
    CreateSessionRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
)
from code_tutor.collaboration.application.services import CollaborationService
from code_tutor.collaboration.infrastructure.repository import (
    SQLAlchemyCollaborationRepository,
)
from code_tutor.shared.infrastructure.database import get_async_session

router = APIRouter(prefix="/collaboration", tags=["collaboration"])


async def get_collaboration_service(db=Depends(get_async_session)) -> CollaborationService:
    """Get collaboration service with repository."""
    repository = SQLAlchemyCollaborationRepository(db)
    return CollaborationService(repository)


@router.post("/sessions", response_model=SessionDetailResponse)
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


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    active_only: bool = True,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionListResponse:
    """List user's collaboration sessions."""
    sessions = await service.get_user_sessions(current_user.id, active_only)
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/active", response_model=SessionListResponse)
async def list_active_sessions(
    limit: int = 10,
    current_user: UserResponse = Depends(get_current_user),
    service: CollaborationService = Depends(get_collaboration_service),
) -> SessionListResponse:
    """List active public sessions."""
    sessions = await service.get_active_sessions(limit)
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
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


@router.post("/sessions/{session_id}/join", response_model=SessionDetailResponse)
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


@router.post("/sessions/{session_id}/leave")
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


@router.post("/sessions/{session_id}/close")
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
