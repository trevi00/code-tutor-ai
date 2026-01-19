"""AI Tutor API routes"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_active_user
from code_tutor.shared.exceptions import AppException
from code_tutor.shared.infrastructure.database import get_async_session
from code_tutor.shared.middleware import ai_chat_rate_limit, ai_review_rate_limit
from code_tutor.tutor.application.dto import (
    ChatRequest,
    ChatResponse,
    CodeReviewRequest,
    CodeReviewResponse,
    ConversationResponse,
    ConversationSummaryResponse,
)
from code_tutor.tutor.application.services import TutorService
from code_tutor.tutor.domain.repository import ConversationRepository
from code_tutor.tutor.infrastructure.repository import SQLAlchemyConversationRepository

router = APIRouter(prefix="/tutor", tags=["AI Tutor"])


# Dependencies
async def get_conversation_repository(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ConversationRepository:
    return SQLAlchemyConversationRepository(session)


async def get_tutor_service(
    repo: Annotated[ConversationRepository, Depends(get_conversation_repository)],
) -> TutorService:
    return TutorService(repo)


# Endpoints
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    responses={429: {"description": "요청 한도 초과 (10/min)"}},
)
async def chat(
    request: ChatRequest,
    service: Annotated[TutorService, Depends(get_tutor_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _: Annotated[None, Depends(ai_chat_rate_limit)],
) -> ChatResponse:
    """
    Send a message and get AI tutor response.

    Rate limited to 10 messages per minute.
    """
    try:
        return await service.chat(current_user.id, request)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)


@router.get(
    "/conversations",
    response_model=list[ConversationSummaryResponse],
    summary="List my conversations",
)
async def list_conversations(
    service: Annotated[TutorService, Depends(get_tutor_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ConversationSummaryResponse]:
    """List current user's conversations"""
    return await service.list_conversations(current_user.id, limit, offset)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get conversation details",
)
async def get_conversation(
    conversation_id: UUID,
    service: Annotated[TutorService, Depends(get_tutor_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> ConversationResponse:
    """Get conversation with all messages"""
    try:
        return await service.get_conversation(current_user.id, conversation_id)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post(
    "/conversations/{conversation_id}/close",
    response_model=ConversationResponse,
    summary="Close a conversation",
)
async def close_conversation(
    conversation_id: UUID,
    service: Annotated[TutorService, Depends(get_tutor_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
) -> ConversationResponse:
    """Close a conversation"""
    try:
        return await service.close_conversation(current_user.id, conversation_id)
    except AppException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post(
    "/review",
    response_model=CodeReviewResponse,
    summary="Get AI code review",
    responses={429: {"description": "요청 한도 초과 (5/min)"}},
)
async def review_code(
    request: CodeReviewRequest,
    service: Annotated[TutorService, Depends(get_tutor_service)],
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    _: Annotated[None, Depends(ai_review_rate_limit)],
) -> CodeReviewResponse:
    """
    Get AI-powered code review with feedback on:
    - Code quality and style
    - Potential bugs and issues
    - Performance suggestions
    - Best practices

    Rate limited to 5 reviews per minute (expensive operation).
    """
    return await service.review_code(current_user.id, request)
