"""Collaboration module for real-time coding sessions."""

from code_tutor.collaboration.application import (
    CollaborationService,
    CreateSessionRequest,
    SessionDetailResponse,
    SessionResponse,
)
from code_tutor.collaboration.domain import (
    CodeChange,
    CodeOperation,
    CollaborationRepository,
    CollaborationSession,
    CursorPosition,
    OperationType,
    Participant,
    SelectionRange,
    SessionStatus,
)
from code_tutor.collaboration.interface import http_router, websocket_router

__all__ = [
    # Domain
    "CollaborationSession",
    "Participant",
    "CodeChange",
    "CollaborationRepository",
    "SessionStatus",
    "OperationType",
    "CursorPosition",
    "SelectionRange",
    "CodeOperation",
    # Application
    "CollaborationService",
    "CreateSessionRequest",
    "SessionResponse",
    "SessionDetailResponse",
    # Interface
    "http_router",
    "websocket_router",
]
