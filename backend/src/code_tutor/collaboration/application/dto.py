"""Collaboration DTOs for request/response handling."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# === Request DTOs ===


class CreateSessionRequest(BaseModel):
    """Request to create a collaboration session."""

    title: str = Field(..., min_length=1, max_length=255)
    problem_id: UUID | None = None
    language: str = Field(default="python", max_length=20)
    max_participants: int = Field(default=5, ge=2, le=10)


class JoinSessionRequest(BaseModel):
    """Request to join a collaboration session."""

    session_id: UUID


class CodeChangeRequest(BaseModel):
    """Request to apply a code change."""

    operation_type: str  # insert, delete, replace
    position: int = Field(..., ge=0)
    content: str = ""
    length: int = Field(default=0, ge=0)


class CursorUpdateRequest(BaseModel):
    """Request to update cursor position."""

    line: int = Field(..., ge=0)
    column: int = Field(..., ge=0)


class SelectionUpdateRequest(BaseModel):
    """Request to update selection range."""

    start_line: int = Field(..., ge=0)
    start_column: int = Field(..., ge=0)
    end_line: int = Field(..., ge=0)
    end_column: int = Field(..., ge=0)


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, max_length=1000)


# === Response DTOs ===


class ParticipantResponse(BaseModel):
    """Response for a session participant."""

    id: UUID
    user_id: UUID
    username: str
    cursor_line: int | None = None
    cursor_column: int | None = None
    selection_start_line: int | None = None
    selection_start_column: int | None = None
    selection_end_line: int | None = None
    selection_end_column: int | None = None
    is_active: bool
    color: str
    joined_at: datetime

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    """Response for a collaboration session."""

    id: UUID
    problem_id: UUID | None
    host_id: UUID
    title: str
    status: str
    language: str
    version: int
    participant_count: int
    max_participants: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """Detailed response for a collaboration session."""

    code_content: str
    participants: list[ParticipantResponse]


class SessionListResponse(BaseModel):
    """Response for list of sessions."""

    sessions: list[SessionResponse]
    total: int


class CodeSyncResponse(BaseModel):
    """Response for code synchronization."""

    code_content: str
    version: int
    participants: list[ParticipantResponse]


# === WebSocket Message DTOs ===


class WebSocketMessage(BaseModel):
    """Base WebSocket message."""

    type: str
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserJoinedData(BaseModel):
    """Data for user joined event."""

    user_id: str
    username: str
    color: str


class UserLeftData(BaseModel):
    """Data for user left event."""

    user_id: str
    username: str


class CodeUpdateData(BaseModel):
    """Data for code update event."""

    user_id: str
    username: str
    operation_type: str
    position: int
    content: str
    length: int
    version: int


class CursorUpdateData(BaseModel):
    """Data for cursor update event."""

    user_id: str
    username: str
    line: int
    column: int
    color: str


class SelectionUpdateData(BaseModel):
    """Data for selection update event."""

    user_id: str
    username: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    color: str


class ChatMessageData(BaseModel):
    """Data for chat message event."""

    user_id: str
    username: str
    message: str
    timestamp: datetime
