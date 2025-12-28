"""AI Tutor DTOs (Data Transfer Objects)"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from code_tutor.tutor.domain.value_objects import ConversationType


# Request DTOs
class CodeContextRequest(BaseModel):
    """Code context for chat request"""
    code: str
    language: str = "python"
    problem_id: UUID | None = None
    submission_id: UUID | None = None


class ChatRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    conversation_type: ConversationType = ConversationType.GENERAL
    problem_id: UUID | None = None
    code_context: CodeContextRequest | None = None


class CreateConversationRequest(BaseModel):
    """Create new conversation request"""
    conversation_type: ConversationType = ConversationType.GENERAL
    problem_id: UUID | None = None
    title: str = ""


# Response DTOs
class MessageResponse(BaseModel):
    """Message response"""
    id: UUID
    role: str
    content: str
    code_context: CodeContextRequest | None = None
    tokens_used: int
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Full conversation response"""
    id: UUID
    user_id: UUID
    problem_id: UUID | None
    conversation_type: str
    title: str
    messages: list[MessageResponse]
    total_tokens: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationSummaryResponse(BaseModel):
    """Conversation summary for list views"""
    id: UUID
    problem_id: UUID | None
    conversation_type: str
    title: str
    message_count: int
    is_active: bool
    updated_at: datetime


class ChatResponse(BaseModel):
    """Chat response"""
    conversation_id: UUID
    message: MessageResponse
    is_new_conversation: bool = False
