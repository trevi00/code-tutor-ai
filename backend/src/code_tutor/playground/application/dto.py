"""Playground DTOs for request/response handling."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# === Request DTOs ===


class CreatePlaygroundRequest(BaseModel):
    """Request to create a new playground."""

    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    code: str = Field(default="")
    language: str = Field(default="python")
    visibility: str = Field(default="private")
    stdin: str = Field(default="")


class UpdatePlaygroundRequest(BaseModel):
    """Request to update a playground."""

    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=2000)
    code: str | None = None
    language: str | None = None
    visibility: str | None = None
    stdin: str | None = None


class ExecutePlaygroundRequest(BaseModel):
    """Request to execute playground code."""

    code: str | None = None  # If None, use saved code
    stdin: str = Field(default="")
    timeout_seconds: int = Field(default=10, ge=1, le=30)


class ForkPlaygroundRequest(BaseModel):
    """Request to fork a playground."""

    title: str | None = None  # If None, use original title + (Fork)


# === Response DTOs ===


class PlaygroundResponse(BaseModel):
    """Basic playground response."""

    id: UUID
    owner_id: UUID
    title: str
    description: str
    language: str
    visibility: str
    share_code: str
    is_forked: bool
    forked_from_id: UUID | None
    run_count: int
    fork_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PlaygroundDetailResponse(PlaygroundResponse):
    """Detailed playground response with code."""

    code: str
    stdin: str


class PlaygroundListResponse(BaseModel):
    """Response for list of playgrounds."""

    playgrounds: list[PlaygroundResponse]
    total: int


class ExecutionResponse(BaseModel):
    """Response for code execution."""

    execution_id: UUID
    status: str  # success, error, timeout
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    is_success: bool


class TemplateResponse(BaseModel):
    """Response for code template."""

    id: UUID
    title: str
    description: str
    code: str
    language: str
    category: str
    tags: list[str]
    usage_count: int

    class Config:
        from_attributes = True


class TemplateListResponse(BaseModel):
    """Response for list of templates."""

    templates: list[TemplateResponse]
    total: int


class LanguageInfo(BaseModel):
    """Information about a supported language."""

    id: str
    display_name: str
    extension: str


class LanguagesResponse(BaseModel):
    """Response for supported languages."""

    languages: list[LanguageInfo]
