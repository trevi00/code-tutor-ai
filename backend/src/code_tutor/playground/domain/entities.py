"""Playground domain entities."""

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)


def generate_share_code() -> str:
    """Generate a unique share code."""
    return secrets.token_urlsafe(8)


@dataclass
class Playground:
    """Playground entity - a saved code snippet."""

    id: UUID
    owner_id: UUID
    title: str
    description: str
    code: str
    language: PlaygroundLanguage
    visibility: PlaygroundVisibility = PlaygroundVisibility.PRIVATE
    share_code: str = field(default_factory=generate_share_code)
    stdin: str = ""
    is_forked: bool = False
    forked_from_id: UUID | None = None
    run_count: int = 0
    fork_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        owner_id: UUID,
        title: str,
        code: str,
        language: PlaygroundLanguage,
        description: str = "",
        visibility: PlaygroundVisibility = PlaygroundVisibility.PRIVATE,
        stdin: str = "",
    ) -> "Playground":
        """Factory method to create a new playground."""
        return cls(
            id=uuid4(),
            owner_id=owner_id,
            title=title,
            description=description,
            code=code,
            language=language,
            visibility=visibility,
            stdin=stdin,
        )

    def fork(self, new_owner_id: UUID) -> "Playground":
        """Create a fork of this playground."""
        self.fork_count += 1
        return Playground(
            id=uuid4(),
            owner_id=new_owner_id,
            title=f"{self.title} (Fork)",
            description=self.description,
            code=self.code,
            language=self.language,
            visibility=PlaygroundVisibility.PRIVATE,
            stdin=self.stdin,
            is_forked=True,
            forked_from_id=self.id,
        )

    def update(
        self,
        title: str | None = None,
        description: str | None = None,
        code: str | None = None,
        language: PlaygroundLanguage | None = None,
        visibility: PlaygroundVisibility | None = None,
        stdin: str | None = None,
    ) -> None:
        """Update playground fields."""
        if title is not None:
            self.title = title
        if description is not None:
            self.description = description
        if code is not None:
            self.code = code
        if language is not None:
            self.language = language
        if visibility is not None:
            self.visibility = visibility
        if stdin is not None:
            self.stdin = stdin
        self.updated_at = utc_now()

    def increment_run_count(self) -> None:
        """Increment the run count."""
        self.run_count += 1

    def regenerate_share_code(self) -> str:
        """Regenerate the share code."""
        self.share_code = generate_share_code()
        return self.share_code

    def is_owner(self, user_id: UUID) -> bool:
        """Check if user is the owner."""
        return self.owner_id == user_id

    def can_view(self, user_id: UUID | None) -> bool:
        """Check if user can view this playground."""
        if self.visibility == PlaygroundVisibility.PUBLIC:
            return True
        if self.visibility == PlaygroundVisibility.UNLISTED:
            return True  # Anyone with link can view
        return user_id is not None and self.is_owner(user_id)

    def can_edit(self, user_id: UUID | None) -> bool:
        """Check if user can edit this playground."""
        return user_id is not None and self.is_owner(user_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id),
            "title": self.title,
            "description": self.description,
            "code": self.code,
            "language": self.language.value,
            "visibility": self.visibility.value,
            "share_code": self.share_code,
            "stdin": self.stdin,
            "is_forked": self.is_forked,
            "forked_from_id": str(self.forked_from_id) if self.forked_from_id else None,
            "run_count": self.run_count,
            "fork_count": self.fork_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class CodeTemplate:
    """Predefined code template."""

    id: UUID
    title: str
    description: str
    code: str
    language: PlaygroundLanguage
    category: TemplateCategory
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def increment_usage(self) -> None:
        """Increment usage count."""
        self.usage_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "code": self.code,
            "language": self.language.value,
            "category": self.category.value,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ExecutionHistory:
    """Record of playground execution."""

    id: UUID
    playground_id: UUID
    user_id: UUID | None
    code: str
    stdin: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    is_success: bool
    executed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "playground_id": str(self.playground_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "is_success": self.is_success,
            "executed_at": self.executed_at.isoformat(),
        }
