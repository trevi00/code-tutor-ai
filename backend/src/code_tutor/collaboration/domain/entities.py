"""Collaboration domain entities."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)

from code_tutor.collaboration.domain.value_objects import (
    CodeOperation,
    CursorPosition,
    SelectionRange,
    SessionStatus,
    get_participant_color,
)


@dataclass
class Participant:
    """Session participant entity."""

    id: UUID
    user_id: UUID
    session_id: UUID
    username: str
    cursor_position: CursorPosition | None = None
    selection_range: SelectionRange | None = None
    is_active: bool = True
    color: str = ""
    joined_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.color:
            # Will be set when added to session
            pass

    def update_cursor(self, position: CursorPosition) -> None:
        """Update cursor position."""
        self.cursor_position = position

    def update_selection(self, selection: SelectionRange | None) -> None:
        """Update selection range."""
        self.selection_range = selection

    def deactivate(self) -> None:
        """Mark participant as inactive."""
        self.is_active = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "username": self.username,
            "cursor_position": self.cursor_position.to_dict() if self.cursor_position else None,
            "selection_range": self.selection_range.to_dict() if self.selection_range else None,
            "is_active": self.is_active,
            "color": self.color,
            "joined_at": self.joined_at.isoformat(),
        }


@dataclass
class CodeChange:
    """Code change record for history."""

    id: UUID
    session_id: UUID
    user_id: UUID
    operation: CodeOperation
    version: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "user_id": str(self.user_id),
            "operation": self.operation.to_dict(),
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CollaborationSession:
    """Collaboration session aggregate root."""

    id: UUID
    problem_id: UUID | None
    host_id: UUID
    title: str
    status: SessionStatus = SessionStatus.WAITING
    code_content: str = ""
    language: str = "python"
    version: int = 0  # For OT versioning
    participants: list[Participant] = field(default_factory=list)
    max_participants: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        host_id: UUID,
        title: str,
        problem_id: UUID | None = None,
        initial_code: str = "",
        language: str = "python",
    ) -> "CollaborationSession":
        """Factory method to create a new session."""
        return cls(
            id=uuid4(),
            problem_id=problem_id,
            host_id=host_id,
            title=title,
            code_content=initial_code,
            language=language,
        )

    def add_participant(self, user_id: UUID, username: str) -> Participant:
        """Add a participant to the session."""
        if len(self.active_participants) >= self.max_participants:
            raise ValueError("Session is full")

        # Check if already in session
        existing = self.get_participant_by_user_id(user_id)
        if existing:
            existing.is_active = True
            return existing

        # Assign color based on participant index
        color_index = len(self.participants)
        participant = Participant(
            id=uuid4(),
            user_id=user_id,
            session_id=self.id,
            username=username,
            color=get_participant_color(color_index),
        )
        self.participants.append(participant)

        # Start session when first participant joins
        if self.status == SessionStatus.WAITING:
            self.status = SessionStatus.ACTIVE

        self.updated_at = utc_now()
        return participant

    def remove_participant(self, user_id: UUID) -> None:
        """Remove a participant from the session."""
        participant = self.get_participant_by_user_id(user_id)
        if participant:
            participant.deactivate()
            self.updated_at = utc_now()

        # Close session if no active participants
        if not self.active_participants:
            self.status = SessionStatus.CLOSED

    def get_participant_by_user_id(self, user_id: UUID) -> Participant | None:
        """Get participant by user ID."""
        for p in self.participants:
            if p.user_id == user_id:
                return p
        return None

    def get_participant(self, user_id: UUID) -> Participant | None:
        """Alias for get_participant_by_user_id."""
        return self.get_participant_by_user_id(user_id)

    @property
    def active_participants(self) -> list[Participant]:
        """Get list of active participants."""
        return [p for p in self.participants if p.is_active]

    def apply_operation(self, operation: CodeOperation, user_id: UUID) -> int:
        """Apply a code operation and return new version."""
        self.version += 1

        if operation.operation_type.value == "insert":
            pos = operation.position
            self.code_content = (
                self.code_content[:pos] + operation.content + self.code_content[pos:]
            )
        elif operation.operation_type.value == "delete":
            pos = operation.position
            self.code_content = (
                self.code_content[:pos] + self.code_content[pos + operation.length :]
            )
        elif operation.operation_type.value == "replace":
            pos = operation.position
            self.code_content = (
                self.code_content[:pos]
                + operation.content
                + self.code_content[pos + operation.length :]
            )

        self.updated_at = utc_now()
        return self.version

    def update_participant_cursor(
        self, user_id: UUID, position: CursorPosition, selection: SelectionRange | None = None
    ) -> None:
        """Update participant cursor position."""
        participant = self.get_participant_by_user_id(user_id)
        if participant:
            participant.update_cursor(position)
            participant.update_selection(selection)

    def update_cursor(self, user_id: UUID, line: int, column: int) -> None:
        """Update participant cursor by line and column."""
        participant = self.get_participant_by_user_id(user_id)
        if participant:
            participant.update_cursor(CursorPosition(line=line, column=column))

    def update_selection(
        self,
        user_id: UUID,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> None:
        """Update participant selection range."""
        participant = self.get_participant_by_user_id(user_id)
        if participant:
            selection = SelectionRange(
                start=CursorPosition(line=start_line, column=start_column),
                end=CursorPosition(line=end_line, column=end_column),
            )
            participant.update_selection(selection)

    def close(self) -> None:
        """Close the session."""
        self.status = SessionStatus.CLOSED
        for p in self.participants:
            p.deactivate()
        self.updated_at = utc_now()

    def is_host(self, user_id: UUID) -> bool:
        """Check if user is the host."""
        return self.host_id == user_id

    def can_join(self, user_id: UUID) -> bool:
        """Check if user can join the session."""
        if self.status == SessionStatus.CLOSED:
            return False
        if len(self.active_participants) >= self.max_participants:
            # Allow if already a participant
            existing = self.get_participant_by_user_id(user_id)
            return existing is not None
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "problem_id": str(self.problem_id) if self.problem_id else None,
            "host_id": str(self.host_id),
            "title": self.title,
            "status": self.status.value,
            "code_content": self.code_content,
            "language": self.language,
            "version": self.version,
            "participants": [p.to_dict() for p in self.active_participants],
            "max_participants": self.max_participants,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
