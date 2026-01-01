"""Collaboration domain repository interfaces."""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.collaboration.domain.entities import CodeChange, CollaborationSession


class CollaborationRepository(ABC):
    """Repository interface for collaboration sessions."""

    @abstractmethod
    async def get_by_id(self, session_id: UUID) -> CollaborationSession | None:
        """Get session by ID."""
        ...

    @abstractmethod
    async def save(self, session: CollaborationSession) -> CollaborationSession:
        """Save or update a session."""
        ...

    @abstractmethod
    async def delete(self, session_id: UUID) -> None:
        """Delete a session."""
        ...

    @abstractmethod
    async def get_user_sessions(
        self, user_id: UUID, active_only: bool = True
    ) -> list[CollaborationSession]:
        """Get sessions for a user."""
        ...

    @abstractmethod
    async def get_active_sessions(self, limit: int = 10) -> list[CollaborationSession]:
        """Get active public sessions."""
        ...

    @abstractmethod
    async def save_code_change(self, change: CodeChange) -> None:
        """Save a code change to history."""
        ...

    @abstractmethod
    async def get_session_changes(
        self, session_id: UUID, from_version: int = 0
    ) -> list[CodeChange]:
        """Get code changes for a session from a specific version."""
        ...
