"""AI Tutor domain repository interfaces"""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.tutor.domain.entities import Conversation


class ConversationRepository(ABC):
    """Abstract repository interface for Conversation aggregate"""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Conversation | None:
        """Get conversation by ID"""
        ...

    @abstractmethod
    async def add(self, conversation: Conversation) -> Conversation:
        """Add a new conversation"""
        ...

    @abstractmethod
    async def update(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation"""
        ...

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete conversation by ID"""
        ...

    @abstractmethod
    async def get_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        """Get conversations by user"""
        ...

    @abstractmethod
    async def get_active_by_user(
        self,
        user_id: UUID,
        problem_id: UUID | None = None,
    ) -> Conversation | None:
        """Get active conversation for user (optionally for a specific problem)"""
        ...
