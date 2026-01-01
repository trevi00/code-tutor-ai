"""Playground repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.playground.domain.entities import (
    CodeTemplate,
    ExecutionHistory,
    Playground,
)
from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)


class PlaygroundRepository(ABC):
    """Repository interface for playgrounds."""

    @abstractmethod
    async def get_by_id(self, playground_id: UUID) -> Playground | None:
        """Get playground by ID."""
        pass

    @abstractmethod
    async def get_by_share_code(self, share_code: str) -> Playground | None:
        """Get playground by share code."""
        pass

    @abstractmethod
    async def save(self, playground: Playground) -> Playground:
        """Save or update a playground."""
        pass

    @abstractmethod
    async def delete(self, playground_id: UUID) -> None:
        """Delete a playground."""
        pass

    @abstractmethod
    async def get_user_playgrounds(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Playground]:
        """Get playgrounds owned by a user."""
        pass

    @abstractmethod
    async def get_public_playgrounds(
        self,
        language: PlaygroundLanguage | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Playground]:
        """Get public playgrounds."""
        pass

    @abstractmethod
    async def get_popular_playgrounds(
        self,
        limit: int = 10,
    ) -> list[Playground]:
        """Get most popular (by run count) public playgrounds."""
        pass

    @abstractmethod
    async def search_playgrounds(
        self,
        query: str,
        language: PlaygroundLanguage | None = None,
        limit: int = 20,
    ) -> list[Playground]:
        """Search public playgrounds by title/description."""
        pass


class TemplateRepository(ABC):
    """Repository interface for code templates."""

    @abstractmethod
    async def get_by_id(self, template_id: UUID) -> CodeTemplate | None:
        """Get template by ID."""
        pass

    @abstractmethod
    async def get_all(
        self,
        category: TemplateCategory | None = None,
        language: PlaygroundLanguage | None = None,
    ) -> list[CodeTemplate]:
        """Get all templates with optional filtering."""
        pass

    @abstractmethod
    async def get_popular(self, limit: int = 10) -> list[CodeTemplate]:
        """Get most popular templates."""
        pass

    @abstractmethod
    async def save(self, template: CodeTemplate) -> CodeTemplate:
        """Save a template."""
        pass


class ExecutionHistoryRepository(ABC):
    """Repository interface for execution history."""

    @abstractmethod
    async def save(self, history: ExecutionHistory) -> ExecutionHistory:
        """Save execution history."""
        pass

    @abstractmethod
    async def get_playground_history(
        self,
        playground_id: UUID,
        limit: int = 10,
    ) -> list[ExecutionHistory]:
        """Get recent execution history for a playground."""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: UUID,
        limit: int = 20,
    ) -> list[ExecutionHistory]:
        """Get recent execution history for a user."""
        pass
