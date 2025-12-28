"""Identity domain repository interfaces"""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.identity.domain.entities import User


class UserRepository(ABC):
    """Abstract repository interface for User aggregate"""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> User | None:
        """Get user by ID"""
        ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Get user by email"""
        ...

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None:
        """Get user by username"""
        ...

    @abstractmethod
    async def add(self, user: User) -> User:
        """Add a new user"""
        ...

    @abstractmethod
    async def update(self, user: User) -> User:
        """Update an existing user"""
        ...

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete user by ID"""
        ...

    @abstractmethod
    async def exists_by_email(self, email: str) -> bool:
        """Check if user with email exists"""
        ...

    @abstractmethod
    async def exists_by_username(self, username: str) -> bool:
        """Check if user with username exists"""
        ...
