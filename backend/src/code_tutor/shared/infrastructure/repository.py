"""Base repository classes for data access"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.shared.domain.base import Entity
from code_tutor.shared.infrastructure.database import Base

# Type variables for generic repository
T = TypeVar("T", bound=Entity)
M = TypeVar("M", bound=Base)


class Repository(ABC, Generic[T]):
    """Abstract base repository interface"""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> T | None:
        """Get entity by ID"""
        ...

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add new entity"""
        ...

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        ...

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete entity by ID"""
        ...


class SQLAlchemyRepository(Repository[T], Generic[T, M]):
    """
    Base SQLAlchemy repository implementation.
    Provides common CRUD operations for entities.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    @abstractmethod
    def _model_class(self) -> type[M]:
        """Return the SQLAlchemy model class"""
        ...

    @abstractmethod
    def _to_entity(self, model: M) -> T:
        """Convert SQLAlchemy model to domain entity"""
        ...

    @abstractmethod
    def _to_model(self, entity: T) -> M:
        """Convert domain entity to SQLAlchemy model"""
        ...

    async def get_by_id(self, id: UUID) -> T | None:
        """Get entity by ID"""
        result = await self._session.get(self._model_class, id)
        if result is None:
            return None
        return self._to_entity(result)

    async def add(self, entity: T) -> T:
        """Add new entity"""
        model = self._to_model(entity)
        self._session.add(model)
        await self._session.flush()
        return self._to_entity(model)

    async def update(self, entity: T) -> T:
        """Update existing entity"""
        model = self._to_model(entity)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def delete(self, id: UUID) -> bool:
        """Delete entity by ID"""
        model = await self._session.get(self._model_class, id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def exists(self, id: UUID) -> bool:
        """Check if entity exists"""
        result = await self._session.get(self._model_class, id)
        return result is not None

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """Get all entities with pagination"""
        stmt = select(self._model_class).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def count(self) -> int:
        """Count total entities"""
        from sqlalchemy import func

        stmt = select(func.count()).select_from(self._model_class)
        result = await self._session.execute(stmt)
        return result.scalar() or 0
