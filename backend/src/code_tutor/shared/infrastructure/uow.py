"""Unit of Work pattern implementation"""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.shared.infrastructure.database import async_session_factory


class UnitOfWork(ABC):
    """Abstract Unit of Work interface"""

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        ...

    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction"""
        ...

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction"""
        ...


class SQLAlchemyUnitOfWork(UnitOfWork):
    """
    SQLAlchemy implementation of Unit of Work.
    Manages database transactions and repository lifecycle.
    """

    def __init__(self) -> None:
        self._session: AsyncSession | None = None

    @property
    def session(self) -> AsyncSession:
        """Get current session"""
        if self._session is None:
            raise RuntimeError("UnitOfWork not started. Use 'async with' context manager.")
        return self._session

    async def __aenter__(self) -> Self:
        factory = async_session_factory()
        self._session = factory()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()
        await self._close()

    async def commit(self) -> None:
        """Commit the transaction"""
        if self._session is not None:
            await self._session.commit()

    async def rollback(self) -> None:
        """Rollback the transaction"""
        if self._session is not None:
            await self._session.rollback()

    async def _close(self) -> None:
        """Close the session"""
        if self._session is not None:
            await self._session.close()
            self._session = None
