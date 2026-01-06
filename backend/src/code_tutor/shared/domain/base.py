"""Base domain classes for DDD implementation"""

from abc import ABC
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)


class ValueObject(ABC):
    """
    Base class for Value Objects.
    Value Objects are immutable and compared by their attributes.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


class Entity(ABC):
    """
    Base class for Entities.
    Entities have identity and are compared by their ID.
    """

    def __init__(self, id: UUID | None = None) -> None:
        self._id = id or uuid4()
        self._created_at = utc_now()
        self._updated_at = utc_now()

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    def _touch(self) -> None:
        """Update the updated_at timestamp"""
        self._updated_at = utc_now()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id})"


class AggregateRoot(Entity):
    """
    Base class for Aggregate Roots.
    Aggregate Roots are the entry point to an aggregate and manage domain events.
    """

    def __init__(self, id: UUID | None = None) -> None:
        super().__init__(id)
        self._domain_events: list[Any] = []
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def domain_events(self) -> list[Any]:
        return self._domain_events.copy()

    def add_domain_event(self, event: Any) -> None:
        """Add a domain event to be dispatched"""
        self._domain_events.append(event)

    def clear_domain_events(self) -> list[Any]:
        """Clear and return all domain events"""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events

    def increment_version(self) -> None:
        """Increment version for optimistic locking"""
        self._version += 1
        self._touch()
