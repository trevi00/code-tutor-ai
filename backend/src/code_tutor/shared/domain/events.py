"""Domain Events base classes"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for Domain Events.
    Domain Events are immutable records of something that happened in the domain.
    """

    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_type(self) -> str:
        """Return the event type name"""
        return self.__class__.__name__


@dataclass(frozen=True)
class IntegrationEvent(ABC):
    """
    Base class for Integration Events.
    Integration Events are used for cross-bounded context communication.
    """

    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    source_context: str = ""

    @property
    def event_type(self) -> str:
        return self.__class__.__name__
