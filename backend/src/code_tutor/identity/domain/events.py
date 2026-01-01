"""Identity domain events"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from code_tutor.shared.domain.events import DomainEvent


@dataclass(frozen=True)
class UserCreated(DomainEvent):
    """Event raised when a new user is created"""

    user_id: UUID = field(default_factory=uuid4)
    email: str = ""
    username: str = ""


@dataclass(frozen=True)
class UserLoggedIn(DomainEvent):
    """Event raised when a user logs in"""

    user_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class PasswordChanged(DomainEvent):
    """Event raised when a user changes their password"""

    user_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class EmailVerified(DomainEvent):
    """Event raised when a user verifies their email"""

    user_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class UserDeactivated(DomainEvent):
    """Event raised when a user is deactivated"""

    user_id: UUID = field(default_factory=uuid4)
